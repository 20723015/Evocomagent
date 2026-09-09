"""SqlSessionStore（阶段八核心）：消息行式追加 + 会话 CAS，正本永不整包覆写。

写路径（一轮对话结束时单事务）：
    1. INSERT chat_messages（本轮新增，seq 递增；new_messages 为空则全量 state.messages）；
    2. UPDATE sessions SET version=version+1, ... WHERE version=?  —— 影响行数 0 → CAS 冲突 → 409；
    3. 同事务写 outbox（同步 ES message_search 用，可重建）。

Redis 热缓存（可选，保留并发角色）：save 后 write-through 写 session:{key}，
load 优先读缓存；缓存只是加速，正确性由 MySQL 兜底。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

from sqlalchemy import case, func, select
from sqlalchemy.exc import IntegrityError

from app.config.settings import settings
from app.stores.base import SessionConflictError, SessionState, StorageUnavailableError
from app.stores.sql.schema import chat_messages, memory_jobs, outbox_rows, sessions


def _parse_msg(row) -> dict:
    """content 存的是完整消息 dict 的 JSON（含 tool_calls），role/tool_name 是查询投影。"""
    try:
        return json.loads(row.content)
    except (json.JSONDecodeError, TypeError):
        return {"role": row.role, "content": row.content}


def _has_tool_calls(message: dict) -> bool:
    return bool(message.get("tool_calls"))


def normalize_model_history(rows: list[tuple[str, dict]]) -> list[dict]:
    """按 turn_id 分组归一化为模型历史：user + 最后一条非 tool-call assistant。

    Review 修复：SQL 正本保存完整审计（含中间 tool 消息），load 只返回
    模型需要的部分——中间 assistant(tool_calls)/tool 消息不进模型历史，
    避免孤儿 tool 消息与重复上下文。缺失终答的轮次只保留 user 消息。
    rows: (turn_id, 消息 dict)，按 seq 升序。
    """
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for turn, msg in rows:
        turn = str(turn or "")
        if turn not in groups:
            groups[turn] = []
            order.append(turn)
        groups[turn].append(msg)

    out: list[dict] = []
    for turn in order:
        group = groups[turn]
        final_assistant = None
        for msg in group:
            if msg.get("role") == "assistant" and not _has_tool_calls(msg):
                final_assistant = msg  # 最后一条非 tool-call assistant
        for msg in group:
            role = msg.get("role")
            if role == "user":
                out.append(msg)
            elif (role == "assistant" and not _has_tool_calls(msg)
                  and msg is final_assistant):
                out.append(msg)
            # 其余（中间 assistant tool-call / tool / 其他）只在审计正本
    return out


def invalidate_session_cache(redis, user_id: str, session_id: str) -> None:
    """worker 推进水位后失效会话热缓存（Review 修复：缓存不得返回旧水位）。"""
    if redis is None:
        return
    key = f"session:{user_id}/{session_id or 'session'}"
    try:
        redis.delete(key)
    except Exception:
        return


class SqlSessionStore:
    """SQLAlchemy 实现（生产 mysql+pymysql；本地 sqlite）。"""

    memory_jobs_enabled = True  # 阶段F：save 同事务入队 memory job

    def __init__(self, engine, redis=None, outbox_enabled: bool | None = None,
                 hot_ttl: int | None = None):
        self._engine = engine
        self._redis = redis
        self._outbox = (
            settings.message_index_outbox_enabled if outbox_enabled is None else outbox_enabled
        )
        self._hot_ttl = (
            settings.session_hot_cache_ttl_seconds if hot_ttl is None else hot_ttl
        )

    @staticmethod
    def _key(user_id: str, session_id: str) -> str:
        return f"{user_id}/{session_id or 'session'}"

    def _cache_key(self, key: str) -> str:
        return f"session:{key}"

    # ---------- 热缓存 ----------
    def _cache_get(self, key: str) -> SessionState | None:
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(self._cache_key(key))
        except Exception:  # noqa: BLE001 —— 缓存故障回源 MySQL
            return None
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            return SessionState(**data)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    def _cache_set(self, key: str, state: SessionState) -> None:
        if self._redis is None:
            return
        try:
            self._redis.set(
                self._cache_key(key),
                json.dumps(state.__dict__, ensure_ascii=False),
                ex=self._hot_ttl,
            )
        except Exception:  # noqa: BLE001 —— 写缓存失败不影响正本
            return

    def _cache_del(self, key: str) -> None:
        if self._redis is None:
            return
        try:
            self._redis.delete(self._cache_key(key))
        except Exception:  # noqa: BLE001
            return

    # ---------- 协议 ----------
    def load(self, user_id: str, session_id: str) -> SessionState | None:
        key = self._key(user_id, session_id)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    select(sessions).where(sessions.c.session_key == key)
                ).mappings().first()
                if row is None:
                    return None
                msgs = conn.execute(
                    select(chat_messages.c.turn_id, chat_messages.c.content)
                    .where(chat_messages.c.session_key == key)
                    .order_by(chat_messages.c.seq)
                ).all()
        except Exception as e:
            raise StorageUnavailableError(f"SQL 读取失败: {e}") from e

        state = SessionState(
            session_id=row["session_uuid"] or session_id,
            user_id=row["user_id"],
            summary=row["summary"],
            messages=normalize_model_history(
                [(row.turn_id, _parse_msg(row)) for row in msgs]
            ),
            short_term_memory=json.loads(row["stm_json"]) if row["stm_json"] else None,
            version=int(row["version"]),
            consolidated_len=int(row["consolidated_len"] or 0),
            updated_at=row["updated_at"].isoformat(sep=" ", timespec="seconds")
            if row["updated_at"] else "",
        )
        self._cache_set(key, state)
        return state

    def save(self, user_id: str, session_id: str, state: SessionState,
             new_messages: list[dict] | None = None,
             enqueue_memory_job: bool = False) -> SessionState:
        key = self._key(user_id, session_id)
        to_insert = new_messages if new_messages is not None else state.messages
        actual_consolidated_len = int(state.consolidated_len or 0)
        try:
            with self._engine.begin() as conn:
                row = conn.execute(
                    select(sessions).where(sessions.c.session_key == key)
                ).mappings().first()
                current_version = int(row["version"]) if row else 0
                if current_version != state.version:
                    raise SessionConflictError(
                        f"session {user_id}/{session_id} 版本冲突："
                        f"SQL v{current_version} ≠ 期望 v{state.version}"
                    )
                if row is None:
                    conn.execute(sessions.insert().values(
                        session_key=key, user_id=user_id,
                        session_uuid=state.session_id or session_id,
                        version=0, summary=state.summary,
                        stm_json=json.dumps(state.short_term_memory, ensure_ascii=False)
                        if state.short_term_memory else None,
                        consolidated_len=state.consolidated_len,
                        status="active",
                    ))
                max_seq = conn.execute(
                    select(func.coalesce(func.max(chat_messages.c.seq), 0))
                    .where(chat_messages.c.session_key == key)
                ).scalar_one()
                created = datetime.now()
                # 一轮一次 save() = 一轮对话：turn_id 轮次粒度（DDL 语义：一轮一个 turn）
                turn_id = uuid.uuid4().hex[:32]
                seq = int(max_seq)
                for i, msg in enumerate(to_insert):
                    seq = int(max_seq) + 1 + i
                    conn.execute(chat_messages.insert().values(
                        session_key=key, user_id=user_id,
                        turn_id=turn_id,
                        seq=seq, role=str(msg.get("role", "")),
                        content=json.dumps(msg, ensure_ascii=False),
                        tool_name=_tool_name_of(msg),
                    ))
                    if self._outbox:
                        conn.execute(outbox_rows.insert().values(
                            session_key=key, seq=seq,
                            payload=json.dumps(msg, ensure_ascii=False),
                        ))
                # 阶段F：消息保存与 memory job 入队同一事务（唯一键
                # session_key+through_seq 幂等；重复入队被唯一约束吸收）
                if enqueue_memory_job and to_insert:
                    conn.execute(memory_jobs.insert().values(
                        session_key=key, user_id=user_id,
                        session_uuid=state.session_id or session_id,
                        through_seq=seq, status="pending",
                    ))
                # Review 修复：水位原子单调——数据库内 clamp（MAX(当前, 期望)），
                # 旧 Agent 状态/历史压缩不得把 memory_consolidated_seq 写小；
                # worker 与 save 并发时以数据库现值为准。
                if row is not None and int(row["consolidated_len"] or 0) > state.consolidated_len:
                    from app.observability.metrics import record_watermark_regression_blocked

                    record_watermark_regression_blocked()
                updated = conn.execute(
                    sessions.update()
                    .where(sessions.c.session_key == key,
                           sessions.c.version == state.version)
                    .values(
                        version=state.version + 1,
                        session_uuid=state.session_id or session_id,
                        summary=state.summary,
                        stm_json=json.dumps(state.short_term_memory, ensure_ascii=False)
                        if state.short_term_memory else None,
                        consolidated_len=case(
                            (sessions.c.consolidated_len > state.consolidated_len,
                             sessions.c.consolidated_len),
                            else_=state.consolidated_len,
                        ),
                        updated_at=created,
                    )
                )
                if updated.rowcount != 1:
                    raise SessionConflictError(
                        f"session {user_id}/{session_id} CAS 更新未生效（并发写入）"
                    )
                # UPDATE 使用数据库内 MAX/clamp；不要把调用方的旧水位写回
                # 热缓存。这里读回同一事务中的最终值，确保返回值和缓存都反映
                # DB 正本（worker 可能刚刚推进过 consolidated_len）。
                actual_consolidated_len = int(conn.execute(
                    select(sessions.c.consolidated_len).where(
                        sessions.c.session_key == key
                    )
                ).scalar_one() or 0)
        except SessionConflictError:
            raise
        except IntegrityError as e:
            # 并发首写/seq 冲突：唯一键撞车是「可重试的并发」而非「存储不可用」
            # （低概率竞态：两请求同看无会话行、同时插入 sessions 主键）
            raise SessionConflictError(
                f"session {user_id}/{session_id} 并发首写冲突（唯一键）: {e.orig}"
            ) from e
        except Exception as e:
            raise StorageUnavailableError(f"SQL 写入失败: {e}") from e

        updated_state = SessionState(**{
            **state.__dict__,
            "version": state.version + 1,
            "consolidated_len": actual_consolidated_len,
        })
        self._cache_set(key, updated_state)
        return updated_state

    def delete(self, user_id: str, session_id: str) -> None:
        key = self._key(user_id, session_id)
        try:
            with self._engine.begin() as conn:
                # Review 修复：reset 同事务删除该会话的 memory jobs
                # （旧 worker 已领取的按 session_uuid 比对判 obsolete）
                conn.execute(chat_messages.delete().where(chat_messages.c.session_key == key))
                conn.execute(outbox_rows.delete().where(outbox_rows.c.session_key == key))
                conn.execute(memory_jobs.delete().where(memory_jobs.c.session_key == key))
                conn.execute(sessions.delete().where(sessions.c.session_key == key))
        except Exception as e:
            raise StorageUnavailableError(f"SQL 删除失败: {e}") from e
        self._cache_del(key)

    def iter_all(self) -> list[tuple[str, str]]:
        """枚举 (user_id, session_id)：idle 兜底巩固 job 用。"""
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    select(sessions.c.session_key)
                ).scalars().all()
        except Exception as e:
            raise StorageUnavailableError(f"SQL 枚举失败: {e}") from e
        out = []
        for key in rows:
            if "/" in key:
                user_id, rest = key.split("/", 1)
                out.append((user_id, rest))
            else:
                out.append((key, ""))
        return out


def _tool_name_of(msg: dict) -> str | None:
    if msg.get("role") != "tool":
        return None
    return None  # tool 消息本身不含工具名；审计用 tool_call_id 关联 assistant 消息
