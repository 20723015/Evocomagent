"""memory_jobs：持久化记忆巩固任务（单 Agent 全量优化计划·阶段F）。

模型：
- 入队：消息保存同事务 INSERT memory_jobs（唯一键 session_key+through_seq）；
- 领取：UPDATE ... WHERE status='pending' OR 租约过期 → status='processing'，
  lease_until=now+lease（崩溃后可被其他 worker 接管）；
- 处理：按水位读取增量消息（seq > memory_consolidated_seq 且 ≤ through_seq），
  LLM 提取复杂事实 → LTM merge；成功后原子推进 sessions.consolidated_len
  （= memory_consolidated_seq 水位）并置 done；
- 幂等：至少一次投递下，水位 ≥ through_seq 的重复任务直接确认完成；
- 重试：attempts+1、next_run_at 指数退避；超上限置 failed（死信）+ 脱敏错误。

开发模式（无 SQL）：文件轻量队列（memory_dir/memory_jobs.jsonl），
现有 idle scanner 作为漏单修复器。
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta

from sqlalchemy import select, update

from app.config.settings import settings
from app.observability.logging import get_logger
from app.observability.metrics import (
    record_memory_job_dead,
    record_memory_job_latency,
    record_memory_job_retry,
    set_memory_job_backlog,
)

log = get_logger("app.agent.memory.jobs")

_MAX_ERROR_CHARS = 300


def _now() -> datetime:
    return datetime.now()


def _sanitize_error(exc: Exception) -> str:
    """脱敏错误摘要：只留异常类型与首行，不落用户内容/SQL 细节。"""
    text = f"{type(exc).__name__}: {exc}".splitlines()[0][:MAX_ERROR_CHARS]
    return text


MAX_ERROR_CHARS = _MAX_ERROR_CHARS


class SqlMemoryJobStore:
    """SQL 实现（MySQL/SQLite）：与消息正本同库。"""

    def __init__(self, engine, max_attempts: int | None = None,
                 lease_seconds: int | None = None, redis=None):
        self._engine = engine
        self._max_attempts = max_attempts or settings.memory_job_max_attempts
        self._lease = lease_seconds or settings.memory_job_lease_seconds
        self._redis = redis  # Review 修复：水位推进后失效会话热缓存

    # ---------- 入队（由 SqlSessionStore.save 同事务调用） ----------
    def enqueue(self, conn, session_key: str, user_id: str, through_seq: int,
                session_uuid: str = "") -> None:
        from app.stores.sql.schema import memory_jobs

        conn.execute(memory_jobs.insert().values(
            session_key=session_key, user_id=user_id,
            session_uuid=session_uuid,
            through_seq=through_seq, status="pending",
        ))

    def session_uuid_of(self, session_key: str) -> str:
        """当前 session 实例 UUID（reset 后变化；Review 修复：旧任务判 obsolete）。"""
        from app.stores.sql.schema import sessions

        try:
            with self._engine.connect() as conn:
                value = conn.execute(
                    select(sessions.c.session_uuid)
                    .where(sessions.c.session_key == session_key)
                ).scalar_one_or_none()
        except Exception:  # noqa: BLE001
            return ""
        return str(value or "")

    def obsolete(self, job: dict) -> None:
        """reset 后领取的旧任务：标记 obsolete，绝不读取新会话消息。"""
        from app.stores.sql.schema import memory_jobs

        with self._engine.begin() as conn:
            conn.execute(
                memory_jobs.update()
                .where(memory_jobs.c.id == job["id"])
                .values(status="obsolete", updated_at=_now())
            )
        from app.observability.metrics import record_memory_job_obsolete

        record_memory_job_obsolete()

    # ---------- 领取 ----------
    def claim(self, worker_id: str, limit: int = 5) -> list[dict]:
        """原子领取可执行任务（pending 或租约过期）；返回任务字段列表。"""
        from app.stores.sql.schema import memory_jobs

        now = _now()
        try:
            with self._engine.begin() as conn:
                rows = conn.execute(
                    select(memory_jobs)
                    .where(
                        memory_jobs.c.status == "pending",
                        (memory_jobs.c.next_run_at.is_(None))
                        | (memory_jobs.c.next_run_at <= now),
                    )
                    .order_by(memory_jobs.c.id)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                ).mappings().all()
                # 租约过期的 processing（worker 崩溃）接管
                if len(rows) < limit:
                    stale = conn.execute(
                        select(memory_jobs)
                        .where(
                            memory_jobs.c.status == "processing",
                            memory_jobs.c.lease_until < now,
                        )
                        .order_by(memory_jobs.c.id)
                        .limit(limit - len(rows))
                        .with_for_update(skip_locked=True)
                    ).mappings().all()
                    rows = list(rows) + list(stale)
                claimed: list[dict] = []
                for row in rows:
                    conn.execute(
                        memory_jobs.update()
                        .where(memory_jobs.c.id == row["id"])
                        .values(
                            status="processing",
                            leased_by=worker_id,
                            lease_until=now + timedelta(seconds=self._lease),
                            updated_at=now,
                        )
                    )
                    claimed.append(dict(row))
                return claimed
        except Exception as e:  # noqa: BLE001 —— 领取失败下轮再试
            log.warning("memory_job.claim_failed err=%s", type(e).__name__)
            return []

    # ---------- 完成/失败/幂等 ----------
    def complete(self, job: dict) -> None:
        from app.stores.sql.schema import memory_jobs

        with self._engine.begin() as conn:
            conn.execute(
                memory_jobs.update()
                .where(memory_jobs.c.id == job["id"])
                .values(status="done", updated_at=_now(), error="")
            )
        created = job.get("created_at")
        if created is not None:
            try:
                record_memory_job_latency((_now() - created).total_seconds())
            except (TypeError, ValueError):
                pass

    def fail(self, job: dict, exc: Exception) -> None:
        from app.stores.sql.schema import memory_jobs

        attempts = int(job.get("attempts", 0) or 0) + 1
        now = _now()
        with self._engine.begin() as conn:
            if attempts >= self._max_attempts:
                conn.execute(
                    memory_jobs.update()
                    .where(memory_jobs.c.id == job["id"])
                    .values(
                        status="failed", attempts=attempts,
                        error=_sanitize_error(exc), updated_at=now,
                    )
                )
                record_memory_job_dead()
            else:
                backoff = min(300, 5 * (2 ** attempts))
                conn.execute(
                    memory_jobs.update()
                    .where(memory_jobs.c.id == job["id"])
                    .values(
                        status="pending", attempts=attempts,
                        next_run_at=now + timedelta(seconds=backoff),
                        error=_sanitize_error(exc), updated_at=now,
                    )
                )
                record_memory_job_retry()

    def is_duplicate(self, job: dict) -> bool:
        """水位幂等：sessions.consolidated_len ≥ through_seq → 直接确认完成。"""
        from app.stores.sql.schema import sessions

        with self._engine.connect() as conn:
            value = conn.execute(
                select(sessions.c.consolidated_len)
                .where(sessions.c.session_key == job["session_key"])
            ).scalar_one_or_none()
        return value is not None and int(value) >= int(job["through_seq"])

    def backlog(self) -> int:
        from app.stores.sql.schema import memory_jobs

        try:
            with self._engine.connect() as conn:
                return int(conn.execute(
                    select(memory_jobs.c.id).where(
                        memory_jobs.c.status == "pending",
                    )
                ).scalars().all().__len__()
                )
        except Exception:  # noqa: BLE001
            return 0

    def due_messages(self, session_key: str, through_seq: int,
                     watermark: int) -> list[dict]:
        """读取 (watermark, through_seq] 的消息（parse 回消息 dict）。"""
        from app.stores.sql.schema import chat_messages

        with self._engine.connect() as conn:
            rows = conn.execute(
                select(chat_messages.c.content)
                .where(
                    chat_messages.c.session_key == session_key,
                    chat_messages.c.seq > watermark,
                    chat_messages.c.seq <= through_seq,
                )
                .order_by(chat_messages.c.seq)
            ).scalars().all()
        out = []
        for raw in rows:
            try:
                out.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                continue
        return out

    def watermark(self, session_key: str) -> int:
        from app.stores.sql.schema import sessions

        with self._engine.connect() as conn:
            value = conn.execute(
                select(sessions.c.consolidated_len)
                .where(sessions.c.session_key == session_key)
            ).scalar_one_or_none()
        return int(value or 0)

    def advance_watermark(self, session_key: str, through_seq: int) -> None:
        """成功后原子推进 consolidated_len 水位（单调不回退）并失效会话热缓存。"""
        from app.stores.sql.schema import sessions

        with self._engine.begin() as conn:
            conn.execute(
                update(sessions)
                .where(
                    sessions.c.session_key == session_key,
                    sessions.c.consolidated_len < through_seq,
                )
                .values(consolidated_len=through_seq)
            )
        if self._redis is not None and "/" in session_key:
            from app.stores.sql.session_store import invalidate_session_cache

            user_id, _, session_id = session_key.partition("/")
            invalidate_session_cache(self._redis, user_id, session_id)


# ============================================================
# 文件轻量队列（开发模式；无 SQL 时）
# ============================================================
class FileMemoryJobStore:
    """JSONL 轻量队列（单机开发；Review 修复后为负载自包含模式）。

    - 任务条目携带待巩固消息负载（messages），不再依赖「user/session 拼(marker)
      文件名」或「可能因压缩回退的消息长度」推导增量；
    - leased_by/lease_until + 过期 processing 接管：进程崩溃后可恢复；
    - 原子重写（tmp + os.replace，Windows 兼容）；
    - 完成记录写 done log，保证「至少一次投递 + 幂等去重」下不重复抽取；
    - 同一轮 turn_id 幂等：enqueue 时已存在（任意状态）或已完成即跳过；
      没有 turn_id 的旧调用继续使用 session_key:through_seq 兼容 ID。
    - enqueue/claim/complete/fail/purge 全部在跨进程 lock-file 保护下完成，
      防止读-改-写与追加同时发生时丢任务。
    """

    LEASE_SECONDS = 300

    def __init__(self, memory_dir: str, max_attempts: int | None = None):
        from pathlib import Path

        self._dir = Path(memory_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._queue = self._dir / "memory_jobs.jsonl"
        self._done_log = self._dir / "memory_jobs.done.jsonl"
        self._lock = self._dir / "memory_jobs.lock"
        self._max_attempts = max_attempts or settings.memory_job_max_attempts

    # ---------- 内部 ----------
    @contextmanager
    def _locked(self):
        """跨线程/进程的轻量互斥锁（Windows/POSIX 通用）。

        O_EXCL 创建锁文件是原子的，且不依赖第三方包。异常退出留下的锁
        只在足够老时回收，避免主动删除仍在执行的长任务锁。
        """
        deadline = time.monotonic() + 15.0
        fd = None
        while fd is None:
            try:
                fd = os.open(
                    self._lock,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                os.write(fd, f"{os.getpid()}\n".encode("ascii"))
                os.close(fd)
                fd = -1
            except FileExistsError:
                try:
                    age = time.time() - self._lock.stat().st_mtime
                    if age > 120.0:
                        self._lock.unlink()
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError("memory job queue lock timeout")
                time.sleep(0.01)
        try:
            yield
        finally:
            try:
                self._lock.unlink()
            except FileNotFoundError:
                pass

    def _read_all_unlocked(self) -> list[dict]:
        if not self._queue.exists():
            return []
        rows: list[dict] = []
        with self._queue.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def _read_all(self) -> list[dict]:
        with self._locked():
            return self._read_all_unlocked()

    def _write_all_unlocked(self, rows: list[dict]) -> None:
        tmp = self._queue.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for entry in rows:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        tmp.replace(self._queue)

    def _write_all(self, rows: list[dict]) -> None:
        with self._locked():
            self._write_all_unlocked(rows)

    def _read_done_ids_unlocked(self) -> set[str]:
        if not self._done_log.exists():
            return set()
        out: set[str] = set()
        try:
            with self._done_log.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    job_id = str(item.get("id") or "")
                    if job_id:
                        out.add(job_id)
        except OSError:
            return set()
        return out

    def _upsert_unlocked(self, job: dict, *, drop: bool = False) -> None:
        rows = [r for r in self._read_all_unlocked() if r.get("id") != job.get("id")]
        if not drop:
            rows.append(dict(job))
        self._write_all_unlocked(rows)

    def _upsert(self, job: dict, *, drop: bool = False) -> None:
        with self._locked():
            self._upsert_unlocked(job, drop=drop)

    # ---------- 协议 ----------
    def enqueue(self, session_key: str, user_id: str, through_seq: int,
                session_uuid: str = "", messages: list[dict] | None = None,
                turn_id: str = "") -> None:
        """入队（负载自包含）并保证已完成任务不可重新入队。

        ``turn_id`` 是轮次稳定幂等键；旧调用不传时保留原
        ``session_key:through_seq`` ID 语义，避免升级时破坏已有队列。
        """
        turn_id = str(turn_id or "")
        job_id = (
            f"{session_key}:turn:{turn_id}"
            if turn_id else f"{session_key}:{through_seq}"
        )
        with self._locked():
            rows = self._read_all_unlocked()
            done_ids = self._read_done_ids_unlocked()
            if job_id in done_ids or any(r.get("id") == job_id for r in rows):
                return
            entry = {
                "id": job_id,
                "turn_id": turn_id,
                "session_key": session_key,
                "user_id": user_id,
                "session_uuid": session_uuid,
                "through_seq": through_seq,
                "status": "pending",
                "attempts": 0,
                "next_run_at": "",
                "leased_by": "",
                "lease_until": "",
                "created_at": _now().isoformat(timespec="seconds"),
                "messages": list(messages or []),  # 负载：待巩固的折叠后消息
            }
            # 统一用原子重写，避免并发 enqueue 的追加与其他读改写交叉。
            self._write_all_unlocked(rows + [entry])

    def claim(self, worker_id: str, limit: int = 5) -> list[dict]:
        """原子领取：pending（到 next_run_at）+ 租约过期的 processing。"""
        now = _now()
        with self._locked():
            rows = self._read_all_unlocked()
            done_ids = self._read_done_ids_unlocked()
            # complete() 先写 done、后删队列时进程若崩溃，下一次 claim
            # 仍不能重新执行该任务；顺手清理残留队列行。
            kept = [r for r in rows if r.get("id") not in done_ids]
            changed = len(kept) != len(rows)
            rows = kept
            claimed: list[dict] = []
            for entry in rows:
                if len(claimed) >= limit:
                    break
                status = entry.get("status")
                if status == "pending":
                    next_run = str(entry.get("next_run_at") or "")
                    if next_run:
                        try:
                            if _now() < datetime.fromisoformat(next_run):
                                continue
                        except ValueError:
                            pass
                elif status == "processing":
                    lease_until = str(entry.get("lease_until") or "")
                    try:
                        if lease_until and _now() < datetime.fromisoformat(lease_until):
                            continue  # 租约未到期（他人持有）
                    except ValueError:
                        pass  # 租约损坏 → 视为过期接管
                else:
                    continue  # done/failed/obsolete 不再领取
                entry["status"] = "processing"
                entry["leased_by"] = worker_id
                entry["lease_until"] = (
                    now + timedelta(seconds=self.LEASE_SECONDS)
                ).isoformat(timespec="seconds")
                claimed.append(entry)
                changed = True
            if changed:
                self._write_all_unlocked(rows)
            return claimed

    def complete(self, job: dict) -> None:
        """完成：移出队列 + 追加完成记录（审计/崩溃后不重复抽取的依据）。"""
        with self._locked():
            done_ids = self._read_done_ids_unlocked()
            job_id = str(job.get("id") or "")
            if job_id and job_id not in done_ids:
                with self._done_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "id": job_id,
                        "turn_id": job.get("turn_id", ""),
                        "session_key": job.get("session_key"),
                        "through_seq": job.get("through_seq"),
                        "finished_at": _now().isoformat(timespec="seconds"),
                    }, ensure_ascii=False) + "\n")
            # done log is the durable idempotency marker; a crash before this
            # removal is recovered by claim() using the same marker.
            self._upsert_unlocked(job, drop=True)

    def fail(self, job: dict, exc: Exception) -> None:
        attempts = int(job.get("attempts", 0) or 0) + 1
        job["attempts"] = attempts
        if attempts >= self._max_attempts:
            job["status"] = "failed"
            job["error"] = type(exc).__name__
            record_memory_job_dead()
        else:
            job["status"] = "pending"
            backoff = min(300, 5 * (2 ** attempts))
            job["next_run_at"] = (
                _now() + timedelta(seconds=backoff)
            ).isoformat(timespec="seconds")
            record_memory_job_retry()
        self._upsert(job)

    def obsolete(self, job: dict) -> None:
        job["status"] = "obsolete"
        self._upsert(job)
        from app.observability.metrics import record_memory_job_obsolete

        record_memory_job_obsolete()

    def is_duplicate(self, job: dict) -> bool:
        # 负载自包含 + 完成即出队：重复投递由 enqueue 幂等与 done log 保证
        return False

    def backlog(self) -> int:
        return sum(
            1 for entry in self._read_all() if entry.get("status") == "pending"
        )

    def due_messages(self, session_key: str, through_seq: int,
                     watermark: int) -> list[dict]:
        """负载自包含模式：worker 直接用 job["messages"]，此方法不再被消费。"""
        return []

    def watermark(self, session_key: str) -> int:
        return 0

    def advance_watermark(self, session_key: str, through_seq: int) -> None:
        return None

    def purge_session(self, session_key: str) -> int:
        """reset：清除该会话的全部任务（Review 修复）。"""
        with self._locked():
            rows = self._read_all_unlocked()
            kept = [r for r in rows if r.get("session_key") != session_key]
            removed = len(rows) - len(kept)
            if removed:
                self._write_all_unlocked(kept)
            return removed

    def jobs_snapshot(self) -> list[dict]:
        """测试/运维视图。"""
        return self._read_all()


def purge_session_file_jobs(user_id: str, session_id: str) -> None:
    """reset 清理入口：文件模式清除该会话任务；SQL 模式由 delete 同事务处理。"""
    from pathlib import Path

    queue = Path(settings.memory_dir) / "jobs"
    if not queue.exists():
        return
    store = FileMemoryJobStore(str(queue))
    store.purge_session(f"{user_id}/{session_id or 'session'}")


def build_memory_job_store(engine=None, redis=None):
    """按配置选择队列：SQL 正本优先；否则文件轻量队列。"""
    if engine is not None:
        return SqlMemoryJobStore(engine, redis=redis)
    from pathlib import Path

    return FileMemoryJobStore(str(Path(settings.memory_dir) / "jobs"))


# ============================================================
# Worker
# ============================================================
def build_memory_job_worker(components) -> MemoryJobWorker:
    """按 pod 组件装配 worker：SQL 正本优先，否则文件轻量队列。"""
    from app.agent.memory.long_term import LongTermMemory

    engine = getattr(components, "db_engine", None)
    store = build_memory_job_store(engine, redis=getattr(components, "redis", None))

    def _ltm_factory(user_id: str) -> LongTermMemory:
        return LongTermMemory(
            user_id=user_id,
            memory_dir=settings.memory_dir,
            max_facts=settings.max_ltm_facts,
            store=getattr(components, "ltm_store", None),
        )

    return MemoryJobWorker(
        store,
        _ltm_factory,
        getattr(components, "client", None),
        settings.model_name,
        worker_id=f"pod-{id(components) % 100000}",
    )


class MemoryJobWorker:
    """后台 worker：领取 → 幂等检查 → 增量 LLM 巩固 → 水位推进 → done。"""

    def __init__(self, store, ltm_factory, llm_client, model: str,
                 worker_id: str = ""):
        self._store = store
        self._ltm_factory = ltm_factory  # callable(user_id) -> LongTermMemory
        self._client = llm_client
        self._model = model
        self._worker_id = worker_id or "worker-1"

    def process_once(self, limit: int = 5) -> int:
        """处理一批任务；返回完成数（供循环/测试断言）。"""
        claimed = self._store.claim(self._worker_id, limit=limit)
        done = 0
        for job in claimed:
            try:
                if self._store.is_duplicate(job):
                    # 至少一次投递：重复任务按水位直接确认（幂等）
                    self._store.complete(job)
                    done += 1
                    continue
                processed = self._process_job(job)
                if processed is False:
                    # Review 修复：obsolete 任务状态已由 store 落库，
                    # 绝不确认完成（避免把新会话水位误推进）
                    continue
                self._store.complete(job)
                done += 1
            except Exception as e:  # noqa: BLE001 —— 单任务失败不拖垮 worker
                log.warning(
                    "memory_job.failed session=%s seq=%s err=%s",
                    job.get("session_key"), job.get("through_seq"),
                    type(e).__name__,
                )
                self._store.fail(job, e)
        backlog = self._store.backlog()
        set_memory_job_backlog(backlog)
        return done

    def _process_job(self, job: dict) -> None:
        session_key = str(job.get("session_key", ""))
        through_seq = int(job.get("through_seq", 0))
        user_id = str(job.get("user_id", ""))
        job_uuid = str(job.get("session_uuid", "") or "")
        # Review 修复：reset（同 session_id 重建）后 job_uuid ≠ 当前会话 UUID
        # → 旧任务直接 obsolete，绝不读取新会话消息
        if job_uuid:
            current_uuid = ""
            uuid_of = getattr(self._store, "session_uuid_of", None)
            if uuid_of is not None:
                current_uuid = uuid_of(session_key)
            if current_uuid and job_uuid != current_uuid:
                self._store.obsolete(job)
                log.info(
                    "memory_job.obsolete session=%s job_uuid=%s current=%s",
                    session_key, job_uuid[:8], current_uuid[:8],
                )
                return False
        # 消息来源：负载自包含（文件模式）优先；SQL 模式按水位读增量
        messages = job.get("messages")
        if messages is None:
            watermark = self._store.watermark(session_key)
            if watermark >= through_seq:
                return  # 已巩固过（幂等）
            messages = self._store.due_messages(session_key, through_seq, watermark)
        # 只巩固到本轮用户可见消息（工具中间消息在折叠后的正本里已不存在）
        if not messages:
            advance = getattr(self._store, "advance_watermark", None)
            if advance is not None:
                advance(session_key, through_seq)
            return True
        ltm = self._ltm_factory(user_id)
        summary_text = ""
        from app.agent.tools.digest import render_tool_result_line, tool_call_name_map

        call_map = tool_call_name_map(messages)
        transcript: list[dict] = []
        for msg in messages:
            if msg.get("role") == "tool":
                transcript.append({
                    "role": "tool",
                    "content": render_tool_result_line(
                        call_map.get(msg.get("tool_call_id"), "?"),
                        str(msg.get("content") or ""),
                    ),
                })
            else:
                transcript.append(msg)
        ltm.extract_and_save(self._client, self._model, transcript, summary_text)
        advance = getattr(self._store, "advance_watermark", None)
        if advance is not None:
            advance(session_key, through_seq)
        return True
