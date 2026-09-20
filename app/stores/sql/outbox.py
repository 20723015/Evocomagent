"""Outbox → ES 同步（阶段八；修复计划·二轮 2/3/4 重写）。

MySQL 为正本，ES message_search 是可重建衍生品。

消息 Outbox（outbox_rows）：
- 消息 INSERT 与 outbox 同事务（见 SqlSessionStore.save）；
- worker 短事务领取（FOR UPDATE SKIP LOCKED + 租约 token）：同时领取已到期的
  `pending` 与 `lease_until` 已过期的 `processing`（崩溃接管）；
- ES bulk 调用在数据库事务外；调用前与结算前都做三重陈旧检查
  （租约 token 仍归自己、会话 UUID 未变、无删除 tombstone）；陈旧行结算为
  `obsolete` 终态，不再重试或写入；
- 逐行按 item 响应结算：成功 done；429/5xx 指数退避；确定性 4xx dead-letter；
  坏 JSON 单行 dead-letter；
- 所有结算更新必须满足 `status=processing AND lease_token=当前 token`，并清空
  owner/token/lease_until（fencing，防旧 worker 覆盖新租约）。

删除 Outbox（message_delete_outbox）：
- Reset 事务写唯一删除事件（session_key + 旧 session_uuid）；
- **领取前置屏障**：同 session_key + 旧 UUID（含 legacy 空 UUID）的消息 Outbox
  全部进入终态（done/dead_letter/obsolete）后才可领取，避免删完又有迟到写入；
- delete_by_query（refresh=True）删除旧 UUID + legacy 空 UUID 文档；成功后标 done；
- done tombstone 在保留期内仍参与搜索过滤；保留期结束时**再次执行删除**，成功
  后才清理该事件行（幂等）。
"""

from __future__ import annotations

import json
import logging
import os
import random
import socket
import uuid
from datetime import datetime, timedelta

from sqlalchemy import and_, delete, func, or_, select, update

from app.config.settings import settings
from app.observability.logging import get_logger
from app.stores.sql.schema import (
    message_delete_outbox,
    outbox_rows,
    sessions,
)

log = get_logger("app.stores.sql.outbox")

_RETRYABLE_STATUS = 429
_IN_FLIGHT_STATUSES = ("pending", "processing")
_TERMINAL_STATUSES = ("done", "dead_letter", "obsolete")


class MessageTombstoneUnavailable(RuntimeError):
    """删除 tombstone 无法可靠读取（DB 缺失/查询失败）：搜索 fail-closed → 503。"""


class OutboxStateUnavailable(RuntimeError):
    """会话/删除状态的可靠性查询失败。

    修复计划·三轮 P1-2：此时**不得**判定行陈旧（不得结算为 obsolete）；
    按 token fencing 退回 pending 并退避，不写 ES、不因重试上限进死信。
    """


def _invalidate_es(reason: str) -> None:
    try:
        from app.agent.rag.es_util import invalidate_es_client

        invalidate_es_client(reason)
    except Exception:  # noqa: BLE001
        pass


def _metric(fn: str, *args) -> None:
    try:
        from app.observability import metrics

        getattr(metrics, fn)(*args)
    except Exception:  # noqa: BLE001 —— 指标失败不影响主流程
        pass


def _worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"[:64]


def _now() -> datetime:
    return datetime.now()


def _backoff_seconds(attempts: int) -> float:
    base = min(settings.outbox_max_backoff_seconds, 2 * (2 ** max(int(attempts) - 1, 0)))
    return float(base) + random.random() * 0.5


def _item_status(item: dict) -> int | None:
    if not isinstance(item, dict):
        return None
    for value in item.values():
        if isinstance(value, dict) and "status" in value:
            try:
                return int(value["status"])
            except (TypeError, ValueError):
                return None
    return None


def _error_column(table) -> str:
    return "error" if hasattr(table.c, "error") else "sync_error"


# ------------------------------------------------------------
# 领取 / 结算（短事务，fencing）
# ------------------------------------------------------------
def _claim(
    engine, table, limit: int, lease_seconds: int, kind: str,
    worker_id: str | None = None, eligible=None,
) -> list[dict]:
    """原子领取：pending 到期 或 processing 租约过期；写入新 token。

    - worker_id 传入时实际生效（同批只解析一次）；
    - eligible(conn, row) 为额外领取条件（删除事件的终态屏障）；
    - 更新以 (status, lease_token) 为 fencing 条件，避免覆盖刚被他人接管的行。
    """
    now = _now()
    wid = worker_id or _worker_id()
    claimed: list[dict] = []
    try:
        with engine.begin() as conn:
            rows = conn.execute(
                select(table)
                .where(or_(
                    and_(
                        table.c.status == "pending",
                        or_(table.c.next_run_at.is_(None), table.c.next_run_at <= now),
                    ),
                    and_(
                        table.c.status == "processing",
                        table.c.lease_until.is_not(None),
                        table.c.lease_until <= now,
                    ),
                ))
                .order_by(table.c.id)
                .limit(max(int(limit), 1))
                .with_for_update(skip_locked=True)
            ).mappings().all()
            for row in rows:
                if eligible is not None and not eligible(conn, row):
                    continue  # 条件不满足（如同 session 仍有在途消息）：本轮跳过
                prev_status = row["status"]
                prev_token = row["lease_token"] or ""
                token = uuid.uuid4().hex
                result = conn.execute(
                    update(table)
                    .where(
                        table.c.id == row["id"],
                        table.c.status == prev_status,
                        table.c.lease_token == prev_token,
                    )
                    .values(
                        status="processing",
                        lease_owner=wid,
                        lease_token=token,
                        lease_until=now + timedelta(seconds=lease_seconds),
                        attempts=int(row["attempts"] or 0) + 1,
                    )
                )
                if result.rowcount != 1:
                    _metric("record_outbox_fence_rejected", kind)
                    continue
                if prev_status == "processing":
                    _metric("record_outbox_takeover", kind)
                item = dict(row)
                item.update(
                    status="processing", lease_owner=wid, lease_token=token,
                    attempts=int(row["attempts"] or 0) + 1,
                )
                claimed.append(item)
    except Exception as e:  # noqa: BLE001 —— 领取失败下轮再试
        log.warning("outbox.claim_failed kind=%s err=%s", kind, type(e).__name__)
        return []
    return claimed


def _settle(engine, table, row: dict, kind: str, **values) -> bool:
    """按 (status=processing, lease_token) fencing 结算；并清空租约字段。

    False = 租约已易主/行已终态（旧 worker 不得覆盖）。
    """
    try:
        with engine.begin() as conn:
            result = conn.execute(
                update(table)
                .where(
                    table.c.id == row["id"],
                    table.c.status == "processing",
                    table.c.lease_token == row["lease_token"],
                )
                .values(
                    lease_owner="", lease_token="", lease_until=None, **values,
                )
            )
        if result.rowcount != 1:
            _metric("record_outbox_fence_rejected", kind)
            return False
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("outbox.settle_failed id=%s err=%s", row["id"], type(e).__name__)
        _metric("record_outbox_settle_failure", kind)
        return False


def _finish_ok(engine, table, row: dict, kind: str, extra: dict) -> bool:
    return _settle(engine, table, row, kind, status="done", next_run_at=None, **extra)


def _retry_or_die(engine, table, row: dict, kind: str, error: str) -> None:
    attempts = int(row.get("attempts") or 0)
    if attempts >= max(int(settings.outbox_max_attempts), 1):
        _dead_letter(engine, table, row, kind, reason="max_attempts", error=error)
        return
    delay = _backoff_seconds(attempts)
    if _settle(
        engine, table, row, kind,
        status="pending",
        next_run_at=_now() + timedelta(seconds=delay),
        **{_error_column(table): error},
    ):
        _metric("record_outbox_retry", kind)


def _dead_letter(engine, table, row: dict, kind: str, reason: str, error: str) -> None:
    extra = (
        {"dead_lettered_at": _now()}
        if hasattr(table.c, "dead_lettered_at")
        else {"finished_at": _now()}
    )
    if _settle(
        engine, table, row, kind,
        status="dead_letter", next_run_at=None,
        **{_error_column(table): error}, **extra,
    ):
        _metric("record_outbox_dead_letter", kind, reason)
        log.warning("outbox.dead_letter kind=%s id=%s reason=%s", kind, row["id"], reason)


def _mark_obsolete(engine, table, row: dict, kind: str, reason: str) -> None:
    """Reset 后旧会话（或已有 tombstone）的消息：终态 obsolete，不再重试/写入。"""
    if _settle(
        engine, table, row, kind,
        status="obsolete", next_run_at=None,
        **{_error_column(table): reason},
    ):
        _metric("record_outbox_obsolete", kind)
        log.info("outbox.obsolete kind=%s id=%s reason=%s", kind, row["id"], reason)


def _defer_state(engine, table, row: dict, kind: str, error: str) -> None:
    """状态查询失败：退回 pending 退避（绝不据此 obsolete / 死信）。"""
    delay = _backoff_seconds(int(row.get("attempts") or 1))
    if _settle(
        engine, table, row, kind,
        status="pending",
        next_run_at=_now() + timedelta(seconds=delay),
        **{_error_column(table): error},
    ):
        _metric("record_outbox_retry", kind)
        log.warning("outbox.state_unavailable kind=%s id=%s err=%s", kind, row["id"], error)


# ------------------------------------------------------------
# 会话陈旧判定（Reset 竞态屏障；查询失败 → OutboxStateUnavailable）
# ------------------------------------------------------------
def _load_session_uuid(engine, session_key: str) -> str | None:
    """会话实例 UUID；None = 会话行不存在。查询异常 → OutboxStateUnavailable。"""
    try:
        with engine.connect() as conn:
            return conn.execute(
                select(sessions.c.session_uuid).where(
                    sessions.c.session_key == session_key
                )
            ).scalar_one_or_none()
    except Exception as e:  # noqa: BLE001
        raise OutboxStateUnavailable(
            f"session uuid 查询失败: {type(e).__name__}"
        ) from e


def _load_tombstone_uuids(engine, session_key: str) -> set[str]:
    """该 session_key 的全部删除 tombstone（含空 UUID legacy 事件）。"""
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                select(message_delete_outbox.c.session_uuid).where(
                    message_delete_outbox.c.session_key == session_key
                )
            ).scalars().all()
        return {(r or "") for r in rows}
    except Exception as e:  # noqa: BLE001
        raise OutboxStateUnavailable(
            f"tombstone 查询失败: {type(e).__name__}"
        ) from e


def _stale_verdict(engine, row, cache: dict | None = None) -> str:
    """返回 'ok' | 'obsolete'；任一状态查询异常抛 OutboxStateUnavailable。

    只有「会话不存在」「UUID 明确不同」「明确存在 tombstone」才判 obsolete。
    """
    key = row["session_key"]
    uid = row.get("session_uuid") or ""
    ck = (key, uid)
    if cache is not None and ck in cache:
        return cache[ck]
    current = _load_session_uuid(engine, key)
    if current is None or current != uid:
        verdict = "obsolete"  # 会话已删除，或 UUID 明确不同（Reset 重建）
    else:
        tombstones = _load_tombstone_uuids(engine, key)
        if uid and uid in tombstones:
            verdict = "obsolete"
        elif not uid and tombstones:
            verdict = "obsolete"  # legacy 空 UUID 行：该会话有任何删除事件即陈旧
        else:
            verdict = "ok"
    if cache is not None:
        cache[ck] = verdict
    return verdict


# ------------------------------------------------------------
# 计数
# ------------------------------------------------------------
def count_pending_outbox(engine) -> int:
    try:
        with engine.connect() as conn:
            return int(conn.execute(
                select(func.count())
                .select_from(outbox_rows)
                .where(outbox_rows.c.status.in_(_IN_FLIGHT_STATUSES))
            ).scalar() or 0)
    except Exception:  # noqa: BLE001
        return 0


def count_pending_delete_events(engine) -> tuple[int, float]:
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                select(func.count(), func.min(message_delete_outbox.c.created_at))
                .select_from(message_delete_outbox)
                .where(message_delete_outbox.c.status.in_(_IN_FLIGHT_STATUSES))
            ).one()
        count = int(rows[0] or 0)
        oldest = rows[1]
        lag = (_now() - oldest).total_seconds() if oldest is not None else 0.0
        return count, max(0.0, lag)
    except Exception:  # noqa: BLE001
        return 0, 0.0


# ------------------------------------------------------------
# 消息 → ES
# ------------------------------------------------------------
def sync_outbox_to_es(
    engine, es_client, index: str, batch_size: int = 200, worker_id: str | None = None,
) -> int:
    """搬运一批消息到 ES；返回成功结算条数（含 obsolete），无待处理返回 0。"""
    claimed = _claim(
        engine, outbox_rows, batch_size, settings.outbox_lease_seconds, "message",
        worker_id=worker_id,
    )
    if not claimed:
        return 0

    # 调用前陈旧检查：明确陈旧 → obsolete；状态查询失败 → 退 pending 退避（不写 ES）
    live: list[dict] = []
    settled = 0
    stale_cache: dict = {}
    for row in claimed:
        try:
            verdict = _stale_verdict(engine, row, stale_cache)
        except OutboxStateUnavailable as e:
            _defer_state(engine, outbox_rows, row, "message", f"状态查询失败: {e}")
            continue
        if verdict == "obsolete":
            _mark_obsolete(engine, outbox_rows, row, "message", "session_reset_before_write")
            settled += 1
        else:
            live.append(row)

    if not live:
        return settled

    actions: list[dict] = []
    action_rows: list[dict] = []
    for row in live:
        try:
            msg = json.loads(row["payload"])
        except (json.JSONDecodeError, TypeError) as e:
            _dead_letter(
                engine, outbox_rows, row, "message", reason="json",
                error=f"payload JSON 损坏: {type(e).__name__}",
            )
            continue
        user_id = row["session_key"].split("/", 1)[0]
        actions.append({"index": {
            "_index": index,
            # 文档 ID 含 session UUID：Reset 后 seq 重新从 1 开始也不会覆盖新消息
            "_id": f"{row['session_key']}:{row.get('session_uuid') or ''}:{row['seq']}",
        }})
        actions.append({
            "session_key": row["session_key"],
            "user_id": user_id,
            "session_uuid": row.get("session_uuid") or "",
            "seq": row["seq"],
            "role": msg.get("role", ""),
            "content": msg.get("content", ""),
            "tool_calls": msg.get("tool_calls"),
            "ts": row["created_at"].isoformat() if row["created_at"] else None,
        })
        action_rows.append(row)

    if not actions:
        return settled

    try:
        resp = es_client.bulk(operations=actions, index=index, refresh=False)
    except Exception as e:  # noqa: BLE001 —— 整批失败：逐行退避重试
        _invalidate_es(type(e).__name__)
        for row in action_rows:
            _retry_or_die(
                engine, outbox_rows, row, "message",
                error=f"ES bulk 调用失败: {type(e).__name__}",
            )
        return settled

    items = resp.get("items") or []
    errors = bool(resp.get("errors", False))
    stale_cache.clear()  # ES 调用期间可能发生 Reset：结算前复查需重新读
    for i, row in enumerate(action_rows):
        item = items[i] if i < len(items) else {}
        status = _item_status(item)
        if not errors or (status is not None and status < 300):
            # 结算前复查：Reset → obsolete；状态查询失败 → 退 pending（幂等重写）
            try:
                verdict = _stale_verdict(engine, row, stale_cache)
            except OutboxStateUnavailable as e:
                _defer_state(engine, outbox_rows, row, "message", f"状态查询失败: {e}")
                continue
            if verdict == "obsolete":
                _mark_obsolete(engine, outbox_rows, row, "message", "session_reset_after_write")
            elif _finish_ok(
                engine, outbox_rows, row, "message",
                {"synced_at": _now(), "sync_error": None},
            ):
                settled += 1
        elif status is None or status == _RETRYABLE_STATUS or status >= 500:
            _retry_or_die(engine, outbox_rows, row, "message", error=f"ES item 状态 {status}")
        else:
            _dead_letter(
                engine, outbox_rows, row, "message", reason="deterministic_4xx",
                error=f"ES item 状态 {status}（确定性 4xx）",
            )
    return settled


# ------------------------------------------------------------
# Reset 删除事件 → ES
# ------------------------------------------------------------
def _delete_eligible(conn, row) -> bool:
    """屏障：同 session_key + 旧 UUID（含 legacy 空 UUID）无在途消息行。

    查询失败返回 False（本轮跳过，下轮再试）——绝不因状态读取异常丢弃事件。
    """
    uuids = (row["session_uuid"] or "", "")
    try:
        pending = conn.execute(
            select(func.count()).select_from(outbox_rows).where(
                outbox_rows.c.session_key == row["session_key"],
                outbox_rows.c.session_uuid.in_(uuids),
                outbox_rows.c.status.in_(_IN_FLIGHT_STATUSES),
            )
        ).scalar_one()
    except Exception:  # noqa: BLE001 —— 状态不可读：本轮不领取
        return False
    return int(pending or 0) == 0


def sync_delete_outbox_to_es(
    engine, es_client, index: str, batch_size: int = 100, worker_id: str | None = None,
) -> int:
    """执行一批 ES 删除事件（delete_by_query，幂等），返回成功条数。"""
    claimed = _claim(
        engine, message_delete_outbox, batch_size,
        settings.outbox_lease_seconds, "delete",
        worker_id=worker_id, eligible=_delete_eligible,
    )
    if not claimed:
        return 0
    settled = 0
    for row in claimed:
        if not _delete_documents(es_client, index, row["session_key"], row["session_uuid"] or ""):
            _retry_or_die(
                engine, message_delete_outbox, row, "delete",
                error="delete_by_query 失败",
            )
            continue
        if _finish_ok(
            engine, message_delete_outbox, row, "delete",
            {"finished_at": _now(), "error": None},
        ):
            settled += 1
    return settled


def _delete_documents(es_client, index: str, session_key: str, stale_uuid: str) -> bool:
    """删除旧 UUID + legacy（空/缺失 UUID）文档；True=成功。"""
    query = {"bool": {"filter": [
        {"term": {"session_key": session_key}},
        {"bool": {"should": [
            {"term": {"session_uuid": stale_uuid}},
            {"term": {"session_uuid": ""}},
            {"bool": {"must_not": [{"exists": {"field": "session_uuid"}}]}},
        ], "minimum_should_match": 1}},
    ]}}
    try:
        # refresh=True：删除可见后才标记 done（清 tombstone 前确保旧文档不可见）
        es_client.delete_by_query(index=index, query=query, refresh=True)
        return True
    except Exception as e:  # noqa: BLE001
        _invalidate_es(type(e).__name__)
        log.warning("outbox.delete_failed key=%s err=%s", session_key, type(e).__name__)
        return False


def reap_finished_delete_events(
    engine, es_client, index: str, older_than_seconds: int | None = None,
    limit: int = 100,
) -> int:
    """保留期结束的 done tombstone：再次执行删除，成功后才清理事件行（幂等）。"""
    ttl = (
        settings.message_search_tombstone_ttl_seconds
        if older_than_seconds is None else older_than_seconds
    )
    cutoff = _now() - timedelta(seconds=max(int(ttl), 0))
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                select(message_delete_outbox)
                .where(
                    message_delete_outbox.c.status == "done",
                    message_delete_outbox.c.finished_at.is_not(None),
                    # <= 而非 <：Windows 定时器粒度下 now() 可能相等（保留期已到）
                    message_delete_outbox.c.finished_at <= cutoff,
                )
                .order_by(message_delete_outbox.c.id)
                .limit(max(int(limit), 1))
            ).mappings().all()
    except Exception:  # noqa: BLE001
        return 0
    reaped = 0
    for row in rows:
        if not _delete_documents(es_client, index, row["session_key"], row["session_uuid"] or ""):
            continue  # 删不掉就保留 tombstone（继续过滤 + 下轮再试）
        try:
            with engine.begin() as conn:
                conn.execute(
                    delete(message_delete_outbox).where(
                        message_delete_outbox.c.id == row["id"],
                        message_delete_outbox.c.status == "done",
                    )
                )
            reaped += 1
        except Exception:  # noqa: BLE001
            pass
    return reaped


def load_message_tombstones(engine, user_id: str) -> dict[str, set[str]]:
    """该用户全部未物理清理的删除 tombstone：{session_key: {旧 uuid, ...}}。

    **fail-closed**（修复计划·二轮 4）：DB 缺失或查询失败抛
    MessageTombstoneUnavailable（搜索返回 503），绝不返回空集合假象。
    done/pending/processing/dead_letter 在物理清理前都参与过滤（含 legacy 空 UUID）。
    """
    if engine is None:
        raise MessageTombstoneUnavailable("删除 tombstone 存储未配置（DB 缺失）")
    prefix = f"{user_id}/"
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                select(
                    message_delete_outbox.c.session_key,
                    message_delete_outbox.c.session_uuid,
                ).where(message_delete_outbox.c.session_key.like(prefix + "%"))
            ).all()
    except Exception as e:  # noqa: BLE001
        raise MessageTombstoneUnavailable(
            f"删除 tombstone 查询失败: {type(e).__name__}"
        ) from e
    out: dict[str, set[str]] = {}
    for key, stale in rows:
        out.setdefault(key, set()).add(stale or "")
    return out


# ------------------------------------------------------------
# 索引 mapping
# ------------------------------------------------------------
_MESSAGE_MAPPING = {
    "properties": {
        "session_key": {"type": "keyword"},
        "user_id": {"type": "keyword"},
        "session_uuid": {"type": "keyword"},
        "seq": {"type": "integer"},
        "role": {"type": "keyword"},
        "content": {"type": "text"},
        "tool_calls": {"type": "object", "enabled": False},
        "ts": {"type": "date"},
    }
}


def ensure_message_index(es_client, index: str) -> None:
    """消息检索索引：不存在则创建；已存在则 put_mapping 补齐（不重建索引）。"""
    try:
        exists = es_client.indices.exists(index=index)
    except Exception:  # noqa: BLE001
        _invalidate_es("indices_exists_failed")
        raise
    if exists:
        try:
            es_client.indices.put_mapping(index=index, properties=_MESSAGE_MAPPING["properties"])
        except Exception:  # noqa: BLE001 —— 补列失败不影响既有检索
            log.warning("message_search put_mapping 失败（session_uuid 补齐）", exc_info=True)
        return
    es_client.indices.create(
        index=index,
        settings={"number_of_shards": 1, "number_of_replicas": 0},
        mappings=_MESSAGE_MAPPING,
    )


def run_outbox_once(engine, es_client, index: str, worker_id: str | None = None) -> int:
    """一轮搬运：消息 + 删除事件 + 保留期复查；返回处理总条数。

    修复计划·三轮 P2-1：消费者**始终**处理/重试/清理已存在的删除事件；
    MESSAGE_DELETE_OUTBOX_ENABLED 只是生产开关（控制 Reset 是否写入新事件）。
    """
    if engine is None or es_client is None:
        return 0
    done = sync_outbox_to_es(engine, es_client, index, worker_id=worker_id)
    done += sync_delete_outbox_to_es(engine, es_client, index, worker_id=worker_id)
    done += reap_finished_delete_events(engine, es_client, index)
    return done
