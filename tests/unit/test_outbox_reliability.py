"""修复计划·二：Reset 删除事件 + Outbox 可靠性测试。

覆盖：
- Reset 事务写唯一 ES 删除事件（旧 session_uuid）并删除 MySQL 正本；
- 删除事件 delete_by_query 按旧 UUID + legacy 空 UUID 过滤，新 UUID 不受影响；
- 运营搜索 tombstone 过滤（重置后立即不可搜索）；
- Outbox 每行独立处理：坏 JSON dead-letter 不阻塞后续行；
- 确定性 4xx → dead-letter；5xx → 退避重试（lease token 结算）。
"""

from __future__ import annotations

import json

from sqlalchemy import create_engine, select
from sqlalchemy.pool import StaticPool

from app.stores.base import SessionState
from app.stores.sql.outbox import (
    load_message_tombstones,
    sync_delete_outbox_to_es,
    sync_outbox_to_es,
)
from app.stores.sql.schema import (
    chat_messages,
    message_delete_outbox,
    metadata,
    outbox_rows,
    sessions,
)
from app.stores.sql.session_store import SqlSessionStore


def _engine():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    metadata.create_all(engine)
    return engine


def _state(session_id="s-uuid-1", version=0, messages=None):
    return SessionState(
        session_id=session_id, user_id="u1", summary=None,
        messages=messages or [], version=version, updated_at="",
    )


def _msg(role, content):
    return {"role": role, "content": content}


# ------------------------------------------------------------
# Reset → 删除事件
# ------------------------------------------------------------
def test_reset_writes_delete_event_and_clears_mysql():
    engine = _engine()
    store = SqlSessionStore(engine, outbox_enabled=True)
    store.save("u1", "s1", _state(messages=[_msg("user", "你好")]))
    with engine.connect() as conn:
        old_uuid = conn.execute(
            select(sessions.c.session_uuid).where(sessions.c.session_key == "u1/s1")
        ).scalar_one()

    store.delete("u1", "s1")

    with engine.connect() as conn:
        events = conn.execute(message_delete_outbox.select()).mappings().all()
        assert conn.execute(chat_messages.select()).all() == []
        assert conn.execute(sessions.select()).all() == []
        # 修复计划·二轮 3：旧 UUID 的 pending 消息标 obsolete（保留为审计/屏障），
        # 不再直接整表删除
        outbox_all = conn.execute(outbox_rows.select()).mappings().all()
        assert all(r["status"] == "obsolete" for r in outbox_all)
    assert len(events) == 1
    assert events[0]["session_key"] == "u1/s1"
    assert events[0]["session_uuid"] == old_uuid
    assert events[0]["status"] == "pending"


def test_reset_without_session_row_writes_no_event():
    engine = _engine()
    store = SqlSessionStore(engine, outbox_enabled=True)
    store.delete("u1", "never")  # 无 sessions 行 → 无删除事件
    with engine.connect() as conn:
        assert conn.execute(message_delete_outbox.select()).all() == []


# ------------------------------------------------------------
# 删除事件执行：旧 UUID + legacy 空 UUID
# ------------------------------------------------------------
class _DeleteES:
    def __init__(self):
        self.calls: list[dict] = []

    def delete_by_query(self, index=None, query=None, refresh=False):
        self.calls.append({"index": index, "query": query})
        return {"deleted": 1}


def test_delete_event_targets_stale_uuid_and_legacy():
    engine = _engine()
    store = SqlSessionStore(engine, outbox_enabled=True)
    store.save("u1", "s1", _state(session_id="old-uuid", messages=[_msg("user", "a")]))
    store.delete("u1", "s1")

    es = _DeleteES()
    assert sync_delete_outbox_to_es(engine, es, "ecom-messages") == 1
    assert len(es.calls) == 1
    q = es.calls[0]["query"]
    assert q["bool"]["filter"][0] == {"term": {"session_key": "u1/s1"}}
    should = q["bool"]["filter"][1]["bool"]["should"]
    assert {"term": {"session_uuid": "old-uuid"}} in should
    assert {"term": {"session_uuid": ""}} in should  # legacy 空 UUID
    assert {"bool": {"must_not": [{"exists": {"field": "session_uuid"}}]}} in should

    with engine.connect() as conn:
        row = conn.execute(message_delete_outbox.select()).mappings().one()
    assert row["status"] == "done"


def test_delete_event_idempotent_second_run_noop():
    engine = _engine()
    store = SqlSessionStore(engine, outbox_enabled=True)
    store.save("u1", "s1", _state(session_id="old-uuid", messages=[_msg("user", "a")]))
    store.delete("u1", "s1")
    es = _DeleteES()
    assert sync_delete_outbox_to_es(engine, es, "ecom-messages") == 1
    assert sync_delete_outbox_to_es(engine, es, "ecom-messages") == 0  # 已完成


# ------------------------------------------------------------
# tombstone 过滤
# ------------------------------------------------------------
def test_load_message_tombstones_keeps_done_during_retention():
    """修复计划·二轮 4：done tombstone 在物理清理前仍参与过滤（防删除可见性延迟）。"""
    engine = _engine()
    store = SqlSessionStore(engine, outbox_enabled=True)
    store.save("u1", "s1", _state(session_id="old-uuid", messages=[_msg("user", "a")]))
    store.delete("u1", "s1")
    tombstones = load_message_tombstones(engine, "u1")
    assert tombstones == {"u1/s1": {"old-uuid"}}

    # 删除事件完成后（done）在保留期内仍返回该 tombstone
    sync_delete_outbox_to_es(engine, _DeleteES(), "ecom-messages")
    assert load_message_tombstones(engine, "u1") == {"u1/s1": {"old-uuid"}}


# ------------------------------------------------------------
# Outbox 每行独立处理
# ------------------------------------------------------------
class _BulkES:
    """记录 bulk 调用；按需对指定下标返回失败状态。"""

    def __init__(self, fail_status: dict | None = None):
        self.fail_status = fail_status or {}
        self.ops: list = []

    def bulk(self, operations=None, index=None, refresh=False):
        self.ops = operations
        n = len(operations) // 2
        items = []
        for i in range(n):
            if i in self.fail_status:
                items.append({"index": {"status": self.fail_status[i], "error": {"reason": "x"}}})
            else:
                items.append({"index": {"status": 201}})
        return {"errors": bool(self.fail_status), "items": items}


def test_bad_json_row_dead_lettered_without_blocking_others():
    engine = _engine()
    store = SqlSessionStore(engine, outbox_enabled=True)
    store.save("u1", "s1", _state(messages=[_msg("user", "ok1"), _msg("user", "ok2")]))
    with engine.begin() as conn:
        # 把第一行 payload 改坏
        first = conn.execute(select(outbox_rows.c.id).order_by(outbox_rows.c.id)).scalars().first()
        conn.execute(
            outbox_rows.update().where(outbox_rows.c.id == first).values(payload="{bad json")
        )

    es = _BulkES()
    settled = sync_outbox_to_es(engine, es, "ecom-messages")
    assert settled == 1  # 正常行仍被同步
    with engine.connect() as conn:
        rows = conn.execute(
            outbox_rows.select().order_by(outbox_rows.c.id)
        ).mappings().all()
    assert rows[0]["status"] == "dead_letter"
    assert rows[0]["dead_lettered_at"] is not None
    assert "JSON" in (rows[0]["sync_error"] or "")
    assert rows[1]["status"] == "done"


def test_deterministic_4xx_dead_lettered_5xx_retried():
    engine = _engine()
    store = SqlSessionStore(engine, outbox_enabled=True)
    store.save("u1", "s1", _state(messages=[_msg("user", "a"), _msg("user", "b")]))
    es = _BulkES(fail_status={0: 400, 1: 503})
    assert sync_outbox_to_es(engine, es, "ecom-messages") == 0
    with engine.connect() as conn:
        rows = conn.execute(
            outbox_rows.select().order_by(outbox_rows.c.id)
        ).mappings().all()
    assert rows[0]["status"] == "dead_letter"  # 确定性 4xx
    assert "400" in (rows[0]["sync_error"] or "")
    assert rows[1]["status"] == "pending"  # 5xx → 退避重试
    assert rows[1]["attempts"] == 1
    assert rows[1]["next_run_at"] is not None


def test_max_attempts_dead_letters():
    from app.config.settings import settings

    engine = _engine()
    store = SqlSessionStore(engine, outbox_enabled=True)
    store.save("u1", "s1", _state(messages=[_msg("user", "a")]))
    # 预置 attempts 已达上限（模拟多次退避后）
    with engine.begin() as conn:
        conn.execute(outbox_rows.update().values(attempts=settings.outbox_max_attempts))
    es = _BulkES(fail_status={0: 500})
    assert sync_outbox_to_es(engine, es, "ecom-messages") == 0
    with engine.connect() as conn:
        row = conn.execute(outbox_rows.select()).mappings().one()
    assert row["status"] == "dead_letter"


# ------------------------------------------------------------
# 运营搜索 tombstone 过滤（端点层）
# ------------------------------------------------------------
class _SearchES:
    def __init__(self, docs):
        self.docs = docs

    def search(self, index=None, query=None, sort=None, size=10, source=None, **kwargs):
        return {"hits": {"hits": [
            {"_id": f"{d['session_key']}:{d['seq']}", "_source": d} for d in self.docs[:size]
        ]}}


def test_message_search_filters_tombstoned_uuid(monkeypatch):
    from fastapi.testclient import TestClient

    import app.server.main as main_mod
    from test_server_api import _FakeComponents

    engine = _engine()
    store = SqlSessionStore(engine, outbox_enabled=True)
    store.save("u1", "s1", _state(session_id="old-uuid", messages=[_msg("user", "旧消息")]))
    store.delete("u1", "s1")  # 留下 pending 删除事件（ES 尚未删除）

    docs = [
        {"session_key": "u1/s1", "user_id": "u1", "session_uuid": "old-uuid",
         "seq": 1, "role": "user", "content": "旧消息", "ts": "2026-09-12T10:00:01"},
        {"session_key": "u1/s1", "user_id": "u1", "session_uuid": "new-uuid",
         "seq": 2, "role": "user", "content": "新消息", "ts": "2026-09-12T10:00:02"},
        {"session_key": "u1/s1", "user_id": "u1", "seq": 3, "role": "user",
         "content": "legacy 无 uuid", "ts": "2026-09-12T10:00:03"},
    ]
    comps = _FakeComponents()
    comps.es_provider = lambda: _SearchES(docs)
    comps.message_index = "ecom-messages"
    comps.db_engine = engine
    monkeypatch.setattr(main_mod, "build_pod_components", lambda: comps)
    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda uid, sid="", comps=None, **_k: __import__("test_server_api")._ScriptedAgent(uid, sid),
    )
    with TestClient(main_mod.create_app()) as client:
        resp = client.get("/v1/messages/search", params={"user_id": "u1", "q": "消息"})
    assert resp.status_code == 200
    contents = [h["content"] for h in resp.json()["hits"]]
    assert "旧消息" not in contents  # 旧 UUID 被 tombstone 过滤
    assert "legacy 无 uuid" not in contents  # legacy 空 UUID 同样过滤
    assert "新消息" in contents  # 新会话实例不受影响


# ------------------------------------------------------------
# dead-letter CLI
# ------------------------------------------------------------
def test_outbox_admin_list_and_replay(monkeypatch, capsys):
    from app.scripts import outbox_admin
    from app.stores.sql import engine as engine_mod

    engine = _engine()
    store = SqlSessionStore(engine, outbox_enabled=True)
    store.save("u1", "s1", _state(messages=[_msg("user", "a")]))
    es = _BulkES(fail_status={0: 400})  # 确定性 4xx → dead_letter
    sync_outbox_to_es(engine, es, "ecom-messages")
    monkeypatch.setattr(engine_mod, "get_engine", lambda: engine)

    assert outbox_admin.main(["list", "--kind", "message"]) == 0
    out = capsys.readouterr().out
    listed = json.loads(out[out.index("["):out.rindex("]") + 1])  # 跳过日志行
    assert len(listed) == 1 and listed[0]["kind"] == "message"

    assert outbox_admin.main(["replay", "--kind", "message", "--all"]) == 0
    with engine.connect() as conn:
        row = conn.execute(outbox_rows.select()).mappings().one()
    assert row["status"] == "pending"
    assert row["attempts"] == 0
    assert row["sync_error"] is None
