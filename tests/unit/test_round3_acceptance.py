"""修复计划·三轮验收：搜索重读 tombstone / OutboxStateUnavailable /
空 UUID legacy Reset 删除事件。"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.pool import StaticPool

from app.stores.base import SessionState
from app.stores.sql import outbox as ob
from app.stores.sql.outbox import (
    OutboxStateUnavailable,
    sync_delete_outbox_to_es,
    sync_outbox_to_es,
)
from app.stores.sql.schema import (
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


def _state(session_id="uuid-1", messages=None):
    return SessionState(
        session_id=session_id, user_id="u1", summary=None,
        messages=messages or [], version=0, updated_at="",
    )


def _seed_session(engine, key="u1/s1", uid="uuid-1"):
    with engine.begin() as conn:
        conn.execute(sessions.insert().values(
            session_key=key, user_id=key.split("/", 1)[0],
            session_uuid=uid, version=0, status="active",
        ))


def _seed_outbox(engine, key="u1/s1", uid="uuid-1", seq=1, status="pending"):
    with engine.begin() as conn:
        conn.execute(outbox_rows.insert().values(
            session_key=key, seq=seq, session_uuid=uid,
            payload=json.dumps({"role": "user", "content": "hi"}), status=status,
        ))


class _BulkES:
    def __init__(self):
        self.ops: list = []

    def bulk(self, operations=None, index=None, refresh=False):
        self.ops = operations
        n = len(operations) // 2
        return {"errors": False, "items": [{"index": {"status": 201}} for _ in range(n)]}


# ------------------------------------------------------------
# P1-1：ES 查询期间发生 Reset → 二次读取必须拦住旧文档
# ------------------------------------------------------------
def test_search_reloads_tombstones_after_es_query(monkeypatch):
    import app.server.main as main_mod
    from test_server_api import _FakeComponents, _ScriptedAgent

    engine = _engine()
    store = SqlSessionStore(engine, outbox_enabled=True)
    store.save("u1", "s1", _state(messages=[{"role": "user", "content": "旧消息"}]))

    docs = [
        # 旧 UUID 文档
        {"session_key": "u1/s1", "session_uuid": "uuid-1", "seq": 1,
         "role": "user", "content": "旧消息", "ts": "2026-09-12T10:00:01"},
        # legacy 无 UUID 文档
        {"session_key": "u1/s1", "seq": 2, "role": "user",
         "content": "legacy 旧消息", "ts": "2026-09-12T10:00:02"},
    ]

    class _ResetDuringSearchES:
        def search(self, **kwargs):
            store.delete("u1", "s1")  # 查询期间 Reset（提交删除事件）
            return {"hits": {"hits": [{"_source": d} for d in docs]}}

    comps = _FakeComponents()
    comps.db_engine = engine
    comps.es_provider = lambda: _ResetDuringSearchES()
    comps.message_index = "ecom-messages"
    monkeypatch.setattr(main_mod, "build_pod_components", lambda: comps)
    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda uid, sid="", components=None, **_k: _ScriptedAgent(uid, sid),
    )
    with TestClient(main_mod.create_app()) as client:
        resp = client.get("/v1/messages/search", params={"user_id": "u1", "q": "消息"})
    assert resp.status_code == 200
    assert resp.json()["hits"] == []  # 旧 UUID 与 legacy 均不返回


def test_search_503_when_second_tombstone_read_fails(monkeypatch):
    import app.server.main as main_mod
    from test_server_api import _FakeComponents, _ScriptedAgent

    real = _engine()
    store = SqlSessionStore(real, outbox_enabled=True)
    store.save("u1", "s1", _state(messages=[{"role": "user", "content": "x"}]))

    class _FlakyEngine:
        """第一次 connect 正常（查询前读取），之后失败（查询后重读）。"""

        def __init__(self, engine):
            self._engine = engine
            self.calls = 0

        def connect(self):
            self.calls += 1
            if self.calls > 1:
                raise ConnectionError("db down")
            return self._engine.connect()

    class _ES:
        def search(self, **kwargs):
            return {"hits": {"hits": []}}

    comps = _FakeComponents()
    comps.db_engine = _FlakyEngine(real)
    comps.es_provider = lambda: _ES()
    comps.message_index = "ecom-messages"
    monkeypatch.setattr(main_mod, "build_pod_components", lambda: comps)
    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda uid, sid="", components=None, **_k: _ScriptedAgent(uid, sid),
    )
    with TestClient(main_mod.create_app()) as client:
        resp = client.get("/v1/messages/search", params={"user_id": "u1", "q": "x"})
    assert resp.status_code == 503
    assert resp.json()["code"] == "message_tombstone_unavailable"


# ------------------------------------------------------------
# P1-2：状态查询异常 → 退 pending 可重试（不 obsolete、不死信）
# ------------------------------------------------------------
@pytest.mark.parametrize("fail_fn", ["_load_session_uuid", "_load_tombstone_uuids"])
def test_state_query_failure_defers_message(monkeypatch, fail_fn):
    engine = _engine()
    _seed_session(engine)
    _seed_outbox(engine)

    def _boom(*a, **k):
        raise OutboxStateUnavailable("db down")

    monkeypatch.setattr(ob, fail_fn, _boom)
    # attempts 已超上限也绝不因状态故障进死信
    monkeypatch.setattr(ob.settings, "outbox_max_attempts", 1)
    es = _BulkES()
    assert sync_outbox_to_es(engine, es, "ecom-messages") == 0
    assert es.ops == []  # 不写 ES
    with engine.connect() as conn:
        row = conn.execute(outbox_rows.select()).mappings().one()
    assert row["status"] == "pending"  # 可重试，非 obsolete/dead_letter
    assert row["next_run_at"] is not None
    assert "状态查询失败" in (row["sync_error"] or "")

    # 恢复后成功写入 ES（退避到点，模拟下轮 worker）
    monkeypatch.undo()
    with engine.begin() as conn:
        conn.execute(outbox_rows.update().values(next_run_at=None))
    es2 = _BulkES()
    assert sync_outbox_to_es(engine, es2, "ecom-messages") == 1
    assert len(es2.ops) == 2
    with engine.connect() as conn:
        assert conn.execute(outbox_rows.select()).mappings().one()["status"] == "done"


def test_uuid_mismatch_and_missing_session_are_obsolete():
    engine = _engine()
    # 会话 UUID 与行不一致 → obsolete
    _seed_session(engine, uid="new-uuid")
    _seed_outbox(engine, uid="old-uuid")
    es = _BulkES()
    assert sync_outbox_to_es(engine, es, "ecom-messages") == 1
    assert es.ops == []
    with engine.connect() as conn:
        assert conn.execute(outbox_rows.select()).mappings().one()["status"] == "obsolete"


def test_missing_session_row_is_obsolete():
    engine = _engine()
    _seed_outbox(engine, uid="uuid-1")  # 无 sessions 行
    es = _BulkES()
    assert sync_outbox_to_es(engine, es, "ecom-messages") == 1
    with engine.connect() as conn:
        assert conn.execute(outbox_rows.select()).mappings().one()["status"] == "obsolete"


# ------------------------------------------------------------
# P1-3：空 UUID legacy 会话 Reset 产生并消费删除事件
# ------------------------------------------------------------
def test_legacy_empty_uuid_reset_creates_and_consumes_delete_event():
    engine = _engine()
    _seed_session(engine, uid="")  # legacy：存在但 UUID 为空
    _seed_outbox(engine, uid="")
    store = SqlSessionStore(engine, outbox_enabled=True)

    store.delete("u1", "s1")
    with engine.connect() as conn:
        events = conn.execute(message_delete_outbox.select()).mappings().all()
    assert len(events) == 1
    assert events[0]["session_uuid"] == ""  # 空 UUID 事件

    class _DeleteES:
        def __init__(self):
            self.queries: list = []

        def delete_by_query(self, index=None, query=None, refresh=True):
            self.queries.append(query)
            return {"deleted": 1}

    es = _DeleteES()
    assert sync_delete_outbox_to_es(engine, es, "ecom-messages") == 1
    should = es.queries[0]["bool"]["filter"][1]["bool"]["should"]
    assert {"term": {"session_uuid": ""}} in should
    assert {"bool": {"must_not": [{"exists": {"field": "session_uuid"}}]}} in should


# ------------------------------------------------------------
# P2-1：消费者与生产开关解耦（存量事件始终处理）
# ------------------------------------------------------------
def test_delete_consumer_runs_even_when_production_switch_off(monkeypatch):
    engine = _engine()
    store = SqlSessionStore(engine, outbox_enabled=True)
    store.save("u1", "s1", _state(messages=[{"role": "user", "content": "a"}]))
    store.delete("u1", "s1")  # 事件已存在

    class _DeleteES:
        def __init__(self):
            self.calls = 0

        def delete_by_query(self, index=None, query=None, refresh=True):
            self.calls += 1
            return {"deleted": 1}

    monkeypatch.setattr(ob.settings, "message_delete_outbox_enabled", False)
    es = _DeleteES()
    from app.stores.sql.outbox import run_outbox_once

    run_outbox_once(engine, es, "ecom-messages")
    assert es.calls == 1  # 生产开关关闭，消费者仍处理存量事件
