"""修复计划·二轮：Outbox 租约状态机 / Reset 竞态 / tombstone fail-closed /
shutdown 锁语义 / 预算故障契约 / 后台记忆计费。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.pool import StaticPool

from app.security.ratelimit import (
    BudgetStoreUnavailable,
    UserLimiter,
    budget_user_scope,
    current_budget_user,
)
from app.stores.base import SessionState
from app.stores.sql import outbox as ob
from app.stores.sql.outbox import (
    MessageTombstoneUnavailable,
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


def _state(session_id="u1-uuid", version=0, messages=None):
    return SessionState(
        session_id=session_id, user_id="u1", summary=None,
        messages=messages or [], version=version, updated_at="",
    )


def _msg(role, content):
    return {"role": role, "content": content}


def _seed_pending(engine, key="u1/s1", uid="u1-uuid", seq=1, status="pending"):
    """直接插入一条 outbox 行（含 sessions 行以通过陈旧检查）。"""
    with engine.begin() as conn:
        conn.execute(sessions.insert().values(
            session_key=key, user_id=key.split("/", 1)[0],
            session_uuid=uid, version=0, status="active",
        ))
        conn.execute(outbox_rows.insert().values(
            session_key=key, seq=seq, session_uuid=uid,
            payload=json.dumps(_msg("user", "hi")), status=status,
        ))


class _BulkES:
    def __init__(self):
        self.ops: list = []

    def bulk(self, operations=None, index=None, refresh=False):
        self.ops = operations
        n = len(operations) // 2
        return {"errors": False, "items": [{"index": {"status": 201}} for _ in range(n)]}


class _DeleteES:
    def __init__(self):
        self.calls: list = []

    def delete_by_query(self, index=None, query=None, refresh=True):
        self.calls.append(query)
        return {"deleted": 1}


# ------------------------------------------------------------
# 2：租约状态机（接管 / fencing / worker_id）
# ------------------------------------------------------------
def test_claim_takes_over_expired_processing_and_sets_worker_id():
    engine = _engine()
    _seed_pending(engine, status="processing")
    with engine.begin() as conn:
        conn.execute(outbox_rows.update().values(
            lease_owner="dead", lease_token="oldtoken",
            lease_until=datetime.now() - timedelta(seconds=5), attempts=1,
        ))
    claimed = ob._claim(
        engine, outbox_rows, 10, 60, "message", worker_id="w-new",
    )
    assert len(claimed) == 1
    assert claimed[0]["lease_owner"] == "w-new"
    assert claimed[0]["lease_token"] != "oldtoken"
    assert claimed[0]["attempts"] == 2  # 接管计数


def test_stale_token_cannot_settle():
    engine = _engine()
    _seed_pending(engine, status="processing")
    with engine.begin() as conn:
        conn.execute(outbox_rows.update().values(
            status="processing", lease_token="tok-new", lease_owner="w2",
        ))
        row = conn.execute(outbox_rows.select()).mappings().one()
    stale = dict(row)  # 旧 worker 手里的 row（token 已被接管）
    stale["lease_token"] = "tok-old"
    assert ob._settle(engine, outbox_rows, stale, "message", status="done") is False
    with engine.connect() as conn:
        cur = conn.execute(outbox_rows.select()).mappings().one()
    assert cur["status"] == "processing"  # 旧 token 不得覆盖
    assert cur["lease_token"] == "tok-new"


def test_worker_id_is_effective_per_batch():
    engine = _engine()
    _seed_pending(engine)
    claimed = ob._claim(engine, outbox_rows, 5, 60, "message", worker_id="explicit-w")
    assert claimed and claimed[0]["lease_owner"] == "explicit-w"


# ------------------------------------------------------------
# 3：Reset 竞态
# ------------------------------------------------------------
def test_reset_marks_pending_obsolete_keeps_processing_barrier():
    engine = _engine()
    store = SqlSessionStore(engine, outbox_enabled=True)
    store.save("u1", "s1", _state(session_id="old-uuid", messages=[_msg("user", "a")]))
    with engine.begin() as conn:
        conn.execute(outbox_rows.update().values(status="processing"))

    # 再造一条 pending
    with engine.begin() as conn:
        conn.execute(outbox_rows.insert().values(
            session_key="u1/s1", seq=2, session_uuid="old-uuid",
            payload=json.dumps(_msg("user", "b")), status="pending",
        ))
    store.delete("u1", "s1")
    with engine.connect() as conn:
        rows = conn.execute(outbox_rows.select().order_by(outbox_rows.c.seq)).mappings().all()
    by_seq = {r["seq"]: r["status"] for r in rows}
    assert by_seq[1] == "processing"  # 在途屏障保留
    assert by_seq[2] == "obsolete"  # pending → obsolete


def test_old_worker_skips_stale_after_reset():
    """旧 worker 领取后发生 Reset：结算为 obsolete，不写 ES。"""
    engine = _engine()
    store = SqlSessionStore(engine, outbox_enabled=True)
    store.save("u1", "s1", _state(session_id="old-uuid", messages=[_msg("user", "a")]))
    claimed = ob._claim(engine, outbox_rows, 5, 60, "message")  # 旧 worker 领取
    assert len(claimed) == 1
    store.delete("u1", "s1")  # Reset（旧 UUID 出现 tombstone）
    # 模拟旧 worker 租约到期（新 worker 可接管），此时应判陈旧 → obsolete
    with engine.begin() as conn:
        conn.execute(outbox_rows.update().values(
            lease_until=datetime.now() - timedelta(seconds=1),
        ))

    es = _BulkES()
    settled = sync_outbox_to_es(engine, es, "ecom-messages")  # 新 worker 轮询
    assert es.ops == []  # 未向 ES 写入旧会话数据
    with engine.connect() as conn:
        row = conn.execute(outbox_rows.select()).mappings().one()
    assert row["status"] == "obsolete"
    assert settled == 1  # obsolete 计入已处理


def test_delete_event_waits_for_message_terminal_state():
    engine = _engine()
    store = SqlSessionStore(engine, outbox_enabled=True)
    store.save("u1", "s1", _state(session_id="old-uuid", messages=[_msg("user", "a")]))
    store.delete("u1", "s1")  # 生成删除事件 + pending→obsolete

    # 制造一条在途 processing（屏障）
    with engine.begin() as conn:
        conn.execute(outbox_rows.update().values(status="processing"))
    es = _DeleteES()
    assert sync_delete_outbox_to_es(engine, es, "ecom-messages") == 0  # 屏障未清，不领取
    assert es.calls == []

    # 屏障进入终态后可领取
    with engine.begin() as conn:
        conn.execute(outbox_rows.update().values(status="done"))
    assert sync_delete_outbox_to_es(engine, es, "ecom-messages") == 1
    assert len(es.calls) == 1


def test_doc_id_includes_session_uuid():
    engine = _engine()
    _seed_pending(engine, uid="uuid-x", seq=1)
    es = _BulkES()
    sync_outbox_to_es(engine, es, "ecom-messages")
    assert es.ops[0]["index"]["_id"] == "u1/s1:uuid-x:1"


def test_reap_delete_event_reexecutes_before_purge():
    engine = _engine()
    store = SqlSessionStore(engine, outbox_enabled=True)
    store.save("u1", "s1", _state(session_id="old-uuid", messages=[_msg("user", "a")]))
    store.delete("u1", "s1")
    es = _DeleteES()
    sync_delete_outbox_to_es(engine, es, "ecom-messages")  # → done
    assert len(es.calls) == 1

    # 保留期结束：复查删除（再次 delete_by_query）后才清理事件行
    n = ob.reap_finished_delete_events(engine, es, "ecom-messages", older_than_seconds=0)
    assert n == 1
    assert len(es.calls) == 2  # 清理前再删一次
    with engine.connect() as conn:
        assert conn.execute(message_delete_outbox.select()).all() == []

    # 复查失败（ES 抛错）→ 保留 tombstone
    store.save("u1", "s2", _state(session_id="old2", messages=[_msg("user", "b")]))
    store.delete("u1", "s2")
    sync_delete_outbox_to_es(engine, es, "ecom-messages")

    class _DownES:
        def delete_by_query(self, **kw):
            raise ConnectionError("down")

    assert ob.reap_finished_delete_events(
        engine, _DownES(), "ecom-messages", older_than_seconds=0
    ) == 0
    with engine.connect() as conn:
        assert len(conn.execute(message_delete_outbox.select()).all()) == 1


# ------------------------------------------------------------
# 4：tombstone fail-closed
# ------------------------------------------------------------
def test_load_tombstones_raises_when_db_missing():
    with pytest.raises(MessageTombstoneUnavailable):
        load_message_tombstones(None, "u1")


def test_load_tombstones_raises_on_query_failure():
    class _BrokenEngine:
        def connect(self):
            raise ConnectionError("db down")

    with pytest.raises(MessageTombstoneUnavailable):
        load_message_tombstones(_BrokenEngine(), "u1")


def test_search_503_with_code_when_tombstone_unavailable(monkeypatch):
    import app.server.main as main_mod
    from test_server_api import _FakeComponents, _ScriptedAgent

    class _BrokenEngine:
        def connect(self):
            raise ConnectionError("db down")

    comps = _FakeComponents()
    comps.db_engine = _BrokenEngine()
    comps.es_provider = lambda: object()
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
# 6：shutdown 锁语义
# ------------------------------------------------------------
class _StubRedis:
    def __init__(self):
        self.kv = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        return True

    def get(self, key):
        return self.kv.get(key)

    def delete(self, key):
        return self.kv.pop(key, None)

    def eval(self, script, numkeys, key, *args):
        if "expire" in script:
            return 1 if self.kv.get(key) == args[0] else 0
        if self.kv.get(key) == args[0]:
            del self.kv[key]
            return 1
        return 0


def test_stop_renew_does_not_delete_lock():
    from app.stores.locks import SessionLease, SessionLockManager

    r = _StubRedis()
    mgr = SessionLockManager(r, ttl_seconds=60)
    lease = SessionLease(mgr, "u1", "s1")
    lease.__enter__()
    token = lease.token
    assert r.kv.get("session_lock:u1:s1") == token

    lease.stop_renew()  # 关闭超时移交：只停续租
    assert lease.abandoned is True
    lease.release()  # abandoned → 不主动删锁
    assert r.kv.get("session_lock:u1:s1") == token  # 仍在，等 TTL 回收


def test_normal_release_still_deletes_lock():
    from app.stores.locks import SessionLease, SessionLockManager

    r = _StubRedis()
    lease = SessionLease(SessionLockManager(r, ttl_seconds=60), "u1", "s1")
    lease.__enter__()
    lease.release()
    assert r.kv.get("session_lock:u1:s1") is None  # 正常完成立即释放


# ------------------------------------------------------------
# 5：ES provider 单源 + 按操作取客户端
# ------------------------------------------------------------
def test_es_backend_uses_provider_per_operation(monkeypatch, reset_settings):
    from app.agent.rag.backends.es_backend import ESBackend

    calls = {"n": 0}

    class _ES:
        def __init__(self, tag):
            self.tag = tag
            self.indices = self

        def get_alias(self, name=None):
            return {f"{self.tag}": {}}

        def get_mapping(self, index=None):
            return {index: {"mappings": {"_meta": {"embedding_model": "m"}}}}

    def provider():
        calls["n"] += 1
        return _ES(f"idx-{calls['n']}")

    backend = ESBackend(es_provider=provider)
    assert backend._resolve_index() == "idx-1"
    backend._index = ""  # 强制重新解析 → 走 provider 第二次
    assert backend._resolve_index() == "idx-2"
    assert calls["n"] == 2  # 不是长期持有同一个实例


# ------------------------------------------------------------
# 7：预算故障契约 / budget_user_scope / Memory Worker 计费
# ------------------------------------------------------------
def test_budget_user_scope_restores_previous():
    outer = current_budget_user()
    with budget_user_scope("outer"):
        assert current_budget_user() == "outer"
        with budget_user_scope("inner"):
            assert current_budget_user() == "inner"
        assert current_budget_user() == "outer"  # 退出恢复
    assert current_budget_user() == outer


def test_llm_rejects_anonymous_when_budget_configured():
    from app.llm.client import ResilientLLM

    limiter = UserLimiter(None, daily_token_budget=1000)
    rl = ResilientLLM(object(), "m", limiter=limiter)
    with budget_user_scope(""):  # 显式空用户（后台任务缺归属）
        with pytest.raises(BudgetStoreUnavailable):
            rl._invoke(lambda: object(), {"messages": [], "max_tokens": 1})


def test_llm_unset_user_does_not_reserve_or_reject():
    """从未绑定（离线 CLI）：不预留、不拒绝（不误伤演进/评测离线调用）。"""
    from app.llm.client import ResilientLLM

    limiter = UserLimiter(None, daily_token_budget=1000)
    rl = ResilientLLM(object(), "m", limiter=limiter)
    with budget_user_scope(current_budget_user()):  # 保持未绑定哨兵
        pass
    sentinel = current_budget_user()
    if sentinel == "":
        pytest.skip("ContextVar 被其他用例污染为显式空用户")
    out = rl._invoke(lambda: object(), {"messages": [], "max_tokens": 1})
    assert out is not None
    assert limiter._resv_meta == {}  # 未建立预留


def test_chat_503_on_budget_store_unavailable(monkeypatch):
    import app.server.main as main_mod
    from test_server_api import _FakeComponents, _ScriptedAgent

    class _DownLimiter:
        def allow_rps(self, user_id):
            return True

        def allow_budget(self, user_id):
            raise BudgetStoreUnavailable("redis down")

        def bind_user(self, user_id):
            pass

    comps = _FakeComponents()
    comps.limiter = _DownLimiter()
    monkeypatch.setattr(main_mod, "build_pod_components", lambda: comps)
    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda uid, sid="", components=None, **_k: _ScriptedAgent(uid, sid),
    )
    with TestClient(main_mod.create_app()) as client:
        resp = client.post("/v1/chat", json={
            "user_id": "u1", "session_id": "s1", "message": "hi",
        })
    assert resp.status_code == 503


def test_stream_budget_store_unavailable_error_contract(monkeypatch):
    import app.server.main as main_mod
    from test_server_api import _FakeComponents, _ScriptedAgent

    class _BoomAgent(_ScriptedAgent):
        def chat(self, message):
            raise BudgetStoreUnavailable("redis down")

    comps = _FakeComponents()
    monkeypatch.setattr(main_mod, "build_pod_components", lambda: comps)
    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda uid, sid="", components=None, **_k: _BoomAgent(uid, sid),
    )
    with TestClient(main_mod.create_app()) as client:
        resp = client.post("/v1/chat/stream", json={
            "user_id": "u1", "session_id": "s1", "message": "hi",
        })
    assert resp.status_code == 200  # SSE 已开始，错误在流内
    assert "budget_store_unavailable" in resp.text
    assert '"ok": false' in resp.text


def test_stream_close_budget_failure_reported_before_success(monkeypatch):
    """成功 end 之前先 close；close 预算故障 → error + end ok=false，不报成功。"""
    import app.server.main as main_mod
    from test_server_api import _FakeComponents, _ScriptedAgent

    class _CloseBoom(_ScriptedAgent):
        def close(self):
            raise BudgetStoreUnavailable("redis down")

    comps = _FakeComponents()
    monkeypatch.setattr(main_mod, "build_pod_components", lambda: comps)
    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda uid, sid="", components=None, **_k: _CloseBoom(uid, sid),
    )
    with TestClient(main_mod.create_app()) as client:
        resp = client.post("/v1/chat/stream", json={
            "user_id": "u1", "session_id": "s1", "message": "hi",
        })
    assert "budget_store_unavailable" in resp.text
    assert '"ok": true' not in resp.text


def test_memory_worker_rejects_missing_user_and_scopes_per_job(monkeypatch):
    from app.agent.memory.jobs import MemoryJobWorker

    processed: list[str] = []

    class _Store:
        def __init__(self):
            self.jobs = [
                {"session_key": "u1/s1", "through_seq": 1, "user_id": ""},
                {"session_key": "u2/s2", "through_seq": 1, "user_id": "u2"},
            ]
            self.failed: list = []

        def claim(self, worker_id, limit=5):
            jobs, self.jobs = self.jobs, []
            return jobs

        def is_duplicate(self, job):
            return False

        def complete(self, job):
            processed.append(f"complete:{current_budget_user()}")

        def fail(self, job, exc):
            self.failed.append((job.get("user_id"), type(exc).__name__))

        def backlog(self):
            return 0

    store = _Store()
    before_scope = current_budget_user()
    worker = MemoryJobWorker(
        store, ltm_factory=lambda uid, sid: object(),
        llm_client=object(), model="m", worker_id="w1",
    )
    # _process_job 需要 store 方法；用打桩替换，聚焦计费作用域
    monkeypatch.setattr(
        worker, "_process_job",
        lambda job: processed.append(f"process:{current_budget_user()}") or True,
    )
    worker.process_once()
    assert ("", "ValueError") in store.failed  # 空 user 任务被拒绝
    assert "process:u2" in processed  # 有 user 的任务按自身 user 计费
    assert current_budget_user() == before_scope  # 退出后恢复（未污染全局）


# ------------------------------------------------------------
# 8：迁移 011 文本断言（真实 MySQL 集成测试在 CI 覆盖）
# ------------------------------------------------------------
def test_migration_011_backfills_status_and_uuid_unique_key():
    from pathlib import Path

    sql = (Path(__file__).resolve().parents[2]
           / "deploy" / "sql" / "011_message_outbox_reliability.sql").read_text(encoding="utf-8")
    assert "CASE WHEN synced_at IS NOT NULL THEN 'done' ELSE 'pending' END" in sql
    assert "DROP INDEX uq_outbox_session_seq" in sql
    assert "ADD UNIQUE KEY uq_outbox_session_seq (session_key, session_uuid, seq)" in sql
    assert "obsolete" in sql  # 状态注释包含终态
