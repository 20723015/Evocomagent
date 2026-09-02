"""安全修复（0.3.0 基线）验收：P1 RBAC/退款/MCP/审核后台、P2 租约/穿越/
timeout/水位/LTM merge、P3 钉版与弃用。

对应《docs/修复计划执行基线（含评审修订）》的验收清单。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import fakeredis
import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.agent.memory.long_term import LongTermMemory, MemoryFact
from app.config.settings import settings
from app.security.identifiers import InvalidIdentifier, validate_identifier
from app.security.jwt import create_token, decode_claims
from app.stores.base import SessionState
from app.stores.idle_consolidator import find_idle_sessions
from app.stores.memory_store import LocalFileLTMStore, RedisLTMStore
from app.stores.session_store import LocalFileSessionStore, RedisSessionStore
from conftest import FakeChatClient

JWT_SECRET = "test-secret-0123456789abcdef0123456789"


def _fake_redis():
    return fakeredis.FakeRedis(server=fakeredis.FakeServer())


# ============================================================
# P1 RBAC：运营端点 scope 校验 + issue_token 配套
# ============================================================
def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _ops_client(monkeypatch):
    """auth_enabled=true + 打桩 Agent 的 TestClient。"""
    import app.server.main as main_mod
    from test_server_api import _FakeComponents, _ScriptedAgent

    settings.jwt_secret = JWT_SECRET
    settings.auth_enabled = True

    app = main_mod.create_app()
    monkeypatch.setattr(main_mod, "build_pod_components", lambda: _FakeComponents())
    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda uid, sid="", comps=None, **_k: _ScriptedAgent(uid, sid),
    )
    return TestClient(app)


def test_ops_endpoints_reject_scopeless_jwt(monkeypatch, reset_settings):
    client = _ops_client(monkeypatch)
    no_scope = create_token("agent1", scopes="chat", ttl_minutes=5)
    with client as c:
        for method, url, kwargs in (
            ("get", "/v1/handoffs", {}),
            ("post", "/v1/handoffs", {"json": {"user_id": "u1", "session_id": "s1"}}),
            ("post", "/v1/handoffs/t1/resolve", {"json": {"user_id": "u1"}}),
            ("get", "/v1/messages/search", {"params": {"q": "订单"}}),
        ):
            resp = getattr(c, method)(url, headers=_auth(no_scope), **kwargs)
            assert resp.status_code == 403, url
            assert "scope" in resp.json()["detail"]


def test_ops_endpoints_accept_ops_scope(monkeypatch, reset_settings):
    client = _ops_client(monkeypatch)
    ops = create_token("agent1", scopes="chat ops", ttl_minutes=5)
    with client as c:
        assert c.get("/v1/handoffs", headers=_auth(ops)).status_code == 200
        resp = c.post("/v1/handoffs", json={"user_id": "u1", "session_id": "s1"},
                      headers=_auth(ops))
        assert resp.status_code == 200
        # message_search：无 ES → 200 + degraded（鉴权先于业务）
        resp = c.get("/v1/messages/search", params={"q": "订单"}, headers=_auth(ops))
        assert resp.status_code == 200
        assert resp.json()["degraded"] is True


def test_ops_endpoints_require_token_when_auth_on(monkeypatch, reset_settings):
    client = _ops_client(monkeypatch)
    with client as c:
        assert c.get("/v1/handoffs").status_code == 401
        assert c.get("/v1/messages/search", params={"q": "x"}).status_code == 401


def test_ops_rbac_required_forces_jwt_even_with_auth_off(monkeypatch, reset_settings):
    import app.server.main as main_mod
    from test_server_api import _FakeComponents, _ScriptedAgent

    settings.auth_enabled = False
    settings.ops_rbac_required = True
    settings.jwt_secret = JWT_SECRET
    app = main_mod.create_app()
    monkeypatch.setattr(main_mod, "build_pod_components", lambda: _FakeComponents())
    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda uid, sid="", comps=None, **_k: _ScriptedAgent(uid, sid),
    )
    with TestClient(app) as c:
        # auth 关闭但 ops 强制开启：无 token → 401，有 ops scope → 200
        assert c.get("/v1/handoffs").status_code == 401
        token = create_token("agent1", scopes="ops", ttl_minutes=5)
        assert c.get("/v1/handoffs", headers=_auth(token)).status_code == 200


def test_scopes_roundtrip_and_dev_passthrough(reset_settings):
    settings.jwt_secret = JWT_SECRET
    token = create_token("u1", scopes="chat,ops")  # issue_token 的逗号风格
    claims = decode_claims(token)
    assert "ops" in claims["scope"] and "chat" in claims["scope"]

    # 开发直通：auth off 且未强制 → 匿名 Principal（ops 视为已授予）
    settings.auth_enabled = False
    settings.ops_rbac_required = False
    from fastapi import Request  # noqa: F401 —— 仅语义提示；authorize 只读 headers

    from app.security.principal import SCOPE_OPS, authorize_scopes

    class _Req:
        headers: dict = {}

    principal = authorize_scopes(_Req(), SCOPE_OPS)
    assert principal.via == "anonymous"


def test_issue_token_script_emits_valid_jwt(reset_settings, capsys):
    settings.jwt_secret = JWT_SECRET
    import sys as _sys

    from app.scripts import issue_token

    _sys.argv = ["issue_token", "--user", "u1", "--scopes", "chat,ops", "--ttl", "30"]
    issue_token.main()
    token = capsys.readouterr().out.strip()
    claims = decode_claims(token)
    assert claims["sub"] == "u1"
    assert "ops" in claims["scope"].split()


# ============================================================
# P2 路径穿越：标识符白名单
# ============================================================
@pytest.mark.parametrize("bad", [
    "../etc/passwd", "a/b", "a\\b", "..", "a:b", "a b", "", "x" * 129,
])
def test_identifier_rejects_invalid(bad):
    with pytest.raises(InvalidIdentifier):
        validate_identifier(bad, "user_id")


def test_identifier_allows_empty_session_id_only_for_session():
    assert validate_identifier("", "session_id", allow_empty=True) == ""
    with pytest.raises(InvalidIdentifier):
        validate_identifier("", "user_id")
    assert validate_identifier("u1", "user_id") == "u1"
    assert validate_identifier("A-b._9", "session_id", allow_empty=True) == "A-b._9"


def test_session_store_rejects_traversal(tmp_path):
    store = LocalFileSessionStore(tmp_path)
    with pytest.raises(InvalidIdentifier):
        store.path_for("../../other-user", "s1")
    with pytest.raises(InvalidIdentifier):
        store.path_for("u1", "../../other-session")
    # 未创建任何目录外文件
    assert list(tmp_path.rglob("*.json")) == []


def _anonymous_client(monkeypatch):
    import app.server.main as main_mod
    from test_server_api import _FakeComponents, _ScriptedAgent

    app = main_mod.create_app()
    monkeypatch.setattr(main_mod, "build_pod_components", lambda: _FakeComponents())
    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda uid, sid="", comps=None, **_k: _ScriptedAgent(uid, sid),
    )
    return TestClient(app)


def test_request_ids_validated_422(monkeypatch):
    with _anonymous_client(monkeypatch) as client:
        resp = client.post("/v1/chat", json={
            "user_id": "../etc/passwd", "message": "hi",
        })
        assert resp.status_code == 422
        resp = client.post("/v1/chat", json={
            "user_id": "u1", "session_id": "a/b", "message": "hi",
        })
        assert resp.status_code == 422
        resp = client.get("/v1/chat/stream", params={
            "user_id": "../x", "message": "hi",
        })
        assert resp.status_code == 422


# ============================================================
# P1 ownership fail-closed（MCP 无身份路径不可绕过）
# ============================================================
def test_ownership_fail_closed_without_ctx(reset_settings):
    from app.agent.tools.mock_data import ORDERS
    from app.agent.tools.ownership import check_order_ownership

    settings.enforce_order_ownership = True
    order_id = next(iter(ORDERS))
    denied = check_order_ownership(order_id, None)
    assert denied is not None and denied["success"] is False
    assert "身份" in denied["error"]


def test_refund_fail_closed_without_ctx(reset_settings):
    from app.agent.tools.mock_data import ORDERS
    from app.agent.tools.refund import apply_refund

    settings.enforce_order_ownership = True
    settings.refund_confirmation_required = False
    order_id = next(iter(ORDERS))
    out = apply_refund(order_id, "原因", None)
    assert out["success"] is False
    assert "身份" in out["error"]


# ============================================================
# P1 MCP server：Bearer 中间件 + actor 身份
# ============================================================
def test_mcp_bearer_middleware_blocks_anonymous():
    from mcp_server.server import _BearerAuthMiddleware

    reached: list[bool] = []

    async def _inner(scope, receive, send):
        reached.append(True)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def _receive():
        return {"type": "http.request"}

    app = _BearerAuthMiddleware(_inner, "secret-token")

    def _sender(bucket):
        async def _send(message):
            bucket.append(message)
        return _send

    async def _drive():
        # 匿名 → 401，且请求不达内层
        sent: list[dict] = []
        await app({"type": "http", "path": "/mcp", "headers": []},
                  _receive, _sender(sent))
        assert sent[0]["status"] == 401
        # 正确 token → 放行
        sent2: list[dict] = []
        headers = [(b"authorization", b"Bearer secret-token")]
        await app({"type": "http", "path": "/mcp", "headers": headers},
                  _receive, _sender(sent2))
        assert sent2[0]["status"] == 200
        # 非 /mcp 路径不拦
        sent3: list[dict] = []
        await app({"type": "http", "path": "/other", "headers": []},
                  _receive, _sender(sent3))
        assert sent3[0]["status"] == 200
        assert len(reached) == 2  # 仅两次真正到达内层

    asyncio.run(_drive())


def test_mcp_tools_carry_actor_identity(reset_settings):
    """修复计划：MCP 服务端从 actor token 重建 ToolContext——缺失/过期/
    scope 不足一律拒绝（订单/物流/退款都有比对主体，归属校验不再绕过）。"""
    from types import SimpleNamespace

    import pytest
    from app.mcp_client.actor import (
        SCOPE_ORDERS_READ,
        SCOPE_REFUND_WRITE,
        issue_actor_token,
    )
    from mcp_server import server as mcp_srv

    settings.mcp_actor_secret = "actor-secret-0123456789-abcdefghijklmnop"

    def fake_ctx(token):
        return SimpleNamespace(
            request_context=SimpleNamespace(
                meta=SimpleNamespace(actor=token if token else None)
            )
        )

    token = issue_actor_token("u1", "s1", (SCOPE_ORDERS_READ,))
    user_ctx = mcp_srv._actor_ctx(fake_ctx(token))
    assert user_ctx.user_id == "u1" and user_ctx.session_id == "s1"

    # 缺失 actor token → 拒绝
    with pytest.raises(ValueError):
        mcp_srv._actor_ctx(fake_ctx(None))
    # 退款 scope 不足 → 拒绝（orders:read 不能退款）
    with pytest.raises(ValueError):
        mcp_srv._actor_ctx_refund(fake_ctx(token))
    refund_token = issue_actor_token("u1", "s1", (SCOPE_REFUND_WRITE,))
    assert mcp_srv._actor_ctx_refund(fake_ctx(refund_token)).user_id == "u1"

    # 过期 token → 拒绝
    from datetime import datetime, timedelta, timezone

    past = datetime.now(timezone.utc) - timedelta(seconds=5)
    expired = issue_actor_token("u1", "s1", (SCOPE_ORDERS_READ,),
                                ttl_seconds=1, now_utc=past)
    with pytest.raises(ValueError):
        mcp_srv._actor_ctx(fake_ctx(expired))


# ============================================================
# P2 SSE：流式全程持锁 + 断连有界收尾
# ============================================================
def test_sse_stream_holds_session_lease_during_turn(monkeypatch):
    """核心回归：租约覆盖整个流式生命周期（历史实现在 return 时即释放）。"""
    import app.server.main as main_mod
    from test_server_api import _FakeComponents, _ScriptedAgent

    comps = _FakeComponents()
    observed: dict = {}

    async def fake_turn(agent, message, on_start=None):
        if on_start is not None:
            on_start()
        # turn 进行中：同一会话锁必须被流持有（再次获取应失败）
        observed["held"] = comps.locks.acquire(agent.user_id, agent.session_id) is None
        from conftest import sample_response

        return sample_response(reply="ok")

    monkeypatch.setattr(main_mod.runtime, "run_agent_turn", fake_turn)
    monkeypatch.setattr(main_mod, "build_pod_components", lambda: comps)
    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda uid, sid="", comps=None, **_k: _ScriptedAgent(uid, sid),
    )
    with TestClient(main_mod.create_app()) as client:
        resp = client.get("/v1/chat/stream", params={
            "user_id": "u1", "session_id": "sX", "message": "hi",
        })
        assert resp.status_code == 200
    assert observed["held"] is True
    # 流结束后锁已释放（可重新获取）
    token = comps.locks.acquire("u1", "sX")
    assert token is not None
    comps.locks.release("u1", "sX", token)


def test_sse_conflict_surfaced_inside_stream(monkeypatch):
    """锁被占时，第二条流在流内收到冲突 error 事件（租约在生成器内获取）。"""
    import app.server.main as main_mod
    from test_server_api import _FakeComponents, _ScriptedAgent

    comps = _FakeComponents()
    hold_token = comps.locks.acquire("u1", "sDup")  # 外部占锁
    assert hold_token is not None
    monkeypatch.setattr(main_mod, "build_pod_components", lambda: comps)
    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda uid, sid="", comps=None, **_k: _ScriptedAgent(uid, sid),
    )
    try:
        with TestClient(main_mod.create_app()) as client:
            resp = client.get("/v1/chat/stream", params={
                "user_id": "u1", "session_id": "sDup", "message": "hi",
            })
            assert resp.status_code == 200
            assert "event: error" in resp.text
            assert "正在处理中" in resp.text
    finally:
        comps.locks.release("u1", "sDup", hold_token)


def test_disconnected_finalize_closes_agent_when_work_done():
    from app.security.ratelimit import UsageTracker, UserLimiter
    from app.server.main import _schedule_disconnected_finalize

    closed: list[bool] = []

    class _Agent:
        def close(self):
            closed.append(True)

    class _Comps:
        usage_tracker = UsageTracker()

    limiter = UserLimiter(None, max_rps=0, daily_token_budget=0)

    async def _drive():
        agent = _Agent()

        async def _work():
            return 42

        task = asyncio.create_task(_work())
        await asyncio.sleep(0.01)  # 让工作完成
        _schedule_disconnected_finalize(
            task, agent, _Comps(), "u1", limiter, "req-ok",
        )
        await asyncio.sleep(0.05)  # 等收尾任务执行
        assert closed == [True]

    asyncio.run(_drive())


def test_disconnected_finalize_bounded_wait_skips_close(monkeypatch):
    """评审·坑1：断连等待有上限——超时不再 close（线程仍持有 agent），
    已发生的用量仍按 request 入账（评审二轮 B2）。"""
    import app.server.main as main_mod
    from app.security.ratelimit import UsageTracker, UserLimiter

    closed: list[bool] = []

    class _Agent:
        def close(self):
            closed.append(True)

    class _Comps:
        usage_tracker = UsageTracker()

    monkeypatch.setattr(main_mod, "_disconnect_wait_bound_seconds", lambda: 0.05)

    async def _drive():
        agent = _Agent()
        comps = _Comps()
        comps.usage_tracker.begin_request("req-timeout")  # 模拟已归集用量
        comps.usage_tracker._totals["req-timeout"] = [123, time.time()]

        async def _work():
            await asyncio.sleep(30)  # 模拟不可中断的长工作

        task = asyncio.create_task(_work())
        await asyncio.sleep(0)
        limiter = UserLimiter(None, max_rps=0, daily_token_budget=0)
        main_mod._schedule_disconnected_finalize(
            task, agent, comps, "u1", limiter, "req-timeout",
        )
        await asyncio.sleep(0.3)  # 远小于 30s 工作，但大于 0.05s 上限
        assert closed == []  # 超时分支：不 close
        # 已发生用量照常入账（end_request 幂等配对）
        assert comps.usage_tracker.end_request("req-timeout") == 0  # 已被收尾取走
        task.cancel()  # 清理测试任务

    asyncio.run(_drive())


def test_disconnect_wait_bound_formula(reset_settings):
    """Agent能力强化计划·改造一：断连等待上限由 turn_budget 推导（不再按
    max_react_steps × 单次超时估算——步数扩到 8 后该估算与真实墙钟脱钩）。"""
    from app.server.main import _disconnect_wait_bound_seconds

    settings.turn_budget_seconds = 120
    settings.max_react_steps = 5
    settings.llm_timeout_seconds = 60
    assert _disconnect_wait_bound_seconds() == pytest.approx(120 * 1.5)

    settings.turn_budget_seconds = 3  # 下限 30s（防微预算配置）
    assert _disconnect_wait_bound_seconds() == pytest.approx(30 * 1.5)


# ============================================================
# P2 LLM timeout 接线
# ============================================================
def test_openai_client_wires_explicit_timeout(monkeypatch, reset_settings):
    import app.server.deps as deps

    recorded: dict = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

    monkeypatch.setattr(deps, "OpenAI", _FakeOpenAI)
    settings.llm_timeout_seconds = 42
    deps.build_openai_client()
    assert recorded["timeout"] == 42
    assert recorded["max_retries"] == 0  # 重试由韧性包装负责，不叠加


def test_agent_fallback_client_wires_timeout(reset_settings):
    from app.agent.chat import EcomAgent

    client = EcomAgent._build_default_client()
    assert client.timeout == settings.llm_timeout_seconds
    assert client.max_retries == 0


# ============================================================
# P2 SESSION_STORE_BACKEND 生效
# ============================================================
def _patch_storage_sources(monkeypatch):
    monkeypatch.setattr("app.stores.sql.engine.get_engine", lambda: None)
    monkeypatch.setattr("app.stores.redis_client.get_redis", lambda: _fake_redis())


def test_backend_file_forces_local_file_even_with_redis(monkeypatch, reset_settings):
    import app.server.deps as deps

    _patch_storage_sources(monkeypatch)
    settings.session_store_backend = "file"
    comps = deps.build_pod_components()
    assert type(comps.session_store) is LocalFileSessionStore
    assert type(comps.ltm_store) is LocalFileLTMStore


def test_backend_redis_uses_redis(monkeypatch, reset_settings):
    import app.server.deps as deps

    _patch_storage_sources(monkeypatch)
    settings.session_store_backend = "redis"
    comps = deps.build_pod_components()
    assert type(comps.session_store) is RedisSessionStore
    assert type(comps.ltm_store) is RedisLTMStore


def test_backend_auto_prefers_redis(monkeypatch, reset_settings):
    import app.server.deps as deps

    _patch_storage_sources(monkeypatch)
    settings.session_store_backend = "auto"
    comps = deps.build_pod_components()
    assert type(comps.session_store) is RedisSessionStore


# ============================================================
# P2 巩固水位持久化 + LTM 原子 merge
# ============================================================
def test_session_watermark_roundtrip(tmp_path):
    store = LocalFileSessionStore(tmp_path)
    store.save("u", "s", SessionState(
        session_id="sid", user_id="u",
        messages=[{"role": "user", "content": "m"}] * 5,
        version=0, consolidated_len=3,
    ))
    loaded = store.load("u", "s")
    assert loaded.consolidated_len == 3


def test_session_watermark_legacy_payload_defaults_consolidated(tmp_path):
    """旧格式（无 consolidated_len 字段）→ 视为已全量巩固，升级不重复摘要。"""
    store = LocalFileSessionStore(tmp_path)
    legacy = {
        "version": 2, "session_id": "sid", "user_id": "u",
        "summary": None,
        "messages": [{"role": "user", "content": "m"}] * 4,
        "lock_version": 0, "updated_at": "2026-01-01T00:00:00",
    }
    path = tmp_path / "u" / "s.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(legacy), encoding="utf-8")
    loaded = store.load("u", "s")
    assert loaded.consolidated_len == 4


def test_agent_restores_consolidation_watermark(tmp_path):
    from app.agent.chat import EcomAgent

    store = LocalFileSessionStore(tmp_path)
    store.save("wm", "s", SessionState(
        session_id="uuid-wm", user_id="wm",
        messages=[{"role": "user", "content": "hi"}] * 4,
        version=0, consolidated_len=4,
    ))
    agent = EcomAgent(
        user_id="wm", session_id="s", client=FakeChatClient(),
        memory_enabled=False, session_store=store, ltm_store=None,
    )
    assert agent._consolidated_len == 4
    # close() 只巩固水位之后的尾部（这里为空 → 零重复巩固）
    agent.memory_manager.consolidate_to_long_term(
        agent.raw_messages[agent._consolidated_len:], agent.summary,
    )


def test_find_idle_sessions_skips_fully_consolidated():
    store = RedisSessionStore(_fake_redis())
    old = "2020-01-01T00:00:00"
    msg = {"role": "user", "content": "m"}
    store.save("u1", "s1", SessionState(
        session_id="a", user_id="u1", messages=[msg], version=0,
        consolidated_len=1, updated_at=old,
    ))
    store.save("u1", "s2", SessionState(
        session_id="b", user_id="u1", messages=[msg], version=0,
        consolidated_len=0, updated_at=old,
    ))
    found = find_idle_sessions(store, idle_minutes=5)
    assert ("u1", "s2") in found
    assert ("u1", "s1") not in found


def test_ltm_file_merge_concurrent_no_lost_facts(tmp_path):
    """竞态验收走文件锁路径（评审·坑4：fakeredis 逼不出 WATCH 竞态）。"""
    store = LocalFileLTMStore(tmp_path)

    def _apply_for(tag, worker_name):
        def _apply(current):
            payload = current or {"facts": [], "interaction_summaries": []}
            facts = list(payload.get("facts", []))
            facts.append({
                "content": f"fact-{tag}-{worker_name}",
                "category": "preference", "created_at": "", "source_session": "",
            })
            return {**payload, "facts": facts}
        return _apply

    def _worker(tag):
        for i in range(10):
            store.merge("u1", _apply_for(tag, f"{tag}{i}"))

    threads = [threading.Thread(target=_worker, args=(t,)) for t in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = store.load("u1")
    # 并发 merge 零丢失（历史 last-write-wins 会丢到 ~10 条）
    assert len(final["facts"]) == 20


def test_ltm_redis_merge_correctness():
    """fakeredis 只验证合并逻辑正确性（非竞态——评审·坑4）。"""
    store = RedisLTMStore(_fake_redis())
    store.save("u1", {
        "facts": [{"content": "a", "category": "x", "created_at": "", "source_session": ""}],
        "interaction_summaries": [],
    })

    def _apply(current):
        facts = list((current or {}).get("facts", []))
        facts.append({"content": "b", "category": "x", "created_at": "", "source_session": ""})
        return {**(current or {}), "facts": facts, "interaction_summaries": []}

    merged = store.merge("u1", _apply)
    assert [f["content"] for f in merged["facts"]] == ["a", "b"]
    reloaded = store.load("u1")
    assert [f["content"] for f in reloaded["facts"]] == ["a", "b"]


def test_ltm_sql_merge_sequential_no_lost_facts():
    from app.stores.sql.memory_store import SqlLTMStore
    from app.stores.sql.schema import metadata

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    metadata.create_all(engine)
    store = SqlLTMStore(engine)

    def _apply(tag):
        def _m(current):
            facts = list((current or {}).get("facts", []))
            facts.append({"content": f"fact-{tag}", "category": "x",
                          "created_at": "", "source_session": ""})
            return {**(current or {}), "facts": facts}
        return _m

    store.merge("u1", _apply("a"))
    store.merge("u1", _apply("b"))
    assert [f["content"] for f in store.load("u1")["facts"]] == ["fact-a", "fact-b"]


def test_long_term_memory_extract_uses_merge(monkeypatch):
    """LongTermMemory 写路径走 store.merge（并发不覆盖），并回写内存态。"""
    merges: list = []

    class _MergeStore:
        def load(self, user_id):
            return {"facts": [
                {"content": "已有事实", "category": "identity",
                 "created_at": "", "source_session": ""},
            ], "interaction_summaries": []}

        def merge(self, user_id, merger):
            merges.append(user_id)
            return merger(self.load(user_id))

        def save(self, user_id, payload):  # 不应被走到
            raise AssertionError("merge 能力存在时不应整包 save")

    ltm = LongTermMemory(user_id="u1", store=_MergeStore(), max_facts=50)

    def _fake_extract(client, model, messages, summary, known_facts):
        return (
            [
                MemoryFact(content="新事实", category="preference",
                           created_at="now", source_session="s1"),
                MemoryFact(content="已有事实", category="identity",
                           created_at="old", source_session=""),
            ],
            "本轮摘要",
        )

    monkeypatch.setattr(
        "app.agent.memory.long_term.extract_long_term_facts", _fake_extract,
    )
    ltm.extract_and_save(None, "m", [{"role": "user", "content": "hi"}], None)
    assert merges == ["u1"]
    # 内存态与合并结果一致：已有事实保留 + 新事实追加 + 摘要落账
    contents = {f.content for f in ltm.facts}
    assert contents == {"已有事实", "新事实"}
    assert ltm.interaction_summaries[-1]["summary"] == "本轮摘要"


# ============================================================
# 评审二轮 A：consolidated_len 升级迁移（老库不炸）
# ============================================================
def _create_legacy_db(db_file):
    """按阶段八老 schema（无 consolidated_len 列）建库并写入一行会话。"""
    import sqlite3

    conn = sqlite3.connect(db_file)
    conn.execute(
        """
        CREATE TABLE sessions (
            session_key  VARCHAR(200) PRIMARY KEY,
            user_id      VARCHAR(64)  NOT NULL,
            session_uuid CHAR(32)     NOT NULL DEFAULT '',
            version      INT          NOT NULL DEFAULT 0,
            summary      TEXT         NULL,
            stm_json     TEXT         NULL,
            status       VARCHAR(12)  NOT NULL DEFAULT 'active',
            updated_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
            expires_at   DATETIME     NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO sessions (session_key, user_id, session_uuid, version) "
        "VALUES ('legacy-u/legacy-s', 'legacy-u', 'abc', 0)"
    )
    conn.commit()
    conn.close()


def test_startup_migration_adds_missing_column(tmp_path, monkeypatch, reset_settings):
    """老库升级：get_engine 探测到缺列自动 ALTER——升级后 SELECT/写入正常。"""
    from sqlalchemy import inspect

    from app.stores.sql import engine as engine_mod
    from app.stores.sql.session_store import SqlSessionStore

    db_file = tmp_path / "legacy.sqlite"
    _create_legacy_db(str(db_file))

    settings.db_url = f"sqlite:///{db_file}"
    engine_mod.reset_engine()
    try:
        engine = engine_mod.get_engine()
        columns = {c["name"] for c in inspect(engine).get_columns("sessions")}
        assert "consolidated_len" in columns
        # 老数据可读、可按新 schema 写回（水位回填 0）
        store = SqlSessionStore(engine)
        state = store.load("legacy-u", "legacy-s")
        assert state is not None and state.user_id == "legacy-u"
        assert state.consolidated_len == 0
    finally:
        engine_mod.reset_engine()


def test_startup_migration_skips_when_table_absent(tmp_path, monkeypatch, reset_settings):
    """空库（表未建）：探测跳过，由 create_all 带列建表，不报错。"""
    from sqlalchemy import inspect

    from app.stores.sql import engine as engine_mod
    from app.stores.sql.schema import metadata

    db_file = tmp_path / "fresh.sqlite"
    settings.db_url = f"sqlite:///{db_file}"
    engine_mod.reset_engine()
    try:
        engine = engine_mod.get_engine()
        metadata.create_all(engine)
        columns = {c["name"] for c in inspect(engine).get_columns("sessions")}
        assert "consolidated_len" in columns
    finally:
        engine_mod.reset_engine()


# ============================================================
# 评审二轮 B1：reset 抢会话锁
# ============================================================
def test_reset_conflicts_while_session_locked(monkeypatch):
    """会话被流式/对话持有时，reset 返回 409（写操作不再裸奔）。"""
    import app.server.main as main_mod
    from test_server_api import _FakeComponents, _ScriptedAgent

    comps = _FakeComponents()
    hold_token = comps.locks.acquire("u1", "sLock")
    assert hold_token is not None
    monkeypatch.setattr(main_mod, "build_pod_components", lambda: comps)
    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda uid, sid="", comps=None, **_k: _ScriptedAgent(uid, sid),
    )
    try:
        with TestClient(main_mod.create_app()) as client:
            resp = client.post(
                "/v1/sessions/reset",
                json={"user_id": "u1", "session_id": "sLock"},
            )
            assert resp.status_code == 409
            assert "正在处理中" in resp.json()["detail"]
    finally:
        comps.locks.release("u1", "sLock", hold_token)


def test_reset_succeeds_and_releases_lock(monkeypatch):
    import app.server.main as main_mod
    from test_server_api import _FakeComponents, _ScriptedAgent

    comps = _FakeComponents()
    monkeypatch.setattr(main_mod, "build_pod_components", lambda: comps)
    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda uid, sid="", comps=None, **_k: _ScriptedAgent(uid, sid),
    )
    with TestClient(main_mod.create_app()) as client:
        resp = client.post(
            "/v1/sessions/reset",
            json={"user_id": "u1", "session_id": "sOk"},
        )
        assert resp.status_code == 200
        # reset 完成后锁已释放
        token = comps.locks.acquire("u1", "sOk")
        assert token is not None
        comps.locks.release("u1", "sOk", token)


# ============================================================
# 评审二轮 B3：LLM 重试单调时钟总截止
# ============================================================
def test_llm_retry_deadline_bounds_wall_clock():
    """总预算耗尽 → 提前止损：调用次数与墙钟都有上界（无截止时会全量重试）。"""
    import httpx

    from app.llm.client import install_resilience
    from conftest import FakeChatClient

    client = FakeChatClient()
    err = httpx.Request("POST", "http://x")

    def _always_retryable(kind, kwargs):
        raise APIConnectionError(request=err)

    for _ in range(8):
        client.enqueue_callable(_always_retryable)

    install_resilience(client, "m1", max_retries=50, max_concurrent=4,
                       fallback_model="", timeout_seconds=0.02)
    t0 = time.monotonic()
    with pytest.raises(APIConnectionError):
        client.chat.completions.create(
            model="m1", messages=[{"role": "user", "content": "hi"}],
        )
    elapsed = time.monotonic() - t0
    # 预算 = (50+1)×0.02 ≈ 1.0s，第一次退避睡眠即耗尽预算 → 只打了 1~2 次
    assert len(client.calls) <= 3
    assert elapsed < 4.5


def test_llm_retry_deadline_skips_fallback_when_budget_spent():
    """预算耗尽时连降级链也不启动（不可能完成的调用不发起）。"""
    import httpx

    from app.llm.client import install_resilience
    from conftest import FakeChatClient

    client = FakeChatClient()

    def _always_retryable(kind, kwargs):
        raise APIConnectionError(request=httpx.Request("POST", "http://x"))

    client.enqueue_callable(_always_retryable)
    install_resilience(client, "m1", max_retries=3, max_concurrent=4,
                       fallback_model="m2", timeout_seconds=0.02)
    with pytest.raises(APIConnectionError):
        client.chat.completions.create(model="m1", messages=[])
    # 只有主模型那次尝试；降级模型从未被调用
    assert len(client.calls) == 1
    assert client.calls[0][1]["model"] == "m1"


# ============================================================
# 评审二轮 B5：500 响应携带 trace_id
# ============================================================
def test_500_response_carries_trace_id(monkeypatch):
    """兜底 500：固定文案 + trace_id（服务端日志同 id 可对账），不回显异常。"""
    import app.server.main as main_mod
    from test_server_api import _FakeComponents, _ScriptedAgent

    class _FailingAgent(_ScriptedAgent):
        def chat(self, message: str):
            raise RuntimeError("boom-internal-detail")

    app = main_mod.create_app()
    monkeypatch.setattr(main_mod, "build_pod_components", lambda: _FakeComponents())
    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda uid, sid="", comps=None, **_k: _FailingAgent(uid, sid),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post("/v1/chat", json={"user_id": "u1", "message": "hi"})
    assert resp.status_code == 500
    body = resp.json()
    assert body["detail"] == "内部错误，请稍后重试"
    assert "boom-internal-detail" not in resp.text
    assert body["trace_id"] and len(body["trace_id"]) == 16
