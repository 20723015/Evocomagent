"""阶段一 1.1/1.2 验收：API 骨架、请求级上下文、多用户隔离。

- /healthz /readyz /v1/chat /v1/sessions/reset
- build_agent 每请求注入 user_id/session_id（脚本化替身，全程无网络）
- 两用户交错 50 轮并发：user/session 配对零串扰
- 会话文件按用户隔离（真实 EcomAgent + FakeChatClient，离线）
"""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from app.config.settings import settings
from app.handoff.board import InProcessHandoffBoard
from app.security.ratelimit import UsageTracker, UserLimiter
from app.stores.locks import SessionLockManager
from conftest import FakeChatClient, sample_response


class _FakeComponents:
    client = None
    skill_manager = None
    mcp_client = None
    redis = None  # 无 Redis → 进程内会话锁 + 文件 store
    ltm_store = None
    turns_archive = None
    session_store = None
    locks = SessionLockManager(None, ttl_seconds=60)  # 进程内互斥，足够 API 层测试
    limiter = UserLimiter(None, max_rps=0, daily_token_budget=0)  # 本组测试不限额
    usage_tracker = UsageTracker()
    handoff_board = InProcessHandoffBoard()
    db_engine = None  # 阶段八：本组测试不启用 SQL 正本
    es_client = None
    message_index = ""
    tool_executor = None  # 修复计划：lifespan 关闭前判空跳过


_RECORD_LOCK = threading.Lock()
_RECORDED_CALLS: list[tuple[str, str, str]] = []


class _ScriptedAgent:
    """记录 (user_id, session_id, message) 的 Agent 替身。"""

    def __init__(self, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id or f"sess-{user_id}"
        self.closed = False

    def chat(self, message: str):
        with _RECORD_LOCK:
            _RECORDED_CALLS.append((self.user_id, self.session_id, message))
        resp = sample_response(reply=f"{self.user_id}:{message}")
        return resp

    def reset(self):
        return None

    def close(self):
        self.closed = True


def _patch_factory(monkeypatch, factory):
    monkeypatch.setattr("app.server.main.build_agent", factory)
    monkeypatch.setattr("app.server.main.build_pod_components", lambda: _FakeComponents())


def _make_client(monkeypatch, raise_server_exceptions=True):
    from app.server.main import create_app

    app = create_app()

    def factory(user_id, session_id="", components=None, **_kwargs):
        return _ScriptedAgent(user_id, session_id)

    _patch_factory(monkeypatch, factory)
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def test_healthz_and_readyz(monkeypatch):
    with _make_client(monkeypatch) as client:
        assert client.get("/healthz").json() == {"status": "ok", "components": {}}
        # 4.1：readyz 返回 status + components 明细（未配置组件为 not_configured）
        body = client.get("/readyz").json()
        assert body["status"] == "ready"
        assert isinstance(body.get("components"), dict)
        for value in body["components"].values():
            assert value in ("ok", "not_configured", "degraded") or value.startswith("error")


def test_chat_returns_structured_response(monkeypatch):
    with _make_client(monkeypatch) as client:
        resp = client.post("/v1/chat", json={
            "user_id": "u1", "session_id": "s1", "message": "我的订单到哪了",
        })
        assert resp.status_code == 200
        data = resp.json()
        # session_id 回显请求携带的会话标识（无则服务端生成）
        assert data["session_id"] == "s1"
        assert data["reply"] == "u1:我的订单到哪了"
        assert data["intent"] == "order_query"
        assert data["requires_human"] is False


def test_build_agent_receives_request_scope(monkeypatch):
    with _make_client(monkeypatch) as client:
        client.post("/v1/chat", json={"user_id": "u1", "session_id": "s42", "message": "hi"})
        got = _RECORDED_CALLS[-1]
        assert got[0] == "u1"
        assert got[1] == "s42"
        assert got[2] == "hi"


def test_sessions_reset(monkeypatch):
    with _make_client(monkeypatch) as client:
        resp = client.post("/v1/sessions/reset", json={"user_id": "u2", "session_id": "s2"})
        assert resp.status_code == 200
        assert resp.json() == {"session_id": "s2", "reset": True}


def test_validation_errors(monkeypatch):
    with _make_client(monkeypatch) as client:
        # 空消息 / 缺 user_id → 422
        assert client.post("/v1/chat", json={"user_id": "u1", "message": ""}).status_code == 422
        assert client.post("/v1/chat", json={"message": "hi"}).status_code == 422
        # 超长 user_id / message → 422
        assert client.post("/v1/chat", json={
            "user_id": "x" * 65, "message": "hi"}).status_code == 422
        assert client.post("/v1/chat", json={
            "user_id": "u1", "message": "x" * 8001}).status_code == 422


def test_agent_failure_returns_500(monkeypatch):
    class _FailingAgent(_ScriptedAgent):
        def chat(self, message: str):
            raise RuntimeError("boom")

    app = _make_client(monkeypatch, raise_server_exceptions=False)

    def factory(user_id, session_id="", components=None, **_kwargs):
        return _FailingAgent(user_id, session_id)

    _patch_factory(monkeypatch, factory)
    with app as client:
        resp = client.post("/v1/chat", json={"user_id": "u1", "message": "hi"})
        assert resp.status_code == 500
        # 安全修复 P2：500 响应体脱敏——异常细节只进日志，不回显给客户端
        assert resp.json()["detail"] == "内部错误，请稍后重试"
        assert "boom" not in resp.text
        # 评审二轮 B5：携带 trace_id（与服务端日志对账）
        assert resp.json()["trace_id"] and len(resp.json()["trace_id"]) == 16


def test_interleaved_users_50_turns_no_crosstalk(monkeypatch):
    """验收：两个 user_id 交错并发 50 轮，user/session 与消息严格配对。"""
    with _make_client(monkeypatch) as client:
        errors: list[str] = []

        def worker(user_id: str, session_id: str, seq: int):
            try:
                for i in range(seq):
                    msg = f"{user_id}-msg-{i}"
                    resp = client.post("/v1/chat", json={
                        "user_id": user_id, "session_id": session_id, "message": msg,
                    })
                    assert resp.status_code == 200, resp.text
                    body = resp.json()
                    assert body["reply"] == f"{user_id}:{msg}", body
            except Exception as e:  # noqa: BLE001 —— 收集断言错误到主线程
                errors.append(f"{user_id}: {e!r}")

        threads = [
            threading.Thread(target=worker, args=("u1", "s-a", 25)),
            threading.Thread(target=worker, args=("u2", "s-b", 25)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


# ============================================================
# 阶段三：鉴权 / 限流 / 归属（API 级）
# ============================================================
def test_auth_required_without_token_401(monkeypatch, reset_settings):
    settings.auth_enabled = True
    settings.jwt_secret = "test-secret-0123456789abcdef0123456789"  # ≥32 字节，启动校验通过
    with _make_client(monkeypatch) as client:
        resp = client.post("/v1/chat", json={"user_id": "u1", "message": "hi"})
        assert resp.status_code == 401


def test_auth_resolves_user_from_token(monkeypatch, reset_settings):
    from app.security.jwt import create_token

    settings.auth_enabled = True
    settings.jwt_secret = "test-secret-0123456789abcdef0123456789"
    token = create_token("u-token-user")
    with _make_client(monkeypatch) as client:
        resp = client.post(
            "/v1/chat",
            json={"user_id": "u1", "message": "hi"},  # 请求体 user_id 被忽略
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert _RECORDED_CALLS[-1][0] == "u-token-user"  # user 从 token 解出


def test_rate_limit_429(monkeypatch, reset_settings):
    from app.security.ratelimit import UserLimiter

    settings.auth_enabled = False
    client = _make_client(monkeypatch)
    with client:
        # 替换成严格限额的组件：RPS=1
        client.app.state.components.limiter = UserLimiter(
            None, max_rps=1, daily_token_budget=0,
        )
        assert client.post("/v1/chat", json={"user_id": "u1", "message": "a"}).status_code == 200
        resp = client.post("/v1/chat", json={"user_id": "u1", "message": "b"})
        assert resp.status_code == 429
        # 其他用户不受影响
        assert client.post("/v1/chat", json={"user_id": "u2", "message": "c"}).status_code == 200


def test_session_ownership_403(monkeypatch, reset_settings, tmp_path):
    from app.server.deps import PodComponents
    from app.stores.base import SessionState
    from app.stores.locks import SessionLockManager
    from app.stores.session_store import LocalFileSessionStore

    settings.session_dir = str(tmp_path)
    store = LocalFileSessionStore(tmp_path)
    # 同一存储键（u1 的 s1）下文档归属是 other：u1 来接管 → 403
    store.save("u1", "s1", SessionState(
        session_id="s1", user_id="other",
        messages=[{"role": "user", "content": "别人的会话"}], version=0,
    ))

    # 用真实 Agent 路径（不 patch build_agent）验证 403 映射
    from app.server.deps import PodComponents
    from app.server.main import create_app
    import app.server.main as main_mod
    from app.security.ratelimit import UserLimiter

    comps = PodComponents(
        client=None, skill_manager=None, mcp_client=None, redis=None,
        session_store=store,
        ltm_store=None,
        locks=SessionLockManager(None, ttl_seconds=60),
        turns_archive=None,
        limiter=UserLimiter(None, max_rps=0, daily_token_budget=0),
        usage_tracker=_FakeComponents.usage_tracker,
        refund_store=None,
    )
    monkeypatch.setattr(main_mod, "build_pod_components", lambda: comps)
    with TestClient(create_app()) as client:
        resp = client.post("/v1/chat", json={"user_id": "u1", "session_id": "s1", "message": "hi"})
        assert resp.status_code == 403
        assert "属于用户" in resp.json()["detail"]


# ============================================================
# 隔离性（文件级）
# ============================================================
def test_session_files_isolated_per_user(tmp_path, reset_settings, monkeypatch):
    """真实 EcomAgent：会话文件按 {session_dir}/{user_id}/session.json 隔离。"""
    settings.session_dir = str(tmp_path / "sessions")
    settings.evolve_capture_enabled = False

    monkeypatch.setattr("app.config.settings.settings.session_dir", str(tmp_path / "sessions"))
    from app.agent.chat import EcomAgent

    def make(user_id):
        client = FakeChatClient()
        client.enqueue("你好，请问有什么可以帮您？").enqueue(sample_response(
            reply="您好，很高兴为您服务！"))
        return EcomAgent(user_id=user_id, client=client,
                         memory_enabled=False, use_mcp=False), client

    agent_a, _ = make("u1")
    agent_b, _ = make("u2")
    agent_a.chat("你好")
    agent_b.chat("你好")

    p_a = tmp_path / "sessions" / "u1" / "session.json"
    p_b = tmp_path / "sessions" / "u2" / "session.json"
    assert p_a.exists() and p_b.exists()
    assert p_a != p_b
