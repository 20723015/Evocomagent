"""修复计划·一：会话锁运行模式与租约语义测试。

覆盖：
- 生产 redis_required=true：Redis 故障 fail-closed（SessionLockBackendUnavailable），
  不降级进程内锁；
- 开发模式：Redis 故障仍降级进程内锁（兼容既有行为）；
- 租约 assert_owned / lost 标记 / 幂等 release / handover 移交；
- 写工具在租约失效时被拒绝（registry execute_tool fail-closed）。
"""

from __future__ import annotations

import time

import pytest

from app.stores.locks import (
    SessionLease,
    SessionLockBackendUnavailable,
    SessionLockLost,
    SessionLockManager,
)


class _StubRedis:
    """最小 Redis 桩：set/get/delete/eval（eval 语义与真 Redis 一致）。"""

    def __init__(self):
        self.kv: dict = {}
        self.fail = False

    def set(self, key, value, nx=False, ex=None):
        if self.fail:
            raise ConnectionError("redis down")
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        return True

    def get(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        return self.kv.get(key)

    def delete(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        return self.kv.pop(key, None)

    def eval(self, script, numkeys, key, *args):
        if self.fail:
            raise ConnectionError("redis down")
        if "expire" in script:
            return 1 if self.kv.get(key) == args[0] else 0
        if self.kv.get(key) == args[0]:
            del self.kv[key]
            return 1
        return 0


# ------------------------------------------------------------
# 生产模式：fail-closed
# ------------------------------------------------------------
def test_redis_required_raises_when_redis_fails():
    r = _StubRedis()
    r.fail = True
    mgr = SessionLockManager(r, ttl_seconds=60, redis_required=True)
    with pytest.raises(SessionLockBackendUnavailable):
        mgr.acquire("u1", "s1")


def test_redis_required_raises_when_redis_absent():
    mgr = SessionLockManager(None, ttl_seconds=60, redis_required=True)
    with pytest.raises(SessionLockBackendUnavailable):
        mgr.acquire("u1", "s1")


def test_dev_mode_still_degrades_to_in_process():
    r = _StubRedis()
    r.fail = True
    mgr = SessionLockManager(r, ttl_seconds=60)  # redis_required 默认 False
    token = mgr.acquire("u2", "s2")
    assert token is not None and ":" in token
    assert mgr.acquire("u2", "s2") is None  # 进程内互斥仍生效
    mgr.release("u2", "s2", token)


# ------------------------------------------------------------
# 租约所有权校验 / lost
# ------------------------------------------------------------
def test_assert_owned_ok_then_lost_after_key_removed():
    r = _StubRedis()
    mgr = SessionLockManager(r, ttl_seconds=60)
    token = mgr.acquire("u1", "s1")
    mgr.assert_owned("u1", "s1", token)  # 持有中：通过
    r.kv.clear()  # 模拟 TTL 过期/被抢占
    with pytest.raises(SessionLockLost):
        mgr.assert_owned("u1", "s1", token)


def test_assert_owned_redis_error_is_lost_fail_closed():
    r = _StubRedis()
    mgr = SessionLockManager(r, ttl_seconds=60)
    token = mgr.acquire("u1", "s1")
    r.fail = True
    with pytest.raises(SessionLockLost):
        mgr.assert_owned("u1", "s1", token)


def test_lease_assert_owned_and_unrenewed_becomes_lost():
    r = _StubRedis()
    mgr = SessionLockManager(r, ttl_seconds=1)  # 续期间隔 max(1, ttl/3)=1s
    lease = SessionLease(mgr, "u1", "s1")
    with lease:
        assert lease.token is not None
        lease.assert_owned()  # 正常
        r.kv.clear()  # 模拟锁被抢占/过期
        # 等续期线程观察到 eval 返回 0 → 标记 lost
        deadline = time.time() + 3
        while not lease.lost and time.time() < deadline:
            time.sleep(0.05)
        assert lease.lost is True
        with pytest.raises(SessionLockLost):
            lease.assert_owned()


def test_lease_release_is_idempotent():
    r = _StubRedis()
    mgr = SessionLockManager(r, ttl_seconds=60)
    lease = SessionLease(mgr, "u1", "s1")
    lease.__enter__()
    lease.release()
    lease.release()  # 幂等：不抛
    assert mgr.acquire("u1", "s1") is not None


def test_lease_handover_skips_exit_release():
    r = _StubRedis()
    mgr = SessionLockManager(r, ttl_seconds=60)
    with SessionLease(mgr, "u1", "s1") as lease:
        assert lease.token is not None
        lease.handover()
    # handover 后 __exit__ 不释放：键仍在
    assert r.kv.get("session_lock:u1:s1") == lease.token
    lease.release()  # 后台收尾任务负责释放
    assert r.kv.get("session_lock:u1:s1") is None


# ------------------------------------------------------------
# 写工具租约门禁（修复计划·二轮 1：统一在 ToolManager.execute_tool）
# ------------------------------------------------------------
class _FakeMCP:
    """MCP 客户端替身：记录调用并返回普通字符串结果。"""

    def __init__(self):
        self.calls: list = []

    def call_tool(self, name, arguments, **kwargs):
        self.calls.append((name, arguments, kwargs))
        return '{"success": true, "status": "committed"}'

    def close(self):
        pass


def _write_manager(kind: str):
    """构造写工具调用路径：local 或 mcp（模拟 MCP 分流）。"""
    from app.agent.tools.manager import ToolManager

    mgr = ToolManager(use_mcp=False)
    if kind == "mcp":
        fake = _FakeMCP()
        mgr._mcp_client = fake
        mgr._owns_mcp_client = False
        mgr._tool_source["submit_refund_application"] = "mcp"
        return mgr, fake
    return mgr, None


@pytest.mark.parametrize("kind", ["local", "mcp"])
def test_write_tool_lease_lost_blocked_both_paths(kind):
    from app.agent.context import ToolContext

    def _lost():
        raise SessionLockLost("租约已失效")

    mgr, fake = _write_manager(kind)
    ctx = ToolContext(user_id="u1", session_id="s1", lease_guard=_lost)
    out = mgr.execute_tool("submit_refund_application", {"order_id": "o1", "reason": "x"}, ctx=ctx)
    assert "SESSION_LOCK_LOST" in out
    if fake is not None:
        assert fake.calls == []  # MCP 不得被调用


@pytest.mark.parametrize("kind", ["local", "mcp"])
def test_write_tool_backend_unavailable_blocked_both_paths(kind):
    from app.agent.context import ToolContext

    def _down():
        raise SessionLockBackendUnavailable("redis down")

    mgr, fake = _write_manager(kind)
    ctx = ToolContext(user_id="u1", session_id="s1", lease_guard=_down)
    out = mgr.execute_tool("submit_refund_application", {"order_id": "o1", "reason": "x"}, ctx=ctx)
    assert "SESSION_LOCK_UNAVAILABLE" in out
    if fake is not None:
        assert fake.calls == []


@pytest.mark.parametrize("kind", ["local", "mcp"])
def test_write_tool_missing_guard_requires_lock(kind):
    from app.agent.context import ToolContext

    mgr, fake = _write_manager(kind)
    ctx = ToolContext(user_id="u1", session_id="s1")  # 未绑定 lease_guard
    out = mgr.execute_tool("submit_refund_application", {"order_id": "o1", "reason": "x"}, ctx=ctx)
    assert "SESSION_LOCK_REQUIRED" in out
    # 无 ctx 同样拒绝
    assert "SESSION_LOCK_REQUIRED" in mgr.execute_tool(
        "submit_refund_application", {"order_id": "o1", "reason": "x"}, ctx=None
    )
    if fake is not None:
        assert fake.calls == []


@pytest.mark.parametrize("kind", ["local", "mcp"])
def test_write_tool_unexpected_guard_error_propagates(kind):
    from app.agent.context import ToolContext

    def _boom():
        raise ValueError("programming bug")

    mgr, _ = _write_manager(kind)
    ctx = ToolContext(user_id="u1", session_id="s1", lease_guard=_boom)
    with pytest.raises(ValueError):
        mgr.execute_tool("submit_refund_application", {"order_id": "o1", "reason": "x"}, ctx=ctx)


@pytest.mark.parametrize("kind", ["local", "mcp"])
def test_write_tool_lease_ok_reaches_tool(kind, reset_settings):
    from app.agent.context import ToolContext
    from app.config.settings import settings

    settings.mcp_actor_secret = "x" * 40  # 供 MCP actor token 签发
    calls: list[int] = []

    def _ok():
        calls.append(1)

    mgr, fake = _write_manager(kind)
    ctx = ToolContext(user_id="u1", session_id="s1", lease_guard=_ok)
    out = mgr.execute_tool("submit_refund_application", {"order_id": "o1", "reason": "x"}, ctx=ctx)
    assert calls == [1]  # 门禁被调用
    assert "SESSION_LOCK_" not in out  # 未因租约被拦
    if fake is not None:
        assert len(fake.calls) == 1
