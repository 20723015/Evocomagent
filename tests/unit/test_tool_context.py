"""阶段一 1.3/1.5：ToolContext 透传 + 按用户隔离。

覆盖：ctx 注入的 memory/skill 句柄、list_user_orders 按 user_id 过滤、
registry/ToolManager 的 ctx 全链路透传、无 ctx 的旧脚本兼容。
"""

from __future__ import annotations

import json

import pytest

from app.agent.context import ToolContext
from app.agent.tools import registry
from app.agent.tools.manager import ToolManager
from app.agent.tools.memory_tool import recall_user_memory, set_memory_manager
from app.agent.tools.skill_tool import load_skill, set_skill_manager
from app.agent.tools.user_orders import list_user_orders


# ------------------------------------------------------------
# ToolContext 构造
# ------------------------------------------------------------
def test_context_carries_user_and_session():
    ctx = ToolContext(user_id="u1", session_id="s1")
    assert ctx.user_id == "u1"
    assert ctx.session_id == "s1"
    assert ctx.credentials is None


# ------------------------------------------------------------
# list_user_orders：按 user_id 过滤（1.5）
# ------------------------------------------------------------
def test_list_user_orders_filters_by_user_id():
    u1 = list_user_orders(ctx=ToolContext(user_id="u1"))
    assert u1["success"] is True
    assert u1["count"] == 1
    assert u1["orders"][0]["order_id"] == "ORD-20240115-001"

    u2 = list_user_orders(ctx=ToolContext(user_id="u2"))
    assert u2["count"] == 1
    assert u2["orders"][0]["order_id"] == "ORD-20240120-002"

    nobody = list_user_orders(ctx=ToolContext(user_id="ghost"))
    assert nobody["count"] == 0


def test_list_user_orders_legacy_without_ctx_returns_all():
    # 无 ctx（旧脚本直调）保持不做归属过滤，教学场景不回退；
    # P2-1 扩容后返回有界（最近 N 条），总量见 total。
    all_orders = list_user_orders()
    assert all_orders["total"] > 1
    assert all_orders["count"] <= 20
    if all_orders["total"] > all_orders["count"]:
        assert all_orders["truncated"] is True


# ------------------------------------------------------------
# memory / skill 句柄：ctx 优先，全局回落仅作旧兼容
# ------------------------------------------------------------
class _StubLTM:
    facts = []
    interaction_summaries = [{"summary": "老客，偏好顺丰"}]


class _StubMemory:
    memory_enabled = True
    ltm = _StubLTM()


class _StubSkillManager:
    def load_skill(self, skill_name):
        return {"success": True, "skill_name": skill_name, "instructions": "指令"}


def test_recall_user_memory_uses_ctx_memory():
    ctx = ToolContext(user_id="u1", memory=_StubMemory())
    result = recall_user_memory(ctx=ctx)
    assert result["success"] is True
    assert result["long_term_facts"] == []
    assert result["recent_interactions"] == ["老客，偏好顺丰"]


def test_recall_user_memory_legacy_global_fallback():
    legacy = _StubMemory()
    set_memory_manager(legacy)
    try:
        result = recall_user_memory()  # 无 ctx
        assert result["success"] is True
        assert result["long_term_facts"] == []
    finally:
        set_memory_manager(None)


def test_load_skill_uses_ctx_skill_manager():
    ctx = ToolContext(user_id="u1", skill_manager=_StubSkillManager())
    result = load_skill("process-return", ctx=ctx)
    assert result["success"] is True
    assert result["instructions"] == "指令"


def test_load_skill_legacy_global_fallback():
    set_skill_manager(_StubSkillManager())
    try:
        result = load_skill("track-order")
        assert result["success"] is True
    finally:
        set_skill_manager(None)


# ------------------------------------------------------------
# 全链路透传：registry / ToolManager
# ------------------------------------------------------------
def test_registry_execute_tool_passes_ctx():
    out = json.loads(registry.execute_tool("list_user_orders", {}, ToolContext(user_id="u1")))
    assert out["count"] == 1


def test_registry_execute_tool_legacy_without_ctx():
    out = json.loads(registry.execute_tool("list_user_orders", {}))
    assert out["total"] > 1  # 无身份 → 不做归属过滤（legacy 契约）
    assert out["count"] <= 20


def test_registry_unknown_tool_returns_error():
    out = json.loads(registry.execute_tool("no_such_tool", {}))
    assert out["error"].startswith("未知工具")


def test_tool_manager_passes_ctx_to_local_tools():
    tm = ToolManager(use_mcp=False)
    out = json.loads(tm.execute_tool("list_user_orders", {}, ToolContext(user_id="u3")))
    assert out["count"] == 1
    assert out["orders"][0]["order_id"] == "ORD-20240110-003"


def test_data_tools_accept_ctx_kwarg_and_legacy_positional():
    # 阶段一后所有工具签名带 ctx（末位），旧式调用仍可用
    from app.agent.tools.order import query_order
    from app.agent.tools.refund import submit_refund_application
    from app.integrations.commerce.mock import MockCommerceGateway
    from app.integrations.commerce import set_gateway

    assert query_order("ORD-20240115-001")["success"] is True
    assert query_order("ORD-20240115-001", ctx=ToolContext(user_id="u1"))["success"] is True

    set_gateway(MockCommerceGateway())
    try:
        ctx = ToolContext(user_id="u1", session_id="s")
        assert submit_refund_application(
            "ORD-20240115-001", "尺码不合适", ctx=ctx)["success"] is True
    finally:
        set_gateway(None)


# ------------------------------------------------------------
# MCP connect 握手超时（中危修复 A3）
# ------------------------------------------------------------
def test_mcp_connect_handshake_timeout_raises(monkeypatch):
    """传输层接受连接但 initialize 永不完成 → connect 显式抛 ConnectionError，
    不再静默返回空工具列表（上层误判「MCP 可用但无工具」）。"""
    import asyncio
    from contextlib import asynccontextmanager

    import app.mcp_client.client as mcp_mod

    class _HangSession:
        def __init__(self, read, write):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def initialize(self):
            await asyncio.Event().wait()  # 永不完成

    @asynccontextmanager
    async def _fake_transport(url, http_client=None):
        yield (object(), object(), None)

    monkeypatch.setattr(mcp_mod, "streamable_http_client", _fake_transport)
    monkeypatch.setattr(mcp_mod, "ClientSession", _HangSession)
    monkeypatch.setattr(mcp_mod, "_CONNECT_TIMEOUT_SECONDS", 0.2)

    client = mcp_mod.MCPClient("http://mcp.example/sse")
    with pytest.raises(ConnectionError, match="握手超时"):
        client.connect()
    # 后台协程仍挂在 initialize（daemon 线程），进程退出时自然回收


def test_tool_manager_mcp_connect_failure_falls_back_to_local(monkeypatch):
    """connect 抛错（含握手超时）→ ToolManager 降级本地工具并清掉 MCP 客户端。"""
    import app.mcp_client

    class _TimeoutClient:
        def __init__(self, server_url, auth_token=""):
            pass

        def connect(self):
            raise ConnectionError("MCP 握手超时（30s）")

        def close(self):
            pass

    monkeypatch.setattr(app.mcp_client, "MCPClient", _TimeoutClient)
    manager = ToolManager(use_mcp=True, mcp_server_url="http://mcp.example/sse")
    names = {td["function"]["name"] for td in manager.tool_definitions}
    assert "list_user_orders" in names
    assert manager._mcp_client is None


# ------------------------------------------------------------
# search_knowledge queries 类型守卫（低危修复 A8）
# ------------------------------------------------------------
def test_validate_arguments_drops_non_list_queries():
    """schema 声明 array 但值不是 list → 键被丢弃（可选参数回退单 query
    路径）；修复前字符串原样放行，被逐字符当作子查询。"""
    from app.agent.tools.registry import _validate_arguments

    cleaned, err = _validate_arguments(
        "search_knowledge",
        {"query": "七天无理由退货", "queries": "七天无理由退货"},
    )
    assert err is None
    assert "queries" not in cleaned
    assert cleaned["query"] == "七天无理由退货"

    # 合法数组不受影响
    cleaned2, err2 = _validate_arguments(
        "search_knowledge",
        {"query": "q", "queries": ["退货", "运费"]},
    )
    assert err2 is None
    assert cleaned2["queries"] == ["退货", "运费"]
