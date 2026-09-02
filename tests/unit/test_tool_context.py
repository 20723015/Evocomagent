"""阶段一 1.3/1.5：ToolContext 透传 + 按用户隔离。

覆盖：ctx 注入的 memory/skill 句柄、list_user_orders 按 user_id 过滤、
registry/ToolManager 的 ctx 全链路透传、无 ctx 的旧脚本兼容。
"""

from __future__ import annotations

import json

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
    # 无 ctx（旧脚本直调）保持返回全部，教学场景不回退
    all_orders = list_user_orders()
    assert all_orders["count"] == 5


# ------------------------------------------------------------
# memory / skill 句柄：ctx 优先，全局回落仅作旧兼容
# ------------------------------------------------------------
class _StubLTM:
    facts = []
    interaction_summaries = [{"summary": "老客，偏好顺丰"}]


class _StubSTM:
    facts = ["用户住在深圳"]


class _StubMemory:
    memory_enabled = True
    stm = _StubSTM()
    ltm = _StubLTM()


class _StubSkillManager:
    def load_skill(self, skill_name):
        return {"success": True, "skill_name": skill_name, "instructions": "指令"}


def test_recall_user_memory_uses_ctx_memory():
    ctx = ToolContext(user_id="u1", memory=_StubMemory())
    result = recall_user_memory(ctx=ctx)
    assert result["success"] is True
    assert result["short_term_facts"] == ["用户住在深圳"]
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
    assert out["count"] == 5


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
    from app.agent.tools.refund import apply_refund

    assert query_order("ORD-20240115-001")["success"] is True
    assert query_order("ORD-20240115-001", ctx=ToolContext(user_id="u1"))["success"] is True
    assert apply_refund("ORD-20240115-001", "尺码不合适",
                        ctx=ToolContext(user_id="u1"))["success"] is True
