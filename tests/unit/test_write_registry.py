"""批次 2：写工具注册表单一来源一致性（P2-b）。

所有写工具清单一律派生自 ``app/agent/write_registry``；本文件保证：
- 任何一处清单与单一源漂移 → 测试直接失败；
- 新增写工具只注册一行（``_REGISTRY``），漏注册的工具执行前被拦截
  （fail-closed），且派生 API 立即可见。
"""

from __future__ import annotations

import app.agent.write_registry as write_registry
from app.agent.tools import batch_executor
from app.agent.tools.manager import ToolManager
from app.agent.write_ops import WriteOpTracker, WRITE_TOOL_STATE_MACHINES
from app.agent.write_registry import WriteToolSpec


def test_write_tool_lists_derive_from_single_source():
    """四处模块级清单必须与 write_registry 单一源相等。"""
    single = write_registry.write_tools()
    assert single == {"submit_refund_application", "cancel_refund_application"}

    # batch_executor：写工具集合 / 状态机工具集合
    assert batch_executor.WRITE_TOOLS == single
    assert batch_executor._WRITE_STATE_MACHINE_TOOLS == frozenset(
        WRITE_TOOL_STATE_MACHINES
    )

    # manager：写工具与敏感 MCP 集合（敏感 ⊇ 写）
    assert ToolManager._WRITE_TOOLS == single
    assert ToolManager._SENSITIVE_MCP_TOOLS >= single

    # write_ops：状态机表键集合（兼容导出亦是派生）
    assert frozenset(WRITE_TOOL_STATE_MACHINES) == single

    # 每个写工具都有目标标识参数名
    for name in single:
        assert write_registry.write_tool_spec(name).target_arg in (
            "order_id", "application_id",
        )


def test_registry_accessors_and_fail_closed_defaults():
    """spec 派生 API：目标标识 / 未注册工具 fail-closed。"""
    assert write_registry.target_of(
        "submit_refund_application", {"order_id": " O1 "}
    ) == "O1"
    assert write_registry.target_of(
        "cancel_refund_application", {"application_id": "RA-1"}
    ) == "RA-1"
    # 未注册工具：目标回落 order_id（不抛错，由各层拦截）
    assert write_registry.write_tool_spec("not_a_tool") is None
    assert write_registry.target_of("not_a_tool", {"order_id": "O1"}) == "O1"

    # 未注册写工具执行前被拦截（fail-closed）
    tracker = WriteOpTracker()
    blocked = tracker.check("not_a_tool", {"order_id": "O1"})
    assert blocked is not None and "WRITE_TOOL_UNREGISTERED" in blocked


def test_new_write_tool_registration_propagates_via_accessors(monkeypatch):
    """新增写工具只在 _REGISTRY 注册一行：派生 API 全部可见、漏注册被拦截。"""
    monkeypatch.setitem(write_registry._REGISTRY, "resolve_ticket", WriteToolSpec(
        state_machine="ticket_state_machine",
        target_arg="ticket_id",
    ))

    assert "resolve_ticket" in write_registry.write_tools()
    assert write_registry.state_machines()["resolve_ticket"] == "ticket_state_machine"
    assert write_registry.target_of(
        "resolve_ticket", {"ticket_id": "T-1"}
    ) == "T-1"

    # 漏注册的写工具不在任何派生清单中 → 执行前拦截兜底
    tracker = WriteOpTracker()
    assert tracker.check("unregistered_new_tool", {"ticket_id": "T-2"}) is not None
