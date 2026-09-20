"""退款申请（申请制）：工具行为 + 终答状态白名单 + 写状态机观察（无网络）。

覆盖：
- 工具：任意可退状态统一创建 merchant_reviewing 申请且订单状态不变、
  重复提交幂等、不可退状态结构化拒绝、撤回仅 merchant_reviewing、
  查询必须且只能一个条件、结果永不出现确认类中间态；
- 终答：只有对应业务回执才能宣称对应状态（merchant_reviewing /
  refund_processing / refunded），无回执越级宣称被确定性改写；
- 状态机：网关 indeterminate 不误判 rejected 且保留对账锚点。
"""

from __future__ import annotations

import json

import pytest

from app.agent.context import ToolContext
from app.agent.tools.refund import (
    cancel_refund_application,
    query_refund_application,
    submit_refund_application,
)
from app.agent.write_ops import WriteOpTracker
from app.integrations.commerce import get_gateway, set_gateway
from app.integrations.commerce.mock import MockCommerceGateway


@pytest.fixture(autouse=True)
def _isolated():
    set_gateway(MockCommerceGateway())
    yield
    set_gateway(None)


def _ctx(user_id="u1", session="s"):
    return ToolContext(user_id=user_id, session_id=session)


def _submit_confirmed(order_id, reason, ctx):
    """两阶段协议（P1-2）：草稿轮 → 确认轮，返回真正提交的结果。

    首调只登记草稿、不落库；本助手模拟「用户已确认」的下一轮调用，让既有
    「提交语义」用例继续测各自的关注点（确认闸门本身见 test_write_confirmation）。
    """
    ctx.write_confirm = "none"      # 新的一轮：表态从零开始（真实 agent 每轮重算）
    ctx.write_executed = False
    draft = submit_refund_application(order_id, reason, ctx)
    assert draft["status"] == "awaiting_confirmation"
    assert ctx.pending_write is not None
    ctx.write_confirm = "confirm"   # 用户确认轮
    return submit_refund_application(order_id, reason, ctx)


# ============================================================
# 工具层：统一创建语义
# ============================================================
def test_submit_pending_creates_merchant_reviewing_without_order_change():
    """未发货（pending）同样直接创建申请，订单状态不做前置流转。"""
    out = _submit_confirmed("ORD-20240120-002", "不想要了", _ctx("u2"))
    assert out["success"] is True
    assert out["status"] == "merchant_reviewing"
    assert out["can_withdraw"] is True
    assert out["application_id"].startswith("RA-")
    # 订单状态保持不变（不再进入 cancelled）
    order = get_gateway().get_order("u2", "ORD-20240120-002").data
    assert order["status"] == "pending"


def test_submit_shipped_creates_merchant_reviewing():
    out = _submit_confirmed("ORD-20240115-001", "尺码不合适", _ctx("u1"))
    assert out["success"] is True
    assert out["status"] == "merchant_reviewing"
    assert out["can_withdraw"] is True


def test_submit_duplicate_returns_existing_application():
    """重复提交：网关按进行中申请短路，返回同一申请（不重复创建）。"""
    ctx = _ctx("u1")
    first = _submit_confirmed("ORD-20240115-001", "尺码不合适", ctx)
    # 第二次提交：重新走两阶段（草稿 → 确认），网关按进行中申请短路
    second = _submit_confirmed("ORD-20240115-001", "质量问题", ctx)
    assert second["success"] is True
    assert second["existing"] is True
    assert second["code"] == "REFUND_APPLICATION_EXISTS"
    assert second["application_id"] == first["application_id"]
    assert second["status"] == "merchant_reviewing"


def test_submit_not_refundable_state_rejected():
    # ORD-20240118-004 处于 refund_processing（不可退）
    out = _submit_confirmed("ORD-20240118-004", "不想要了", _ctx("u4"))
    assert out["success"] is False
    assert out["code"] == "REFUND_ORDER_STATE_NOT_REFUNDABLE"


def test_submit_result_never_contains_confirmation_midstate():
    """提交轮结果不出现任何授权/确认类内部字段；草稿轮同样不泄漏凭证。

    P1-2 后草稿轮会返回 status=awaiting_confirmation（协议中间态，属对外
    可见的草稿语义），但两轮都不得出现 authorization 凭证字段。
    """
    ctx = _ctx("u2")
    draft = submit_refund_application("ORD-20240120-002", "不想要了", ctx)
    draft_blob = json.dumps(draft, ensure_ascii=False).lower()
    assert "authorization" not in draft_blob
    assert "token" not in draft_blob

    ctx.write_confirm = "confirm"
    out = submit_refund_application("ORD-20240120-002", "不想要了", ctx)
    blob = json.dumps(out, ensure_ascii=False).lower()
    assert "authorization" not in blob
    assert "token" not in blob


def test_query_requires_exactly_one_condition():
    out = query_refund_application(ctx=_ctx())
    assert out["success"] is False and out["code"] == "REFUND_APPLICATION_QUERY_INVALID"
    out2 = query_refund_application(
        application_id="RA-1", order_id="ORD-1", ctx=_ctx(),
    )
    assert out2["success"] is False


def test_cancel_only_merchant_reviewing_then_rejected():
    created = _submit_confirmed("ORD-20240115-001", "质量问题", _ctx("u1"))
    app_id = created["application_id"]
    done = cancel_refund_application(app_id, _ctx("u1"))
    assert done["success"] is True and done["status"] == "withdrawn"
    # 再次撤回：网关结构化拒绝（非 merchant_reviewing）
    again = cancel_refund_application(app_id, _ctx("u1"))
    assert again["success"] is False
    assert again["code"] == "REFUND_APPLICATION_NOT_WITHDRAWABLE"


# ============================================================
# 终答状态白名单
# ============================================================
def test_final_reply_guard_requires_matching_business_status():
    tracker = WriteOpTracker()
    tracker.business_statuses.add("merchant_reviewing")
    # 有 merchant_reviewing 证据：申请已提交可说
    reply, rewritten = tracker.final_reply_guard("您的退款申请已提交，等待商家审核。")
    assert rewritten is False
    # 无 refunded 证据：退款完成被改写
    reply2, rewritten2 = tracker.final_reply_guard("您的退款已完成，款项已退回。")
    assert rewritten2 is True and "尚未返回对应的业务回执" in reply2
    # 无 refund_processing 证据：处理中宣称被改写
    _, rewritten3 = tracker.final_reply_guard("您的退款正在处理中。")
    assert rewritten3 is True


def test_final_reply_guard_allows_refunded_with_evidence():
    tracker = WriteOpTracker()
    tracker.business_statuses.add("refunded")
    _, rewritten = tracker.final_reply_guard("您的退款已完成。")
    assert rewritten is False


def test_tracker_observe_records_business_status():
    tracker = WriteOpTracker()
    tracker.observe(
        "submit_refund_application", {"order_id": "O1", "reason": "x"},
        {"success": True, "status": "merchant_reviewing",
         "application_id": "RA-1", "client_request_id": "r1"},
    )
    assert tracker.business_statuses == {"merchant_reviewing"}
    assert tracker.committed and tracker.committed[0].target == "O1"


def test_tracker_observe_gateway_indeterminate_not_rejected():
    """网关超时返回 success=False + status=indeterminate：不得误判 rejected。

    结果载荷自带 client_request_id（工具层自生成并回写），对账锚点由此承载。
    """
    tracker = WriteOpTracker()
    tracker.observe(
        "submit_refund_application",
        {"order_id": "O1", "reason": "x"},
        {"success": False, "status": "indeterminate", "client_request_id": "r1"},
    )
    assert tracker.records[-1].phase == "indeterminate"
    assert tracker.has_indeterminate is True
    assert tracker.indeterminate == [{
        "tool": "submit_refund_application",
        "order_id": "O1",
        "application_id": "",
        "client_request_id": "r1",
    }]


def test_tracker_observe_plain_failure_still_rejected():
    """普通失败（无 indeterminate 状态）仍走 rejected，行为不变。"""
    tracker = WriteOpTracker()
    tracker.observe(
        "submit_refund_application", {"order_id": "O1", "reason": "x"},
        {"success": False, "code": "ORDER_ACCESS_DENIED"},
    )
    assert tracker.records[-1].phase == "rejected"
    assert tracker.has_indeterminate is False
    assert "O1" in tracker.denied_targets
