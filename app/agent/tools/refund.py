"""退款申请工具（申请制：Agent 只创建、查询、撤回申请）。

三个公开工具（无旧 apply_refund 别名）：

- ``submit_refund_application(order_id, reason)``：创建申请。身份校验通过后
  **两阶段提交（P1-2）**：首次调用不落库，只登记草稿并返回结构化草稿 + 追问
  话术；用户确认轮之后的调用复用草稿的幂等键真正提交，统一生成可撤回的
  ``merchant_reviewing`` 申请并进入商家审核。订单状态不做任何前置流转；
  是否可创建由订单状态决定，不可退状态由网关 fail-closed 结构化拒绝。
- ``query_refund_application(application_id?, order_id?)``：查询申请，必须且
  只能提供一个条件，统一返回 ``applications`` 数组。
- ``cancel_refund_application(application_id)``：撤回申请，仅
  ``merchant_reviewing`` 可撤回；其余状态由网关结构化拒绝。

``client_request_id`` 由本层自生成（申请级幂等键 + 超时对账锚点），不由模型
提供、也不进入模型可见参数面（registry 层拒绝未声明参数）；两阶段下它在
**草稿登记时**生成，确认后执行复用同一个键——重放安全。
"""

from __future__ import annotations

import uuid
from typing import Optional

from app.agent.context import ToolContext
from app.agent.tools.ownership import (
    actor_of,
    credentials_of,
    gateway_failure,
    require_identity,
)
from app.agent.write_gate import build_draft, is_expired, pending_of
from app.integrations.commerce import get_gateway

TOOL_SUBMIT = "submit_refund_application"
TOOL_QUERY = "query_refund_application"
TOOL_CANCEL = "cancel_refund_application"

QUERY_INVALID = "REFUND_APPLICATION_QUERY_INVALID"
# 草稿态结果码（P1-2）：非错误，是「等待用户确认」的中间态
DRAFT_AWAITING = "REFUND_DRAFT_AWAITING_CONFIRMATION"
DRAFT_CANCELLED = "REFUND_DRAFT_CANCELLED"
DRAFT_EXPIRED = "REFUND_DRAFT_EXPIRED"
DRAFT_TARGET_MISMATCH = "REFUND_DRAFT_TARGET_MISMATCH"
DRAFT_DUPLICATE_IN_TURN = "REFUND_DUPLICATE_IN_TURN"

_AWAIT_TEMPLATE = (
    "已生成退款申请草稿（**尚未提交**）：订单 {order_id}，退款原因「{reason}」。"
    "请把订单号与退款原因复述给用户并请求确认；"
    "用户明确确认后，带 confirm=true 再次调用本工具才会真正提交，"
    "用户取消或改口则不要提交。"
)


def _new_request_id() -> str:
    """写请求的幂等键与超时对账锚点（每次草稿登记唯一，确认后复用）。"""
    return uuid.uuid4().hex


def _draft_reply(draft: dict) -> dict:
    """草稿轮返回：结构化草稿 + 追问话术（零网关调用、零落库）。"""
    arguments = draft.get("arguments") or {}
    order_id = str(arguments.get("order_id", "") or "")
    reason = str(arguments.get("reason", "") or "")
    return {
        "success": True,
        "status": "awaiting_confirmation",
        "code": DRAFT_AWAITING,
        "draft": {
            "order_id": order_id,
            "reason": reason,
            "client_request_id": draft.get("client_request_id", ""),
        },
        "requires_user_confirmation": True,
        "message": _AWAIT_TEMPLATE.format(order_id=order_id, reason=reason),
    }


def _register_draft(ctx: Optional[ToolContext], order_id: str,
                   reason: str) -> dict:
    """登记（或复用）草稿并返回草稿轮响应；绝不落库。"""
    draft = build_draft(
        TOOL_SUBMIT, _new_request_id(),
        {"order_id": order_id, "reason": reason},
        {"order_id": order_id, "reason": reason,
         "summary": f"退款申请：订单 {order_id}，原因「{reason}」"},
    )
    if ctx is not None:
        ctx.pending_write = draft
        persist = getattr(ctx, "persist_pending_write", None)
        if persist is not None:
            persist(draft)
    return _draft_reply(draft)


def _clear_draft(ctx: Optional[ToolContext]) -> None:
    """清除会话草稿（取消/执行完成后调用）。"""
    if ctx is None:
        return
    ctx.pending_write = None
    persist = getattr(ctx, "persist_pending_write", None)
    if persist is not None:
        persist(None)


# ============================================================
# 创建
# ============================================================
def _do_submit(order_id: str, reason: str, ctx: Optional[ToolContext],
               client_request_id: str) -> dict:
    gateway = get_gateway()
    res = gateway.submit_refund_application(
        actor_of(ctx), order_id, reason, client_request_id,
        credentials_of(ctx),
    )
    if not res.success:
        out = gateway_failure(res)
        if res.indeterminate:
            out["client_request_id"] = client_request_id  # 对账锚点
        return out
    data = res.data or {}
    app = dict(data.get("application") or {})
    out = {"success": True, **app}
    if res.code:
        out["code"] = res.code
    if data.get("existing"):
        out["existing"] = True
    if data.get("replayed"):
        out["replayed"] = True
    out.setdefault("message", res.message or "退款申请已创建")
    out["client_request_id"] = client_request_id
    return out


def submit_refund_application(
    order_id: str,
    reason: str,
    ctx: Optional["ToolContext"] = None,
    confirm: bool = False,
) -> dict:
    """为指定订单创建退款申请（两阶段：草稿 → 用户确认 → 提交）。

    执行判据：**`ctx.write_confirm == "confirm"`（服务端判定）才落库**，
    否则一律只返回草稿。草稿的作用是让确认轮可被判定（给用户一个可复述的
    对象）并让幂等键跨轮稳定；Agent 侧只在存在草稿时才可能产生 confirm 结论，
    因此线上路径必然是两阶段。

    ``confirm=true`` 是模型对「用户已确认」的显式声明，参与审计并在「无确认轮
    却声明确认」时打点告警；但**它不是放行依据**——放行只看服务端判定，因此
    模型漏带该参数不会导致「用户确认了却提交不了」，而模型谎报也不会绕过确认轮。
    """
    denied = require_identity(ctx)
    if denied is not None:
        return denied

    order_id = str(order_id or "").strip()
    reason = str(reason or "").strip()
    draft = getattr(ctx, "pending_write", None) if ctx is not None else None
    if draft and is_expired(draft):
        draft = None  # 超期草稿视为不存在（重新登记，不复用旧幂等键）
    action = str(getattr(ctx, "write_confirm", "none") or "none")

    if action == "cancel":
        if draft:
            _clear_draft(ctx)
            return {
                "success": False,
                "code": DRAFT_CANCELLED,
                "message": "用户已取消该退款申请草稿，草稿已作废，未提交任何申请。",
            }
        return {
            "success": False,
            "code": DRAFT_CANCELLED,
            "message": "用户已取消该退款操作，未提交任何申请。",
        }

    if action == "confirm":
        if getattr(ctx, "write_executed", False):
            # 同一轮内二次提交：确认轮每次调用都是新幂等键，必须在此拦住
            return {
                "success": False,
                "code": DRAFT_DUPLICATE_IN_TURN,
                "message": (
                    "本轮已经提交过退款申请，请勿重复提交。"
                    "如需确认进度，请调用 query_refund_application。"
                ),
            }
        entry = (pending_of(draft) or [None])[0]
        if entry is None:
            # 不变量：没有草稿就没有「可确认的对象」→ 绝不落库。
            # 可信的机器调用方（MCP 包装层）同样先走一次草稿轮再声明确认，
            # 因此不存在「凭一句 confirm 直接写」的旁路。
            return _register_draft(ctx, order_id, reason)
        if order_id and entry.order_id and order_id != entry.order_id:
            # 草稿指向另一笔订单：不得把本次调用解释成对该草稿的确认
            return {
                "success": False,
                "code": DRAFT_TARGET_MISMATCH,
                "message": (
                    f"当前待确认的退款草稿是订单 {entry.order_id}，与本次请求的订单 "
                    f"{order_id} 不一致。请先与用户确认要处理哪一笔，"
                    "不要用旧草稿提交新订单。"
                ),
            }
        # 确认轮：复用草稿的幂等键执行写（重放安全）
        result = _do_submit(
            entry.order_id or order_id, entry.reason or reason, ctx,
            entry.client_request_id or _new_request_id(),
        )
        if result.get("success") is not False:
            # 无论成功/失败都标记「本轮已执行」：indeterminate（结果未知）同样
            # 禁止同轮重试——换新幂等键重试等于制造第二笔申请。
            if ctx is not None:
                ctx.write_executed = True
            if draft:
                _clear_draft(ctx)
        return result

    # 未确认（none/ambiguous）：保持或登记草稿，**绝不落库**——本协议的核心不变量
    if draft:
        out = _draft_reply(draft)
        out["message"] = (
            f"退款申请草稿（订单 {out['draft']['order_id']}）仍在等待用户明确确认，"
            "本轮用户消息不构成确认，**未提交**。请先复述订单号与退款原因"
            "并请求确认。"
        )
        return out

    return _register_draft(ctx, order_id, reason)


# ============================================================
# 查询
# ============================================================
def query_refund_application(
    application_id: Optional[str] = None,
    order_id: Optional[str] = None,
    ctx: Optional["ToolContext"] = None,
) -> dict:
    """查询退款申请；必须且只能提供 application_id 或 order_id 之一。"""
    denied = require_identity(ctx)
    if denied is not None:
        return denied
    application_id = str(application_id or "").strip() or None
    order_id = str(order_id or "").strip() or None
    if bool(application_id) == bool(order_id):
        return {
            "success": False,
            "code": QUERY_INVALID,
            "error": "必须且只能提供一个查询条件：application_id 或 order_id",
        }
    gateway = get_gateway()
    res = gateway.query_refund_application(
        actor_of(ctx), application_id=application_id, order_id=order_id,
        credentials=credentials_of(ctx),
    )
    if not res.success:
        return gateway_failure(res)
    return {
        "success": True,
        "applications": list((res.data or {}).get("applications", []) or []),
    }


# ============================================================
# 撤回
# ============================================================
def _do_cancel(application_id: str, ctx: Optional[ToolContext],
               client_request_id: str) -> dict:
    gateway = get_gateway()
    res = gateway.cancel_refund_application(
        actor_of(ctx), application_id, client_request_id, credentials_of(ctx),
    )
    if not res.success:
        out = gateway_failure(res)
        if res.indeterminate:
            out["client_request_id"] = client_request_id
        return out
    data = res.data or {}
    app = dict(data.get("application") or {})
    out = {"success": True, **app}
    if data.get("replayed"):
        out["replayed"] = True
    out.setdefault("message", res.message or "退款申请已撤回")
    out["client_request_id"] = client_request_id
    return out


def cancel_refund_application(
    application_id: str,
    ctx: Optional["ToolContext"] = None,
) -> dict:
    """撤回退款申请（仅 merchant_reviewing 可撤回）。"""
    denied = require_identity(ctx)
    if denied is not None:
        return denied
    return _do_cancel(application_id, ctx, _new_request_id())
