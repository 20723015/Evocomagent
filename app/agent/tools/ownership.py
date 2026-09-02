"""工具级授权：订单归属校验（阶段三 3.1/3.2；2.2 统一机器可判定错误码）。

enforce_order_ownership 开启时，跨用户访问/操作一律拒绝（403 语义）。
拒绝发生在工具内部——数据不出工具层，绝不会以「部分内容」泄给提问者。

错误码（2.2 冻结，供评测与客户端机器判定；success/error 字段保持兼容）：
- IDENTITY_REQUIRED：无操作者身份（ctx 缺失 / user_id 为空）→ fail-closed；
- ORDER_ACCESS_DENIED：订单存在但不属于当前用户；
- ORDER_NOT_FOUND：订单不存在（由各工具在下游返回，不泄露归属信息）。

开关解析顺序：ctx.enforce_order_ownership 显式设置 > 全局 settings——
评测沙箱显式传 True，测试不再依赖全局 .env。
"""

from __future__ import annotations

from typing import Optional

from app.agent.context import ToolContext
from app.config.settings import settings
from app.agent.tools.mock_data import ORDERS

# 机器可判定错误码（2.2）
IDENTITY_REQUIRED = "IDENTITY_REQUIRED"
ORDER_ACCESS_DENIED = "ORDER_ACCESS_DENIED"
ORDER_NOT_FOUND = "ORDER_NOT_FOUND"
CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
CONFIRMATION_INVALID = "CONFIRMATION_INVALID"


def effective_enforce(ctx: Optional[ToolContext]) -> bool:
    """归属强制开关：请求级覆盖优先，否则跟随全局配置。"""
    if ctx is not None and ctx.enforce_order_ownership is not None:
        return ctx.enforce_order_ownership
    return settings.enforce_order_ownership


def identity_required_error(message: str = "") -> dict:
    return {
        "success": False,
        "code": IDENTITY_REQUIRED,
        "error": message or "无法验证操作者身份，已拒绝该操作（订单归属不可校验）",
    }


def access_denied_error(message: str = "") -> dict:
    return {
        "success": False,
        "code": ORDER_ACCESS_DENIED,
        "error": message or "无权访问该订单（不属于当前用户）",
    }


def not_found_error(order_id: str) -> dict:
    return {
        "success": False,
        "code": ORDER_NOT_FOUND,
        "error": f"未找到订单 {order_id}，请核实订单号",
    }


def check_order_ownership(
    order_id: str, ctx: Optional[ToolContext],
) -> Optional[dict]:
    """通过返回 None；拒绝返回带 code 的错误结果 dict。

    安全修复 P1 + 2.2：enforce 开启而拿不到操作者身份（ctx=None，如 MCP/
    旧直调）→ fail-closed 拒绝；订单不存在返回 None（由工具返回
    ORDER_NOT_FOUND），不泄露「存在性」之外的任何信息。
    """
    if not effective_enforce(ctx):
        return None
    user_id = ctx.user_id if ctx is not None else ""
    if not user_id:
        return identity_required_error()
    order = ORDERS.get(order_id)
    if order is None:
        return None  # 不存在由各工具返回「未找到」，不泄露存在性之外的任何信息
    if order.get("user_id") != user_id:
        return access_denied_error()
    return None


# ---------- 2.3 网关接入辅助 ----------
def actor_of(ctx: Optional[ToolContext]) -> str:
    """操作者身份（空串=无身份直调，仅 mock 网关兼容开发）。"""
    return ctx.user_id if ctx is not None else ""


def credentials_of(ctx: Optional[ToolContext]) -> Optional[dict]:
    """请求级外部凭证（Bearer 来源）；永不进 prompt/轨迹/日志。"""
    return getattr(ctx, "credentials", None) if ctx is not None else None


def require_identity(ctx: Optional[ToolContext], message: str = "") -> Optional[dict]:
    """fail-closed：enforce 开启且无操作者身份 → IDENTITY_REQUIRED。"""
    if effective_enforce(ctx) and not actor_of(ctx):
        return identity_required_error(message)
    return None


def gateway_failure(res) -> dict:
    """把网关失败结果映射为工具返回（success/error/code 兼容；indeterminate 带状态）。"""
    out: dict = {"success": False, "error": getattr(res, "message", None) or "请求失败"}
    code = getattr(res, "code", None)
    if code:
        out["code"] = code
    if getattr(res, "indeterminate", False):
        out["status"] = "indeterminate"
    return out