from __future__ import annotations

from typing import Optional

from app.agent.context import ToolContext
from app.integrations.commerce import get_gateway
from app.agent.tools.ownership import (
    actor_of,
    credentials_of,
    gateway_failure,
    require_identity,
)

STATUS_LABELS = {
    "pending": "待发货",
    "shipped": "已发货",
    "delivered": "已签收",
    "refund_processing": "退款中",
}


def list_user_orders(ctx: Optional[ToolContext] = None) -> dict:
    """查询当前用户的所有订单概要列表（数据源：CommerceGateway，2.3）。

    2.2：归属强制开启时无身份上下文 → fail-closed（IDENTITY_REQUIRED），
    不再静默返回全部订单；ctx 缺省（旧脚本/CLI 直调且未强制）时返回全部，
    保持教学习惯。
    """
    denied = require_identity(
        ctx, "无法验证操作者身份，已拒绝查询订单列表（订单归属不可校验）",
    )
    if denied is not None:
        return denied
    res = get_gateway().list_orders(actor_of(ctx), credentials_of(ctx))
    if not res.success:
        return gateway_failure(res)
    orders = [
        {
            "order_id": o["order_id"],
            "status": STATUS_LABELS.get(o["status"], o["status"]),
            "items_summary": "、".join(item["name"] for item in o["items"]),
            "total": o["total"],
            "created_at": o["created_at"],
        }
        for o in (res.data or {}).get("orders", [])
    ]
    return {"success": True, "count": len(orders), "orders": orders}