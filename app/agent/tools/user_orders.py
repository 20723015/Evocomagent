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

# 订单列表返回条数上限（P2-1 扩容后单用户订单可达数十条、全量千级）：
# 工具结果不设界会撑爆 Agent 上下文；返回最近 N 条 + total 提示总量，
# 需要具体某单时由 query_order 按订单号查（与真实客服系统一致）。
ORDER_LIST_LIMIT = 20


def list_user_orders(ctx: Optional[ToolContext] = None) -> dict:
    """查询当前用户的订单概要列表（数据源：CommerceGateway，2.3）。

    2.2：归属强制开启时无身份上下文 → fail-closed（IDENTITY_REQUIRED），
    不再静默返回全部订单；ctx 缺省（旧脚本/CLI 直调且未强制）时返回全部，
    保持教学习惯。

    P2-1：按创建时间倒序返回最近 ``ORDER_LIST_LIMIT`` 条，并给出 ``total``
    总量与 ``truncated`` 标记——扩容后不再无界返回。
    """
    denied = require_identity(
        ctx, "无法验证操作者身份，已拒绝查询订单列表（订单归属不可校验）",
    )
    if denied is not None:
        return denied
    res = get_gateway().list_orders(
        actor_of(ctx), credentials_of(ctx), limit=ORDER_LIST_LIMIT,
    )
    if not res.success:
        return gateway_failure(res)
    data = res.data or {}
    orders = [
        {
            "order_id": o["order_id"],
            "status": STATUS_LABELS.get(o["status"], o["status"]),
            "items_summary": "、".join(item["name"] for item in o["items"]),
            "total": o["total"],
            "created_at": o["created_at"],
        }
        for o in data.get("orders", [])
    ]
    total = int(data.get("total", len(orders)) or 0)
    out = {"success": True, "count": len(orders), "total": total, "orders": orders}
    if total > len(orders):
        out["truncated"] = True
        out["note"] = (
            f"仅返回最近 {len(orders)} 条（共 {total} 条）；"
            "需要查看某笔订单请提供订单号调用 query_order。"
        )
    return out