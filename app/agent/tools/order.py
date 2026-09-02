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


def query_order(order_id: str, ctx: Optional["ToolContext"] = None) -> dict:
    """根据订单号查询订单详情，包括状态、商品、金额、物流等信息。

    2.2：错误统一带机器可判定 code（ORDER_NOT_FOUND / ORDER_ACCESS_DENIED /
    IDENTITY_REQUIRED），success/error 字段保持兼容；
    2.3：数据源为 CommerceGateway（mock=内存数据 / http=真实后端）。
    """
    denied = require_identity(ctx)
    if denied is not None:
        return denied
    res = get_gateway().get_order(
        actor_of(ctx), order_id, credentials_of(ctx),
    )
    if not res.success:
        return gateway_failure(res)
    return {"success": True, "order": res.data}