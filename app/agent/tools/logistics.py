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


def query_logistics(order_id: str, ctx: Optional["ToolContext"] = None) -> dict:
    """根据订单号查询物流轨迹信息（归属校验与订单查询同级）。

    2.2：错误统一带机器可判定 code；
    2.3：数据源为 CommerceGateway。
    """
    denied = require_identity(ctx)
    if denied is not None:
        return denied
    res = get_gateway().get_logistics(
        actor_of(ctx), order_id, credentials_of(ctx),
    )
    if not res.success:
        return gateway_failure(res)
    data = res.data or {}
    return {"success": True, "logistics": data.get("logistics", {})}