"""商家业务网关：订单/物流/退款的统一数据源抽象（2.3）。

工具层不再直读 mock 数据；所有订单域数据经 CommerceGateway 获取。
- MockCommerceGateway：开发/离线评测（内存 ORDERS/LOGISTICS）；
- HTTPCommerceGateway：真实商家后端（契约见 http.py）。
"""

from __future__ import annotations

from app.integrations.commerce.base import CommerceGateway, CommerceResult
from app.integrations.commerce.factory import (
    COMMERCE_BACKEND_HTTP,
    COMMERCE_BACKEND_MOCK,
    CommerceConfigError,
    create_commerce_gateway,
    get_gateway,
    set_gateway,
)

__all__ = [
    "CommerceGateway",
    "CommerceResult",
    "CommerceConfigError",
    "COMMERCE_BACKEND_MOCK",
    "COMMERCE_BACKEND_HTTP",
    "create_commerce_gateway",
    "get_gateway",
    "set_gateway",
]
