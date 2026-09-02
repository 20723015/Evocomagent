"""CommerceGateway 工厂：按 COMMERCE_BACKEND 装配（2.3）。

- mock：MockCommerceGateway（开发/离线评测）；
- http：HTTPCommerceGateway（生产；COMMERCE_BASE_URL 未配置时 fail-fast）。

生产配置（values-production.yaml）使用 http；没有真实商家 staging 环境前，
简历只写「业务工具调用」，不写「接入真实订单系统」。
"""

from __future__ import annotations

import threading
from typing import Optional

from app.config.settings import settings
from app.integrations.commerce.base import CommerceGateway

COMMERCE_BACKEND_MOCK = "mock"
COMMERCE_BACKEND_HTTP = "http"


class CommerceConfigError(ValueError):
    """commerce 配置错误（生产缺 URL 等）。"""


_instance: Optional[CommerceGateway] = None
_instance_lock = threading.Lock()


def create_commerce_gateway() -> CommerceGateway:
    """按配置构造网关（模块单例由 get_gateway 持有）。"""
    backend = settings.commerce_backend
    if backend == COMMERCE_BACKEND_MOCK:
        from app.integrations.commerce.mock import MockCommerceGateway
        return MockCommerceGateway()
    if backend == COMMERCE_BACKEND_HTTP:
        if not settings.commerce_base_url:
            raise CommerceConfigError(
                "COMMERCE_BACKEND=http 但 COMMERCE_BASE_URL 未配置（拒绝启动）"
            )
        from app.integrations.commerce.http import HTTPCommerceGateway
        return HTTPCommerceGateway(
            base_url=settings.commerce_base_url,
            connect_timeout=settings.commerce_connect_timeout_seconds,
            read_timeout=settings.commerce_timeout_seconds,
        )
    raise CommerceConfigError(
        f"未知 COMMERCE_BACKEND={backend!r}（可选: mock / http）"
    )


def get_gateway() -> CommerceGateway:
    """进程级单例（懒加载；测试用 set_gateway 注入替身）。"""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = create_commerce_gateway()
    return _instance


def set_gateway(gateway: Optional[CommerceGateway]) -> None:
    """测试注入：传 None 复位为按配置重建。"""
    global _instance
    with _instance_lock:
        _instance = gateway


def validate_commerce_config() -> None:
    """生产启动校验：backend=http 缺 URL 直接失败（fail-fast）。"""
    create_commerce_gateway()  # 仅校验配置可构造，不持有单例