"""Redis 连接与就绪检查（阶段二 2.7）。

- build_redis：连不上返回 None（降级走文件实现，开发环境友好）；
- require_redis：settings.redis_required=True 时连不上直接抛（生产 fast-fail）；
- get_redis 单例：pod 内共享一个连接池。
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from app.config.settings import settings

logger = logging.getLogger("app.stores")

_UNSET = object()
_client: object = _UNSET
_client_lock = threading.Lock()


def build_redis(timeout: float = 2.0):
    """构建并 ping 一次；失败返回 None（日志 warn，不阻断启动）。"""
    try:
        import redis
    except ImportError:
        logger.warning("redis 未安装，状态外置降级为本地文件")
        return None
    try:
        client = redis.Redis.from_url(
            settings.redis_url,
            socket_timeout=timeout,
            socket_connect_timeout=timeout,
        )
        client.ping()
        return client
    except Exception as e:  # noqa: BLE001 —— 连接失败视为不可用
        logger.warning("Redis 不可用（%s），状态外置降级为本地文件", e)
        return None


def require_redis() -> object:
    """生产启动检查：redis_required=True 时连不上直接 fast-fail。"""
    client = build_redis()
    if client is None:
        raise RuntimeError(
            "Redis 不可用且 redis_required=True（生产部署必须：会话/记忆/锁外置）"
        )
    return client


def get_redis() -> Optional[object]:
    """pod 级共享 Redis 客户端：首次构建后缓存（含「不可用=None」缓存，不再重复探活）。"""
    global _client
    if _client is _UNSET:
        with _client_lock:
            if _client is _UNSET:
                _client = require_redis() if settings.redis_required else build_redis()
    return _client if _client is not _UNSET else None


def set_redis_for_test(client) -> None:
    """测试注入：替换共享客户端（fakeredis）；传 None 表示明确不可用（不再探活）。"""
    global _client
    _client = client
