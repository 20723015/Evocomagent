"""ES 客户端单例（阶段八）：按 settings.es_url 构建；无配置/连不上 → None（降级）。

与 redis_client 同构：首次构建后缓存（含「不可用」缓存）；测试可注入替身。
"""

from __future__ import annotations

import threading

from app.config.settings import settings

_UNSET = object()
_client = _UNSET
_lock = threading.Lock()


def build_es_client(timeout: float = 5.0):
    if not settings.es_url:
        return None
    from elasticsearch import Elasticsearch

    try:
        params = dict(hosts=[settings.es_url], request_timeout=timeout)
        if settings.es_user:
            params["basic_auth"] = (settings.es_user, settings.es_password)
        client = Elasticsearch(**params)
        client.info()  # 探测：失败返回 None（降级）
        return client
    except Exception as e:  # noqa: BLE001
        return None


def get_es_client():
    global _client
    if _client is _UNSET:
        with _lock:
            if _client is _UNSET:
                _client = build_es_client()
    return _client if _client is not _UNSET else None


def set_es_for_test(client) -> None:
    """测试注入（FakeTransport）；传 None 表示明确不可用。"""
    global _client
    _client = client
