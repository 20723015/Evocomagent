"""ES 客户端 provider（阶段八；修复计划·三：可恢复，不再永久缓存失败）。

- 按 settings.es_url 构建；未配置 → None（不依赖 ES）；
- **可恢复**：构建/探测失败记录最近错误并进入 5 秒重连冷却；冷却结束后的下次
  请求会重新初始化（历史实现把 None 永久缓存，ES 恢复后必须重启 Pod）；
- 操作失败时调用 invalidate_es_client() 使当前客户端失效，后续请求可重建；
- readiness、RAG、消息搜索、Outbox 共用同一个 provider；
- 测试可注入替身（set_es_for_test）。
"""

from __future__ import annotations

import threading
import time

from app.config.settings import settings

_UNSET = object()
_client = _UNSET
_last_error: str = ""
_retry_after: float = 0.0
_injected = False  # 测试注入的替身：invalidate 不使其失效
RECONNECT_COOLDOWN_SECONDS = 5.0
_lock = threading.Lock()


def build_es_client(timeout: float = 5.0):
    """构建并探测 ES 客户端；失败返回 None（调用方走冷却/降级）。"""
    if not settings.es_url:
        return None
    try:
        from elasticsearch import Elasticsearch

        params = dict(hosts=[settings.es_url], request_timeout=timeout)
        if settings.es_user:
            params["basic_auth"] = (settings.es_user, settings.es_password)
        client = Elasticsearch(**params)
        client.info()  # 探测：失败返回 None（降级）
        return client
    except Exception:  # noqa: BLE001
        return None


def get_es_client():
    """取 ES 客户端；失败进入冷却，冷却后自动重试（ES 恢复无需重启）。"""
    global _client, _last_error, _retry_after
    if _client is not _UNSET and _client is not None:
        return _client
    if _client is None:
        # set_es_for_test(None)/显式不可用：保持不可用，不自动探测
        return None
    if time.monotonic() < _retry_after:
        return None
    with _lock:
        if _client is not _UNSET and _client is not None:
            return _client
        if _client is None:
            return None
        client = build_es_client()
        if client is None:
            _last_error = "connect_failed"
            _retry_after = time.monotonic() + RECONNECT_COOLDOWN_SECONDS
            return None
        _client = client
        _last_error = ""
        _retry_after = 0.0
        return _client


def _close_quietly(client) -> None:
    """best-effort 关闭旧连接（不阻断失效/重建路径）。"""
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001
            pass


def invalidate_es_client(reason: str = "") -> None:
    """操作失败时使当前客户端失效：后续请求可在冷却后重新初始化。

    旧客户端 best-effort 关闭（释放连接池）；测试注入的替身不参与恢复语义。
    """
    global _client, _last_error, _retry_after
    if _injected:
        return
    old = None
    with _lock:
        if _client is not _UNSET and _client is not None:
            old = _client
        _client = _UNSET
        _last_error = reason or "operation_failed"
        _retry_after = time.monotonic() + RECONNECT_COOLDOWN_SECONDS
    if old is not None:
        _close_quietly(old)


def set_es_for_test(client) -> None:
    """测试注入（FakeTransport）；传 None 表示明确不可用（不自动恢复）。"""
    global _client, _last_error, _retry_after, _injected
    with _lock:
        _client = client
        _injected = True
        _last_error = "" if client is not None else "test_injected_unavailable"
        _retry_after = 0.0


def reset_es_for_test() -> None:
    """测试复位：回到未探测状态。"""
    global _client, _last_error, _retry_after, _injected
    with _lock:
        _client = _UNSET
        _injected = False
        _last_error = ""
        _retry_after = 0.0


def last_es_error() -> str:
    return _last_error


def es_dependency_state() -> str:
    """`not_configured` / `ok` / `unavailable`（readiness 用，会实际探活）。"""
    if not settings.es_url:
        return "not_configured"
    client = get_es_client()
    if client is None:
        return "unavailable"
    try:
        client.info()
    except Exception as e:  # noqa: BLE001 —— 探活失败：失效并标不可用
        invalidate_es_client(f"probe_failed:{type(e).__name__}")
        return "unavailable"
    return "ok"
