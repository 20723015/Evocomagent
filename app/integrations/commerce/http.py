"""HTTPCommerceGateway：真实商家后端接入（2.3 契约）。

HTTP 契约（与 kind 的 fake commerce 服务对齐，见 deploy/compose/fake_commerce.py）：
- 归属由下游服务端执行：请求头带 X-Actor-Id（操作者），Authorization Bearer
  携带服务级凭证（从 ToolContext.credentials 取，禁止进入 prompt/轨迹/日志）；
- GET /orders/{order_id}        → 200 {"order": {...}} | 403/404/{code,message}
- GET /orders                   → 200 {"orders": [...]}
- GET /orders/{order_id}/logistics → 200 {"logistics": {...}}
- POST /orders/{order_id}/refunds  body {"reason", "refund_id"} → 200 {"message": ...}

错误映射：401 → IDENTITY_REQUIRED；403 → ORDER_ACCESS_DENIED；
404 → ORDER_NOT_FOUND；其余 4xx/5xx → 通用失败（message 透传）。

韧性策略（2.3 硬性要求）：
- 显式 connect/read 超时（connect 默认 2s、读取默认 5s，可配置）；
- 只对幂等查询（GET）重试（1 次快速重试）；退款 POST 绝不自动重放；
- 退款超时时返回 indeterminate=True（结果未知，转人工对账），
  禁止调用方/工具层自行重放。
"""

from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import quote

from app.agent.tools.ownership import (
    IDENTITY_REQUIRED,
    ORDER_ACCESS_DENIED,
    ORDER_NOT_FOUND,
)
from app.integrations.commerce.base import CommerceGateway, CommerceResult

log = logging.getLogger("app.integrations.commerce.http")

# 凭证键：ToolContext.credentials 中的服务级 Bearer
CRED_TOKEN_KEY = "commerce_token"


class HTTPCommerceGateway(CommerceGateway):
    """基于 httpx 的商家后端（持久化 client，trust_env=False）。"""

    def __init__(
        self,
        base_url: str,
        connect_timeout: float = 2.0,
        read_timeout: float = 5.0,
        client=None,
        read_retries: int = 1,
    ):
        self._base_url = base_url.rstrip("/")
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._read_retries = read_retries
        self._client = client
        self._owns_client = client is None

    def _get_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._read_timeout,
                                      connect=self._connect_timeout),
                trust_env=False,  # 内网商家服务不走系统代理
            )
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    # ---------------- 内部 ----------------
    def _headers(self, actor_id: str,
                 credentials: Optional[dict]) -> dict:
        headers = {}
        if actor_id:
            headers["X-Actor-Id"] = actor_id
        token = ""
        if credentials:
            token = credentials.get(CRED_TOKEN_KEY) or ""
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _map_status(self, status: int, payload: dict, fallback: str) -> CommerceResult:
        """把下游 HTTP 状态映射为机器可判定错误码。"""
        if status in (200, 201):
            return CommerceResult(True, data=payload)
        code = {
            401: IDENTITY_REQUIRED,
            403: ORDER_ACCESS_DENIED,
            404: ORDER_NOT_FOUND,
        }.get(status)
        message = str(payload.get("message") or
                      payload.get("error") or f"商家后端返回 {status}")
        if code:
            return CommerceResult(False, code, message=message)
        return CommerceResult(False, message=message)

    def _request(self, method: str, path: str, headers: dict,
                 json_body: Optional[dict] = None,
                 timeout: Optional[float] = None,
                 idempotent: bool = False) -> tuple[int, dict]:
        """幂等 GET 快速重试（read_retries 次）；写操作绝不重试。

        返回 (status, payload)；网络异常统一按「后端不可用」处理。
        """
        import httpx

        url = httpx.URL(self._base_url).join(path)
        attempts = (self._read_retries + 1) if idempotent else 1
        last_exc: Exception | None = None
        for _ in range(attempts):
            try:
                resp = self._get_client().request(
                    method, url, headers=headers, json=json_body,
                    timeout=timeout,
                )
                try:
                    payload = resp.json()
                except ValueError:
                    payload = {}
                # 幂等读在 5xx（后端瞬时可恢复）时重试；写操作一律不重试
                if idempotent and resp.status_code >= 500:
                    last_exc = ConnectionError(
                        f"商家后端返回 {resp.status_code}"
                    )
                    continue
                return resp.status_code, payload
            except Exception as e:  # noqa: BLE001 —— 网络/超时/连接失败统一降级
                last_exc = e
        raise ConnectionError(f"商家后端请求失败: {last_exc}")

    # ---------------- 契约实现 ----------------
    def get_order(self, actor_id: str, order_id: str,
                  credentials: Optional[dict] = None) -> CommerceResult:
        if not actor_id:
            return CommerceResult(False, IDENTITY_REQUIRED,
                                  message="无法验证操作者身份（缺少 actor_id）")
        try:
            status, payload = self._request(
                "GET", f"/orders/{quote(order_id, safe='')}",
                self._headers(actor_id, credentials), idempotent=True,
            )
            return self._map_status(status, payload, "订单查询失败")
        except ConnectionError as e:
            return CommerceResult(False, message=f"订单查询失败（后端不可用）: {e}")

    def list_orders(self, actor_id: str,
                    credentials: Optional[dict] = None) -> CommerceResult:
        if not actor_id:
            return CommerceResult(False, IDENTITY_REQUIRED,
                                  message="无法验证操作者身份（缺少 actor_id）")
        try:
            status, payload = self._request(
                "GET", "/orders", self._headers(actor_id, credentials),
                idempotent=True,
            )
            return self._map_status(status, payload, "订单列表查询失败")
        except ConnectionError as e:
            return CommerceResult(False, message=f"订单列表查询失败（后端不可用）: {e}")

    def get_logistics(self, actor_id: str, order_id: str,
                      credentials: Optional[dict] = None) -> CommerceResult:
        if not actor_id:
            return CommerceResult(False, IDENTITY_REQUIRED,
                                  message="无法验证操作者身份（缺少 actor_id）")
        try:
            status, payload = self._request(
                "GET", f"/orders/{quote(order_id, safe='')}/logistics",
                self._headers(actor_id, credentials), idempotent=True,
            )
            return self._map_status(status, payload, "物流查询失败")
        except ConnectionError as e:
            return CommerceResult(False, message=f"物流查询失败（后端不可用）: {e}")

    def request_refund(
        self,
        actor_id: str,
        order_id: str,
        reason: str,
        idempotency_key: str,
        credentials: Optional[dict] = None,
    ) -> CommerceResult:
        """发起退款：必带 refund_id 幂等键；绝不自动重试；超时 → indeterminate。"""
        if not actor_id:
            return CommerceResult(False, IDENTITY_REQUIRED,
                                  message="无法验证操作者身份（缺少 actor_id）")
        try:
            status, payload = self._request(
                "POST",
                f"/orders/{quote(order_id, safe='')}/refunds",
                self._headers(actor_id, credentials),
                json_body={"reason": reason, "refund_id": idempotency_key},
            )
        except ConnectionError as e:
            # 超时/网络断开：结果未知，禁止重放
            return CommerceResult(
                False, indeterminate=True,
                message=f"退款结果未知（后端超时），请勿重复提交，人工对账: {e}",
            )
        return self._map_status(status, payload, "退款申请失败")