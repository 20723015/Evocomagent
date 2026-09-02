"""退款工具（阶段三 3.3：两段式硬确认；3.2：归属校验；2.2：统一错误码；
2.3：经 CommerceGateway 执行退款）。

- refund_confirmation_required=True（生产）：apply_refund 第一段只签发
  refund_id + 一次性 confirmation_token（5min TTL），第二段带 token +
  refund_id 才真正执行；同 refund_id 重复确认返回首次结果（replayed=true）；
- enforce_order_ownership=True：订单归属不符直接拒绝（工具级授权，403 语义；
  凭证永不进 prompt——agent 只看到拒绝话术，看不到他人订单数据）。
  安全修复 P1：无 ctx / 无身份时 **fail-closed 拒绝**（MCP 等无身份路径
  不再因 `if user_id and ...` 被跳过而绕过归属校验）；
- 2.3：网关退款不自动重放——超时返回 indeterminate（结果未知，转人工对账）。

2.2 机器可判定错误码（success/error 字段保持兼容）：
- IDENTITY_REQUIRED / ORDER_ACCESS_DENIED / ORDER_NOT_FOUND（归属层）；
- CONFIRMATION_REQUIRED：确认开启但本次调用只走了第一段（签发阶段）；
- CONFIRMATION_INVALID：确认令牌无效/过期/已使用/不匹配。
"""

from __future__ import annotations

import threading
from typing import Optional

from app.agent.context import ToolContext
from app.config.settings import settings

from app.integrations.commerce import get_gateway
from app.agent.tools.ownership import (
    CONFIRMATION_INVALID,
    CONFIRMATION_REQUIRED,
    actor_of,
    credentials_of,
    gateway_failure,
    require_identity,
)

_store_instance = None
_store_lock = threading.Lock()


def _confirmation_store():
    """token + 幂等账本共用的存储：Redis 可用用 Redis，否则进程内（3.3）。"""
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                from app.security.refunds import (
                    InProcessConfirmationStore,
                    RedisConfirmationStore,
                )
                from app.stores.redis_client import get_redis

                redis = get_redis()
                _store_instance = (
                    RedisConfirmationStore(redis) if redis is not None
                    else InProcessConfirmationStore()
                )
    return _store_instance


def _do_refund(order_id: str, reason: str, ctx: Optional[ToolContext],
               idempotency_key: str = "") -> dict:
    """真正的退款执行（两段确认之后）。

    idempotency_key（refund_id）透传下游做幂等；网关不自动重放：
    超时结果 indeterminate（结果未知，转人工对账，禁止重试重放）。
    """
    gateway = get_gateway()
    res = gateway.request_refund(
        actor_of(ctx), order_id, reason, idempotency_key, credentials_of(ctx),
    )
    if not res.success:
        out = gateway_failure(res)
        if res.indeterminate:
            out["refund_id"] = idempotency_key  # 对账锚点
        return out
    return {"success": True, **(res.data or {})}


def apply_refund(
    order_id: str,
    reason: str,
    ctx: Optional["ToolContext"] = None,
    confirmation_token: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    refund_id: Optional[str] = None,
) -> dict:
    """为指定订单申请退款（两段式：先确认后执行）。

    refund_id / idempotency_key：第一段返回的幂等锚点（同值重复 confirm
    幂等重放；不传则退回旧哈希键——仅供兼容，新调用应传 refund_id）。
    """
    denied = require_identity(ctx)
    if denied is not None:
        return denied

    if not settings.refund_confirmation_required:
        return _do_refund(order_id, reason, ctx)

    from app.security.refunds import ConfirmationInvalid, RefundConfirmation

    controller = RefundConfirmation(_confirmation_store())
    if not confirmation_token:
        # 第一段：先过归属（网关侧校验防越权签发），再签发
        # refund_id + 一次性确认凭证，绝不直接执行
        gateway = get_gateway()
        pre = gateway.get_order(actor_of(ctx), order_id, credentials_of(ctx))
        if not pre.success:
            return gateway_failure(pre)
        return {**controller.request(order_id, reason),
                "code": CONFIRMATION_REQUIRED}
    try:
        return controller.confirm(
            confirmation_token,
            executor=lambda oid, rsn, key: _do_refund(oid, rsn, ctx, key),
            order_id=order_id,
            reason=reason,
            refund_id=refund_id or idempotency_key or "",
        )
    except ConfirmationInvalid as e:
        return {
            "success": False,
            "status": "confirmation_failed",
            "code": CONFIRMATION_INVALID,
            "error": str(e),
        }