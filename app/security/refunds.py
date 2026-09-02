"""退款两段式硬确认（阶段三 3.3；安全修复 P1 幂等状态机）。

协议：apply_refund 不再直接执行——
1. 发起：签发 refund_id（幂等锚点）+ 一次性 confirm_token（TTL 内有效）；
2. 确认：带 token + refund_id 执行；token 单次有效（天然拒绝并发双确认）；
3. 幂等：同一 refund_id 重复 confirm 返回首次结果（replayed=true，
   防网络重试误执行两次）。

历史缺陷（本次修复）：
- 结果账本原是控制器实例字段，而 apply_refund 每次调用新建控制器——
  幂等从未生效。现把账本放进与 token 同源的共享 store（进程内 / Redis）。
- 幂等键原由「订单+原因」哈希派生，同单同因的正当二次退款会被误判为
  重放。现改由 request() 签发的 refund_id 锚定；订单层防双退仍由
  退款执行器的 refund_processing 状态兜底。

**[评审修订·坑2]** mock 执行器没有下游去重语义，「执行后崩溃重试只执行
一次」只有真实后端能保证。测试只断言：executor 收到的幂等键一致、状态机
拒绝并发双确认（token 单次）、重复 confirm 跨控制器实例 replayed 且
executor 恰好执行一次。不写「下游只执行一次」的假验收。

store 实现：Redis（生产，SET NX + TTL / GETDEL）；进程内 dict（开发/测试）。
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from typing import Callable, Optional

from app.config.settings import settings


class ConfirmationInvalid(Exception):
    """token 无效/过期/已使用，或与 refund_id 不匹配。"""


# ============================================================
# ConfirmationStore：一次性确认令牌 + 幂等结果账本
# ============================================================
class InProcessConfirmationStore:
    """进程内实现：单进程互斥足够（测试/单机）。无跨进程保证——生产配 Redis。"""

    def __init__(self):
        self._data: dict[str, tuple[str, float]] = {}
        self._results: dict[str, tuple[dict, float]] = {}
        self._lock = threading.Lock()

    def put(self, token: str, payload: str, ttl_seconds: int) -> None:
        with self._lock:
            self._data[token] = (payload, time.time() + ttl_seconds)

    def take(self, token: str) -> Optional[str]:
        """取走并失效（单次使用）；不存在/过期返回 None。"""
        with self._lock:
            entry = self._data.pop(token, None)
            if entry is None:
                return None
            payload, expires_ts = entry
            if expires_ts < time.time():
                return None
            return payload

    def put_result(self, refund_id: str, result: dict) -> None:
        ttl = settings.refund_idempotency_ttl_seconds
        with self._lock:
            self._results[refund_id] = (dict(result), time.time() + ttl)

    def get_result(self, refund_id: str) -> Optional[dict]:
        with self._lock:
            entry = self._results.get(refund_id)
            if entry is None:
                return None
            result, expires_ts = entry
            if expires_ts < time.time():
                self._results.pop(refund_id, None)
                return None
            return dict(result)


class RedisConfirmationStore:
    """Redis 实现：token SET NX + EX / GETDEL（严格单次）；结果 SET NX + EX。"""

    TOKEN_PREFIX = "refund_confirm:"
    RESULT_PREFIX = "refund_result:"

    def __init__(self, redis):
        self._redis = redis

    @staticmethod
    def _decode(value) -> Optional[str]:
        if value is None:
            return None
        return value.decode("utf-8") if isinstance(value, bytes) else value

    def put(self, token: str, payload: str, ttl_seconds: int) -> None:
        self._redis.set(self.TOKEN_PREFIX + token, payload, nx=True, ex=ttl_seconds)

    def take(self, token: str) -> Optional[str]:
        try:
            return self._decode(self._redis.getdel(self.TOKEN_PREFIX + token))
        except Exception:  # noqa: BLE001 —— 老版本 Redis 用 get+del 降级
            value = self._redis.get(self.TOKEN_PREFIX + token)
            if value is not None:
                self._redis.delete(self.TOKEN_PREFIX + token)
            return self._decode(value)

    def put_result(self, refund_id: str, result: dict) -> None:
        ttl = settings.refund_idempotency_ttl_seconds
        self._redis.set(
            self.RESULT_PREFIX + refund_id,
            json.dumps(result, ensure_ascii=False),
            nx=True, ex=ttl,
        )

    def get_result(self, refund_id: str) -> Optional[dict]:
        raw = self._decode(self._redis.get(self.RESULT_PREFIX + refund_id))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None


# ============================================================
# 两段式入口
# ============================================================
def _payload_of(order_id: str, reason: str) -> str:
    return f"{order_id}|{reason}"


def idempotency_key_of(order_id: str, reason: str) -> str:
    """兼容旧调用保留；幂等锚点已改为 request() 签发的 refund_id。

    该哈希把「同单同因的正当二次退款」误判为重放，不应再用于新路径。
    """
    return hashlib.sha256(f"{order_id}|{reason}".encode("utf-8")).hexdigest()[:24]


class RefundConfirmation:
    """退款确认控制器：发起/确认两段式 + 幂等。

    控制器本身无状态——token 与幂等账本都在共享 store 里，跨实例/跨请求一致
    （历史缺陷：账本在实例字段上，apply_refund 每次新建控制器导致永远为空）。
    """

    def __init__(self, store, ttl_seconds: Optional[int] = None):
        self._store = store
        self._ttl = ttl_seconds or settings.refund_confirm_ttl_seconds

    def request(self, order_id: str, reason: str) -> dict:
        """第一段：签发 refund_id（幂等锚点）+ 一次性确认 token。"""
        token = uuid.uuid4().hex
        refund_id = uuid.uuid4().hex
        self._store.put(
            token,
            json.dumps(
                {"order_id": order_id, "reason": reason, "refund_id": refund_id},
                ensure_ascii=False,
            ),
            self._ttl,
        )
        return {
            "success": True,
            "status": "pending_confirmation",
            "refund_id": refund_id,
            "confirmation_token": token,
            "expires_in_seconds": self._ttl,
            "idempotency_key": refund_id,
        }

    def confirm(
        self,
        confirmation_token: str,
        executor: Callable[[str, str, str], dict],
        order_id: str = "",
        reason: str = "",
        refund_id: str = "",
    ) -> dict:
        """第二段：校验 token 并执行退款；refund_id 幂等去重。

        executor(order_id, reason, idempotency_key) -> dict：真实退款执行
        （真实后端应按 idempotency_key 做下游去重；mock 无此语义，见模块注释）。

        状态机：
        - 账本命中 → 返回首次结果 + replayed=true（网络重试安全）；
        - token take 单次有效 → 并发双确认只有一个成功；
        - token 与 refund_id/订单不匹配 → ConfirmationInvalid。
        """
        key = refund_id or (order_id and idempotency_key_of(order_id, reason)) or ""
        if not key:
            raise ConfirmationInvalid("缺少幂等锚点（refund_id）")

        prior = self._store.get_result(key)
        if prior is not None:
            return {**prior, "replayed": True, "idempotency_key": key}

        raw = self._store.take(confirmation_token)
        if raw is None:
            raise ConfirmationInvalid("确认令牌无效/过期/已使用")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ConfirmationInvalid("确认令牌载荷损坏") from e
        if payload.get("refund_id") != key:
            raise ConfirmationInvalid("确认令牌与退款请求不匹配")
        if order_id and payload.get("order_id") != order_id:
            raise ConfirmationInvalid("确认令牌与订单不匹配")

        oid = payload.get("order_id") or order_id
        rsn = payload.get("reason") or reason
        result = executor(oid, rsn, key)
        stored = {**dict(result), "idempotency_key": key, "confirmed": True}
        self._store.put_result(key, stored)
        return stored
