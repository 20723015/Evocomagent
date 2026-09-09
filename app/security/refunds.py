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
import hmac
import json
import threading
import time
import uuid
from typing import Callable, Optional

from app.config.settings import settings


class ConfirmationInvalid(Exception):
    """token 无效/过期/已使用，或与 refund_id 不匹配。"""


_RESULT_BINDING_KEY = "_refund_confirmation_binding"


def _token_digest(token: str) -> str:
    """只保存确认 token 的指纹，结果账本中永不保存原始凭证。"""
    return hashlib.sha256(str(token).encode("utf-8")).hexdigest()


def _public_result(result: dict) -> dict:
    """移除结果账本内部绑定信息，避免凭证关联元数据泄露给调用方。"""
    return {
        key: value for key, value in dict(result).items()
        if key != _RESULT_BINDING_KEY
    }


# ============================================================
# ConfirmationStore：一次性确认令牌 + 幂等结果账本
# ============================================================
class InProcessConfirmationStore:
    """进程内实现：单进程互斥足够（测试/单机）。无跨进程保证——生产配 Redis。"""

    def __init__(self):
        self._data: dict[str, tuple[str, float]] = {}
        self._results: dict[str, tuple[dict, float]] = {}
        self._session_pending: dict[tuple[str, str], dict[str, tuple[dict, float]]] = {}
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

    # ---- Review 修复：会话级待确认注册表（token 只存这里，反向解析）----
    def put_session_refund(self, user_id: str, session_id: str,
                           payload: dict, ttl_seconds: int) -> None:
        """按 user+session 注册待确认退款（payload 含 token/order/reason/refund_id）。"""
        key = (user_id, session_id)
        with self._lock:
            self._session_pending.setdefault(key, {})[payload["refund_id"]] = (
                dict(payload), time.time() + ttl_seconds,
            )

    def get_session_refunds(self, user_id: str, session_id: str) -> list[dict]:
        with self._lock:
            entries = self._session_pending.get((user_id, session_id), {})
            now = time.time()
            return [
                dict(payload)
                for payload, expires_ts in entries.values()
                if expires_ts >= now
            ]

    def take_session_refund(self, user_id: str, session_id: str,
                            refund_id: str) -> Optional[dict]:
        with self._lock:
            entries = self._session_pending.get((user_id, session_id), {})
            entry = entries.pop(refund_id, None)
        if entry is None:
            return None
        payload, expires_ts = entry
        if expires_ts < time.time():
            return None
        return dict(payload)

    def drop_token(self, token: str) -> None:
        """取消路径：让未消费的确认令牌立即失效。"""
        with self._lock:
            self._data.pop(token, None)


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
        except Exception:  # noqa: BLE001 —— 老客户端用 Lua 保持原子 GET+DEL
            # 不能退化成分离的 get()/delete()：两个确认请求会同时读到 token，
            # 破坏一次性语义。Redis 2.6+ 均支持 EVAL；再失败则安全拒绝。
            try:
                value = self._redis.eval(
                    "local v=redis.call('GET',KEYS[1]);"
                    "if v then redis.call('DEL',KEYS[1]) end;return v",
                    1,
                    self.TOKEN_PREFIX + token,
                )
                return self._decode(value)
            except Exception:  # noqa: BLE001
                return None

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

    # ---- Review 修复：会话级待确认注册表（token 只存这里，反向解析）----
    PENDING_PREFIX = "refund_pending:"
    PENDING_INDEX_SUFFIX = ":idx"

    @staticmethod
    def _pending_key(user_id: str, session_id: str, refund_id: str) -> str:
        return f"{RedisConfirmationStore.PENDING_PREFIX}{user_id}:{session_id}:{refund_id}"

    @staticmethod
    def _pending_index_key(user_id: str, session_id: str) -> str:
        return (f"{RedisConfirmationStore.PENDING_PREFIX}{user_id}:{session_id}"
                f"{RedisConfirmationStore.PENDING_INDEX_SUFFIX}")

    def put_session_refund(self, user_id: str, session_id: str,
                           payload: dict, ttl_seconds: int) -> None:
        """按 user+session 注册待确认退款（payload 含 token/order/reason/refund_id）。"""
        refund_id = str(payload.get("refund_id", ""))
        key = self._pending_key(user_id, session_id, refund_id)
        index = self._pending_index_key(user_id, session_id)
        try:
            self._redis.set(
                key, json.dumps(payload, ensure_ascii=False),
                ex=max(int(ttl_seconds), 1),
            )
            self._redis.sadd(index, refund_id)
            self._redis.expire(index, max(int(ttl_seconds), 1))
        except Exception:  # noqa: BLE001 —— 注册失败下一轮可重试（签发段未完成语义）
            return

    def get_session_refunds(self, user_id: str, session_id: str) -> list[dict]:
        index = self._pending_index_key(user_id, session_id)
        out: list[dict] = []
        try:
            refund_ids = self._redis.smembers(index)
        except Exception:  # noqa: BLE001
            return []
        for raw_id in refund_ids:
            refund_id = self._decode(raw_id)
            if not refund_id:
                continue
            raw = self._redis.get(self._pending_key(user_id, session_id, refund_id))
            if raw is None:
                continue  # 过期即视为不存在
            try:
                out.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                continue
        return out

    def take_session_refund(self, user_id: str, session_id: str,
                            refund_id: str) -> Optional[dict]:
        key = self._pending_key(user_id, session_id, refund_id)
        try:
            raw = self._redis.get(key)
            if raw is None:
                return None
            self._redis.delete(key)
            self._redis.srem(
                self._pending_index_key(user_id, session_id), refund_id,
            )
            return json.loads(raw)
        except Exception:  # noqa: BLE001
            return None

    def drop_token(self, token: str) -> None:
        """取消路径：让未消费的确认令牌立即失效。"""
        try:
            self._redis.delete(self.TOKEN_PREFIX + token)
        except Exception:  # noqa: BLE001
            return


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
        # Review 修复：区分 None（跟随配置）与 0（立即过期，测试/急停用）
        self._ttl = (
            settings.refund_confirm_ttl_seconds if ttl_seconds is None else ttl_seconds
        )

    def request(self, order_id: str, reason: str,
                user_id: str = "", session_id: str = "") -> dict:
        """第一段：签发 refund_id（幂等锚点）+ 一次性确认 token。

        Review 修复：token 载荷绑定用户/会话/订单/原因/refund_id；同时把
        待确认条目注册进会话级注册表（token 只存确认存储，不进消息/上下文）。
        """
        token = uuid.uuid4().hex
        refund_id = uuid.uuid4().hex
        payload = {
            "order_id": order_id,
            "reason": reason,
            "refund_id": refund_id,
            "user_id": user_id,
            "session_id": session_id,
        }
        self._store.put(
            token,
            json.dumps(payload, ensure_ascii=False),
            self._ttl,
        )
        if user_id:
            store_put = getattr(self._store, "put_session_refund", None)
            if store_put is not None:
                store_put(
                    user_id, session_id,
                    {**payload, "token": token}, self._ttl,
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
        user_id: str = "",
        session_id: str = "",
    ) -> dict:
        """第二段：校验 token 并执行退款；refund_id 幂等去重。

        executor(order_id, reason, idempotency_key) -> dict：真实退款执行
        （真实后端应按 idempotency_key 做下游去重；mock 无此语义，见模块注释）。

        状态机：
        - 账本命中 → 返回首次结果 + replayed=true（网络重试安全）；
        - token take 单次有效 → 并发双确认只有一个成功；
        - token 与 refund_id/订单/用户/会话不匹配 → ConfirmationInvalid
          （Review 修复：载荷绑定全量校验）。
        """
        key = refund_id or (order_id and idempotency_key_of(order_id, reason)) or ""
        if not key:
            raise ConfirmationInvalid("缺少幂等锚点（refund_id）")

        token = str(confirmation_token or "")
        prior = self._store.get_result(key)
        if prior is not None:
            # 幂等 replay 不能只凭 refund_id 命中结果账本。否则任意用户只要
            # 猜到/拿到一个 refund_id，带 bogus token 即可读取并伪造一次成功
            # 结果。结果账本保存 token 指纹 + user/session 绑定，重放先完成
            # 全量校验，旧的无绑定结果宁可拒绝（fail-closed）。
            self._validate_replay_binding(
                prior, token, key, order_id, reason, user_id, session_id,
            )
            return {
                **_public_result(prior), "replayed": True,
                "idempotency_key": key,
            }

        if not token:
            raise ConfirmationInvalid("确认令牌无效/过期/已使用")
        raw = self._store.take(token)
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
        if reason and payload.get("reason") != reason:
            raise ConfirmationInvalid("确认令牌与退款原因不匹配")
        # Review 修复：用户/会话绑定必须严格相等。不能因为调用方漏传
        # user/session 就跳过载荷中的绑定检查。
        if str(payload.get("user_id") or "") != str(user_id or ""):
            raise ConfirmationInvalid("确认令牌与用户不匹配")
        if str(payload.get("session_id") or "") != str(session_id or ""):
            raise ConfirmationInvalid("确认令牌与会话不匹配")

        oid = payload.get("order_id") or order_id
        rsn = payload.get("reason") or reason
        result = executor(oid, rsn, key)
        stored = {
            **dict(result),
            "idempotency_key": key,
            "confirmed": True,
            # 只进结果账本内部字段；_public_result() 确保不返回给工具/模型。
            _RESULT_BINDING_KEY: {
                "refund_id": key,
                "order_id": str(payload.get("order_id") or oid),
                "reason": str(payload.get("reason") or rsn),
                "user_id": str(payload.get("user_id") or ""),
                "session_id": str(payload.get("session_id") or ""),
                "token_hash": _token_digest(token),
            },
        }
        self._store.put_result(key, stored)
        # 成功提交 → 清掉会话级待确认条目
        take_pending = getattr(self._store, "take_session_refund", None)
        bound_user = payload.get("user_id") or user_id
        bound_session = payload.get("session_id") or session_id
        if take_pending is not None and bound_user:
            take_pending(bound_user, bound_session, key)
        return _public_result(stored)

    @staticmethod
    def _validate_replay_binding(
        prior: dict,
        token: str,
        key: str,
        order_id: str,
        reason: str,
        user_id: str,
        session_id: str,
    ) -> None:
        """校验结果 replay 的完整归属，不信任 refund_id 单字段。"""
        binding = prior.get(_RESULT_BINDING_KEY)
        if not isinstance(binding, dict):
            raise ConfirmationInvalid("幂等结果缺少确认绑定，拒绝重放")
        if binding.get("refund_id") != key:
            raise ConfirmationInvalid("幂等结果与退款请求不匹配")
        expected_hash = str(binding.get("token_hash") or "")
        if not token or not expected_hash or not hmac.compare_digest(
            expected_hash, _token_digest(token),
        ):
            raise ConfirmationInvalid("确认令牌与幂等结果不匹配")
        if order_id and binding.get("order_id") != order_id:
            raise ConfirmationInvalid("幂等结果与订单不匹配")
        if reason and binding.get("reason") != reason:
            raise ConfirmationInvalid("幂等结果与退款原因不匹配")
        if str(binding.get("user_id") or "") != str(user_id or ""):
            raise ConfirmationInvalid("幂等结果与用户不匹配")
        if str(binding.get("session_id") or "") != str(session_id or ""):
            raise ConfirmationInvalid("幂等结果与会话不匹配")
