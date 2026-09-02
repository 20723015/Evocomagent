"""用户级限流与配额（阶段三 3.7）+ LLM 用量归集。

- UserLimiter：per-user RPS 滑动窗口（Redis zset；无 Redis 降级进程内）+ 
  每日 token 预算（Redis INCRBY 按日计数；无 Redis 进程内）。
  与阶段四 4.4 pod 级信号量互补：那是护系统，这是护成本、防滥用。
- UsageTracker：按 request 归集 LLM token 用量（评审二轮 B2：request 粒度）
  → 消费预算 / 成本看板（4.3 的 llm_tokens_total 也走这里）。
"""

from __future__ import annotations

import threading
import time
import uuid
from contextvars import ContextVar
from typing import Optional

from app.config.settings import settings

RPS_KEY_PREFIX = "ratelimit:rps:"
BUDGET_KEY_PREFIX = "ratelimit:budget:"


class _InProcessWindow:
    """进程内滑动窗口（无 Redis 降级；单机互斥即可）。"""

    def __init__(self):
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window: float) -> bool:
        now = time.time()
        with self._lock:
            hits = self._hits.setdefault(key, [])
            cutoff = now - window
            hits[:] = [t for t in hits if t >= cutoff]
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True

    def count(self, key: str) -> int:
        with self._lock:
            return len(self._hits.get(key, []))


class UserLimiter:
    """per-user 限流与配额。redis 为 None 时进程内降级（弱一致性）。"""

    def __init__(self, redis=None, max_rps: Optional[int] = None,
                 daily_token_budget: Optional[int] = None):
        self._redis = redis
        self._window = _InProcessWindow()
        self._budgets: dict[str, int] = {}
        self._budget_lock = threading.Lock()
        self._max_rps = settings.rate_limit_rps if max_rps is None else max_rps
        self._daily = (
            settings.rate_limit_daily_tokens
            if daily_token_budget is None else daily_token_budget
        )
        self._window_seconds = settings.rate_limit_window_seconds

    # ---------- RPS 滑动窗口 ----------
    def allow_rps(self, user_id: str) -> bool:
        if self._max_rps <= 0:
            return True
        if self._redis is not None:
            try:
                key = f"{RPS_KEY_PREFIX}{user_id}"
                now = time.time()
                # 成员唯一化：同微秒多次请求的 time.time() 会撞值（zset 去重漏计）
                member = f"{now:.6f}:{uuid.uuid4().hex[:8]}"
                pipe = self._redis.pipeline()
                pipe.zremrangebyscore(key, 0, now - self._window_seconds)
                pipe.zadd(key, {member: now})
                pipe.zcard(key)
                pipe.expire(key, self._window_seconds + 5)
                count = pipe.execute()[-2]
                return int(count) <= self._max_rps
            except Exception:  # noqa: BLE001 —— Redis 故障降级进程内滑动窗口
                pass
        return self._window.allow(user_id, self._max_rps, self._window_seconds)

    # ---------- 每日 token 预算 ----------
    def allow_budget(self, user_id: str, estimated_tokens: int = 0) -> bool:
        """请求前置检查：今日已消耗 + 本次预估是否超预算。"""
        if self._daily <= 0:
            return True
        used = self._used_today(user_id)
        return (used + estimated_tokens) < self._daily

    def consume_tokens(self, user_id: str, tokens: int) -> None:
        """实际用量入账（请求结束后调用）。"""
        if tokens <= 0 or self._daily <= 0:
            return
        if self._redis is not None:
            try:
                key = f"{BUDGET_KEY_PREFIX}{user_id}:{self._day()}"
                self._redis.incrby(key, tokens)
                self._redis.expire(key, 2 * 86400)
                return
            except Exception:  # noqa: BLE001 —— Redis 故障降级进程内计数
                pass
        with self._budget_lock:
            self._budgets[f"{user_id}:{self._day()}"] = \
                self._budgets.get(f"{user_id}:{self._day()}", 0) + tokens

    def _used_today(self, user_id: str) -> int:
        if self._redis is not None:
            try:
                key = f"{BUDGET_KEY_PREFIX}{user_id}:{self._day()}"
                value = self._redis.get(key)
                return int(value) if value else 0
            except Exception:  # noqa: BLE001 —— Redis 故障降级进程内计数
                pass
        with self._budget_lock:
            return self._budgets.get(f"{user_id}:{self._day()}", 0)

    @staticmethod
    def _day() -> str:
        return time.strftime("%Y%m%d")


# ============================================================
# UsageTracker：LLM token 用量按请求归集（ContextVar 请求标签）
# ============================================================
_USAGE_REQUEST_ID: ContextVar[str] = ContextVar("usage_request_id", default="")

# 未认领桶上限（断连弃单的兜底清扫阈值；防慢速攻击者撑爆内存）
_PENDING_CAP = 4096
_PENDING_STALE_SECONDS = 3600


class UsageTracker:
    """包裹 LLM client 统计每请求 token 消耗（评审二轮 B2）。

    历史实现按 user_id 归集：同用户并发会话混账。现按 request_id 归集：
    路由在发起点（async 上下文）调用 begin_request 设定 ContextVar——
    anyio to_thread 会拷贝当前上下文，turn 线程与 close 线程（LTM 巩固
    的 LLM 调用）都能正确归集，修掉「close 阶段用量丢失」的老问题。
    end_request(request_id) 返回并清除该请求累计（须幂等配对调用）。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._totals: dict[str, list] = {}  # request_id -> [tokens, last_ts]

    def begin_request(self, request_id: str) -> None:
        _USAGE_REQUEST_ID.set(request_id)

    def end_request(self, request_id: str) -> int:
        """取出并清除该请求累计；重复调用返回 0（幂等）。"""
        with self._lock:
            entry = self._totals.pop(request_id, None)
        return int(entry[0]) if entry else 0

    def wrap(self, fn):
        """包裹 openai 端点方法（chat.completions.create / beta.chat.completions.parse）。"""
        def wrapper(*args, **kwargs):
            response = fn(*args, **kwargs)
            request_id = _USAGE_REQUEST_ID.get()
            usage = getattr(response, "usage", None)
            if request_id and usage is not None:
                total = getattr(usage, "total_tokens", 0) or 0
                with self._lock:
                    if len(self._totals) >= _PENDING_CAP:
                        self._sweep_stale()
                    entry = self._totals.get(request_id)
                    if entry is None:
                        self._totals[request_id] = [int(total), time.time()]
                    else:
                        entry[0] += int(total)
                        entry[1] = time.time()
            return response
        return wrapper

    def snapshot(self, request_id: str) -> int:
        with self._lock:
            entry = self._totals.get(request_id)
            return int(entry[0]) if entry else 0

    def _sweep_stale(self) -> None:
        """仅在持有 _lock 时调用：丢弃长时间未被 end 的陈旧桶。"""
        cutoff = time.time() - _PENDING_STALE_SECONDS
        stale = [k for k, v in self._totals.items() if v[1] < cutoff]
        for k in stale:
            self._totals.pop(k, None)


def install_usage_tracking(client, tracker: UsageTracker) -> UsageTracker:
    """在共享 OpenAI client 上安装用量包裹（幂等：重复安装返回已有 tracker）。"""
    existing = getattr(client, "_usage_tracker", None)
    if existing is not None:
        return existing
    completions = client.chat.completions
    completions.create = tracker.wrap(completions.create)
    beta_completions = client.beta.chat.completions
    beta_completions.parse = tracker.wrap(beta_completions.parse)
    client._usage_tracker = tracker
    return tracker
