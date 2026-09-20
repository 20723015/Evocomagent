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
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

from app.config.settings import settings

RPS_KEY_PREFIX = "ratelimit:rps:"
BUDGET_KEY_PREFIX = "ratelimit:budget:"
BUDGET_RESV_PREFIX = "ratelimit:resv:"
RESERVATION_TTL_SECONDS = 300


class BudgetStoreUnavailable(RuntimeError):
    """预算存储（Redis）不可用：redis_required 下 fail-closed（HTTP 503），
    不得降级为本地计数（多 Pod 会各算各的，成本护栏失效）。"""


# 请求级用户标签（LLM 包装层据此预留/结算；随 anyio 线程池传播）。
# 哨兵值区分「从未绑定」（离线 CLI/演进任务，不参与请求预算）与「显式空用户」
# （后台任务缺归属，必须拒绝匿名绕过预算）。
UNSET_BUDGET_USER = "__budget_user_unset__"
_BUDGET_USER: ContextVar[str] = ContextVar("budget_user_id", default=UNSET_BUDGET_USER)


def bind_budget_user(user_id: str) -> None:
    _BUDGET_USER.set(user_id)


def current_budget_user() -> str:
    return _BUDGET_USER.get()


@contextmanager
def budget_user_scope(user_id: str):
    """临时绑定预算用户并在退出时恢复原 ContextVar（后台 worker 逐任务计费）。

    修复计划·二轮 7：Memory Worker 每个任务用任务自身 user_id 建立作用域，
    空 user_id 的异常任务不得以匿名方式绕过预算。
    """
    token = _BUDGET_USER.set(user_id or "")
    try:
        yield
    finally:
        _BUDGET_USER.reset(token)


# 原子预留： 已消费 + 在途预留 + 本次估算 >= 日预算 → 拒绝（返回 0）；
# 同时清扫过期预留（hash 值格式 "estimate:ts"）。
_RESERVE_LUA = """
local budget = tonumber(redis.call('GET', KEYS[1]) or '0')
local now = tonumber(ARGV[4])
local ttl = tonumber(ARGV[3])
local items = redis.call('HGETALL', KEYS[2])
local reserved = 0
for i=1,#items,2 do
  local f = items[i]
  local v = items[i+1]
  local est = tonumber(string.match(v, '^(%d+):') or '0') or 0
  local ts = tonumber(string.match(v, ':(%d+)$') or '0') or 0
  if ts > 0 and now - ts > ttl then
    redis.call('HDEL', KEYS[2], f)
  else
    reserved = reserved + est
  end
end
local limit = tonumber(ARGV[5])
local estimate = tonumber(ARGV[2])
if limit > 0 and budget + reserved + estimate >= limit then
  return 0
end
redis.call('HSET', KEYS[2], ARGV[1], ARGV[2] .. ':' .. ARGV[4])
redis.call('EXPIRE', KEYS[2], ttl + 60)
return 1
"""

# 结算：移除预留 + 实际用量全额入账（异常超预留也完整计费）；返回超预估量。
_SETTLE_LUA = """
local v = redis.call('HGET', KEYS[2], ARGV[1])
local est = 0
if v then
  est = tonumber(string.match(v, '^(%d+):') or '0') or 0
end
redis.call('HDEL', KEYS[2], ARGV[1])
local actual = tonumber(ARGV[2])
if actual > 0 then
  redis.call('INCRBY', KEYS[1], actual)
  redis.call('EXPIRE', KEYS[1], ARGV[3])
end
if est > 0 and actual > est then
  return actual - est
end
return 0
"""


class _InProcessWindow:
    """进程内滑动窗口（无 Redis 降级；单机互斥即可）。"""

    def __init__(self):
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window: float) -> bool:
        now = time.time()
        with self._lock:
            cutoff = now - window
            hits = self._hits.get(key)
            if hits is not None:
                hits[:] = [t for t in hits if t >= cutoff]
                if not hits:
                    # 全过期：删除键，防进程内结构无界增长（低危修复 C8）
                    del self._hits[key]
            hits = self._hits.setdefault(key, [])
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
                 daily_token_budget: Optional[int] = None,
                 redis_required: Optional[bool] = None):
        self._redis = redis
        self._window = _InProcessWindow()
        self._budgets: dict[str, int] = {}
        self._budget_lock = threading.Lock()
        self._budget_sweep_day = ""  # 上次跨日清扫的日期（低危修复 C8）
        # 预留登记：reservation_id -> (user_id, day, estimate)
        self._resv_meta: dict[str, tuple[str, str, int]] = {}
        self._required = (
            settings.redis_required if redis_required is None else redis_required
        )
        self._max_rps = settings.rate_limit_rps if max_rps is None else max_rps
        self._daily = (
            settings.rate_limit_daily_tokens
            if daily_token_budget is None else daily_token_budget
        )
        self._window_seconds = settings.rate_limit_window_seconds

    @property
    def daily_budget(self) -> int:
        return self._daily

    def bind_user(self, user_id: str) -> None:
        """绑定请求级用户标签：LLM 包装层的预留/结算据此定位预算键。"""
        bind_budget_user(user_id)

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
        """请求前置检查：今日已消耗 + 本次预估是否超预算。

        修复计划·四：redis_required 下预算存储故障 → BudgetStoreUnavailable
        （HTTP 503），不降级为本地计数。
        """
        if self._daily <= 0:
            return True
        if self._redis is not None:
            try:
                value = self._redis.get(f"{BUDGET_KEY_PREFIX}{user_id}:{self._day()}")
                used = int(value) if value else 0
                return (used + estimated_tokens) < self._daily
            except Exception as e:  # noqa: BLE001
                if self._required:
                    raise BudgetStoreUnavailable(
                        f"预算存储不可用：{type(e).__name__}"
                    ) from e
        with self._budget_lock:
            day = self._day()
            self._sweep_stale_budgets_locked(day)
            used = self._budgets.get(f"{user_id}:{day}", 0)
            return (used + estimated_tokens) < self._daily

    def consume_tokens(self, user_id: str, tokens: int) -> None:
        """实际用量入账（非预留路径；LLM 包装层已改为预留/结算）。"""
        if tokens <= 0 or self._daily <= 0:
            return
        if self._redis is not None:
            try:
                key = f"{BUDGET_KEY_PREFIX}{user_id}:{self._day()}"
                self._redis.incrby(key, tokens)
                self._redis.expire(key, 2 * 86400)
                return
            except Exception as e:  # noqa: BLE001 —— Redis 故障降级进程内计数
                if self._required:
                    raise BudgetStoreUnavailable(
                        f"预算存储不可用：{type(e).__name__}"
                    ) from e
        with self._budget_lock:
            day = self._day()
            self._sweep_stale_budgets_locked(day)
            self._budgets[f"{user_id}:{day}"] = \
                self._budgets.get(f"{user_id}:{day}", 0) + tokens

    # ---------- 原子预留 / 结算（修复计划·四） ----------
    def reserve_tokens(self, user_id: str, estimate: int, reservation_id: str,
                       ttl: int = RESERVATION_TTL_SECONDS) -> bool:
        """预留本次 LLM 尝试的估算用量；False = 超出日预算。

        Redis Lua 原子完成「已消费 + 在途预留 + 本次估算 ≥ 日预算」判定并登记，
        过期预留自动清扫；redis_required 下存储故障抛 BudgetStoreUnavailable。
        """
        if self._daily <= 0:
            return True
        estimate = max(int(estimate), 0)
        day = self._day()
        if self._redis is not None:
            try:
                key = f"{BUDGET_KEY_PREFIX}{user_id}:{day}"
                rkey = f"{BUDGET_RESV_PREFIX}{user_id}:{day}"
                res = self._redis.eval(
                    _RESERVE_LUA, 2, key, rkey,
                    reservation_id, estimate, int(ttl), int(time.time()),
                    int(self._daily),
                )
                if res:
                    self._resv_meta[reservation_id] = (user_id, day, estimate)
                return bool(res)
            except Exception as e:  # noqa: BLE001
                if self._required:
                    raise BudgetStoreUnavailable(
                        f"预算存储不可用：{type(e).__name__}"
                    ) from e
        with self._budget_lock:
            self._sweep_stale_budgets_locked(day)
            self._sweep_reservations_locked(ttl)
            used = self._budgets.get(f"{user_id}:{day}", 0)
            reserved = sum(
                est for (u, d, est) in self._resv_meta.values()
                if u == user_id and d == day
            )
            if used + reserved + estimate >= self._daily:
                return False
            self._resv_meta[reservation_id] = (user_id, day, estimate)
            return True

    def settle_tokens(self, reservation_id: str, actual_tokens: int) -> None:
        """按真实 usage 结算预留：移除预留 + 全额入账（超预估也完整计费）。"""
        meta = self._resv_meta.pop(reservation_id, None)
        actual = max(int(actual_tokens or 0), 0)
        if meta is None:
            return  # 未登记（如 unlimited 模式）或重复结算：幂等
        user_id, day, estimate = meta
        overrun = 0
        if self._redis is not None:
            try:
                key = f"{BUDGET_KEY_PREFIX}{user_id}:{day}"
                rkey = f"{BUDGET_RESV_PREFIX}{user_id}:{day}"
                overrun = int(self._redis.eval(
                    _SETTLE_LUA, 2, key, rkey, reservation_id, actual, 2 * 86400,
                ) or 0)
            except Exception as e:  # noqa: BLE001
                if self._required:
                    raise BudgetStoreUnavailable(
                        f"预算结算失败：{type(e).__name__}"
                    ) from e
                self._settle_in_process(user_id, day, actual)
        else:
            self._settle_in_process(user_id, day, actual)
        if actual > estimate:
            overrun = overrun or (actual - estimate)
        if overrun > 0:
            self._record_overrun(overrun)

    def release_reservation(self, reservation_id: str) -> None:
        """失败调用释放预留（不入账）。"""
        meta = self._resv_meta.pop(reservation_id, None)
        if meta is None or self._redis is None:
            return
        user_id, day, _ = meta
        try:
            self._redis.hdel(f"{BUDGET_RESV_PREFIX}{user_id}:{day}", reservation_id)
        except Exception:  # noqa: BLE001 —— 释放失败靠 TTL 清扫
            pass

    def _settle_in_process(self, user_id: str, day: str, actual: int) -> None:
        if actual <= 0:
            return
        with self._budget_lock:
            key = f"{user_id}:{day}"
            self._budgets[key] = self._budgets.get(key, 0) + actual

    def _sweep_reservations_locked(self, ttl: int) -> None:
        """进程内过期预留清扫（须持 _budget_lock）：跨日即失效。"""
        day = self._day()
        for rid in [r for r, (_, d, _) in self._resv_meta.items() if d != day]:
            self._resv_meta.pop(rid, None)

    @staticmethod
    def _record_overrun(overrun: int) -> None:
        try:
            from app.observability.metrics import record_budget_estimator_overrun

            record_budget_estimator_overrun(overrun)
        except Exception:  # noqa: BLE001
            pass

    def _used_today(self, user_id: str) -> int:
        if self._redis is not None:
            try:
                key = f"{BUDGET_KEY_PREFIX}{user_id}:{self._day()}"
                value = self._redis.get(key)
                return int(value) if value else 0
            except Exception:  # noqa: BLE001 —— Redis 故障降级进程内计数
                pass
        with self._budget_lock:
            self._sweep_stale_budgets_locked(self._day())
            return self._budgets.get(f"{user_id}:{self._day()}", 0)

    def _sweep_stale_budgets_locked(self, day: str) -> None:
        """跨日清扫非当日预算键（低危修复 C8：进程内 dict 无界增长；
        O(n) 每日至多一次，须持有 _budget_lock 调用）。"""
        if day != self._budget_sweep_day:
            self._budgets = {
                k: v for k, v in self._budgets.items() if k.endswith(f":{day}")
            }
            self._budget_sweep_day = day

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
