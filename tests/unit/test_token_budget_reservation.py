"""修复计划·四：Token 预算原子预留 / 结算 / 释放与 LLM 包装层计费测试。"""

from __future__ import annotations

import pytest

from app.agent.turn_budget import LLMBudgetExhausted
from app.security.ratelimit import (
    BudgetStoreUnavailable,
    UserLimiter,
    bind_budget_user,
)


class _FailingRedis:
    def get(self, *a, **k):
        raise ConnectionError("down")

    def set(self, *a, **k):
        raise ConnectionError("down")

    def incrby(self, *a, **k):
        raise ConnectionError("down")

    def expire(self, *a, **k):
        raise ConnectionError("down")

    def eval(self, *a, **k):
        raise ConnectionError("down")

    def hget(self, *a, **k):
        raise ConnectionError("down")

    def hdel(self, *a, **k):
        raise ConnectionError("down")

    def hset(self, *a, **k):
        raise ConnectionError("down")


def _redis():
    import fakeredis

    return fakeredis.FakeStrictRedis()


# ------------------------------------------------------------
# 原子预留语义（Redis 与进程内同语义）
# ------------------------------------------------------------
@pytest.mark.parametrize("use_redis", [True, False])
def test_reserve_respects_daily_budget(use_redis):
    limiter = UserLimiter(
        _redis() if use_redis else None, daily_token_budget=1000,
    )
    assert limiter.reserve_tokens("u1", 600, "r1") is True
    # 600 + 600 >= 1000 → 拒绝（在途预留计入）
    assert limiter.reserve_tokens("u1", 600, "r2") is False
    limiter.settle_tokens("r1", 600)
    assert limiter._used_today("u1") == 600
    assert limiter.reserve_tokens("u1", 400, "r3") is False  # 600+400 >= 1000
    assert limiter.reserve_tokens("u1", 399, "r4") is True  # 600+399 < 1000


@pytest.mark.parametrize("use_redis", [True, False])
def test_settle_over_estimate_bills_full(use_redis):
    limiter = UserLimiter(
        _redis() if use_redis else None, daily_token_budget=10000,
    )
    assert limiter.reserve_tokens("u1", 100, "r1") is True
    limiter.settle_tokens("r1", 500)  # 实际远超预估
    assert limiter._used_today("u1") == 500  # 仍完整入账


@pytest.mark.parametrize("use_redis", [True, False])
def test_release_frees_reservation(use_redis):
    limiter = UserLimiter(
        _redis() if use_redis else None, daily_token_budget=1000,
    )
    assert limiter.reserve_tokens("u1", 600, "r1") is True
    assert limiter.reserve_tokens("u1", 600, "r2") is False
    limiter.release_reservation("r2")  # 未登记也幂等
    limiter.release_reservation("r1")
    assert limiter.reserve_tokens("u1", 600, "r3") is True  # 释放后额度回来
    assert limiter._used_today("u1") == 0  # 释放不入账


def test_concurrent_reservations_only_fit_budget():
    """多会话竞争最后一段预算：只能有符合额度的预留成功。"""
    limiter = UserLimiter(_redis(), daily_token_budget=1000)
    outcomes = [
        limiter.reserve_tokens("u1", 400, f"r{i}") for i in range(5)
    ]
    assert sum(1 for o in outcomes if o) == 2  # 400+400<1000，第三个 1200 拒绝


# ------------------------------------------------------------
# redis_required：预算存储故障 fail-closed
# ------------------------------------------------------------
def test_budget_store_unavailable_when_required():
    limiter = UserLimiter(_FailingRedis(), daily_token_budget=1000, redis_required=True)
    with pytest.raises(BudgetStoreUnavailable):
        limiter.allow_budget("u1")
    with pytest.raises(BudgetStoreUnavailable):
        limiter.reserve_tokens("u1", 100, "r1")
    with pytest.raises(BudgetStoreUnavailable):
        limiter.consume_tokens("u1", 10)


def test_budget_degrades_when_not_required():
    limiter = UserLimiter(_FailingRedis(), daily_token_budget=1000)  # 开发模式
    assert limiter.allow_budget("u1") is True
    assert limiter.reserve_tokens("u1", 100, "r1") is True


# ------------------------------------------------------------
# LLM 包装层：立即计费 / 失败释放 / 超额拒绝
# ------------------------------------------------------------
class _Usage:
    def __init__(self, total):
        self.total_tokens = total


class _Resp:
    def __init__(self, total):
        self.usage = _Usage(total)


def _wrapper(limiter):
    from app.llm.client import ResilientLLM

    return ResilientLLM(object(), "m", limiter=limiter)


def test_llm_wrapper_bills_actual_usage_immediately():
    limiter = UserLimiter(None, daily_token_budget=10000)
    bind_budget_user("u1")
    rl = _wrapper(limiter)
    kwargs = {"messages": [{"role": "user", "content": "你好"}], "max_tokens": 10}
    out = rl._invoke(lambda: _Resp(123), kwargs)
    assert out.usage.total_tokens == 123
    assert limiter._used_today("u1") == 123  # 成功即以真实 usage 入账
    assert limiter._resv_meta == {}  # 预留已结算清除


def test_llm_wrapper_releases_reservation_on_failure():
    limiter = UserLimiter(None, daily_token_budget=10000)
    bind_budget_user("u1")
    rl = _wrapper(limiter)
    kwargs = {"messages": [{"role": "user", "content": "你好"}], "max_tokens": 10}

    def _boom():
        raise RuntimeError("llm down")

    with pytest.raises(RuntimeError):
        rl._invoke(_boom, kwargs)
    assert limiter._used_today("u1") == 0  # 失败不入账
    assert limiter._resv_meta == {}  # 预留已释放


def test_llm_wrapper_rejects_when_budget_exhausted():
    limiter = UserLimiter(None, daily_token_budget=10)
    bind_budget_user("u1")
    rl = _wrapper(limiter)
    kwargs = {"messages": [{"role": "user", "content": "x" * 4000}], "max_tokens": 100}
    with pytest.raises(LLMBudgetExhausted):
        rl._invoke(lambda: _Resp(1), kwargs)
