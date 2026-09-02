"""LLM 韧性客户端（阶段四 4.4）。

在共享 OpenAI client 上原地包裹（与 install_usage_tracking 一样不改调用面）：
1. pod 级并发信号量（护系统，与 3.7 用户级配额互补）；
2. 重试：指数退避 + 抖动，仅幂等读（网络/5xx/限流）；BadRequest 不重试；
3. 模型降级链：主模型 → fallback_model（重试耗尽后尝试）；
4. 显式 timeout（客户端构造时设，取代 SDK 默认 600s）；
5. token 归集（usage）与 4.3 指标、4.2 span 属性；
6. 小任务（提取/STM/摘要）路由廉价模型：purpose_models。

要点：双 LLM 调用治理——`_extract_structured_response` 走到这里时
purpose="extract"，经 extraction_model 切到廉价模型（路径①，最小改动）。
"""

from __future__ import annotations

import random
import threading
import time
from typing import Callable, Optional

from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from app.agent.turn_budget import LLMBudgetExhausted, current_budget
from app.config.settings import settings
from app.observability import metrics, tracing

RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)

# 轮次剩余低于该阈值即视为耗尽：Windows 定时器粒度（~15.6ms）下
# 睡眠可能提前返回，剩 1e-9 秒会让重试链借门口穿过——保守夹逼（修复计划）
_LLM_MIN_REMAINING = 0.05


def infer_purpose(kind: str, kwargs: dict) -> str:
    """按调用特征猜用途（用作指标/廉价模型路由）。"""
    if kind == "parse":
        return "extract"
    if kwargs.get("max_tokens") == 10:
        return "router"
    messages = kwargs.get("messages") or []
    first = str(messages[0].get("content", "")) if messages else ""
    if "压缩" in first or "摘要" in first:
        return "summarize"
    if "历史记忆" in first or "短期记忆" in first or "长期记忆" in first:
        return "memory"
    if kwargs.get("tools"):
        return "react"
    return "react"


def _purpose_model(purpose: str, model: str) -> str:
    """小任务路由廉价模型（4.4）：提取/摘要/记忆/路由走 extraction_model。"""
    cheap = settings.extraction_model
    if not cheap or purpose not in ("extract", "summarize", "memory", "router"):
        return model
    return cheap


class ResilientLLM:
    """包装器：原地替换 client 的 create/parse 方法。"""

    def __init__(
        self,
        client,
        model: str,
        *,
        fallback_model: Optional[str] = None,
        max_retries: Optional[int] = None,
        max_concurrent: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        semaphore: Optional[threading.BoundedSemaphore] = None,
    ):
        self._client = client
        self._model = model
        self._fallback = settings.llm_fallback_model if fallback_model is None else fallback_model
        self._retries = settings.llm_max_retries if max_retries is None else max_retries
        self._semaphore = semaphore or threading.BoundedSemaphore(
            settings.llm_max_concurrent if max_concurrent is None else max_concurrent
        )
        self._timeout_seconds = (
            settings.llm_timeout_seconds if timeout_seconds is None else timeout_seconds
        )

    # ---------- 安装 ----------
    def install(self) -> None:
        completions = self._client.chat.completions
        completions.create = self._wrap(
            completions.create, kind="chat",
        )
        beta_completions = self._client.beta.chat.completions
        beta_completions.parse = self._wrap(
            beta_completions.parse, kind="parse",
        )

    def _wrap(self, fn: Callable, kind: str) -> Callable:
        def wrapped(*args, **kwargs):
            purpose = infer_purpose(kind, kwargs)
            model = _purpose_model(purpose, self._model)
            kwargs["model"] = model
            start = time.time()
            # 改造一：轮次预算贯通——LLM 调用在线程内发起，ContextVar 已由
            # chat()/close 绑定；timeout 与并发许可等待均以 remaining 为上限
            budget = current_budget()
            if _budget_exhausted_for_llm(budget):
                raise LLMBudgetExhausted("轮次预算耗尽，LLM 调用被拒绝")
            # 评审二轮 B3：单调时钟总截止——重试+降级的最坏墙钟有上界
            deadline = time.monotonic() + self._wall_clock_budget()
            got_permit = self._acquire_permit(budget)
            if not got_permit:
                metrics.record_budget_exhausted("semaphore")
                raise LLMBudgetExhausted(
                    f"等待 LLM 并发许可超时（剩余 {max(budget.remaining(), 0.0):.1f}s）"
                )
            try:
                attempt = 0
                last_exc: Optional[Exception] = None
                while attempt <= self._retries:
                    # 剩余预算放不下一次完整调用 → 不再发起（评审二轮 B3）。
                    # 内部 deadline 与 turn deadline 双夹逼：两者任一耗尽即停
                    if attempt > 0 and not self._can_attempt(deadline, budget):
                        break
                    self._apply_attempt_timeout(kwargs, budget)
                    try:
                        response = fn(*args, **kwargs)
                        self._record(model, purpose, kwargs, response, start)
                        return response
                    except RETRYABLE as e:
                        attempt += 1
                        last_exc = e
                        metrics.LLM_RETRIES.labels(model=model, kind="retryable").inc()
                        if attempt > self._retries:
                            break
                        backoff = min(2.0, 0.5 * (2 ** attempt)) + random.random() * 0.2
                        time.sleep(self._bound_backoff(backoff, deadline, budget))
                    except BadRequestError:
                        # 请求本身非法：重试无意义
                        raise
                # 重试耗尽 → 模型降级链（备用模型再试一次；同样受总预算约束）
                can_fallback = (
                    self._fallback and model != self._fallback
                    and self._can_attempt(deadline, budget)
                )
                if can_fallback:
                    metrics.LLM_RETRIES.labels(model=model, kind="fallback").inc()
                    try:
                        self._apply_attempt_timeout(kwargs, budget)
                        response = fn(*args, **{**kwargs, "model": self._fallback})
                        self._record(self._fallback, purpose, kwargs, response, start)
                        return response
                    except BadRequestError:
                        raise
                    except RETRYABLE as e:
                        last_exc = e
                if _budget_exhausted_for_llm(budget):
                    raise LLMBudgetExhausted(
                        "轮次预算耗尽，重试/降级路径终止"
                    ) from last_exc
                raise last_exc or RuntimeError("LLM 调用失败且无降级路径")
            finally:
                self._semaphore.release()

        return wrapped

    def _acquire_permit(self, budget) -> bool:
        """获取 pod 级并发许可；有轮次预算时以 remaining 为等待上限（不排队穿越 deadline）。

        返回 False = 等待超时（budget 耗尽）：调用方以 LLMBudgetExhausted 拒绝本次调用。
        """
        if budget is None:
            self._semaphore.acquire()
            return True
        remaining = budget.remaining()
        if remaining <= 0:
            return False
        return self._semaphore.acquire(timeout=remaining)

    def _can_attempt(self, deadline: float, budget) -> bool:
        """内部墙钟 deadline（按本包装器 timeout 基准）与轮次 deadline 双夹逼：
        任一耗尽即不允许发起下次尝试。"""
        if time.monotonic() + self._timeout_seconds > deadline:
            return False
        if _budget_exhausted_for_llm(budget):
            return False
        return True

    def _bound_backoff(self, backoff: float, deadline: float, budget) -> float:
        """退避睡眠同时受内部 deadline 与轮次 remaining 限制。"""
        cap = max(0.0, deadline - time.monotonic())
        if budget is not None:
            cap = min(cap, max(0.0, budget.remaining()))
        return min(backoff, cap)

    @staticmethod
    def _apply_attempt_timeout(kwargs: dict, budget) -> None:
        """单次尝试 timeout = min(wrapper 基准, 调用方既有 timeout, remaining)。

        调用前 remaining ≤ 0 直接放弃（LLMBudgetExhausted）；无轮次预算时
        只保留『调用方既有 timeout 与 wrapper 基准取小』——不再动用 SDK 600s 默认。
        """
        attempt_timeout = settings.llm_timeout_seconds  # wrapper 基准（显式 timeout 接线）
        caller_timeout = kwargs.get("timeout")
        if caller_timeout:
            attempt_timeout = min(attempt_timeout, float(caller_timeout))
        if budget is not None:
            remaining = budget.remaining()
            if remaining <= _LLM_MIN_REMAINING:
                raise LLMBudgetExhausted("轮次预算耗尽，LLM 调用被拒绝")
            attempt_timeout = min(attempt_timeout, remaining)
        kwargs["timeout"] = attempt_timeout

    def _wall_clock_budget(self) -> float:
        """总墙钟预算：全部计划尝试（含降级一次）× 单次超时。

        退避睡眠被夹到剩余预算内，最坏墙钟 ≈ 预算 + 一次超时余量，
        不再是「重试次数 × (超时 + 无界退避)」的开放级数（评审二轮 B3）。
        """
        attempts = self._retries + 1 + (1 if self._fallback else 0)
        return max(attempts, 1) * max(self._timeout_seconds or 0.0, 0.0)

    @staticmethod
    def _record(model, purpose, kwargs, response, start) -> None:
        latency_ms = (time.time() - start) * 1000
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total = getattr(usage, "total_tokens", 0) or 0
        metrics.record_tokens("prompt", purpose, prompt_tokens, model)
        metrics.record_tokens("completion", purpose, completion_tokens, model)
        tracing.record_llm_call(model, purpose, latency_ms, prompt_tokens, completion_tokens)


def _budget_exhausted_for_llm(budget) -> bool:
    """LLM 门的预算耗尽判定：严格 0 过早（Windows 定时器粒度），留 50ms 保守余量。"""
    return budget is not None and budget.remaining() <= _LLM_MIN_REMAINING


def install_resilience(client, model: str, **kwargs) -> ResilientLLM:
    """原地安装韧性包装（幂等：重复安装返回已有实例）。"""
    existing = getattr(client, "_resilient", None)
    if existing is not None:
        return existing
    wrapper = ResilientLLM(client, model, **kwargs)
    wrapper.install()
    client._resilient = wrapper
    return wrapper
