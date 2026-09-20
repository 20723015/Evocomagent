"""LLM 韧性客户端（阶段四 4.4；推理模型全量适配计划·T1/T7）。

在共享 OpenAI client 上原地包裹（与 install_usage_tracking 一样不改调用面）：
1. pod 级并发信号量（护系统，与 3.7 用户级配额互补）；
2. 重试：指数退避 + 抖动，仅幂等读（网络/5xx/限流）；BadRequest 不重试；
3. 模型降级链：主模型 → fallback_model（重试耗尽后尝试）；
4. 显式 timeout（客户端构造时设，取代 SDK 默认 600s）；
5. token 归集（usage）与 4.3 指标、4.2 span 属性；
6. 小任务（提取/STM/摘要）路由廉价模型：purpose_models；
7. **模型画像参数改写（T1）**：按画像增删 temperature、改 max_tokens 参数名、
   抬输出下限、注入思考预算——17+ 调用点零改动的唯一改写出口（含降级链按
   备用模型画像重写）。

要点：双 LLM 调用治理——`_extract_structured_response` 走到这里时
purpose="extract"，经 extraction_model 切到廉价模型（路径①，最小改动）。
"""

from __future__ import annotations

import json
import math
import random
import threading
import time
import uuid
from collections.abc import Callable

from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from app.agent.turn_budget import LLMBudgetExhausted, current_budget
from app.config.settings import settings
from app.llm.model_profile import apply_profile, effective_profile
from app.observability import metrics, tracing
from app.observability.logging import get_logger
from app.security.ratelimit import current_budget_user

log = get_logger("app.llm.client")

RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)


def estimate_llm_tokens(kwargs: dict) -> int:
    """本次尝试的 token 预估：ceil(消息启发式 token × 1.5) + 本次 max_tokens。

    修复计划·四：预留按此估算；真实 usage 结算差额，低估也全额入账。

    推理模型适配 T1：画像可能把 `max_tokens` 改名为 `max_completion_tokens`
    （o 系列只认后者），因此输出上限按两个键取值——否则改名后预留只算输入，
    单次预留被系统性低估。公式本身不变：预留本就取上限，`llm_max_tokens`
    抬到 8192 后预留相应变大是正确行为（T7）。
    """
    messages = kwargs.get("messages") or []
    chars = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif content is not None:
            chars += len(json.dumps(content, ensure_ascii=False))
    heuristic = max(1, chars // 4)
    max_tokens = kwargs.get("max_tokens")
    if max_tokens is None:
        max_tokens = kwargs.get("max_completion_tokens")
    try:
        budget_tokens = int(max_tokens or 0)
    except (TypeError, ValueError):
        budget_tokens = 0
    return int(math.ceil(heuristic * 1.5)) + budget_tokens


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
    # 记忆链路只剩 LTM：提取/巩固 sweep 的 system prompt 均含「长期记忆」
    # （短期记忆槽位层已删除；注入段「历史偏好」不作为首条消息发给模型）
    if "历史记忆" in first or "长期记忆" in first:
        return "memory"
    # T5：结构化提取的 JSON 兜底路径（beta.parse 不被支持时）走的是
    # 普通 create，语义仍是提取 → 摘离到 extraction_model，别拖进推理主模型
    if "提取结构化" in first or "提取结构化信息" in first:
        return "extract"
    if kwargs.get("tools"):
        return "react"
    return "react"



def _purpose_model(purpose: str, model: str) -> str:
    """小任务路由廉价模型（4.4）：提取/摘要/记忆/路由走 extraction_model。"""
    cheap = settings.extraction_model
    if not cheap or purpose not in ("extract", "summarize", "memory", "router"):
        return model
    return cheap


# 辅助用途未摘离推理模型的告警只出一次/进程（T1 的部署动作提醒）
_WARNED_AUX_ON_REASONING = False


def _warn_aux_on_reasoning_model(model: str, purpose: str) -> None:
    """主模型是推理画像且 extraction_model 未配：辅助调用会被一起拖进推理模型。

    这正是 T1「配置 extraction_model」要解决的成本/延迟问题，因此在此处（唯一
    的路由决策点）告警，而不是在启动期加一段需要维护的探针。
    """
    global _WARNED_AUX_ON_REASONING
    if _WARNED_AUX_ON_REASONING:
        return
    profile = effective_profile(model)
    if profile.reasoning_field == "none" or profile.temperature_mode == "free":
        return  # 非推理画像（含未命中画像）：无该问题
    _WARNED_AUX_ON_REASONING = True
    log.warning(
        "llm.aux_on_reasoning_model purpose=%s model=%s：extraction_model 未配置，"
        "摘要/提取/记忆/路由等辅助调用仍走推理模型（推理模型适配 T1）",
        purpose, model,
    )


class ResilientLLM:
    """包装器：原地替换 client 的 create/parse 方法。"""

    def __init__(
        self,
        client,
        model: str,
        *,
        fallback_model: str | None = None,
        max_retries: int | None = None,
        max_concurrent: int | None = None,
        timeout_seconds: float | None = None,
        semaphore: threading.BoundedSemaphore | None = None,
        limiter=None,
    ):
        self._client = client
        self._model = model
        self._limiter = limiter  # 修复计划·四：预算预留/结算（None=不接预算）
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
            if model == self._model:
                _warn_aux_on_reasoning_model(model, purpose)
            # T1：画像改写前的调用方原参快照——降级链要按**备用模型**的画像重写，
            # 否则会把主模型的参数名/温度语义带给备用模型（如 o 系列改名后的
            # max_completion_tokens 传给 gpt-4o → 400）
            caller_kwargs = dict(kwargs)
            kwargs["model"] = model
            apply_profile(kwargs, model)
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
                last_exc: Exception | None = None
                while attempt <= self._retries:
                    # 剩余预算放不下一次完整调用 → 不再发起（评审二轮 B3）。
                    # 内部 deadline 与 turn deadline 双夹逼：两者任一耗尽即停
                    if attempt > 0 and not self._can_attempt(deadline, budget):
                        break
                    self._apply_attempt_timeout(kwargs, budget)
                    try:
                        response = self._invoke(lambda: fn(*args, **kwargs), kwargs)
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
                    # 备用模型独立画像改写（T1）：从快照重建，不继承主模型的改写结果
                    fallback_kwargs = {**caller_kwargs, "model": self._fallback}
                    apply_profile(fallback_kwargs, self._fallback)
                    try:
                        self._apply_attempt_timeout(fallback_kwargs, budget)
                        response = self._invoke(
                            lambda: fn(*args, **fallback_kwargs), fallback_kwargs,
                        )
                        self._record(
                            self._fallback, purpose, fallback_kwargs, response, start,
                        )
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

    def _invoke(self, call, kwargs: dict):
        """执行一次真实 LLM 尝试，并做预算预留/结算（修复计划·四）。

        - 预留（原子）：估算不足 → LLMBudgetExhausted（确定性收尾）；
          存储故障 → BudgetStoreUnavailable（503）；
        - 成功：按真实 usage 结算（低估也全额入账并记 overrun 指标）；
        - 失败：释放预留（重试/降级各自重新预留，独立计费）。
        """
        limiter = self._limiter
        reservation_id = None
        if limiter is not None and getattr(limiter, "daily_budget", 0) > 0:
            from app.security.ratelimit import UNSET_BUDGET_USER

            user_id = current_budget_user()
            if user_id == UNSET_BUDGET_USER:
                pass  # 离线/CLI（从未绑定请求用户）：不参与请求级预算
            elif not user_id:
                # 修复计划·二轮 7：显式空用户（后台任务缺归属）→ 拒绝匿名绕过
                from app.security.ratelimit import BudgetStoreUnavailable

                raise BudgetStoreUnavailable("缺少预算用户标识，拒绝匿名 LLM 调用")
            else:
                reservation_id = uuid.uuid4().hex
                if not limiter.reserve_tokens(
                    user_id, estimate_llm_tokens(kwargs), reservation_id
                ):
                    raise LLMBudgetExhausted(
                        "日 token 预算不足，LLM 调用被拒绝"
                    )
        try:
            response = call()
        except BaseException:  # noqa: BLE001 —— 含取消：失败/中断释放预留
            if reservation_id is not None:
                limiter.release_reservation(reservation_id)
            raise
        if reservation_id is not None:
            usage = getattr(response, "usage", None)
            actual = getattr(usage, "total_tokens", 0) or 0
            limiter.settle_tokens(reservation_id, actual)
        return response

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

    def _apply_attempt_timeout(self, kwargs: dict, budget) -> None:
        """单次尝试 timeout = min(wrapper 基准, 调用方既有 timeout, remaining)。

        调用前 remaining ≤ 0 直接放弃（LLMBudgetExhausted）；无轮次预算时
        只保留『调用方既有 timeout 与 wrapper 基准取小』——不再动用 SDK 600s 默认。
        wrapper 基准用实例 timeout_seconds（与 _can_attempt/_wall_clock_budget
        同口径；历史缺陷：@staticmethod 直读 settings，构造时注入的自定义
        timeout 被忽略）。
        """
        attempt_timeout = self._timeout_seconds  # wrapper 基准（显式 timeout 接线）
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
        reasoning_tokens = _reasoning_tokens(usage)
        # 阶段G：真实 usage 拆分方向计费（不再 70/30 估算）+ 调用次数指标
        # 推理模型适配 T7：reasoning token 从 completion 中拆出单列（N5：思考成本
        # 记成回答成本会让方向计费失真）
        metrics.record_llm_usage(
            model, purpose, prompt_tokens, completion_tokens, reasoning_tokens,
        )
        metrics.record_llm_call_count(purpose)
        tracing.record_llm_call(
            model, purpose, latency_ms, prompt_tokens, completion_tokens,
            reasoning_tokens,
        )


def _reasoning_tokens(usage) -> int:
    """从 usage 读推理 token（不同厂商字段位置不一；缺省 0）。

    - OpenAI 兼容：`usage.completion_tokens_details.reasoning_tokens`
    - 部分网关直接平铺：`usage.reasoning_tokens`
    """
    if usage is None:
        return 0
    details = getattr(usage, "completion_tokens_details", None)
    value = getattr(details, "reasoning_tokens", 0) if details is not None else 0
    if not value:
        value = getattr(usage, "reasoning_tokens", 0)
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0



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
