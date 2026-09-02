"""轮次预算（Agent能力强化计划·改造一）：全链路 deadline 的载体。

设计要点（评审·三轮1/2）：
- TurnBudget 是对象而非裸 ContextVar：deadline 在 chat() 起点一次确定
  （monotonic 时钟），同时存两处——Agent 实例属性 + ContextVar；
- turn 与 close 两个 worker 线程各自**显式绑定** ContextVar 并在 finally
  reset（close 在另一线程执行，无法继承 chat() 内的 ContextVar——绑定
  点在 server/runtime.run_agent_close_sync）；
- ContextVar 不能中断已在运行的调用——工具提交侧的预算规则在
  ToolBatchExecutor（提交前查剩余 / 只读到期弃等 / 写工具同步等幂等结果）。

口径：turn_budget_seconds=120 是安全熔断值，与 15s P95 SLO 无关；
SSE 断连等待上限由它推导（server/main._disconnect_wait_bound_seconds）。
"""

from __future__ import annotations

import time
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Optional


class LLMBudgetExhausted(RuntimeError):
    """轮次预算耗尽：LLM 调用被拒绝（ReAct/收尾/提取/记忆辅助共用）。"""


@dataclass(frozen=True)
class TurnBudget:
    """一轮对话的墙钟预算（monotonic deadline）。"""

    deadline: float

    @classmethod
    def start(cls, seconds: float) -> "TurnBudget":
        return cls(deadline=time.monotonic() + max(seconds, 0.0))

    def remaining(self) -> float:
        """剩余秒数；可为负（已超时）。"""
        return self.deadline - time.monotonic()

    def expired(self) -> bool:
        return self.remaining() <= 0

    def remaining_or(self, fallback: float) -> float:
        """有预算返回剩余（下限 0），无预算返回 fallback。"""
        if self.expired():
            return 0.0
        return self.remaining()


_CURRENT: ContextVar[Optional[TurnBudget]] = ContextVar("turn_budget", default=None)


def bind_budget(budget: Optional[TurnBudget]) -> Optional[Token]:
    """当前线程/上下文绑定轮次预算；返回 token 供 finally reset。"""
    if budget is None:
        return None
    return _CURRENT.set(budget)


def reset_budget(token: Optional[Token]) -> None:
    if token is not None:
        _CURRENT.reset(token)


def current_budget() -> Optional[TurnBudget]:
    return _CURRENT.get()


def budget_fallback_response():
    """预算耗尽的确定性结构化 fallback（改造一：零 LLM 强制收尾）。

    固定话术 + requires_human=true + intent=other；STM/摘要/LTM 等辅助
    LLM 任务在调用方直接跳过——辅助任务不得毁掉已生成的回复。
    """
    from app.schemas.response import CustomerServiceResponse, IntentType

    return CustomerServiceResponse(
        reply="很抱歉，本轮处理时间已达上限，已为您转接人工客服，请稍候。",
        intent=IntentType.OTHER,
        confidence=0.0,
        requires_human=True,
        follow_up_question=None,
    )
