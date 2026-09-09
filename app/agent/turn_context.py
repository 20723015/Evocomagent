"""AgentTurnContext（单 Agent 全量优化计划·阶段A）：统一轮次上下文。

每轮持有全部轮次状态，所有组件显式接收该对象，逐步替代实例临时字段：
- turn_id、用户原始输入与脱敏输入；
- 全局 deadline（TurnBudget）与 token 预算；
- 工具调用轨迹、检索证据、写操作状态（WriteOpTracker）；
- Guardrail 结论、引用 verdict、事实校验结论、Handoff 原因；
- 各阶段耗时与 LLM 用量（调用次数 / token）。

ContextVar 只保留底层兼容用途（turn_budget.py 的线程绑定）。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from app.agent.reliability import ReliabilitySignal
from app.agent.token_budget import budget_shares
from app.agent.turn_budget import TurnBudget
from app.agent.write_ops import WriteOpTracker


@dataclass
class ToolTraceEntry:
    """一次工具调用的轮内轨迹（含审计所需的完整结果）。"""

    call_id: str
    name: str
    arguments: dict
    result: str                 # 完整 JSON 字符串（仅审计存储）
    digest: str = ""            # 结构化摘要（进入历史 metadata）
    ok: bool = True
    code: str = ""              # 稳定结果码（ORDER_FOUND / error / ...）
    side_effect_status: str = "none"  # none|pending|committed|indeterminate
    skipped: bool = False
    sequence: int = 0


@dataclass
class StageTimings:
    """各阶段耗时（毫秒）。"""

    context_build_ms: float = 0.0
    react_ms: float = 0.0
    finalize_ms: float = 0.0
    persist_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "context_build_ms": round(self.context_build_ms, 1),
            "react_ms": round(self.react_ms, 1),
            "finalize_ms": round(self.finalize_ms, 1),
            "persist_ms": round(self.persist_ms, 1),
        }


@dataclass
class AgentTurnContext:
    """一轮对话的全部状态（每轮新建；组件显式接收）。"""

    user_input: str
    budget: TurnBudget
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    sanitized_input: str = ""

    # 输入策略
    input_action: str = "continue"   # continue | guardrail_block | scope_block
    input_reason: str = ""

    # 上下文构建
    current_query: str = ""          # 记忆相关性 query（= 原始输入）
    context_tokens: int = 0          # 本轮最后一次构建的上下文估算 token
    context_truncated: bool = False  # 历史被水位裁剪/压缩
    truncation_stage: str = ""       # 截断发生位置（history | memory | none）

    # ReAct 执行
    react_steps: int = 0
    llm_calls: int = 0
    final_args: object | None = None       # FinalResponseArgs | None
    final_text: str = ""                       # 兼容路径的纯文本终答
    forced_finalize: bool = False              # 步数耗尽强制终答

    # 工具轨迹与证据
    tool_trace: list[ToolTraceEntry] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)   # 本轮检索来源（规范化值）
    evidence_texts: list[str] = field(default_factory=list)  # EvidencePack 片段原文
    write_ops: WriteOpTracker = field(default_factory=WriteOpTracker)

    # 收尾
    citation_verdict: dict | None = None
    fact_guard: dict | None = None
    escalation: str | None = None             # complaint | purchase | None
    escalation_cap: float | None = None       # 业务升级硬规则可靠度上限
    handoff_reason: str = ""                     # requires_human 时的归因
    handoff_emitted: bool = False
    reliability_signal: ReliabilitySignal = field(default_factory=ReliabilitySignal)
    reliability: float = 0.0
    guardrail_output_block: bool = False
    refund_decision: object | None = None        # Review 修复：退款确认三态判定

    # 写操作与兜底
    budget_fallback: bool = False                # 预算耗尽确定性 fallback（可靠度 0.0）
    forced_finalize_failed: bool = False         # 强制终答失败（视为无法生成有效终答）
    extra_indeterminate: list[dict] = field(default_factory=list)  # 执行器写超时清单
    indeterminate_writes: list[dict] = field(default_factory=list)  # 最终对账清单

    # 消息窗口切片（本轮在 raw_messages 中的起止）
    slice_start: int = 0
    full_turn_messages: list[dict] = field(default_factory=list)  # 含中间工具消息（审计）

    timings: StageTimings = field(default_factory=StageTimings)
    started_at: float = field(default_factory=time.monotonic)

    def token_shares(self, context_window_tokens: int) -> dict[str, int]:
        return budget_shares(context_window_tokens)

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.started_at) * 1000

    def successful_tools(self) -> list[ToolTraceEntry]:
        return [t for t in self.tool_trace if t.ok and not t.skipped]

    def to_usage_dict(self) -> dict:
        """轮次用量摘要（指标/评估报告共用，不含用户原文）。"""
        return {
            "turn_id": self.turn_id,
            "react_steps": self.react_steps,
            "llm_calls": self.llm_calls,
            "tool_calls": len(self.tool_trace),
            "context_tokens": self.context_tokens,
            "context_truncated": self.context_truncated,
            "truncation_stage": self.truncation_stage,
            "timings": self.timings.as_dict(),
            "latency_ms": round(self.elapsed_ms(), 1),
        }
