"""运行轨迹：沙箱采集的 Agent 运行过程与结果载体（第9期）。

沙箱通过给共享的 OpenAI client 与 ToolManager 插桩，把一次会话里发生的所有
LLM 调用（token / 被请求的工具 / 延迟）和工具返回都记录到 RunTrace 里。
评估器随后只读 RunTrace，不再碰 Agent 内部，实现「采集」与「评分」解耦。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.schemas.response import CustomerServiceResponse


@dataclass
class LLMCallRecord:
    """单次 LLM 调用的记录。"""

    purpose: str  # 启发式标注：router / react / extract
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tool_calls: list[dict] = field(default_factory=list)  # 本次响应请求的工具 [{name, arguments}]
    latency_ms: float = 0.0


@dataclass
class ToolObservation:
    """单次工具调用的输入与返回（幻觉检测的 ground truth）。

    outcome：结构化结果摘要（2.2）——从 result JSON 中抽取的可判定字段
    （success / code / count / status），供安全评测比对，不整段拷贝原始结果。
    """

    name: str
    arguments: dict
    result: str  # 工具返回的 JSON 字符串
    outcome: dict | None = None  # 结构化摘要：{"success": bool, "code": str|None, ...}


@dataclass
class RunTrace:
    """一条用例完整运行的轨迹。"""

    case_id: str
    turns: list[str]
    final_response: Optional["CustomerServiceResponse"] = None
    route: str | None = None  # 多 Agent 模式下实际路由到的子 Agent
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    tool_observations: list[ToolObservation] = field(default_factory=list)
    # Agent能力强化计划·改造三：QuoteCheck 输入/输出
    retrieved_sources: list[str] = field(default_factory=list)  # 实际来源集合（规范化值）
    citation_verdict: dict | None = None  # Agent 内部引用校验 verdict（最后轮）
    error: str | None = None  # 运行异常信息，None=正常

    # ---------- 便捷聚合属性 ----------
    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.llm_calls)

    @property
    def num_llm_calls(self) -> int:
        return len(self.llm_calls)

    @property
    def num_tool_calls(self) -> int:
        return len(self.tool_observations)

    @property
    def tool_call_names(self) -> list[str]:
        """实际调用过的工具名（按调用顺序）。"""
        return [obs.name for obs in self.tool_observations]

    @property
    def tool_outcomes(self) -> list[dict]:
        """结构化工具结果摘要列表（2.2）：只携带可判定字段，避免整段原始结果入库。"""
        return [
            {"name": obs.name, "arguments": obs.arguments, "outcome": obs.outcome}
            for obs in self.tool_observations
        ]

    def to_dict(self) -> dict:
        """精简快照，供报告 JSON 输出。"""
        resp = self.final_response
        return {
            "case_id": self.case_id,
            "turns": self.turns,
            "route": self.route,
            "reply": resp.reply if resp else None,
            "intent": resp.intent.value if resp else None,
            "requires_human": resp.requires_human if resp else None,
            "total_tokens": self.total_tokens,
            "num_llm_calls": self.num_llm_calls,
            "num_tool_calls": self.num_tool_calls,
            "tool_calls": self.tool_call_names,
            "tool_outcomes": self.tool_outcomes,
            "tool_outputs": [
                {"name": obs.name, "arguments": obs.arguments, "result": obs.result}
                for obs in self.tool_observations
            ],
            "retrieved_sources": self.retrieved_sources,
            "citation_verdict": self.citation_verdict,
            "error": self.error,
        }
