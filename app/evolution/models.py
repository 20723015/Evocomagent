"""第10期 QA 自动沉淀：核心数据结构。

- TurnRecord：一轮对话的落盘原始记录（脱敏后）。
- CandidateQA：从 turn 挖掘出的候选问答对。
- EvolutionReport：单次运行报告。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class SourceRef:
    """一条检索命中来源（source_path 由 chunker 提供，空表示 legacy 无法溯源）。"""

    source_path: str = ""
    doc: str = ""
    section: str = ""
    score: float = 0.0
    text: str = ""


@dataclass
class TurnRecord:
    """一轮对话的原始记录（已脱敏）。"""

    turn_id: str
    session_id: str
    mode: str  # "single" | "multi" | "legacy"
    ts: str
    question: str
    reply: str
    user_id: str = ""  # 阶段一/3.6：审计溯源用，旧 turn 文件缺省空串
    intent: str = ""
    confidence: float = 0.0
    requires_human: bool = False
    follow_up: Optional[str] = None
    sources: list[SourceRef] = field(default_factory=list)
    status: str = "captured"  # captured | mined | rejected
    # 3.6 审计：本轮工具调用明细（谁/什么参数/结果码），凭证已脱敏
    tool_calls: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TurnRecord":
        sources = [SourceRef(**s) for s in data.get("sources", [])]
        return cls(**{**data, "sources": sources})


@dataclass
class CandidateQA:
    """从 turn 挖掘出的候选问答对（进入 Judge 前后）。"""

    candidate_id: str
    turn_id: str
    question: str  # 规范化问题
    answer: str  # 规范化答案
    raw_question: str = ""  # 原始脱敏问题（预去重 0.95 用）
    intent: str = ""
    confidence: float = 0.0
    sources: list[SourceRef] = field(default_factory=list)
    filter_state: str = "pending"  # pending | judged | published | rejected
    # 价值 Judge 的质量分（_judge_round 赋值；沉淀到 frontmatter，
    # 供 P2-1 替换决策与 P1-2 重接地判断）
    quality_score: float = 0.0
    # P2-1：最终去重命中 evolved/ 旧沉淀且质量分更高 → 发布时替换的旧文件名（空=不替换）
    replaces: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CandidateQA":
        sources = [SourceRef(**s) for s in data.get("sources", [])]
        return cls(**{**data, "sources": sources})


@dataclass
class EvolutionReport:
    """单次运行报告（dry-run 与正式运行共用结构）。"""

    mined: int = 0
    pre_deduped: int = 0
    judged: int = 0
    pending: int = 0  # 值得沉淀但未发布（待审核或容量不足）
    sedimented: int = 0  # 实际发布文档数
    skipped: dict = field(
        default_factory=lambda: {
            "low_confidence": 0,
            "requires_human": 0,
            "no_sources": 0,
            "short": 0,
            "sensitive": 0,
            "duplicate": 0,
            "judge_rejected": 0,
        }
    )
    api_calls: int = 0  # 预计/实际 Judge API 调用数
    failures: int = 0  # 运行中的异常次数（如 embedding 不可用）
    aging: int = 0  # 本轮标记为 aging 的 pending 数量
    revalidated_checked: int = 0  # 本轮重接地核对的自进化文档数
    revalidated_passed: int = 0  # 重接地通过（刷新 last_validated）的文档数
    revalidated_failed: int = 0  # 重接地失败（隔离到 pending）的文档数
    revalidated_remaining: int = 0  # 当前 generation 尚待后续批次核对的文档数
    replaced: int = 0  # 本轮发布的近重复新答案替换掉的旧 evolved 文档数
    per_candidate: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
