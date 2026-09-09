"""确定性可靠度计算（单 Agent 全量优化计划·阶段E）。

外部 confidence 保持 0–1 兼容，但语义改为程序计算的可靠度：
- 模型自评分不再进入结果（final_response 不含 confidence 字段）；
- 基础档由本轮最强证据决定，硬风险规则只能压低上限，不能提高；
- 全部确定性、无 LLM 参与，单测可穷举。

档位（计划原文）：
- 1.0：纯规则响应或确定性安全拒绝；
- 0.9：成功工具结果直接支持全部关键事实；
- 0.8：知识回答全部声明有有效证据；
- 0.6：非事实型回复或存在轻微不确定性；
- 上限 0.4：检索不足、参数缺失、工具可重试错误；
- 上限 0.2：引用缺失、事实冲突或写操作结果未知；
- 0.0：预算耗尽、无法生成有效终答。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 基础档（证据由强到弱取最高可达档）
TIER_RULE_BASED = 1.0        # 纯规则响应/确定性安全拒绝（护栏拦截、范围闸门、固定话术）
TIER_TOOL_SUPPORTED = 0.9    # 成功工具结果直接支持全部关键事实
TIER_KNOWLEDGE_GROUNDED = 0.8  # 知识回答全部声明有有效证据
TIER_NON_FACTUAL = 0.6       # 非事实型回复或轻微不确定
TIER_BUDGET_EXHAUSTED = 0.0  # 预算耗尽/无法生成有效终答

# 硬风险上限
CAP_RETRIEVAL_INSUFFICIENT = 0.4
CAP_PARAM_MISSING = 0.4
CAP_TOOL_RETRYABLE_ERROR = 0.4
CAP_CITATION_MISSING = 0.2
CAP_FACT_CONFLICT = 0.2
CAP_WRITE_INDETERMINATE = 0.2


@dataclass
class ReliabilitySignal:
    """一轮终答的可靠度信号（TurnFinalizer 收集，计算前只读）。"""

    rule_based: bool = False                 # 固定话术/确定性拒绝（护栏/范围闸门/兜底）
    budget_exhausted: bool = False           # 预算耗尽 fallback
    tool_committed_evidence: bool = False    # 本轮成功工具结果直接支持关键事实
    tool_data_used: bool = False             # 回复基于工具数据（非知识检索）
    knowledge_grounded: bool = False         # 知识回答全部声明有有效证据（阶段D声明级校验）
    knowledge_partial: bool = False          # 知识回答但存在未接地声明 → 轻微不确定
    retrieval_insufficient: bool = False     # 检索不足（空结果/低于阈值）
    param_missing: bool = False              # 关键参数缺失（只追问）
    tool_retryable_error: bool = False       # 工具可重试错误（超时/跳过，非写副作用）
    citation_missing: bool = False           # 引用缺失（verdict missing / 零来源有引用）
    fact_conflict: bool = False              # 证据冲突/事实接地失败
    write_indeterminate: bool = False        # 写操作结果未知
    write_rejected: bool = False             # 写操作被拒（越权/确认失败）
    overrides: dict[str, float] = field(default_factory=dict)  # 特殊规则直接定值

    def cap(self, value: float) -> float:
        return round(value, 4)


def compute_reliability(signal: ReliabilitySignal) -> float:
    """按信号计算 0–1 可靠度（确定性；硬风险只降不升）。"""
    if signal.budget_exhausted:
        return TIER_BUDGET_EXHAUSTED

    # 特殊规则覆盖（如业务升级上限由调用方直接给出）
    if signal.overrides:
        value = max(signal.overrides.values())
        value = _apply_caps(value, signal)
        return round(value, 4)

    if signal.rule_based:
        value = TIER_RULE_BASED
    elif signal.tool_committed_evidence or (signal.tool_data_used and not signal.knowledge_partial):
        value = TIER_TOOL_SUPPORTED
    elif signal.knowledge_grounded:
        value = TIER_KNOWLEDGE_GROUNDED
    elif signal.knowledge_partial or signal.param_missing:
        value = min(TIER_NON_FACTUAL, _apply_caps(TIER_NON_FACTUAL, signal))
    else:
        value = TIER_NON_FACTUAL

    value = _apply_caps(value, signal)
    return round(value, 4)


def _apply_caps(value: float, signal: ReliabilitySignal) -> float:
    if signal.write_indeterminate:
        value = min(value, CAP_WRITE_INDETERMINATE)
    if signal.write_rejected:
        value = min(value, CAP_TOOL_RETRYABLE_ERROR)
    if signal.citation_missing:
        value = min(value, CAP_CITATION_MISSING)
    if signal.fact_conflict:
        value = min(value, CAP_FACT_CONFLICT)
    if signal.retrieval_insufficient:
        value = min(value, CAP_RETRIEVAL_INSUFFICIENT)
    if signal.tool_retryable_error:
        value = min(value, CAP_TOOL_RETRYABLE_ERROR)
    return max(value, 0.0)


def escalation_cap(intent_value: str, requires_human: bool) -> float | None:
    """业务升级硬规则（2026-08 评测接线延续）：强投诉/强购买 → 上限 0.5。

    返回 None 表示无升级；返回值作为可靠度上限。
    """
    if requires_human and intent_value in ("complaint",):
        return 0.5
    if requires_human and intent_value == "order_query":
        return None
    return None
