"""声明级事实接地（单 Agent 全量优化计划·阶段D）。

对终答抽取事实声明（金额、时效、比例、资格、流程步骤），逐条映射到
本轮证据（工具结果 / EvidencePack 检索片段）：
- 数字声明必须能在证据原文中找到出处，无证据数字 → 删除该句；
- 全部内容被删除 → 回复转为「需核实/转人工」提示；
- 跨文档推断/证据冲突检测（同一数字出现不一致值 → conflict）。

校验是确定性的字符串级匹配（不做 NLU 判断），宁可少删不误删：
引用语境的数字（「7天无理由」政策名等）也要求证据里出现同样数字。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 数字声明：金额 / 时效 / 比例 / 计数
_NUMBER_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:元|块钱|天|个工作日|工作日|小时|分钟|秒|个月|月|年|%|％|倍|单|次|件|kg|千米|公里)"
)
_NUM_ALL = re.compile(r"\d+(?:\.\d+)?")
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])")

# 用业务主题而不是句首修饰语做冲突键。顺序表示更具体的概念优先；
# 未命中时仍回退到首个内容词，兼容未知领域声明。
_TOPIC_TERMS: tuple[str, ...] = (
    "退款", "退货", "换货", "到货", "发货", "物流", "配送",
    "到账", "支付", "订单", "保修", "客服", "响应",
)
_LEADING_MODIFIERS_RE = re.compile(
    r"^(?:本次|此次|这次|该次|当前|目前|预计|通常|一般|大约|约)+"
)


@dataclass
class FactGuardVerdict:
    """声明级校验结论（审计/指标用）。"""

    claims_total: int = 0
    claims_grounded: int = 0
    removed_sentences: int = 0
    conflicts: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.removed_sentences == 0 and not self.conflicts

    def as_dict(self) -> dict:
        return {
            "claims_total": self.claims_total,
            "claims_grounded": self.claims_grounded,
            "removed_sentences": self.removed_sentences,
            "conflicts": self.conflicts,
        }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _sentence_has_evidence(sentence: str, evidence_blob: str) -> bool:
    """句子中的每个数字声明都要在证据中找到（归一化包含匹配）。"""
    claims = _NUMBER_RE.findall(sentence)
    if not claims:
        return True  # 无数字声明：不做词面拦截（声明级校验聚焦数字）
    for claim in claims:
        if _normalize(claim) not in evidence_blob:
            return False
    return True


def _claim_clauses(text: str) -> list[tuple[str, str]]:
    """拆子句并抽取（业务主题锚点, 子句归一化文本）。

    主题锚点优先取已知业务概念，并忽略「本次/目前」等句首修饰语；
    未命中领域词时回退到汉字首 bigram 或 ASCII 首词。这样「退款将在7天内…」与
    「退款周期为15天…」都锚定「退款」，而「退货…7天」与「到货…15天」
    分别锚定「退货」「到货」不同主题，不互判冲突。
    """
    import re as _re
    import unicodedata as _uni

    clauses: list[tuple[str, str]] = []
    for raw in _re.split(r"[，,；;、\n]", str(text or "")):
        norm = _normalize(raw)
        if not norm:
            continue
        stripped = _NUMBER_RE.sub(" ", raw)
        stripped = _NUM_ALL.sub(" ", stripped)
        clean = _uni.normalize("NFKC", stripped)
        clean = _LEADING_MODIFIERS_RE.sub("", clean.strip())
        anchor = next((term for term in _TOPIC_TERMS if term in clean), "")
        ascii_match = _re.search(r"[A-Za-z0-9]{2,}", clean)
        if not anchor:
            for m in _re.finditer(r"([一-鿿]{2})", clean):
                anchor = m.group(1)
                break
        if not anchor and ascii_match:
            anchor = ascii_match.group(0).casefold()
        clauses.append((anchor, norm))
    return clauses


def _claim_key_of_clause(clause_norm: str) -> list[tuple[str, str]]:
    """子句内的（单位 → 值）清单。"""
    out: list[tuple[str, str]] = []
    for m in _NUMBER_RE.finditer(clause_norm):
        value = _NUM_ALL.match(m.group(0)).group(0)
        unit = m.group(0)[len(value):]
        if unit:
            out.append((unit, value))
    return out


def _detect_conflicts(reply: str, evidence_texts: list[str]) -> list[str]:
    """证据冲突（Review 修复）：按「回复句主题锚点 + 单位」匹配证据。

    同一声明键（同主题锚点 + 同单位）在证据间存在多个不同值且回复选取
    其中之一 → conflict；不同主题但单位相同（如 7天退货 vs 15天到货）
    不判冲突。
    """
    conflicts: list[str] = []
    # 证据声明键：(锚点, 单位) → 值集合
    evidence_keys: dict[tuple[str, str], set[str]] = {}
    for text in evidence_texts:
        for anchor, norm in _claim_clauses(text):
            for unit, value in _claim_key_of_clause(norm):
                evidence_keys.setdefault((anchor, unit), set()).add(value)

    for anchor, norm in _claim_clauses(reply):
        for unit, value in _claim_key_of_clause(norm):
            values = evidence_keys.get((anchor, unit))
            if values and len(values) > 1 and value in values:
                conflicts.append(f"{value}{unit}")
    return conflicts


def ground_reply(reply: str, evidence_texts: list[str],
                 *, enabled: bool = True) -> tuple[str, FactGuardVerdict]:
    """声明级接地校验。

    返回 (校验后回复, verdict)。无证据（空证据集）时不拦截——工具/知识
    都没用的纯流程回复（道歉、确认收货地址）不该被误杀。

    Review 修复：证据冲突（同主题+同单位多证据值且回复取其一）的句子
    一并删除并改为核实/转人工话术（不再保留可能过时的单一取值）。
    """
    verdict = FactGuardVerdict()
    if not enabled or not reply:
        return reply, verdict
    evidence_blob = _normalize("".join(evidence_texts))
    if not evidence_blob:
        return reply, verdict  # 无证据：放行（不误杀非事实回复）

    sentences = [s for s in _SENT_SPLIT_RE.split(reply) if s]
    kept: list[str] = []
    for sentence in sentences:
        claims = _NUMBER_RE.findall(sentence)
        verdict.claims_total += len(claims)
        sentence_conflicts = _detect_conflicts(sentence, evidence_texts)
        for conflict in sentence_conflicts:
            if conflict not in verdict.conflicts:
                verdict.conflicts.append(conflict)
        if _sentence_has_evidence(sentence, evidence_blob) and not sentence_conflicts:
            kept.append(sentence)
            verdict.claims_grounded += len(claims)
        else:
            verdict.removed_sentences += 1

    cleaned = "".join(kept).strip()
    if verdict.removed_sentences and not cleaned:
        # 全部被删：转核实提示（不做内容编造）
        cleaned = "您咨询的具体数字需要进一步核实，我暂时无法给出准确答复；已为您转人工确认，请稍候。"
    if verdict.removed_sentences and cleaned and not cleaned.endswith(("。", "！", "？", "！", "?")):
        cleaned += "。"
    if verdict.removed_sentences and cleaned:
        cleaned += "以上信息以官方政策与实际到账为准，如需精确确认可转人工核实。"
    return cleaned, verdict
