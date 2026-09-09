"""输入策略（单 Agent 全量优化计划·阶段A）：一套核心实现，多入口共用。

API（server/main.py）、CLI（main.py）、评估（sandbox）统一经过 InputPolicy：
- guardrail（注入 → block；PII → 脱敏继续）；
- 业务范围闸门（非业务/闲聊 → 固定引导话术）；
- 业务升级规则（强投诉/强购买意图识别，纯规则零 LLM）。

拦截类判定返回 action，调用方按 action 生成规则响应（零 LLM 消耗）。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.security.guardrails import check_input, guardrail_enabled
from app.security.scope_gate import SCOPE_BLOCK_REPLY, check_scope, scope_gate_enabled


@dataclass
class InputDecision:
    """一次输入评估结论。

    action:
    - "guardrail_block"：注入/敏感内容 → 固定拒绝话术 + 转人工；
    - "scope_block"：非业务 → 固定引导话术（不转人工）；
    - "continue"：放行（text 为脱敏后文本）。
    """

    action: str
    text: str
    reason: str = ""
    reply: str = ""       # 拦截类固定话术（continue 时为空）
    requires_human: bool = False


def evaluate_input(text: str, ctx=None, *, client=None, model: str = "") -> InputDecision:
    """统一输入评估：guardrail → 范围闸门（顺序与历史行为一致）。"""
    if guardrail_enabled(ctx):
        verdict = check_input(text)
        if verdict.blocked:
            return InputDecision(
                action="guardrail_block",
                text=text,
                reason=verdict.reason,
                reply=(
                    "抱歉，您的消息包含疑似指令注入/敏感内容，出于安全考虑已被拦截，"
                    "本条消息已转接人工客服处理，请稍候。"
                ),
                requires_human=True,
            )
        if verdict.masked:
            text = verdict.text

    if scope_gate_enabled(ctx):
        verdict = check_scope(text, client, model)
        if not verdict.in_scope:
            return InputDecision(
                action="scope_block",
                text=text,
                reason=verdict.reason,
                reply=SCOPE_BLOCK_REPLY,
            )

    return InputDecision(action="continue", text=text)


# ============================================================
# 业务升级规则（2026-08 评测后接线；从 chat.py 迁入，唯一实现）
# ============================================================
_COMPLAINT_SIGNALS = (
    "投诉", "举报", "消协", "工商局", "起诉", "曝光", "赔偿",
    "告你们", "态度差", "敷衍", "欺诈",
)
# 强购买意图（平台无下单工具）→ 转人工；同句含咨询词（价格/介绍等）视为咨询不升级
_PURCHASE_SIGNALS = ("我要买", "买一个", "拍下", "立即购买", "帮我下单", "直接买")
_CONSULT_HINTS = ("多少钱", "价格", "批发", "介绍", "推荐", "有货吗", "怎么卖")


def has_complaint_signal(text: str) -> bool:
    return any(k in text for k in _COMPLAINT_SIGNALS)


def has_purchase_signal(text: str) -> bool:
    if any(k in text for k in _PURCHASE_SIGNALS):
        return not any(k in text for k in _CONSULT_HINTS)
    return False


def business_escalation(text: str) -> str | None:
    """返回升级类别（"complaint" | "purchase"）或 None。"""
    if has_complaint_signal(text):
        return "complaint"
    if has_purchase_signal(text):
        return "purchase"
    return None
