"""TurnFinalizer（单 Agent 全量优化计划·阶段A/C/D/E）：固定顺序收尾管线。

顺序（计划原文，不得调整）：
1. 写操作状态校验（indeterminate → 转人工；无 committed 证据禁止宣称成功）；
2. 引用与事实接地校验（引用来源真实性 + 声明级数字接地）；
3. 业务升级规则（强投诉/强购买 → 转人工 + 上限 0.5）；
4. 输出安全检查与脱敏（输出侧 guardrail + PII 脱敏）；
5. 可靠度计算（确定性，模型不能提高程序给出的等级）；
6. 保存安全后的最终结果（TurnRepository.commit_turn）；
7. 产生 Handoff 事件（指标 + 事件回调）。

普通与 SSE 接口复用同一完成器：输出安全、Handoff 语义一致。
"""

from __future__ import annotations

from app.agent.citations import apply_citation_policy
from app.agent.fact_guard import ground_reply
from app.agent.final_response import FinalResponseArgs
from app.agent.input_policy import business_escalation
from app.agent.reliability import (
    compute_reliability,
)
from app.agent.turn_context import AgentTurnContext
from app.schemas.response import CustomerServiceResponse, IntentType
from app.security.guardrails import check_output
from app.security.scope_gate import SCOPE_BLOCK_REPLY

SAFE_FALLBACK_REPLY = "抱歉，我无法处理您这条请求，已为您转接人工客服，请稍候。"
SCOPE_REPLY = SCOPE_BLOCK_REPLY

# 规则拦截/确定性拒绝 → 可靠度 1.0（计划阶段E档位）
RULE_BASED_RELIABILITY = 1.0
# 输出安全命中（内容替换为安全话术）属硬风险 → 上限 0.2
OUTPUT_SAFETY_CAP = 0.2
# 业务升级硬规则上限
ESCALATION_CAP = 0.5


class TurnFinalizer:
    """响应完成器（每 pod 可共享；无请求状态驻留）。"""

    def __init__(self, repository):
        self._repository = repository

    # ---------- 入口 ----------
    def finalize(self, ctx: AgentTurnContext) -> CustomerServiceResponse:
        result = self._build_response(ctx)
        # 1. 写操作状态校验
        self._apply_write_op_state(ctx, result)
        # 2. 引用与事实接地校验
        self._apply_citation_and_facts(ctx, result)
        # 3. 业务升级规则
        self._apply_escalation(ctx, result)
        # 4. 输出安全检查与脱敏
        self._apply_output_safety(ctx, result)
        # 5. 可靠度计算
        ctx.reliability = self._compute_reliability(ctx, result)
        result.confidence = ctx.reliability
        # 6. 保存安全后的最终结果
        self._repository.commit_turn(ctx, result)
        # 7. Handoff 事件
        self._emit_handoff(ctx, result)
        return result

    # ---------- 规则响应（零 LLM 路径） ----------
    def finalize_rule_response(
        self, ctx: AgentTurnContext, reply: str, *, requires_human: bool,
        intent: IntentType = IntentType.OTHER, follow_up: str | None = None,
        handoff_reason: str = "",
    ) -> CustomerServiceResponse:
        """固定话术（护栏拦截/范围闸门/预算 fallback）的确定性收尾。

        rule_based=True → 可靠度 1.0（预算 fallback 例外 → 0.0）。
        """
        ctx.reliability_signal.rule_based = not ctx.budget_fallback
        if ctx.budget_fallback:
            ctx.reliability_signal.rule_based = False
            ctx.reliability_signal.budget_exhausted = True
        ctx.handoff_reason = handoff_reason or (
            "budget_exhausted" if ctx.budget_fallback else handoff_reason
        )
        result = CustomerServiceResponse(
            intent=intent,
            confidence=0.0,  # 由可靠度计算覆盖
            reply=reply,
            requires_human=requires_human,
            follow_up_question=follow_up,
        )
        ctx.reliability = compute_reliability(ctx.reliability_signal)
        result.confidence = ctx.reliability
        self._repository.commit_turn(ctx, result)
        self._emit_handoff(ctx, result)
        return result

    # ---------- 各步 ----------
    def _build_response(self, ctx: AgentTurnContext) -> CustomerServiceResponse:
        # Review 修复：唯一成功路径 = final_response 协议（纯文本路径已删除，
        # ReAct 无法产出合法终答时由 pipeline 直接走规则 fallback）
        final = ctx.final_args
        return CustomerServiceResponse(
            intent=final.intent if final is not None else IntentType.OTHER,
            confidence=0.0,  # 模型自评不采纳；由可靠度计算覆盖
            reply=final.reply if final is not None else "",
            requires_human=final.requires_human if final is not None else False,
            follow_up_question=final.follow_up_question if final is not None else None,
        )

    def _apply_write_op_state(self, ctx: AgentTurnContext, result) -> None:
        tracker = ctx.write_ops
        refund = tracker.refund
        # 执行器写超时与状态机观察可能记录同一事件 → 按锚点去重
        seen: set[tuple] = set()
        indeterminate: list[dict] = []
        for entry in list(refund.indeterminate) + list(ctx.extra_indeterminate):
            key = (
                entry.get("tool"), entry.get("order_id"),
                entry.get("idempotency_key"),
            )
            if key in seen:
                continue
            seen.add(key)
            indeterminate.append(entry)
        if indeterminate:
            ctx.reliability_signal.write_indeterminate = True
            result.requires_human = True
            ctx.handoff_reason = ctx.handoff_reason or "tool_indeterminate"
            ctx.indeterminate_writes = indeterminate
        # 无本轮 committed 证据 → 禁止宣称退款成功（确定性改写）
        safe_reply, rewritten = refund.final_reply_guard(result.reply)
        if rewritten:
            result.reply = safe_reply
            result.requires_human = True
            ctx.reliability_signal.write_indeterminate = True
            ctx.handoff_reason = ctx.handoff_reason or "refund_claim_without_commit"
            from app.observability.metrics import record_write_claim_blocked

            record_write_claim_blocked()

    def _apply_citation_and_facts(self, ctx: AgentTurnContext, result) -> None:
        from app.config.settings import settings

        if settings.citation_check_enabled:
            verdict = apply_citation_policy(result, ctx.sources)
            ctx.citation_verdict = verdict
            if verdict and verdict.get("missing"):
                ctx.reliability_signal.citation_missing = True
        if settings.fact_guard_enabled and result.reply:
            cleaned, fact_verdict = ground_reply(
                result.reply, ctx.evidence_texts,
                enabled=True,
            )
            ctx.fact_guard = fact_verdict.as_dict()
            if fact_verdict.removed_sentences or fact_verdict.conflicts:
                result.reply = cleaned
                if fact_verdict.conflicts:
                    ctx.reliability_signal.fact_conflict = True
                else:
                    # 无证据数字被删：检索/证据不足档
                    ctx.reliability_signal.retrieval_insufficient = True
                if fact_verdict.removed_sentences and "转人工" in cleaned:
                    result.requires_human = True
        # 工具证据 → 可靠度基础档
        successful = ctx.successful_tools()
        if successful:
            ctx.reliability_signal.tool_data_used = True
            if ctx.fact_guard and ctx.fact_guard["claims_total"] > 0:
                grounded = (
                    ctx.fact_guard["claims_grounded"] == ctx.fact_guard["claims_total"]
                )
                if grounded:
                    ctx.reliability_signal.tool_committed_evidence = True
            else:
                ctx.reliability_signal.tool_committed_evidence = True
        if any(t.name == "search_knowledge" for t in successful):
            ctx.reliability_signal.knowledge_grounded = (
                ctx.fact_guard is None or ctx.fact_guard["removed_sentences"] == 0
            )
            if not ctx.reliability_signal.knowledge_grounded:
                ctx.reliability_signal.knowledge_partial = True
        # 检索不足：发了检索请求但没有任何成功结果
        if any(t.name == "search_knowledge" for t in ctx.tool_trace) and not any(
            t.name == "search_knowledge" and t.ok and not t.skipped
            for t in ctx.tool_trace
        ):
            ctx.reliability_signal.retrieval_insufficient = True
        # 工具可重试错误（跳过/超时/错误码，非写副作用）
        for entry in ctx.tool_trace:
            if entry.name in ("apply_refund",):
                continue
            if entry.skipped or (not entry.ok):
                ctx.reliability_signal.tool_retryable_error = True
        if ctx.forced_finalize_failed:
            ctx.reliability_signal.budget_exhausted = True  # 无法生成有效终答 → 0.0

    def _apply_escalation(self, ctx: AgentTurnContext, result) -> None:
        escalation = business_escalation(ctx.sanitized_input or ctx.user_input)
        if escalation == "complaint":
            result.intent = IntentType.COMPLAINT
            result.requires_human = True
            ctx.escalation = "complaint"
            ctx.escalation_cap = ESCALATION_CAP
            ctx.handoff_reason = ctx.handoff_reason or "complaint"
        elif escalation == "purchase":
            result.requires_human = True
            ctx.escalation = "purchase"
            ctx.escalation_cap = ESCALATION_CAP
            ctx.handoff_reason = ctx.handoff_reason or "purchase_request"

    def _apply_output_safety(self, ctx: AgentTurnContext, result) -> None:
        from app.config.settings import settings

        if not settings.guardrails_enabled:
            return
        verdict = check_output(result.reply)
        if verdict.blocked:
            ctx.guardrail_output_block = True
            result.reply = SAFE_FALLBACK_REPLY
            result.requires_human = True
            ctx.handoff_reason = ctx.handoff_reason or "output_guardrail"
        # PII 脱敏（落库/工单/响应统一脱敏后文本）
        try:
            from app.evolution.sanitizer import sanitize_text

            masked = sanitize_text(result.reply)
            if masked != result.reply:
                result.reply = masked
        except Exception:  # noqa: BLE001 —— 脱敏失败不影响回复
            pass

    def _compute_reliability(self, ctx: AgentTurnContext,
                             result) -> float:
        signal = ctx.reliability_signal
        if ctx.escalation_cap is not None:
            signal.overrides["escalation"] = min(
                signal.overrides.get("escalation", 1.0), ctx.escalation_cap,
            )
        if ctx.guardrail_output_block:
            signal.overrides["output_safety"] = OUTPUT_SAFETY_CAP
        value = compute_reliability(signal)
        if not signal.budget_exhausted:
            value = max(value, 0.0)
        return value

    def _emit_handoff(self, ctx: AgentTurnContext, result) -> None:
        if not result.requires_human:
            return
        from app.observability.metrics import record_handoff

        record_handoff(ctx.handoff_reason or "requires_human")
        ctx.handoff_emitted = True
