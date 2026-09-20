"""ReactRunner（单 Agent 全量优化计划·阶段A/A2 + Review 修复）：ReAct 循环。

Review 修复后的终答协议（唯一成功路径）：
- 每一步只能调用业务工具或 `final_response`（虚拟终止工具，不进 ToolManager）；
- `final_response` 参数合法 → 结构化终答，循环结束（无第二次结构化提取调用）；
  被接受的 final call 也写入 assistant(tool_calls) + 恰好一个 tool 结果
  （审计消息序完整：所有 tool call 都有且仅有一个对应结果）；
- 参数不合法 → 结构化纠错信息回模型，可在剩余步数内修复；
- **纯文本输出不再视为终答**：追加协议纠错并重试（消耗剩余步数）；
- 混合业务工具 + final_response：只写一条 assistant tool-call 消息，
  每个 call ID 恰好一个 tool 结果（final 全部返回协议错误）；
- 达到最大步数 → 最后一轮只挂 final_response 强制收尾（显式 tool_choice；
  画像不支持强制 tool_choice 时降级为 system 软强制，见 T4）；
  仍失败 → ForcedFinalizeFailed（上层确定性转人工 fallback）；
- 步数余量提示：剩余步数（含当前步）≤ 2 时注入一条 system 预告（只注一次，
  零额外 LLM 调用），与耗尽后的最后通牒语义互补；
- 预算耗尽 → LLMBudgetExhausted（上层确定性 fallback，零 LLM）；
- 工具批次经 ToolBatchExecutor（原序分段并行 + 守卫 + 写状态机拦截 + 退款
  确认注入）。

推理模型全量适配（T2/T3/T4/T6）：
- **参数**：调用点传的是「调用方偏好值」，是否发给厂商由 `llm/client.py` 的
  画像层决定（T1）——这里直传 `temperature`/`max_tokens` 不改，语义已降级为
  偏好，勿在调用点按模型分支（那会把改写面重新摊到 17 处）；
- **reasoning 双通道**（T3）：窗口通道按画像剥离/原样回传，审计通道
  （`full_turn_messages`）始终留存 `reasoning` 附加字段；
- **强制收尾**（T4）：达成手段可按画像降级为软强制，`ForcedFinalizeFailed`
  兜底语义与上层转人工路径零改动；
- **SSE**（T6）：`thought` 仍是受控状态说明；模型推理原文只在
  `settings.sse_reasoning_enabled`（默认关，产品/合规决策）打开时经
  `emit_reasoning` 透出，且过输出 guardrails。
"""

from __future__ import annotations

import json
import time

from app.agent.final_response import (
    FINAL_RESPONSE_TOOL_DEFINITION,
    FINAL_RESPONSE_TOOL_NAME,
    FinalResponseArgs,
    validate_final_response,
)
from app.agent.tools.batch_executor import ToolTurnState
from app.agent.tools.digest import digest_tool_result
from app.agent.turn_budget import LLMBudgetExhausted
from app.agent.turn_context import AgentTurnContext, ToolTraceEntry
from app.llm.model_profile import effective_profile
from app.llm.reasoning import (
    CapturedReasoning,
    capture_reasoning,
    merge_audit,
    window_content,
    window_extra_fields,
)

FINAL_ACCEPTED_RESULT = json.dumps(
    {"success": True, "accepted": True, "code": "FINAL_RESPONSE_ACCEPTED"},
    ensure_ascii=False,
)
PLAIN_TEXT_CORRECTION = (
    "你上一步输出了纯文本而没有调用 final_response 工具。"
    "请基于你上面的答复内容，立即调用 final_response 工具提交最终答复"
    "（intent/reply/requires_human/follow_up_question）；不要再输出纯文本。"
)
# T4：画像不支持强制 tool_choice 时的软强制提示（挂终止工具 + 该 system 提示，
# 提高仍能走协议的概率；仍失败则 ForcedFinalizeFailed 语义不变）
FORCED_FINALIZE_SOFT_PROMPT = (
    "已达到本轮的步骤上限。你现在必须调用 final_response 工具提交最终答复"
    "（intent/reply/requires_human/follow_up_question），不要再输出纯文本、"
    "也不要再调用其他工具。"
)
# 步数余量提示：模型看不到自己还剩几步，只有耗尽后才收到最后通牒
# （FORCED_FINALIZE_SOFT_PROMPT），实测缺口是「最后一两步还在开新查询方向」。
# 阈值取 2（含当前步）：完成必要的最后查询后尽快收尾。与最后通牒语义互补——
# 该提示是预告（仍可不调用 final_response 继续干活），不是协议强制。
STEPS_MARGIN_REMAINING = 2
STEPS_MARGIN_PROMPT = (
    "提示：本轮工具调用步骤即将用尽（含当前这一步，还剩 {remaining} 步）。"
    "请尽快完成必要的最后一次查询并调用 final_response 提交最终答复，"
    "不要再开启新的查询方向。"
)

_FINAL_PREMATURE_ERROR = json.dumps({
    "error": "FINAL_RESPONSE_PREMATURE",
    "message": "本轮还有业务工具刚返回结果，请先基于结果回答，再重新调用 final_response",
}, ensure_ascii=False)
_FINAL_MULTIPLE_ERROR = json.dumps({
    "error": "FINAL_RESPONSE_PROTOCOL",
    "message": "一次只能调用一个 final_response；请只保留一个最终答复调用",
}, ensure_ascii=False)


class ForcedFinalizeFailed(RuntimeError):
    """步数耗尽后的强制终答失败（无效输出）→ 上层确定性 fallback。"""


class ReactRunner:
    """ReAct 循环（每轮由 AgentTurnContext 驱动；无请求状态驻留）。"""

    def __init__(self, agent):
        self._agent = agent
        self._ctx: AgentTurnContext | None = None

    # ---------- 入口 ----------
    def run(self, ctx: AgentTurnContext, state: ToolTurnState) -> FinalResponseArgs:
        """执行循环；唯一成功返回 FinalResponseArgs。

        抛出 LLMBudgetExhausted（预算耗尽）/ ForcedFinalizeFailed（强制收尾
        失败）；其余异常向上传播（500 语义）。
        """
        self._ctx = ctx
        agent = self._agent
        max_steps = agent.max_react_steps
        margin_hint_sent = False
        for step in range(max_steps):
            ctx.react_steps = step + 1
            if ctx.budget.expired():
                raise LLMBudgetExhausted("轮次预算耗尽，停止 ReAct 循环")
            # 步数余量提示（含当前步）：只注一次，注入先于本步 build()，
            # 当步即生效。窗口从尾保留，该消息不会被 trim 裁掉。
            remaining = max_steps - step
            hint_now = remaining <= STEPS_MARGIN_REMAINING and not margin_hint_sent
            if hint_now:
                margin_hint_sent = True
                ctx.steps_margin_hint = True
                self._push_window_message({
                    "role": "system",
                    "content": STEPS_MARGIN_PROMPT.format(remaining=remaining),
                })
                from app.observability.metrics import record_steps_margin_hint

                record_steps_margin_hint()
            # 受控状态说明（SSE thought 事件：不再透出模型原始思考）
            agent.emit_status(
                (
                    f"正在整理您的请求（第 {step + 1}/{max_steps} 步）"
                    if hint_now
                    else f"正在处理您的请求（第 {step + 1}/{max_steps} 步）"
                ),
                step=step + 1,
            )
            final = self._step(ctx, state)
            if final is not None:
                return final

        # 步数耗尽：强制选择 final_response（只挂终止工具 + 显式 tool_choice）
        ctx.forced_finalize = True
        return self._forced_finalize(ctx)

    # ---------- 单步 ----------
    def _step(self, ctx: AgentTurnContext, state: ToolTurnState):
        agent = self._agent
        t0 = time.monotonic()
        messages = agent.context_builder.build(agent, ctx.current_query)
        ctx.context_tokens = agent.context_builder.last_context_tokens
        ctx.timings.context_build_ms += (time.monotonic() - t0) * 1000

        tools = list(agent.tool_manager.tool_definitions) + [
            FINAL_RESPONSE_TOOL_DEFINITION,
        ]
        # temperature/max_tokens 是「调用方偏好值」：是否真的发给厂商由
        # llm/client.py 的画像层（T1）决定（推理模型删 temperature、改参数名、
        # 抬下限）。调用点保持直传，避免改写面重新摊到 17 个调用点。
        response = agent.client.chat.completions.create(
            model=agent.model,
            messages=messages,
            temperature=agent.temperature,
            tools=tools,
            max_tokens=agent.llm_max_tokens,
        )
        ctx.llm_calls += 1
        choice = response.choices[0]
        assistant_msg = choice.message
        captured = capture_reasoning(assistant_msg)
        self._emit_reasoning(captured)
        tool_calls = list(assistant_msg.tool_calls or [])

        # 纯文本：未走终止协议 → 协议纠错并重试（Review 修复：不再视为终答）
        if not tool_calls:
            self._push_window_message(
                *self._assistant_message(assistant_msg, captured),
            )
            self._push_window_message({
                "role": "system", "content": PLAIN_TEXT_CORRECTION,
            })
            from app.observability.metrics import record_final_protocol_correction

            ctx.protocol_corrections += 1
            record_final_protocol_correction()
            return None

        # 拆分业务工具与终止工具
        business = [tc for tc in tool_calls
                    if tc.function.name != FINAL_RESPONSE_TOOL_NAME]
        finals = [tc for tc in tool_calls
                  if tc.function.name == FINAL_RESPONSE_TOOL_NAME]

        # 只写一条 assistant tool-call 消息（全部 calls，保持 API 消息序合法）
        self._append_assistant_tool_calls(assistant_msg, tool_calls, captured)

        if business:
            self._execute_business(ctx, state, business)
        if business or len(finals) > 1:
            # 每个终止调用恰好一个结果：混合 → PREMATURE；多个 final → 协议错误
            for tc in finals:
                error = _FINAL_PREMATURE_ERROR if business else _FINAL_MULTIPLE_ERROR
                self._append_tool_message(tc.id, error, ctx)
            return None

        # 唯一 final call：校验；合法 → 记录接受结果并结束
        tc = finals[0]
        args, error = validate_final_response(tc.function.arguments)
        if error is not None:
            self._append_tool_message(tc.id, error, ctx)
            return None
        self._append_tool_message(tc.id, FINAL_ACCEPTED_RESULT, ctx)
        return args

    # ---------- 强制终答 ----------
    def _forced_finalize(self, ctx: AgentTurnContext) -> FinalResponseArgs:
        agent = self._agent
        if ctx.budget.expired():
            raise LLMBudgetExhausted("轮次预算耗尽，强制收尾被拒绝")
        profile = effective_profile(agent.model)
        soft = not profile.supports_forced_tool_choice
        messages = agent.context_builder.build(agent, ctx.current_query)
        ctx.context_tokens = agent.context_builder.last_context_tokens
        kwargs: dict = {
            "model": agent.model,
            "tools": [FINAL_RESPONSE_TOOL_DEFINITION],
            "temperature": agent.temperature,
            "max_tokens": agent.llm_max_tokens,
        }
        if soft:
            # T4：画像不支持强制 tool_choice（探针第 2 项）→ 达成手段改为
            # 「只挂终止工具 + system 软强制」。兜底语义不变：仍纯文本即
            # ForcedFinalizeFailed（上层确定性转人工）。提示同时进窗口与审计，
            # 保证审计切片能解释这一轮的收尾方式。
            hint = {"role": "system", "content": FORCED_FINALIZE_SOFT_PROMPT}
            self._push_window_message(hint)
            messages = [*messages, hint]
        else:
            kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": FINAL_RESPONSE_TOOL_NAME},
            }
        from app.observability.metrics import (
            record_forced_finalize_attempt,
            record_forced_finalize_fallback,
        )

        mode = "soft" if soft else "hard"
        record_forced_finalize_attempt(mode)
        response = agent.client.chat.completions.create(messages=messages, **kwargs)
        ctx.llm_calls += 1
        assistant_msg = response.choices[0].message
        captured = capture_reasoning(assistant_msg)
        self._emit_reasoning(captured)
        tool_calls = list(assistant_msg.tool_calls or [])
        if not tool_calls:
            # 未按协议输出 → 强制收尾失败（确定性 fallback）
            record_forced_finalize_fallback(mode)
            raise ForcedFinalizeFailed("强制终答未返回 final_response 调用")
        tc = tool_calls[0]
        args, error = validate_final_response(tc.function.arguments)
        if error is not None:
            record_forced_finalize_fallback(mode)
            raise ForcedFinalizeFailed(f"强制终答参数非法: {error}")
        # 审计消息序完整：assistant(tool_calls) + 恰好一个接受结果
        self._append_assistant_tool_calls(assistant_msg, [tc], captured)
        self._append_tool_message(tc.id, FINAL_ACCEPTED_RESULT, ctx)
        return args


    # ---------- 工具执行与轨迹 ----------
    def _execute_business(self, ctx: AgentTurnContext, state: ToolTurnState,
                          business_calls) -> None:
        agent = self._agent
        outcomes = agent.tool_executor.execute(
            [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in business_calls
            ],
            state, agent.ctx, agent.tool_manager, budget=ctx.budget,
        )
        for oc in outcomes:
            display = oc.result if len(oc.result) <= 300 else oc.result[:300] + "..."
            agent.log_tool_result(oc.name, oc.sequence, display)
            self._append_tool_message(oc.call_id, oc.result, ctx)
            ctx.tool_trace.append(self._trace_entry(ctx, oc))

    def _trace_entry(self, ctx: AgentTurnContext, oc) -> ToolTraceEntry:
        """ToolOutcome → 轮次轨迹（含稳定结果码与副作用状态）。"""
        payload: dict = {}
        try:
            parsed = json.loads(oc.result)
            if isinstance(parsed, dict):
                payload = parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            payload = {}
        has_error = "error" in payload or payload.get("success") is False
        ok = not has_error
        code = str(payload.get("code", "") or ("ERROR" if has_error else "OK"))
        side_effect = "none"
        if oc.name in ("submit_refund_application", "cancel_refund_application"):
            status = str(payload.get("status") or "")
            if status == "indeterminate":
                side_effect = "indeterminate"
            elif status in (
                "merchant_reviewing", "approved", "rejected",
                "refund_processing", "refunded", "withdrawn",
            ) or payload.get("replayed"):
                side_effect = "submitted"
        # 写状态机观察（归属验证/申请结果推进）；工具结果载荷自带
        # client_request_id（工具层自生成并回写），indeterminate 对账锚点由此承载
        if payload:
            ctx.write_ops.observe(oc.name, dict(oc.arguments), payload)
        # 检索证据（声明级接地用；完整结果只进审计）
        if oc.name == "search_knowledge" and ok:
            for item in payload.get("results", []) or []:
                if isinstance(item, dict) and not item.get("tainted"):
                    text = str(item.get("text") or item.get("matched_text") or "")
                    if text:
                        ctx.evidence_texts.append(text)
        elif ok and payload:
            ctx.evidence_texts.append(oc.result[:2000])
        return ToolTraceEntry(
            call_id=oc.call_id,
            name=oc.name,
            arguments=oc.arguments,
            result=oc.result,
            digest=digest_tool_result(oc.name, oc.result),
            ok=ok,
            code=code,
            side_effect_status=side_effect,
            skipped=oc.skipped,
            sequence=oc.sequence,
        )

    # ---------- 消息窗口 ----------
    def _push_window_message(self, msg: dict, audit_msg: dict | None = None) -> None:
        """同时进模型窗口（raw_messages）与轮次审计切片（full_turn_messages）。

        T3 双通道：窗口消息与审计消息可以**不同内容**（窗口按画像剥离
        reasoning，审计附加 reasoning）。`audit_msg` 缺省 = 两通道同一对象，
        即非推理路径的现状行为。
        """
        self._agent.raw_messages.append(msg)
        self._ctx.full_turn_messages.append(msg if audit_msg is None else audit_msg)

    def _assistant_message(self, assistant_msg,
                           captured: CapturedReasoning | None,
                           tool_calls=None) -> tuple[dict, dict]:
        """assistant 消息的双通道形态：返回 (窗口消息, 审计消息)。

        - 窗口：content 按画像取值（`required_signed` 原样回传 thinking block，
          其余压平为文本）+ 仅在 `required_signed` 时回填 reasoning 字段；
        - 审计：附加 `reasoning` 字段（additive），供 turns archive 对账。
        """
        profile = effective_profile(self._agent.model)
        content = window_content(assistant_msg, profile, captured)
        window: dict = {"role": "assistant", "content": content}
        window.update(window_extra_fields(profile, captured))
        if tool_calls is not None:
            window["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        return window, merge_audit(window, captured)

    def _emit_reasoning(self, captured: CapturedReasoning | None) -> None:
        """T6：reasoning 透出（开关 + guardrails 都在 chat.emit_reasoning 内）。"""
        if captured is None:
            return
        emit = getattr(self._agent, "emit_reasoning", None)
        if emit is None:
            return
        emit(captured.sse_text(), step=self._ctx.react_steps)

    def _append_assistant_tool_calls(self, assistant_msg, tool_calls,
                                     captured: CapturedReasoning | None = None) -> None:
        if captured is None:
            captured = capture_reasoning(assistant_msg)
        self._push_window_message(
            *self._assistant_message(assistant_msg, captured, tool_calls),
        )

    def _append_tool_message(self, call_id: str, result: str,
                             ctx: AgentTurnContext) -> None:
        self._push_window_message(
            {"role": "tool", "tool_call_id": call_id, "content": result},
        )

