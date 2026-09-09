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
- 达到最大步数 → 最后一轮只挂 final_response 且显式 tool_choice 强制收尾；
  仍失败 → ForcedFinalizeFailed（上层确定性转人工 fallback）；
- 预算耗尽 → LLMBudgetExhausted（上层确定性 fallback，零 LLM）；
- 工具批次经 ToolBatchExecutor（原序分段并行 + 守卫 + 写状态机拦截 + 退款
  确认注入）；SSE `thought` 事件为受控状态说明，不透出模型原始思考。
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

FINAL_ACCEPTED_RESULT = json.dumps(
    {"success": True, "accepted": True, "code": "FINAL_RESPONSE_ACCEPTED"},
    ensure_ascii=False,
)
PLAIN_TEXT_CORRECTION = (
    "你上一步输出了纯文本而没有调用 final_response 工具。"
    "请基于你上面的答复内容，立即调用 final_response 工具提交最终答复"
    "（intent/reply/requires_human/follow_up_question）；不要再输出纯文本。"
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
        for step in range(max_steps):
            ctx.react_steps = step + 1
            if ctx.budget.expired():
                raise LLMBudgetExhausted("轮次预算耗尽，停止 ReAct 循环")
            # 受控状态说明（SSE thought 事件：不再透出模型原始思考）
            agent.emit_status(
                f"正在处理您的请求（第 {step + 1}/{max_steps} 步）", step=step + 1,
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
        tool_calls = list(assistant_msg.tool_calls or [])

        # 纯文本：未走终止协议 → 协议纠错并重试（Review 修复：不再视为终答）
        if not tool_calls:
            content = assistant_msg.content or ""
            self._push_window_message({"role": "assistant", "content": content})
            self._push_window_message({
                "role": "system", "content": PLAIN_TEXT_CORRECTION,
            })
            from app.observability.metrics import record_final_protocol_correction

            record_final_protocol_correction()
            return None

        # 拆分业务工具与终止工具
        business = [tc for tc in tool_calls
                    if tc.function.name != FINAL_RESPONSE_TOOL_NAME]
        finals = [tc for tc in tool_calls
                  if tc.function.name == FINAL_RESPONSE_TOOL_NAME]

        # 只写一条 assistant tool-call 消息（全部 calls，保持 API 消息序合法）
        self._append_assistant_tool_calls(assistant_msg, tool_calls)

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
        messages = agent.context_builder.build(agent, ctx.current_query)
        ctx.context_tokens = agent.context_builder.last_context_tokens
        response = agent.client.chat.completions.create(
            model=agent.model,
            messages=messages,
            temperature=agent.temperature,
            tools=[FINAL_RESPONSE_TOOL_DEFINITION],
            tool_choice={
                "type": "function",
                "function": {"name": FINAL_RESPONSE_TOOL_NAME},
            },
            max_tokens=agent.llm_max_tokens,
        )
        ctx.llm_calls += 1
        assistant_msg = response.choices[0].message
        tool_calls = list(assistant_msg.tool_calls or [])
        if not tool_calls:
            # 未按协议输出 → 强制收尾失败（确定性 fallback）
            raise ForcedFinalizeFailed("强制终答未返回 final_response 调用")
        tc = tool_calls[0]
        args, error = validate_final_response(tc.function.arguments)
        if error is not None:
            raise ForcedFinalizeFailed(f"强制终答参数非法: {error}")
        # 审计消息序完整：assistant(tool_calls) + 恰好一个接受结果
        self._append_assistant_tool_calls(assistant_msg, [tc])
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
        if oc.name == "apply_refund":
            if payload.get("status") == "indeterminate":
                side_effect = "indeterminate"
            elif payload.get("confirmed") or payload.get("replayed"):
                side_effect = "committed"
            elif payload.get("status") == "pending_confirmation":
                side_effect = "pending"
        # 写状态机观察（归属验证/退款推进）；合并 internal_args：
        # confirm 段的 confirmation_token/idempotency_key 在服务端注入通道，
        # 不合并则 tokens_issued 无法回收、indeterminate 条目幂等键恒为空
        if payload:
            ctx.write_ops.observe(
                oc.name, {**oc.arguments, **(oc.internal_args or {})}, payload,
            )
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
    def _push_window_message(self, msg: dict) -> None:
        """同时进模型窗口（raw_messages）与轮次审计切片（full_turn_messages）。"""
        self._agent.raw_messages.append(msg)
        self._ctx.full_turn_messages.append(msg)

    def _append_assistant_tool_calls(self, assistant_msg, tool_calls) -> None:
        msg_dict = {"role": "assistant", "content": assistant_msg.content or ""}
        msg_dict["tool_calls"] = [
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
        self._push_window_message(msg_dict)

    def _append_tool_message(self, call_id: str, result: str,
                             ctx: AgentTurnContext) -> None:
        self._push_window_message(
            {"role": "tool", "tool_call_id": call_id, "content": result},
        )
