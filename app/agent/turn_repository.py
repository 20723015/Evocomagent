"""TurnRepository（单 Agent 全量优化计划·阶段A/B）：轮次持久化唯一出口。

阶段B约定：
- 每轮历史只保存一个 assistant 消息：content = 用户可见回复，结构化字段
  （intent/confidence/requires_human/follow_up/tools/...）
  放入消息 metadata；本轮中间的 assistant(tool_calls) 与 tool 消息在收尾
  时从模型窗口折叠移除——完整结果只进入审计存储（append_log/SQL 行式
  正本 + evolution turn 记录）；
- 读取旧会话时旧格式在 ContextBuilder 折叠，正本不修改；
- 历史压缩按 token 水位触发（ContextBuilder.history_overflow）；
- SQL 模式下消息保存与 memory job 入队同一事务（save 透传 memory_job_seq）。
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.agent.context_builder import METADATA_KEY
from app.agent.turn_context import AgentTurnContext
from app.observability.logging import get_logger

log = get_logger("app.agent.turn_repository")

# 新格式 assistant 消息 metadata 的 schema 标记
METADATA_SCHEMA = 2


class TurnRepository:
    """会话/审计/演进记录的持久化门面（请求级使用，无内部请求状态）。"""

    def __init__(self, agent):
        self._agent = agent

    # ---------- 主入口 ----------
    def begin_turn(self, ctx: AgentTurnContext) -> None:
        """轮次开始：把本轮 user 消息先落库，并置 pending_turn 草稿标记。

        与消息同一 save 事务（SQL 同事务；文件/Redis 同一次整包写）。
        **best-effort**：失败只告警——提交阶段仍会整轮落库，草稿标记只是
        增强（丢了不丢消息），不得因它中断用户请求。

        草稿在轮次开始（ReAct 之前）置位而非收尾阶段：崩溃窗口最大的一段
        是 ReAct 循环本身，收尾时才置位等于只覆盖收尾窗口。
        """
        agent = self._agent
        try:
            agent.flush_session(pending_turn={
                "turn_id": ctx.turn_id,
                "user_message": ctx.sanitized_input or ctx.user_input,
                "started_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
            })
        except Exception:  # noqa: BLE001 —— 草稿失败不影响本轮
            log.warning("turn.begin_persist_failed", exc_info=True)

    def commit_turn(self, ctx: AgentTurnContext,
                    result) -> None:
        """收尾步骤6：折叠本轮消息 → 单条 assistant 消息 → 保存。

        保存同时清除 pending_turn（收尾成功 = 本轮已完整落库）。
        """
        agent = self._agent
        # 1. 折叠本轮中间消息，追加最终 assistant 消息（新格式）
        self._collapse_turn(ctx, result)
        # 2. 审计切片（含工具完整结果）进 append_log（SQL 正本逐行落库）
        agent.append_audit_messages(ctx.full_turn_messages)
        # 3. token 水位压缩（辅助 LLM；预算耗尽跳过，不毁本轮结果）
        self._compress_if_overflow(ctx)
        # 4. 保存会话（SQL 同事务入队 memory job；清 pending_turn 草稿）
        self._save_session(ctx)
        # 5. 演进 turn 记录（脱敏；失败不影响主流程）
        self._record_turn(ctx, result)

    # ---------- 折叠 ----------
    def _collapse_turn(self, ctx: AgentTurnContext, result) -> None:
        agent = self._agent
        messages = agent.raw_messages
        start = ctx.slice_start
        # 保留本轮 user 消息；清掉中间 assistant(tool_calls)/tool 消息
        if 0 <= start < len(messages):
            messages[start + 1:] = []
        # 工具轨迹 metadata（digest；完整结果只在审计存储）
        tools_meta = [
            {
                "name": t.name,
                "ok": t.ok,
                "code": t.code,
                "side_effect_status": t.side_effect_status,
                "digest": t.digest,
            }
            for t in ctx.tool_trace
        ]
        metadata: dict = {
            "schema": METADATA_SCHEMA,
            "turn_id": ctx.turn_id,
            "intent": result.intent.value if hasattr(result.intent, "value") else str(result.intent),
            "confidence": result.confidence,
            "requires_human": result.requires_human,
            "follow_up_question": result.follow_up_question,
            "tools": tools_meta,
        }
        if ctx.sources:
            metadata["sources"] = sorted(ctx.sources)
        if ctx.indeterminate_writes:
            metadata["indeterminate_writes"] = ctx.indeterminate_writes
        final_message = {
            "role": "assistant",
            "content": result.reply,
            METADATA_KEY: metadata,
        }
        messages.append(final_message)
        # Review 修复：最终 assistant 同时加入审计增量——SQL/outbox/重载
        # 历史均包含最终答复
        ctx.full_turn_messages.append(final_message)

    # ---------- 辅助任务 ----------
    def _compress_if_overflow(self, ctx: AgentTurnContext) -> None:
        agent = self._agent
        try:
            overflow, folded = agent.context_builder.history_overflow(agent.raw_messages)
            if not overflow:
                return
            agent.compress_history_by_tokens(folded)
            ctx.context_truncated = True
            ctx.truncation_stage = "history"
        except Exception as e:
            from app.agent.turn_budget import LLMBudgetExhausted

            if isinstance(e, LLMBudgetExhausted):
                log.info("aux.summary_skipped 轮次预算耗尽，历史摘要跳过")
            else:
                log.warning("history.compress_failed", exc_info=True)

    def _save_session(self, ctx: AgentTurnContext) -> None:
        agent = self._agent
        # 文件 memory queue 使用轮次稳定 ID 做幂等键；SQL store 仍以其
        # append-only seq/事务唯一键为正本语义。
        # pending_turn 显式传 None：收尾成功即清草稿标记。
        agent.flush_session(
            enqueue_memory_job=True,
            turn_messages=ctx.full_turn_messages,
            turn_id=ctx.turn_id,
            pending_turn=None,
        )

    def _record_turn(self, ctx: AgentTurnContext, result) -> None:
        agent = self._agent
        recorder = agent.turn_recorder
        if recorder is None:
            return
        try:
            recorder.record(
                session_id=agent.session_id,
                mode="single",
                question=agent.current_turn_query or "",
                structured_reply=result,
                turn_slice=ctx.full_turn_messages,  # 审计：完整轮次切片
                user_id=agent.user_id,
                usage=ctx.to_usage_dict(),
            )
        except TypeError:
            # 旧 TurnRecorder 签名无 usage
            recorder.record(
                session_id=agent.session_id,
                mode="single",
                question=agent.current_turn_query or "",
                structured_reply=result,
                turn_slice=ctx.full_turn_messages,
                user_id=agent.user_id,
            )
        except Exception:
            log.warning("turn.record_failed", exc_info=True)
