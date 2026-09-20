import uuid
from collections.abc import Callable
from pathlib import Path

from openai import OpenAI

from app.agent.context import ToolContext
from app.agent.context_builder import ContextBuilder
from app.agent.input_policy import evaluate_input
from app.agent.react_runner import (
    PLAIN_TEXT_CORRECTION,
    ForcedFinalizeFailed,
    ReactRunner,
)
from app.agent.tools.batch_executor import ToolBatchExecutor, ToolTurnState
from app.agent.tools.manager import ToolManager
from app.agent.turn_budget import (
    LLMBudgetExhausted,
    TurnBudget,
    bind_budget,
    reset_budget,
)
from app.agent.turn_context import AgentTurnContext
from app.agent.turn_finalizer import (
    TurnFinalizer,
)
from app.agent.turn_repository import TurnRepository
from app.agent import write_gate
from app.agent.write_ops import DRAFT_CANCELLED_STATUS, WriteOpTracker
from app.config.settings import settings
from app.observability.logging import get_logger
from app.observability.metrics import record_budget_exhausted
from app.schemas.response import CustomerServiceResponse
from app.security.scope_gate import SCOPE_BLOCK_REPLY
from app.stores.base import (
    SessionOwnershipError,
    SessionState,
    SessionStore,
)
from app.stores.session_store import LocalFileSessionStore

log = get_logger("app.agent.chat")

# flush_session 的哨兵默认值：未显式传 pending_write 时保持会话现值
# （与「显式传 None = 作废草稿」区分开）
_KEEP_PENDING_WRITE = object()

# 预算耗尽的确定性 fallback 话术（零 LLM 收尾）
BUDGET_FALLBACK_REPLY = "很抱歉，本轮处理时间已达上限，已为您转接人工客服，请稍候。"


def _memory_job_payload(folded: list[dict]) -> list[dict]:
    """memory job 巩固负载：模型可见的 user + 最终 assistant 消息。

    工具中间消息不进负载；纯文本协议纠错的 assistant 稿（react_runner 在其后
    成对压入 PLAIN_TEXT_CORRECTION system 消息）是中间纠错文本而非终答，
    带前瞻跳过——不改变模型可见消息结构，只过滤巩固负载（低危修复 A4）。
    """
    correction_starts = {
        i for i, m in enumerate(folded)
        if m.get("role") == "system" and m.get("content") == PLAIN_TEXT_CORRECTION
    }
    return [
        m for i, m in enumerate(folded)
        if m.get("role") in ("user", "assistant")
        and "tool_calls" not in m
        and str(m.get("content") or "").strip()
        and not (m.get("role") == "assistant" and i + 1 in correction_starts)
    ]


class EcomAgent:
    """电商客服 Agent —— 单 Agent 全量优化后的统一执行流水线。

    输入策略 → 上下文构建 → ReAct/工具执行（final_response 终止协议）→
    结构化终答 → 安全与事实校验（TurnFinalizer 固定顺序）→ 持久化
    （TurnRepository）→ 异步记忆（memory job 增量巩固 LTM）。

    Agent 是「无状态算子」：构造 → 处理一轮 → 写回 → 丢弃。
    user_id/session_id 为请求级参数，pod 级资源（client/skill_manager/
    mcp_client）注入共享；轮次状态统一在 AgentTurnContext，实例不再持有
    请求级临时字段。
    """

    def __init__(
        self,
        user_id: str = "default",
        session_id: str | None = None,
        session_path: str | None = None,
        client: OpenAI | None = None,
        skill_manager=None,
        mcp_client=None,
        tool_manager: ToolManager | None = None,
        memory_enabled: bool | None = None,
        use_mcp: bool | None = None,
        temperature: float | None = None,
        model: str | None = None,
        session_store: SessionStore | None = None,
        ltm_store=None,
        consolidate_every: int | None = None,
        turns_archive=None,
        credentials: dict | None = None,
        event_callback=None,
        tool_executor: ToolBatchExecutor | None = None,
        turn_state_factory=None,
        enforce_order_ownership: bool | None = None,
    ):
        self.user_id = user_id
        self.client = client or self._build_default_client()
        self.model = model or settings.model_name
        self.temperature = (
            settings.temperature if temperature is None else temperature
        )
        self.session_path = session_path or self._derive_session_path(user_id, session_id)
        self.session_store = session_store or LocalFileSessionStore(
            settings.session_dir,
            exact_path=session_path,
        )
        self._state_version = 0
        # 兼容保留：历史压缩水位已改为 token（ContextBuilder）；增量巩固由
        # memory job 承担（close()/轮内不再做 LLM 巩固）
        self._consolidate_every = (
            settings.memory_consolidate_every if consolidate_every is None else consolidate_every
        )
        self._consolidated_len = 0
        self.history_threshold = settings.history_threshold
        self.history_keep_recent = settings.history_keep_recent
        self.max_react_steps = settings.max_react_steps
        self.llm_max_tokens = settings.llm_max_tokens

        self.tool_manager = tool_manager or ToolManager(
            use_mcp=settings.mcp_enabled if use_mcp is None else use_mcp,
            mcp_server_url=settings.mcp_server_url,
            mcp_client=mcp_client,
        )

        from app.agent.memory import MemoryManager

        self.memory_manager = MemoryManager(
            client=self.client,
            model=self.model,
            user_id=user_id,
            memory_dir=settings.memory_dir,
            memory_enabled=settings.memory_enabled if memory_enabled is None else memory_enabled,
            max_ltm_facts=settings.max_ltm_facts,
            ltm_store=ltm_store,
        )

        from app.agent.skills import SkillManager

        self.skill_manager = skill_manager or SkillManager(
            skills_dir=settings.skills_dir,
            enabled=settings.skills_enabled,
        )

        self.raw_messages: list[dict] = []
        self.summary: str | None = None
        # 全量追加日志：压缩只裁剪模型窗口，append_log 永续（SQL 正本据它
        # 行式追加，完整工具结果只进审计存储）
        self._append_log: list[dict] = []
        self._append_flushed = 0
        self.event_callback = event_callback
        self._react_steps_count = 0

        self._owns_tool_executor = tool_executor is None
        self.tool_executor = tool_executor or ToolBatchExecutor()
        self._turn_state_factory = turn_state_factory
        # 写结果未知清单（indeterminate → handoff 对账）；finalize 后随 ctx 刷新
        self._indeterminate_writes: list[dict] = []

        # 流水线组件（阶段A）：上下文构建 / ReAct / 收尾 / 持久化
        self.context_builder = ContextBuilder(
            memory_manager=self.memory_manager,
            skill_manager=self.skill_manager,
            context_window_tokens=settings.context_window_tokens,
        )
        self._react_runner = ReactRunner(self)
        self._finalizer = TurnFinalizer(TurnRepository(self))

        self.current_turn_query = ""
        self._last_turn_ctx = None  # 最近一轮 AgentTurnContext（观测/测试用）

        self._session_key = session_id or ""
        loaded = self.session_store.load(user_id, self._session_key)
        if loaded is not None and loaded.user_id and loaded.user_id != user_id:
            raise SessionOwnershipError(
                f"session {user_id}/{self._session_key} 属于用户 {loaded.user_id}"
            )
        self.session_id = loaded.session_id if loaded else (
            session_id or uuid.uuid4().hex
        )
        self.memory_manager.bind_session(self.session_id)
        # 掉线恢复（Step6）：加载时带回上次遗留的草稿标记（非空 = 上次回复
        # 未完成）；API 层据此提示客户端重发。本轮收尾会清除它。
        self.pending_turn = loaded.pending_turn if loaded else None
        # 写确认两阶段（P1-2）：待确认写草稿随会话持久化，重启/崩溃后仍有效
        self.pending_write = loaded.pending_write if loaded else None
        if loaded:
            self._state_version = loaded.version
            self.summary = loaded.summary
            self.raw_messages = loaded.messages
            self._append_log = list(loaded.messages)
            self._append_flushed = len(self._append_log)
            self._consolidated_len = loaded.consolidated_len

        # 修复计划·一：会话租约校验回调（路由层持有 SessionLease 时注入）
        self._lease_guard: Callable[[], None] | None = None
        self.ctx = ToolContext(
            user_id=user_id,
            session_id=self.session_id,
            memory=self.memory_manager,
            skill_manager=self.skill_manager,
            credentials=credentials,
            enforce_order_ownership=enforce_order_ownership,
            pending_write=self.pending_write,
            persist_pending_write=self._persist_pending_write,
        )

        if settings.evolve_capture_enabled:
            from app.evolution.recorder import TurnRecorder

            self._turn_recorder = TurnRecorder(
                Path(settings.evolve_turns_dir), archive=turns_archive,
            )
        else:
            self._turn_recorder = None

    def bind_lease_guard(self, guard: Callable[[], None] | None) -> None:
        """绑定会话租约校验回调：save/reset 与写工具提交前都会调用。

        回调失效（SessionLockLost）时抛出，调用方据此拒绝提交副作用/会话状态。
        """
        self._lease_guard = guard
        if getattr(self, "ctx", None) is not None:
            self.ctx.lease_guard = guard

    def _assert_lease(self) -> None:
        if self._lease_guard is not None:
            self._lease_guard()

    # ============================================================
    # 构造辅助
    # ============================================================
    @staticmethod
    def _build_default_client() -> OpenAI:
        return OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )

    @staticmethod
    def _derive_session_path(user_id: str, session_id: str | None) -> str:
        return str(Path(settings.session_dir) / user_id / f"{session_id or 'session'}.json")

    @property
    def turn_recorder(self):
        return self._turn_recorder

    @property
    def history_size(self) -> int:
        return len(self.raw_messages)

    # ============================================================
    # 主流程
    # ============================================================
    def chat(self, user_input: str) -> CustomerServiceResponse:
        """处理一轮：InputPolicy → ReAct(final_response) → TurnFinalizer。"""
        budget = TurnBudget.start(settings.turn_budget_seconds)
        ctx = AgentTurnContext(user_input=user_input, budget=budget)
        ctx.sanitized_input = user_input
        ctx.current_query = user_input
        token = bind_budget(budget)
        try:
            from app.observability.tracing import span

            with span("agent.turn", turn_id=ctx.turn_id):
                return self._chat_pipeline(ctx)
        finally:
            # 所有返回路径（含护栏/范围/预算兜底）都刷新观测上下文：
            # 评估沙箱逐轮读 _last_turn_ctx，只有成功路径赋值会让被拦截轮
            # 读到上一轮的陈旧 ctx（步数翻倍、verdict 串轮）
            self._last_turn_ctx = ctx
            reset_budget(token)

    def _chat_pipeline(self, ctx: AgentTurnContext) -> CustomerServiceResponse:
        state = (
            self._turn_state_factory()
            if self._turn_state_factory is not None
            else ToolTurnState(event_callback=self.event_callback)
        )
        # 写状态机唯一实例：执行器拦截与终答校验共用（ctx 与 state 同源）
        ctx.write_ops = state.write_tracker()
        self._react_steps_count = 0
        self.current_turn_query = ctx.user_input

        # —— 输入策略（唯一实现；拦截走规则响应，零 LLM）——
        try:
            decision = evaluate_input(
                ctx.user_input, self.ctx, client=self.client, model=self.model,
            )
        except LLMBudgetExhausted:
            # 预算耗尽连 scope 判定都拒绝 → 直接确定性 fallback
            record_budget_exhausted("input")
            ctx.budget_fallback = True
            self._open_turn_window(ctx, ctx.user_input)
            return self._finalizer.finalize_rule_response(
                ctx, BUDGET_FALLBACK_REPLY, requires_human=True,
                handoff_reason="budget_exhausted",
            )
        ctx.input_action = decision.action
        ctx.input_reason = decision.reason
        if decision.action == "guardrail_block":
            log.warning("guardrail.block user=%s reason=%s", self.user_id, decision.reason)
            self._open_turn_window(ctx, ctx.user_input)
            return self._finalizer.finalize_rule_response(
                ctx, decision.reply, requires_human=True,
                handoff_reason="guardrail_block",
            )
        if decision.action == "scope_block":
            log.info("scope.block user=%s reason=%s", self.user_id, decision.reason)
            self._open_turn_window(ctx, ctx.user_input)
            return self._finalizer.finalize_rule_response(
                ctx, SCOPE_BLOCK_REPLY, requires_human=False,
            )
        ctx.sanitized_input = decision.text
        ctx.current_query = decision.text
        self.current_turn_query = decision.text

        # —— 写确认两阶段协议（P1-2）：判定本轮消息对草稿的表态 ——
        self._resolve_pending_write(ctx.sanitized_input, write_ops=ctx.write_ops)

        # —— 用户消息进窗口（审计切片同步记录）——
        self._open_turn_window(ctx, ctx.sanitized_input)

        # —— ReAct 循环（final_response 终止协议）——
        try:
            final = self._react_runner.run(ctx, state)
        except LLMBudgetExhausted:
            record_budget_exhausted("react")
            ctx.budget_fallback = True
            # 中危修复 B6：写超时等 indeterminate 记录可能已被截断在 state 里，
            # fallback 收尾同样要落账（工单对账建议依赖 _indeterminate_writes）
            ctx.extra_indeterminate = list(state.indeterminate_writes)
            ctx.sources = set(state.sources)
            result = self._finalizer.finalize_rule_response(
                ctx, BUDGET_FALLBACK_REPLY, requires_human=True,
                handoff_reason="budget_exhausted",
            )
            self._indeterminate_writes = list(ctx.indeterminate_writes)
            return result
        except ForcedFinalizeFailed as e:
            log.warning("react.forced_finalize_failed user=%s err=%s", self.user_id, e)
            ctx.forced_finalize_failed = True
            ctx.budget_fallback = True  # 无法生成有效终答 → 可靠度 0.0
            ctx.extra_indeterminate = list(state.indeterminate_writes)
            ctx.sources = set(state.sources)
            from app.observability.metrics import record_turn_missing_final

            record_turn_missing_final()
            result = self._finalizer.finalize_rule_response(
                ctx, BUDGET_FALLBACK_REPLY, requires_human=True,
                handoff_reason="forced_finalize_failed",
            )
            self._indeterminate_writes = list(ctx.indeterminate_writes)
            return result
        if ctx.budget.expired():
            record_budget_exhausted("react")
            ctx.budget_fallback = True
            ctx.extra_indeterminate = list(state.indeterminate_writes)
            ctx.sources = set(state.sources)
            from app.observability.metrics import record_turn_missing_final

            record_turn_missing_final()
            result = self._finalizer.finalize_rule_response(
                ctx, BUDGET_FALLBACK_REPLY, requires_human=True,
                handoff_reason="budget_exhausted",
            )
            self._indeterminate_writes = list(ctx.indeterminate_writes)
            return result

        ctx.final_args = final  # Review 修复：唯一成功路径 = final_response 协议
        ctx.extra_indeterminate = list(state.indeterminate_writes)
        ctx.sources = set(state.sources)
        ctx.timings.react_ms = ctx.elapsed_ms()
        self._react_steps_count = ctx.react_steps

        # —— 指标（每轮工具数/步数；Phase G 补 token/工具码）——
        from app.observability.metrics import (
            record_react_steps,
            record_tool_calls_per_turn,
            record_turn_usage,
        )

        record_tool_calls_per_turn(sum(state.per_name_counts.values()))
        record_react_steps(ctx.react_steps)
        record_turn_usage(ctx, state)

        # —— 收尾管线（固定顺序 1-7）——
        result = self._finalizer.finalize(ctx)
        self._indeterminate_writes = list(ctx.indeterminate_writes)
        return result

    # ============================================================
    # 流水线支撑（供 ReactRunner/TurnRepository 调用）
    # ============================================================
    def _resolve_pending_write(
        self, user_text: str, write_ops: WriteOpTracker | None = None,
    ) -> None:
        """写确认两阶段协议（P1-2）：判定本轮消息对草稿的表态并注入工具上下文。

        判定在**工具执行之前**完成（ReAct 之前），工具只读结论——写与不写的
        决定权在服务端，不在模型。语义：
        - confirm → 工具可复用草稿幂等键真正提交；
        - cancel → 立即作废草稿（不依赖模型是否调用工具）；
        - ambiguous/none → 草稿保留，本轮禁止任何写落库。
        超期草稿在此作废（避免陈旧草稿被后来一句无关的确认词触发）。

        **每轮复位**：`write_executed` 是「本轮已执行过写」的标记，必须与
        `write_confirm` 一样每轮重算——否则上一轮确认提交后置位的 True 会
        残留到后续轮次，把新一轮的草稿登记误判成「同轮重复提交」。
        """
        # 轮边界：三个写状态一并复位（write_confirm 本来就在下面每条分支重算，
        # 这里统一置位以免漏项）
        self.ctx.write_executed = False
        draft = self.pending_write
        if not draft:
            self.ctx.pending_write = None
            self.ctx.write_confirm = "none"
            return
        if write_gate.is_expired(draft):
            log.info("write.draft_expired user=%s", self.user_id)
            self._persist_pending_write(None)
            self.ctx.pending_write = None
            self.ctx.write_confirm = "none"
            return

        decision = write_gate.judge_write_confirmation(
            user_text, write_gate.pending_of(draft),
        )
        self.ctx.pending_write = draft
        self.ctx.write_confirm = decision.action
        if decision.action == "cancel":
            # 取消即作废：不依赖模型是否调用写工具
            log.info("write.draft_cancelled user=%s", self.user_id)
            self._persist_pending_write(None)
            self.ctx.pending_write = None
            if write_ops is not None:
                # 服务端已确知取消事实，直接注入守卫可见的状态位：取消轮模型
                # 可能只调读工具（终轮实测路径），observe 永远等不到取消回执。
                write_ops.business_statuses.add(DRAFT_CANCELLED_STATUS)
            return
        if decision.action == "confirm":
            self._persist_pending_write(write_gate.mark_confirmed(draft))

    def _persist_pending_write(self, draft: dict | None) -> None:
        """登记/清除待确认写草稿（与消息同一 save 事务落库，崩溃可恢复）。"""
        self.pending_write = draft
        self.ctx.pending_write = draft
        self.flush_session(
            pending_turn=self.pending_turn, pending_write=draft,
        )

    def _open_turn_window(self, ctx: AgentTurnContext, user_text: str) -> None:
        ctx.slice_start = len(self.raw_messages)
        message = {"role": "user", "content": user_text}
        self.raw_messages.append(message)
        ctx.full_turn_messages.append(message)
        # 掉线恢复（Step6）：user 消息先落库并置 pending_turn 草稿标记。
        # 必须在本轮（ReAct 循环）开始前置位——若放在收尾阶段，只能覆盖
        # 收尾窗口的崩溃，ReAct 中途掉线（最常见）仍会整轮丢失。
        self._finalizer.begin_turn(ctx)

    def emit_status(self, text: str, step: int = 0) -> None:
        """SSE thought 事件：受控状态说明（不透出模型原始思考）。"""
        log.info("react.status step=%s", step)
        self._emit_event("thought", {"text": text, "step": step})

    def emit_reasoning(self, text: str, step: int = 0) -> None:
        """SSE reasoning 事件：模型推理原文（推理模型适配 T6）。

        **默认关闭**（`settings.sse_reasoning_enabled=False`）。开启是对
        「thought 只发受控文案、不透出模型原始思考」这一有意设计的反向修改，
        属产品/合规决策：推理原文可能含内部措辞与未过滤内容，因此开启时先过
        输出 guardrails（复用 `guardrails_enabled` 管道），命中敏感词即不透出
        （审计仍留存，见 T3）。关闭时本方法是纯 no-op——断言"不透出"的现有
        测试因此继续成立。
        """
        if not settings.sse_reasoning_enabled or not text:
            return
        if settings.guardrails_enabled:
            from app.security.guardrails import check_output

            verdict = check_output(text)
            if verdict.blocked:
                log.warning("react.reasoning_blocked reason=%s", verdict.reason)
                return
        self._emit_event("reasoning", {"text": text, "step": step})

    def log_tool_result(self, name: str, sequence: int, display: str) -> None:
        """工具结果审计日志：摘要展示（完整结果不进日志）。"""
        log.info("react.tool_result", name=name, sequence=sequence, display=display)

    def append_audit_messages(self, messages: list[dict]) -> None:
        """审计切片进 append_log（完整工具结果只进审计存储）。"""
        self._append_log.extend(messages)

    def flush_session(self, enqueue_memory_job: bool = False,
                      turn_messages: list[dict] | None = None,
                      turn_id: str = "",
                      pending_turn: dict | None = None,
                      pending_write=_KEEP_PENDING_WRITE) -> None:
        """经 SessionStore 保存（CAS）；SQL 同事务入队 memory job。

        pending_turn：掉线恢复草稿标记（None = 清空）。轮次开始置位、成功
        收尾清除，与消息同一 save 事务落库——崩溃后遗留非空即「半截回复」。
        pending_write：写确认两阶段草稿（P1-2）。**未显式传参时保持会话现值**
        （哨兵默认值）——收尾保存等既有调用点不关心该字段，若默认成 None 会把
        本轮刚登记的草稿误清空；显式传 None 才是「作废草稿」。
        非 SQL 开发模式（文件/Redis）：走轻量文件队列（阶段F），任务条目
        携带本轮待巩固消息负载（Review 修复：不再依赖可能回退的消息长度）。
        """
        if pending_write is _KEEP_PENDING_WRITE:
            pending_write = self.pending_write
        pending = self._append_log[self._append_flushed:]
        supports_jobs = getattr(self.session_store, "memory_jobs_enabled", False)
        # 修复计划·一：保存会话前再次校验租约——失效则拒绝落库（防无锁写入）
        self._assert_lease()
        # idle consolidator is the explicit fallback when the durable worker is
        # disabled; do not leave a second queue behind for a future worker to
        # replay the same turns.
        queue_memory_job = enqueue_memory_job and settings.memory_job_worker_enabled
        updated = self.session_store.save(
            self.user_id,
            self._session_key,
            SessionState(
                session_id=self.session_id,
                user_id=self.user_id,
                summary=self.summary,
                messages=self.raw_messages,
                pending_turn=pending_turn,
                pending_write=pending_write,
                version=self._state_version,
                consolidated_len=self._consolidated_len,
            ),
            new_messages=pending,
            enqueue_memory_job=queue_memory_job and bool(pending),
        )
        self._state_version = updated.version
        self._append_flushed = len(self._append_log)
        if queue_memory_job and pending and not supports_jobs:
            from pathlib import Path as _P

            from app.agent.context_builder import fold_history
            from app.agent.memory.jobs import FileMemoryJobStore

            slice_msgs = turn_messages if turn_messages is not None else pending
            # 负载只含模型可见消息（user + 最终 assistant）；工具中间消息与
            # 纯文本协议纠错稿不进巩固（见 _memory_job_payload）
            payload = _memory_job_payload(fold_history(slice_msgs))
            store = FileMemoryJobStore(str(_P(settings.memory_dir) / "jobs"))
            store.enqueue(
                session_key=f"{self.user_id}/{self._session_key or 'session'}",
                user_id=self.user_id,
                through_seq=len(self.raw_messages),
                session_uuid=self.session_id,
                messages=payload,
                turn_id=turn_id,
            )

    def compress_history_by_tokens(self, folded: list[dict]) -> None:
        """token 水位压缩：老消息折叠视图摘要，正本只保留窗口尾部。

        folded 与 raw_messages 一一对应（fold 不增删消息），split 索引通用。
        """
        from app.agent.token_budget import budget_shares, trim_messages_to_budget

        shares = budget_shares(self.context_builder._window)
        target = int(shares["dialog"] * 0.6)
        kept = trim_messages_to_budget(folded, target)
        split = len(folded) - len(kept)
        if split <= 0:
            return
        old_messages = folded[:split]
        from app.agent.summarizer import summarize

        new_summary = summarize(
            client=self.client,
            model=self.model,
            old_messages=old_messages,
            prev_summary=self.summary,
        )
        self.summary = new_summary
        self.raw_messages = list(self.raw_messages[split:])
        # Review 修复：SQL 水位语义 = chat_messages.seq（memory_consolidated_seq），
        # 历史压缩只裁模型窗口，不得把水位写小（SQL save 侧另有数据库内
        # 单调 clamp 兜底）；文件模式水位随 job 负载携带，不再依赖该值。
        log.info(
            "history.compressed",
            compressed=len(old_messages),
            summary_len=len(new_summary),
        )
        self._emit_event(
            "compressed", {"count": len(old_messages), "summary_len": len(new_summary)},
        )

    # ============================================================
    # 会话管理
    # ============================================================
    def reset(self):
        # 修复计划·一：删除会话前校验租约——失效则拒绝删除（防无锁破坏状态）
        self._assert_lease()
        # Review 修复：以重置前的会话标识清理（确认存储按 session_id(uuid)
        # 注册；SQL memory_jobs 随 delete 同事务清理；文件模式按 session_key）
        old_session_key = self._session_key or "session"
        self.raw_messages = []
        self.summary = None
        self.session_id = uuid.uuid4().hex
        self.ctx.session_id = self.session_id
        self.memory_manager.bind_session(self.session_id)
        self._state_version = 0
        self._consolidated_len = 0
        self._append_log = []
        self._append_flushed = 0
        self.pending_turn = None  # reset 清空掉线草稿标记（新会话无半截轮次）
        # 写确认两阶段：reset 一并作废待确认草稿（超时/取消的第三种出口）
        self.pending_write = None
        self.ctx.pending_write = None
        self.ctx.write_confirm = "none"
        self.ctx.write_executed = False
        self.session_store.delete(self.user_id, old_session_key)
        try:
            from app.agent.memory.jobs import purge_session_file_jobs

            purge_session_file_jobs(self.user_id, old_session_key)
        except Exception:
            log.warning("reset.memory_jobs_cleanup_failed", exc_info=True)
        # 轮次状态一并复位：indeterminate 对账清单与上一轮 ctx 不跨会话残留
        self._indeterminate_writes = []
        self._last_turn_ctx = None

    def save(self) -> None:
        self.flush_session()

    def close(self):
        """阶段F：只释放本地资源，不再调用 LLM。

        复杂事实巩固由持久化 memory job 异步承担（SQL 同事务入队）；
        文件/Redis 开发模式由轻量队列 + idle scanner 漏单修复兜底。
        """
        self.tool_manager.close()
        if self._owns_tool_executor:
            self.tool_executor.close()

    def _emit_event(self, event_type: str, data: dict) -> None:
        if self.event_callback is not None:
            self.event_callback(event_type, data)
