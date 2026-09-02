"""Multi-Agent 编排器：协调 Router 和子 Agent 完成用户请求。

流程：Router 分类意图 → 选择子 Agent → ReAct 执行 → 结构化提取 → 持久化。
"""

import json
import uuid
from pathlib import Path
from typing import Optional

from openai import OpenAI

from app.agent.context import ToolContext
from app.agent.summarizer import summarize
from app.agent.tools.batch_executor import ToolBatchExecutor, ToolTurnState
from app.agent.turn_budget import (
    LLMBudgetExhausted,
    TurnBudget,
    bind_budget,
    budget_fallback_response,
    reset_budget,
)
from app.config.settings import settings
from app.multi_agent.agents import AGENT_CONFIGS, SubAgent
from app.multi_agent.router import Router
from app.schemas.response import CustomerServiceResponse, IntentType
from app.security.scope_gate import SCOPE_BLOCK_REPLY, check_scope, scope_gate_enabled
from app.agent.tools.manager import ToolManager
from app.observability.logging import get_logger
from app.observability.metrics import record_budget_exhausted
from app.stores.base import (
    SessionOwnershipError,
    SessionState,
    SessionStore,
)
from app.stores.session_store import LocalFileSessionStore

log = get_logger("app.multi_agent.orchestrator")


class MultiAgentOrchestrator:
    """多 Agent 编排器，对外接口与 EcomAgent 一致。

    阶段一 1.5：user_id/session_id 请求级参数化，与 EcomAgent 同构。
    """

    def __init__(
        self,
        user_id: str = "default",
        session_id: Optional[str] = None,
        session_path: Optional[str] = None,
        client: Optional[OpenAI] = None,
        skill_manager=None,
        mcp_client=None,
        memory_enabled: Optional[bool] = None,
        use_mcp: Optional[bool] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        session_store: Optional[SessionStore] = None,
        ltm_store=None,
        consolidate_every: Optional[int] = None,
        turns_archive=None,
        credentials: Optional[dict] = None,
        event_callback=None,
        tool_executor: Optional[ToolBatchExecutor] = None,
        turn_state_factory=None,
    ):
        self.event_callback = event_callback
        # Agent能力强化计划：工具批次执行器（pod 共享经注入；CLI/评估自建）
        # + 请求级状态工厂（评估沙箱注入 on_outcomes 采集轨迹）
        # 修复计划：自建时才负责 close（pod 共享交由 FastAPI lifespan 关闭）
        self._owns_tool_executor = tool_executor is None
        self._tool_executor = tool_executor or ToolBatchExecutor()
        self._turn_state_factory = turn_state_factory
        # 改造一/三：轮次预算与引用 verdict（每次 chat() 刷新）
        self._turn_budget: Optional[TurnBudget] = None
        self._turn_state: Optional[ToolTurnState] = None
        self._last_citation_verdict: Optional[dict] = None
        self._react_steps_count = 0
        self._indeterminate_writes: list[dict] = []
        self.user_id = user_id
        self.client = client or OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_seconds,  # 安全修复 P2：显式 timeout
            max_retries=0,  # 重试由韧性包装负责
        )
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
        self._consolidate_every = (
            settings.memory_consolidate_every if consolidate_every is None else consolidate_every
        )
        self._consolidated_len = 0
        self.history_threshold = settings.history_threshold
        self.history_keep_recent = settings.history_keep_recent
        self.max_react_steps = settings.max_react_steps

        self.router = Router(self.client, self.model)

        self.agents: dict[str, SubAgent] = {}
        for key, cfg in AGENT_CONFIGS.items():
            tm = ToolManager(
                use_mcp=settings.mcp_enabled if use_mcp is None else use_mcp,
                mcp_server_url=settings.mcp_server_url,
                allowed_tools=cfg["tools"],
                mcp_client=mcp_client,
            )
            self.agents[key] = SubAgent(
                name=cfg["name"],
                system_prompt=cfg["prompt"],
                tool_manager=tm,
                client=self.client,
                model=self.model,
                temperature=self.temperature,
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
        self.summary: Optional[str] = None
        # 阶段八：追加日志（压缩裁剪窗口不裁正本）
        self._append_log: list[dict] = []
        self._append_flushed = 0

        # 存储键 = 请求级 session_id（稳定）；payload 内 uuid 是会话身份
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
        if loaded:
            self._state_version = loaded.version
            self.summary = loaded.summary
            self.raw_messages = loaded.messages
            self._append_log = list(loaded.messages)
            self._append_flushed = len(self._append_log)
            # 安全修复 P2：巩固水位随会话文档恢复（防重启重复巩固）
            self._consolidated_len = loaded.consolidated_len
            if loaded.short_term_memory:
                self.memory_manager.restore_stm(loaded.short_term_memory)

        # 1.3：工具调用上下文（子 Agent 共享同一 ctx，3.2 携带外部凭证）
        self.ctx = ToolContext(
            user_id=user_id,
            session_id=self.session_id,
            memory=self.memory_manager,
            skill_manager=self.skill_manager,
            credentials=credentials,
        )

        if settings.evolve_capture_enabled:
            from app.evolution.recorder import TurnRecorder

            self._turn_recorder = TurnRecorder(
                Path(settings.evolve_turns_dir), archive=turns_archive,
            )
        else:
            self._turn_recorder = None

    @staticmethod
    def _derive_session_path(user_id: str, session_id: Optional[str]) -> str:
        return str(Path(settings.session_dir) / user_id / f"{session_id or 'session'}.json")

    @property
    def history_size(self) -> int:
        return len(self.raw_messages)

    def chat(self, user_input: str) -> CustomerServiceResponse:
        """处理用户输入：路由 → 子 Agent 执行 → 结构化提取 → 返回结果。

        改造一：chat() 起点确定 turn deadline，同时存实例属性（close 线程
        显式绑定用）与 ContextVar（本线程 LLM/工具调用用）。
        """
        budget = TurnBudget.start(settings.turn_budget_seconds)
        self._turn_budget = budget
        token = bind_budget(budget)
        try:
            return self._chat_locked(user_input, budget)
        finally:
            reset_budget(token)

    def _chat_locked(self, user_input: str, budget: TurnBudget) -> CustomerServiceResponse:
        state = (
            self._turn_state_factory()
            if self._turn_state_factory is not None
            else ToolTurnState(event_callback=self.event_callback)
        )
        self._turn_state = state
        self._last_citation_verdict = None
        self._indeterminate_writes = []
        self._react_steps_count = 0  # 修复计划：每轮重置，避免复用实例跨轮累计
        self._current_query = user_input  # 改造四：记忆相关性 query（显式传入）
        start = len(self.raw_messages)

        # 业务范围闸门（与单 Agent chat.py 同约定）：非业务/闲聊 → 固定引导话术。
        # 放在路由器之前：闲聊不进路由、不触发子 Agent ReAct。
        if scope_gate_enabled(getattr(self, "ctx", None)):
            verdict = check_scope(user_input, self.client, self.model)
            if not verdict.in_scope:
                return self._scope_block_response(user_input, verdict, start)

        self.raw_messages.append({"role": "user", "content": user_input})

        try:
            agent_key = self.router.route(user_input, self.raw_messages)
            agent = self.agents[agent_key]
            log.info("multi.route", target=agent.name)
            if getattr(self, "event_callback", None) is not None:
                self.event_callback("route", {"target": agent.name})
        except LLMBudgetExhausted:
            record_budget_exhausted("react")
            return self._finish_budget_fallback(user_input, start)

        messages = self._build_messages(agent, self._current_query)
        try:
            final_text, new_messages, steps = agent.handle(
                messages, ctx=self.ctx, max_steps=self.max_react_steps,
                executor=self._tool_executor, state=state, budget=budget,
            )
        except LLMBudgetExhausted:
            record_budget_exhausted("react")
            return self._finish_budget_fallback(user_input, start)
        # 改造一：orchestrator 回填真实 step count（此前恒为 1，分布指标失真）
        self._react_steps_count += steps
        self.raw_messages.extend(new_messages)

        if budget.expired():
            # 预算耗尽路径：不调 LLM 强制收尾/提取，返回确定性 fallback
            return self._finish_budget_fallback(user_input, start)
        try:
            result = self._extract_structured_response(final_text)
        except LLMBudgetExhausted:
            record_budget_exhausted("extract")
            return self._finish_budget_fallback(user_input, start)

        # 改造三：引用真实性校验——先于落库与演进采集（单/多 Agent 同约定）
        from app.agent.citations import apply_citation_policy

        self._last_citation_verdict = apply_citation_policy(result, state.sources)

        # 修复计划：写结果未知 → 强制转人工 + handoff 对账
        self._apply_indeterminate(result, state)

        try:
            self.memory_manager.update_short_term(self.raw_messages[-6:])
        except LLMBudgetExhausted:
            record_budget_exhausted("stm")
            log.info("aux.stm_skipped 轮次预算耗尽，短期记忆更新跳过")

        self.raw_messages.append(
            {"role": "assistant", "content": json.dumps(result.model_dump(), ensure_ascii=False)}
        )

        self._record_turn(user_input, result, start)
        self._append_log.extend(self.raw_messages[start:])

        if len(self.raw_messages) > self.history_threshold:
            try:
                self._compress_history()
            except LLMBudgetExhausted:
                record_budget_exhausted("summary")
                log.info("aux.summary_skipped 轮次预算耗尽，历史摘要跳过")

        try:
            self._maybe_consolidate_incremental()
        except LLMBudgetExhausted:
            record_budget_exhausted("ltm")
            log.info("aux.ltm_skipped 轮次预算耗尽，增量巩固跳过")
        self._save_session()
        return result

    def _apply_indeterminate(self, result, state: ToolTurnState) -> None:
        """写结果未知（indeterminate）：强制转人工 + 压置信，操作清单留给 handoff。"""
        if not state.indeterminate_writes:
            return
        self._indeterminate_writes = list(state.indeterminate_writes)
        result.requires_human = True
        result.confidence = round(min(result.confidence, 0.5), 4)
        log.warning(
            "tool.indeterminate user=%s writes=%s", self.user_id,
            self._indeterminate_writes,
        )
        from app.observability.metrics import record_handoff

        record_handoff("tool_indeterminate")

    def _scope_block_response(
        self, user_input: str, verdict, start: int,
    ) -> CustomerServiceResponse:
        """业务范围闸门拦截：非业务/闲聊 → 固定引导话术（不转人工、不进路由）。"""
        log.info("scope.block user=%s reason=%s", self.user_id, verdict.reason)
        result = CustomerServiceResponse(
            intent=IntentType.OTHER,
            confidence=0.1,
            reply=SCOPE_BLOCK_REPLY,
            requires_human=False,
        )
        self.raw_messages.append({"role": "user", "content": user_input})
        self.raw_messages.append(
            {"role": "assistant", "content": json.dumps(result.model_dump(), ensure_ascii=False)}
        )
        self._record_turn(user_input, result, start)
        self._append_log.extend(self.raw_messages[start:])
        self._save_session()
        return result

    def _finish_budget_fallback(self, user_input: str, start: int) -> CustomerServiceResponse:
        """零 LLM 收尾：固定话术 + requires_human；STM/摘要/LTM 辅助任务跳过。"""
        log.warning("turn.budget_exhausted user=%s", self.user_id)
        result = budget_fallback_response()
        self.raw_messages.append(
            {"role": "assistant", "content": json.dumps(result.model_dump(), ensure_ascii=False)}
        )
        self._record_turn(user_input, result, start)
        self._append_log.extend(self.raw_messages[start:])
        self._save_session()
        return result

    def _save_session(self) -> None:
        pending = self._append_log[self._append_flushed:]
        updated = self.session_store.save(
            self.user_id,
            self._session_key,
            SessionState(
                session_id=self.session_id,
                user_id=self.user_id,
                summary=self.summary,
                messages=self.raw_messages,
                short_term_memory=self.memory_manager.stm_to_dict(),
                version=self._state_version,
                consolidated_len=self._consolidated_len,
            ),
            new_messages=pending,
        )
        self._state_version = updated.version
        self._append_flushed = len(self._append_log)

    def _maybe_consolidate_incremental(self) -> None:
        if not self.memory_manager.memory_enabled:
            return
        every = self._consolidate_every
        if not every or every <= 0:
            return
        if len(self.raw_messages) - self._consolidated_len >= every:
            self.memory_manager.consolidate_to_long_term(
                self.raw_messages[self._consolidated_len:], self.summary
            )
            self._consolidated_len = len(self.raw_messages)

    def _record_turn(self, user_input: str, result, start: int) -> None:
        """第10期：本轮轮次切片落盘（脱敏），失败不影响主流程。"""
        if self._turn_recorder is None:
            return
        self._turn_recorder.record(
            session_id=self.session_id,
            mode="multi",
            question=user_input,
            structured_reply=result,
            turn_slice=self.raw_messages[start:],
            user_id=self.user_id,
        )

    def reset(self):
        self.raw_messages = []
        self.summary = None
        self.session_id = uuid.uuid4().hex
        self.ctx.session_id = self.session_id
        self.memory_manager.bind_session(self.session_id)
        self._state_version = 0
        self._consolidated_len = 0
        self._append_log = []
        self._append_flushed = 0
        self.memory_manager.reset_short_term()
        self.session_store.delete(self.user_id, self._session_key)

    def save(self) -> None:
        self._save_session()

    def close(self):
        # 改造一（评审·三轮1）：close 在另一 worker 线程执行，无法继承
        # chat() 内的 ContextVar——从实例属性显式绑定并在 finally reset。
        # 预算耗尽 → 巩固 LLM 调用被拒（辅助任务跳过），不拖垮收尾。
        token = bind_budget(self._turn_budget)
        try:
            self.memory_manager.consolidate_to_long_term(
                self.raw_messages[self._consolidated_len:], self.summary
            )
        except LLMBudgetExhausted:
            log.info("close.budget_exhausted: LTM 巩固跳过（本轮预算已耗尽）")
        except Exception:  # noqa: BLE001 —— 巩固失败不影响连接清理
            log.warning("close.consolidate_failed", exc_info=True)
        finally:
            reset_budget(token)
        for agent in self.agents.values():
            agent.tool_manager.close()
        # 修复计划：自建执行器由 Agent 所有者关闭（pod 共享由 lifespan 负责）
        if self._owns_tool_executor:
            self._tool_executor.close()

    def _build_messages(self, agent: SubAgent, query: str = "") -> list[dict]:
        """用子 Agent 的 system prompt 构建消息列表。"""
        system_content = agent.system_prompt
        if self.skill_manager and self.skill_manager.enabled:
            system_content += self.skill_manager.build_catalog_prompt()

        messages: list[dict] = [
            {"role": "system", "content": system_content}
        ]
        # 改造四：query = 本轮原始 user_input（记忆相关性筛选）
        messages.extend(self.memory_manager.build_memory_prompt_sections(query))
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"以下是此前对话的摘要，用于延续上下文记忆：\n{self.summary}",
            })
        messages.extend(self.raw_messages)
        return messages

    def _extract_structured_response(self, text: str) -> CustomerServiceResponse:
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "基于以下客服回复内容，提取结构化信息。"
                            "reply 字段直接使用原文，不要修改或缩减。"
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                response_format=CustomerServiceResponse,
            )
            return response.choices[0].message.parsed
        except Exception:
            return self._extract_structured_fallback(text)

    def _extract_structured_fallback(self, text: str) -> CustomerServiceResponse:
        """当 response_format 不被 API 支持时，用 prompt 引导 JSON 输出。"""
        intent_values = ", ".join(f'"{e.value}"' for e in IntentType)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "基于以下客服回复内容，提取结构化信息并输出 JSON。\n"
                        "reply 字段直接使用原文，不要修改或缩减。\n\n"
                        "必须严格按照以下 JSON 格式输出（不要加 markdown 代码块）：\n"
                        "{\n"
                        f'  "intent": <从以下选择: {intent_values}>,\n'
                        '  "confidence": <0.0到1.0的浮点数>,\n'
                        '  "reply": <原文回复内容>,\n'
                        '  "requires_human": <true或false>,\n'
                        '  "follow_up_question": <追问问题或null>\n'
                        "}"
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return CustomerServiceResponse.model_validate_json(raw)

    def _compress_history(self) -> None:
        keep = self.history_keep_recent
        split = len(self.raw_messages) - keep
        while split > 0 and self.raw_messages[split].get("role") in ("tool",):
            split -= 1
        if split <= 0:
            return
        old_messages = self.raw_messages[:split]
        recent = self.raw_messages[split:]

        new_summary = summarize(
            client=self.client,
            model=self.model,
            old_messages=old_messages,
            prev_summary=self.summary,
        )
        self.summary = new_summary
        self.raw_messages = recent
        self._consolidated_len = 0
        log.info(
            "history.compressed",
            compressed=len(old_messages),
            summary_len=len(new_summary),
        )
