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
from app.prompts.customer_service import SYSTEM_PROMPT
from app.schemas.response import CustomerServiceResponse, IntentType
from app.security.guardrails import check_input, guardrail_enabled
from app.security.scope_gate import SCOPE_BLOCK_REPLY, check_scope, scope_gate_enabled
from app.agent.tools.manager import ToolManager
from app.observability.logging import get_logger
from app.observability.metrics import record_budget_exhausted
from app.stores.base import (
    SessionConflictError,
    SessionOwnershipError,
    SessionState,
    SessionStore,
)
from app.stores.session_store import LocalFileSessionStore

log = get_logger("app.agent.chat")

# 业务升级规则（2026-08 评测后接线：模型对投诉/购买类倾向自行消化，需规则兜底）
# 命中任一强投诉信号 → 归为投诉意图并强制转人工
_COMPLAINT_SIGNALS = (
    "投诉", "举报", "消协", "工商局", "起诉", "曝光", "赔偿",
    "告你们", "态度差", "敷衍", "欺诈",
)
# 强购买意图（平台无下单工具）→ 转人工；同句含咨询词（价格/介绍等）视为咨询不升级
_PURCHASE_SIGNALS = ("我要买", "买一个", "拍下", "立即购买", "帮我下单", "直接买")
_CONSULT_HINTS = ("多少钱", "价格", "批发", "介绍", "推荐", "有货吗", "怎么卖")


def _has_complaint_signal(text: str) -> bool:
    return any(k in text for k in _COMPLAINT_SIGNALS)


def _has_purchase_signal(text: str) -> bool:
    if any(k in text for k in _PURCHASE_SIGNALS):
        return not any(k in text for k in _CONSULT_HINTS)
    return False


class EcomAgent:
    """电商客服 Agent —— 阶段一 1.5：请求级参数化（user_id/session_id）。

    Agent 是「无状态算子」：构造 → 处理一轮 → 写回 → 丢弃。
    user_id/session_id 为请求级参数，pod 级资源（client/skill_manager/
    mcp_client）可注入共享，不再读全局套用单一用户。
    """

    def __init__(
        self,
        user_id: str = "default",
        session_id: Optional[str] = None,
        session_path: Optional[str] = None,
        client: Optional[OpenAI] = None,
        skill_manager=None,
        mcp_client=None,
        tool_manager: Optional[ToolManager] = None,
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
        enforce_order_ownership: Optional[bool] = None,
    ):
        self.user_id = user_id
        self.client = client or self._build_default_client()
        self.model = model or settings.model_name
        self.temperature = (
            settings.temperature if temperature is None else temperature
        )
        self.session_path = session_path or self._derive_session_path(user_id, session_id)
        # 阶段二 2.1：会话/记忆经 SessionStore/LTMStore 协议读写（可外置 Redis）
        self.session_store = session_store or LocalFileSessionStore(
            settings.session_dir,
            exact_path=session_path,  # 沙箱等显式单文件场景
        )
        self._state_version = 0  # 乐观锁版本（CAS）
        self._consolidate_every = (
            settings.memory_consolidate_every if consolidate_every is None else consolidate_every
        )
        self._consolidated_len = 0  # 已增量巩固到的消息位置
        self.history_threshold = settings.history_threshold
        self.history_keep_recent = settings.history_keep_recent
        self.max_react_steps = settings.max_react_steps

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
        self.summary: Optional[str] = None
        # 阶段八：全量追加日志——压缩只裁剪 raw_messages（LLM 窗口），
        # append_log 永续，SQL 正本据它做行式追加（历史压缩不丢正本）
        self._append_log: list[dict] = []
        self._append_flushed = 0
        # 4.5：思考/工具/结果逐条事件回调（SSE 流式）；None = 不推流
        self.event_callback = event_callback
        self._react_steps_count = 0

        # Agent能力强化计划：工具批次执行器（pod 共享经注入；CLI/评估自建）
        # + 请求级状态工厂（评估沙箱注入 on_outcomes 采集轨迹）
        # 修复计划：自建时才负责 close（pod 共享交由 FastAPI lifespan 关闭）
        self._owns_tool_executor = tool_executor is None
        self._tool_executor = tool_executor or ToolBatchExecutor()
        self._turn_state_factory = turn_state_factory
        # 改造一/三：轮次预算与引用 verdict（每次 chat() 刷新）
        self._turn_budget: Optional[TurnBudget] = None
        self._last_citation_verdict: Optional[dict] = None
        # 修复计划：写结果未知清单（indeterminate → 强制转人工 + handoff 对账）
        self._indeterminate_writes: list[dict] = []

        # 存储键 = 请求级 session_id（稳定：缺省 "" → session.json）；
        # payload 里的 session_id（uuid）是会话身份，二者分离
        self._session_key = session_id or ""
        loaded = self.session_store.load(user_id, self._session_key)
        # 3.1：session 归属校验（用户 B 拿用户 A 的 session_id → 403）
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
            self._append_log = list(loaded.messages)  # 已入库历史不回灌
            self._append_flushed = len(self._append_log)
            # 安全修复 P2：巩固水位随会话文档恢复（历史只在内存，重启归零
            # → close()/兜底巩固对全量历史重复摘要）
            self._consolidated_len = loaded.consolidated_len
            if loaded.short_term_memory:
                self.memory_manager.restore_stm(loaded.short_term_memory)

        # 1.3：工具调用上下文（memory/skill 句柄随 Agent 实例走）
        # 3.2：credentials（外部凭证）随 ctx 注入，永不进 prompt/日志
        # 2.2：请求级归属强制开关（None=跟随全局配置；评测沙箱显式传 True）
        self.ctx = ToolContext(
            user_id=user_id,
            session_id=self.session_id,
            memory=self.memory_manager,
            skill_manager=self.skill_manager,
            credentials=credentials,
            enforce_order_ownership=enforce_order_ownership,
        )

        if settings.evolve_capture_enabled:
            from app.evolution.recorder import TurnRecorder

            self._turn_recorder = TurnRecorder(
                Path(settings.evolve_turns_dir), archive=turns_archive,
            )
        else:
            self._turn_recorder = None

    @staticmethod
    def _build_default_client() -> OpenAI:
        """兜底自建 client（生产路径 client 注入；安全修复 P2：显式 timeout
        取代 SDK 默认 600s）。"""
        return OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )

    @staticmethod
    def _derive_session_path(user_id: str, session_id: Optional[str]) -> str:
        """默认按 {session_dir}/{user_id}/{session_id}.json 派生；未给 session_id 用 session.json。"""
        return str(Path(settings.session_dir) / user_id / f"{session_id or 'session'}.json")

    @property
    def history_size(self) -> int:
        return len(self.raw_messages)

    def chat(self, user_input: str) -> CustomerServiceResponse:
        """处理用户输入：ReAct 循环 → 结构化提取 → 返回结果。

        改造一：chat() 起点确定 turn deadline，同时存 Agent 实例属性
        （close 线程显式绑定用）与 ContextVar（本线程 LLM/工具调用用）。
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
        self._current_query = user_input  # 改造四：记忆相关性 query（显式传入）
        start = len(self.raw_messages)

        # 输入侧护栏（3.5 接线）：注入 → 零 LLM 拦截并转人工；PII → 脱敏后继续
        if guardrail_enabled(getattr(self, "ctx", None)):
            verdict = check_input(user_input)
            if verdict.blocked:
                return self._guardrail_block_response(user_input, verdict, start)
            if verdict.masked:
                user_input = verdict.text

        # 业务范围闸门：非业务/闲聊 → 固定引导话术（不转人工；先于任何工具/主 LLM）
        if scope_gate_enabled(getattr(self, "ctx", None)):
            verdict = check_scope(user_input, self.client, self.model)
            if not verdict.in_scope:
                return self._scope_block_response(user_input, verdict, start)

        self.raw_messages.append({"role": "user", "content": user_input})

        try:
            final_text = self._react_loop(state, budget)
        except LLMBudgetExhausted:
            record_budget_exhausted("react")
            return self._finish_budget_fallback(user_input, start)
        if budget.expired():
            # 预算耗尽路径：不调 LLM 强制收尾/提取，返回确定性 fallback
            return self._finish_budget_fallback(user_input, start)
        # 2.5：每轮工具调用数直方图（评估报告与生产看板同源）
        from app.observability.metrics import record_tool_calls_per_turn
        record_tool_calls_per_turn(sum(state.per_name_counts.values()))
        try:
            result = self._extract_structured_response(final_text)
        except LLMBudgetExhausted:
            record_budget_exhausted("extract")
            return self._finish_budget_fallback(user_input, start)

        # 改造三：引用真实性校验——先于落库与演进采集（单/多 Agent 同约定）
        from app.agent.citations import apply_citation_policy

        self._last_citation_verdict = apply_citation_policy(result, state.sources)

        # 修复计划：写结果未知 → 强制转人工（不改已生成回复，只改派发信号）
        self._apply_indeterminate(result, state)

        # 业务升级规则（2026-08 评测后接线）：强投诉 → 归为投诉并转人工；
        # 强购买意图 → 转人工（平台无下单工具）
        if _has_complaint_signal(user_input):
            result.intent = IntentType.COMPLAINT
            result.requires_human = True
            result.confidence = round(min(result.confidence, 0.5), 4)
        elif _has_purchase_signal(user_input):
            result.requires_human = True
            result.confidence = round(min(result.confidence, 0.5), 4)

        # 辅助任务（STM/摘要/增量 LTM）：可跳过阶段——预算耗尽绝不毁掉已生成回复
        try:
            self.memory_manager.update_short_term(self.raw_messages[-6:])
        except LLMBudgetExhausted:
            record_budget_exhausted("stm")
            log.info("aux.stm_skipped 轮次预算耗尽，短期记忆更新跳过")

        self.raw_messages.append(
            {"role": "assistant", "content": json.dumps(result.model_dump(), ensure_ascii=False)}
        )

        self._record_turn(user_input, result, start)
        # 阶段八：压缩前把本轮消息快照进追加日志（压缩裁剪窗口不裁正本）
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

    def _guardrail_block_response(
        self, user_input: str, verdict, start: int
    ) -> CustomerServiceResponse:
        """输入侧护栏拦截（3.5 接线）：固定拒绝话术 + 转人工，不消耗 LLM 预算。

        复用零 LLM 收尾的落库/记录流程；写入占位 assistant 消息供会话还原。
        """
        log.warning("guardrail.block user=%s reason=%s", self.user_id, verdict.reason)
        result = CustomerServiceResponse(
            intent=IntentType.OTHER,
            confidence=0.1,
            reply=(
                "抱歉，您的消息包含疑似指令注入/敏感内容，出于安全考虑已被拦截，"
                "本条消息已转接人工客服处理，请稍候。"
            ),
            requires_human=True,
        )
        self.raw_messages.append({"role": "user", "content": user_input})
        self.raw_messages.append(
            {"role": "assistant", "content": json.dumps(result.model_dump(), ensure_ascii=False)}
        )
        self._record_turn(user_input, result, start)
        self._append_log.extend(self.raw_messages[start:])
        self._save_session()
        return result

    def _scope_block_response(
        self, user_input: str, verdict, start: int,
    ) -> CustomerServiceResponse:
        """业务范围闸门拦截：非业务/闲聊 → 固定引导话术（不转人工）。

        复用零 LLM 收尾的落库/记录流程；写入占位 assistant 消息供会话还原。
        """
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
        """经 SessionStore 保存（CAS；版本冲突抛 SessionConflictError → 409）。

        new_messages：未入库的增量（整包覆写型 store 忽略；SQL store 行式追加）。
        """
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
        """2.3：每 N 轮增量巩固 LTM——close() 在 K8s 里不可靠，这是主路径。"""
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
            mode="single",
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
        # 只巩固未增量覆盖的尾部（2.3 增量后 close 只是兜底）。
        # 改造一（评审·三轮1）：close 在另一 worker 线程执行，无法继承
        # chat() 内的 ContextVar——这里从实例属性显式绑定并在 finally reset。
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
        self.tool_manager.close()
        # 修复计划：自建执行器由 Agent 所有者关闭（pod 共享由 lifespan 负责）
        if self._owns_tool_executor:
            self._tool_executor.close()

    def _print_thought(self, text: str) -> None:
        log.info("react.thought", text=text)
        self._emit_event("thought", {"text": text})

    def _react_loop(self, state: ToolTurnState, budget: TurnBudget) -> str:
        """ReAct 循环：调用 LLM → 执行工具 → 观察结果 → 重复，直到模型给出最终回答。

        工具批次经 ToolBatchExecutor（原序分段并行 + 重复拦截 + 预算规则）；
        每步前检查轮次预算，耗尽抛 LLMBudgetExhausted（上层零 LLM fallback）。
        """
        self._react_steps_count = 0
        for step in range(self.max_react_steps):
            self._react_steps_count = step + 1
            if budget.expired():
                raise LLMBudgetExhausted("轮次预算耗尽，停止 ReAct 循环")
            messages = self._build_messages(self._current_query)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                tools=self.tool_manager.tool_definitions,
            )
            choice = response.choices[0]
            assistant_msg = choice.message

            if assistant_msg.content:
                self._print_thought(assistant_msg.content)

            if not assistant_msg.tool_calls:
                content = assistant_msg.content or ""
                self.raw_messages.append({"role": "assistant", "content": content})
                return content

            msg_dict = {"role": "assistant", "content": assistant_msg.content}
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_msg.tool_calls
            ]
            self.raw_messages.append(msg_dict)

            outcomes = self._tool_executor.execute(
                [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in assistant_msg.tool_calls
                ],
                state, self.ctx, self.tool_manager, budget=budget,
            )
            for oc in outcomes:
                display = oc.result if len(oc.result) <= 300 else oc.result[:300] + "..."
                log.info("react.tool_result", name=oc.name,
                         sequence=oc.sequence, display=display)
                self.raw_messages.append({
                    "role": "tool",
                    "tool_call_id": oc.call_id,
                    "content": oc.result,
                })

        if budget.expired():
            raise LLMBudgetExhausted("轮次预算耗尽，强制收尾被拒绝")
        messages = self._build_messages(self._current_query)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content or ""
        self.raw_messages.append({"role": "assistant", "content": content})
        return content

    def _extract_structured_response(self, text: str) -> CustomerServiceResponse:
        """从最终文本中提取结构化元数据（意图、置信度等）。"""
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

    def _build_messages(self, query: str = "") -> list[dict]:
        system_content = SYSTEM_PROMPT
        if self.skill_manager and self.skill_manager.enabled:
            system_content += self.skill_manager.build_catalog_prompt()

        messages: list[dict] = [
            {"role": "system", "content": system_content}
        ]
        # 改造四：query = 本轮原始 user_input（记忆相关性筛选）
        messages.extend(self.memory_manager.build_memory_prompt_sections(query))
        if self.summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"以下是此前对话的摘要，用于延续上下文记忆：\n{self.summary}",
                }
            )
        messages.extend(self.raw_messages)
        return messages

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
        self._consolidated_len = 0  # 增量巩固窗口随压缩重置
        log.info(
            "history.compressed",
            compressed=len(old_messages),
            summary_len=len(new_summary),
        )
        self._emit_event(
            "compressed", {"count": len(old_messages), "summary_len": len(new_summary)},
        )

    def _emit_event(self, event_type: str, data: dict) -> None:
        """4.5 SSE：回调由服务层注入（线程安全由调用方保证）。"""
        if self.event_callback is not None:
            self.event_callback(event_type, data)

    # 说明：tool_call / tool_result 的 SSE 事件与日志已迁移到
    # ToolBatchExecutor（带 tool_call_id + sequence）；Agent 不再自行发工具事件。
