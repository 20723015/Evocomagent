"""评估沙箱：隔离、可复现地重跑测试集，并采集运行全过程（第9期）。

沙箱做三件事：
1. 隔离：每条用例独立的临时 session 文件；关闭记忆读写（否则 default.json 会注入
   prompt 污染评分）；关闭 MCP 只用本地 mock 工具（保证可复现，且让幻觉检测有确定
   的 ground truth）。
2. 插桩：被测 Agent 只共享一个 OpenAI client 实例，给它的
   chat.completions.create / beta.chat.completions.parse 打补丁，即可捕获整个会话
   所有 LLM 调用的 token、被请求的工具、延迟——无需改动 chat.py。
   工具返回值则通过包裹 ToolManager.execute_tool 采集。
3. 执行：顺序跑完用例的多轮输入，把过程与结果填进 RunTrace 返回。

关键：绝不调用 agent.close()（会触发长期记忆巩固的 LLM 写入，污染且烧钱）；
所有补丁在 finally 中还原。
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

from app.agent.tools.batch_executor import ToolBatchExecutor
from app.config.settings import settings
from app.evaluation.dataset import EvalCase
from app.llm.model_profile import resolve_profile

# 沙箱评测固定温度（低危修复 C5）：温度归零让输出更确定（此前用全局 0.7，
# 评分存在随机抖动）。manifest 的 runtime 指纹必须记录该常量——指纹描述的是
# 实际运行的配置，而非 settings 里被沙箱忽略的值。
SANDBOX_TEMPERATURE = 0.0
# 沙箱评测固定开启订单归属校验（2.2）：评测沙箱不依赖全局 .env，安全用例
# 无论本机配置如何都执行 fail-closed 语义。manifest 必须记录该常量而非
# settings.enforce_order_ownership——指纹描述实际运行配置（与温度 C5 同口径）
SANDBOX_ENFORCE_ORDER_OWNERSHIP = True
from app.evaluation.trace import LLMCallRecord, RunTrace, ToolObservation  # noqa: E402


def sandbox_temperature(model: str) -> float | None:
    """沙箱被测模型温度（推理模型适配 T9）：None = **不传** temperature。

    推理模型画像下 temperature 被忽略（DeepSeek）/禁传（o 系列）/必须为 1
    （Claude thinking），硬设 0.0 是「假装可控」——指纹会记下一个实际未生效的值。
    因此非 free 画像返回 None，由 `llm/client.py` 画像层决定请求体（删除或置 1）；
    free 画像与未命中画像仍返回 `SANDBOX_TEMPERATURE`（现状行为）。
    """
    profile = resolve_profile(model)
    if profile is not None and profile.temperature_mode != "free":
        return None
    return SANDBOX_TEMPERATURE



def _structured_outcome(result_str: str) -> dict | None:
    """从工具返回 JSON 抽取可判定字段（2.2）：success/code/status/count。

    只保留机器可判定的摘要字段，不整段拷贝原始结果（轨迹体量可控、
    不携带他人物品/金额等敏感明细进入报告）。
    """
    try:
        data = json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    return {k: data[k] for k in ("success", "code", "status", "count") if k in data}


class Sandbox:
    """Agent 评估沙箱：构建隔离环境、插桩采集、跑用例产出 RunTrace。"""

    def __init__(self, tmp_root: str | None = None, resilient: bool = False):
        # resilient=True：给 Agent 的 LLM client 安装韧性层（429/超时/连接错误
        # 指数退避重试 + 降级链，见 app/llm/client.py）。评测路径默认 client
        # max_retries=0，外部 API 限流会直接打穿整条用例；启用后仍受轮次预算约束。
        self.resilient = resilient
        self.tmp_root = Path(tmp_root) if tmp_root else Path(tempfile.mkdtemp(prefix="eval_sandbox_"))
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def session_path_for(self, case_id: str) -> str:
        return str(self.tmp_root / f"{case_id}.json")

    def _reset_session(self, case: EvalCase) -> None:
        """删除上一轮残留的会话文件与（记忆用例）播种目录。"""
        for stale in (self.tmp_root / f"{case.id}.json",
                      self.tmp_root / f"{case.id}_memory"):
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
            elif stale.exists():
                stale.unlink()

    def _build_agent(self, session_path: str, case: EvalCase | None = None):
        """在隔离配置下构建被测 Agent（阶段一 1.4：显式 override 参数）。

        memory/mcp/temperature 不再 monkey-patch 全局 settings，而是以
        构造参数显式传入（覆盖默认）：关闭记忆读写与 MCP 保证可复现，
        temperature 归零让输出更确定（此前用全局 0.7，评分存在随机抖动）。

        2.2：user_id 取 case.actor_user_id（None → u1）；归属强制显式传 True——
        评测沙箱不依赖全局 .env，安全用例无论本机配置如何都执行 fail-closed 语义。

        改造二（评审·三轮3）：turn_state_factory 注入 on_outcomes 钩子——
        工具观察按模型给定顺序一次性写入 RunTrace（旧 execute_tool 包装层
        拿不到 sequence，并行后执行顺序≠模型顺序）。tool_executor 显式注入
        独立实例：沙箱内并行度/全局并发不受 pod 单例状态污染。

        推理模型适配 T9：温度按画像分支——非 free 画像**不传** temperature 参数
        （而不是传一个会被忽略/拒绝的 0.0），请求体的最终形态由画像层决定。
        """
        user_id = (case.actor_user_id if case is not None else None) or "u1"
        # 记忆系统重构·阶段0：记忆用例——独立 memory_dir + 显式播种，其余
        # 用例维持 memory_enabled=False（历史可复现约定不变）
        memory_on = bool(getattr(case, "memory_enabled", False))
        if memory_on:
            user_id = (case.actor_user_id if case is not None else None) or "mem_u1"
        temperature = sandbox_temperature(settings.model_name)
        common = dict(
            user_id=user_id,
            session_path=session_path,
            memory_enabled=memory_on,
            use_mcp=False,
            tool_executor=ToolBatchExecutor(),
            turn_state_factory=self._make_state_factory(),
            enforce_order_ownership=SANDBOX_ENFORCE_ORDER_OWNERSHIP,
            **({} if temperature is None else {"temperature": temperature}),
        )

        from app.agent.chat import EcomAgent
        agent = EcomAgent(**common)
        # 写工具租约门禁：沙箱独占会话（无并发），注入「始终持有租约」的
        # no-op 守卫——与生产路由层 `agent.bind_lease_guard(lease.assert_owned)`
        # 等价。缺这一步时所有写工具都被 SESSION_LOCK_REQUIRED 挡下，
        # 写路径（退款提交/撤回）在 agent 级评测里等于零覆盖
        # （2026-09-18 由确认流用例实测暴露）。
        agent.bind_lease_guard(lambda: None)
        if memory_on:
            self._seed_memory(agent, case, user_id)
        if self.resilient:
            from app.llm.client import ResilientLLM

            ResilientLLM(
                agent.client, settings.model_name,
                fallback_model=settings.llm_fallback_model,
            ).install()
        return agent

    def _seed_memory(self, agent, case: EvalCase, user_id: str) -> None:
        """向沙箱 Agent 的 LTM 播种（文件存储，独立 memory_dir）。

        播种只写存储层（不走 LLM 提取）——记忆用例度量的是注入/召回行为，
        而非提取质量；ltm 实例内 facts 与落盘保持一致，保证当轮即可注入。
        """
        from app.agent.memory.long_term import LongTermMemory
        from app.agent.memory.models import MemoryFact

        memory_dir = self.tmp_root / f"{case.id}_memory"
        ltm = LongTermMemory(
            user_id=user_id, memory_dir=str(memory_dir),
            max_facts=settings.max_ltm_facts,
        )
        ltm.load()
        facts = [
            MemoryFact.from_dict({**item, "status": item.get("status", "active")})
            for item in (case.seed_ltm_facts or [])
            if isinstance(item, dict) and item.get("content")
        ]
        if facts:
            ltm.facts = facts
            ltm.save()
        manager = agent.memory_manager
        # 沙箱 Agent 构造时用的是默认 memory_dir；换成用例专属并重载
        manager.ltm = ltm

    def _make_state_factory(self):
        """请求级 ToolTurnState 工厂：on_outcomes 按稳定顺序采集工具观察。

        trace 在 self._current_trace（run() 设定）上累积；每轮 chat() 新建
        state（pod 单例绝不持有请求状态）。
        """
        def factory():
            from app.agent.tools.batch_executor import ToolTurnState
            return ToolTurnState(
                on_outcomes=lambda outcomes: self._current_trace.tool_observations.extend(
                    ToolObservation(
                        name=o.name,
                        arguments=dict(o.arguments),
                        result=o.result,
                        outcome=_structured_outcome(o.result),
                    )
                    for o in outcomes
                )
            )
        return factory

    def run(self, case: EvalCase) -> RunTrace:
        """跑一条用例，返回采集到的运行轨迹。"""
        trace = RunTrace(case_id=case.id, turns=list(case.turns))
        # 每用例独立会话：同一 Sandbox 实例重跑同 case 时，固定命名的
        # session 文件会带着上次的会话历史进入本轮（复现性依赖此清理）
        self._reset_session(case)
        session_path = self.session_path_for(case.id)
        self._current_trace = trace

        agent = None
        patches: list[tuple] = []  # (obj, attr, original) 供还原
        try:
            agent = self._build_agent(session_path, case)
            self._instrument(agent, trace, patches)

            result = None
            for turn in case.turns:
                result = agent.chat(turn)
                # 改造三：每轮后收来源集合（并集）与引用 verdict（最后轮生效）。
                # 单 Agent 重构后这些观测全部落在 AgentTurnContext——沙箱只读
                # _last_turn_ctx，不再摸 _turn_state / _last_citation_verdict
                # 等私有属性（后者已被移除，读 None 会让引用用例全数假阴性）
                turn_ctx = getattr(agent, "_last_turn_ctx", None)
                if turn_ctx is not None:
                    if getattr(turn_ctx, "sources", None):
                        trace.retrieved_sources = sorted(
                            set(trace.retrieved_sources) | set(turn_ctx.sources)
                        )
                    if getattr(turn_ctx, "citation_verdict", None) is not None:
                        trace.citation_verdict = turn_ctx.citation_verdict
                    # ReAct 步数余量感知（修改5）：步数/纠错按轮累计；
                    # 预告与强制终答任一轮触发即真
                    trace.react_steps += turn_ctx.react_steps
                    trace.steps_margin_hint = (
                        trace.steps_margin_hint or turn_ctx.steps_margin_hint
                    )
                    trace.forced_finalize = (
                        trace.forced_finalize or turn_ctx.forced_finalize
                    )
                    trace.protocol_corrections += turn_ctx.protocol_corrections
                if result is not None:
                    # 全轮回复留痕：敏感泄露检查覆盖每一轮（多轮套取类用例
                    # 中间轮泄露此前不设防）；末轮指标仍读 final_response
                    trace.turn_replies.append(result.reply)
            trace.final_response = result

        except Exception as e:  # noqa: BLE001 —— 单条用例异常不应中断整轮评估
            trace.error = f"{type(e).__name__}: {e}"
        finally:
            for obj, attr, original in patches:
                setattr(obj, attr, original)
            if agent is not None:
                self._close_tool_managers(agent)
                # 修复计划：沙箱自建 ToolBatchExecutor，由其负责 close
                executor = getattr(agent, "_tool_executor", None)
                if executor is not None:
                    try:
                        executor.close()
                    except Exception:  # noqa: BLE001
                        pass
            # 注意：刻意不调用 agent.close()，避免长期记忆巩固写入

        return trace

    # ---------- 插桩 ----------
    def _instrument(self, agent, trace: RunTrace, patches: list[tuple]) -> None:
        """给共享 client 与各 ToolManager 打补丁。"""
        # 1) LLM client：create + beta.parse
        completions = agent.client.chat.completions
        patches.append((completions, "create", completions.create))
        completions.create = self._wrap_create(completions.create, trace)

        beta_completions = agent.client.beta.chat.completions
        patches.append((beta_completions, "parse", beta_completions.parse))
        beta_completions.parse = self._wrap_parse(beta_completions.parse, trace)

        # 2) 工具执行：不再包裹 execute_tool——改造二后工具经 ToolBatchExecutor
        #    执行（并行/超时参数变化），观察由 on_outcomes 钩子按模型顺序采集
        #    （见 _make_state_factory）。

    def _wrap_create(self, original, trace: RunTrace):
        def wrapper(*args, **kwargs):
            start = time.time()
            response = original(*args, **kwargs)
            latency_ms = (time.time() - start) * 1000
            self._record_llm_call(trace, response, latency_ms, purpose="react")
            return response
        return wrapper

    def _wrap_parse(self, original, trace: RunTrace):
        def wrapper(*args, **kwargs):
            start = time.time()
            response = original(*args, **kwargs)
            latency_ms = (time.time() - start) * 1000
            self._record_llm_call(trace, response, latency_ms, purpose="extract")
            return response
        return wrapper

    # ---------- 辅助 ----------
    @staticmethod
    def _record_llm_call(trace: RunTrace, response, latency_ms: float, purpose: str) -> None:
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0
        # 推理模型适配 T9：推理 token 进轨迹，报告/指纹才能区分「思考成本」
        reasoning_tokens = _reasoning_tokens(usage)

        tool_calls: list[dict] = []
        try:
            message = response.choices[0].message
            for tc in (getattr(message, "tool_calls", None) or []):
                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                })
        except (AttributeError, IndexError):
            pass

        model = getattr(response, "model", "") or ""
        trace.llm_calls.append(LLMCallRecord(
            purpose=purpose,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tool_calls=tool_calls,
            latency_ms=latency_ms,
            reasoning_tokens=reasoning_tokens,
        ))


    def _tool_managers(self, agent) -> list:
        if hasattr(agent, "tool_manager"):
            return [agent.tool_manager]
        return []

    def _close_tool_managers(self, agent) -> None:
        for tm in self._tool_managers(agent):
            try:
                tm.close()
            except Exception:  # noqa: BLE001
                pass


def _reasoning_tokens(usage) -> int:
    """与 llm/client._reasoning_tokens 同口径（沙箱不能依赖韧性层已安装）。"""
    if usage is None:
        return 0
    details = getattr(usage, "completion_tokens_details", None)
    value = getattr(details, "reasoning_tokens", 0) if details is not None else 0
    if not value:
        value = getattr(usage, "reasoning_tokens", 0)
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0
