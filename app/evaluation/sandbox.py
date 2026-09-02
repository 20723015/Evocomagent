"""评估沙箱：隔离、可复现地重跑测试集，并采集运行全过程（第9期）。

沙箱做三件事：
1. 隔离：每条用例独立的临时 session 文件；关闭记忆读写（否则 default.json 会注入
   prompt 污染评分）；关闭 MCP 只用本地 mock 工具（保证可复现，且让幻觉检测有确定
   的 ground truth）。
2. 插桩：单/多 Agent 都只共享一个 OpenAI client 实例，给它的
   chat.completions.create / beta.chat.completions.parse 打补丁，即可捕获整个会话
   所有 LLM 调用的 token、被请求的工具、延迟——无需改动 chat.py / orchestrator.py。
   工具返回值则通过包裹 ToolManager.execute_tool 采集。
3. 执行：顺序跑完用例的多轮输入，把过程与结果填进 RunTrace 返回。

关键：绝不调用 agent.close()（会触发长期记忆巩固的 LLM 写入，污染且烧钱）；
所有补丁在 finally 中还原。
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from app.agent.tools.batch_executor import ToolBatchExecutor
from app.config.settings import settings
from app.evaluation.dataset import EvalCase
from app.evaluation.trace import LLMCallRecord, RunTrace, ToolObservation


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

    def __init__(self, mode: str = "single", tmp_root: str | None = None,
                 resilient: bool = False):
        self.mode = mode  # "single" / "multi"
        # resilient=True：给 Agent 的 LLM client 安装韧性层（429/超时/连接错误
        # 指数退避重试 + 降级链，见 app/llm/client.py）。评测路径默认 client
        # max_retries=0，外部 API 限流会直接打穿整条用例；启用后仍受轮次预算约束。
        self.resilient = resilient
        self.tmp_root = Path(tmp_root) if tmp_root else Path(tempfile.mkdtemp(prefix="eval_sandbox_"))
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def session_path_for(self, case_id: str) -> str:
        return str(self.tmp_root / f"{case_id}.json")

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
        """
        user_id = (case.actor_user_id if case is not None else None) or "u1"
        common = dict(
            user_id=user_id,
            session_path=session_path,
            memory_enabled=False,
            use_mcp=False,
            temperature=0.0,
            tool_executor=ToolBatchExecutor(),
            turn_state_factory=self._make_state_factory(),
            enforce_order_ownership=True,
        )
        if self.mode == "multi":
            from app.multi_agent.orchestrator import MultiAgentOrchestrator
            agent = MultiAgentOrchestrator(**common)
        else:
            from app.agent.chat import EcomAgent
            agent = EcomAgent(**common)
        if self.resilient:
            from app.llm.client import ResilientLLM

            ResilientLLM(
                agent.client, settings.model_name,
                fallback_model=settings.llm_fallback_model,
            ).install()
        return agent

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
                # 改造三：每轮后收来源集合（并集）与引用 verdict（最后轮生效）
                turn_state = getattr(agent, "_turn_state", None)
                if turn_state is not None and getattr(turn_state, "sources", None):
                    trace.retrieved_sources = sorted(
                        set(trace.retrieved_sources) | set(turn_state.sources)
                    )
                verdict = getattr(agent, "_last_citation_verdict", None)
                if verdict is not None:
                    trace.citation_verdict = verdict
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
        """给共享 client、各 ToolManager、（多 Agent）Router 打补丁。"""
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

        # 3) 多 Agent 路由
        if self.mode == "multi" and hasattr(agent, "router"):
            patches.append((agent.router, "route", agent.router.route))
            agent.router.route = self._wrap_route(agent.router.route, trace)

    def _wrap_create(self, original, trace: RunTrace):
        def wrapper(*args, **kwargs):
            start = time.time()
            response = original(*args, **kwargs)
            latency_ms = (time.time() - start) * 1000
            self._record_llm_call(
                trace, response, latency_ms,
                purpose=self._guess_purpose(kwargs),
            )
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

    def _wrap_route(self, original, trace: RunTrace):
        def wrapper(*args, **kwargs):
            agent_key = original(*args, **kwargs)
            trace.route = agent_key
            return agent_key
        return wrapper

    # ---------- 辅助 ----------
    @staticmethod
    def _guess_purpose(kwargs: dict) -> str:
        """启发式标注 LLM 调用用途，仅供报告可读，不作硬断言。"""
        if kwargs.get("max_tokens") == 10:
            return "router"
        if kwargs.get("tools"):
            return "react"
        return "react"

    @staticmethod
    def _record_llm_call(trace: RunTrace, response, latency_ms: float, purpose: str) -> None:
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0

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
        ))

    def _tool_managers(self, agent) -> list:
        if self.mode == "multi" and hasattr(agent, "agents"):
            return [a.tool_manager for a in agent.agents.values()]
        if hasattr(agent, "tool_manager"):
            return [agent.tool_manager]
        return []

    def _close_tool_managers(self, agent) -> None:
        for tm in self._tool_managers(agent):
            try:
                tm.close()
            except Exception:  # noqa: BLE001
                pass
