"""子 Agent 定义：每个子 Agent 有专属的 system prompt 和工具子集。

SubAgent 封装了一个轻量级 ReAct 循环，由 Orchestrator 调度执行。
"""

from typing import Optional

from openai import OpenAI

from app.agent.context import ToolContext
from app.agent.tools.batch_executor import ToolBatchExecutor, ToolTurnState
from app.agent.turn_budget import LLMBudgetExhausted
from app.config.settings import settings
from app.observability.logging import get_logger
from app.prompts.agents import COMPLAINT_PROMPT, POSTSALE_PROMPT, PRESALE_PROMPT
from app.agent.tools.manager import ToolManager

log = get_logger("app.multi_agent.agents")


AGENT_CONFIGS = {
    "presale": {
        "name": "小夕-售前",
        "prompt": PRESALE_PROMPT,
        "tools": {"query_product", "search_knowledge", "list_user_orders", "load_skill"},
    },
    "postsale": {
        "name": "小夕-售后",
        "prompt": POSTSALE_PROMPT,
        "tools": {
            "query_order", "query_logistics", "apply_refund",
            "list_user_orders", "search_knowledge", "load_skill",
        },
    },
    "complaint": {
        "name": "小夕-投诉",
        "prompt": COMPLAINT_PROMPT,
        "tools": {"query_order", "search_knowledge", "load_skill"},
    },
}


class SubAgent:
    """专业子 Agent：拥有独立的 prompt 和工具子集，执行 ReAct 循环。"""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tool_manager: ToolManager,
        client: OpenAI,
        model: str,
        temperature: float,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tool_manager = tool_manager
        self.client = client
        self.model = model
        self.temperature = temperature

    def handle(
        self, messages: list[dict], ctx: Optional[ToolContext] = None, *,
        max_steps: Optional[int] = None,
        executor: Optional[ToolBatchExecutor] = None,
        state: Optional[ToolTurnState] = None,
        budget=None,
    ) -> tuple[str, list[dict], int]:
        """执行 ReAct 循环，返回 (最终文本, 新增消息列表, 实际步数)。

        Agent能力强化计划：
        - max_steps 缺省读 settings.max_react_steps（不再硬编码 5，改造一）；
        - 工具批次与主 Agent 共用 ToolBatchExecutor/ToolTurnState（改造二收敛，
          来源集合/事件/重复拦截/预算规则同约定）；
        - budget 为轮次预算（改造一），每步与强制收尾前检查，耗尽抛
          LLMBudgetExhausted（上层零 LLM fallback）；
        - 步数随返回值上抛，orchestrator 累计回填（分布指标不再恒为 1）。
        """
        steps_limit = (
            settings.max_react_steps if max_steps is None else max_steps
        )
        executor = executor or ToolBatchExecutor()
        state = state if state is not None else ToolTurnState()
        new_messages: list[dict] = []
        working = list(messages)
        steps = 0

        for _ in range(steps_limit):
            steps += 1
            if budget is not None and budget.expired():
                raise LLMBudgetExhausted("轮次预算耗尽，停止 ReAct 循环")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=working,
                temperature=self.temperature,
                tools=self.tool_manager.tool_definitions,
            )
            assistant_msg = response.choices[0].message

            if assistant_msg.content:
                self._print_thought(assistant_msg.content)

            if not assistant_msg.tool_calls:
                content = assistant_msg.content or ""
                msg = {"role": "assistant", "content": content}
                new_messages.append(msg)
                return content, new_messages, steps

            msg_dict: dict = {"role": "assistant", "content": assistant_msg.content}
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
            new_messages.append(msg_dict)
            working.append(msg_dict)

            outcomes = executor.execute(
                [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in assistant_msg.tool_calls
                ],
                state, ctx, self.tool_manager, budget=budget,
            )
            for oc in outcomes:
                self._print_action(oc.name, oc.arguments)
                self._print_observation(oc.result)
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": oc.call_id,
                    "content": oc.result,
                }
                new_messages.append(tool_msg)
                working.append(tool_msg)

        if budget is not None and budget.expired():
            raise LLMBudgetExhausted("轮次预算耗尽，强制收尾被拒绝")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=working,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content or ""
        new_messages.append({"role": "assistant", "content": content})
        return content, new_messages, steps

    def _print_thought(self, text: str) -> None:
        log.info(f"\n  💭 [{self.name}·思考] {text}")

    def _print_action(self, func_name: str, func_args: dict) -> None:
        args_str = ", ".join(f"{k}={v!r}" for k, v in func_args.items())
        log.info(f"  🔧 [{self.name}·调用工具] {func_name}({args_str})")

    def _print_observation(self, result: str) -> None:
        display = result if len(result) <= 300 else result[:300] + "..."
        log.info(f"  📋 [{self.name}·工具结果] {display}")
