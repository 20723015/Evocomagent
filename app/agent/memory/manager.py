"""记忆管理器：统一管理短期记忆和长期记忆。

EcomAgent 和 MultiAgentOrchestrator 通过此管理器与记忆系统交互。
"""

from __future__ import annotations

from openai import OpenAI

from app.agent.memory.long_term import LongTermMemory
from app.agent.memory.short_term import ShortTermMemory


class MemoryManager:
    """记忆管理器：统一管理短期记忆和长期记忆。"""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        user_id: str = "default",
        memory_dir: str = "app/sessions/memory",
        memory_enabled: bool = True,
        max_ltm_facts: int = 50,
        ltm_store=None,  # 阶段二 2.3：LTMStore（Redis 外置）
        session_id: str = "",
    ):
        self.client = client
        self.model = model
        self.memory_enabled = memory_enabled

        self.stm = ShortTermMemory()
        self.ltm = LongTermMemory(
            user_id=user_id,
            memory_dir=memory_dir,
            max_facts=max_ltm_facts,
            store=ltm_store,
            source_session=session_id,
        )

        if self.memory_enabled:
            self.ltm.load()

    def bind_session(self, session_id: str) -> None:
        """Bind the current session for memory audit provenance."""
        self.ltm.source_session = session_id

    def update_short_term(self, recent_messages: list[dict]) -> None:
        """每轮对话后更新短期记忆（LLM 路径；阶段F后仅离线工具使用）。

        主链路已切换为 update_short_term_deterministic（零 LLM）。
        """
        if not self.memory_enabled:
            return
        self.stm.update(self.client, self.model, recent_messages)

    def update_short_term_deterministic(
        self, recent_messages: list[dict], query: str = "",
    ) -> None:
        """阶段F：确定性规则即时提取会话槽位（零 LLM，不抛错）。"""
        if not self.memory_enabled:
            return
        from app.agent.memory.stm_rules import extract_stm_slots

        changes = extract_stm_slots(
            recent_messages,
            [fact for fact in self.stm.records if fact.status == "active"],
        )
        if not changes:
            return
        from app.agent.memory.models import MemoryMutation, apply_memory_mutations

        if all(isinstance(item, MemoryMutation) for item in changes):
            self.stm.records = apply_memory_mutations(
                self.stm.records, changes, max_active=50,
            )
        else:
            self.stm.facts = changes

    def build_memory_prompt_sections(self, query: str = "") -> list[dict]:
        """生成所有记忆相关的 system prompt 消息列表。

        改造四：query = 本轮原始 user_input（chat() 显式传入，不从消息尾部
        猜测）；LTM 注入按相关性筛选（严格 ≤8 条 + 保底），STM/交互摘要不变。
        """
        if not self.memory_enabled:
            return []

        sections = []
        ltm_section = self.ltm.build_prompt_section(query)
        if ltm_section:
            sections.append({"role": "system", "content": ltm_section})
        stm_section = self.stm.build_prompt_section()
        if stm_section:
            sections.append({"role": "system", "content": stm_section})
        return sections

    def consolidate_to_long_term(
        self, messages: list[dict], summary: str | None,
    ) -> None:
        """会话结束时，将本次对话的关键事实巩固到长期记忆。"""
        if not self.memory_enabled:
            return
        self.ltm.extract_and_save(
            self.client, self.model, messages, summary,
        )

    def reset_short_term(self) -> None:
        """重置短期记忆（会话内重置时调用）。"""
        self.stm.reset()

    def reset_all(self) -> None:
        """重置所有记忆（短期+长期）。"""
        self.stm.reset()
        self.ltm.reset()

    def stm_to_dict(self) -> dict:
        return self.stm.to_dict()

    def restore_stm(self, data: dict) -> None:
        self.stm = ShortTermMemory.from_dict(data)
