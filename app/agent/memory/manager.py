"""记忆管理器：长期记忆门面。

EcomAgent 通过此管理器与记忆系统交互。

记忆系统重构后收敛为业界两层：
- 会话内上下文 = 对话历史 + rolling 摘要（Agent 自身承担，不经本模块）；
- 跨会话个性化 = per-user LTM 事实库（注入 + 异步巩固，本模块唯一职责）。

原会话内「槽位层」（ShortTermMemory/stm_rules 确定性规则提取）已整体删除：
槽位是历史/摘要的冗余副本，且与 LTM 单值键冲突（改名后新旧值同屏）；
身份类信息由 LTM identity 单值键每轮必注入承担（见 long_term.py）。
"""

from __future__ import annotations

from openai import OpenAI

from app.agent.memory.long_term import LongTermMemory


def _derive_embedding_store(ltm_store, memory_dir: str):
    """记忆系统重构·阶段2：按 LTM 存储后端派生嵌入存储（同介质，不混写）。

    语义关闭时返回 None 也无妨——LongTermMemory._semantic_active() 已
    短路；这里常驻构建只是让开启语义的部署零额外接线。

    cache-aside 包装层（CachedLTMStore）自身不持有介质信息，先解包到正本，
    否则 SQL 部署会因 engine 缺失而把嵌入落到 Redis（与事实正本异介质）。
    """
    from app.agent.memory.embeddings import build_memory_embedding_store

    base = getattr(ltm_store, "_inner", ltm_store)
    engine = getattr(base, "_engine", None)
    redis = getattr(base, "_redis", None)
    try:
        return build_memory_embedding_store(
            engine=engine, redis=redis, memory_dir=memory_dir,
        )
    except Exception:  # noqa: BLE001 —— 派生失败 = 无语义路（降级词面）
        return None


class MemoryManager:
    """记忆管理器：长期记忆门面。"""

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
        embedding_store=None,  # 记忆系统重构·阶段2：显式嵌入存储（可选，缺省按 ltm_store 派生）
    ):
        self.client = client
        self.model = model
        self.memory_enabled = memory_enabled

        self.ltm = LongTermMemory(
            user_id=user_id,
            memory_dir=memory_dir,
            max_facts=max_ltm_facts,
            store=ltm_store,
            source_session=session_id,
            embedding_store=(
                embedding_store
                if embedding_store is not None
                else _derive_embedding_store(ltm_store, memory_dir)
            ),
        )

        if self.memory_enabled:
            self.ltm.load()

    def bind_session(self, session_id: str) -> None:
        """Bind the current session for memory audit provenance."""
        self.ltm.source_session = session_id

    def build_memory_prompt_sections(
        self, query: str = "", query_embedding: list[float] | None = None,
    ) -> list[dict]:
        """生成所有记忆相关的 system prompt 消息列表。

        改造四：query = 本轮原始 user_input（chat() 显式传入，不从消息尾部
        猜测）；LTM 注入按相关性筛选（严格 ≤8 条 + identity 单值键保底）。
        阶段2：query_embedding 由 ContextBuilder 每轮至多计算一次传入。
        """
        if not self.memory_enabled:
            return []
        sections = []
        ltm_section = self.ltm.build_prompt_section(
            query, query_embedding=query_embedding,
        )
        if ltm_section:
            sections.append({"role": "system", "content": ltm_section})
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

    def reset_all(self) -> None:
        """重置长期记忆（会话 reset 时调用）。"""
        self.ltm.reset()
