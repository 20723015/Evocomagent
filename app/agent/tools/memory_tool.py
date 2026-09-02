"""记忆查询工具：让 Agent 在 ReAct 循环中主动查询用户记忆。

阶段一 1.3：生产路径改为「ctx 注入」——ToolContext 携带所属 Agent 的
MemoryManager，多用户时各自隔离；全局 set_memory_manager 仅保留给旧版
脚本/CLI 直调（无 ctx 时降级回落），新代码一律走 ctx。

Agent能力强化计划·改造四：query 非空 → 按相关性公式（Dice + 类别权重 +
新近度）返回 top-10；query 为空 → 全量（兼容旧行为）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from app.agent.context import ToolContext

if TYPE_CHECKING:
    from app.agent.memory.manager import MemoryManager

_memory_manager: MemoryManager | None = None


def set_memory_manager(manager: MemoryManager) -> None:
    """旧注入方式：仅兼容无 ctx 的历史调用（tests/ 顶层脚本等），生产路径不再使用。"""
    global _memory_manager
    _memory_manager = manager


def recall_user_memory(query: str = "", ctx: Optional[ToolContext] = None) -> dict:
    """查询当前用户的记忆信息（长期记忆和短期记忆）。

    ctx 为 None 或 ctx.memory 为空时回落到全局管理器（旧脚本兼容）。
    """
    manager = ctx.memory if ctx is not None and ctx.memory is not None else None
    if manager is None:
        manager = _memory_manager

    if manager is None or not manager.memory_enabled:
        return {"success": False, "error": "记忆系统未启用"}

    if query and query.strip():
        ranked = manager.ltm.ranked_facts(query, limit=10)
    else:
        ranked = getattr(manager.ltm, "active_facts", manager.ltm.facts)
    long_term_facts = [
        {
            "content": f.content,
            "category": f.category,
            "fact_key": getattr(f, "fact_key", ""),
            "updated_at": getattr(f, "updated_at", ""),
        }
        for f in ranked
    ]

    result: dict = {
        "success": True,
        "query": query,
        "short_term_facts": manager.stm.facts,
        "long_term_facts": long_term_facts,
    }

    if manager.ltm.interaction_summaries:
        result["recent_interactions"] = [
            s["summary"] for s in manager.ltm.interaction_summaries[-3:]
        ]

    return result
