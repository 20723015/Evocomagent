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
    """查询当前用户的长期记忆信息（跨会话事实库）。

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
            # 记忆系统重构·阶段1.2：双时态语义透出（只读，无行为变化）——
            # created_at 即 valid_from；superseded/deleted 的 updated_at 即
            # invalid_at；recall 只读 active 故 invalid_at 恒空
            "valid_from": getattr(f, "created_at", ""),
            "invalid_at": "",
        }
        for f in ranked
    ]

    # 记忆系统重构·阶段0：漏注直接度量——Agent 主动召回命中、而本轮自动注入
    # 未命中的 fact_id（注入漏召 = recall 命中集 − 注入快照）
    if query and query.strip():
        try:
            import logging

            from app.observability.metrics import record_memory_recall_miss

            injected = getattr(manager.ltm, "_last_injected", {}) or {}
            missed = [f.fact_id for f in ranked if f.fact_id not in injected]
            if missed:
                record_memory_recall_miss(len(missed))
                logging.getLogger("app.agent.tools.memory_tool").debug(
                    "memory.recall_miss user=%s missed_fact_ids=%s",
                    getattr(manager.ltm, "user_id", ""), missed,
                )
        except Exception:  # noqa: BLE001 —— 度量失败不影响工具返回
            pass

    result: dict = {
        "success": True,
        "query": query,
        "long_term_facts": long_term_facts,
    }

    if manager.ltm.interaction_summaries:
        result["recent_interactions"] = [
            s["summary"] for s in manager.ltm.interaction_summaries[-3:]
        ]

    return result
