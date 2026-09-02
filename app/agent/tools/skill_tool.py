"""技能加载工具：让 Agent 在 ReAct 循环中按需加载 Skill 指令。

模式与 memory_tool.py 一致：阶段一 1.3 起由 ToolContext 注入 SkillManager，
无 ctx 时回落全局管理器（旧脚本兼容）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from app.agent.context import ToolContext

if TYPE_CHECKING:
    from app.agent.skills.loader import SkillManager

_skill_manager: SkillManager | None = None


def set_skill_manager(manager: SkillManager) -> None:
    """旧注入方式：仅兼容无 ctx 的历史调用，生产路径不再使用。"""
    global _skill_manager
    _skill_manager = manager


def load_skill(skill_name: str, ctx: Optional[ToolContext] = None) -> dict:
    """加载指定技能的完整指令。Agent 调用后按指令处理用户问题。"""
    manager = ctx.skill_manager if ctx is not None and ctx.skill_manager is not None else None
    if manager is None:
        manager = _skill_manager
    if manager is None:
        return {"success": False, "error": "技能系统未启用"}
    return manager.load_skill(skill_name)
