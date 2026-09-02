"""ToolContext：工具调用的请求级上下文（阶段一 1.3）。

从前「set_memory_manager/set_skill_manager 全局单例 + 工具函数直接读全局」，
一个进程只能服务一个用户；现在每个工具调用都携带 ctx，天然支持多用户并发。

ctx 里只放「工具执行所需的请求级状态」，不放模块级资源：
- user_id / session_id：归属与隔离依据
- memory / skill_manager：由所属 Agent 实例注入（每个 Agent 自己的实例）
- credentials：阶段三 3.2 引入的外部凭证（订单/退款后端），永不进 prompt/日志
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolContext:
    """一次工具调用所处的请求上下文。"""

    user_id: str
    session_id: str = ""
    memory: Optional[Any] = None  # MemoryManager（仅类型标注，避免运行时循环导入）
    skill_manager: Optional[Any] = None  # SkillManager（同上）
    credentials: Optional[dict] = field(default=None, repr=False)  # 3.2 外部凭证
    # 2.2 订单归属强制开关（请求级覆盖）：None=跟随全局配置；
    # 生产由配置注入，评测沙箱显式设为 True，避免测试依赖全局 .env。
    enforce_order_ownership: Optional[bool] = None
