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
from typing import Any, Callable, Optional


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
    # 修复计划·一：会话租约校验回调（写工具提交前调用；失效抛 SessionLockLost）。
    # 由路由层在持有 SessionLease 期间注入，Agent 的 save/reset 同样经此校验。
    lease_guard: Optional[Callable[[], None]] = field(default=None, repr=False)
    # 写操作两阶段协议（P1-2，工具层强制用户确认）：
    # - pending_write：本会话待确认的写草稿（SessionState 持久化记录，Agent 每轮
    #   装载；工具只读，不直接改写）；
    # - write_confirm：本轮用户消息的表态判定 none|confirm|cancel|ambiguous
    #   （Agent 每轮用 write_gate.judge_write_confirmation 计算后注入）；
    # - persist_pending_write：登记/清除草稿的回调（Agent 提供，与消息同一
    #   save 事务落库）。三者缺一即视为「无确认能力」——写工具 fail-closed，
    #   只登记草稿不落库。
    pending_write: Optional[dict] = None
    write_confirm: str = "none"
    # 本轮是否已真正执行过写（P1-2）：同一轮内模型重复调用写工具时防止二次提交
    # ——确认轮里每次调用都会生成新幂等键，仅靠网关幂等挡不住「同轮两次提交」。
    write_executed: bool = False
    persist_pending_write: Optional[Callable[[Optional[dict]], None]] = field(
        default=None, repr=False,
    )
