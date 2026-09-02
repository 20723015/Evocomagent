"""存储抽象：SessionStore / LTMStore / ObjectStore 协议与共享数据结构。

设计原则：Agent 只依赖协议，不关心背后是本地文件还是 Redis/对象存储；
实现可替换（阶段二 2.1），生产用 Redis 版，开发/测试用文件版。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


class SessionConflictError(Exception):
    """乐观锁冲突：保存时的版本与当前不一致，调用方应 409 或重读重试。"""


@dataclass
class SessionState:
    """一份会话文档（与磁盘/Redis 上的 v2 格式一一对应）。"""

    session_id: str = ""
    user_id: str = ""  # 阶段三 3.1：session 归属校验（旧数据为空串 → 不校验）
    summary: Optional[str] = None
    messages: list[dict] = field(default_factory=list)
    short_term_memory: Optional[dict] = None
    version: int = 0  # 乐观锁版本号（乐观锁 CAS 依据）
    consolidated_len: int = 0  # 安全修复 P2：增量巩固水位（持久化防重启重复巩固）
    updated_at: str = ""


class SessionOwnershipError(Exception):
    """session 不属于当前用户（3.1：session.user_id ≠ token sub → 403）。"""


class StorageUnavailableError(Exception):
    """外置存储不可用（Redis 断连/超时）——调用方映射 503（5.2 故障注入）。"""


@runtime_checkable
class SessionStore(Protocol):
    """会话文档存储协议（2.1）。save 携带期望版本做 CAS，冲突抛 SessionConflictError。"""

    def load(self, user_id: str, session_id: str) -> Optional[SessionState]:
        """读取会话；不存在或损坏返回 None（降级为新会话）。"""
        ...

    def save(
        self, user_id: str, session_id: str, state: SessionState,
        new_messages: Optional[list[dict]] = None,
    ) -> SessionState:
        """写入会话（CAS）。state.version 是期望的当前版本；返回带上新版本的状态。

        new_messages：本轮新增消息（阶段八 SQL 行式追加用，append-only 正本
        不随压缩丢弃）；整包覆写型实现忽略该参数（沿用 state.messages）。
        """
        ...

    def delete(self, user_id: str, session_id: str) -> None:
        ...


@runtime_checkable
class LTMStore(Protocol):
    """长期记忆外置存储：payload 与 LongTermMemory 的 JSON 结构一致。"""

    def load(self, user_id: str) -> Optional[dict]:
        """读取 payload；无记录返回 None。"""
        ...

    def save(self, user_id: str, payload: dict) -> None:
        ...


@runtime_checkable
class LTMStoreMerge(Protocol):
    """原子读-改-写能力（安全修复 P2，可选实现）。

    merge(user_id, merger)：merger 接收当前 payload（无记录为 None），返回
    合并后的完整 payload；实现须保证「读-改-写」对同用户串行（锁/事务/
    WATCH-MULTI）。LongTermMemory 经 getattr 探测该能力，未实现则退回
    load/save 整包写。
    """

    def merge(self, user_id: str, merger) -> dict:
        ...


@runtime_checkable
class ObjectStore(Protocol):
    """对象存储抽象（知识文档 / turns 归档 / 上传分片 / 原件，2.4/2.8）。
    key 为相对路径（prefix 用 / 分隔）。"""

    def put(self, key: str, data: bytes) -> None:
        ...

    def get(self, key: str) -> Optional[bytes]:
        ...

    def list(self, prefix: str = "") -> list[str]:
        ...

    def delete(self, key: str) -> None:
        """删除单对象（幂等：不存在不报错）。"""

    def delete_prefix(self, prefix: str) -> None:
        """删除 prefix 下的全部对象（幂等清理，2.8）。"""

    def healthcheck(self) -> bool:
        """连通性探活（/readyz 用）：可读可写返回 True。"""
