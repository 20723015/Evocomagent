"""app.stores：状态外置（阶段二 2.1~2.4）。

- SessionStore：会话文档的 load/save/delete（CAS 乐观锁，写冲突抛 SessionConflictError）
- LTMStore：长期记忆外置（memory:{user_id}）——Redis 版在 Prod 使用，本地文件版开发用
- SessionLockManager：同一会话同一时刻只允许一个写入者（SETNX + TTL + 续期）
- ObjectStore：turns 归档 / 知识文档的对象存储抽象
"""

from app.stores.base import SessionConflictError, SessionState, SessionStore, LTMStore, ObjectStore
from app.stores.locks import SessionLease, SessionLockManager
from app.stores.memory_store import LocalFileLTMStore, RedisLTMStore
from app.stores.object_store import LocalDirObjectStore, S3ObjectStore
from app.stores.session_store import LocalFileSessionStore, RedisSessionStore

__all__ = [
    "SessionConflictError",
    "SessionState",
    "SessionStore",
    "LTMStore",
    "ObjectStore",
    "SessionLease",
    "SessionLockManager",
    "LocalFileLTMStore",
    "RedisLTMStore",
    "LocalDirObjectStore",
    "S3ObjectStore",
    "LocalFileSessionStore",
    "RedisSessionStore",
]
