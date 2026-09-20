"""app.stores：状态外置（阶段二 2.1~2.4）。

- SessionStore：会话文档的 load/save/delete（CAS 乐观锁，写冲突抛 SessionConflictError）
- LTMStore：长期记忆外置——SQL 为唯一正本，Redis 只做 cache-aside 读缓存
  （CachedLTMStore）；文件版仅供开发/测试
- SessionLockManager：同一会话同一时刻只允许一个写入者（SETNX + TTL + 续期）
- ObjectStore：turns 归档 / 知识文档的对象存储抽象
"""

from app.stores.base import SessionConflictError, SessionState, SessionStore, LTMStore, ObjectStore
from app.stores.locks import (
    SessionLease,
    SessionLockBackendUnavailable,
    SessionLockLost,
    SessionLockManager,
)
from app.stores.memory_store import CachedLTMStore, LocalFileLTMStore, RedisLTMStore
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
    "SessionLockBackendUnavailable",
    "SessionLockLost",
    "CachedLTMStore",
    "LocalFileLTMStore",
    "RedisLTMStore",
    "LocalDirObjectStore",
    "S3ObjectStore",
    "LocalFileSessionStore",
    "RedisSessionStore",
]
