"""会话锁（阶段二 2.2）：同一会话同一时刻只有一个写入者。

场景：用户「连点」、双 pod 同时服务同一 session 的竞态。
- Redis 版：SET NX EX —— 跨 pod 一致；lease 期间后台续期，防单轮超时误释放。
- 无 Redis 时降级进程内锁（同进程互斥；跨 pod 一致性是 Redis 版的职责）。
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Optional

LOCK_PREFIX = "session_lock:"
DEFAULT_TTL_SECONDS = 300  # ≥ 单轮最长耗时的经验值；lease 会自动续期


class _InProcessLocks:
    """进程内互斥锁表：按 lock key 复用 threading.Lock。"""

    def __init__(self):
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def get(self, key: str) -> threading.Lock:
        with self._guard:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]


_in_process = _InProcessLocks()


class SessionLockManager:
    """取得/释放会话锁。Redis 可用用 Redis（跨 pod），否则进程内锁。"""

    def __init__(self, redis=None, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self._redis = redis
        self._ttl = ttl_seconds

    def acquire(self, user_id: str, session_id: str) -> Optional[str]:
        """尝试获取锁；成功返回 token（lease 续期/释放凭据），失败返回 None（→409）。

        Redis 故障（断连/超时）自动降级进程内锁——锁是「同一会话单写者」的
        一致性护栏，进程内互斥保证同 pod 内不双写；跨 pod 一致性交给恢复后的 Redis。
        """
        key = f"{LOCK_PREFIX}{user_id}:{session_id}"
        token = uuid.uuid4().hex
        if self._redis is not None:
            try:
                ok = self._redis.set(key, token, nx=True, ex=self._ttl)
                return token if ok else None
            except Exception:  # noqa: BLE001 —— 5.2 故障注入：Redis 断连降级
                pass

        lock = _in_process.get(key)
        if not lock.acquire(blocking=False):
            return None
        return f"{token}:{id(lock)}"  # 进程内令牌，释放时校验

    def release(self, user_id: str, session_id: str, token: Optional[str]) -> None:
        if not token:
            return
        key = f"{LOCK_PREFIX}{user_id}:{session_id}"
        released_redis = False
        if self._redis is not None:
            try:
                # 仅当 token 匹配才删除（防误删他人锁）
                self._redis.eval(
                    "if redis.call('get', KEYS[1]) == ARGV[1] "
                    "then return redis.call('del', KEYS[1]) else return 0 end",
                    1, key, token,
                )
                return
            except Exception:  # noqa: BLE001 —— EVAL 不可用（如 fakeredis）降级 GET+DEL
                pass
            try:
                cur = self._redis.get(key)
                if cur is not None:
                    cur_str = cur.decode("utf-8") if isinstance(cur, bytes) else cur
                    if cur_str == token:
                        self._redis.delete(key)
                        released_redis = True
            except Exception:  # noqa: BLE001 —— Redis 故障：继续走进程内兜底
                released_redis = False
            if released_redis:
                return
            # 注意：Redis 分支失败不 return——若锁实际在进程内（acquire 降级时
            # 取的），必须走到下面的进程内释放，否则锁泄漏
        if ":" not in token:
            return
        lock_id = token.split(":", 1)[-1]
        try:
            lock = self._locks_for_release(int(lock_id))
        except (ValueError, KeyError):
            return
        lock.release()

    def _locks_for_release(self, lock_id: int):
        for lock in _in_process._locks.values():
            if id(lock) == lock_id:
                return lock
        raise KeyError(lock_id)


class SessionLease:
    """会话锁租约：持有期间后台续期（Redis 版），退出时释放。"""

    def __init__(self, manager: SessionLockManager, user_id: str, session_id: str):
        self._manager = manager
        self._user_id = user_id
        self._session_id = session_id
        self.token: Optional[str] = None
        self._renewer: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def __enter__(self) -> "SessionLease":
        self.token = self._manager.acquire(self._user_id, self._session_id)
        if self.token is not None and self._manager._redis is not None:
            self._renewer = threading.Thread(
                target=self._renew_loop, daemon=True, name="session-lock-renewal",
            )
            self._renewer.start()
        return self

    def _renew_loop(self) -> None:
        key = f"{LOCK_PREFIX}{self._user_id}:{self._session_id}"
        ttl = self._manager._ttl
        interval = max(1.0, ttl / 3)
        redis = self._manager._redis
        while not self._stop.wait(interval):
            try:
                redis.eval(
                    "if redis.call('get', KEYS[1]) == ARGV[1] then "
                    "return redis.call('expire', KEYS[1], ARGV[2]) else return 0 end",
                    1, key, self.token, ttl,
                )
            except Exception:  # noqa: BLE001 —— 续期失败：TTL 到期后锁自动失效
                return

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._renewer is not None:
            self._renewer.join(timeout=2)
        if self.token is not None:
            self._manager.release(self._user_id, self._session_id, self.token)
        return False
