"""会话锁（阶段二 2.2）：同一会话同一时刻只有一个写入者。

场景：用户「连点」、双 pod 同时服务同一 session 的竞态。
- Redis 版：SET NX EX —— 跨 pod 一致；lease 期间后台续期，防单轮超时误释放。
- 生产 redis_required=true：Redis 获取/续租/所有权校验异常一律 fail-closed，
  抛 SessionLockBackendUnavailable（HTTP 503）——不做本地降级，避免多 Pod 各写各的。
- 单机开发模式（默认）：保留进程内锁降级（同进程互斥；跨 pod 一致性是 Redis 的职责）。

租约语义（SessionLease）：
- TTL 过期/被抢占/续租失败 → 标记 lost，后续 assert_owned() 抛 SessionLockLost；
- release() 幂等；handover() 把租约移交给后台收尾任务后，本对象不再释放。
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Optional

LOCK_PREFIX = "session_lock:"
DEFAULT_TTL_SECONDS = 300  # ≥ 单轮最长耗时的经验值；lease 会自动续期


class SessionLockBackendUnavailable(RuntimeError):
    """锁后端（Redis）不可用：redis_required 下 fail-closed（→ HTTP 503）。"""


class SessionLockLost(RuntimeError):
    """租约已失效：TTL 过期、被抢占或续租失败（禁止再提交副作用）。"""


class _InProcessLocks:
    """进程内互斥锁表：按 lock key 复用 threading.Lock，并记录持有者令牌。"""

    def __init__(self):
        self._locks: dict[str, threading.Lock] = {}
        self._owners: dict[int, str] = {}  # id(lock) → 持有者令牌
        self._guard = threading.Lock()

    def get(self, key: str) -> threading.Lock:
        with self._guard:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def set_owner(self, lock: threading.Lock, token: str) -> None:
        with self._guard:
            self._owners[id(lock)] = token

    def owner(self, lock_id: int) -> Optional[str]:
        with self._guard:
            return self._owners.get(lock_id)

    def clear_owner(self, lock_id: int) -> None:
        with self._guard:
            self._owners.pop(lock_id, None)

    def lock_for_id(self, lock_id: int) -> threading.Lock:
        with self._guard:
            for lock in self._locks.values():
                if id(lock) == lock_id:
                    return lock
        raise KeyError(lock_id)


_in_process = _InProcessLocks()


class SessionLockManager:
    """取得/释放会话锁。redis_required 时 Redis 故障 fail-closed，否则进程内降级。"""

    def __init__(
        self,
        redis=None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        redis_required: bool = False,
    ):
        self._redis = redis
        self._ttl = ttl_seconds
        self._required = redis_required

    @property
    def redis_required(self) -> bool:
        return self._required

    def acquire(self, user_id: str, session_id: str) -> Optional[str]:
        """尝试获取锁；成功返回 token（lease 续期/释放凭据），失败返回 None（→409）。

        redis_required=true：Redis 不可用/获取异常 → SessionLockBackendUnavailable
        （生产多 Pod 不得降级为进程内锁，否则各 Pod 可同时写同一会话）。
        开发模式（默认）：Redis 故障降级进程内锁——同 pod 内不双写。
        """
        key = f"{LOCK_PREFIX}{user_id}:{session_id}"
        token = uuid.uuid4().hex
        if self._redis is not None:
            try:
                ok = self._redis.set(key, token, nx=True, ex=self._ttl)
                return token if ok else None
            except Exception as e:  # noqa: BLE001 —— 5.2 故障注入
                if self._required:
                    raise SessionLockBackendUnavailable(
                        f"会话锁后端（Redis）不可用：{type(e).__name__}"
                    ) from e
                # 开发模式：降级进程内锁
        elif self._required:
            raise SessionLockBackendUnavailable(
                "会话锁要求 Redis（redis_required=true），但 Redis 不可用"
            )

        lock = _in_process.get(key)
        if not lock.acquire(blocking=False):
            return None
        token = f"{token}:{id(lock)}"  # 进程内令牌，释放/校验时比对
        _in_process.set_owner(lock, token)
        return token

    def release(self, user_id: str, session_id: str, token: Optional[str]) -> None:
        if not token:
            return
        key = f"{LOCK_PREFIX}{user_id}:{session_id}"
        released_redis = False
        if self._redis is not None:
            try:
                # 仅当 token 匹配才删除（防误删他人锁）。
                # eval 返回 0 = 锁不在 Redis（acquire 降级时锁在进程内），
                # 不能 return——必须继续走下面的进程内释放，否则进程内锁泄漏
                deleted = self._redis.eval(
                    "if redis.call('get', KEYS[1]) == ARGV[1] "
                    "then return redis.call('del', KEYS[1]) else return 0 end",
                    1, key, token,
                )
                if deleted:
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
            lock = _in_process.lock_for_id(int(lock_id))
        except (ValueError, KeyError):
            return
        lock.release()
        _in_process.clear_owner(int(lock_id))

    def assert_owned(self, user_id: str, session_id: str, token: Optional[str]) -> None:
        """校验本进程仍持有该会话锁；失效 → SessionLockLost（fail-closed）。

        Redis 键值不匹配（TTL 过期/被抢占）或校验调用异常都按丢失处理——
        宁可拒绝副作用，不可在无锁状态下写。
        """
        if not token:
            raise SessionLockLost("会话租约不存在")
        key = f"{LOCK_PREFIX}{user_id}:{session_id}"
        if ":" not in token:
            if self._redis is None:
                raise SessionLockLost("会话租约校验失败：Redis 不可用")
            try:
                cur = self._redis.get(key)
            except Exception as e:  # noqa: BLE001 —— 校验失败按丢失（fail-closed）
                raise SessionLockLost(f"会话租约校验失败：{type(e).__name__}") from e
            cur_str = cur.decode("utf-8") if isinstance(cur, bytes) else cur
            if cur_str != token:
                raise SessionLockLost("会话租约已丢失（TTL 过期或被抢占）")
            return
        lock_id = int(token.split(":", 1)[-1])
        if _in_process.owner(lock_id) != token:
            raise SessionLockLost("会话租约已丢失（进程内锁已释放/被抢占）")


class SessionLease:
    """会话锁租约：持有期间后台续期（Redis 版），退出时释放。

    - lost 状态：续租返回 0 或异常即标记失效，assert_owned() 抛 SessionLockLost；
    - release() 幂等；
    - handover()：断连时把租约移交给后台收尾任务，本对象不再释放（续期继续，
      直到后台任务真正结束 Agent 后 release）。
    """

    def __init__(self, manager: SessionLockManager, user_id: str, session_id: str):
        self._manager = manager
        self._user_id = user_id
        self._session_id = session_id
        self.token: Optional[str] = None
        self._renewer: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lost = False
        self._released = False
        self._handed_off = False
        self._abandoned = False

    @property
    def lost(self) -> bool:
        return self._lost

    @property
    def handed_off(self) -> bool:
        return self._handed_off

    @property
    def abandoned(self) -> bool:
        """已停止续租且放弃主动释放（剩余 Redis 锁由 TTL 回收）。"""
        return self._abandoned

    def __enter__(self) -> "SessionLease":
        self.token = self._manager.acquire(self._user_id, self._session_id)
        if (
            self.token is not None
            and ":" not in self.token
            and self._manager._redis is not None
        ):
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
                res = redis.eval(
                    "if redis.call('get', KEYS[1]) == ARGV[1] then "
                    "return redis.call('expire', KEYS[1], ARGV[2]) else return 0 end",
                    1, key, self.token, ttl,
                )
            except Exception:  # noqa: BLE001 —— 续租失败：标记丢失，禁止后续副作用
                self._mark_lost()
                return
            if not res:
                # 0 = 键不存在或已被抢占：TTL 到期/他人持有 → 租约失效
                self._mark_lost()
                return

    def _mark_lost(self) -> None:
        self._lost = True
        self._stop.set()

    def assert_owned(self) -> None:
        """副作用前校验：失效 → SessionLockLost（拒绝提交外部副作用/会话状态）。"""
        if self._lost:
            raise SessionLockLost("会话租约已丢失（续租失败/被抢占）")
        self._manager.assert_owned(self._user_id, self._session_id, self.token)

    def release(self) -> None:
        """幂等释放；handover 后由后台收尾任务调用。

        abandoned（关闭超时移交）状态禁止主动删除锁：底层 Agent 线程可能仍在
        运行，删锁会让第二个请求并发写同一会话——剩余 Redis 锁交给 TTL 回收。
        """
        if self._released or self._abandoned:
            return
        self._released = True
        self._stop.set()
        if self._renewer is not None:
            self._renewer.join(timeout=2)
        if self.token is not None:
            self._manager.release(self._user_id, self._session_id, self.token)

    def handover(self) -> "SessionLease":
        """断连移交：本对象不再释放（__exit__ 变成 no-op），续期继续运行。

        后台收尾任务在 Agent 真正结束后调用 release() 完成释放。
        """
        self._handed_off = True
        return self

    def stop_renew(self) -> None:
        """关闭超时移交：只停止续租，绝不主动删锁（TTL 回收）。

        修复计划·二轮 6：应用关闭等待窗口用尽时，底层 Agent 线程可能仍在运行；
        此时若删除锁（或 cancel 会走到 finally 释放锁的收尾任务），第二个请求
        会在无锁状态下并发写同一会话。这里只停止续租并置 abandoned。
        """
        self._handed_off = True
        self._abandoned = True
        self._stop.set()
        if self._renewer is not None:
            self._renewer.join(timeout=2)

    def __exit__(self, exc_type, exc, tb):
        if self._handed_off:
            return False
        self.release()
        return False
