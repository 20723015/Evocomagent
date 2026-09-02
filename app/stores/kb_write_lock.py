"""KB 统一写锁（v7 冻结）：部署期选择后端，运行时不降级。

选择规则（get_kb_write_lock，进程内解析一次；**只按配置/方言，与连通性无关**
——多 Pod 在任何故障时序下锁定同一后端，杜绝锁域分裂）：

    auto + 方言 mysql          → MysqlAdvisoryLock；失败 → 503，不降级
    auto + 非 mysql + Redis    → RenewableRedisLock；失败 → 503，不降级
    auto + 两者皆不可用        → 报错，要求显式配置 kb_write_lock_backend=file
    mysql/redis/file           → 指定后端；file 仅显式配置（单机开发）

语义边界（评审确认，docstring 即承诺）：
- MysqlAdvisoryLock 是「生产级连接租约互斥」：锁生命周期 = 专用连接生命周期，
  进程崩溃/断连由 MySQL 自动释放；**不是严格 fencing**——跨 MySQL/ES 的单调
  epoch 提交不可行，接受「assert_held 与 ES alias 更新之间」的极小窗口。
- RenewableRedisLock 是租约锁：存在「暂停越过租约期再苏醒」的极小竞态，
  assert_held 在移动文件/build/alias 切换前 + alias 切换后拦截；失锁即停。
- 统一接口 acquire(phase)/assert_held()/release()；assert_held 失败抛
  KbWriteLockError——提交点未过可回滚、已过只能等待恢复。
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy import text

from app.config.settings import settings


class KbWriteLockError(RuntimeError):
    """锁获取失败/锁已丢失（映射 503；调用方不得降级到其它锁域）。"""


class KbWriteLockBackendError(KbWriteLockError):
    """后端选择/配置错误（auto 双不可用、指定后端无对应设施）。"""


class MysqlAdvisoryLock:
    """MySQL 连接级互斥：SELECT GET_LOCK 于专用连接，连接存活即持锁。

    - 进程崩溃/网络断开 → MySQL 自动释放（无固定 TTL，不会因超时被抢）；
    - assert_held：IS_USED_LOCK == 本连接 connection_id（连接丢失/锁被抢 → 抛错）；
    - release：RELEASE_LOCK + 显式归还连接（失败由连接关闭兜底）。
    """

    def __init__(self, engine, lock_name: Optional[str] = None, timeout: Optional[int] = None):
        self._engine = engine
        self._lock_name = lock_name or (
            f"ecom-agent:{settings.app_env}:kb_write"
        )
        self._timeout = int(timeout if timeout is not None else settings.kb_write_lock_timeout)
        self._conn = None
        self._connection_id: Optional[int] = None
        self._held = False

    def acquire(self, phase: str = "run") -> None:
        if self._held:
            return
        try:
            conn = self._engine.connect()
            result = conn.execute(
                text("SELECT GET_LOCK(:name, :timeout)"),
                {"name": self._lock_name, "timeout": self._timeout},
            ).scalar()
            if result != 1:
                conn.close()
                raise KbWriteLockError(
                    f"MySQL GET_LOCK 未取得（result={result}），等待 {self._timeout}s 超时"
                )
            cid = conn.execute(text("SELECT CONNECTION_ID()")).scalar()
        except KbWriteLockError:
            raise
        except Exception as e:  # noqa: BLE001 —— 连接失败：fail-closed，不降级
            raise KbWriteLockError(f"MySQL 锁不可用: {e}") from e
        self._conn = conn
        self._connection_id = int(cid or 0)
        self._held = True

    def assert_held(self) -> None:
        if not self._held or self._conn is None:
            raise KbWriteLockError("MySQL 锁未被持有")
        try:
            # IS_USED_LOCK 返回持有者的 server connection id；0 = 无人持有
            holder = self._conn.execute(
                text("SELECT IS_USED_LOCK(:name)"), {"name": self._lock_name},
            ).scalar()
            if holder is None or int(holder or 0) != self._connection_id:
                raise KbWriteLockError("MySQL 锁已丢失（连接失效或锁被释放）")
        except KbWriteLockError:
            raise
        except Exception as e:  # noqa: BLE001
            raise KbWriteLockError(f"MySQL 锁校验失败: {e}") from e

    def release(self) -> None:
        conn, self._conn = self._conn, None
        self._held = False
        if conn is None:
            return
        try:
            conn.execute(
                text("SELECT RELEASE_LOCK(:name)"), {"name": self._lock_name},
            )
        except Exception:  # noqa: BLE001 —— 释放失败由连接关闭兜底（MySQL 自动释放）
            pass
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    # 测试探针
    @property
    def held(self) -> bool:
        return self._held

    @property
    def connection_id(self) -> Optional[int]:
        return self._connection_id


class RenewableRedisLock:
    """Redis 租约锁（一般场景）：SET NX EX + 后台续租；续租失败 → lost。

    token 随机值仅保证「续租/释放只对本人生效」——**不是 fencing token**：
    旧进程暂停越过租约期后苏醒仍可能继续执行，由 assert_held 在副作用点拦截。
    """

    KEY_PREFIX = "kb_write_lock:"

    def __init__(self, redis, key: str = "kb_write", ttl_seconds: Optional[int] = None,
                 lock_name: Optional[str] = None):
        self._redis = redis
        self._key = f"{self.KEY_PREFIX}{key}"
        self._ttl = int(ttl_seconds if ttl_seconds is not None
                        else settings.evolve_lock_stale_seconds)
        self._token = uuid.uuid4().hex
        self._held = False
        self._lost = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ---------- Redis 原语 ----------
    def _renew(self) -> bool:
        try:
            ok = self._redis.eval(
                "if redis.call('get', KEYS[1]) == ARGV[1] then "
                "return redis.call('expire', KEYS[1], ARGV[2]) else return 0 end",
                1, self._key, self._token, self._ttl,
            )
            return bool(ok)
        except Exception:  # noqa: BLE001 —— 续租失败 → lost（EVAL 不可用时不降级）
            return False

    def _renew_loop(self) -> None:
        interval = max(self._ttl / 3, 1.0)
        while not self._stop.wait(interval):
            if not self._renew():
                with self._lock:
                    self._lost = True
                return

    # ---------- 接口 ----------
    def acquire(self, phase: str = "run") -> None:
        if self._held:
            return
        try:
            ok = self._redis.set(self._key, self._token, nx=True, ex=self._ttl)
        except Exception as e:  # noqa: BLE001 —— fail-closed，不降级文件
            raise KbWriteLockError(f"Redis 锁不可用: {e}") from e
        if not ok:
            raise KbWriteLockError(f"KB 写锁被其它运行者持有（{self._key}）")
        self._held = True
        self._lost = False
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._renew_loop, name="kb-lock-renew", daemon=True,
        )
        self._thread.start()

    def assert_held(self) -> None:
        if not self._held:
            raise KbWriteLockError("Redis 锁未被持有")
        if self._lost:
            raise KbWriteLockError("Redis 锁已丢失（续租失败），停止本次写入")
        # 主动校验一次（失锁后可能尚未触发 lost 标记）
        if not self._renew():
            with self._lock:
                self._lost = True
            raise KbWriteLockError("Redis 锁已丢失（续租失败），停止本次写入")

    def release(self) -> None:
        if not self._held:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self._held = False
        try:
            cur = self._redis.get(self._key)
            if cur is not None and (
                cur.decode("utf-8") if isinstance(cur, bytes) else cur
            ) == self._token:
                self._redis.delete(self._key)
        except Exception:  # noqa: BLE001 —— 释放失败靠 TTL 兜底
            pass

    # 测试探针
    @property
    def lost(self) -> bool:
        return self._lost


def resolve_lock_backend(engine, redis) -> str:
    """后端解析（只按配置/方言，不看连通性）——多 Pod 一致性的前提。"""
    cfg = (settings.kb_write_lock_backend or "auto").lower()
    if cfg == "auto":
        if engine is not None and engine.dialect.name == "mysql":
            return "mysql"
        if redis is not None:
            return "redis"
        raise KbWriteLockBackendError(
            "KB_WRITE_LOCK_BACKEND=auto 但 MySQL（方言非 mysql/未配置）与 Redis 均不可用；"
            "请显式配置 KB_WRITE_LOCK_BACKEND=file（仅限单机开发）"
        )
    if cfg == "mysql":
        if engine is None or engine.dialect.name != "mysql":
            raise KbWriteLockBackendError(
                "KB_WRITE_LOCK_BACKEND=mysql 但 DB 未配置或非 MySQL 方言（sqlite 不启用 MySQL 锁）；"
                "请改用 redis 或显式 file"
            )
        return "mysql"
    if cfg == "redis":
        if redis is None:
            raise KbWriteLockBackendError(
                "KB_WRITE_LOCK_BACKEND=redis 但 Redis 未配置；请改用 mysql 或显式 file"
            )
        return "redis"
    if cfg == "file":
        return "file"
    raise KbWriteLockBackendError(f"未知的 KB_WRITE_LOCK_BACKEND: {settings.kb_write_lock_backend}")


class _FileLockAdapter:
    """文件锁适配器：与 Mysql/Redis 锁接口对齐（acquire/assert_held/release）。

    文件锁无「续租丢失」语义：assert_held 只校验本进程持有状态（单机开发）。
    """

    def __init__(self, inner):
        self._inner = inner

    def acquire(self, phase: str = "run") -> None:
        self._inner.acquire(phase)

    def assert_held(self) -> None:
        if not self._inner.held:
            raise KbWriteLockError("文件锁未被持有")

    def release(self) -> None:
        self._inner.release()


def get_kb_write_lock(engine=None, redis=None, ttl_seconds: Optional[int] = None):
    """统一写锁工厂：mysql | redis | file（同实例内解析一次，不随故障切换）。"""
    backend = resolve_lock_backend(engine, redis)
    if backend == "mysql":
        return MysqlAdvisoryLock(engine)
    if backend == "redis":
        return RenewableRedisLock(
            redis, key="kb_write", ttl_seconds=ttl_seconds,
        )
    if backend == "file":
        from app.evolution.lock import LockGuard

        return _FileLockAdapter(LockGuard(
            Path(settings.evolve_state_dir, "evolution.lock"),
            stale_seconds=settings.evolve_lock_stale_seconds,
        ))
    raise KbWriteLockBackendError(f"未实现的后端: {backend}")
