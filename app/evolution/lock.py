"""lock.py：单写者锁（PID/hostname/stale 判定）+ 事务 journal（第10期）。

锁语义：
- O_CREAT|O_EXCL 写 {pid, hostname, started_at, phase}。
- 二次获取：同主机且 PID 存活 → LockHeldError；hostname 不同 → CrossHostLockError
  （只能 --force-unlock 处理）。
- 同主机 PID 已死且超 stale 秒数 → 归档旧锁并接管；未超时 → LockHeldError。

journal：staging 事务日志。文档移动前写入 staging 文件清单与索引目标，
下次运行开始时恢复（generation 未切换 → 清理孤立文档与 staging 索引；
已切换 → 由 ledger 补记）。
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from app.evolution.generation import GenerationInfo


class LockHeldError(Exception):
    """锁被同主机存活进程持有（或残留未超 stale）。"""


class CrossHostLockError(Exception):
    """锁被其他主机持有，只能 --force-unlock 处理。"""


def _default_pid_alive(pid: int) -> bool:
    """检查 PID 是否存活：Windows 用 OpenProcess，POSIX 用 os.kill(pid, 0)。"""
    if os.name == "nt":
        try:
            import ctypes

            # PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        except Exception:  # noqa: BLE001 —— 判定失败按不存在处理，避免死锁
            return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _parse_started(text) -> Optional[float]:
    try:
        return datetime.fromisoformat(text).timestamp()
    except (ValueError, TypeError, AttributeError):
        return None


class LockGuard:
    """单写者锁：构造后可注入 clock / hostname / pid_alive 便于测试。"""

    def __init__(
        self,
        path,
        stale_seconds: int,
        clock=None,
        hostname: Optional[str] = None,
        pid_alive_fn: Optional[Callable[[int], bool]] = None,
    ):
        self._path = Path(path)
        self._stale_seconds = stale_seconds
        self._clock = clock
        self._hostname = hostname or socket.gethostname()
        self._pid_alive = pid_alive_fn or _default_pid_alive
        self._acquired = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def held(self) -> bool:
        """本进程是否持有（kb_write 锁适配器 assert_held 用）。"""
        return self._acquired

    def assert_held(self) -> None:
        """2.7：副作用（文件移动/alias/pointer）前后校验锁仍被本进程持有。

        文件锁无「续租丢失」语义，此处校验本进程持有状态（与 kb_write_lock
        的 assert_held 同接口，供上传/演进共用同一发布阶段代码路径）。
        """
        if not self._acquired:
            raise LockHeldError("锁未被本进程持有")

    def _now(self) -> float:
        if self._clock:
            return self._clock.now().timestamp()
        return time.time()

    def acquire(self, phase: str = "run") -> None:
        """获取锁；被持有/残留 → 抛 LockHeldError 或 CrossHostLockError。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "pid": os.getpid(),
            "hostname": self._hostname,
            "started_at": datetime.fromtimestamp(self._now()).isoformat(timespec="seconds"),
            "phase": phase,
        }
        if self._try_create(entry):
            self._acquired = True
            return

        holder = self._read_holder()
        holder_pid = holder.get("pid")
        holder_host = holder.get("hostname", "")

        if holder_host == self._hostname and self._pid_alive(holder_pid):
            raise LockHeldError(
                f"另一个进程（PID {holder_pid}）正在运行 "
                f"{holder.get('phase', '?')}，请稍后重试"
            )
        if holder_host != self._hostname:
            raise CrossHostLockError(
                f"锁被主机 {holder_host}（PID {holder_pid}）持有，"
                f"只能 --force-unlock 处理"
            )

        # 同主机但进程已死：超 stale 才接管
        started = _parse_started(holder.get("started_at"))
        if started is None or self._now() - started < self._stale_seconds:
            raise LockHeldError("锁由已退出进程残留但未超 stale 时限，请稍后重试")

        archive = self._path.with_suffix(self._path.suffix + ".stale")
        try:
            os.replace(self._path, archive)
        except OSError:
            raise LockHeldError(f"无法归档残留锁文件: {self._path}")

        if not self._try_create(entry):
            raise LockHeldError("锁文件在接管重试期间被其他进程抢占")
        self._acquired = True

    def _try_create(self, entry: dict) -> bool:
        try:
            fd = os.open(str(self._path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False)
            return True
        except FileExistsError:
            return False

    def _read_holder(self) -> dict:
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def release(self) -> None:
        if self._acquired:
            self._path.unlink(missing_ok=True)
            self._acquired = False

    def force_unlock(self) -> None:
        """--force-unlock：无条件删除锁文件（跨主机场景的唯一解法）。"""
        self._acquired = False
        self._path.unlink(missing_ok=True)


# ============================================================
# staging 事务 journal
# ============================================================
class Journal:
    """staging 事务日志：文档移动前写入，下次运行开始时恢复。"""

    def __init__(self, path):
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def write(self, entry: dict) -> None:
        """原子写入 journal（staging 文件清单 + staging 索引目标）。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp, self._path)

    def read(self) -> Optional[dict]:
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def clear(self) -> None:
        self._path.unlink(missing_ok=True)

    def archive(self) -> None:
        """归档（保留现场，便于排查中断原因）。"""
        archived = self._path.with_suffix(self._path.suffix + ".archived")
        try:
            os.replace(self._path, archived)
        except OSError:
            self.clear()


def journal_entry(
    staging_docs: list[str],
    backend: str,
    index_info: Optional[GenerationInfo],
) -> dict:
    """构造 journal 内容：staging 文件清单（相对 kb_dir）+ staging 索引目标。"""
    return {
        "phase": "publish",
        "staging_docs": staging_docs,
        "backend": backend,
        "index": index_info.to_dict() if index_info else None,
    }

class RedisLockGuard:
    """evolution 单写者锁的 Redis 实现（阶段六 6.2，K8s CronJob 跨 pod 互斥）。

    与 LockGuard 接口一致（acquire(phase)/release()），语义：
    - SET NX EX：抢到即持有（ttl 大于运行时长，进程崩溃自动过期）；
    - 二次获取 → LockHeldError（另一 pod 在跑，本轮跳过）；
    - Redis 不可用 → CrossHostLockError（宁可失败不双写）；
    - release 仅当 token 匹配才删除（防误释放他人锁）。

    注意：生产环境的 evolution/上传写锁统一走 kb_write_lock.RenewableRedisLock
    （stale 续期 + lost 检测）；本类仅保留给测试与文件锁后端的历史兼容。
    """

    KEY_PREFIX = "evolution_lock:"

    def __init__(self, redis, key: str, ttl_seconds: int):
        self._redis = redis
        self._key = f"{self.KEY_PREFIX}{key}"
        self._ttl = ttl_seconds
        self._token = uuid.uuid4().hex
        self._held = False

    def acquire(self, phase: str = "run") -> None:
        try:
            ok = self._redis.set(self._key, self._token, nx=True, ex=self._ttl)
        except Exception as e:  # noqa: BLE001 —— Redis 不可用：明确失败而非双写
            raise CrossHostLockError(f"Redis 锁不可用（{e}），拒绝并写") from e
        if not ok:
            raise LockHeldError(
                f"evolution 锁被其他运行者持有（{self._key}），本机跳过本次运行"
            )
        self._held = True

    def release(self) -> None:
        if not self._held:
            return
        try:
            self._redis.eval(
                "if redis.call('get', KEYS[1]) == ARGV[1] then "
                "return redis.call('del', KEYS[1]) else return 0 end",
                1, self._key, self._token,
            )
        except Exception:  # noqa: BLE001 —— EVAL 不可用降级 GET+DEL
            try:
                cur = self._redis.get(self._key)
                cur_str = cur.decode("utf-8") if isinstance(cur, bytes) else cur
                if cur_str == self._token:
                    self._redis.delete(self._key)
            except Exception:  # noqa: BLE001 —— 释放失败靠 TTL 兜底
                pass
        self._held = False
