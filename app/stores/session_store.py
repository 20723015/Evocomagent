"""SessionStore 两个实现（阶段二 2.1/2.2）。

- LocalFileSessionStore：现有 storage.py 逻辑的协议化包装（开发/测试用）。
  文件 CAS 是「读-改-写」最佳努力——真正的并发一致性靠 Redis 版。
- RedisSessionStore：生产用。WATCH/MULTI 事务做 CAS，多 pod 一致。
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from redis.exceptions import WatchError

from app.security.identifiers import validate_identifier
from app.stores.base import (
    SessionConflictError,
    SessionState,
    StorageUnavailableError,
)

SESSION_VERSION = 2
_LOCK_VERSION_KEY = "lock_version"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _parse_payload(data: dict, session_id: str) -> SessionState:
    messages = data.get("messages", [])
    # 巩固水位：新 payload 显式携带；旧格式（无字段，安全修复前）视为已
    # 全量巩固——升级后首轮 close 不再对整个历史重复摘要
    raw_len = data.get("consolidated_len")
    consolidated_len = (
        int(raw_len) if raw_len is not None else len(messages)
    )
    return SessionState(
        session_id=data.get("session_id") or session_id,
        user_id=str(data.get("user_id", "") or ""),
        summary=data.get("summary"),
        messages=messages,
        short_term_memory=data.get("short_term_memory"),
        version=int(data.get(_LOCK_VERSION_KEY, 0) or 0),
        consolidated_len=consolidated_len,
        updated_at=data.get("updated_at", ""),
    )


def _payload_of(state: SessionState) -> dict:
    return {
        "version": SESSION_VERSION,
        "session_id": state.session_id,
        "user_id": state.user_id,
        "updated_at": state.updated_at or _now(),
        "summary": state.summary,
        "messages": state.messages,
        "short_term_memory": state.short_term_memory,
        "consolidated_len": state.consolidated_len,
        _LOCK_VERSION_KEY: state.version,
    }


class LocalFileSessionStore:
    """本地文件实现：root 下 {user_id}/{session_id}.json；exact_path 指定单文件（沙箱用）。"""

    def __init__(self, root: str | Path, exact_path: str | Path | None = None):
        self._root = Path(root)
        self._exact = Path(exact_path) if exact_path is not None else None

    def path_for(self, user_id: str, session_id: str) -> Path:
        if self._exact is not None:
            return self._exact
        # 安全修复 P2：标识符白名单（纵深防御第二层；服务端入口已校验）。
        # `../` 类标识直接拒绝，绝不落盘到目录外。
        validate_identifier(user_id, "user_id")
        validate_identifier(session_id, "session_id", allow_empty=True)
        return self._root / user_id / f"{session_id or 'session'}.json"

    def load(self, user_id: str, session_id: str) -> Optional[SessionState]:
        path = self.path_for(user_id, session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, dict) or "messages" not in data:
            return None
        return _parse_payload(data, session_id)

    def _acquire_write_lock(self, path: Path, timeout: float = 2.0):
        """文件级写锁（O_EXCL）：串行化读-判-写，让本地 CAS 具备真单写者语义。

        锁文件随进程 crash 残留时由时间窗兜底（超时 → 冲突语义）。
        """
        lock_path = path.with_suffix(path.suffix + ".writelock")
        deadline = time.monotonic() + timeout
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return lock_path
            except FileExistsError:
                if time.monotonic() > deadline:
                    raise SessionConflictError(
                        f"session 写入锁获取超时（{lock_path.name}）"
                    )
                time.sleep(0.02)

    def save(self, user_id: str, session_id: str, state: SessionState,
             new_messages: Optional[list[dict]] = None,
             enqueue_memory_job: bool = False) -> SessionState:
        # 整包覆写实现：new_messages 忽略（阶段八 SQL store 用行式追加）；
        # enqueue_memory_job 仅 SQL 实现消费（同事务入队 memory job），此处忽略
        path = self.path_for(user_id, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self._acquire_write_lock(path)
        try:
            # 单写者下读-判-写原子；CAS：期望版本与磁盘不一致 → 409 语义
            current = self.load(user_id, session_id)
            if current is not None and current.version != state.version:
                raise SessionConflictError(
                    f"session {user_id}/{session_id} 版本冲突："
                    f"磁盘 v{current.version} ≠ 期望 v{state.version}（可能被并发修改）"
                )

            updated = SessionState(**{**state.__dict__, "version": state.version + 1})
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(_payload_of(updated), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, path)
            return updated
        finally:
            if lock_path.exists():
                lock_path.unlink()

    def delete(self, user_id: str, session_id: str) -> None:
        path = self.path_for(user_id, session_id)
        if path.exists():
            path.unlink()


class RedisSessionStore:
    """Redis 实现：session:{user_id}:{session_id} = JSON（含 lock_version）。

    CAS 用 WATCH/MULTI：读版本 → 事务内校验并写回；并发修改时 WATCH 失败重试。
    """

    KEY_PREFIX = "session:"

    def __init__(self, redis):
        self._redis = redis

    @staticmethod
    def key(user_id: str, session_id: str) -> str:
        # 白名单校验防 Redis key 注入（`:` 是分隔符，白名单本就禁止）
        validate_identifier(user_id, "user_id")
        validate_identifier(session_id, "session_id", allow_empty=True)
        return f"{RedisSessionStore.KEY_PREFIX}{user_id}:{session_id}"

    def load(self, user_id: str, session_id: str) -> Optional[SessionState]:
        try:
            raw = self._redis.get(self.key(user_id, session_id))
        except Exception as e:  # noqa: BLE001 —— 5.2：断连即不可用（503 语义）
            raise StorageUnavailableError(f"Redis 读取失败: {e}") from e
        if raw is None:
            return None
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        return _parse_payload(data, session_id)

    def save(self, user_id: str, session_id: str, state: SessionState,
             new_messages: Optional[list[dict]] = None,
             enqueue_memory_job: bool = False) -> SessionState:
        # 整包覆写实现：new_messages/enqueue_memory_job 忽略（SQL store 专用）
        key = self.key(user_id, session_id)
        redis = self._redis
        payload = _payload_of(state)

        for attempt in range(8):
            try:
                with redis.pipeline() as pipe:
                    pipe.watch(key)
                    current_raw = pipe.get(key)
                    current_version = 0
                    if current_raw:
                        try:
                            current_version = int(
                                json.loads(current_raw).get(_LOCK_VERSION_KEY, 0) or 0
                            )
                        except (json.JSONDecodeError, TypeError, ValueError):
                            current_version = 0  # 损坏视作新会话
                    if current_version != state.version:
                        pipe.unwatch()
                        raise SessionConflictError(
                            f"session {user_id}/{session_id} 版本冲突："
                            f"Redis v{current_version} ≠ 期望 v{state.version}"
                        )
                    pipe.multi()
                    pipe.set(key, json.dumps(
                        {**payload, _LOCK_VERSION_KEY: current_version + 1},
                        ensure_ascii=False,
                    ))
                    pipe.execute()
                    loaded = self.load(user_id, session_id)
                    if loaded is None:  # 语义兜底：读回失败视为冲突，让客户端重读
                        raise SessionConflictError(
                            f"session {user_id}/{session_id} 写回后读回为空"
                        )
                    return loaded
            except SessionConflictError:
                raise
            except WatchError:
                continue  # 事务期间被并发修改 → 重读重试
            except Exception as e:  # noqa: BLE001 —— 5.2：断连即不可用（503 语义）
                raise StorageUnavailableError(f"Redis 写入失败: {e}") from e
        raise SessionConflictError(
            f"session {user_id}/{session_id} 写回重试超限（并发写繁忙）"
        )

    def delete(self, user_id: str, session_id: str) -> None:
        try:
            self._redis.delete(self.key(user_id, session_id))
        except Exception as e:  # noqa: BLE001
            raise StorageUnavailableError(f"Redis 删除失败: {e}") from e

    def iter_all(self) -> list[tuple[str, str]]:
        """枚举全部会话 (user_id, session_id)：idle 兜底巩固 job 用（SCAN）。"""
        out: list[tuple[str, str]] = []
        for key_raw in self._redis.scan_iter(match=f"{self.KEY_PREFIX}*", count=500):
            key = key_raw.decode("utf-8") if isinstance(key_raw, bytes) else key_raw
            rest = key[len(self.KEY_PREFIX):] if key.startswith(self.KEY_PREFIX) else key
            if ":" not in rest:
                continue
            user_id, session_id = rest.split(":", 1)
            out.append((user_id, session_id))
        return out
