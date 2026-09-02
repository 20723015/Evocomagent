"""长期记忆外置（阶段二 2.3）：memory:{user_id} → Redis（hash）。

LocalFileLTMStore：现有 memory_dir/{user_id}.json 布局（开发/测试）。
RedisLTMStore：hash 字段 facts / interaction_summaries / updated_at。
LongTermMemory 只依赖 LTMStore 协议。

安全修复 P2：LTMStore 增加 merge(user_id, merger) 原子读-改-写——
同用户多会话并发巩固时，历史 load/extend/save 整包覆写 last-write-wins
互相覆盖（丢事实）。文件版 per-user 进程内锁；Redis 版 WATCH/MULTI 重试，
重试超限或存储异常时 fail-closed（fakeredis 对 WATCH/MULTI 支持有限，
竞态测试主要跑文件/SQL 路径）。
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from redis.exceptions import WatchError

from app.security.identifiers import validate_identifier
from app.stores.base import LTMStore, StorageUnavailableError


class LocalFileLTMStore:
    """文件实现：{memory_dir}/{user_id}.json（与既有 LongTermMemory 布局一致）。"""

    def __init__(self, memory_dir: str | Path):
        self._dir = Path(memory_dir)
        self._merge_locks: dict[str, threading.Lock] = {}
        self._merge_locks_guard = threading.Lock()

    def _path(self, user_id: str) -> Path:
        # 安全修复 P2：标识符白名单（纵深防御；`../` 不再逃逸目录）
        validate_identifier(user_id, "user_id")
        return self._dir / f"{user_id}.json"

    def load(self, user_id: str) -> Optional[dict]:
        path = self._path(user_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def save(self, user_id: str, payload: dict) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path(user_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _merge_lock(self, user_id: str) -> threading.Lock:
        with self._merge_locks_guard:
            if user_id not in self._merge_locks:
                self._merge_locks[user_id] = threading.Lock()
            return self._merge_locks[user_id]

    def merge(self, user_id: str, merger: Callable[[Optional[dict]], dict]) -> dict:
        """原子读-改-写（安全修复 P2）：per-user 锁内 load → merger → save。

        进程内互斥保证同用户多会话并发巩固不互相覆盖；跨进程共享目录不在
        文件版的职责范围（生产走 Redis/SQL 版）。
        """
        with self._merge_lock(user_id):
            current = self.load(user_id)
            updated = merger(current)
            self.save(user_id, updated)
            return updated


class RedisLTMStore:
    """Redis 实现：key memory:{user_id}，hash 存 facts / interaction_summaries / updated_at。"""

    KEY_PREFIX = "memory:"

    def __init__(self, redis):
        self._redis = redis

    @staticmethod
    def key(user_id: str) -> str:
        validate_identifier(user_id, "user_id")
        return f"{RedisLTMStore.KEY_PREFIX}{user_id}"

    def load(self, user_id: str) -> Optional[dict]:
        raw_data = self._redis.hgetall(self.key(user_id))
        if not raw_data:
            return None
        # redis-py 返回 bytes 键值，统一解码
        data = {
            k.decode("utf-8") if isinstance(k, bytes) else k:
            v.decode("utf-8") if isinstance(v, bytes) else v
            for k, v in raw_data.items()
        }
        payload: dict = {}
        for field in ("facts", "interaction_summaries"):
            raw = data.get(field)
            if raw is None:
                continue
            try:
                payload[field] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                payload[field] = []
        payload["updated_at"] = data.get("updated_at", "")
        payload["schema_version"] = self._int_field(data, "schema_version", 1)
        payload["version"] = self._int_field(data, "version", 1)
        return payload

    def save(self, user_id: str, payload: dict) -> None:
        mapping = {
            "facts": json.dumps(payload.get("facts", []), ensure_ascii=False),
            "interaction_summaries": json.dumps(
                payload.get("interaction_summaries", []), ensure_ascii=False
            ),
            "updated_at": payload.get("updated_at")
            or datetime.now().isoformat(timespec="seconds"),
            "schema_version": str(payload.get("schema_version", 2)),
            "version": str(payload.get("version", 2)),
        }
        self._redis.hset(self.key(user_id), mapping=mapping)

    def merge(self, user_id: str, merger: Callable[[Optional[dict]], dict]) -> dict:
        """原子读-改-写（安全修复 P2）：WATCH/MULTI 事务内校验重写。

        并发修改 → WatchError → 重读重试；重试超限或 Redis 异常时
        fail-closed，避免 last-write-wins 静默覆盖其他实例的记忆变更。
        注意：fakeredis 对 WATCH/MULTI 支持有限，竞态正确性只在文件/SQL
        路径验证（评审·坑4）；此处用 fakeredis 验证合并逻辑本身。
        """
        key = self.key(user_id)
        redis = self._redis
        for _ in range(8):
            try:
                with redis.pipeline() as pipe:
                    pipe.watch(key)
                    raw = pipe.hgetall(key)
                    current = self._decode_hash(raw)
                    updated = merger(current)
                    pipe.multi()
                    pipe.hset(key, mapping={
                        "facts": json.dumps(updated.get("facts", []), ensure_ascii=False),
                        "interaction_summaries": json.dumps(
                            updated.get("interaction_summaries", []),
                            ensure_ascii=False,
                        ),
                        "updated_at": updated.get("updated_at")
                        or datetime.now().isoformat(timespec="seconds"),
                        "schema_version": str(updated.get("schema_version", 2)),
                        "version": str(updated.get("version", 2)),
                    })
                    pipe.execute()
                    return updated
            except WatchError:
                continue
            except Exception as e:  # noqa: BLE001
                raise StorageUnavailableError(f"Redis LTM merge 失败: {e}") from e
        raise StorageUnavailableError(f"Redis LTM merge 冲突重试超限: {user_id}")

    @staticmethod
    def _decode_hash(raw: dict) -> dict:
        if not raw:
            return {}
        data = {
            (k.decode("utf-8") if isinstance(k, bytes) else k):
            (v.decode("utf-8") if isinstance(v, bytes) else v)
            for k, v in raw.items()
        }
        payload: dict = {}
        for field in ("facts", "interaction_summaries"):
            value = data.get(field)
            if value is None:
                continue
            try:
                payload[field] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                payload[field] = []
        payload["updated_at"] = data.get("updated_at", "")
        payload["schema_version"] = RedisLTMStore._int_field(
            data, "schema_version", 1,
        )
        payload["version"] = RedisLTMStore._int_field(data, "version", 1)
        return payload

    @staticmethod
    def _int_field(data: dict, field: str, default: int) -> int:
        try:
            return int(data.get(field, default) or default)
        except (TypeError, ValueError):
            return default
