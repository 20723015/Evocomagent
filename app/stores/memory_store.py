"""长期记忆外置（阶段二 2.3）：memory:{user_id} → Redis（hash）。

LocalFileLTMStore：现有 memory_dir/{user_id}.json 布局（开发/测试）。
RedisLTMStore：hash 字段 facts / interaction_summaries / updated_at。
CachedLTMStore：cache-aside 包装（SQL 唯一正本 + Redis 读缓存）。
LongTermMemory 只依赖 LTMStore 协议。

安全修复 P2：LTMStore 增加 merge(user_id, merger) 原子读-改-写——
同用户多会话并发巩固时，历史 load/extend/save 整包覆写 last-write-wins
互相覆盖（丢事实）。文件版 per-user 进程内锁；Redis 版 WATCH/MULTI 重试，
重试超限或存储异常时 fail-closed（fakeredis 对 WATCH/MULTI 支持有限，
竞态测试主要跑文件/SQL 路径）。
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from redis.exceptions import WatchError

from app.security.identifiers import validate_identifier
from app.stores.base import LTMStore, StorageUnavailableError

_log = logging.getLogger("app.stores.memory_store")


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


class CachedLTMStore:
    """cache-aside 读缓存：SQL 唯一正本 + Redis 加速读（与会话热缓存同模式）。

    语义（与 SqlSessionStore 热缓存一致）：
    - **SQL 是唯一正本**；Redis 里的 ``memory:{user_id}`` 只是可丢弃的副本，
      flush/eviction/丢失都只影响性能，不影响正确性（下次 load 回填）；
    - load：命中缓存直接返回；未命中/缓存异常 → 透传 inner（SQL）并回填；
    - save/merge：**先写 inner**，成功后才改写缓存——缓存写失败只告警，
      绝不回滚/报错（正本已落库，缓存由下次读回填）；
    - StorageUnavailableError 一律不兜：正本不可用时 fail-closed（503），
      不拿缓存冒充正本（缓存可能是被 evict 后重新加载的旧值）。

    为什么不把 Redis 当 LTM 唯一主存储：eviction 静默丢记忆（用户画像
    无声消失）与跨存储无事务（Redis 写了 SQL 没写无从对账）两个坑；
    cache-aside 把 Redis 降级为纯加速层即可同时避开。

    merge 的读-改-写始终在 inner 上原子完成（SQL GET_LOCK/事务），缓存
    不参与并发控制——因此这里不做 WATCH/MULTI（缓存只是写后刷新）。

    键名沿用 ``memory:{user_id}``（RedisLTMStore 布局）：若某环境此前以
    Redis 作 LTM 唯一主存储且刚切到 SQL，旧键会被当缓存读到——TTL（默认
    1800s）内可能读到旧副本，到期或下一次写自动纠正；生产一直配 DB_URL
    走 SQL，不受影响。
    """

    def __init__(self, inner, redis, ttl_seconds: int | None = None):
        self._inner = inner
        self._redis = redis
        # 编解码/键名复用 RedisLTMStore（hash 字段与 load/save 结构一致）
        self._codec = RedisLTMStore(redis)
        if ttl_seconds is None:
            from app.config.settings import settings

            ttl_seconds = settings.session_hot_cache_ttl_seconds
        self._ttl = int(ttl_seconds)

    # ---------- 缓存原语（全部 fail-open：缓存故障绝不冒泡） ----------
    def _cache_load(self, user_id: str) -> Optional[dict]:
        try:
            raw = self._redis.hgetall(self._codec.key(user_id))
        except Exception as e:  # noqa: BLE001 —— 缓存读失败 = 未命中，回源正本
            _log.warning("LTM 缓存读取失败（回源正本）: %s", type(e).__name__)
            return None
        if not raw:
            return None
        if not self._cache_payload_valid(raw):
            # 损坏副本绝不冒充正本（RedisLTMStore 解码会把坏字段降级成空表，
            # 直接返回等于静默清空用户记忆）→ 视为未命中并回源修复
            _log.warning("LTM 缓存损坏（回源正本）: user=%s", user_id)
            return None
        try:
            return RedisLTMStore._decode_hash(raw)
        except Exception as e:  # noqa: BLE001
            _log.warning("LTM 缓存解码失败（回源正本）: %s", type(e).__name__)
            return None

    @staticmethod
    def _cache_payload_valid(raw: dict) -> bool:
        """缓存 hash 结构校验：列表字段必须是合法 JSON 数组。"""
        for key, value in raw.items():
            name = key.decode("utf-8") if isinstance(key, bytes) else key
            if name not in ("facts", "interaction_summaries"):
                continue
            text = value.decode("utf-8") if isinstance(value, bytes) else value
            try:
                if not isinstance(json.loads(text), list):
                    return False
            except (json.JSONDecodeError, TypeError):
                return False
        return True

    def _cache_store(self, user_id: str, payload: dict) -> None:
        try:
            self._codec.save(user_id, payload)
            if self._ttl > 0:
                self._redis.expire(self._codec.key(user_id), self._ttl)
        except Exception as e:  # noqa: BLE001 —— 正本已落库，缓存写失败只降速
            _log.warning("LTM 缓存写入失败（正本已落库）: %s", type(e).__name__)

    # ---------- 协议 ----------
    def load(self, user_id: str) -> Optional[dict]:
        cached = self._cache_load(user_id)
        if cached is not None:
            return cached
        payload = self._inner.load(user_id)  # StorageUnavailableError 不兜
        if payload is not None:
            self._cache_store(user_id, payload)
        return payload

    def save(self, user_id: str, payload: dict) -> None:
        self._inner.save(user_id, payload)
        self._cache_store(user_id, payload)

    def merge(self, user_id: str, merger: Callable[[Optional[dict]], dict]) -> dict:
        updated = self._inner.merge(user_id, merger)
        self._cache_store(user_id, updated)
        return updated
