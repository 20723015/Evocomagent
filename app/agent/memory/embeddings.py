"""记忆嵌入存储（记忆系统重构·阶段2.2）：派生数据，与 memory_facts 正本分离。

- **SQL**：``memory_fact_embeddings`` 表（user_id, fact_id, model,
  content_hash, vector JSON, updated_at）——不碰 memory_facts 正本；
  迁移见 deploy/sql/012_memory_fact_embeddings.sql；
- **文件开发模式**：``memory_dir/embeddings/{user_id}.json``（独立文件，
  不进 MemoryFact.to_dict，避免污染审计链与版本快照）；
- Redis 开发模式：独立 hash ``memory:embeddings:{user_id}``（同样不进
  LTM payload 正本）。

派生语义：向量是 content 的函数，丢失/损坏可按 model + content_hash
重建（``backfill_embeddings``），正本永远是 memory_facts。
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path

from sqlalchemy import delete, select

from app.observability.logging import get_logger

log = get_logger("app.agent.memory.embeddings")


class SqlMemoryEmbeddingStore:
    """SQL 实现：``memory_fact_embeddings`` 表（MySQL/SQLite，与正本同库）。"""

    def __init__(self, engine):
        self._engine = engine

    def get(self, user_id: str, fact_id: str, model: str) -> list[float] | None:
        from app.stores.sql.schema import memory_fact_embeddings

        try:
            with self._engine.connect() as conn:
                raw = conn.execute(
                    select(memory_fact_embeddings.c.vector).where(
                        memory_fact_embeddings.c.user_id == user_id,
                        memory_fact_embeddings.c.fact_id == fact_id,
                        memory_fact_embeddings.c.model == model,
                    )
                ).scalar_one_or_none()
        except Exception:  # noqa: BLE001 —— 嵌入读取失败降级词面
            return None
        return _parse_vector(raw)

    def put(self, user_id: str, fact_id: str, model: str,
            content_hash: str, vector: list[float]) -> None:
        from app.stores.sql.schema import memory_fact_embeddings

        payload = json.dumps(vector, ensure_ascii=False)
        with self._engine.begin() as conn:
            dialect = conn.dialect.name
            if dialect == "sqlite":
                conn.execute(
                    memory_fact_embeddings.delete().where(
                        memory_fact_embeddings.c.user_id == user_id,
                        memory_fact_embeddings.c.fact_id == fact_id,
                        memory_fact_embeddings.c.model == model,
                    )
                )
                conn.execute(memory_fact_embeddings.insert().values(
                    user_id=user_id, fact_id=fact_id, model=model,
                    content_hash=content_hash, vector=payload,
                    updated_at=datetime.now(),
                ))
            else:
                from sqlalchemy.dialects.mysql import insert as mysql_insert

                stmt = mysql_insert(memory_fact_embeddings).values(
                    user_id=user_id, fact_id=fact_id, model=model,
                    content_hash=content_hash, vector=payload,
                    updated_at=datetime.now(),
                )
                conn.execute(stmt.on_duplicate_key_update(
                    content_hash=stmt.inserted.content_hash,
                    vector=stmt.inserted.vector,
                    updated_at=stmt.inserted.updated_at,
                ))

    def delete_user(self, user_id: str) -> None:
        from app.stores.sql.schema import memory_fact_embeddings

        with self._engine.begin() as conn:
            conn.execute(
                delete(memory_fact_embeddings).where(
                    memory_fact_embeddings.c.user_id == user_id,
                )
            )


class FileMemoryEmbeddingStore:
    """文件实现：``memory_dir/embeddings/{user_id}.json``（独立 sidecar）。"""

    def __init__(self, memory_dir: str):
        self._dir = Path(memory_dir) / "embeddings"
        self._lock = threading.Lock()

    def _path(self, user_id: str) -> Path:
        safe = "".join(
            ch if ch.isalnum() or ch in "-_." else "_" for ch in user_id
        )
        return self._dir / f"{safe}.json"

    def _load(self, user_id: str) -> dict:
        path = self._path(user_id)
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self, user_id: str, data: dict) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path(user_id)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)

    def get(self, user_id: str, fact_id: str, model: str) -> list[float] | None:
        with self._lock:
            entry = self._load(user_id).get(fact_id)
        if not isinstance(entry, dict) or entry.get("model") != model:
            return None
        return _parse_vector(entry.get("vector"))

    def put(self, user_id: str, fact_id: str, model: str,
            content_hash: str, vector: list[float]) -> None:
        with self._lock:
            data = self._load(user_id)
            data[fact_id] = {
                "model": model,
                "content_hash": content_hash,
                "vector": vector,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            self._save(user_id, data)

    def delete_user(self, user_id: str) -> None:
        with self._lock:
            path = self._path(user_id)
            if path.exists():
                path.unlink()


class RedisMemoryEmbeddingStore:
    """Redis 实现：独立 hash（不进 LTM payload 正本，键独立可整体重建）。"""

    def __init__(self, redis):
        self._redis = redis

    @staticmethod
    def _key(user_id: str) -> str:
        return f"memory:embeddings:{user_id}"

    def get(self, user_id: str, fact_id: str, model: str) -> list[float] | None:
        try:
            raw = self._redis.hget(self._key(user_id), f"{model}:{fact_id}")
        except Exception:  # noqa: BLE001
            return None
        return _parse_vector(raw)

    def put(self, user_id: str, fact_id: str, model: str,
            content_hash: str, vector: list[float]) -> None:
        try:
            self._redis.hset(
                self._key(user_id), f"{model}:{fact_id}",
                json.dumps(vector, ensure_ascii=False),
            )
        except Exception as e:  # noqa: BLE001 —— 嵌入缓存写失败不阻塞
            log.info("memory.embedding_redis_put_failed err=%s", type(e).__name__)

    def delete_user(self, user_id: str) -> None:
        try:
            self._redis.delete(self._key(user_id))
        except Exception:  # noqa: BLE001
            pass


def _parse_vector(raw) -> list[float] | None:
    """DB/文件/Redis 原始值 → list[float]；解析失败 None（降级词面）。"""
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, list) or not raw:
        return None
    try:
        return [float(x) for x in raw]
    except (TypeError, ValueError):
        return None


def build_memory_embedding_store(engine=None, redis=None,
                                 memory_dir: str = ""):
    """按配置选择嵌入存储：SQL 正本优先 → Redis → 文件（与 LTM 存储同策略）。"""
    if engine is not None:
        return SqlMemoryEmbeddingStore(engine)
    if redis is not None:
        return RedisMemoryEmbeddingStore(redis)
    from app.config.settings import settings

    return FileMemoryEmbeddingStore(memory_dir or settings.memory_dir)


def backfill_embeddings(user_id: str, facts, embedder, store,
                        *, stage: str = "backfill") -> int:
    """对 active 事实补算嵌入（一次性，量小：active ≤ max_facts）。

    - content_hash 命中（内容未变且模型未换）→ 跳过，避免重复计费；
    - 单项失败跳过（日志已有计数），不阻塞其他事实；
    - 返回本次新写入的向量数。
    """
    if embedder is None or store is None:
        return 0
    pending = []
    for fact in facts:
        if not getattr(fact, "active", False):
            continue
        cached = store.get(user_id, fact.fact_id, embedder.model)
        if cached is not None:
            continue
        pending.append(fact)
    if not pending:
        return 0
    vectors = embedder.encode_batch(
        [f.content for f in pending], stage=stage,
    )
    written = 0
    for fact, vector in zip(pending, vectors):
        if vector is None:
            continue
        try:
            store.put(
                user_id, fact.fact_id, embedder.model,
                embedder.content_hash(embedder.model, fact.content), vector,
            )
            written += 1
        except Exception as e:  # noqa: BLE001 —— 后补数据，失败不阻塞
            log.info(
                "memory.embedding_backfill_put_failed fact=%s err=%s",
                fact.fact_id, type(e).__name__,
            )
    return written
