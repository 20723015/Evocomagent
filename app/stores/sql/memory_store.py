"""SqlLTMStore：长期记忆行式化（phase 八）。

memory_facts / interaction_summaries 两张表；写策略为「按用户事务内
全量替换」。活跃事实最多 50 条，失效版本继续保留审计；替换保证与
LongTermMemory 的内存态严格一致并保持幂等。
"""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime
from typing import Optional

from sqlalchemy import delete, select, text

from app.stores.base import StorageUnavailableError
from app.stores.sql.schema import interaction_summaries, memory_facts


_merge_locks: dict[str, threading.Lock] = {}
_merge_locks_guard = threading.Lock()


class SqlLTMStore:
    """SQLAlchemy 实现（生产 mysql+pymysql；本地 sqlite）。"""

    def __init__(self, engine):
        self._engine = engine

    def _merge_lock(self, user_id: str) -> threading.Lock:
        key = f"{self._engine.url}:{user_id}"
        with _merge_locks_guard:
            if key not in _merge_locks:
                _merge_locks[key] = threading.Lock()
            return _merge_locks[key]

    def load(self, user_id: str) -> Optional[dict]:
        try:
            with self._engine.connect() as conn:
                return self._load_via(conn, user_id)
        except StorageUnavailableError:
            raise
        except Exception as e:  # noqa: BLE001
            raise StorageUnavailableError(f"SQL 记忆读取失败: {e}") from e

    def _load_via(self, conn, user_id: str) -> dict:
        """事务/连接内读取（load 与 merge 复用）。"""
        facts = conn.execute(
            select(memory_facts.c.category, memory_facts.c.content,
                   memory_facts.c.source_session, memory_facts.c.created_at,
                   memory_facts.c.fact_id, memory_facts.c.fact_key,
                   memory_facts.c.status, memory_facts.c.confidence,
                   memory_facts.c.supersedes_id, memory_facts.c.updated_at,
                   memory_facts.c.evidence)
            .where(memory_facts.c.user_id == user_id)
            .order_by(memory_facts.c.id)
        ).all()
        summaries = conn.execute(
            select(
                interaction_summaries.c.summary,
                interaction_summaries.c.source_session,
                interaction_summaries.c.created_at,
            )
            .where(interaction_summaries.c.user_id == user_id)
            .order_by(interaction_summaries.c.id)
        ).all()

        return {
            "schema_version": 3,
            "version": 3,
            "facts": [{
                "content": f.content,
                "category": f.category,
                "created_at": f.created_at.isoformat(sep=" ", timespec="seconds")
                if f.created_at else "",
                "source_session": f.source_session,
                "fact_id": f.fact_id,
                "fact_key": f.fact_key,
                "status": f.status,
                "confidence": f.confidence,
                "supersedes_id": f.supersedes_id,
                "evidence": f.evidence or "",
                "updated_at": (
                    f.updated_at or f.created_at
                ).isoformat(sep=" ", timespec="seconds")
                if (f.updated_at or f.created_at) else "",
            } for f in facts],
            "interaction_summaries": [{
                "summary": s.summary,
                "timestamp": s.created_at.isoformat(sep=" ", timespec="seconds")
                if s.created_at else "",
                "source_session": s.source_session,
            } for s in summaries],
        }

    @staticmethod
    def _parse_ts(value: str) -> datetime:
        """payload 里的原始时间优先（保留「这条记忆多久前形成」的信息）。"""
        if value:
            try:
                return datetime.fromisoformat(value.replace(" ", "T"))
            except (ValueError, TypeError):
                pass
        return datetime.now()

    def save(self, user_id: str, payload: dict) -> None:
        try:
            with self._engine.begin() as conn:
                self._write_via(conn, user_id, payload)
        except Exception as e:  # noqa: BLE001
            raise StorageUnavailableError(f"SQL 记忆写入失败: {e}") from e

    def _write_via(self, conn, user_id: str, payload: dict) -> None:
        """事务内全量替换（save 与 merge 复用）。"""
        conn.execute(delete(memory_facts).where(memory_facts.c.user_id == user_id))
        conn.execute(delete(interaction_summaries)
                     .where(interaction_summaries.c.user_id == user_id))
        for fact in payload.get("facts", []):
            conn.execute(memory_facts.insert().values(
                user_id=user_id,
                category=fact.get("category", "other"),
                content=fact.get("content", ""),
                source_session=fact.get("source_session", ""),
                fact_id=fact.get("fact_id", ""),
                fact_key=fact.get("fact_key", ""),
                status=fact.get("status", "active"),
                confidence=float(fact.get("confidence", 1.0) or 0.0),
                supersedes_id=fact.get("supersedes_id", ""),
                evidence=fact.get("evidence", "") or "",
                created_at=self._parse_ts(fact.get("created_at", "")),
                updated_at=self._parse_ts(fact.get("updated_at", "")),
            ))
        for s in payload.get("interaction_summaries", []):
            conn.execute(interaction_summaries.insert().values(
                user_id=user_id, summary=s.get("summary", ""),
                source_session=s.get("source_session", ""),
                created_at=self._parse_ts(s.get("timestamp", "")),
            ))

    def merge(self, user_id: str, merger) -> dict:
        """原子读-改-写（安全修复 P2）：单事务内读-改-写。

        进程内按用户互斥；MySQL 额外使用连接级 GET_LOCK 覆盖多实例并发。

        MySQL GET_LOCK is connection-scoped rather than transaction-scoped.  The
        lock therefore must remain held until the transaction has reached its
        commit/rollback boundary.  Releasing it from a ``finally`` nested inside
        ``engine.begin()`` lets another pod read stale rows while this transaction
        is still uncommitted, causing a lost merge.
        """
        try:
            with self._merge_lock(user_id):
                # Use an explicit transaction so RELEASE_LOCK is unambiguously
                # after either commit or rollback.  ``with engine.begin()`` cannot
                # express that ordering when release is in its body/finally block.
                conn = self._engine.connect()
                mysql_lock = ""
                lock_acquired = False
                transaction = None
                try:
                    if conn.dialect.name == "mysql":
                        digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:32]
                        mysql_lock = f"ecom_ltm_{digest}"
                        acquired = conn.execute(
                            text("SELECT GET_LOCK(:name, 10)"), {"name": mysql_lock},
                        ).scalar()
                        if acquired != 1:
                            raise StorageUnavailableError(
                                f"SQL 记忆锁获取超时: {user_id}"
                            )
                        lock_acquired = True
                        # SQLAlchemy starts an implicit transaction for the
                        # GET_LOCK statement.  End that transaction before
                        # beginning the data transaction: on MySQL REPEATABLE
                        # READ this makes the subsequent read snapshot start only
                        # after the cross-pod lock has been acquired.
                        conn.commit()
                    transaction = conn.begin()
                    current = self._load_via(conn, user_id)
                    updated = merger(current)
                    self._write_via(conn, user_id, updated)
                    transaction.commit()
                    return updated
                except BaseException:
                    # Roll back before releasing a connection-level MySQL lock so
                    # waiters never observe a partially committed merge.
                    if transaction is not None and transaction.is_active:
                        transaction.rollback()
                    elif conn.in_transaction():
                        # Covers GET_LOCK/acquire or begin failures before the
                        # explicit data transaction exists.
                        conn.rollback()
                    raise
                finally:
                    if lock_acquired:
                        try:
                            conn.execute(
                                text("SELECT RELEASE_LOCK(:name)"),
                                {"name": mysql_lock},
                            )
                        except Exception:  # noqa: BLE001 - connection close is fallback
                            # The transaction has already completed.  A failed
                            # release is safe to contain: closing the connection
                            # makes MySQL release the connection-level lock.
                            pass
                    conn.close()
        except StorageUnavailableError:
            raise
        except Exception as e:  # noqa: BLE001
            raise StorageUnavailableError(f"SQL 记忆合并失败: {e}") from e
