"""SqlDocumentStore / KbControlStore：KB 文档上传元数据与全局控制表的 SQL 正本。

状态机（v7 冻结；全部迁移走 CAS：WHERE status + version，处理路径加
operation_id 三条件；update 同事务刷新 status_changed_at）：

    uploading → validating → indexing → indexed
    uploading/validating → cancelled（主动取消/会话过期）
    输入错误（hash/magic/解析/消毒/超限）→ failed（终态，无 failed→validating）
    系统错误（锁冲突/构建异常/提交点前失败）→ 回退 uploading
    indexed → deleting → deleted；deleting 仅提交点前回滚 indexed

异常语义与 SqlSessionStore 一致：IntegrityError → 可重试冲突（409 语义），
其余 → StorageUnavailableError（503）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from app.stores.base import StorageUnavailableError
from app.stores.sql.schema import kb_control, kb_documents

# ---- 状态常量 ----
STATUS_UPLOADING = "uploading"
STATUS_VALIDATING = "validating"
STATUS_INDEXING = "indexing"
STATUS_INDEXED = "indexed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_DELETING = "deleting"
STATUS_DELETED = "deleted"

# 合法迁移表（起点 → 可达终点）
_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    STATUS_UPLOADING: {STATUS_VALIDATING, STATUS_FAILED, STATUS_CANCELLED},
    STATUS_VALIDATING: {STATUS_INDEXING, STATUS_UPLOADING, STATUS_FAILED, STATUS_CANCELLED},
    STATUS_INDEXING: {STATUS_INDEXED, STATUS_UPLOADING, STATUS_FAILED},
    STATUS_INDEXED: {STATUS_DELETING},
    STATUS_DELETING: {STATUS_DELETED, STATUS_INDEXED},
    STATUS_FAILED: set(),
    STATUS_CANCELLED: set(),
    STATUS_DELETED: set(),
}

TERMINAL = frozenset({STATUS_INDEXED, STATUS_FAILED, STATUS_CANCELLED, STATUS_DELETED})


class DocumentStateError(ValueError):
    """非法状态迁移（映射 409）。"""


@dataclass
class DocumentRecord:
    doc_id: str
    upload_id: str
    storage_key: str
    filename: str
    format: str
    size_bytes: int
    sha256: str
    uploader: str = ""
    original_key: str = ""
    content_type: str = ""
    chunk_size: int = 0
    status: str = STATUS_UPLOADING
    operation_id: str = ""
    pending_generation_id: str = ""
    upload_chunk_count: int = 0
    indexed_chunk_count: int = 0
    generation_id: str = ""
    owner: str = "ops"
    provenance: str = ""
    error: str = ""
    version: int = 0
    status_changed_at: str = ""
    expires_at: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def as_dto_dict(self) -> dict:
        return _iso_free(self.to_dict())


def _iso_free(data: dict) -> dict:
    """去掉 Python 化字段（status_changed_at 等时间列由 SQL 层维护，不回写）。"""
    return data


def assert_transition(src: str, dst: str) -> None:
    allowed = _ALLOWED_TRANSITIONS.get(src)
    if allowed is None:
        raise DocumentStateError(f"未知状态: {src}")
    if dst not in allowed:
        raise DocumentStateError(f"非法状态迁移: {src} → {dst}")


def _row_to_record(row) -> DocumentRecord:
    d = dict(row)
    for key in ("status_changed_at", "expires_at", "created_at", "updated_at"):
        v = d.get(key)
        if v is not None and hasattr(v, "isoformat"):
            d[key] = v.isoformat(sep=" ", timespec="seconds")
        else:
            d[key] = str(v) if v is not None else ""
    return DocumentRecord(**d)


class SqlDocumentStore:
    """SQLAlchemy 实现（生产 mysql+pymysql；本地/测试 sqlite 同表）。"""

    def __init__(self, engine):
        self._engine = engine

    # ---------- 读 ----------
    def _fetch_rows(self, condition, limit: int = 100):
        try:
            with self._engine.connect() as conn:
                q = select(kb_documents).order_by(
                    kb_documents.c.updated_at.desc()
                ).limit(min(max(limit, 1), 500))
                if condition is not None:
                    q = q.where(condition)
                return [dict(r) for r in conn.execute(q).mappings().all()]
        except Exception as e:  # noqa: BLE001 —— 断连即存储不可用（503 语义）
            raise StorageUnavailableError(f"SQL 读取失败: {e}") from e

    def get(self, doc_id: str) -> Optional[DocumentRecord]:
        rows = self._fetch_rows(kb_documents.c.doc_id == doc_id, limit=1)
        return _row_to_record(rows[0]) if rows else None

    def get_by_upload_id(self, upload_id: str) -> Optional[DocumentRecord]:
        rows = self._fetch_rows(kb_documents.c.upload_id == upload_id, limit=1)
        return _row_to_record(rows[0]) if rows else None

    def get_by_storage_key(self, storage_key: str) -> Optional[DocumentRecord]:
        rows = self._fetch_rows(kb_documents.c.storage_key == storage_key, limit=1)
        return _row_to_record(rows[0]) if rows else None

    def list(self, status: str = "", limit: int = 100) -> list[DocumentRecord]:
        cond = kb_documents.c.status == status if status else None
        return [_row_to_record(r) for r in self._fetch_rows(cond, limit=limit)]

    def list_before_status_changed(self, statuses: tuple[str, ...], cutoff: datetime) -> list[DocumentRecord]:
        """保留期扫描（GC）：状态在集合内且 status_changed_at 早于 cutoff。"""
        cond = (
            kb_documents.c.status.in_(statuses)
            & kb_documents.c.status_changed_at.isnot(None)
            & (kb_documents.c.status_changed_at < cutoff)
        )
        return [_row_to_record(r) for r in self._fetch_rows(cond, limit=500)]

    # ---------- 写 ----------
    def create(self, record: DocumentRecord) -> DocumentRecord:
        """新建（幂等：upload_id 已存在 → 返回现有行；参数一致性比对由服务层负责）。"""
        existing = self.get_by_upload_id(record.upload_id)
        if existing is not None:
            return existing
        values = _record_to_db_values(record)
        values.update({
            "status_changed_at": datetime.now(),
            "expires_at": _as_dt(record.expires_at),
        })
        try:
            with self._engine.begin() as conn:
                conn.execute(kb_documents.insert().values(**values))
        except IntegrityError as e:
            existing = self.get_by_upload_id(record.upload_id)
            if existing is not None:
                return existing  # 并发创建撞唯一键 → 幂等返回
            raise StorageUnavailableError(
                f"SQL 写入失败（唯一键冲突且非重复创建）: {e.orig}"
            ) from e
        except Exception as e:  # noqa: BLE001
            raise StorageUnavailableError(f"SQL 写入失败: {e}") from e
        got = self.get_by_upload_id(record.upload_id)
        assert got is not None
        return got

    def update_status(
        self,
        doc_id: str,
        src: str,
        dst: str,
        expected_version: int,
        *,
        require_op: Optional[str] = None,
        **fields,
    ) -> int:
        """CAS 状态迁移：WHERE status=src AND version=expected（+ require_op 三条件）。

        - require_op：处理路径上校验行中现有 operation_id（旧值），防止已过期请求
          覆盖新处理；抢占迁移（uploading→validating）行上还没有本 op，
          通过 fields 传入 operation_id=new 并在 VALUES 写入；require_op 留 None。
        - fields 中的 operation_id 是**新值**（随迁移写入），非 WHERE 条件。
        返回新版本号；非法迁移 DocumentStateError；并发/失配返回 -1（调用方 409）。
        """
        assert_transition(src, dst)
        values = dict(fields)
        values["status"] = dst
        values["version"] = expected_version + 1
        values["status_changed_at"] = datetime.now()
        where = [
            kb_documents.c.doc_id == doc_id,
            kb_documents.c.status == src,
            kb_documents.c.version == expected_version,
        ]
        if require_op is not None:
            where.append(kb_documents.c.operation_id == require_op)
        try:
            with self._engine.begin() as conn:
                updated = conn.execute(update(kb_documents).where(*where).values(**values))
        except Exception as e:  # noqa: BLE001
            raise StorageUnavailableError(f"SQL 写入失败: {e}") from e
        return expected_version + 1 if updated.rowcount == 1 else -1

    def expire_orphans(self, before: Optional[datetime] = None) -> int:
        """懒清理：expires_at 早于 before 的 uploading 行 → cancelled（不删审计行）。"""
        if before is None:
            before = datetime.now().replace(microsecond=0)
        try:
            with self._engine.begin() as conn:
                updated = conn.execute(
                    update(kb_documents)
                    .where(
                        kb_documents.c.status == STATUS_UPLOADING,
                        kb_documents.c.expires_at.isnot(None),
                        kb_documents.c.expires_at < before,
                    )
                    .values(
                        status=STATUS_CANCELLED,
                        error="上传会话已过期，自动取消",
                        version=kb_documents.c.version + 1,
                        status_changed_at=datetime.now(),
                    )
                )
        except Exception as e:  # noqa: BLE001
            raise StorageUnavailableError(f"SQL 写入失败: {e}") from e
        return int(updated.rowcount or 0)

    def delete_row(self, doc_id: str) -> None:
        """物理删除审计行（reconcile 手工处理路径用；常规删除只置 deleted 状态）。"""
        try:
            with self._engine.begin() as conn:
                conn.execute(kb_documents.delete().where(kb_documents.c.doc_id == doc_id))
        except Exception as e:  # noqa: BLE001
            raise StorageUnavailableError(f"SQL 删除失败: {e}") from e


def _record_to_db_values(record: DocumentRecord) -> dict:
    return {
        "doc_id": record.doc_id,
        "upload_id": record.upload_id,
        "uploader": record.uploader,
        "storage_key": record.storage_key,
        "original_key": record.original_key,
        "filename": record.filename,
        "content_type": record.content_type,
        "format": record.format,
        "chunk_size": record.chunk_size,
        "size_bytes": record.size_bytes,
        "sha256": record.sha256,
        "status": record.status,
        "operation_id": record.operation_id,
        "pending_generation_id": record.pending_generation_id,
        "upload_chunk_count": record.upload_chunk_count,
        "indexed_chunk_count": record.indexed_chunk_count,
        "generation_id": record.generation_id,
        "owner": record.owner,
        "provenance": record.provenance,
        "error": record.error,
        "version": record.version,
    }


def _as_dt(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace(" ", "T"))
    except (ValueError, TypeError):
        return None


# ============================================================
# KbControlStore：全局控制表（blk 标记等；值走 SQL，多 Pod 一致）
# ============================================================
class KbControlStore:
    """kb_control 表访问：get/set/cas/delete（version 递增；兼容 sqlite/mysql）。"""

    def __init__(self, engine):
        self._engine = engine

    def get(self, key: str) -> str:
        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    select(kb_control.c.value).where(kb_control.c.key == key)
                ).mappings().first()
        except Exception as e:  # noqa: BLE001
            raise StorageUnavailableError(f"SQL 读取失败: {e}") from e
        return str(row["value"]) if row is not None else ""

    def version(self, key: str) -> int:
        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    select(kb_control.c.version).where(kb_control.c.key == key)
                ).mappings().first()
        except Exception as e:  # noqa: BLE001
            raise StorageUnavailableError(f"SQL 读取失败: {e}") from e
        return int(row["version"]) if row is not None else 0

    def set(self, key: str, value: str) -> int:
        """upsert（不递增 version：初始化/整体覆盖语义）。"""
        dialect = self._engine.dialect.name
        try:
            with self._engine.begin() as conn:
                if dialect == "mysql":
                    from sqlalchemy.dialects.mysql import insert as _ins

                    stmt = _ins(kb_control).values(key=key, value=value)
                    stmt = stmt.on_duplicate_key_update(value=stmt.inserted.value)
                else:
                    from sqlalchemy.dialects.sqlite import insert as _ins

                    stmt = _ins(kb_control).values(key=key, value=value)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=[kb_control.c.key], set_={"value": value},
                    )
                conn.execute(stmt)
        except Exception as e:  # noqa: BLE001
            raise StorageUnavailableError(f"SQL 写入失败: {e}") from e
        return self.version(key)

    def cas(self, key: str, expected_version: int, value: str) -> bool:
        """CAS 写：WHERE version=expected → value 更新 + version+1；返回是否生效。"""
        where = [
            kb_control.c.key == key,
            kb_control.c.version == expected_version,
        ]
        try:
            with self._engine.begin() as conn:
                updated = conn.execute(
                    update(kb_control)
                    .where(*where)
                    .values(value=value, version=expected_version + 1,
                            updated_at=datetime.now())
                )
        except Exception as e:  # noqa: BLE001
            raise StorageUnavailableError(f"SQL 写入失败: {e}") from e
        return updated.rowcount == 1

    def delete(self, key: str) -> None:
        try:
            with self._engine.begin() as conn:
                conn.execute(kb_control.delete().where(kb_control.c.key == key))
        except Exception as e:  # noqa: BLE001
            raise StorageUnavailableError(f"SQL 删除失败: {e}") from e
