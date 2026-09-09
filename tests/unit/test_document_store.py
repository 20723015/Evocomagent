"""kb_documents / kb_control 存储层单测（sqlite 方言，全程无网络）。"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.stores.base import StorageUnavailableError
from app.stores.sql.document_store import (
    DocumentRecord,
    DocumentStateError,
    KbControlStore,
    SqlDocumentStore,
    STATUS_CANCELLED,
    STATUS_DELETING,
    STATUS_DELETED,
    STATUS_DELETE_QUEUED,
    STATUS_FAILED,
    STATUS_INDEXED,
    STATUS_INDEXING,
    STATUS_UPLOADING,
    STATUS_VALIDATING,
)


@pytest.fixture()
def engine():
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    from app.stores.sql.schema import metadata

    metadata.create_all(eng)
    return eng


def _record(upload_id="up-1", **overrides) -> DocumentRecord:
    base = dict(
        doc_id="doc-1",
        upload_id=upload_id,
        uploader="ops-a",
        storage_key=f"x-{upload_id}.md",
        filename="退换货政策.pdf",
        format="pdf",
        size_bytes=1024,
        sha256="a" * 64,
        chunk_size=1024 * 1024,
        upload_chunk_count=1,
        owner="ops",
        provenance=f"upload:{upload_id}",
        expires_at=(datetime.now() + timedelta(days=1)).isoformat(timespec="seconds"),
    )
    base.update(overrides)
    return DocumentRecord(**base)


class TestSqlDocumentStore:
    def test_create_and_get(self, engine):
        store = SqlDocumentStore(engine)
        rec = store.create(_record())
        assert rec.status == STATUS_UPLOADING
        got = store.get("doc-1")
        assert got.doc_id == "doc-1"
        assert got.uploader == "ops-a"
        assert got.provenance == "upload:up-1"
        assert got.version == 0

    def test_create_idempotent_by_upload_id(self, engine):
        store = SqlDocumentStore(engine)
        store.create(_record(upload_id="up-1", doc_id="doc-1"))
        second = store.create(_record(upload_id="up-1", doc_id="doc-2"))
        assert second.doc_id == "doc-1"  # 幂等返回现有行

    def test_get_by_upload_id_and_storage_key(self, engine):
        store = SqlDocumentStore(engine)
        store.create(_record())
        assert store.get_by_upload_id("up-1").doc_id == "doc-1"
        assert store.get_by_storage_key("x-up-1.md").doc_id == "doc-1"

    def test_update_status_cas_happy_path(self, engine):
        store = SqlDocumentStore(engine)
        rec = store.create(_record())
        new_ver = store.update_status(
            "doc-1", STATUS_UPLOADING, STATUS_VALIDATING, rec.version,
            operation_id="op-1",  # 抢占：VALUES 写入新 op
        )
        assert new_ver == 1
        got = store.get("doc-1")
        assert got.status == STATUS_VALIDATING
        assert got.operation_id == "op-1"
        assert got.version == 1
        assert got.status_changed_at  # 时间戳随迁移刷新

    def test_update_status_cas_conflict_returns_minus_one(self, engine):
        store = SqlDocumentStore(engine)
        rec = store.create(_record())
        # 版本不同步 → -1（并发写入者）
        assert store.update_status(
            "doc-1", STATUS_UPLOADING, STATUS_VALIDATING, rec.version + 99,
        ) == -1
        # 状态不同步 → -1（他处已迁移）
        assert store.update_status(
            "doc-1", STATUS_UPLOADING, STATUS_VALIDATING, rec.version,
        ) == 1
        assert store.update_status(
            "doc-1", STATUS_UPLOADING, STATUS_VALIDATING, rec.version + 1,
        ) == -1

    def test_update_status_requires_operation_id_when_given(self, engine):
        store = SqlDocumentStore(engine)
        rec = store.create(_record())
        store.update_status("doc-1", STATUS_UPLOADING, STATUS_VALIDATING,
                            rec.version, operation_id="op-1")
        # require_op 与行上现有 op 不符 → -1（过期请求不得覆盖新处理）
        assert store.update_status(
            "doc-1", STATUS_VALIDATING, STATUS_INDEXING, rec.version + 1,
            require_op="op-other",
        ) == -1
        # 匹配 → 通过
        assert store.update_status(
            "doc-1", STATUS_VALIDATING, STATUS_INDEXING, rec.version + 1,
            require_op="op-1",
        ) == 2

    def test_illegal_transition_rejected(self, engine):
        store = SqlDocumentStore(engine)
        rec = store.create(_record())
        with pytest.raises(DocumentStateError):
            store.update_status("doc-1", STATUS_UPLOADING, STATUS_INDEXED, rec.version)
        with pytest.raises(DocumentStateError):
            store.update_status("doc-1", STATUS_FAILED, STATUS_VALIDATING, rec.version)
        # failed 是终态：无 failed→validating（评审冻结）
        assert store.get("doc-1").status == STATUS_UPLOADING

    def test_full_lifecycle_cas_chain(self, engine):
        store = SqlDocumentStore(engine)
        rec = store.create(_record())
        v = store.update_status("doc-1", STATUS_UPLOADING, STATUS_VALIDATING, rec.version, operation_id="op")
        v = store.update_status("doc-1", STATUS_VALIDATING, STATUS_INDEXING, v, require_op="op")
        v = store.update_status(
            "doc-1", STATUS_INDEXING, STATUS_INDEXED, v, require_op="op",
            generation_id="g-1", indexed_chunk_count=5,
        )
        assert v == 3
        got = store.get("doc-1")
        assert got.status == STATUS_INDEXED
        assert got.generation_id == "g-1"
        assert got.indexed_chunk_count == 5
        # indexed → deleting → deleted
        v = store.update_status("doc-1", STATUS_INDEXED, STATUS_DELETING, v, require_op="op")
        v = store.update_status("doc-1", STATUS_DELETING, STATUS_DELETED, v, require_op="op")
        assert store.get("doc-1").status == STATUS_DELETED

    def test_deleting_rolls_back_to_delete_queued(self, engine):
        """下架遇锁等待回退 delete_queued 重试是设计内路径（indexed →
        delete_queued → deleting → deleted 状态链，upload_service 依赖）。"""
        store = SqlDocumentStore(engine)
        rec = store.create(_record())
        v = store.update_status("doc-1", STATUS_UPLOADING, STATUS_VALIDATING, rec.version, operation_id="op")
        v = store.update_status("doc-1", STATUS_VALIDATING, STATUS_INDEXING, v, require_op="op")
        v = store.update_status("doc-1", STATUS_INDEXING, STATUS_INDEXED, v, require_op="op")
        v = store.update_status("doc-1", STATUS_INDEXED, STATUS_DELETE_QUEUED, v, require_op="op")
        v = store.update_status("doc-1", STATUS_DELETE_QUEUED, STATUS_DELETING, v, require_op="op")
        # 锁等待回退：deleting → delete_queued（重试）
        v2 = store.update_status("doc-1", STATUS_DELETING, STATUS_DELETE_QUEUED, v, require_op="op")
        assert v2 == v + 1
        assert store.get("doc-1").status == STATUS_DELETE_QUEUED
        # 重试后仍可走到 deleted
        v3 = store.update_status("doc-1", STATUS_DELETE_QUEUED, STATUS_DELETING, v2, require_op="op")
        store.update_status("doc-1", STATUS_DELETING, STATUS_DELETED, v3, require_op="op")
        assert store.get("doc-1").status == STATUS_DELETED

    def test_system_error_rolls_back_to_uploading(self, engine):
        store = SqlDocumentStore(engine)
        rec = store.create(_record())
        v = store.update_status("doc-1", STATUS_UPLOADING, STATUS_VALIDATING, rec.version, operation_id="op")
        v = store.update_status("doc-1", STATUS_VALIDATING, STATUS_INDEXING, v, require_op="op")
        # 系统错误回退（锁冲突/构建异常）
        v2 = store.update_status("doc-1", STATUS_INDEXING, STATUS_UPLOADING, v, require_op="op")
        assert v2 == v + 1
        assert store.get("doc-1").status == STATUS_UPLOADING

    def test_expire_orphans(self, engine):
        store = SqlDocumentStore(engine)
        old = _record(upload_id="up-old", doc_id="doc-old",
                      expires_at=(datetime.now() - timedelta(days=1)).isoformat(timespec="seconds"))
        store.create(old)
        store.create(_record(upload_id="up-fresh", doc_id="doc-fresh"))
        n = store.expire_orphans(datetime.now())
        assert n == 1
        assert store.get("doc-old").status == STATUS_CANCELLED
        # 审计行不删除
        assert store.get("doc-old") is not None
        assert store.get("doc-fresh").status == STATUS_UPLOADING

    def test_list_filters(self, engine):
        store = SqlDocumentStore(engine)
        store.create(_record(upload_id="a", doc_id="d1"))
        store.create(_record(upload_id="b", doc_id="d2"))
        assert len(store.list()) == 2
        assert len(store.list(status=STATUS_UPLOADING)) == 2
        assert len(store.list(status=STATUS_INDEXED)) == 0

    def test_list_before_status_changed(self, engine):
        store = SqlDocumentStore(engine)
        store.create(_record(upload_id="a", doc_id="d1"))
        rec = store.get("d1")
        store.update_status("d1", STATUS_UPLOADING, STATUS_FAILED, rec.version,
                            error="解析失败")
        cutoff = datetime.now() - timedelta(days=31)
        assert store.list_before_status_changed((STATUS_FAILED,), cutoff) == []
        far_cutoff = datetime.now() + timedelta(days=1)
        rows = store.list_before_status_changed((STATUS_FAILED,), far_cutoff)
        assert len(rows) == 1 and rows[0].status == STATUS_FAILED

class TestKbControlStore:
    def test_set_get_cas(self, engine):
        ctl = KbControlStore(engine)
        assert ctl.get("kb_write_blocked") == ""
        v0 = ctl.set("kb_write_blocked", "reconcile-needed")
        assert ctl.get("kb_write_blocked") == "reconcile-needed"
        # 插入行 version=0；cas 后 +1
        assert ctl.cas("kb_write_blocked", v0, "cleared") is True
        assert ctl.version("kb_write_blocked") == v0 + 1

    def test_cas_happy_and_conflict(self, engine):
        ctl = KbControlStore(engine)
        ctl.set("k", "v0")
        v = ctl.version("k")
        assert ctl.cas("k", v, "v1") is True
        assert ctl.get("k") == "v1"
        assert ctl.cas("k", v, "v2") is False  # 版本过期（已被 cas 到 v+1）
        assert ctl.get("k") == "v1"

    def test_cas_missing_key(self, engine):
        ctl = KbControlStore(engine)
        assert ctl.cas("missing", 0, "x") is False

    def test_delete(self, engine):
        ctl = KbControlStore(engine)
        ctl.set("k", "v")
        ctl.delete("k")
        assert ctl.get("k") == ""
