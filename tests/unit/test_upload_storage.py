"""分片/现场存储单测：内容寻址 key、tmp 并发不互踩、同卷 move、journal 原子写。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.stores.upload_storage import (
    LocalChunkStorage,
    OriginalStore,
    S3ChunkStorage,
    StageDir,
    UploadStorageError,
    _object_name,
)


@pytest.fixture()
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path


class TestObjectName:
    def test_full_sha256_in_key(self):
        sha = "x" * 64
        assert _object_name(3, sha) == f"00003-{sha}"
        assert len(_object_name(3, sha).split("-")[1]) == 64


class TestLocalChunkStorage:
    def test_write_finalize_list(self, tmp_root):
        s = LocalChunkStorage(tmp_root / "chunks")
        s.write_temp("u1", 0, "t1", b"hello")
        assert (tmp_root / "chunks" / "u1" / "00000.t1.tmp").exists()
        key = s.finalize_object("u1", 0, "t1", "a" * 64)
        assert key == f"00000-{'a' * 64}"
        assert s.list_chunks("u1") == [0]
        assert s.get_chunk("u1", 0) == b"hello"

    def test_concurrent_same_seq_different_tokens_not_clobber(self, tmp_root):
        s = LocalChunkStorage(tmp_root / "chunks")
        s.write_temp("u1", 0, "tA", b"aaa")
        s.write_temp("u1", 0, "tB", b"bbb")
        items = sorted((tmp_root / "chunks" / "u1").glob("*"))
        assert len(items) == 2  # 两个临时对象并存
        assert s.list_chunks("u1") == []  # 未 finalize 不算分片
        s.finalize_object("u1", 0, "tA", "a" * 64)
        assert s.get_chunk("u1", 0) == b"aaa"

    def test_finalize_missing_tmp_raises(self, tmp_root):
        s = LocalChunkStorage(tmp_root / "chunks")
        with pytest.raises(UploadStorageError):
            s.finalize_object("u1", 0, "tX", "a" * 64)

    def test_chunk_exists_and_delete(self, tmp_root):
        s = LocalChunkStorage(tmp_root / "chunks")
        s.write_temp("u1", 0, "t1", b"x")
        s.finalize_object("u1", 0, "t1", "a" * 64)
        assert s.chunk_exists("u1", 0) is True
        s.delete_object("u1", 0)
        assert s.chunk_exists("u1", 0) is False
        s.write_temp("u1", 1, "t2", b"y")
        s.finalize_object("u1", 1, "t2", "b" * 64)
        s.delete_chunks("u1")
        assert s.list_chunks("u1") == []


class _FakeObjectStore:
    """ObjectStore 协议内存替身（put/get/list/delete/delete_prefix，2.8）。"""

    def __init__(self):
        self._data: dict[str, bytes] = {}

    def put(self, key: str, data: bytes) -> None:
        self._data[key] = data

    def get(self, key: str) -> bytes | None:
        return self._data.get(key)

    def list(self, prefix: str = "") -> list[str]:
        return sorted(k for k in self._data if k.startswith(prefix))

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def delete_prefix(self, prefix: str) -> None:
        for k in list(self._data):
            if k.startswith(prefix):
                self._data.pop(k, None)

    def healthcheck(self) -> bool:
        return True


class TestS3ChunkStorage:
    def test_finalize_and_list(self):
        os_ = _FakeObjectStore()
        s = S3ChunkStorage(os_)
        s.write_temp("u1", 0, "t1", b"hello")
        assert s.list_chunks("u1") == []  # tmp 不算
        key = s.finalize_object("u1", 0, "t1", "a" * 64)
        assert key == f"00000-{'a' * 64}"
        assert s.list_chunks("u1") == [0]
        assert s.get_chunk("u1", 0) == b"hello"
        assert s.chunk_exists("u1", 0) is True

    def test_cross_pod_shared_visibility(self):
        os_ = _FakeObjectStore()
        pod_a = S3ChunkStorage(os_)
        pod_b = S3ChunkStorage(os_)
        pod_a.write_temp("u1", 0, "t1", b"data")
        pod_a.finalize_object("u1", 0, "t1", "a" * 64)
        assert pod_b.list_chunks("u1") == [0]  # Pod B 可见 Pod A 的分片


class TestStageDir:
    def test_write_merged_and_move_into_kb(self, tmp_root):
        stage = StageDir(tmp_root / "kb" / ".staging", tmp_root / "kb" / ".trash")
        stage.write_merged("u1", b"# title\n\nbody")
        assert stage.merged_exists("u1")
        dst = stage.move_into_kb("u1", tmp_root / "kb" / "uploads", "x.md")
        assert dst.exists()
        assert dst.read_text(encoding="utf-8") == "# title\n\nbody"
        assert not stage.merged_exists("u1")

    def test_move_out_of_kb(self, tmp_root):
        stage = StageDir(tmp_root / "kb" / ".staging", tmp_root / "kb" / ".trash")
        uploads = tmp_root / "kb" / "uploads"
        uploads.mkdir(parents=True)
        (uploads / "x.md").write_text("x", encoding="utf-8")
        p = stage.move_out_of_kb(uploads, "x.md", "doc-1")
        assert p.exists()
        assert not (uploads / "x.md").exists()

    def test_trash_roundtrip(self, tmp_root):
        stage = StageDir(tmp_root / "kb" / ".staging", tmp_root / "kb" / ".trash")
        uploads = tmp_root / "kb" / "uploads"
        uploads.mkdir(parents=True)
        (uploads / "x.md").write_text("x", encoding="utf-8")
        stage.move_to_trash(uploads, "x.md")
        assert stage.trash_exists("x.md")
        assert not (uploads / "x.md").exists()
        stage.restore_from_trash(uploads, "x.md")
        assert (uploads / "x.md").exists()
        assert not stage.trash_exists("x.md")
        stage.trash_delete("x.md")

    def test_journal_atomic_write_read_list_clear(self, tmp_root):
        stage = StageDir(tmp_root / "kb" / ".staging", tmp_root / "kb" / ".trash")
        stage.journal_write("up-1", {"phase": "PREPARED", "generation_id": "g1"})
        stage.journal_write("del-doc1", {"phase": "INDEX_BUILT"})
        assert stage.journal_list() == ["del-doc1", "up-1"]
        assert stage.journal_read("up-1")["phase"] == "PREPARED"
        stage.journal_clear("up-1")
        assert stage.journal_read("up-1") is None
        assert stage.journal_list() == ["del-doc1"]

    def test_cleanup_staging_and_failed(self, tmp_root):
        stage = StageDir(tmp_root / "kb" / ".staging", tmp_root / "kb" / ".trash")
        stage.write_merged("u1", b"x")
        stage.journal_write("u1", {"phase": "PREPARED"})
        stage.cleanup_staging("u1")
        assert not stage.merged_exists("u1")
        assert stage.journal_read("u1") is None
        uploads = tmp_root / "kb" / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        (uploads / "y.md").write_text("y", encoding="utf-8")
        stage.move_out_of_kb(uploads, "y.md", "d1")
        assert stage.failed_path("d1", "y.md").exists()
        stage.cleanup_failed("d1")
        assert not stage.failed_path("d1", "y.md").exists()


class TestOriginalStore:
    def test_put_get_uses_validated_ext(self, tmp_root):
        os_ = OriginalStore(tmp_root / "originals")
        key = os_.put("doc-1", "pdf", b"%PDF-1.7")
        assert "用户文件名" not in key  # filename 绝不进路径
        assert key == f"doc-1{os.sep}original.pdf".replace("\\", "/") or key.endswith("original.pdf")
        assert os_.get("doc-1", "pdf") == b"%PDF-1.7"
        assert os_.exists("doc-1", "pdf") is True
        os_.delete("doc-1")
        assert os_.exists("doc-1", "pdf") is False

    def test_diff_ext_isolated(self, tmp_root):
        os_ = OriginalStore(tmp_root / "originals")
        os_.put("doc-1", "pdf", b"pdf")
        assert os_.get("doc-1", "md") is None
