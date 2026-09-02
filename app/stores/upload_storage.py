"""分片/现场存储：本地目录（默认）+ S3 对象存储（生产可选）。

多 Pod 语义（评审 R1，实施约束）：
- LocalUploadStorage：数据在本地盘——**只适用于单 Pod 部署，或多 Pod 共享
  volume 挂载同一目录**；分片可能传到 Pod A、complete 落 Pod B，本地盘不共享
  则无法合并（部署要求见 docs）。
- S3UploadStorage：分片/临时对象走对象存储（key = uploads/{upload_id}/...），
  跨 Pod 天然共享；知识库目录（knowledge/）仍要求共享卷或每 Pod 拉取——
  IndexBuildService 是本地文件扫描（部署由 helm PVC 保证）。

内容寻址（评审）：正式对象 key 含**完整 SHA-256**（{seq:05d}-{sha256}），
无覆盖语义；临时对象 {seq}.{token}.tmp（并发 PUT 不互踩）。
所有目录（staging/trash/uploads）与知识库**同挂载点**：os.replace 原子，
跨卷场景退化为复制 + fsync。
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Protocol

from app.config.settings import settings


class UploadStorageError(ValueError):
    """分片存储不可用/配置错误。"""


class ChunkStorage(Protocol):
    """分片字节存取（临时对象 + 内容寻址正式对象）。"""

    def write_temp(self, upload_id: str, seq: int, token: str, data: bytes) -> None: ...
    def finalize_object(self, upload_id: str, seq: int, token: str, sha256: str) -> str: ...
    def chunk_exists(self, upload_id: str, seq: int) -> bool: ...
    def get_chunk(self, upload_id: str, seq: int) -> Optional[bytes]: ...
    def list_chunks(self, upload_id: str) -> list[int]: ...
    def delete_object(self, upload_id: str, seq: int) -> None: ...
    def delete_temp(self, upload_id: str, seq: int, token: str) -> None: ...
    def delete_chunks(self, upload_id: str) -> None: ...


def _object_name(seq: int, sha256: str) -> str:
    return f"{int(seq):05d}-{sha256}"


class LocalChunkStorage:
    """本地目录分片存储（默认；单 Pod 或共享卷）。"""

    def __init__(self, tmp_dir: str | Path):
        self._root = Path(tmp_dir)

    def _dir(self, upload_id: str) -> Path:
        return self._root / upload_id

    def _tmp_path(self, upload_id: str, seq: int, token: str) -> Path:
        return self._dir(upload_id) / f"{int(seq):05d}.{token}.tmp"

    def _obj_path(self, upload_id: str, seq: int, sha256: str) -> Path:
        return self._dir(upload_id) / _object_name(seq, sha256)

    def write_temp(self, upload_id: str, seq: int, token: str, data: bytes) -> None:
        p = self._tmp_path(upload_id, seq, token)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def finalize_object(self, upload_id: str, seq: int, token: str, sha256: str) -> str:
        """临时对象 → 正式对象（同目录 rename 原子）；返回 object_key。"""
        tmp = self._tmp_path(upload_id, seq, token)
        if not tmp.exists():
            raise UploadStorageError(f"临时分片不存在: {tmp}")
        dst = self._obj_path(upload_id, seq, sha256)
        os.replace(tmp, dst)
        return _object_name(seq, sha256)

    def chunk_exists(self, upload_id: str, seq: int) -> bool:
        d = self._dir(upload_id)
        if not d.exists():
            return False
        for p in d.glob(f"{int(seq):05d}-*"):
            if p.is_file():
                return True
        return False

    def get_chunk(self, upload_id: str, seq: int) -> Optional[bytes]:
        d = self._dir(upload_id)
        if not d.exists():
            return None
        for p in sorted(d.glob(f"{int(seq):05d}-*")):
            if p.is_file():
                return p.read_bytes()
        return None

    def list_chunks(self, upload_id: str) -> list[int]:
        d = self._dir(upload_id)
        if not d.exists():
            return []
        out = set()
        for p in d.glob("*"):
            if not p.is_file():
                continue
            name = p.name
            if "." in name:  # tmp（{seq}.{token}.tmp）不算
                continue
            seq, _, tail = name.partition("-")
            if seq.isdigit() and len(tail) == 64:
                out.add(int(seq))
        return sorted(out)

    def delete_object(self, upload_id: str, seq: int) -> None:
        d = self._dir(upload_id)
        if not d.exists():
            return
        for p in d.glob(f"{int(seq):05d}-*"):
            if p.is_file():
                p.unlink(missing_ok=True)

    def delete_temp(self, upload_id: str, seq: int, token: str) -> None:
        """删除本次上传的临时对象（冲突/关闭场景：不动正式对象）。"""
        p = self._tmp_path(upload_id, seq, token)
        p.unlink(missing_ok=True)

    def delete_chunks(self, upload_id: str) -> None:
        d = self._dir(upload_id)
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


class S3ChunkStorage:
    """S3 分片存储（生产可选，跨 Pod 共享）：复用 ObjectStore 整对象接口。

    2.8：finalize 在正式对象写成功后删除临时对象；delete_temp/delete_object/
    delete_chunks 为真实删除（幂等），不再空操作。
    """

    def __init__(self, object_store, prefix: str = "uploads"):
        self._store = object_store
        self._prefix = prefix

    def _tmp_key(self, upload_id: str, seq: int, token: str) -> str:
        return f"{self._prefix}/{upload_id}/{int(seq):05d}.{token}.tmp"

    def _obj_key(self, upload_id: str, seq: int, sha256: str) -> str:
        return f"{self._prefix}/{upload_id}/{_object_name(seq, sha256)}"

    def write_temp(self, upload_id: str, seq: int, token: str, data: bytes) -> None:
        self._store.put(self._tmp_key(upload_id, seq, token), data)

    def finalize_object(self, upload_id: str, seq: int, token: str, sha256: str) -> str:
        data = self._store.get(self._tmp_key(upload_id, seq, token))
        if data is None:
            raise UploadStorageError(f"临时分片不存在: {self._tmp_key(upload_id, seq, token)}")
        # 正式对象写成功后删除临时对象（写失败保留临时供重试/接管）
        self._store.put(self._obj_key(upload_id, seq, sha256), data)
        self.delete_temp(upload_id, seq, token)  # 2.8：非空操作
        return _object_name(seq, sha256)

    def chunk_exists(self, upload_id: str, seq: int) -> bool:
        return any(
            k.rsplit("/", 1)[-1].startswith(f"{int(seq):05d}-")
            for k in self._store.list(f"{self._prefix}/{upload_id}/")
        )

    def get_chunk(self, upload_id: str, seq: int) -> Optional[bytes]:
        for k in self._store.list(f"{self._prefix}/{upload_id}/"):
            name = k.rsplit("/", 1)[-1]
            if name.startswith(f"{int(seq):05d}-") and "." not in name:
                return self._store.get(k)
        return None

    def list_chunks(self, upload_id: str) -> list[int]:
        out = set()
        for k in self._store.list(f"{self._prefix}/{upload_id}/"):
            name = k.rsplit("/", 1)[-1]
            if "." in name:
                continue
            seq, _, tail = name.partition("-")
            if seq.isdigit() and len(tail) == 64:
                out.add(int(seq))
        return sorted(out)

    def delete_object(self, upload_id: str, seq: int) -> None:
        """删除指定序号的正式对象（内容寻址前缀精确匹配，幂等）。"""
        prefix = f"{self._prefix}/{upload_id}/"
        for k in self._store.list(prefix):
            name = k.rsplit("/", 1)[-1]
            if name.startswith(f"{int(seq):05d}-") and "." not in name:
                self._store.delete(k)

    def delete_temp(self, upload_id: str, seq: int, token: str) -> None:
        """删除本次上传的临时对象（冲突/关闭场景：不动正式对象）。"""
        self._store.delete(self._tmp_key(upload_id, seq, token))

    def delete_chunks(self, upload_id: str) -> None:
        """清理该上传的全部对象（正式 + 残留临时），幂等。"""
        self._store.delete_prefix(f"{self._prefix}/{upload_id}")


# ============================================================
# 同卷现场目录（staging / trash / journal / originals）
# ============================================================
class StageDir:
    """complete/下架/回收的本地现场区（与 knowledge 同挂载点，rename 原子）。

    - .staging/{upload_id}/doc（合并后规范化 .md，doc.tmp 写后 fsync+rename）
    - .staging/failed/{doc_id}/{storage_key}（构建失败保留现场，**不在可扫描路径**）
    - .staging/journal/{op_id}.json（每 op 一个 journal，原子写 + fsync）
    - .trash/{storage_key}（下架隔离）
    """

    def __init__(self, staging_dir: str | Path | None = None,
                 trash_dir: str | Path | None = None):
        self._staging = Path(staging_dir or settings.kb_upload_staging_dir)
        self._trash = Path(trash_dir or settings.kb_upload_trash_dir)

    @property
    def staging_root(self) -> Path:
        return self._staging

    def merged_path(self, upload_id: str, fmt: str = "md") -> Path:
        """原始合并文件（先生成：供原件复制与子进程解析）。

        文件名带格式后缀——子进程 parse_document 按后缀分发（无后缀会被拒）。
        """
        return self._staging / upload_id / f"doc.{fmt}"

    def normalized_path(self, upload_id: str) -> Path:
        """规范化 .md（frontmatter + 正文）：入库文件。"""
        return self._staging / upload_id / "doc.md"

    def failed_path(self, doc_id: str, storage_key: str) -> Path:
        return self._staging / "failed" / doc_id / storage_key

    def journal_path(self, op_id: str) -> Path:
        return self._staging / "journal" / f"{op_id}.json"

    def trash_path(self, storage_key: str) -> Path:
        return self._trash / storage_key

    # ---------- 合并文件 ----------
    def write_merged(self, upload_id: str, data: bytes, fmt: str = "md") -> Path:
        """写合并文件：doc.{fmt}.tmp → fsync → rename（同卷原子）。"""
        p = self.merged_path(upload_id, fmt)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        _fsync_dir(p.parent)
        return p

    def read_merged(self, upload_id: str, fmt: str = "md") -> Optional[bytes]:
        p = self.merged_path(upload_id, fmt)
        return p.read_bytes() if p.exists() else None

    def merged_exists(self, upload_id: str, fmt: str = "md") -> bool:
        return self.merged_path(upload_id, fmt).exists()

    def write_normalized(self, upload_id: str, text: str) -> Path:
        """写规范化 md：doc.md.tmp → fsync → rename（同卷原子）。"""
        p = self.normalized_path(upload_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        _fsync_dir(p.parent)
        return p

    def normalized_exists(self, upload_id: str) -> bool:
        return self.normalized_path(upload_id).exists()

    # ---------- 知识库进出（同卷 rename） ----------
    def move_into_kb(self, upload_id: str, kb_uploads_dir: str | Path,
                     storage_key: str) -> Path:
        src = self.normalized_path(upload_id)
        if not src.exists():
            raise UploadStorageError(f"staging 规范化文件不存在: {src}")
        dst_dir = Path(kb_uploads_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / storage_key
        _atomic_move(src, dst)
        _fsync_dir(dst.parent)
        return dst

    def move_out_of_kb(self, kb_uploads_dir: str | Path, storage_key: str,
                       doc_id: str) -> Path:
        src = Path(kb_uploads_dir) / storage_key
        dst = self.failed_path(doc_id, storage_key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        _atomic_move(src, dst)
        return dst

    def move_to_trash(self, kb_uploads_dir: str | Path, storage_key: str) -> Path:
        src = Path(kb_uploads_dir) / storage_key
        dst = self.trash_path(storage_key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        _atomic_move(src, dst)
        return dst

    def restore_from_trash(self, kb_uploads_dir: str | Path, storage_key: str) -> Path:
        src = self.trash_path(storage_key)
        dst_dir = Path(kb_uploads_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / storage_key
        _atomic_move(src, dst)
        return dst

    # ---------- journal（每 op 一个文件；临时文件 + fsync + rename） ----------
    def journal_write(self, op_id: str, payload: dict) -> None:
        p = self.journal_path(op_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        _fsync_dir(p.parent)

    def journal_read(self, op_id: str) -> Optional[dict]:
        p = self.journal_path(op_id)
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def journal_clear(self, op_id: str) -> None:
        self.journal_path(op_id).unlink(missing_ok=True)

    def journal_list(self) -> list[str]:
        d = self._staging / "journal"
        if not d.exists():
            return []
        return [p.stem for p in sorted(d.glob("*.json")) if p.is_file()]

    # ---------- 清理 ----------
    def cleanup_staging(self, upload_id: str) -> None:
        d = self._staging / upload_id
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
        self.journal_clear(upload_id)

    def cleanup_failed(self, doc_id: str) -> None:
        d = self._staging / "failed" / doc_id
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
        self.journal_clear(f"del-{doc_id}")

    def trash_delete(self, storage_key: str) -> None:
        self.trash_path(storage_key).unlink(missing_ok=True)

    def trash_exists(self, storage_key: str) -> bool:
        return self.trash_path(storage_key).exists()


class OriginalStore:
    """原件（原格式）保存：originals/{doc_id}/original.{ext}，用户 filename 不进路径。

    2.8：可挂对象存储实现（S3ObjectStore 等）——生产原件写入
    {object_prefix}/{doc_id}/original.{ext}，而不是 Pod 本地目录；
    未注入 store 时维持本地目录（开发/单 Pod）。
    """

    def __init__(self, root: str | Path | None = None, object_store=None,
                 object_prefix: str = "originals"):
        self._root = Path(root or settings.kb_upload_original_dir)
        self._store = object_store
        self._object_prefix = object_prefix

    def _path(self, doc_id: str, ext: str) -> Path:
        return self._root / doc_id / f"original.{ext.lower()}"

    def _key(self, doc_id: str, ext: str) -> str:
        return f"{self._object_prefix}/{doc_id}/original.{ext.lower()}"

    def put(self, doc_id: str, ext: str, data: bytes) -> str:
        if self._store is not None:
            # 对象存储版：原件写对象存储（供多 Pod 共享与 /readyz 探活）
            self._store.put(self._key(doc_id, ext), data)
            return self._key(doc_id, ext)
        p = self._path(doc_id, ext)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        return str(p.relative_to(self._root))

    def get(self, doc_id: str, ext: str) -> Optional[bytes]:
        if self._store is not None:
            return self._store.get(self._key(doc_id, ext))
        p = self._path(doc_id, ext)
        return p.read_bytes() if p.exists() else None

    def exists(self, doc_id: str, ext: str) -> bool:
        if self._store is not None:
            return self._store.get(self._key(doc_id, ext)) is not None
        return self._path(doc_id, ext).exists()

    def delete(self, doc_id: str) -> None:
        if self._store is not None:
            self._store.delete_prefix(f"{self._object_prefix}/{doc_id}")
            return
        d = self._root / doc_id
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


def _atomic_move(src: Path, dst: Path) -> None:
    """同卷 os.replace；跨卷（EXDEV）退化为复制+fsync+删除。"""
    try:
        os.replace(src, dst)
    except OSError:
        shutil.move(str(src), str(dst))


def _fsync_dir(path: Path) -> None:
    """目录条目 fsync（POSIX）；Windows 无目录 fsync——跳过（rename 已原子）。"""
    if os.name == "nt":
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:  # noqa: BLE001 —— 目录 fsync 失败不阻断
        return


def build_chunk_storage(kind: Optional[str] = None, object_store=None) -> ChunkStorage:
    """分片存储工厂：local（默认）| s3（需 object_store）。"""
    kind = (kind or settings.kb_upload_storage).lower()
    if kind == "s3":
        if object_store is None:
            from app.stores.object_store import S3ObjectStore

            object_store = S3ObjectStore(settings.s3_bucket, endpoint_url=settings.s3_endpoint_url)
        return S3ChunkStorage(object_store)
    if kind in ("local", ""):
        return LocalChunkStorage(settings.kb_upload_tmp_dir)
    raise UploadStorageError(f"未知分片存储: {kind}（可选 local / s3）")
