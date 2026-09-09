"""KB 异步建库测试公共装配（job_store / worker / API 三层共用）。

与 test_upload_service._build_service 同一伪造基座（全程无网络）：
sqlite + 进程内上传状态 + LocalChunkStorage + numpy 后端 + FakeEmbedder +
文件锁（monkeypatch 隔离到 tmp），额外构造 KbIndexJobStore 并注入 service。
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.agent.rag.job_store import KbIndexJobStore
from app.agent.rag.parsers import chunk_kb_dir
from app.agent.rag.upload_service import DocumentUploadService
from app.config.settings import settings
from app.evolution.generation import GenerationStore
from app.evolution.index_service import IndexBuildService
from app.stores.sql.document_store import KbControlStore, SqlDocumentStore
from app.stores.upload_state import InProcessUploadStateStore
from app.stores.upload_storage import LocalChunkStorage, OriginalStore, StageDir


class FakeEmbedder:
    model = "fake-embed"

    def __init__(self, *, fail_after: int | None = None, batch_hook=None):
        self._fail_after = fail_after  # 成功 N 次 encode 后开始失败（None=从不）
        self._calls = 0
        self._hook = batch_hook

    def encode(self, texts, timeout=None):
        if self._fail_after is not None and self._calls >= self._fail_after:
            raise RuntimeError("embedding 服务不可用（注入）")
        self._calls += 1
        if self._hook is not None:
            self._hook(texts)
        return [[0.1, 0.2]] * len(texts)

    def encode_one(self, text, timeout=None):
        return [0.1, 0.2]


def pad(content: bytes, size: int = 70000) -> bytes:
    """垫长到两片（最小合法分片 64KiB）；截断回退到合法 UTF-8 边界。"""
    out = (content * (size // len(content) + 1))[:size]
    while out:
        try:
            out.decode("utf-8")
            return out
        except UnicodeDecodeError:
            out = out[:-1]
    return out


def build_async_kb(tmp_path: Path, monkeypatch, *, embedder=None,
                   backoff_base: int = 0, **store_kwargs):
    """构造（service, job_store, deps）三元组；deps=(doc_store, control, gen, kb)。"""
    monkeypatch.setattr(settings, "kb_write_lock_backend", "file")
    monkeypatch.setattr(settings, "evolve_state_dir", str(tmp_path / "evo"))
    monkeypatch.setattr(settings, "rag_backend", "numpy")

    kb = tmp_path / "kb"
    (kb / "uploads").mkdir(parents=True)
    (kb / ".staging").mkdir()
    (kb / ".trash").mkdir()
    (kb / "根文档.md").write_text("# 根文档\n\n## 说明\n已有内容\n", encoding="utf-8")

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    from app.stores.sql.schema import metadata

    metadata.create_all(engine)
    doc_store = SqlDocumentStore(engine)
    control = KbControlStore(engine)
    gen = GenerationStore(tmp_path / "kb_generations.json")
    index = IndexBuildService(
        embedder=embedder or FakeEmbedder(), kb_dir=kb, generation_store=gen,
        backend_settings={"kb_index_path": str(tmp_path / "kb_index.json")},
        chunker=chunk_kb_dir,
        strict_build=True,
    )
    job_store = KbIndexJobStore(engine, backoff_base=backoff_base, **store_kwargs)
    svc = DocumentUploadService(
        doc_store=doc_store, control_store=control, generation_store=gen,
        index_service=index, state_store=InProcessUploadStateStore(),
        chunk_storage=LocalChunkStorage(tmp_path / "chunks"),
        stage=StageDir(kb / ".staging", kb / ".trash"),
        originals=OriginalStore(tmp_path / "originals"),
        engine=None, redis=None, kb_root=kb,
        job_store=job_store,
    )
    return svc, job_store, (doc_store, control, gen, kb)


def enable_async(control, enabled: bool = True) -> None:
    """共享开关（kb_control 正本）：生产两阶段上线的第二步。"""
    control.set("kb_async_enabled", "1" if enabled else "0")


def upload_chunks(svc, upload_id="up-1", filename="补充说明.md",
                  content=None, chunk_size=65536):
    """创建会话并传满分片（默认垫长到两片）。"""
    content = content if content is not None else pad(
        "# 退换货补充说明\n\n## 适用范围\n\n支持七天无理由退货。\n".encode(),
    )
    cresp = svc.create_upload(
        uploader="ops-a", filename=filename, size_bytes=len(content),
        content_type="text/markdown", chunk_size=chunk_size, upload_id=upload_id,
    )
    total = cresp["total_chunks"]
    for seq in range(total):
        start = seq * chunk_size
        svc.put_chunk(upload_id, seq, content[start:start + chunk_size])
    return cresp, total
