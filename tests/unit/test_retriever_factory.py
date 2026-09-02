"""2.6 统一检索器工厂：backend/target 指定、hybrid 装配、ES 候选直查。

覆盖：
- open_retriever 按 backend + generation_target 装配（numpy 文件 / chroma
  collection / ES index_name——ES 候选不经过活动 alias）；
- hybrid 关闭返回 KnowledgeRetriever；ES+hybrid 走 ESHybridRetriever；
  numpy+hybrid 走 Python BM25+RRF；
- 候选探针（pipeline._eval_after_and_probe）记录排名/精排分数/source_path
  并阻断未进 Top-K 的候选。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agent.rag.retriever_factory import (
    RetrievalConfig,
    build_backend,
    open_retriever,
    retrieval_config_from_settings,
)


class FakeEmbedder:
    model = "fake-model"

    def encode(self, texts):
        return [[0.1] * 8 for _ in texts]

    def encode_one(self, text, timeout=None):
        return [0.1] * 8


class FakeES:
    """最小 ES 假对象：记录 create_backend('es', index_name=...) 命中。"""

    def __init__(self):
        self.created = []

    @property
    def indices(self):
        return self

    def count(self, index):
        return {"count": 0}

    def get_alias(self, name):
        # 线上路径：alias 存在 → 解析真实索引名（候选直查时不会走到这里）
        return {"ecom-kb-active": {"aliases": {name: {}}}}

    def get_mapping(self, index):
        return {index: {"mappings": {"_meta": {"embedding_model": "fake-model"}}}}

    def search(self, **kwargs):
        return {}


def test_open_retriever_numpy_target(tmp_path):
    """numpy + generation_target → 直接打开该索引文件（不走固定路径）。"""
    from app.agent.rag.backends.numpy_backend import NumpyBackend

    target = str(tmp_path / "kb_index.gen1.json")
    config = RetrievalConfig(backend="numpy")
    # 先造一个真实 numpy 索引
    from app.agent.rag.chunker import Chunk

    chunks = [Chunk(chunk_id="c1", doc="d1", section="s", text="退货政策",
                    source_path="policy.md")]
    be = NumpyBackend(index_path=Path(target))
    be.upsert(chunks=chunks, vectors=[[0.1, 0.2]], embedding_model="fake-model")

    r = open_retriever(config, embedder=FakeEmbedder(), generation_target=target)
    assert type(r.backend).__name__ == "NumpyBackend"
    hits = r.search("退货政策", top_k=1)
    assert hits[0].chunk.chunk_id == "c1"


def test_open_retriever_es_candidate_not_alias(monkeypatch):
    """ES 候选以 index_name 直查（不解析活动 alias）——build_backend 语义。"""
    from app.agent.rag.backends.es_backend import ESBackend

    calls = {}

    def fake_create(name, **kwargs):
        calls.update(name=name, **kwargs)
        es = FakeES()
        return ESBackend(es=es, index_name=kwargs.get("index_name", ""),
                         alias="ecom-kb-active", index_prefix="ecom")

    monkeypatch.setattr(
        "app.agent.rag.backends.create_backend", fake_create,
    )
    config = RetrievalConfig(backend="es", hybrid=True)
    r = open_retriever(config, embedder=FakeEmbedder(), generation_target="ecom-kb-gen1")
    assert calls["index_name"] == "ecom-kb-gen1"  # 候选直查目标，不是 alias
    assert r.backend._index == "ecom-kb-gen1"


def test_open_retriever_es_alias_default_when_no_target(monkeypatch):
    """无 generation_target 时 ES 走 alias（线上路径）。"""
    from app.agent.rag.backends.es_backend import ESBackend

    calls = {}

    def fake_create(name, **kwargs):
        calls.update(kwargs)
        return ESBackend(es=FakeES(), index_name="",
                         alias="ecom-kb-active", index_prefix="ecom")

    monkeypatch.setattr(
        "app.agent.rag.backends.create_backend", fake_create,
    )
    config = RetrievalConfig(backend="es")
    r = open_retriever(config, embedder=FakeEmbedder())
    # 未显式 target → load() 经 alias 解析出真实索引（不再是候选直查）
    assert r.backend._index == "ecom-kb-active"


def test_open_retriever_hybrid_es_uses_es_hybrid(monkeypatch):
    """ES + hybrid → ESHybridRetriever（后端原生 BM25+kNN+RRF）。"""
    from app.agent.rag.backends.es_backend import ESBackend

    monkeypatch.setattr(
        "app.agent.rag.backends.create_backend",
        lambda name, **kw: ESBackend(es=FakeES(), index_name="x",
                                     alias="a", index_prefix="ecom"),
    )
    r = open_retriever(RetrievalConfig(backend="es", hybrid=True),
                       embedder=FakeEmbedder(), generation_target="g1")
    assert type(r).__name__ == "ESHybridRetriever"


def test_open_retriever_hybrid_numpy_uses_python_hybrid(tmp_path):
    """numpy + hybrid → HybridRetriever（Python BM25+RRF 装配）。"""
    from app.agent.rag.backends.numpy_backend import NumpyBackend
    from app.agent.rag.chunker import Chunk

    target = str(tmp_path / "kb_index.h.json")
    chunks = [Chunk(chunk_id="c1", doc="d1", section="s", text="七天无理由退货",
                    source_path="policy.md")]
    NumpyBackend(index_path=Path(target)).upsert(
        chunks=chunks, vectors=[[0.1, 0.2]], embedding_model="fake-model",
    )
    r = open_retriever(
        RetrievalConfig(backend="numpy", hybrid=True, recall_k=10),
        embedder=FakeEmbedder(), generation_target=target,
    )
    assert type(r).__name__ == "HybridRetriever"
    assert r.search("退货", top_k=1)[0].chunk.chunk_id == "c1"


def test_retrieval_config_from_settings_freezes():
    cfg = retrieval_config_from_settings()
    assert isinstance(cfg, RetrievalConfig)
    assert cfg.backend == "numpy"  # 默认 settings.rag_backend


def test_index_service_open_retriever_passes_backend_settings(tmp_path):
    """IndexBuildService.open_retriever 透传 _b（测试目录）到工厂。"""
    from app.agent.rag.backends.numpy_backend import NumpyBackend
    from app.agent.rag.chunker import Chunk
    from app.agent.rag.retriever_factory import build_backend
    from app.evolution.generation import GenerationInfo, GenerationStore
    from app.evolution.index_service import IndexBuildService

    target = str(tmp_path / "idx" / "kb_index.cand.json")
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    chunks = [Chunk(chunk_id="c1", doc="d1", section="s", text="退货政策",
                    source_path="policy.md")]
    NumpyBackend(index_path=Path(target)).upsert(
        chunks=chunks, vectors=[[0.1, 0.2]], embedding_model="fake-model",
    )
    gen_store = GenerationStore(tmp_path / "gens.json")
    svc = IndexBuildService(
        embedder=FakeEmbedder(), kb_dir=tmp_path / "kb",
        generation_store=gen_store,
        backend_settings={"kb_index_path": str(tmp_path / "elsewhere.json")},
    )
    info = GenerationInfo(generation_id="g1", target=target,
                          embedding_model="fake-model")
    r = svc.open_retriever(info)
    assert r.backend._index_path == Path(target)  # 候选 target 优先于 backend_settings


# ============================================================
# 候选探针：记录排名/精排分数/source_path 并阻断
# ============================================================
class _FakeChunk:
    def __init__(self, source_path):
        self.source_path = source_path
        self.chunk_id = "x"
        self.doc = "d"
        self.section = "s"
        self.text = "t"
        self.provenance = ""


class _FakeHit:
    def __init__(self, source_path, score):
        self.chunk = _FakeChunk(source_path)
        self.score = score


class _FakeStagedRetriever:
    """可编程 staging 检索器：top-2 命中 evolved/a.md，top-5 才命中 evolved/b.md。"""

    def __init__(self):
        self._calls: list[str] = []

    def load(self):
        pass

    def search(self, query, top_k=5, timeout=None):
        self._calls.append(query)
        hits = [
            _FakeHit("evolved/a.md", 0.91),
            _FakeHit("kb/base.md", 0.80),
            _FakeHit("evolved/b.md", 0.75),
            _FakeHit("kb/other.md", 0.60),
            _FakeHit("kb/faq.md", 0.50),
        ]
        return hits[:top_k]


def test_probe_records_rank_and_blocks_missing(tmp_path, monkeypatch):
    """探针记录 rank/精排分数/source_path；未进 Top-K 的候选阻断发布。"""
    from types import SimpleNamespace

    from app.evolution.pipeline import EvolutionPipeline

    # 复用 make_services 构建真实 pipeline，只替换 staged retriever
    from tests.unit.test_pipeline import make_services

    svc = make_services(tmp_path)
    pipe: EvolutionPipeline = svc["pipeline"]

    entries = [
        (SimpleNamespace(candidate_id="cand1", question="问题一",
                         raw_question="原问题一"), "a.md"),
        (SimpleNamespace(candidate_id="cand2", question="问题二",
                         raw_question="原问题二"), "b.md"),
    ]

    # stub index_service.open_retriever 返回可编程 staged retriever
    staged = _FakeStagedRetriever()
    monkeypatch.setattr(
        svc["index_service"], "open_retriever", lambda info: staged,
    )
    before = {"summary": {"pass_rate": 1.0, "avg_result_score": 0.8},
              "cases": [{"case_id": "c1", "passed": True, "error": None}]}
    after = {"summary": {"pass_rate": 1.0, "avg_result_score": 0.8},
             "cases": [{"case_id": "c1", "passed": True, "error": None}]}

    monkeypatch.setattr(pipe, "_run_eval", lambda: after)

    detail = pipe._eval_after_and_probe(entries, SimpleNamespace(target="g1"), before)
    # a.md 命中 top-2（rank 1），b.md 需要 top-5（rank 3）→ 都放行
    assert detail["probe_failures"] == []
    assert not detail["blocked"]
    assert len(detail["probe_records"]) == 4  # 2 候选 × 2 标签
    by_cid = {r["candidate_id"]: r for r in detail["probe_records"]}
    assert by_cid["cand1"]["rank"] == 1
    assert by_cid["cand1"]["rerank_score"] == 0.91
    assert by_cid["cand1"]["hit_source_path"] == "evolved/a.md"


def test_probe_blocks_candidate_not_in_top_k(tmp_path, monkeypatch):
    """候选在 Top-5 内未命中 → probe_failures + blocked。"""
    from types import SimpleNamespace

    from app.evolution.pipeline import EvolutionPipeline

    from tests.unit.test_pipeline import make_services

    svc = make_services(tmp_path)
    pipe: EvolutionPipeline = svc["pipeline"]

    class _MissRetriever:
        def load(self):
            pass

        def search(self, query, top_k=5, timeout=None):
            return [_FakeHit("kb/other.md", 0.4)]  # 候选目标永远不命中

    monkeypatch.setattr(
        svc["index_service"], "open_retriever", lambda info: _MissRetriever(),
    )
    before = {"summary": {"pass_rate": 1.0, "avg_result_score": 0.8},
              "cases": [{"case_id": "c1", "passed": True, "error": None}]}
    after = {"summary": {"pass_rate": 1.0, "avg_result_score": 0.8},
             "cases": [{"case_id": "c1", "passed": True, "error": None}]}

    monkeypatch.setattr(pipe, "_run_eval", lambda: after)
    entries = [(SimpleNamespace(candidate_id="cand9", question="q",
                                raw_question="rq"), "new.md")]
    detail = pipe._eval_after_and_probe(entries, SimpleNamespace(target="g1"), before)
    assert detail["probe_failures"]
    assert "cand9" in detail["probe_failures"][0]
    assert detail["blocked"] is True
    assert all(r["rank"] == 0 and r["hit_source_path"] == ""
               for r in detail["probe_records"])