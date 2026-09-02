"""knowledge 单例：generation 感知、热刷新、last-known-good、override 栈。"""

from __future__ import annotations

import json

import pytest

from app.config.settings import settings


@pytest.fixture(autouse=True)
def _reset_retriever():
    from app.agent.tools import knowledge as knowledge_tool

    knowledge_tool.reset_retriever()
    yield
    knowledge_tool.reset_retriever()


def _write_numpy_index(path, texts, model="fake-embedder"):
    from app.agent.rag.chunker import Chunk

    from conftest import FakeEmbedder

    embedder = FakeEmbedder(model=model)
    chunks = [
        Chunk(chunk_id=f"c{i}", doc=f"doc{i}", section="s", text=t,
              source_path=f"d{i}.md")
        for i, t in enumerate(texts)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "embedding_model": model,
        "chunks": [c.to_dict() for c in chunks],
        "vectors": embedder.encode(texts),
    }, ensure_ascii=False), encoding="utf-8")


def _configure(tmp_path, reset_settings):
    settings.rag_backend = "numpy"
    settings.embedding_model = "fake-embedder"
    settings.kb_generation_path = str(tmp_path / "g.json")
    settings.kb_index_path = str(tmp_path / "fixed" / "kb_index.json")


def _activate(tmp_path, generation_id, target):
    from app.evolution.generation import GenerationInfo, GenerationStore

    store = GenerationStore(tmp_path / "g.json")
    store.activate("numpy", GenerationInfo(
        generation_id=generation_id, target=str(target),
        embedding_model="fake-embedder"))
    return store


def test_no_generation_falls_back_to_fixed_path(tmp_path, reset_settings):
    _configure(tmp_path, reset_settings)
    _write_numpy_index(tmp_path / "fixed" / "kb_index.json", ["固定索引内容"])
    from app.agent.tools import knowledge as knowledge_tool

    retriever = knowledge_tool._get_retriever()
    assert retriever.size == 1
    from app.evolution.generation import GenerationStore

    assert GenerationStore(tmp_path / "g.json").active("numpy") is None


def test_generation_pointer_used(tmp_path, reset_settings):
    _configure(tmp_path, reset_settings)
    gen_file = tmp_path / "gen" / "kb_index.g1.json"
    _write_numpy_index(gen_file, ["代际索引内容"])
    _activate(tmp_path, "g1", gen_file)
    from app.agent.tools import knowledge as knowledge_tool

    retriever = knowledge_tool._get_retriever()
    assert retriever.size == 1
    assert retriever.backend._index_path == gen_file


def test_unchanged_pointer_reuses_instance(tmp_path, reset_settings):
    _configure(tmp_path, reset_settings)
    gen_file = tmp_path / "gen" / "kb_index.g1.json"
    _write_numpy_index(gen_file, ["内容"])
    _activate(tmp_path, "g1", gen_file)
    from app.agent.tools import knowledge as knowledge_tool

    r1 = knowledge_tool._get_retriever()
    assert knowledge_tool._get_retriever() is r1  # 指针未变 → 复用


def test_hot_refresh_on_generation_change(tmp_path, reset_settings):
    _configure(tmp_path, reset_settings)
    from app.agent.tools import knowledge as knowledge_tool

    gen1 = tmp_path / "gen" / "kb_index.g1.json"
    _write_numpy_index(gen1, ["第一代"])
    _activate(tmp_path, "g1", gen1)
    r1 = knowledge_tool._get_retriever()

    gen2 = tmp_path / "gen" / "kb_index.g2.json"
    _write_numpy_index(gen2, ["第二代"])
    _activate(tmp_path, "g2", gen2)
    r2 = knowledge_tool._get_retriever()
    assert r2 is not r1
    assert r2.size == 1


def test_pointer_corrupt_keeps_last_known_good(tmp_path, reset_settings):
    _configure(tmp_path, reset_settings)
    from app.agent.tools import knowledge as knowledge_tool

    gen1 = tmp_path / "gen" / "kb_index.g1.json"
    _write_numpy_index(gen1, ["第一代"])
    _activate(tmp_path, "g1", gen1)
    r1 = knowledge_tool._get_retriever()

    settings.kb_generation_path = str(tmp_path / "missing.json")  # 指针消失
    assert knowledge_tool._get_retriever() is r1  # 读取异常 → 沿用内存实例


def test_override_stack_for_staging(tmp_path, reset_settings):
    _configure(tmp_path, reset_settings)
    from conftest import FakeBackend, FakeEmbedder, FakeRetriever
    from app.agent.tools import knowledge as knowledge_tool

    gen1 = tmp_path / "gen" / "kb_index.g1.json"
    _write_numpy_index(gen1, ["第一代"])
    _activate(tmp_path, "g1", gen1)
    r1 = knowledge_tool._get_retriever()

    staging = FakeRetriever(FakeEmbedder(), FakeBackend())
    knowledge_tool.push_retriever_override(staging)
    assert knowledge_tool._get_retriever() is staging  # staging 生效，跳过 generation 检查
    knowledge_tool.pop_retriever_override()
    assert knowledge_tool._get_retriever() is r1

    knowledge_tool.push_retriever_override(staging)
    knowledge_tool.push_retriever_override(staging)
    knowledge_tool.pop_retriever_override()
    knowledge_tool.pop_retriever_override()
    assert knowledge_tool._get_retriever() is r1  # 栈平衡后恢复


def test_reset_rebuilds(tmp_path, reset_settings):
    _configure(tmp_path, reset_settings)
    _write_numpy_index(tmp_path / "fixed" / "kb_index.json", ["内容"])
    from app.agent.tools import knowledge as knowledge_tool

    r1 = knowledge_tool._get_retriever()
    knowledge_tool.reset_retriever()
    r2 = knowledge_tool._get_retriever()
    assert r2 is not r1


def test_search_knowledge_returns_empty_when_all_hits_below_threshold(monkeypatch, reset_settings):
    from app.agent.rag.backends.base import RetrievedChunk
    from app.agent.rag.chunker import Chunk
    from app.agent.tools import knowledge as knowledge_tool

    class LowScoreRetriever:
        def search(self, query, top_k=3, timeout=None):
            return [RetrievedChunk(
                chunk=Chunk(
                    chunk_id="c1", doc="退换货政策", section="s", text="无关内容",
                    source_path="退换货政策.md",
                ),
                score=0.2,
            )]

    monkeypatch.setattr(knowledge_tool, "_get_retriever", lambda: LowScoreRetriever())
    settings.rag_min_relevance_score = 0.5
    result = knowledge_tool.search_knowledge("线下门店在哪里")
    assert result["success"] is True
    assert result["results"] == []
