"""build_kb_index 语义：手动构建切 generation + 运行中 Agent 热刷新。"""

from __future__ import annotations

import json

import pytest

from conftest import FakeEmbedder
from app.config.settings import settings
from app.evolution.generation import GenerationInfo, GenerationStore


@pytest.fixture(autouse=True)
def _reset_knowledge_singleton():
    """knowledge 模块全局单例：每个测试前后复位（测试顺序无关）。"""
    from app.agent.tools import knowledge as knowledge_tool

    knowledge_tool.reset_retriever()
    yield
    knowledge_tool.reset_retriever()


def _configure(tmp_path, reset_settings):
    settings.rag_backend = "numpy"
    settings.embedding_model = "fake-embedder"
    settings.kb_generation_path = str(tmp_path / "g.json")
    settings.kb_index_path = str(tmp_path / "fixed" / "kb_index.json")
    settings.evolve_state_dir = str(tmp_path / "state")


def _write_numpy_index(path, texts, model="fake-embedder"):
    """写一份 NumpyBackend 格式的索引 JSON。"""
    from app.agent.rag.chunker import Chunk

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
    return chunks


def test_manual_build_switches_generation_and_hot_refresh(tmp_path, tmp_kb_dir,
                                                          reset_settings):
    """手动 build（build_kb_index 语义）→ 指针切换 → 运行中 Agent 检索器热刷新。"""
    from app.agent.tools import knowledge as knowledge_tool
    from app.evolution.index_service import IndexBuildService

    _configure(tmp_path, reset_settings)

    # 先有固定索引：Agent 单例指向固定路径
    fixed = tmp_path / "fixed" / "kb_index.json"
    _write_numpy_index(fixed, ["固定知识内容"])
    knowledge_tool.reset_retriever()
    r_before = knowledge_tool._get_retriever()
    assert r_before.size == 1

    # 手动构建 + 激活（build_kb_index.py 的主体逻辑）
    store = GenerationStore(tmp_path / "g.json")
    service = IndexBuildService(
        embedder=FakeEmbedder(),
        kb_dir=tmp_kb_dir,
        generation_store=store,
        backend_settings={"kb_index_path": str(tmp_path / "fixed" / "kb_index.json")},
    )
    info = service.build("numpy")
    assert info.generation_id.startswith("2026")
    service.activate("numpy", info)
    assert store.active("numpy").generation_id == info.generation_id
    assert store.active("numpy").previous_generation_id == ""  # 首次切换

    # 运行中 Agent 热刷新：同一进程内下一次检索即指向新 generation
    r_after = knowledge_tool._get_retriever()
    assert r_after is not r_before
    assert r_after.size == _index_chunk_count(info)


def _index_chunk_count(info):
    import json as _json
    from pathlib import Path

    data = _json.loads(Path(info.target).read_text(encoding="utf-8"))
    return len(data["chunks"])


def test_activate_keeps_previous_generation_for_rollback(tmp_path, tmp_kb_dir,
                                                         reset_settings):
    from app.evolution.index_service import IndexBuildService

    _configure(tmp_path, reset_settings)
    store = GenerationStore(tmp_path / "g.json")
    service = IndexBuildService(
        embedder=FakeEmbedder(), kb_dir=tmp_kb_dir, generation_store=store,
        backend_settings={"kb_index_path": str(tmp_path / "fixed" / "kb_index.json")},
    )
    g1 = service.build("numpy")
    service.activate("numpy", g1)
    g2 = service.build("numpy")
    service.activate("numpy", g2)
    # 上一代索引文件保留（回滚/热刷新过渡期用）
    assert (tmp_path / "fixed" / f"kb_index.{g1.generation_id}.json").exists()
    assert store.active("numpy").previous_generation_id == g1.generation_id