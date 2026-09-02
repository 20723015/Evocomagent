"""generation + index_service：版本化、验证失败不切换、两代清理、Windows 重试。"""

from __future__ import annotations

import json
import re

import pytest

from conftest import FakeEmbedder
from app.evolution.generation import GenerationInfo, GenerationStore, new_generation_id


# ============================================================
# GenerationStore
# ============================================================
def test_generation_roundtrip(tmp_path):
    store = GenerationStore(tmp_path / "kb_generations.json")
    info = GenerationInfo(generation_id="g1", target="/tmp/a.json",
                          embedding_model="m", previous_generation_id="g0")
    store.activate("numpy", info)
    assert store.active("numpy") == info
    assert store.active("Numpy") == info  # 大小写不敏感
    assert store.active("chroma") is None


def test_generation_read_missing_or_corrupt(tmp_path):
    store = GenerationStore(tmp_path / "nope.json")
    assert store.read() == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert GenerationStore(bad).read() == {}


def test_generation_id_format(frozen_clock):
    gid = new_generation_id(frozen_clock)
    assert re.match(r"^20260828\d{6}-[0-9a-f]{8}$", gid)


def test_activate_windows_permission_retry(tmp_path, monkeypatch):
    """Windows PermissionError 指数退避重试：前 2 次失败第 3 次成功。"""
    store = GenerationStore(tmp_path / "kb_generations.json")
    info = GenerationInfo(generation_id="g1", target="/t.json", embedding_model="m")
    real_replace = __import__("os").replace
    attempts = {"n": 0}

    def flaky_replace(src, dst):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise PermissionError("文件被占用")
        return real_replace(src, dst)

    monkeypatch.setattr("app.evolution.generation.os.replace", flaky_replace)
    store.activate("numpy", info)
    assert store.active("numpy").generation_id == "g1"
    assert attempts["n"] == 3


def test_activate_windows_permission_exhausted(tmp_path, monkeypatch):
    store = GenerationStore(tmp_path / "kb_generations.json")

    def always_fail(src, dst):
        raise PermissionError("一直占用")

    monkeypatch.setattr("app.evolution.generation.os.replace", always_fail)
    with pytest.raises(PermissionError):
        store.activate("numpy", GenerationInfo(
            generation_id="g1", target="/t.json", embedding_model="m"))


# ============================================================
# IndexBuildService：版本化构建 + 验证 + 切换 + 清理
# ============================================================
def _service(tmp_kb_dir, tmp_path):
    from app.evolution.index_service import IndexBuildService

    store = GenerationStore(tmp_path / "kb_generations.json")
    service = IndexBuildService(
        embedder=FakeEmbedder(),
        kb_dir=tmp_kb_dir,
        generation_store=store,
        backend_settings={"kb_index_path": str(tmp_path / "idx" / "kb_index.json")},
    )
    return store, service


def test_build_creates_versioned_index(tmp_kb_dir, tmp_path):
    store, service = _service(tmp_kb_dir, tmp_path)
    info = service.build("numpy")
    assert store.active("numpy") is None  # 未激活
    target = tmp_path / "idx" / f"kb_index.{info.generation_id}.json"
    assert target.exists()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["embedding_model"] == "fake-embedder"
    ids = [c["chunk_id"] for c in data["chunks"]]
    assert len(set(ids)) == len(ids)
    assert service.last_built_size == len(data["chunks"])


def test_build_verify_probe(tmp_kb_dir, tmp_path):
    """探针行为：用首个 chunk 的文本向量检索全新实例，top1 必须命中自身。"""
    import json as _json

    from app.agent.rag.backends import create_backend

    store, service = _service(tmp_kb_dir, tmp_path)
    info = service.build("numpy")
    target = tmp_path / "idx" / f"kb_index.{info.generation_id}.json"
    data = _json.loads(target.read_text(encoding="utf-8"))
    first = data["chunks"][0]

    probe_vec = service._embedder.encode_one(first["text"])
    impl = create_backend("numpy", index_path=target)
    impl.load()
    hits = impl.search(probe_vec, top_k=1)
    assert hits and hits[0].chunk.chunk_id == first["chunk_id"]
    assert store.active("numpy") is None  # 构建阶段不切换指针


def test_build_verify_failure_does_not_switch(tmp_kb_dir, tmp_path, monkeypatch):
    store, service = _service(tmp_kb_dir, tmp_path)
    monkeypatch.setattr(service, "_verify",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("验证失败")))
    with pytest.raises(RuntimeError):
        service.build("numpy")
    # 验证失败不切换：旧 generation 指针保持可用
    assert store.active("numpy") is None


def test_activate_keeps_two_generations(tmp_kb_dir, tmp_path):
    store, service = _service(tmp_kb_dir, tmp_path)
    g1 = service.build("numpy")
    service.activate("numpy", g1)
    active = store.active("numpy")
    assert active.generation_id == g1.generation_id

    g2 = service.build("numpy")
    service.activate("numpy", g2)
    assert store.active("numpy").generation_id == g2.generation_id
    assert store.active("numpy").previous_generation_id == g1.generation_id

    g3 = service.build("numpy")
    service.activate("numpy", g3)
    files = sorted((tmp_path / "idx").glob("kb_index.*.json"))
    gens = {f.name[len("kb_index."):-len(".json")] for f in files}
    assert gens == {g2.generation_id, g3.generation_id}  # 只留当前和上一代
    assert store.active("numpy").previous_generation_id == g2.generation_id


def test_build_empty_kb_raises(tmp_path):
    from app.evolution.index_service import IndexBuildService
    from app.evolution.generation import GenerationStore

    empty = tmp_path / "empty_kb"
    empty.mkdir()
    service = IndexBuildService(
        embedder=FakeEmbedder(), kb_dir=empty,
        generation_store=GenerationStore(tmp_path / "g.json"),
        backend_settings={"kb_index_path": str(tmp_path / "idx" / "kb_index.json")},
    )
    with pytest.raises(ValueError):
        service.build("numpy")