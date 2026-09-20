"""修复计划·三：依赖状态、能力分级与可恢复 ES provider 测试。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config.settings import settings


def _client(monkeypatch, comps=None):
    import app.server.main as main_mod
    from test_server_api import _FakeComponents, _ScriptedAgent

    comps = comps if comps is not None else _FakeComponents()
    monkeypatch.setattr(main_mod, "build_pod_components", lambda: comps)
    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda uid, sid="", components=None, **_k: _ScriptedAgent(uid, sid),
    )
    return TestClient(main_mod.create_app())


def test_readyz_chat_dependency_unavailable_is_503_not_ready(monkeypatch, reset_settings):
    """聊天核心依赖不可用（redis_required 且 Redis 缺失）→ 503 not_ready。"""
    settings.redis_required = True
    with _client(monkeypatch) as client:
        resp = client.get("/readyz")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["capabilities"]["chat"] == "unavailable"


def test_readyz_non_chat_unavailable_is_200_degraded(monkeypatch, reset_settings):
    """仅非聊天能力不可用（turn_archive S3 缺失）→ 200 degraded，聊天仍可用。"""
    settings.redis_required = False
    settings.turns_archive_backend = "s3"
    import app.server.main as main_mod
    from test_server_api import _FakeComponents

    monkeypatch.setattr(main_mod, "build_pod_components", lambda: _FakeComponents())
    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda uid, sid="", components=None, **_k: __import__("test_server_api")._ScriptedAgent(uid, sid),
    )
    with TestClient(main_mod.create_app()) as client:
        resp = client.get("/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["capabilities"]["chat"] in ("ok", "not_configured")
    assert body["capabilities"]["turn_archive"] == "unavailable"


# ------------------------------------------------------------
# ES provider 可恢复
# ------------------------------------------------------------
class _StubIndices:
    def get_alias(self, name=None):
        return {f"{name}-000001": {}}

    def get_mapping(self, index=None):
        # 与运行时配置一致的完整 _meta（provider/model/dimensions/指纹）
        from app.agent.rag.fingerprint import (
            config_fingerprint,
            embedding_dimensions,
        )
        from app.config.settings import settings as _s

        return {index: {"mappings": {"_meta": {
            "embedding_model": _s.embedding_model,
            "embedding_provider": (_s.embedding_provider or "openai").lower(),
            "embedding_dimensions": embedding_dimensions(),
            "config_fingerprint": config_fingerprint(),
        }}}}


class _StubES:
    def __init__(self):
        self.indices = _StubIndices()

    def info(self):
        return {"version": {"number": "8.0"}}


def test_es_provider_does_not_cache_failure_permanently(monkeypatch, reset_settings):
    from app.agent.rag import es_util

    es_util.reset_es_for_test()
    settings.es_url = "http://localhost:1"
    monkeypatch.setattr(es_util, "RECONNECT_COOLDOWN_SECONDS", 0.0)
    calls = {"n": 0}

    def _build(timeout=5.0):
        calls["n"] += 1
        return None

    monkeypatch.setattr(es_util, "build_es_client", _build)
    assert es_util.get_es_client() is None
    assert es_util.get_es_client() is None
    assert calls["n"] >= 2  # 冷却为 0 → 每次都会重试（不再永久缓存失败）


def test_es_provider_invalidate_allows_reinit(monkeypatch, reset_settings):
    from app.agent.rag import es_util

    es_util.reset_es_for_test()
    settings.es_url = "http://localhost:2"
    monkeypatch.setattr(es_util, "RECONNECT_COOLDOWN_SECONDS", 0.0)
    stub = _StubES()
    monkeypatch.setattr(es_util, "build_es_client", lambda timeout=5.0: stub)
    assert es_util.get_es_client() is stub

    es_util.invalidate_es_client("boom")
    # 冷却为 0 → 下次请求可重新初始化
    assert es_util.get_es_client() is stub


def test_es_status_not_configured_without_url(reset_settings):
    from app.agent.rag import es_util

    es_util.reset_es_for_test()
    settings.es_url = ""
    assert es_util.es_dependency_state() == "not_configured"


def test_readyz_es_backend_outage_is_503_then_recovers(monkeypatch, reset_settings):
    """rag_backend=es 时 ES 不可用 → chat 不可用（503）；恢复后无需重启变 ready。"""
    from app.agent.rag import es_util

    es_util.reset_es_for_test()
    settings.rag_backend = "es"
    settings.es_url = "http://localhost:1"
    monkeypatch.setattr(es_util, "RECONNECT_COOLDOWN_SECONDS", 0.0)
    monkeypatch.setattr(es_util, "build_es_client", lambda timeout=5.0: None)

    import app.server.main as main_mod
    from test_server_api import _FakeComponents
    from app.agent.rag import health as rag_health

    # 恢复阶段：需要有活动 generation 才能判定 RAG ready
    class _Gen:
        target = "ecom-kb-stub"

    monkeypatch.setattr(rag_health, "_active_generation", lambda: _Gen())

    comps = _FakeComponents()
    monkeypatch.setattr(main_mod, "build_pod_components", lambda: comps)
    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda uid, sid="", components=None, **_k: __import__("test_server_api")._ScriptedAgent(uid, sid),
    )
    with TestClient(main_mod.create_app()) as client:
        down = client.get("/readyz")
        assert down.status_code == 503
        assert down.json()["capabilities"]["chat"] == "unavailable"

        # ES 恢复（provider 冷却为 0，无需重启）：provider 与组件均恢复到可用
        stub = _StubES()
        es_util.set_es_for_test(stub)
        comps.es_provider = lambda: stub
        up = client.get("/readyz")
    assert up.status_code == 200
    assert up.json()["capabilities"]["chat"] in ("ok", "not_configured")


# ------------------------------------------------------------
# S3 上传不得退回本地
# ------------------------------------------------------------
def test_kb_upload_s3_without_object_store_returns_none(monkeypatch, reset_settings):
    from sqlalchemy import create_engine

    import app.server.deps as deps
    from app.stores.sql.schema import metadata

    settings.kb_upload_enabled = True
    settings.kb_upload_storage = "s3"
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    metadata.create_all(engine)

    service = deps._build_upload_service(engine, redis=None, object_store=None, job_store=None)
    assert service is None  # 不退回本地文件（端点 503）
