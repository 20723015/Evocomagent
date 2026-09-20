"""批次7 单测：指纹纳入解析/分块版本，health 明确提示重建。

背景：`config_fingerprint()` 原先只覆盖 embedding/hybrid/rerank，改完解析或
分块逻辑后索引会静默停留在旧切分上、没有任何信号。纳入
``PARSER_CHUNKER_VERSION`` 后，升级会经 health 明确报「请重建索引」（预期信号）。
"""

from __future__ import annotations

import pytest

from app.agent.rag import fingerprint


# ============================================================
# 1. 版本进入指纹
# ============================================================
def test_version_constant_is_non_empty():
    assert isinstance(fingerprint.PARSER_CHUNKER_VERSION, str)
    assert fingerprint.PARSER_CHUNKER_VERSION.strip()


def test_fingerprint_is_stable(reset_settings):
    assert fingerprint.config_fingerprint() == fingerprint.config_fingerprint()


def test_version_change_changes_fingerprint(monkeypatch, reset_settings):
    before = fingerprint.config_fingerprint()

    monkeypatch.setattr(fingerprint, "PARSER_CHUNKER_VERSION", "bumped-for-test")

    after = fingerprint.config_fingerprint()
    assert after != before  # 版本确实参与哈希，而非摆设


def test_other_config_still_affects_fingerprint(monkeypatch, reset_settings):
    from app.config.settings import settings

    before = fingerprint.config_fingerprint()
    monkeypatch.setattr(settings, "rag_hybrid", not settings.rag_hybrid)
    assert fingerprint.config_fingerprint() != before


# ============================================================
# 2. health 指纹不一致时的文案
# ============================================================
def test_fingerprint_mismatch_message_mentions_parser_version(
    monkeypatch, reset_settings,
):
    from app.agent.rag import es_util, health

    class _Indices:
        def get_mapping(self, index=None):
            from app.config.settings import settings as _s

            return {index: {"mappings": {"_meta": {
                "embedding_model": _s.embedding_model,
                "embedding_provider": (_s.embedding_provider or "openai").lower(),
                "embedding_dimensions": fingerprint.embedding_dimensions(),
                # 与运行指纹不一致（模拟旧索引 / 解析版本变更前构建的索引）
                "config_fingerprint": "0" * 16,
            }}}}

    class _StubES:
        indices = _Indices()

    class _Gen:
        target = "ecom-kb-20260914000000-abcdef12"

    monkeypatch.setattr(health, "_active_generation", lambda: _Gen())
    monkeypatch.setattr(es_util, "get_es_client", lambda: _StubES())

    status, reason = health._check_embedding_model()

    assert status == "unavailable"
    assert "解析/分块版本" in reason
    assert "请重建索引" in reason


def test_matching_fingerprint_passes(monkeypatch, reset_settings):
    """对照：指纹一致时不报 unavailable（确认上面的失败来自指纹分支）。"""
    from app.agent.rag import es_util, health

    class _Indices:
        def get_mapping(self, index=None):
            from app.config.settings import settings as _s

            return {index: {"mappings": {"_meta": {
                "embedding_model": _s.embedding_model,
                "embedding_provider": (_s.embedding_provider or "openai").lower(),
                "embedding_dimensions": fingerprint.embedding_dimensions(),
                "config_fingerprint": fingerprint.config_fingerprint(),
            }}}}

    class _StubES:
        indices = _Indices()

    class _Gen:
        target = "ecom-kb-20260914000000-abcdef12"

    monkeypatch.setattr(health, "_active_generation", lambda: _Gen())
    monkeypatch.setattr(es_util, "get_es_client", lambda: _StubES())

    status, reason = health._check_embedding_model()

    assert status == "ok", reason


def test_missing_fingerprint_asks_rebuild(monkeypatch, reset_settings):
    from app.agent.rag import es_util, health

    class _Indices:
        def get_mapping(self, index=None):
            from app.config.settings import settings as _s

            return {index: {"mappings": {"_meta": {
                "embedding_model": _s.embedding_model,
                "embedding_provider": (_s.embedding_provider or "openai").lower(),
            }}}}

    class _StubES:
        indices = _Indices()

    class _Gen:
        target = "ecom-kb-20260914000000-abcdef12"

    monkeypatch.setattr(health, "_active_generation", lambda: _Gen())
    monkeypatch.setattr(es_util, "get_es_client", lambda: _StubES())

    status, reason = health._check_embedding_model()

    assert status == "unavailable"
    assert "config_fingerprint" in reason
