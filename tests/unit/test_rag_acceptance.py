"""RAG 第六轮验收单测：release profile / RRF 禁校准 / degraded 失败 /
reranker 探活协议 / 索引 _meta 指纹。"""

from __future__ import annotations

from app.agent.rag import health
from app.config.settings import settings


# ------------------------------------------------------------
# release profile：门槛只升不降（禁止调为 0）
# ------------------------------------------------------------
class _Args:
    def __init__(self, **kw):
        self.min_positive_recall = 0.0
        self.min_easy_recall = 0.0
        self.min_hard_recall = 0.0
        self.min_mrr = 0.0
        self.min_ndcg = 0.0
        self.min_negative_rejection = 0.0
        self.max_p95_latency_ms = 0.0
        for k, v in kw.items():
            setattr(self, k, v)


def test_release_profile_raises_floors_and_cannot_zero():
    from app.scripts.run_retrieval_eval import apply_release_profile

    args = _Args()
    apply_release_profile(args)
    assert args.min_positive_recall == 0.95
    assert args.min_easy_recall == 0.98
    assert args.min_hard_recall == 0.80
    assert args.min_mrr == 0.90
    assert args.min_ndcg == 0.90
    assert args.min_negative_rejection == 0.90
    assert args.max_p95_latency_ms == 500.0  # 0 被抬到固定上限


def test_release_profile_does_not_loosen_existing_stricter_values():
    from app.scripts.run_retrieval_eval import apply_release_profile

    args = _Args(min_positive_recall=0.99, max_p95_latency_ms=100.0)
    apply_release_profile(args)
    assert args.min_positive_recall == 0.99  # 更严格保留
    assert args.max_p95_latency_ms == 100.0  # 更严格（更小）保留


# ------------------------------------------------------------
# RRF 模式判定 + degraded 失败
# ------------------------------------------------------------
def test_rrf_mode_detection(reset_settings):
    from app.scripts.run_retrieval_eval import rrf_mode

    settings.rag_hybrid = True
    settings.rag_rerank = "none"
    assert rrf_mode() is True
    settings.rag_rerank = "bge-reranker-v2-m3"
    assert rrf_mode() is False
    settings.rag_hybrid = False
    settings.rag_rerank = "none"
    assert rrf_mode() is False


def test_quality_failures_flags_degraded_cases():
    from app.scripts.run_retrieval_eval import quality_failures

    report = {
        "summary": {
            "positive": {"recall_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0},
            "easy": {"cases": 1, "recall_at_k": 1.0},
            "hard": {"cases": 1, "recall_at_k": 1.0},
            "negative": {"cases": 0, "rejection_rate": 1.0},
            "p95_latency_ms": 10.0,
            "degraded_cases": 2,
        },
    }
    args = _Args(min_score=0.1)
    failures = quality_failures(report, args)
    assert any("degraded" in f for f in failures)


def test_quality_failures_flags_p95_latency_over_budget():
    from app.scripts.run_retrieval_eval import quality_failures

    report = {
        "summary": {
            "positive": {"recall_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0},
            "easy": {"cases": 0, "recall_at_k": None},
            "hard": {"cases": 0, "recall_at_k": None},
            "negative": {"cases": 0, "rejection_rate": 1.0},
            "p95_latency_ms": 800.0,
            "degraded_cases": 0,
        },
    }
    args = _Args(min_score=0.1)
    failures = quality_failures(report, args)
    assert any("P95" in f for f in failures)


# ------------------------------------------------------------
# reranker 探活协议（2xx + index=0 有限分）
# ------------------------------------------------------------
def test_probe_pairs_parsing_variants():
    p = health._parse_probe_pairs
    assert p({"results": [{"index": 0, "relevance_score": 0.5}]}) == [(0, 0.5)]
    assert p([[0, 0.3]]) == [(0, 0.3)]
    assert p({"nope": 1}) == []
    assert p("bad") == []
    assert p({"results": [{"index": 0}]}) == []  # 缺分数


def test_check_reranker_rejects_non_2xx(monkeypatch, reset_settings):
    import httpx

    settings.rag_rerank = "bge-reranker-v2-m3"
    settings.rerank_endpoint_url = "http://reranker.invalid/rerank"
    health._RERANKER_CACHE.update(at=0.0, state="", err="")

    def _fake_client(*a, **k):
        class _C:
            def post(self, url, json=None):
                return httpx.Response(401, json={"error": "unauthorized"})

            def close(self):
                pass

        return _C()

    monkeypatch.setattr(httpx, "Client", _fake_client)
    state, err = health._check_reranker()
    assert state == "unavailable" and "401" in err


def test_check_reranker_accepts_valid_and_caches(monkeypatch, reset_settings):
    import httpx

    settings.rag_rerank = "bge-reranker-v2-m3"
    settings.rerank_endpoint_url = "http://reranker.invalid/rerank"
    health._RERANKER_CACHE.update(at=0.0, state="", err="")
    calls = {"n": 0}

    def _fake_client(*a, **k):
        class _C:
            def post(self, url, json=None):
                calls["n"] += 1
                return httpx.Response(200, json=[[0, 0.42]])

            def close(self):
                pass

        return _C()

    monkeypatch.setattr(httpx, "Client", _fake_client)
    assert health._check_reranker() == ("ok", "")
    assert health._check_reranker() == ("ok", "")  # 命中 TTL 缓存
    assert calls["n"] == 1


def test_check_reranker_rejects_malformed_payload(monkeypatch, reset_settings):
    import httpx

    settings.rag_rerank = "bge-reranker-v2-m3"
    settings.rerank_endpoint_url = "http://reranker.invalid/rerank"
    health._RERANKER_CACHE.update(at=0.0, state="", err="")

    def _fake_client(*a, **k):
        class _C:
            def post(self, url, json=None):
                return httpx.Response(200, json={"results": [{"index": 1, "score": 0.5}]})

            def close(self):
                pass

        return _C()

    monkeypatch.setattr(httpx, "Client", _fake_client)
    state, err = health._check_reranker()
    assert state == "unavailable" and "index=0" in err


# ------------------------------------------------------------
# 索引 _meta 指纹
# ------------------------------------------------------------
def test_es_mapping_meta_contains_fingerprint_fields():
    from app.agent.rag.backends.es_backend import ESBackend

    mapping = ESBackend._mapping(
        "bge-m3", 1024, embedding_provider="sophnet",
        dimensions=1024, config_fingerprint="abc123",
    )
    meta = mapping["_meta"]
    assert meta["embedding_model"] == "bge-m3"
    assert meta["embedding_provider"] == "sophnet"
    assert meta["embedding_dimensions"] == 1024
    assert meta["config_fingerprint"] == "abc123"


def test_production_requirements_enforced(monkeypatch, reset_settings):
    monkeypatch.setenv("RAG_PROD_STRICT", "true")
    settings.rag_rerank_fail_closed = False
    settings.rag_doc_metadata_required = False
    settings.rag_min_relevance_score = None
    problems = health._check_production_requirements()
    assert any("RAG_RERANK_FAIL_CLOSED" in p for p in problems)
    assert any("RAG_DOC_METADATA_REQUIRED" in p for p in problems)
    assert any("RAG_MIN_RELEVANCE_SCORE" in p for p in problems)

    settings.rag_rerank_fail_closed = True
    settings.rag_doc_metadata_required = True
    settings.rag_min_relevance_score = 0.26
    assert health._check_production_requirements() == []
