"""真实 ES + 假 embedding / 假 reranker 的 RAG 协议集成测试（RAG 修复计划·4）。

用真实 Elasticsearch（compose，TEST_ES_URL）验证：
- ES 原生 hybrid（BM25+kNN+RRF）能召回；
- rerank=none 时 score_source=rrf，且正数阈值不清空结果；
- 挂上假 reranker（TEI 协议）时 score_source=rerank；
- reranker 断连时降级为 reranker_unavailable。

embedding 用确定性假实现（不依赖外部 API）；无 ES 时本地跳过，CI 提供 TEST_ES_URL。
"""

from __future__ import annotations

import hashlib
import os
import uuid

import pytest

pytestmark = pytest.mark.integration

TEST_ES_URL = os.environ.get("TEST_ES_URL", "")


def _es_or_skip():
    if not TEST_ES_URL:
        pytest.skip("TEST_ES_URL 未配置（需真实 ES）")
    try:
        from elasticsearch import Elasticsearch

        client = Elasticsearch(hosts=[TEST_ES_URL], request_timeout=10)
        client.info()
        return client
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("CI", "").lower() == "true":
            pytest.fail(f"真实 ES 不可用: {exc}")
        pytest.skip(f"真实 ES 不可用: {exc}")


class _FakeEmbedder:
    """确定性假 embedding（8 维）；不依赖任何外部服务。"""

    model = "fake-embed-8"

    def encode_one(self, text: str, timeout=None):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:8]]


def _chunks():
    from app.agent.rag.chunker import Chunk

    return [
        Chunk(chunk_id="c1", doc="退换货政策", section="七天无理由",
              text="七天无理由退货：商品不影响二次销售可退", source_path="p1.md",
              parent_id="p1#00"),
        Chunk(chunk_id="c2", doc="配送说明", section="包邮",
              text="偏远地区不包邮，需补运费", source_path="p2.md",
              parent_id="p2#00"),
        Chunk(chunk_id="c3", doc="会员权益", section="钻石会员",
              text="钻石会员享受专属客服与生日礼", source_path="p3.md",
              parent_id="p3#00"),
    ]


def _build_index(client, index: str):
    from app.agent.rag.backends.es_backend import ESBackend

    backend = ESBackend(client, index_name=index)
    chunks = _chunks()
    vectors = [_FakeEmbedder().encode_one(c.text) for c in chunks]
    backend.upsert(chunks, vectors, embedding_model=_FakeEmbedder.model)
    return backend


def _mock_rerank_client(status=200, payload=None, error=None):
    import httpx

    def handler(request):
        if error is not None:
            raise error
        return httpx.Response(status, json=payload or {})

    return httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)


def test_es_hybrid_protocol_with_real_es():
    from app.agent.rag.hybrid import ESHybridRetriever
    from app.agent.rag.rerank import HTTPReranker
    from app.agent.rag.retriever_factory import final_search

    client = _es_or_skip()
    index = f"ecom-kb-ragtest-{uuid.uuid4().hex[:8]}"
    try:
        backend = _build_index(client, index)
        client.indices.refresh(index=index)

        # 1) rerank=none：RRF 分 + 正数阈值不清空
        plain = ESHybridRetriever(
            embedder=_FakeEmbedder(), backend=backend, recall_k=5, reranker=None,
        )
        out = final_search(plain, "七天无理由退货", top_k=3, min_score=0.99)
        assert out.hits, "ES hybrid + rerank=none 被阈值清空（RRF 分无语义）"
        assert out.score_source == "rrf" and out.degraded is False

        # 2) 假 reranker（TEI 协议）→ score_source=rerank
        payload = [[0, 0.91], [1, 0.42], [2, 0.13]]
        reranker = HTTPReranker(
            "bge-reranker-v2-m3", endpoint_url="http://fake-reranker/rerank",
            client=_mock_rerank_client(payload=payload),
        )
        ranked = ESHybridRetriever(
            embedder=_FakeEmbedder(), backend=backend, recall_k=5, reranker=reranker,
        )
        out2 = final_search(ranked, "偏远地区包邮", top_k=3)
        assert out2.score_source == "rerank" and out2.degraded is False
        assert out2.hits and out2.hits[0].score > 0

        # 3) reranker 断连 → 显式降级（不得当正常精排）
        import httpx

        broken = HTTPReranker(
            "bge-reranker-v2-m3", endpoint_url="http://fake-reranker/rerank",
            client=_mock_rerank_client(error=httpx.ConnectError("down")),
        )
        degraded = ESHybridRetriever(
            embedder=_FakeEmbedder(), backend=backend, recall_k=5, reranker=broken,
        ).search_with_status("钻石会员权益", top_k=3)
        assert degraded.degraded is True
        assert degraded.degraded_reason == "reranker_unavailable"
        assert degraded.score_source == "rrf"
    finally:
        try:
            client.options(ignore_status=[404]).indices.delete(index=index)
        except Exception:  # noqa: BLE001
            pass
