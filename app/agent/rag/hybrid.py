"""混合检索（7.2）：向量路 + BM25 路，RRF（Reciprocal Rank Fusion）融合。

动机：向量检索擅长"语义相似"，但对精确术语（SKU 编号、法条原文、专有名词）
经常漏召回——"七天无理由退货"这类词面强匹配的需求恰恰是 BM25 的主场。
两路互补，本模块就是 7.2「BM25 补向量对精确术语盲区」的落地。

算法：RRF（Reciprocal Rank Fusion），ES 混合检索的同款算法：
    score(d) = Σ 1/(k + rank_i(d))，k = 60
只依赖每路返回的**排名**，不依赖各路的分数尺度——余弦相似度与 BM25
量纲不同，直接加权相加没有意义，按排名融合天然免疫这一点。

工程边界：本类对 vector_retriever 只要求 KnowledgeRetriever 的接口
（search / load / size），对 bm25 只要求 BM25Index。后续换 Elasticsearch，
只需把 vector_retriever 换成 ES 向量检索器、把 bm25 换成 ES 的 BM25/混合查询，
上层（knowledge 单例、Agent）调用方式不变——换后端不动上层。
"""

from __future__ import annotations

from app.agent.rag.backends.base import RetrievedChunk
from app.agent.rag.bm25 import BM25Index
from app.agent.rag.chunker import Chunk
from app.agent.rag.query_normalizer import QueryNormalizer, normalize_query
from app.agent.rag.rerank import RerankerUnavailable
from app.agent.rag.retriever import (
    DEGRADED_RERANKER_UNAVAILABLE,
    SCORE_SOURCE_RERANK,
    SCORE_SOURCE_RRF,
    KnowledgeRetriever,
    RetrievalResult,
)


def _normalized(query: str, normalizer: QueryNormalizer | None) -> str:
    """工作流 A：检索入口做一次表层规范化，改写后的 query 同时喂
    hybrid/BM25/embedder 与 reranker（改写只发生在字符串层，语义不变）。"""
    return normalize_query(query, normalizer)


class ESHybridRetriever:
    """ES 原生混合检索装配（阶段八）：一条查询出 BM25+kNN 并做 RRF。

    hybrid.py 的 Python 侧融合只在 numpy/chroma 路径使用；rag_backend=es
    时经本类走 ESBackend.hybrid_search（rank.rrf 查询级融合，单次网络往返），
    BM25Index 与 Python RRF 在该路径下退役。接口与 KnowledgeRetriever 对齐
    （search/search_with_status/load/size），上层调用方式不变。
    """

    def __init__(self, embedder, backend, recall_k: int = 30, reranker=None,
                 normalizer: QueryNormalizer | None = None):
        self._embedder = embedder
        self._backend = backend
        self._recall_k = recall_k
        self._reranker = reranker
        self._normalizer = normalizer

    @property
    def scores_meaningful(self) -> bool:
        """兼容旧调用：仅当挂了精排器（精排分覆盖 RRF 分）时才为 True。"""
        return self._reranker is not None

    def load(self) -> None:
        self._backend.load()

    @property
    def size(self) -> int:
        return self._backend.size()

    @property
    def backend(self) -> object:
        return self._backend

    def search(self, query: str, top_k: int = 3,
               timeout: float | None = None) -> list[RetrievedChunk]:
        return self.search_with_status(query, top_k, timeout=timeout).hits

    def search_with_status(self, query: str, top_k: int = 3,
                           timeout: float | None = None) -> RetrievalResult:
        query = _normalized(query, self._normalizer)
        hits = self._backend.hybrid_search(
            query_text=query,
            query_vector=self._embedder.encode_one(query, timeout=timeout),
            top_k=top_k,
            recall_k=self._recall_k,
            timeout=timeout,
        )
        if self._reranker is None:
            # RRF 秩融合分无语义：score_source=rrf，调用方不得做阈值门控
            return RetrievalResult(
                hits=hits[:top_k], score_source=SCORE_SOURCE_RRF,
            )
        try:
            reranked = self._reranker.rerank(query, hits, top_k, timeout=timeout)
        except RerankerUnavailable as e:
            # 精排不可用/部分响应：显式降级（不得用 RRF 0 分做阈值判断）
            _record_reranker_unavailable(e)
            return RetrievalResult(
                hits=hits[:top_k],
                score_source=SCORE_SOURCE_RRF,
                degraded=True,
                degraded_reason=getattr(e, "reason", DEGRADED_RERANKER_UNAVAILABLE),
            )
        return RetrievalResult(hits=reranked, score_source=SCORE_SOURCE_RERANK)


class HybridRetriever:
    """向量 + BM25 双路召回，RRF 融合后输出 Top-K；可选精排器（7.3 扩展点）。"""

    RRF_K = 60  # RRF 常数 k：控制排名对分数的衰减速度

    def __init__(
        self,
        vector_retriever: KnowledgeRetriever,
        bm25: BM25Index,
        recall_k: int = 30,
        reranker=None,
        normalizer: QueryNormalizer | None = None,
    ):
        self._vector_retriever = vector_retriever
        self._bm25 = bm25
        self._recall_k = recall_k
        self._reranker = reranker
        self._normalizer = normalizer

    def load(self) -> None:
        """委托向量路加载索引（接口与 KnowledgeRetriever 对齐，供知识单例复用）。"""
        self._vector_retriever.load()

    @property
    def size(self) -> int:
        """已索引 chunk 总数（与向量路一致）。"""
        return self._vector_retriever.size

    @property
    def scores_meaningful(self) -> bool:
        return self._reranker is not None

    def search(self, query: str, top_k: int = 3,
               timeout: float | None = None) -> list[RetrievedChunk]:
        """双路召回 → 按 chunk_id 去重融合 → 可选精排 → 返回 Top-K。"""
        return self.search_with_status(query, top_k, timeout=timeout).hits

    def search_with_status(self, query: str, top_k: int = 3,
                           timeout: float | None = None) -> RetrievalResult:
        query = _normalized(query, self._normalizer)
        vector_hits = self._vector_retriever.search(query, self._recall_k, timeout=timeout)
        bm25_hits = self._bm25.search(query, self._recall_k)
        candidates = self._rrf_fuse([vector_hits, bm25_hits], limit=self._recall_k)
        if self._reranker is None:
            return RetrievalResult(hits=candidates[:top_k], score_source=SCORE_SOURCE_RRF)
        try:
            reranked = self._reranker.rerank(query, candidates, top_k, timeout=timeout)
        except RerankerUnavailable as e:
            _record_reranker_unavailable(e)
            return RetrievalResult(
                hits=candidates[:top_k],
                score_source=SCORE_SOURCE_RRF,
                degraded=True,
                degraded_reason=getattr(e, "reason", DEGRADED_RERANKER_UNAVAILABLE),
            )
        return RetrievalResult(hits=reranked, score_source=SCORE_SOURCE_RERANK)

    @staticmethod
    def _rrf_fuse(
        ranked_lists: list[list[RetrievedChunk]],
        limit: int,
    ) -> list[RetrievedChunk]:
        """把多路 ranked list 融合成一个列表：score = Σ 1/(RRF_K + rank)。"""
        scores: dict[str, float] = {}
        chunk_by_id: dict[str, Chunk] = {}
        for hits in ranked_lists:
            for rank, hit in enumerate(hits, start=1):
                cid = hit.chunk.chunk_id
                chunk_by_id[cid] = hit.chunk
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (HybridRetriever.RRF_K + rank)
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [
            RetrievedChunk(chunk=chunk_by_id[cid], score=score)
            for cid, score in ordered[:limit]
        ]

def _record_reranker_unavailable(exc) -> None:
    try:
        from app.observability.metrics import record_reranker_unavailable

        record_reranker_unavailable(
            getattr(exc, "reason", "reranker_unavailable")
        )
    except Exception:  # noqa: BLE001
        pass
