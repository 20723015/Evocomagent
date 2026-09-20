"""知识库检索器：query → 向量化 → 委托后端检索。

设计上 retriever 只负责"问句怎么变向量""结果怎么聚合"，
存储和打分都交给 VectorBackend 实现，对应两套：
- NumpyBackend：手写余弦 + JSON 持久化（教学透明，零依赖）
- ChromaBackend：嵌入式向量数据库 + HNSW（生产代表性）

校验逻辑：加载后比对 backend 持久化的 embedding_model 与当前 Embedder.model，
不一致直接报错——避免"换了 embedding 但还在用老索引"这种隐蔽问题。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.agent.rag.backends.base import RetrievedChunk, VectorBackend
from app.agent.rag.embedder import Embedder

__all__ = ["KnowledgeRetriever", "RetrievedChunk", "collapse_by_parent",
           "filter_hits_by_score", "RetrievalResult", "SCORE_SOURCE_RRF",
           "SCORE_SOURCE_VECTOR", "SCORE_SOURCE_RERANK",
           "DEGRADED_RERANKER_UNAVAILABLE"]

SCORE_SOURCE_RRF = "rrf"
SCORE_SOURCE_VECTOR = "vector"
SCORE_SOURCE_RERANK = "rerank"
DEGRADED_RERANKER_UNAVAILABLE = "reranker_unavailable"


@dataclass
class RetrievalResult:
    """一次检索的显式结果状态（RAG 修复计划·1）。

    score_source 决定分数是否有绝对语义：
    - vector / rerank：分数可比阈值；
    - rrf：秩融合分无语义（不得做阈值门控）。
    degraded=True 时下游必须按降级处理（生产默认 fail-closed）。
    """

    hits: list = field(default_factory=list)
    score_source: str = SCORE_SOURCE_VECTOR
    degraded: bool = False
    degraded_reason: str = ""

    @property
    def scores_meaningful(self) -> bool:
        return (
            self.score_source in (SCORE_SOURCE_VECTOR, SCORE_SOURCE_RERANK)
            and not self.degraded
        )


def filter_hits_by_score(
    hits: list[RetrievedChunk], min_score: float | None,
) -> list[RetrievedChunk]:
    """按最终相关度过滤候选；未配置阈值时保持原结果。

    阈值属于完整检索配置（embedding/backend/hybrid/reranker），不能跨配置复用。
    等于阈值的候选视为可接受，便于配置与评测使用同一边界语义。
    """
    if min_score is None:
        return list(hits)
    return [hit for hit in hits if hit.score >= min_score]


def collapse_by_parent(
    hits: list[RetrievedChunk], top_k: int,
) -> list[RetrievedChunk]:
    """父子块去重：同一 parent_id 只保留排序最高的首见命中，截取 top_k。

    召回与精排都发生在子块粒度；这里在最终输出前去重，避免同一章节的
    多个子块占满 Top-K。旧索引的 chunk 无 parent_id（空串）→ 视为各自
    独立，行为与旧版本一致。
    """
    out: list[RetrievedChunk] = []
    seen: set[str] = set()
    for hit in hits:
        pid = hit.chunk.parent_id
        if pid:
            if pid in seen:
                continue
            seen.add(pid)
        out.append(hit)
        if len(out) >= top_k:
            break
    return out


class KnowledgeRetriever:
    """对上层暴露统一接口，对下委托给具体 backend。"""

    def __init__(self, embedder: Embedder, backend: VectorBackend):
        self._embedder = embedder
        self._backend = backend
        self._loaded = False

    @property
    def backend(self) -> VectorBackend:
        return self._backend

    @property
    def size(self) -> int:
        return self._backend.size()

    def load(self) -> None:
        if self._loaded:
            return
        self._backend.load()

        expected = self._backend.expected_embedding_model()
        if expected and expected != self._embedder.model:
            raise ValueError(
                f"索引模型({expected}) 与当前 Embedder 模型"
                f"({self._embedder.model}) 不一致，请重建索引。"
            )
        self._loaded = True

    def search(self, query: str, top_k: int = 3,
               timeout: Optional[float] = None) -> list[RetrievedChunk]:
        return self.search_with_status(query, top_k, timeout=timeout).hits

    def search_with_status(self, query: str, top_k: int = 3,
                           timeout: Optional[float] = None) -> RetrievalResult:
        """纯向量检索：score_source=vector，分数有绝对语义。"""
        if not self._loaded:
            self.load()
        q_vec = self._embedder.encode_one(query, timeout=timeout)
        hits = self._backend.search(q_vec, top_k=top_k, timeout=timeout)
        return RetrievalResult(hits=hits, score_source=SCORE_SOURCE_VECTOR)
