"""semantic_dedup.py：人工链路专用语义去重服务（人工知识链路修正）。

与线上检索完全解耦：
- GenerationStore.active(backend) → index_service.open_retriever(
    info, RetrievalConfig(backend=..., hybrid=False, rerank="none"))
  —— 纯向量通道，不经过线上 hybrid/RRF/reranker，也不依赖知识工具单例；
- 双侧（问题/答案）各取 top-1，分数归一化 ``min(max(cos, 0), 1)``
  （截断而非仿射：0.90 阈值语义即「余弦 0.90」，正交 → 0 而非 0.5）；
- 阈值与 evolve_dedup_threshold 彻底分离（human_dedup_question/answer_threshold）。

本服务同时注入评审器（RagScorer）与发布 Worker（批内 pairwise + 最终去重）
——这是修复「生产去重不执行」的落点：发布路径构造期强制注入，不可能再被
默认参数静默跳过。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config.settings import settings
from app.observability.logging import get_logger

log = get_logger("app.evolution.semantic_dedup")


def normalize_cosine(score: float) -> float:
    """原始余弦 → [0,1]（截断，不仿射）。"""
    return min(max(float(score), 0.0), 1.0)


def backend_score_to_cosine(score: float, backend: str) -> float:
    """把后端检索分数还原为统一的原始余弦语义。

    numpy/chroma 已返回 cosine；Elasticsearch dense_vector cosine kNN 的
    ``_score`` 为 ``(1 + cosine) / 2``，必须先做逆变换再参与 0.90 阈值。
    """
    raw = float(score)
    cosine = (2.0 * raw - 1.0) if str(backend).lower() == "es" else raw
    return normalize_cosine(cosine)


@dataclass
class SideHit:
    """单侧 top-1 命中：score 已归一化 [0,1]；text 供更新判定（不落库）。"""

    path: str
    score: float
    text: str = ""


@dataclass
class DedupResult:
    """双侧去重结果：任一侧无命中为 None。"""

    question: SideHit | None = None
    answer: SideHit | None = None


class SemanticDedupService:
    """人工链路语义去重（评审 + 发布共用同一实例与同一阈值口径）。"""

    def __init__(
        self,
        index_service,
        generation_store,
        embedder,
        *,
        q_threshold: float | None = None,
        a_threshold: float | None = None,
    ):
        self._index = index_service
        self._gen_store = generation_store
        self._embedder = embedder
        self._q_threshold = (
            float(q_threshold)
            if q_threshold is not None
            else float(settings.human_dedup_question_threshold)
        )
        self._a_threshold = (
            float(a_threshold)
            if a_threshold is not None
            else float(settings.human_dedup_answer_threshold)
        )

    @property
    def q_threshold(self) -> float:
        return self._q_threshold

    @property
    def a_threshold(self) -> float:
        return self._a_threshold

    @property
    def embedder_model(self) -> str:
        return getattr(self._embedder, "model", "")

    # ---------- 检索通道 ----------
    def _open_retriever(self):
        """活动代 + 纯向量检索器；无活动代（空库）→ None（无任何命中）。"""
        backend = settings.rag_backend.lower()
        info = self._gen_store.active(backend)
        if info is None:
            return None
        from app.agent.rag.retriever_factory import RetrievalConfig

        return self._index.open_retriever(
            info,
            RetrievalConfig(backend=backend, hybrid=False, rerank="none"),
        )

    @staticmethod
    def _top1(retriever, text: str, *, backend: str) -> SideHit | None:
        hits = retriever.search(text, top_k=1)
        if not hits:
            return None
        chunk = getattr(hits[0], "chunk", None)
        return SideHit(
            path=getattr(chunk, "source_path", "") or "",
            score=backend_score_to_cosine(hits[0].score, backend),
            text=(getattr(chunk, "text", "") or "")[:2000],
        )

    # ---------- 对外接口 ----------
    def score(self, question: str, answer: str) -> DedupResult:
        """双侧 top-1 相似度（检索失败抛异常 → 调用方按任务重试处理）。"""
        backend = settings.rag_backend.lower()
        retriever = self._open_retriever()
        if retriever is None:
            return DedupResult(None, None)
        return DedupResult(
            question=self._top1(retriever, question, backend=backend),
            answer=self._top1(retriever, answer, backend=backend),
        )

    def at_threshold(self, hit: SideHit | None, *, side: str) -> bool:
        """单侧命中是否达到该侧阈值（dedup.ge_with_tolerance 同款容差）。"""
        if hit is None:
            return False
        from app.evolution.dedup import ge_with_tolerance

        threshold = self._q_threshold if side == "question" else self._a_threshold
        return ge_with_tolerance(hit.score, threshold)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量向量（供批内 pairwise 复用同一 embedding 口径）。"""
        return self._embedder.encode(list(texts))

    def snapshot_meta(self) -> dict:
        """去重快照的审计元数据（评审代际 + embedding 版本）。"""
        generation = ""
        try:
            info = self._gen_store.active(settings.rag_backend.lower())
            generation = info.generation_id if info is not None else ""
        except Exception:  # noqa: BLE001 - audit metadata is best effort
            generation = ""
        return {"generation": generation, "embedding_version": self.embedder_model}


def drop_in_run_duplicates(
    vectors: list[list[float]],
    items: list[dict],
    *,
    threshold: float,
) -> list[int]:
    """批内 pairwise 去重：返回被淘汰的 items 下标（按传入顺序）。

    语义：高分胜出，同分小 candidate_id 胜出。实现上按 candidate_id 升序
    喂给 dedup.in_run_pairwise（平局保留先到者 → 小 ID 先到），分数字段只传
    value_score——避免 -rank 对 candidate_id 取负把平局判给大 ID。
    """
    from app.evolution.dedup import in_run_pairwise

    order = sorted(range(len(items)), key=lambda i: int(items[i]["candidate_id"]))
    metas = [{"value_score": float(items[i].get("value_score") or 0.0)} for i in order]
    dropped_sorted_order = in_run_pairwise(
        [vectors[i] for i in order],
        metas,
        threshold=threshold,
        score_keys=("value_score",),
    )
    return sorted(order[i] for i in dropped_sorted_order)
