"""统一检索器装配工厂（2.6）：线上 knowledge 单例与候选索引评测共用。

背景：检索装配逻辑（backend 选择/目标解析/hybrid/reranker/recall_k）原先
分散在 knowledge.py 单例与 pipeline._eval_after_and_probe（后者还是
`else -> chroma` 的错误兜底）。本工厂抽出单一装配路径：

open_retriever() 支持：
- 指定 backend（numpy/chroma/es），缺省跟随 settings.rag_backend；
- 指定 generation_target（numpy 索引文件 / chroma collection / ES index_name）——
  传 None 时回退固定路径（kb_index.json / ecom_kb / 活动 alias）；
- 按 retrieval_config 装配 hybrid（ES 原生 BM25+kNN+RRF / Python BM25+RRF）、
  reranker、recall_k 与相关性门控阈值。

final_search() 是**唯一**的最终检索口径（3 倍候选 ≤15 → 阈值门控 →
parent_id 去重 → Top-K）：线上 search_knowledge、阈值校准、retrieval_metrics.
evaluate 与候选索引探针全部经它出结果，保证线上与离线可比。

评测语义：ES 候选以 index_name=info.target 直查——**不经过活动 alias**，
保证测的是候选索引而不是线上已激活索引。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.agent.rag.backends.base import RetrievedChunk
from app.config.settings import settings


@dataclass
class RetrievalConfig:
    """一条完整检索配置（装配 hybrid/reranker/门控的全部输入）。

    评测 A/B 与线上共用同一结构：baseline-v2 只开纯向量，candidate-v2
    开 hybrid+reranker——配置不同才可解释，而不是查代码猜。
    """

    backend: str = "numpy"  # numpy | chroma | es
    hybrid: bool = False
    rerank: str = "none"  # none | cohere | jina | bge-reranker-v2-m3
    recall_k: int = 30
    min_score: float | None = None  # 相关性门控（None=未校准不门控）
    # 工作流 A：query 表层规范化（默认关；词表路径空 = settings 默认路径）
    query_normalize: bool = False
    query_normalize_lexicon_path: str = ""
    llm_timeout_seconds: float = 60.0
    # 后端路径覆盖（测试注入 tmp 目录用；与 IndexBuildService.backend_settings 同构）
    kb_index_path: str = ""  # numpy 无代际时的固定索引文件
    chroma_persist_dir: str = ""
    chroma_collection: str = ""
    backend_settings: dict | None = None  # IndexBuildService._b 透传（测试目录覆盖）

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "hybrid": self.hybrid,
            "rerank": self.rerank,
            "recall_k": self.recall_k,
            "min_score": self.min_score,
            "query_normalize": self.query_normalize,
        }


def retrieval_config_from_settings() -> RetrievalConfig:
    """冻结当前 settings 的检索配置（后续改动不复读 settings）。"""
    return RetrievalConfig(
        backend=settings.rag_backend.lower(),
        hybrid=settings.rag_hybrid,
        rerank=settings.rag_rerank,
        recall_k=settings.rag_hybrid_recall_k,
        min_score=settings.rag_min_relevance_score,
        query_normalize=settings.rag_query_normalize,
        query_normalize_lexicon_path=settings.rag_query_normalize_lexicon_path,
        llm_timeout_seconds=settings.llm_timeout_seconds,
        kb_index_path=settings.kb_index_path,
        chroma_persist_dir=settings.chroma_persist_dir,
        chroma_collection=settings.chroma_collection,
    )


def build_backend(config: RetrievalConfig, generation_target: str | None = None):
    """按配置 + 目标构建后端实例（generation_target 缺省回退固定路径/alias）。

    backend_settings 覆盖（测试注入 tmp 目录）优先级：显式 backend_settings
    > RetrievalConfig 显式字段 > settings。
    """
    from app.agent.rag.backends import create_backend

    b = config.backend_settings or {}
    name = config.backend.lower()
    if name == "numpy":
        target = (
            generation_target
            or b.get("kb_index_path")
            or config.kb_index_path
            or settings.kb_index_path
        )
        return create_backend("numpy", index_path=Path(target))
    if name == "chroma":
        persist_dir = (
            b.get("chroma_persist_dir")
            or config.chroma_persist_dir
            or settings.chroma_persist_dir
        )
        collection = (
            generation_target
            or b.get("chroma_collection")
            or config.chroma_collection
            or settings.chroma_collection
        )
        return create_backend(
            "chroma", persist_dir=Path(persist_dir), collection_name=collection,
        )
    if name == "es":
        if generation_target:
            # 候选索引：直查 index_name（不解析活动 alias）
            return create_backend("es", index_name=generation_target)
        # 线上：无显式 target → ESBackend 内部经 alias 解析
        return create_backend("es")
    raise ValueError(
        f"未知的 RAG 后端: {config.backend}（可选: numpy / chroma / es）"
    )


def open_retriever(
    config: RetrievalConfig | None = None,
    *,
    embedder=None,
    generation_target: str | None = None,
):
    """装配完整检索器（含 hybrid/reranker/recall_k 与相关性门控阈值）。

    返回对象具备 KnowledgeRetriever 接口（search/load/size）：
    - hybrid 关闭 → KnowledgeRetriever；
    - ES + hybrid → ESHybridRetriever（后端原生 BM25+kNN+RRF）；
    - numpy/chroma + hybrid → HybridRetriever（Python BM25 + RRF）。
    """
    from app.agent.rag.embedder import create_embedder
    from app.agent.rag.hybrid import ESHybridRetriever, HybridRetriever
    from app.agent.rag.rerank import create_reranker
    from app.agent.rag.retriever import KnowledgeRetriever

    config = config or retrieval_config_from_settings()
    embedder = embedder or create_embedder(timeout=config.llm_timeout_seconds)
    backend = build_backend(config, generation_target)
    retriever = KnowledgeRetriever(embedder=embedder, backend=backend)
    retriever.load()

    # 工作流 A：词表缺失/损坏 → load_normalizer 返回 None（fail-open 恒等 + 打点）
    normalizer = None
    if config.query_normalize:
        from app.agent.rag.query_normalizer import load_normalizer

        normalizer = load_normalizer(
            config.query_normalize_lexicon_path
            or settings.rag_query_normalize_lexicon_path
        )

    if not config.hybrid:
        return retriever

    # RAG 修复计划·1：rerank=none → 真正的 None（不再传 NullReranker，
    # 否则会被误判为「已挂精排」而恢复分数语义、错误地执行阈值门控）。
    reranker = None
    if config.rerank not in ("", "none", None):
        reranker = create_reranker(config.rerank)
    if hasattr(backend, "hybrid_search"):
        return ESHybridRetriever(
            embedder=embedder, backend=backend,
            recall_k=config.recall_k, reranker=reranker,
            normalizer=normalizer,
        )
    from app.agent.rag.bm25 import BM25Index

    return HybridRetriever(
        vector_retriever=retriever,
        bm25=BM25Index(backend.chunks()),
        recall_k=config.recall_k,
        reranker=reranker,
        normalizer=normalizer,
    )


@dataclass
class FinalSearchOutcome:
    """final_search 的完整结果：最终 Top-K + 评测诊断字段 + 状态。"""

    hits: list  # 最终 Top-K（阈值过滤 + 父块去重后）
    raw_hits: list  # 召回候选（阈值过滤前；校准扫阈值时复用，避免反复检索）
    kept_keys: list[str]  # 父块折叠后的命中键（parent_id 优先，无则 chunk_id）
    raw_candidates: int  # 原始候选数（阈值过滤前）
    gated_candidates: int  # 阈值过滤后候选数
    collapsed_parents: int  # 被去重的 parent 数量（同 parent_id 多候选的组数）
    raw_top_score: float | None  # 候选最高分（阈值过滤前，负例诊断用）
    # RAG 修复计划·1：显式检索状态
    score_source: str = "vector"  # rrf | vector | rerank
    degraded: bool = False
    degraded_reason: str = ""


def _search_with_status(retriever, query: str, top_k: int, timeout):
    """统一取检索结果 + 状态；不支持 status 的旧替身按 scores_meaningful 回落。"""
    fn = getattr(retriever, "search_with_status", None)
    if callable(fn):
        return fn(query, top_k=top_k, timeout=timeout)
    from app.agent.rag.retriever import (
        SCORE_SOURCE_RRF,
        SCORE_SOURCE_VECTOR,
        RetrievalResult,
    )

    hits = retriever.search(query, top_k=top_k, timeout=timeout)
    source = (
        SCORE_SOURCE_VECTOR
        if getattr(retriever, "scores_meaningful", True)
        else SCORE_SOURCE_RRF
    )
    return RetrievalResult(hits=hits, score_source=source)


def final_search(retriever, query: str, top_k: int,
                 min_score: float | None = None,
                 timeout: float | None = None) -> FinalSearchOutcome:
    """线上与离线共用的**最终检索口径**（单一实现，所有调用方走这里）。

    固定流程：
    1. 请求 top_k × 3 个候选（最大 15）——为父块去重预留候选冗余；
    2. 应用检索配置的相关度阈值 min_score（None = 未校准不门控）；
       **仅当分数有绝对语义（score_source=vector/rerank 且未降级）时门控**；
       RRF 秩融合分不得参与阈值判断（>0 会清空全部结果）；
    3. 按 parent_id 去重（同章节多个子块只保留排序最高者）；
    4. 截取最终 Top-K。

    召回与精排仍在子块粒度；这里只负责最终输出的门控与折叠。
    """
    from collections import Counter

    from app.agent.rag.retriever import collapse_by_parent, filter_hits_by_score

    top_k = max(1, int(top_k))
    recall = min(top_k * 3, 15)
    result = _search_with_status(retriever, query, recall, timeout)
    raw = result.hits
    effective_min = min_score if result.scores_meaningful else None
    gated = filter_hits_by_score(raw, effective_min)
    hits = collapse_by_parent(gated, top_k)
    pid_counts = Counter(h.chunk.parent_id for h in gated if h.chunk.parent_id)
    return FinalSearchOutcome(
        hits=hits,
        raw_hits=list(raw),
        kept_keys=[h.chunk.parent_id or h.chunk.chunk_id for h in hits],
        raw_candidates=len(raw),
        gated_candidates=len(gated),
        collapsed_parents=sum(1 for n in pid_counts.values() if n > 1),
        raw_top_score=max((h.score for h in raw), default=None),
        score_source=result.score_source,
        degraded=result.degraded,
        degraded_reason=result.degraded_reason,
    )


def search_with_gate(retriever, query: str, top_k: int,
                     min_score: float | None = None,
                     timeout: float | None = None) -> list:
    """（兼容保留）只取最终命中列表；新调用方请直接用 final_search。"""
    return final_search(retriever, query, top_k,
                        min_score=min_score, timeout=timeout).hits


@dataclass
class FinalMultiSearchOutcome:
    """final_multi_search 的完整结果（多子查询编排，阶段D + RAG修复计划·1）。

    形态与 FinalSearchOutcome 对齐，外加两个多路字段：
    - subqueries：实际参与检索的子查询；
    - hit_queries：与 hits **严格平行**（父块折叠后同步截断），标注每条命中
      来自哪个子查询（证据归因）。
    """

    hits: list  # 最终 Top-K（RRF 合并 + 父块折叠后；门控在合并前逐路生效）
    raw_hits: list  # 各路原始候选（阈值过滤前，跨路累计含重复）
    kept_keys: list[str]
    raw_candidates: int
    gated_candidates: int
    collapsed_parents: int
    raw_top_score: float | None
    score_source: str = "rrf"  # 单子查询路透传 final_search（vector/rerank/rrf）
    degraded: bool = False
    degraded_reason: str = ""
    subqueries: list = field(default_factory=list)
    hit_queries: list = field(default_factory=list)


def _rrf_merge_hits(ranked_lists: list, k: int = 60, top_n: int = 10):
    """chunk 级 RRF 秩融合（与 evidence.rrf_merge 同款语义）。

    ranked_lists = [(子查询, 该路候选), ...]；同一 parent_id（无父块则
    doc:section:chunk_id）只保留首见命中并累计融合分。返回
    (融合后候选, 与之平行的首见子查询列表)——RRF 分永不参与阈值门控。
    """
    scores: dict[str, float] = {}
    hit_by_key: dict[str, RetrievedChunk] = {}
    query_by_key: dict[str, str] = {}
    for subquery, ranked in ranked_lists:
        for rank, hit in enumerate(ranked):
            key = (
                hit.chunk.parent_id
                or f"{hit.chunk.doc}:{hit.chunk.section}:{hit.chunk.chunk_id}"
            )
            contribution = 1.0 / (k + rank + 1)
            hit_by_key.setdefault(key, hit)
            query_by_key.setdefault(key, subquery)
            scores[key] = scores.get(key, 0.0) + contribution
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    merged: list = []
    hit_queries: list = []
    for key, score in ordered[:top_n]:
        hit = hit_by_key[key]
        # P1-4：保留首见命中的精排分（联合拒绝 4 号信号；合并分是 RRF 无语义）
        merged.append(RetrievedChunk(
            chunk=hit.chunk, score=score,
            rerank_score=getattr(hit, "rerank_score", None),
        ))
        hit_queries.append(query_by_key[key])
    return merged, hit_queries


def final_multi_search(retriever, subqueries: list, top_k: int,
                       min_score: float | None = None,
                       timeout: float | None = None) -> FinalMultiSearchOutcome:
    """多子查询的**最终检索口径**（B1 提取：线上 search_knowledge 与多跳评测共用）。

    单子查询：直接等价 final_search（行为逐字段一致，回归保护）。
    多子查询：逐路并行「召回 → 阈值门控（仅当分数有绝对语义）」→ chunk 级
    RRF 秩融合（天然父块去重）→ 父块折叠 → Top-K；合并分是秩融合，永不门控；
    任一路 degraded 即聚合上报（供调用方 fail-closed）。

    min_score 由调用方注入：线上传 settings.rag_min_relevance_score，评测传
    本次校准阈值——与 final_search「阈值是参数而不是全局读取」的惯例一致。
    """
    from collections import Counter

    from app.agent.rag.retriever import (
        SCORE_SOURCE_RRF,
        filter_hits_by_score,
    )

    top_k = max(1, int(top_k))
    if not subqueries:
        raise ValueError("final_multi_search 需要至少一个子查询")

    if len(subqueries) == 1:
        outcome = final_search(retriever, subqueries[0], top_k,
                               min_score=min_score, timeout=timeout)
        return FinalMultiSearchOutcome(
            hits=outcome.hits,
            raw_hits=outcome.raw_hits,
            kept_keys=outcome.kept_keys,
            raw_candidates=outcome.raw_candidates,
            gated_candidates=outcome.gated_candidates,
            collapsed_parents=outcome.collapsed_parents,
            raw_top_score=outcome.raw_top_score,
            score_source=outcome.score_source,
            degraded=outcome.degraded,
            degraded_reason=outcome.degraded_reason,
            subqueries=[subqueries[0]],
            hit_queries=[subqueries[0]] * len(outcome.hits),
        )

    recall = min(top_k * 3, 15)

    def _one(subquery: str):
        result = _search_with_status(retriever, subquery, recall, timeout)
        # 仅当分数有绝对语义时才门控；RRF 秩融合分/降级结果跳过
        effective_min = min_score if result.scores_meaningful else None
        return result, filter_hits_by_score(result.hits, effective_min)

    from concurrent.futures import ThreadPoolExecutor

    workers = min(len(subqueries), 3)
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="rag-multi") as pool:
        pairs = list(pool.map(_one, subqueries))
    raw_all = [hit for result, _ in pairs for hit in result.hits]
    gated_all = [hit for _, gated in pairs for hit in gated]
    merged, merged_queries = _rrf_merge_hits(
        [(subquery, gated) for subquery, (_, gated) in zip(subqueries, pairs)],
        top_n=top_k * 2,
    )
    # 父块折叠与 collapse_by_parent 同语义（首见保留 + 截 top_k），但同步折叠
    # 子查询归因，保证 hit_queries 与 hits 严格平行——review 修复：此前返回
    # 折叠前的全长列表，调用方一旦按下标而非 zip 消费就会归因错位。
    hits: list = []
    hit_queries: list = []
    seen: set[str] = set()
    for hit, hit_query in zip(merged, merged_queries):
        pid = hit.chunk.parent_id
        if pid:
            if pid in seen:
                continue
            seen.add(pid)
        hits.append(hit)
        hit_queries.append(hit_query)
        if len(hits) >= top_k:
            break
    pid_counts = Counter(
        hit.chunk.parent_id for hit in gated_all if hit.chunk.parent_id
    )
    return FinalMultiSearchOutcome(
        hits=hits,
        raw_hits=raw_all,
        kept_keys=[hit.chunk.parent_id or hit.chunk.chunk_id for hit in hits],
        raw_candidates=len(raw_all),
        gated_candidates=len(gated_all),
        collapsed_parents=sum(1 for n in pid_counts.values() if n > 1),
        raw_top_score=max((hit.score for hit in raw_all), default=None),
        score_source=SCORE_SOURCE_RRF,
        degraded=any(result.degraded for result, _ in pairs),
        degraded_reason=next(
            (result.degraded_reason for result, _ in pairs
             if result.degraded_reason),
            "",
        ),
        subqueries=list(subqueries),
        hit_queries=hit_queries,
    )