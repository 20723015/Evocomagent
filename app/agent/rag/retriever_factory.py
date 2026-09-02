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

评测语义：ES 候选以 index_name=info.target 直查——**不经过活动 alias**，
保证测的是候选索引而不是线上已激活索引。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
    min_score: Optional[float] = None  # 相关性门控（None=未校准不门控）
    llm_timeout_seconds: float = 60.0
    # 后端路径覆盖（测试注入 tmp 目录用；与 IndexBuildService.backend_settings 同构）
    kb_index_path: str = ""  # numpy 无代际时的固定索引文件
    chroma_persist_dir: str = ""
    chroma_collection: str = ""
    backend_settings: Optional[dict] = None  # IndexBuildService._b 透传（测试目录覆盖）

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "hybrid": self.hybrid,
            "rerank": self.rerank,
            "recall_k": self.recall_k,
            "min_score": self.min_score,
        }


def retrieval_config_from_settings() -> RetrievalConfig:
    """冻结当前 settings 的检索配置（后续改动不复读 settings）。"""
    return RetrievalConfig(
        backend=settings.rag_backend.lower(),
        hybrid=settings.rag_hybrid,
        rerank=settings.rag_rerank,
        recall_k=settings.rag_hybrid_recall_k,
        min_score=settings.rag_min_relevance_score,
        llm_timeout_seconds=settings.llm_timeout_seconds,
        kb_index_path=settings.kb_index_path,
        chroma_persist_dir=settings.chroma_persist_dir,
        chroma_collection=settings.chroma_collection,
    )


def build_backend(config: RetrievalConfig, generation_target: Optional[str] = None):
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
    config: Optional[RetrievalConfig] = None,
    *,
    embedder=None,
    generation_target: Optional[str] = None,
):
    """装配完整检索器（含 hybrid/reranker/recall_k 与相关性门控阈值）。

    返回对象具备 KnowledgeRetriever 接口（search/load/size）：
    - hybrid 关闭 → KnowledgeRetriever；
    - ES + hybrid → ESHybridRetriever（后端原生 BM25+kNN+RRF）；
    - numpy/chroma + hybrid → HybridRetriever（Python BM25 + RRF）。
    """
    from app.agent.rag.embedder import create_embedder
    from app.agent.rag.hybrid import ESHybridRetriever, HybridRetriever
    from app.agent.rag.retriever import KnowledgeRetriever
    from app.agent.rag.rerank import create_reranker

    config = config or retrieval_config_from_settings()
    embedder = embedder or create_embedder(timeout=config.llm_timeout_seconds)
    backend = build_backend(config, generation_target)
    retriever = KnowledgeRetriever(embedder=embedder, backend=backend)
    retriever.load()

    if not config.hybrid:
        return retriever

    reranker = create_reranker(config.rerank)
    if hasattr(backend, "hybrid_search"):
        return ESHybridRetriever(
            embedder=embedder, backend=backend,
            recall_k=config.recall_k, reranker=reranker,
        )
    from app.agent.rag.bm25 import BM25Index

    return HybridRetriever(
        vector_retriever=retriever,
        bm25=BM25Index(backend.chunks()),
        recall_k=config.recall_k,
        reranker=reranker,
    )


def search_with_gate(retriever, query: str, top_k: int,
                     min_score: Optional[float] = None,
                     timeout: Optional[float] = None):
    """检索 + 相关性门控（与 search_knowledge 工具同口径，供评测探针复用）。"""
    from app.agent.rag.retriever import filter_hits_by_score

    hits = retriever.search(query, top_k=top_k, timeout=timeout)
    return filter_hits_by_score(hits, min_score)