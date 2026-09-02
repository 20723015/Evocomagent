"""知识库检索工具：通过向量检索回答 FAQ、政策类问题。

与查订单/查物流这类「结构化数据查询」工具不同，
search_knowledge 面向非结构化文本（退换货政策、配送说明、FAQ 等），
返回 Top-K 命中片段及其来源，由 LLM 引用回答。

向量后端由 settings.rag_backend 切换：
- numpy ：手写余弦相似度，零依赖，教学透明
- chroma：嵌入式向量数据库，HNSW 索引，工程代表性

索引代际（第10期）：
- 每次 search 前读 generation 指针文件（文件极小，直接读）；有活动 generation
  则按 target 实例化后端，无则回退固定路径（kb_index.json / ecom_kb）。
- generation ID 与单例缓存不一致 → 重建（embedder + backend + load）；
  读取异常或重建失败 → 沿用内存中 last-known-good 实例，保证检索不中断。
- push_retriever_override / pop_retriever_override：with-eval 沙箱临时把检索
  指向 staging 索引（激活时跳过 generation 检查）。
"""

from __future__ import annotations
from app.observability.logging import get_logger
log = get_logger("app.agent.tools.knowledge")


from pathlib import Path
from typing import Optional

from app.agent.context import ToolContext
from app.config.settings import settings
from app.agent.rag.backends import create_backend
from app.agent.rag.embedder import create_embedder
from app.agent.rag.retriever import KnowledgeRetriever
from app.agent.rag.retriever import filter_hits_by_score
from app.evolution.generation import GenerationStore

_retriever: Optional[KnowledgeRetriever] = None
_retriever_generation_id: Optional[str] = None  # 当前单例对应的 generation
_override_stack: list[KnowledgeRetriever] = []  # with-eval staging override


def _create_backend_from_settings():
    """根据 settings.rag_backend 创建对应后端实例（固定路径/alias 回退）。"""
    name = settings.rag_backend.lower()
    if name == "numpy":
        return create_backend("numpy", index_path=Path(settings.kb_index_path))
    if name == "chroma":
        return create_backend(
            "chroma",
            persist_dir=Path(settings.chroma_persist_dir),
            collection_name=settings.chroma_collection,
        )
    if name == "es":
        # collection_name 缺省 → ESBackend 内部派生 alias（{prefix}-kb-active）
        return create_backend("es")
    raise ValueError(
        f"未知的 RAG 后端: {settings.rag_backend}（可选: numpy / chroma / es）"
    )


def _create_backend_for_generation(gen):
    """按 activity generation 的 target 实例化后端（numpy 为绝对路径索引文件；es 为索引名）。"""
    name = settings.rag_backend.lower()
    if name == "numpy":
        return create_backend("numpy", index_path=Path(gen.target))
    if name == "chroma":
        return create_backend(
            "chroma",
            persist_dir=Path(settings.chroma_persist_dir),
            collection_name=gen.target,
        )
    if name == "es":
        return create_backend("es", index_name=gen.target)
    raise ValueError(f"未知的 RAG 后端: {settings.rag_backend}（可选: numpy / chroma / es）")


def _generation_store() -> GenerationStore:
    """代际指针存储：注入 shared Redis 时跨 pod 共享（阶段二 2.4）。"""
    from app.stores.redis_client import get_redis

    return GenerationStore(
        Path(settings.kb_generation_path),
        redis_client=get_redis(),
    )


def _make_retriever() -> KnowledgeRetriever:
    """构建 retriever：优先活动 generation，缺省回退固定路径。

    2.6：装配逻辑（hybrid/reranker/recall_k/相关性门控）统一收敛到
    app.agent.rag.retriever_factory.open_retriever——线上单例与候选索引
    评测共用同一条装配路径，不再各自复制逻辑。
    """
    from app.agent.rag.retriever_factory import (
        open_retriever,
        retrieval_config_from_settings,
    )

    gen = _generation_store().active(settings.rag_backend)
    return open_retriever(
        retrieval_config_from_settings(),
        generation_target=gen.target if gen is not None else None,
    )


def _get_retriever() -> KnowledgeRetriever:
    global _retriever, _retriever_generation_id

    if _override_stack:
        return _override_stack[-1]

    gen_id = None
    try:
        gen = _generation_store().active(settings.rag_backend)
        gen_id = gen.generation_id if gen else ""
    except Exception:  # noqa: BLE001 —— 指针读取异常：沿用内存实例
        if _retriever is not None:
            return _retriever

    if _retriever is not None and gen_id == _retriever_generation_id:
        return _retriever

    try:
        retriever = _make_retriever()
    except Exception as e:  # noqa: BLE001 —— 重建失败：保留 last-known-good
        if _retriever is not None:
            log.info(f"⚠️  检索器重建失败，沿用现有实例（{type(e).__name__}: {e}）")
            return _retriever
        raise

    _retriever = retriever
    _retriever_generation_id = gen_id
    return _retriever


def reset_retriever() -> None:
    """清空单例缓存（测试或切换后端时使用）。"""
    global _retriever, _retriever_generation_id
    _retriever = None
    _retriever_generation_id = None


# ============================================================
# with-eval staging override
# ============================================================
def push_retriever_override(retriever: KnowledgeRetriever) -> None:
    """临时把检索指向 staging 索引（with-eval 沙箱用，跳过 generation 检查）。"""
    _override_stack.append(retriever)


def pop_retriever_override() -> None:
    """恢复上一层检索器；栈空时静默。"""
    if _override_stack:
        _override_stack.pop()


# ============================================================
# 工具入口
# ============================================================
def search_knowledge(
    query: str = "", top_k: int = 3, ctx: Optional[ToolContext] = None,
    timeout: Optional[float] = None,
) -> dict:
    """检索退换货政策、配送说明、会员权益、FAQ 等知识库内容。

    Returns:
        {
          "success": bool,
          "backend": "numpy" | "chroma",
          "query": str,
          "results": [
            {"doc": "...", "section": "...", "score": 0.83, "text": "...",
             "source_path": "..."},
            ...
          ],
          "error": "..."  # 仅失败时存在
        }
    """
    if not query or not query.strip():
        return {"success": False, "error": "query 不能为空", "query": query, "results": []}

    try:
        retriever = _get_retriever()
    except FileNotFoundError as e:
        return {
            "success": False,
            "error": str(e),
            "backend": settings.rag_backend,
            "query": query,
            "results": [],
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"知识库初始化失败: {e}",
            "backend": settings.rag_backend,
            "query": query,
            "results": [],
        }

    top_k = max(1, min(int(top_k or 3), 5))
    # 修复计划：remaining 透传——Embedder/远端 reranker/ES 全部受轮次预算约束
    hits = retriever.search(query, top_k=top_k, timeout=timeout)
    hits = filter_hits_by_score(hits, settings.rag_min_relevance_score)

    results = [
        {
            "doc": h.chunk.doc,
            "section": h.chunk.section,
            "score": round(h.score, 4),
            "text": h.chunk.text,
            "source_path": h.chunk.source_path or h.chunk.doc,
            "provenance": h.chunk.provenance or "",
        }
        for h in hits
    ]
    # 3.5 检索侧：KB 块（尤其 evolved/ 沉淀）视为不可信数据 →
    # 来源围栏包裹 + 注入语句标注 tainted 拦截
    if settings.retrieval_fence_enabled:
        from app.security.guardrails import search_result_fence_check

        results = search_result_fence_check(results)

    return {
        "success": True,
        "backend": settings.rag_backend,
        "query": query,
        "results": results,
    }
