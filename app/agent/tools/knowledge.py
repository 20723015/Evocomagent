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

from app.agent.context import ToolContext
from app.agent.rag.backends import create_backend
from app.agent.rag.chunker import chunk_body
from app.agent.rag.retriever import KnowledgeRetriever
from app.config.settings import settings
from app.evolution.generation import GenerationStore

_retriever: KnowledgeRetriever | None = None
_retriever_generation_id: str | None = None  # 当前单例对应的 generation
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
    except Exception as e:
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
MAX_SUBQUERIES = 3  # 阶段D：最多三个子查询


def _parent_window(c) -> tuple[str, str]:
    """父块窗口优先；父块不含命中子块正文时回退子块（旧索引/构建异常兜底）。"""

    body = chunk_body(c.text)
    if c.parent_text and body and body in c.parent_text:
        return c.parent_text, "parent"
    return c.text, "self"


def _item_from_hit(hit, query: str, raw_score: float | None = None):
    from app.agent.rag.evidence import EvidenceItem

    c = hit.chunk
    context_text, context_type = _parent_window(c)
    return EvidenceItem(
        doc=c.doc,
        section=c.section or "",
        heading_path=c.heading_path or "",
        chunk_id=c.chunk_id,
        parent_id=c.parent_id or "",
        query=query,
        score=hit.score if raw_score is None else float(raw_score),
        text=context_text,
        matched_text=c.text,
        source_path=c.source_path or c.doc,
        context_type=context_type,
    )


def search_knowledge(
    query: str = "", top_k: int = 3, ctx: ToolContext | None = None,
    timeout: float | None = None, queries: list[str] | None = None,
) -> dict:
    """检索退换货政策、配送说明、会员权益、FAQ 等知识库内容。

    阶段D：新增可选 `queries`（最多 3 个子查询，保持原 `query` 参数兼容）——
    比较、组合条件、跨文档问题由 Agent 一次提交多个子查询；检索层并行召回，
    RRF 合并、父块去重、reranker 精排（retriever 配置照常生效）。返回结果
    保持原字段，另附 subqueries 与 evidence 诊断字段。

    Returns:
        {
          "success": bool,
          "backend": "numpy" | "chroma" | "es",
          "query": str, "subqueries": [...],
          "results": [ ...原字段..., ],
          "evidence": {"top1_score": ..., "top1_gap": ..., "n_items": ...},
          "error": "..."  # 仅失败时存在
        }
    """
    subqueries = _normalize_subqueries(query, queries)
    if not subqueries:
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
    # 统一最终检索口径（与评测/校准/探针同一条 final_search）；多子查询并行
    # 召回后 RRF 合并再父块去重（阶段D）
    try:
        pack = _multi_query_search(retriever, subqueries, top_k, timeout)
    except Exception as e:
        return {
            "success": False,
            "error": f"知识库检索失败: {type(e).__name__}",
            "backend": settings.rag_backend,
            "query": query,
            "results": [],
        }

    results = pack.to_results()

    # 3.5 检索侧：KB 块（尤其 evolved/ 沉淀）视为不可信数据 →
    # 来源围栏包裹 + 注入语句标注 tainted 拦截
    if settings.retrieval_fence_enabled:
        from app.security.guardrails import search_result_fence_check

        results = search_result_fence_check(results)
        # 同步 tainted 标注回证据包（污染片段不进证据/来源）
        by_key = {}
        for item in pack.items:
            by_key[(item.doc, item.matched_text)] = item
        for r in results:
            item = by_key.get((r.get("doc"), r.get("matched_text")))
            if item is not None and r.get("tainted"):
                item.tainted = True
                item.text = r.get("text", item.text)

    from app.observability.metrics import record_rag_retrieve

    record_rag_retrieve(pack.diagnostics())

    return {
        "success": True,
        "backend": settings.rag_backend,
        "query": query,
        "subqueries": pack.subqueries,
        "results": results,
        "evidence": pack.diagnostics(),
    }


def _normalize_subqueries(query: str, queries: list[str] | None) -> list[str]:
    """子查询规整：query 必居首；queries 去重去空、最多 MAX_SUBQUERIES 个。"""
    out: list[str] = []
    if query and query.strip():
        out.append(query.strip())
    for item in queries or []:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
        if len(out) >= MAX_SUBQUERIES:
            break
    return out[:MAX_SUBQUERIES]


def _dedup_by_parent(items, top_k: int):
    """EvidenceItem 父块去重：同一 parent_id 只保留首见（排序已就绪），截 top_k。"""
    out: list = []
    seen: set[str] = set()
    for item in items:
        if item.parent_id:
            if item.parent_id in seen:
                continue
            seen.add(item.parent_id)
        out.append(item)
        if len(out) >= top_k:
            break
    return out


def _multi_query_search(retriever, subqueries: list[str], top_k: int,
                        timeout: float | None):
    """多子查询并行召回 → RRF 合并 → 阈值门控 → 父块去重 → Top-K。"""
    from concurrent.futures import ThreadPoolExecutor

    from app.agent.rag.evidence import EvidencePack, rrf_merge
    from app.agent.rag.retriever import filter_hits_by_score

    recall = min(top_k * 3, 15)
    min_score = settings.rag_min_relevance_score

    def _one(subquery: str):
        raw = retriever.search(subquery, top_k=recall, timeout=timeout)
        gated = filter_hits_by_score(raw, min_score)
        return [_item_from_hit(hit, subquery) for hit in gated]

    if len(subqueries) == 1:
        items = _one(subqueries[0])
        kept = _dedup_by_parent(items, top_k)
    else:
        workers = min(len(subqueries), 3)
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="rag-multi") as pool:
            ranked_lists = list(pool.map(_one, subqueries))
        # RRF 融合（天然父块去重）后截取 top_k
        kept = _dedup_by_parent(
            rrf_merge(ranked_lists, top_n=top_k * 2), top_k,
        )

    return EvidencePack(query=subqueries[0], subqueries=list(subqueries), items=kept)
