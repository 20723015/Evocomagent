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
from app.agent.rag.chunker import chunk_body, strip_inherited_table_head
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
    """父块窗口优先；父块不含命中子块正文时回退子块（旧索引/构建异常兜底）。

    续块表头（P0-4）是从同一父块更早位置复制的行，与命中正文拼接后不再是父块的
    连续子串——剥离继承表头后仍需落在父块内才认父块，否则回退命中子块。
    """

    body = chunk_body(c.text)
    if c.parent_text and body:
        if body in c.parent_text:
            return c.parent_text, "parent"
        stripped = strip_inherited_table_head(body)
        if stripped != body and stripped in c.parent_text:
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
    subqueries = _resolve_subqueries(query, queries, timeout)
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
    # 召回后 RRF 合并再父块去重（阶段D）。
    # P1-4 组合语义：联合拒绝是「单阈值不可达」的替代——联合阈值已配置
    # （params.active）时最终门控由联合判定独占，legacy 单阈值不再传入，
    # 避免同一层正例被两道门双重砍；仅当四个联合阈值全为 None 时才回落
    # legacy 单阈值（历史行为逐字段不变）。
    from app.agent.rag import rejection as rejection_mod

    rejection_params = rejection_mod.params_from_settings()
    legacy_min_score = (
        None if rejection_params.active else settings.rag_min_relevance_score
    )
    try:
        bundle = _multi_query_search_bundle(
            retriever, subqueries, top_k, timeout,
            min_score=legacy_min_score,
        )
    except Exception as e:
        return {
            "success": False,
            "error": f"知识库检索失败: {type(e).__name__}",
            "backend": settings.rag_backend,
            "query": query,
            "results": [],
        }

    pack = bundle.pack
    results = pack.to_results()

    # RAG 修复计划·1：精排不可用 → 生产默认 fail-closed（禁止用未验证证据回答）
    from app.observability.metrics import (
        record_rag_degraded,
        record_rag_rejection,
        record_rag_retrieve,
        record_rag_retrieve_failed,
    )

    if pack.degraded:
        record_rag_degraded(pack.degraded_reason or "unknown")
        if settings.rag_rerank_fail_closed:
            record_rag_retrieve_failed(pack.degraded_reason or "degraded")
            return {
                "success": False,
                "backend": settings.rag_backend,
                "query": query,
                "subqueries": pack.subqueries,
                "results": [],
                "error": pack.degraded_reason or "reranker_unavailable",
                "evidence": pack.diagnostics(),
            }

    # P1-4 四信号联合拒绝：在最终口径 hits（父块折叠后）上判定；降级/RRF 秩
    # 融合分无语义 → applicable=False（与 final_search 的阈值门控同一纪律）。
    # 判定为拒绝 → 与 degraded 分支同构的 fail-closed（success=False + 空
    # results），不新造返回结构。
    decision = rejection_mod.decide_rejection(
        bundle.outcome.hits, subqueries[0], rejection_params,
        score_source=bundle.outcome.score_source,
        degraded=bundle.outcome.degraded,
    )
    if decision.applicable:
        record_rag_rejection(
            "rejected" if decision.rejected else "accepted",
            decision.reasons,
            reason=decision.reason,
        )
    elif rejection_params.active:
        record_rag_rejection("skipped", reason=decision.skip_reason)
    if decision.rejected:
        record_rag_retrieve(pack.diagnostics())
        record_rag_retrieve_failed(f"rejected:{decision.reason or 'unknown'}")
        return {
            "success": False,
            "backend": settings.rag_backend,
            "query": query,
            "subqueries": pack.subqueries,
            "results": [],
            "error": f"retrieval_rejected:{decision.reason or 'unknown'}",
            "evidence": pack.diagnostics(),
        }

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


def _resolve_subqueries(query: str, queries: list[str] | None,
                        timeout: float | None) -> list[str]:
    """子查询规整 + P1-3 改写路：原 query 居首，改写结果追加在末尾。

    契约与 ``_normalize_subqueries`` 一致（原 query 必居首、去重、≤3）：
    - 模型已给满 MAX_SUBQUERIES 条 → 不改写（不挤掉模型显式提交的子查询）；
    - 改写失败/等价/未配置 → 原列表原样返回（fail-open，热路径不抛）；
    - 改写成功 → 作为第二/三路与原 query 并行召回，经 final_multi_search 的
      既有语义（逐路门控 → RRF 合并 → 父块折叠 → Top-K）合流。
    """
    subqueries = _normalize_subqueries(query, queries)
    if not subqueries or len(subqueries) >= MAX_SUBQUERIES:
        return subqueries
    import time as _time

    from app.agent.rag.query_rewrite import rewrite_query

    _t0 = _time.perf_counter()
    rewritten = rewrite_query(subqueries[0], timeout=timeout)
    _record_rewrite_latency_delta(_time.perf_counter() - _t0)
    if rewritten and rewritten not in subqueries:
        subqueries.append(rewritten)
    return subqueries[:MAX_SUBQUERIES]


def _record_rewrite_latency_delta(seconds: float) -> None:
    """改写路延迟增量打点（埋点失败不影响检索）。"""
    try:
        from app.observability.metrics import record_query_rewrite_latency_delta

        record_query_rewrite_latency_delta(seconds)
    except Exception:  # noqa: BLE001
        pass


class _MultiSearchBundle:
    """_multi_query_search 的完整结果：证据包 + 原始 outcome（联合拒绝取信号）。"""

    __slots__ = ("pack", "outcome")

    def __init__(self, pack, outcome):
        self.pack = pack
        self.outcome = outcome


def _multi_query_search_bundle(retriever, subqueries: list[str], top_k: int,
                               timeout: float | None,
                               min_score: float | None = None):
    """多子查询编排统一走 final_multi_search（B1 提取，线上/评测共用口径）。

    检索编排（并行召回 → 逐路门控 → RRF 合并 → 父块去重 → Top-K，RRF 分与
    降级状态聚合）收敛到 retriever_factory.final_multi_search；本函数只负责
    在其结果之上构建 EvidencePack（父块窗口文本、子查询归因），并保留原始
    outcome 供 P1-4 联合拒绝在最终 hits 上取信号。
    """
    from app.agent.rag.evidence import EvidencePack
    from app.agent.rag.retriever_factory import final_multi_search

    outcome = final_multi_search(
        retriever, subqueries, top_k,
        # 统一口径：阈值由调用方显式传入（线上传 legacy 单阈值或 None，
        # 评测侧传本次校准阈值），检索层不再自行读取 settings
        min_score=min_score, timeout=timeout,
    )
    items = [
        _item_from_hit(hit, query)
        for hit, query in zip(outcome.hits, outcome.hit_queries)
    ]
    pack = EvidencePack(
        query=subqueries[0], subqueries=list(subqueries), items=items,
        score_source=outcome.score_source, degraded=outcome.degraded,
        degraded_reason=outcome.degraded_reason,
    )
    return _MultiSearchBundle(pack=pack, outcome=outcome)


def _multi_query_search(retriever, subqueries: list[str], top_k: int,
                        timeout: float | None):
    """（兼容保留）只取证据包；联合拒绝调用方请用 _multi_query_search_bundle。"""
    return _multi_query_search_bundle(
        retriever, subqueries, top_k, timeout,
        min_score=settings.rag_min_relevance_score,
    ).pack
