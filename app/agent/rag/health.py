"""RAG 生产配置健康校验（RAG 修复计划·3）。

启动/就绪阶段验证：
1. ES 可连接（rag_backend=es 时）；
2. 活动 generation 存在；
3. 索引 embedding 模型与运行配置一致（防「换了 embedding 还在用老索引」）；
4. reranker 可用（配置了非 none 时，端点探活）；
5. 配置了相关性阈值时，必须有分数有绝对语义的打分器（vector 或 rerank）——
   hybrid + rerank=none 的 RRF 秩融合分无语义，配阈值无意义（fail-closed）。

返回结构化结果，供 /readyz 与启动日志使用；不抛异常（调用方决定 fail-closed）。
"""

from __future__ import annotations

from app.config.settings import settings


def _check_es() -> tuple[str, str]:
    from app.agent.rag.es_util import es_dependency_state

    return es_dependency_state(), ""


def _active_generation():
    try:
        from pathlib import Path

        from app.evolution.generation import GenerationStore
        from app.stores.redis_client import get_redis

        store = GenerationStore(
            Path(settings.kb_generation_path), redis_client=get_redis(),
        )
        return store.active("es")
    except Exception:  # noqa: BLE001
        return None


def _check_generation() -> tuple[str, str]:
    gen = _active_generation()
    if gen is None:
        return "unavailable", "无活动 generation（请先构建/激活索引）"
    return "ok", ""


def _check_embedding_model() -> tuple[str, str]:
    """索引 `_meta` 的 provider/model/dimensions/指纹与运行配置逐项比对。"""
    from app.agent.rag.fingerprint import config_fingerprint, embedding_dimensions
    from app.agent.rag.backends.es_backend import ESBackend
    from app.agent.rag.es_util import get_es_client

    gen = _active_generation()
    if gen is None:
        return "unavailable", "无活动 generation，无法校验 embedding 模型"
    es = get_es_client()
    if es is None:
        return "unavailable", "ES 不可达，无法校验 embedding 模型"
    try:
        backend = ESBackend(es, index_name=gen.target)
        meta = backend.expected_embedding_meta()
        embedding_model = str(meta.get("embedding_model", "") or "")
        if not embedding_model:
            return "unavailable", "索引缺少 embedding_model 元数据（旧索引，请重建）"
        # 逐项比对（旧索引缺字段 → 需重建）
        from app.config.settings import settings as _s

        expected_provider = (_s.embedding_provider or "openai").lower()
        indexed_provider = str(meta.get("embedding_provider", "") or "").lower()
        if indexed_provider != expected_provider:
            return (
                "unavailable",
                f"索引 embedding provider({indexed_provider or '缺失'}) 与运行配置"
                f"({expected_provider}) 不一致，请重建索引",
            )
        expected_dims = embedding_dimensions()
        indexed_dims = int(meta.get("embedding_dimensions", 0) or 0)
        if expected_dims and indexed_dims != expected_dims:
            return (
                "unavailable",
                f"索引 embedding dimensions({indexed_dims or '缺失'}) 与运行配置"
                f"({expected_dims}) 不一致，请重建索引",
            )
        indexed_fp = str(meta.get("config_fingerprint", "") or "")
        if not indexed_fp:
            return "unavailable", "索引缺少 config_fingerprint（旧索引，请重建）"
        if indexed_fp != config_fingerprint():
            return (
                "unavailable",
                "索引配置指纹与运行配置不一致"
                "（解析/分块版本或 embedding/hybrid/rerank 变更），请重建索引",
            )
        # 运行时 embedder 模型名一致性（复用既有校验）
        from app.agent.rag.embedder import create_embedder

        embedder = create_embedder(timeout=settings.llm_timeout_seconds)
        if embedding_model != embedder.model:
            return (
                "unavailable",
                f"索引模型({embedding_model}) 与运行配置({embedder.model}) 不一致，请重建索引",
            )
        return "ok", ""
    except Exception as e:  # noqa: BLE001
        return "unavailable", f"embedding 模型校验失败: {type(e).__name__}"


_RERANKER_CACHE: dict = {"at": 0.0, "state": "", "err": ""}
RERANKER_PROBE_TTL_SECONDS = 15.0


def _parse_probe_pairs(data) -> list[tuple[int, float]]:
    """把探活响应解析为 (index, score) 列表（与 HTTPReranker 协议一致）。"""
    if isinstance(data, dict):
        results = data.get("results")
        if results is None:
            return []
        items = results if isinstance(results, list) else [results]
    elif isinstance(data, list):
        items = data
    else:
        return []
    out: list[tuple[int, float]] = []
    for item in items:
        try:
            if isinstance(item, dict):
                idx = item.get("index")
                score = item.get("relevance_score", item.get("score"))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                idx, score = item[0], item[1]
            else:
                continue
            if idx is None or score is None:
                continue
            out.append((int(idx), float(score)))
        except (TypeError, ValueError):
            continue
    return out


def _probe_reranker(endpoint: str) -> tuple[str, str]:
    """探活：仅接受 2xx + 含 index=0 有限分数的响应；401/404/422/错误协议 → unavailable。"""
    import math

    try:
        import httpx

        client = httpx.Client(trust_env=False, timeout=3.0)
        try:
            resp = client.post(endpoint, json={"query": "ping", "texts": ["pong"]})
        finally:
            client.close()
    except Exception as e:  # noqa: BLE001
        return "unavailable", f"reranker 端点不可达: {type(e).__name__}"
    if not (200 <= resp.status_code < 300):
        return "unavailable", f"reranker 探活返回 {resp.status_code}"
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        return "unavailable", "reranker 探活响应非 JSON"
    pairs = _parse_probe_pairs(data)
    if not any(idx == 0 and math.isfinite(score) for idx, score in pairs):
        return "unavailable", "reranker 探活响应缺少 index=0 的有限分数"
    return "ok", ""


def _check_reranker() -> tuple[str, str]:
    name = (settings.rag_rerank or "none").lower()
    if name in ("", "none"):
        return "not_configured", ""
    endpoint = settings.rerank_endpoint_url or ""
    if not endpoint:
        return "unavailable", "已配置 reranker 但缺少 RERANK_ENDPOINT_URL"
    import time

    now = time.monotonic()
    if (
        _RERANKER_CACHE["state"]
        and now - _RERANKER_CACHE["at"] < RERANKER_PROBE_TTL_SECONDS
    ):
        return _RERANKER_CACHE["state"], _RERANKER_CACHE["err"]
    state, err = _probe_reranker(endpoint)
    _RERANKER_CACHE.update(at=now, state=state, err=err)
    return state, err


def _check_threshold_scorer() -> tuple[str, str]:
    """配置了阈值时，必须有分数有绝对语义的打分器。"""
    if settings.rag_min_relevance_score is None:
        return "not_configured", ""
    backend = (settings.rag_backend or "").lower()
    hybrid = bool(settings.rag_hybrid)
    rerank = (settings.rag_rerank or "none").lower()
    # 纯向量（非 hybrid）或挂了 reranker：分数有绝对语义
    if not hybrid or rerank not in ("", "none"):
        return "ok", ""
    return (
        "unavailable",
        "配置了相关性阈值，但 hybrid+rerank=none 的 RRF 分无语义："
        "请启用 reranker 或移除阈值",
    )


def _check_production_requirements() -> list[str]:
    """生产环境强校验（RAG 修复计划·3）：任一不满足即 unavailable。

    仅当 app_env=prod（或显式 RAG_PROD_STRICT=true）时生效。
    """
    import os

    is_prod = (settings.app_env or "").lower() == "prod" or (
        os.environ.get("RAG_PROD_STRICT", "").strip().lower() == "true"
    )
    if not is_prod:
        return []
    problems: list[str] = []
    if not settings.rag_rerank_fail_closed:
        problems.append("生产要求 RAG_RERANK_FAIL_CLOSED=true")
    if not settings.rag_doc_metadata_required:
        problems.append("生产要求 RAG_DOC_METADATA_REQUIRED=true")
    if settings.rag_min_relevance_score is None:
        problems.append("生产要求配置已校准的 RAG_MIN_RELEVANCE_SCORE")
    return problems


def check_rag_configuration() -> dict:
    """返回 {status, checks, errors}；status ∈ ok|not_configured|unavailable。"""
    checks: dict[str, str] = {}
    errors: list[str] = []

    thr_state, thr_err = _check_threshold_scorer()
    checks["threshold_scorer"] = thr_state
    if thr_err:
        errors.append(thr_err)

    rr_state, rr_err = _check_reranker()
    checks["reranker"] = rr_state
    if rr_err:
        errors.append(rr_err)

    checks["es"] = "not_configured"
    checks["generation"] = "not_configured"
    checks["embedding_model"] = "not_configured"

    if (settings.rag_backend or "").lower() == "es":
        es_state, _ = _check_es()
        gen_state, gen_err = _check_generation()
        emb_state, emb_err = _check_embedding_model()
        checks["es"] = es_state
        checks["generation"] = gen_state
        checks["embedding_model"] = emb_state
        if es_state == "unavailable":
            errors.append("ES 不可连接")
        if gen_err:
            errors.append(gen_err)
        if emb_err:
            errors.append(emb_err)

    # 生产额外强校验（不满足 → 直接 unavailable）
    prod_problems = _check_production_requirements()
    checks["production_config"] = "unavailable" if prod_problems else "not_configured"
    errors.extend(prod_problems)

    if "unavailable" in checks.values():
        status = "unavailable"
    elif all(v == "not_configured" for v in checks.values()):
        status = "not_configured"
    else:
        status = "ok"
    return {"status": status, "checks": checks, "errors": errors}
