"""评测 manifest：一次 v2 评测的完整指纹（3.1 冻结协议）。

manifest 记录：
- 数据集 SHA-256 + 条数
- Git commit（head）
- Prompt 目录 SHA-256（评估/记忆/summarizer 提示词——判定口径的一部分）
- 被测模型、与被测模型不同的 Judge 模型
- 实际 CLI 的 mode、Judge 开关、temperature、token 上限、后端、reranker、阈值
- Python 版本与依赖锁（requirements.txt）哈希

用途：
- 报告自描述（artifacts/eval/v2/<run-id>/manifest.json）；
- run_eval_resilient 断点续跑/合并前校验配置未漂移（不一致拒绝合并）；
- 简历/评估文档引用「该报告对应什么配置」的唯一凭据。
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from app.config.settings import settings

ROOT = Path(__file__).resolve().parent.parent.parent
PROMPT_DIR = ROOT / "app" / "prompts"
REQUIREMENTS = ROOT / "requirements.txt"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _prompt_sha256() -> str:
    """评估/记忆/摘要提示词目录的聚合指纹（口径一部分）。"""
    h = hashlib.sha256()
    for p in sorted(PROMPT_DIR.glob("*.py")):
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


def _deps_sha256() -> str:
    if REQUIREMENTS.exists():
        return sha256_of(REQUIREMENTS)
    return ""


def _endpoint_sha256(value: str | None) -> str:
    """对完整 endpoint（可能含 query 凭据）做不可逆指纹。"""
    if not value:
        return ""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _safe_endpoint(value: str | None) -> str:
    """返回不携带 userinfo/query/fragment 的可展示 endpoint。

    OpenAI-compatible 网关和 reranker 常被配置成带 query token 或
    ``user:password@host`` 的 URL；manifest 只保存规范化地址，完整原值
    通过 ``*_sha256`` 参与指纹，绝不明文落盘。
    """
    raw = str(value or "")
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
        if not parts.scheme or not parts.netloc:
            # 对无 scheme 的内部地址也至少去掉 query/fragment；若形式可疑
            # 且含 userinfo，宁可完全隐藏而不冒险泄露凭据。
            if "@" in raw:
                return "<redacted-endpoint>"
            return raw.split("?", 1)[0].split("#", 1)[0]
        hostname = parts.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        try:
            port = parts.port
        except ValueError:
            return "<redacted-endpoint>"
        netloc = hostname + (f":{port}" if port is not None else "")
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    except ValueError:
        return "<redacted-endpoint>"


def _context_fingerprint() -> dict[str, str]:
    """生成续跑必须保持不变的源码/依赖上下文指纹。"""
    return {
        "prompt_sha256": _prompt_sha256(),
        "deps_sha256": _deps_sha256(),
        "git_commit": git_commit(),
    }


def _runtime_config(
    *,
    dataset_path: str | None = None,
    model: str | None = None,
    judge_model: str | None = None,
    mode: str | None = None,
    use_judge: bool | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """返回参与评测的、可安全落盘的运行配置。

    这里刻意不读取 ``settings.eval_*`` 来代替 CLI 参数：续跑的 manifest
    必须描述「实际运行」的模型、模式与 Judge 开关，而不是当前进程后来
    重新加载出来的默认值。API key 等凭据永远不进入 manifest。
    """
    config: dict[str, Any] = {
        "dataset_path": str(dataset_path if dataset_path is not None else settings.eval_dataset_path),
        "model": str(model if model is not None else settings.model_name),
        "judge_model": str(
            judge_model if judge_model is not None else (settings.eval_judge_model or "")
        ),
        "mode": str(mode if mode is not None else (
            "multi" if settings.multi_agent_enabled else "single"
        )),
        "use_judge": bool(
            use_judge if use_judge is not None else settings.eval_use_judge
        ),
        # Endpoint affects the actual model service and therefore the
        # experiment, while userinfo/query credentials remain redacted.
        "openai_base_url": _safe_endpoint(settings.openai_base_url),
        "openai_base_url_sha256": _endpoint_sha256(settings.openai_base_url),
        "temperature": settings.temperature,
        "llm_max_tokens": settings.llm_max_tokens,
        "multi_agent_enabled": settings.multi_agent_enabled,
        "eval_pass_threshold": settings.eval_pass_threshold,
        "rag": {
            "backend": settings.rag_backend,
            "embedding_model": settings.embedding_model,
            "embedding_provider": settings.embedding_provider,
            "embedding_endpoint": _safe_endpoint(settings.sophnet_embedding_url),
            "embedding_endpoint_sha256": _endpoint_sha256(settings.sophnet_embedding_url),
            "hybrid": settings.rag_hybrid,
            "hybrid_recall_k": settings.rag_hybrid_recall_k,
            "rerank": settings.rag_rerank,
            "rerank_endpoint": _safe_endpoint(settings.rerank_endpoint_url),
            "rerank_endpoint_sha256": _endpoint_sha256(settings.rerank_endpoint_url),
            "rerank_model": settings.rerank_model,
            "min_relevance_score": settings.rag_min_relevance_score,
            "kb_index_path": settings.kb_index_path,
            "chroma_persist_dir": settings.chroma_persist_dir,
        },
        "guardrails": {
            "enabled": settings.guardrails_enabled,
            "block_terms": settings.guardrail_block_terms,
            "retrieval_fence_enabled": settings.retrieval_fence_enabled,
            "citation_check_enabled": settings.citation_check_enabled,
            "business_only_scope": settings.business_only_scope,
        },
        "tool_guard": {
            "enabled": settings.tool_call_guard_enabled,
            "max_calls_per_name": settings.tool_max_calls_per_name,
            "search_max_calls": settings.tool_search_max_calls,
        },
        "authorization": {
            "enforce_order_ownership": settings.enforce_order_ownership,
            "refund_confirmation_required": settings.refund_confirmation_required,
        },
        "commerce_backend": settings.commerce_backend,
    }
    if overrides:
        # Overrides are intended for non-secret, experiment-specific fields
        # (for example an A/B arm). Keep a shallow top-level merge so nested
        # settings remain explicit and deterministic.
        config.update(dict(overrides))
    return config


def _combined_sha256(**kwargs: Any) -> dict[str, Any]:
    """稳定序列化的配置指纹（合并/续跑比较用；不含易变字段）。"""
    return _runtime_config(**kwargs)


def config_hash(
    *,
    dataset_path: str | None = None,
    model: str | None = None,
    judge_model: str | None = None,
    mode: str | None = None,
    use_judge: bool | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> str:
    """配置指纹（续跑/合并校验用）。

    所有 CLI 运行时参数都必须显式传入；不传时才回退到 settings，兼容
    旧的内部调用。这样 ``--mode single`` / ``--no-judge`` 等不会被忽略。
    """
    raw = json.dumps(
        _combined_sha256(
            dataset_path=dataset_path,
            model=model,
            judge_model=judge_model,
            mode=mode,
            use_judge=use_judge,
            overrides=overrides,
        ),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_manifest(
    dataset_path: str,
    num_cases: int,
    model: str,
    judge_model: str,
    rng_seed: str = "",
    *,
    mode: str = "single",
    use_judge: bool = False,
    config_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构建 v2 评测 manifest。"""
    ds = Path(dataset_path)
    context = _context_fingerprint()
    config = _runtime_config(
        dataset_path=str(ds), model=model, judge_model=judge_model,
        mode=mode, use_judge=use_judge, overrides=config_overrides,
    )
    return {
        "protocol": "eval-v2",
        "frozen": True,  # 冻结集：禁止静默改写（--check 保护）
        "dataset": {
            "path": str(ds),
            "sha256": sha256_of(ds) if ds.exists() else "",
            "num_cases": num_cases,
        },
        "git": {"commit": context["git_commit"]},
        "prompts": {"sha256": context["prompt_sha256"]},
        "model": {
            "under_test": model,
            "judge": judge_model,
            "use_judge": use_judge,
            "temperature": settings.temperature or 0.0,
            "max_tokens": settings.llm_max_tokens,
        },
        "retrieval": {
            "backend": settings.rag_backend,
            "hybrid": settings.rag_hybrid,
            "hybrid_recall_k": settings.rag_hybrid_recall_k,
            "rerank": settings.rag_rerank,
            "rerank_model": settings.rerank_model,
            "embedding_model": settings.embedding_model,
            "embedding_provider": settings.embedding_provider,
            "embedding_endpoint": _safe_endpoint(settings.sophnet_embedding_url),
            "embedding_endpoint_sha256": _endpoint_sha256(settings.sophnet_embedding_url),
            "rerank_endpoint": _safe_endpoint(settings.rerank_endpoint_url),
            "rerank_endpoint_sha256": _endpoint_sha256(settings.rerank_endpoint_url),
            "min_score": settings.rag_min_relevance_score,
        },
        "guardrails": {
            "enabled": settings.guardrails_enabled,
            "retrieval_fence_enabled": settings.retrieval_fence_enabled,
            "citation_check_enabled": settings.citation_check_enabled,
            "tool_call_guard_enabled": settings.tool_call_guard_enabled,
        },
        "threshold": {
            "pass": settings.eval_pass_threshold,
        },
        "execution": {
            "mode": mode,
            "use_judge": use_judge,
            "base_url": _safe_endpoint(settings.openai_base_url),
            "base_url_sha256": _endpoint_sha256(settings.openai_base_url),
        },
        "config": config,
        "runtime": {
            "python": platform.python_version(),
            "deps_sha256": context["deps_sha256"],
            "rng_seed": rng_seed,
        },
        "config_hash": config_hash(
            dataset_path=str(ds), model=model, judge_model=judge_model,
            mode=mode, use_judge=use_judge, overrides=config_overrides,
        ),
    }


def verify_manifest_unchanged(
    manifest: dict,
    *,
    dataset_path: str,
    model: str,
    judge_model: str,
    mode: str = "single",
    use_judge: bool = False,
    config_overrides: Mapping[str, Any] | None = None,
    allow_context_mismatch: bool = False,
) -> bool:
    """续跑/合并前校验：配置与源码/依赖上下文均未漂移。

    默认 fail-closed：旧 manifest 缺少 prompts/deps/git 字段，或当前上下文
    任一指纹不同，均拒绝混合 checkpoint。特殊迁移场景若确需绕过，必须由
    调用方显式传入 ``allow_context_mismatch=True``，不改变默认安全语义。
    """
    if not isinstance(manifest, dict):
        return False
    if manifest.get("protocol") != "eval-v2":
        return False
    ds = manifest.get("dataset") or {}
    dataset_ok = (
        ds.get("sha256") == (sha256_of(Path(dataset_path))
                             if Path(dataset_path).exists() else "")
    )
    expected_hash = config_hash(
        dataset_path=str(dataset_path), model=model, judge_model=judge_model,
        mode=mode, use_judge=use_judge, overrides=config_overrides,
    )
    context = _context_fingerprint()
    recorded_context = {
        "prompt_sha256": (manifest.get("prompts") or {}).get("sha256"),
        "deps_sha256": (manifest.get("runtime") or {}).get("deps_sha256"),
        "git_commit": (manifest.get("git") or {}).get("commit"),
    }
    context_ok = all(
        recorded_context[key] is not None
        and recorded_context[key] not in {"", "unknown"}
        and context[key] not in {"", "unknown"}
        and recorded_context[key] == context[key]
        for key in context
    )
    if not allow_context_mismatch and not context_ok:
        return False
    return dataset_ok and manifest.get("config_hash") == expected_hash


def build_retrieval_manifest(
    *,
    dataset_path: str,
    num_cases: int,
    top_k: int,
    min_score: float | None,
    min_score_hard: float | None,
    thresholds: Mapping[str, Any],
    variant: str,
    hard_threshold_overridden: bool = False,
) -> dict[str, Any]:
    """构建独立于端到端 eval-v2 的检索实验清单。

    检索报告不调用 ``build_manifest``，因为它没有 Agent model/Judge/mode；
    但同样必须冻结数据集哈希、git、后端/embedding/reranker 和所有阈值。
    """
    ds = Path(dataset_path)
    context = _context_fingerprint()
    retrieval = {
        "backend": settings.rag_backend,
        "embedding_model": settings.embedding_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_endpoint": _safe_endpoint(settings.sophnet_embedding_url),
        "embedding_endpoint_sha256": _endpoint_sha256(settings.sophnet_embedding_url),
        "openai_base_url": _safe_endpoint(settings.openai_base_url),
        "openai_base_url_sha256": _endpoint_sha256(settings.openai_base_url),
        "hybrid": settings.rag_hybrid,
        "hybrid_recall_k": settings.rag_hybrid_recall_k,
        "rerank": settings.rag_rerank,
        "rerank_endpoint": _safe_endpoint(settings.rerank_endpoint_url),
        "rerank_endpoint_sha256": _endpoint_sha256(settings.rerank_endpoint_url),
        "rerank_model": settings.rerank_model,
        "min_relevance_score": settings.rag_min_relevance_score,
        "kb_index_path": settings.kb_index_path,
        "chroma_persist_dir": settings.chroma_persist_dir,
    }
    threshold_payload = dict(thresholds)
    online_min_score = settings.rag_min_relevance_score
    online_threshold_match = (
        not hard_threshold_overridden
        and min_score == online_min_score
        and min_score_hard == online_min_score
    )
    threshold_payload.update({
        "applied_min_score": min_score,
        "applied_min_score_hard": min_score_hard,
        "hard_threshold_overridden": hard_threshold_overridden,
        "hard_threshold_policy": (
            "explicit_override_non_online"
            if hard_threshold_overridden else "same_as_min_score"
        ),
        "online_threshold_match": online_threshold_match,
        "online_min_score": online_min_score,
    })
    fingerprint_payload = {
        "dataset": {
            "path": str(ds),
            "sha256": sha256_of(ds) if ds.exists() else "",
            "num_cases": num_cases,
        },
        "top_k": top_k,
        "variant": variant,
        "retrieval": retrieval,
        "thresholds": threshold_payload,
    }
    raw = json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"))
    return {
        "protocol": "retrieval-eval-v1",
        "frozen": True,
        **fingerprint_payload,
        "git": {"commit": context["git_commit"]},
        "runtime": {
            "python": platform.python_version(),
            "deps_sha256": context["deps_sha256"],
        },
        "config_hash": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    }
