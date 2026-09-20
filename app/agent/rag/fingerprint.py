"""RAG 索引配置指纹（RAG 修复计划·3 + 批次7）。

索引 `_meta` 记录 embedding provider/model/dimensions 与整体检索配置指纹；
运行时逐项比对，旧索引缺字段或指纹不一致 → 要求重建（fail-closed）。
"""

from __future__ import annotations

import hashlib
import json

from app.config.settings import settings

# 解析 / 分块版本（批次7）——**改动解析器或分块器输出时必须 bump 本常量**，
# 否则机制只生效一次。指纹原先只覆盖 embedding/hybrid/rerank：改完切分逻辑
# 后索引会静默停留在旧切分上、没有任何信号；纳入版本后 health 会明确报
# 「请重建索引」。约定只增不改，历史值不再复用。
PARSER_CHUNKER_VERSION = "2026-09-15.1"


def embedding_dimensions() -> int:
    provider = (settings.embedding_provider or "openai").lower()
    if provider == "sophnet":
        return int(settings.sophnet_embedding_dimensions or 0)
    return 0


def config_fingerprint() -> str:
    """影响检索语义的配置指纹（解析/分块版本 + 切分策略 + embedding + hybrid + rerank）。

    切分策略（rag_parent_merge / rag_prefix_dedup）与构建期上下文增强
    （rag_contextual_index + prompt 版本）都改变索引内容，必须进指纹：否则
    关掉开关后旧索引会被静默继续使用，或改了 prompt 后旧上下文被静默复用。
    """
    payload = {
        "parser_chunker_version": PARSER_CHUNKER_VERSION,
        "rag_parent_merge": bool(settings.rag_parent_merge),
        "rag_prefix_dedup": bool(settings.rag_prefix_dedup),
        "rag_contextual_index": bool(settings.rag_contextual_index),
        "rag_contextual_prompt_version": (
            settings.rag_contextual_prompt_version
            if settings.rag_contextual_index else ""
        ),
        "rag_contextual_model": (
            settings.rag_contextual_model if settings.rag_contextual_index else ""
        ),
        "embedding_provider": (settings.embedding_provider or "openai").lower(),
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": embedding_dimensions(),
        "rag_backend": (settings.rag_backend or "").lower(),
        "rag_hybrid": bool(settings.rag_hybrid),
        "rag_hybrid_recall_k": int(settings.rag_hybrid_recall_k or 0),
        "rag_rerank": (settings.rag_rerank or "none").lower(),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
