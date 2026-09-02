"""dedup.py：精确 / 预去重 0.95 / 最终去重 0.9 / 本轮互查（第10期）。

- 相似度比较统一用 `>= threshold - 1e-4` 容差，避免浮点边界误判。
- Embedding 不可用的处理在 pipeline 层：dry-run 报 dedup_unavailable，
  正式模式在写文件前退出。
"""

from __future__ import annotations

import math
from typing import Sequence

TOLERANCE = 1e-4
PASS_KEYS = ("quality_score", "source_score", "confidence")


def ge_with_tolerance(value: float, threshold: float) -> bool:
    return value >= threshold - TOLERANCE


def exact(candidate_id: str, ledger) -> bool:
    """精确去重：candidate_id 已在 processed / published / rejected / pending。"""
    return ledger.contains(candidate_id)


def pre_dedup(question: str, retriever, threshold: float) -> bool:
    """预去重：原始脱敏问题与已有知识检索，top1 余弦 ≥ threshold 视为重复。"""
    hits = retriever.search(question, top_k=1)
    if not hits:
        return False
    return ge_with_tolerance(hits[0].score, threshold)


def final_dedup(question: str, answer: str, retriever, threshold: float):
    """最终去重：规范化问题、答案分别检索，任一 top1 余弦 ≥ 阈值 → 返回 (命中, 命中侧)。

    返回 (hit, side) 而非 bool：side ∈ {"question", "answer"}，未命中为 (None, "")。
    调用方据此区分近重复的性质——问题侧命中 evolved/ 旧沉淀且新候选质量分更高
    → 走替换（P2-1）；答案侧命中（回答模板化，问题未必相同）与人工文档命中
    → 一律按重复丢弃。
    """
    for side, text in (("question", question), ("answer", answer)):
        hits = retriever.search(text, top_k=1)
        if hits and ge_with_tolerance(hits[0].score, threshold):
            return hits[0], side
    return None, ""


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _rank(meta: dict, keys: tuple[str, ...]) -> tuple[float, ...]:
    """得分元组（负数化后越小越好）；缺失键按 0 处理。"""
    return tuple(-float(meta.get(k) or 0.0) for k in keys)


def in_run_pairwise(
    vectors: list[list[float]],
    metas: list[dict],
    threshold: float = 0.9,
    score_keys: tuple[str, ...] = PASS_KEYS,
) -> list[int]:
    """本轮互查：批量两两余弦，重复时保留得分更高者（平局保留先到者）。

    返回被淘汰的下标列表。
    """
    n = len(vectors)
    dropped: set[int] = set()
    for i in range(n):
        if i in dropped:
            continue
        for j in range(i + 1, n):
            if j in dropped:
                continue
            if not ge_with_tolerance(_cosine(vectors[i], vectors[j]), threshold):
                continue
            if _rank(metas[j], score_keys) < _rank(metas[i], score_keys):
                dropped.add(i)
                break
            dropped.add(j)
    return sorted(dropped)