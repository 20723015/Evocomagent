"""检索质量评估指标（阶段七 7.6，检索回归门禁基础）。

配合 7.1-7.4 的检索改造（embedder / 后端切换 / generation 感知单例）使用：
评估的检索器通过 app.agent.tools.knowledge._get_retriever() 取「线上同款」实例，
检索本身统一走 retriever_factory.final_search（3 倍候选 ≤15 → 阈值门控 →
parent_id 去重 → Top-K），与线上 search_knowledge 完全同口径。

指标定义（同一文档的多个 chunk 按首次出现位置折叠）：
- recall@k  ：前 k 个命中里覆盖的期望文档数 / 期望文档总数。
- MRR       ：首个命中期 望文档位置的倒数（1-based），衡量「第一个结果就得对」。
- nDCG@k    ：binary relevance（命中期望 = 1）的归一化折损累计增益，
              理想列按前 min(k, len(expected)) 位全命中的 DCG 计算。
- negative rejection：expected=[] 时，经相关度阈值过滤后没有任何候选即为正确拒绝。

评测报告逐例记录：原始候选数、父块折叠后的命中键、被去重的 parent 数。
口径变化（引入统一 final_search）后，历史阈值与报告不可直接对比，需在
dev 集重新校准并通过后再冻结 holdout。

评估集格式（retrieval_cases.json）：
    {"cases": [{"id": str, "query": str, "expected": [source_path, ...], "k": 5}]}
expected 是期望命中的文档 source_path 列表；空列表表示知识库没有答案。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

_REQUIRED_CASE_FIELDS = ("id", "query", "expected")


def _case_key(hit) -> str:
    """从一条检索命中（RetrievedChunk）取用于比对的文档键。

    优先 source_path（相对 kb_dir 的 posix 路径，如 "退换货政策.md"），
    为空（旧索引）时回退 doc 字段，保证两种索引形态都能比对。
    """
    return hit.chunk.source_path or hit.chunk.doc


def load_cases(path) -> list[dict]:
    """加载检索评估用例集。

    格式: {"cases": [{"id", "query", "expected": [source_path...], "k": 5}]}
    文件缺失/不可读/JSON 损坏/字段缺失 一律抛 ValueError（CLI 捕获后以码 1 退出）。
    """
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"检索用例集无法读取: {path}（{e}）") from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"检索用例集 JSON 解析失败: {path}（{e}）") from e

    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError(
            f"检索用例集格式错误: {path}（顶层需为 {{\"cases\": [...]}}）"
        )

    cases: list[dict] = []
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    for i, item in enumerate(data["cases"]):
        if not isinstance(item, dict):
            raise ValueError(f"检索用例集格式错误: 第 {i} 条用例不是 JSON 对象")
        missing = [f for f in _REQUIRED_CASE_FIELDS if f not in item]
        if missing:
            raise ValueError(
                f"检索用例集格式错误: 第 {i} 条用例缺少字段 {missing}"
            )
        case_id = item["id"]
        query = item["query"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"检索用例集格式错误: 第 {i} 条用例 id 需为非空字符串")
        if case_id in seen_ids:
            raise ValueError(f"检索用例集格式错误: id 重复 {case_id!r}")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"检索用例集格式错误: 第 {i} 条用例 query 需为非空字符串")
        normalized_query = query.strip()
        if normalized_query in seen_queries:
            raise ValueError(f"检索用例集格式错误: query 重复 {normalized_query!r}")

        expected = item["expected"]
        if not isinstance(expected, list) or not all(
            isinstance(s, str) and s for s in expected
        ):
            raise ValueError(
                f"检索用例集格式错误: 第 {i} 条用例 expected 需为字符串列表"
            )
        if len(set(expected)) != len(expected):
            raise ValueError(f"检索用例集格式错误: 第 {i} 条用例 expected 含重复项")
        k = item.get("k", 5)
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError(f"检索用例集格式错误: 第 {i} 条用例 k 需为正整数")
        tags = item.get("tags", [])
        if not isinstance(tags, list) or not all(
            isinstance(tag, str) and tag.strip() for tag in tags
        ) or len(set(tags)) != len(tags):
            raise ValueError(
                f"检索用例集格式错误: 第 {i} 条用例 tags 需为无重复的非空字符串列表"
            )
        seen_ids.add(case_id)
        seen_queries.add(normalized_query)
        cases.append(item)
    return cases


# ============================================================
# 纯函数指标
# ============================================================
def recall_at_k(hit_keys: list[str], expected_keys: list[str], k: int) -> float:
    """前 k 个命中里覆盖的期望文档数 / 期望总数。

    k <= 0 或 expected 为空时返回 0.0（空期望无从谈覆盖，不记分母为 0 的脏分）。
    """
    if k <= 0 or not expected_keys:
        return 0.0
    ranked = list(dict.fromkeys(hit_keys))
    covered = sum(1 for key in set(expected_keys) if key in ranked[:k])
    return covered / len(expected_keys)


def mrr(hit_keys: list[str], expected_keys: list[str]) -> float:
    """首个命中期望文档位置的倒数（1-based）；整体无命中返回 0.0。

    强调「第一个结果就得对」：位置越靠前权重越大，适合衡量首答即准。
    """
    expected = set(expected_keys)
    for i, key in enumerate(dict.fromkeys(hit_keys), start=1):
        if key in expected:
            return 1.0 / i
    return 0.0


def ndcg_at_k(hit_keys: list[str], expected_keys: list[str], k: int) -> float:
    """二元相关性的归一化折损累计增益（前 k 位）。

    relevance = 命中期望则为 1，否则 0；DCG = Σ rel_i / log2(i+1)（i 从 1 起）。
    理想 DCG 按前 min(k, len(expected)) 位全为 1 计算（即使命中键在列表里
    实际重复也只算一次期望）。k <= 0 或 expected 为空返回 0.0。
    """
    if k <= 0 or not expected_keys:
        return 0.0
    expected = set(expected_keys)
    ranked = list(dict.fromkeys(hit_keys))
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, key in enumerate(ranked[:k], start=1)
        if key in expected
    )
    ideal = sum(
        1.0 / math.log2(i + 1) for i in range(1, min(k, len(expected)) + 1)
    )
    return dcg / ideal if ideal > 0 else 0.0


# ============================================================
# 批量评估
# ============================================================
def _mean(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _unique_keys(hits) -> list[str]:
    """按首次出现顺序折叠同一文档的多个 chunk。"""
    keys: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        key = _case_key(hit)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _positive_summary(results: list[dict]) -> dict:
    return {
        "cases": len(results),
        "recall_at_k": _mean(r["recall_at_k"] for r in results),
        "mrr": _mean(r["mrr"] for r in results),
        "ndcg_at_k": _mean(r["ndcg_at_k"] for r in results),
    }


def calibrate_threshold(
    positive_cases: list[dict],
    negative_queries: list[str],
    retriever,
    *,
    top_k: int = 5,
    min_positive_recall: float = 0.98,
    min_negative_rejection: float = 0.80,
) -> dict:
    """为当前检索配置选择满足正例召回约束的最高分数阈值。

    校准集必须与正式困难题/负例分离。检索走与线上一致的 final_search
    统一口径（此处 min_score=None 取全候选，阈值在候选上扫描）；若原始
    检索或分数分布无法同时满足约束则抛 ValueError，避免提交拍脑袋阈值。
    """
    from app.agent.rag.retriever import collapse_by_parent, filter_hits_by_score
    from app.agent.rag.retriever_factory import final_search

    def _final(hits: list, k: int, threshold: float) -> list:
        """与 final_search 相同的门控→折叠→截断顺序（扫阈值时复用候选）。"""
        return collapse_by_parent(filter_hits_by_score(hits, threshold), k)

    if not positive_cases or not negative_queries:
        raise ValueError("阈值校准需要非空正例和负例")

    positive_runs: list[tuple[list, list[str], int]] = []
    negative_runs: list[list] = []
    scores: list[float] = []
    for case in positive_cases:
        k = int(case.get("k") or top_k)
        outcome = final_search(retriever, case["query"], k)
        positive_runs.append((outcome.raw_hits, list(case["expected"]), k))
        scores.extend(hit.score for hit in outcome.raw_hits)
    for query in negative_queries:
        outcome = final_search(retriever, query, top_k)
        negative_runs.append(outcome.raw_hits)
        scores.extend(hit.score for hit in outcome.raw_hits)
    if not scores:
        raise ValueError("阈值校准未获得任何候选分数")

    candidates = sorted(set(scores))
    candidates.append(max(scores) + 1e-12)
    feasible: list[tuple[float, float, float]] = []
    for threshold in candidates:
        positive_recall = _mean(
            recall_at_k(
                _unique_keys(_final(hits, k, threshold)), expected, k
            )
            for hits, expected, k in positive_runs
        )
        negative_rejection = _mean(
            not _final(hits, top_k, threshold) for hits in negative_runs
        )
        if positive_recall >= min_positive_recall:
            feasible.append((threshold, positive_recall, negative_rejection))

    if not feasible:
        raise ValueError(
            f"原始检索正例 recall@k 无法达到 {min_positive_recall:.1%}"
        )
    threshold, positive_recall, negative_rejection = max(feasible, key=lambda row: row[0])
    if negative_rejection < min_negative_rejection:
        raise ValueError(
            "正负例分数不可分：保持正例 recall@k "
            f"{positive_recall:.1%} 时负例拒绝率仅 {negative_rejection:.1%}"
        )
    return {
        "threshold": threshold,
        "positive_recall_at_k": positive_recall,
        "negative_rejection_rate": negative_rejection,
        "positive_cases": len(positive_runs),
        "negative_cases": len(negative_runs),
    }


def evaluate(
    cases: list[dict], retriever, top_k: int = 5,
    min_score: float | None = None,
    min_score_hard: float | None = None,
) -> dict:
    """逐用例跑检索并聚合 recall@k / MRR / nDCG@k。

    每条用例经 retriever_factory.final_search 统一口径检索（3 倍候选 ≤15 →
    阈值门控 → parent_id 去重 → Top-K），与线上 search_knowledge 完全同源；
    比对键取 chunk.source_path（空则 chunk.doc），指标口径 k 取用例自带的
    "k"（缺省回落 top_k）。返回逐例诊断信息（含原始候选数、折叠后命中键、
    被去重的 parent 数），以及 positive/easy/hard/negative 分层汇总。负例的
    三个 ranking 指标为 None，只进入 rejection_rate，避免与正例平均值混算。

    min_score / min_score_hard：相关度阈值分 easy/hard 两档。hard 用例默认
    使用与线上一致的 ``min_score``；只有调用方显式传入 ``min_score_hard``
    才采用非线上口径。这样评测不会在 hard 集悄悄绕过线上相关度门控。
    """
    from app.agent.rag.retriever_factory import final_search

    results: list[dict] = []
    for case in cases:
        k = int(case.get("k") or top_k)
        threshold = (
            min_score_hard if (
                "hard" in case.get("tags", []) and min_score_hard is not None
            ) else min_score
        )
        outcome = final_search(retriever, case["query"], k, min_score=threshold)
        hits = outcome.hits
        hit_keys = _unique_keys(hits)
        expected = list(case["expected"])
        is_negative = not expected
        result = {
            "id": case["id"],
            "query": case["query"],
            "expected": expected,
            "tags": list(case.get("tags", [])),
            "threshold_applied": threshold,
            "hit_keys": hit_keys,
            # 统一口径诊断：原始候选数 / 阈值过滤后候选数 / 折叠后命中键 /
            # 被去重的 parent 数
            "raw_candidates": outcome.raw_candidates,
            "gated_candidates": outcome.gated_candidates,
            "collapsed_hit_keys": outcome.kept_keys,
            "collapsed_parents": outcome.collapsed_parents,
            "raw_top_score": outcome.raw_top_score,
            "accepted_hits": len(hits),
            "negative_rejected": is_negative and not hits,
            "recall_at_k": None if is_negative else recall_at_k(hit_keys, expected, k),
            "mrr": None if is_negative else mrr(hit_keys, expected),
            "ndcg_at_k": None if is_negative else ndcg_at_k(hit_keys, expected, k),
        }
        results.append(result)

    positives = [r for r in results if r["expected"]]
    negatives = [r for r in results if not r["expected"]]
    hard = [r for r in positives if "hard" in r["tags"]]
    easy = [r for r in positives if "easy" in r["tags"]]
    negative_rejection = _mean(r["negative_rejected"] for r in negatives)

    return {
        "cases": results,
        "summary": {
            "cases": len(results),
            "positive": _positive_summary(positives),
            "easy": _positive_summary(easy),
            "hard": _positive_summary(hard),
            "negative": {
                "cases": len(negatives),
                "rejection_rate": negative_rejection,
                "false_accept_rate": 1.0 - negative_rejection if negatives else 0.0,
            },
        },
    }
