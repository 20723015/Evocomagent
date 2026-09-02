"""检索质量评估（阶段七 7.6）单测。

覆盖：
- 三个纯函数指标（recall@k / MRR / nDCG@k）：完美 / 部分 / 无命中、顺序影响、
  k<=0 与空期望等边界。
- load_cases 格式校验（合法 / 缺失 / 损坏 / 字段不全）。
- FakeEmbedder + FakeBackend + FakeRetriever 的 evaluate 集成（全程无网络）。
- 内置用例集（app/evaluation/retrieval_cases.json）完整性。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent.rag.chunker import Chunk
from app.agent.rag.backends.base import RetrievedChunk
from app.config.settings import settings
from app.evaluation.retrieval_metrics import (
    calibrate_threshold,
    evaluate,
    load_cases,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

from conftest import FakeBackend, FakeEmbedder, FakeRetriever

ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# recall@k
# ============================================================
@pytest.mark.parametrize(
    "hit_keys,expected,k,want",
    [
        (["退换货政策.md", "配送说明.md"], ["退换货政策.md"], 5, 1.0),  # 完美命中
        (["退换货政策.md", "配送说明.md"], ["退换货政策.md", "会员权益.md"], 5, 0.5),  # 部分命中
        (["配送说明.md", "会员权益.md"], ["退换货政策.md"], 5, 0.0),  # 无命中
        (["配送说明.md", "退换货政策.md"], ["退换货政策.md"], 1, 0.0),  # 命中在第 2 位，超出 k
        ([], ["退换货政策.md"], 5, 0.0),  # 空结果
    ],
)
def test_recall_at_k(hit_keys, expected, k, want):
    assert recall_at_k(hit_keys, expected, k) == pytest.approx(want)


def test_recall_at_k_edges():
    assert recall_at_k(["退换货政策.md"], ["退换货政策.md"], 0) == 0.0  # k<=0
    assert recall_at_k(["退换货政策.md"], [], 5) == 0.0  # 空期望
    assert recall_at_k(["退换货政策.md"], ["退换货政策.md"], -1) == 0.0


# ============================================================
# MRR
# ============================================================
@pytest.mark.parametrize(
    "hit_keys,expected,want",
    [
        (["退换货政策.md", "配送说明.md"], ["退换货政策.md"], 1.0),  # 首位即中
        (["配送说明.md", "退换货政策.md"], ["退换货政策.md"], 0.5),  # 次位命中
        (["配送说明.md", "会员权益.md"], ["退换货政策.md"], 0.0),  # 无命中
        ([], ["退换货政策.md"], 0.0),  # 空结果
        (["退换货政策.md"], [], 0.0),  # 空期望
    ],
)
def test_mrr(hit_keys, expected, want):
    assert mrr(hit_keys, expected) == pytest.approx(want)


def test_mrr_first_hit_position_matters():
    """同一命中集合，期望文档位置越靠前 MRR 越高。"""
    first = mrr(["退换货政策.md", "配送说明.md"], ["退换货政策.md"])
    second = mrr(["配送说明.md", "退换货政策.md"], ["退换货政策.md"])
    assert first == 1.0
    assert second == pytest.approx(0.5)
    assert second < first


# ============================================================
# nDCG@k
# ============================================================
def test_ndcg_perfect_hit():
    assert ndcg_at_k(
        ["退换货政策.md", "配送说明.md"], ["退换货政策.md"], 5
    ) == pytest.approx(1.0)
    # 两个期望都在前 2 位 → 满分
    assert ndcg_at_k(
        ["退换货政策.md", "配送说明.md"],
        ["退换货政策.md", "配送说明.md"], 5,
    ) == pytest.approx(1.0)


def test_ndcg_order_sensitivity():
    """同样命中/期望组合，期望文档排在首位时 nDCG 更高。"""
    first = ndcg_at_k(["退换货政策.md", "配送说明.md"], ["退换货政策.md"], 5)
    second = ndcg_at_k(["配送说明.md", "退换货政策.md"], ["退换货政策.md"], 5)
    assert first == pytest.approx(1.0)
    assert second == pytest.approx(1.0 / math.log2(3))
    assert second < first


def test_ndcg_multi_expected_partial():
    """两个期望只命中一个（在首位）：dcg / 理想（两个全中的 DCG）。"""
    got = ndcg_at_k(
        ["退换货政策.md", "会员权益.md"],
        ["退换货政策.md", "配送说明.md"], 5,
    )
    ideal = 1.0 + 1.0 / math.log2(3)
    assert got == pytest.approx(1.0 / ideal)


def test_ndcg_edges():
    assert ndcg_at_k(["退换货政策.md"], ["退换货政策.md"], 0) == 0.0  # k<=0
    assert ndcg_at_k(["退换货政策.md"], [], 5) == 0.0  # 空期望
    assert ndcg_at_k(["会员权益.md"], ["退换货政策.md"], 5) == 0.0  # 无命中
    assert ndcg_at_k(["退换货政策.md", "会员权益.md"], ["退换货政策.md"], 1) == pytest.approx(1.0)  # 首位即中


def test_metrics_collapse_duplicate_chunks_from_same_document():
    hits = ["退换货政策.md", "退换货政策.md", "配送说明.md"]
    assert mrr(hits, ["配送说明.md"]) == pytest.approx(0.5)
    assert ndcg_at_k(hits, ["退换货政策.md"], 5) == pytest.approx(1.0)


# ============================================================
# load_cases 格式校验
# ============================================================
def test_load_cases_ok(tmp_path):
    p = tmp_path / "cases.json"
    p.write_text(json.dumps({
        "cases": [
            {"id": "c1", "query": "七天无理由退货可以吗", "expected": ["退换货政策.md"], "k": 5},
            {"id": "c2", "query": "偏远地区包邮吗", "expected": ["配送说明.md"]},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    cases = load_cases(p)
    assert len(cases) == 2
    assert cases[0]["id"] == "c1"
    assert cases[0]["expected"] == ["退换货政策.md"]
    assert cases[0]["k"] == 5
    assert cases[1].get("k", 5) == 5  # 缺省 k 由 evaluate 回落 top_k


def test_load_cases_missing_file(tmp_path):
    with pytest.raises(ValueError, match="无法读取"):
        load_cases(tmp_path / "missing.json")


def test_load_cases_corrupt_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON 解析失败"):
        load_cases(p)


def test_load_cases_wrong_shape(tmp_path):
    """顶层不是 {"cases": [...]}。"""
    p = tmp_path / "shape.json"
    p.write_text(json.dumps({"items": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="格式错误"):
        load_cases(p)


def test_load_cases_missing_required_field(tmp_path):
    p = tmp_path / "field.json"
    p.write_text(json.dumps({
        "cases": [{"id": "c1", "query": "七天无理由退货可以吗"}],
    }, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="expected"):
        load_cases(p)


def test_load_cases_bad_expected(tmp_path):
    """expected 必须是字符串列表。"""
    p = tmp_path / "exp.json"
    p.write_text(json.dumps({
        "cases": [{"id": "c1", "query": "q", "expected": "退换货政策.md"}],
    }, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="expected"):
        load_cases(p)


def test_load_cases_accepts_negative_case(tmp_path):
    p = tmp_path / "negative.json"
    p.write_text(json.dumps({
        "cases": [{"id": "n1", "query": "线下门店在哪", "expected": [], "tags": ["no_hit"]}],
    }, ensure_ascii=False), encoding="utf-8")
    assert load_cases(p)[0]["expected"] == []


@pytest.mark.parametrize(
    "case,match",
    [
        ({"id": "dup", "query": "q", "expected": ["a.md", "a.md"]}, "重复"),
        ({"id": "bad-k", "query": "q", "expected": [], "k": 0}, "正整数"),
        ({"id": "bad-tags", "query": "q", "expected": [], "tags": [""]}, "tags"),
    ],
)
def test_load_cases_rejects_invalid_case_fields(tmp_path, case, match):
    p = tmp_path / "invalid.json"
    p.write_text(json.dumps({"cases": [case]}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_cases(p)


# ============================================================
# evaluate 集成（FakeEmbedder + FakeBackend + FakeRetriever）
# ============================================================
def _build_fake_retriever() -> FakeRetriever:
    """两个知识库 chunk：退换货政策 / 配送说明（FakeEmbedder 按共享字符算相似度）。"""
    embedder = FakeEmbedder()
    backend = FakeBackend()
    chunks = [
        Chunk(
            chunk_id="c1", doc="退换货政策", section="七天无理由",
            text="七天无理由退货 退换货政策 退货退款流程",
            source_path="退换货政策.md",
        ),
        Chunk(
            chunk_id="c2", doc="配送说明", section="运费规则",
            text="偏远地区配送 运费说明 满九十九包邮",
            source_path="配送说明.md",
        ),
    ]
    backend.upsert(chunks, embedder.encode([c.text for c in chunks]), embedding_model=embedder.model)
    return FakeRetriever(embedder, backend)


def test_evaluate_with_fake_retriever():
    retriever = _build_fake_retriever()
    cases = [
        {"id": "r1", "query": "七天无理由退货可以吗", "expected": ["退换货政策.md"], "k": 5},
        {"id": "r2", "query": "偏远地区包邮吗", "expected": ["配送说明.md"], "k": 5},
        {"id": "r3", "query": "今天天气怎么样", "expected": ["天气查询.md"], "k": 5},  # 无对应文档
    ]
    report = evaluate(cases, retriever, top_k=5)

    # 三条都是正例：两中一空 → 2/3
    assert report["summary"]["cases"] == 3
    assert report["summary"]["positive"]["recall_at_k"] == pytest.approx(2 / 3)
    assert report["summary"]["positive"]["mrr"] == pytest.approx(2 / 3)
    assert report["summary"]["positive"]["ndcg_at_k"] == pytest.approx(2 / 3)
    assert report["summary"]["negative"]["cases"] == 0

    # 逐用例：期望文档能被命中，且字段齐全
    by_id = {c["id"]: c for c in report["cases"]}
    assert set(by_id) == {"r1", "r2", "r3"}
    for cid in ("r1", "r2"):
        assert by_id[cid]["recall_at_k"] == 1.0
        assert by_id[cid]["mrr"] == 1.0
        assert by_id[cid]["ndcg_at_k"] == 1.0
    assert by_id["r3"]["recall_at_k"] == 0.0
    assert by_id["r3"]["mrr"] == 0.0
    assert by_id["r3"]["ndcg_at_k"] == 0.0


def test_evaluate_negative_cases_use_filtered_empty_results():
    retriever = _build_fake_retriever()
    cases = [{
        "id": "n1", "query": "今天天气怎么样", "expected": [],
        "k": 5, "tags": ["no_hit"],
    }]
    without_threshold = evaluate(cases, retriever, min_score=None)
    assert without_threshold["summary"]["negative"]["rejection_rate"] == 0.0

    rejected = evaluate(cases, retriever, min_score=1.1)
    assert rejected["summary"]["negative"]["rejection_rate"] == 1.0
    assert rejected["cases"][0]["negative_rejected"] is True
    assert rejected["cases"][0]["recall_at_k"] is None


def test_hard_cases_default_to_the_online_threshold():
    """未显式覆盖 hard 阈值时，hard 不得绕过线上相关度过滤。"""
    class ScoredRetriever:
        def search(self, query, top_k=5):
            return [RetrievedChunk(
                chunk=Chunk(
                    chunk_id="hard", doc="退换货政策", section="s", text="t",
                    source_path="退换货政策.md",
                ),
                score=0.2,
            )]

    report = evaluate(
        [{"id": "hard-1", "query": "口语化问题", "expected": ["退换货政策.md"],
          "tags": ["hard"]}],
        ScoredRetriever(), min_score=0.5,
    )
    case = report["cases"][0]
    assert case["threshold_applied"] == 0.5
    assert case["accepted_hits"] == 0
    assert case["recall_at_k"] == 0.0


def test_retrieval_manifest_contains_dataset_hash_and_thresholds(tmp_path, reset_settings):
    from app.evaluation.manifest import build_retrieval_manifest

    ds = tmp_path / "retrieval.json"
    ds.write_text('{"cases": []}', encoding="utf-8")
    manifest = build_retrieval_manifest(
        dataset_path=str(ds), num_cases=0, top_k=5,
        min_score=0.4, min_score_hard=0.2,
        thresholds={"min_mrr": 0.9}, variant="hybrid-rerank",
        hard_threshold_overridden=True,
    )
    assert manifest["protocol"] == "retrieval-eval-v1"
    assert manifest["dataset"]["sha256"]
    assert manifest["git"]["commit"]
    assert manifest["retrieval"]["backend"] == settings.rag_backend
    assert manifest["thresholds"]["applied_min_score"] == 0.4
    assert manifest["thresholds"]["applied_min_score_hard"] == 0.2
    assert manifest["thresholds"]["hard_threshold_overridden"] is True
    assert manifest["thresholds"]["online_threshold_match"] is False
    assert manifest["config_hash"]


def test_calibrate_threshold_keeps_positive_and_rejects_negative():
    class ScoredRetriever:
        def search(self, query, top_k=5):
            score = 0.9 if query == "answerable" else 0.2
            doc = "退换货政策.md" if query == "answerable" else "配送说明.md"
            return [RetrievedChunk(
                chunk=Chunk(chunk_id=query, doc=doc, section="s", text="t", source_path=doc),
                score=score,
            )]

    result = calibrate_threshold(
        [{"id": "p", "query": "answerable", "expected": ["退换货政策.md"], "k": 5}],
        ["unknown"],
        ScoredRetriever(),
        min_positive_recall=1.0,
        min_negative_rejection=1.0,
    )
    assert result["threshold"] == pytest.approx(0.9)
    assert result["positive_recall_at_k"] == 1.0
    assert result["negative_rejection_rate"] == 1.0


def test_quality_gate_requires_threshold_for_negative_cases():
    from app.scripts.run_retrieval_eval import quality_failures

    report = {"summary": {
        "positive": {"cases": 1, "recall_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0},
        "easy": {"cases": 1, "recall_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0},
        "hard": {"cases": 1, "recall_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0},
        "negative": {"cases": 1, "rejection_rate": 1.0, "false_accept_rate": 0.0},
    }}
    args = SimpleNamespace(
        min_positive_recall=0.95, min_easy_recall=0.98, min_hard_recall=0.8, min_mrr=0.85,
        min_ndcg=0.85, min_negative_rejection=0.8, min_score=None,
    )
    assert "未配置" in quality_failures(report, args)[0]
    args.min_score = 0.5
    assert quality_failures(report, args) == []


def test_evaluate_falls_back_to_doc_when_source_path_empty():
    """旧索引 chunk 无 source_path 时，比对键回退 doc 字段。"""
    embedder = FakeEmbedder()
    backend = FakeBackend()
    chunk = Chunk(
        chunk_id="c1", doc="退换货政策", section="七天无理由",
        text="七天无理由退货 退换货政策 退货退款流程",
    )  # source_path 缺省为空
    backend.upsert([chunk], embedder.encode([chunk.text]), embedding_model=embedder.model)
    retriever = FakeRetriever(embedder, backend)

    report = evaluate(
        [{"id": "legacy", "query": "七天无理由退货可以吗", "expected": ["退换货政策"]}],
        retriever, top_k=5,
    )
    assert report["cases"][0]["recall_at_k"] == 1.0
    assert report["cases"][0]["mrr"] == 1.0


# ============================================================
# 内置用例集完整性
# ============================================================
def test_builtin_caseset_complete():
    """仓库内置主集包含完整的普通、困难和负例切片。"""
    dataset_path = ROOT / "app" / "evaluation" / "retrieval_cases.json"
    assert dataset_path.exists(), f"用例集不存在: {dataset_path}"
    cases = load_cases(dataset_path)

    ids = [c["id"] for c in cases]
    assert len(ids) >= 170  # 2026-08-31 知识库扩容后主集 294 条，下限断言防缩水
    assert len(set(ids)) == len(ids), "用例 id 必须唯一"
    for c in cases:
        assert c["query"], f"{c['id']} 的 query 为空"
        assert all(isinstance(p, str) and p for p in c["expected"])
    assert sum("hard" in c.get("tags", []) for c in cases) >= 20
    assert sum(not c["expected"] for c in cases) == 15


def test_builtin_caseset_matches_generator():
    from app.scripts.generate_eval_data import build_retrieval_cases

    dataset_path = ROOT / "app" / "evaluation" / "retrieval_cases.json"
    assert load_cases(dataset_path) == build_retrieval_cases(None)
