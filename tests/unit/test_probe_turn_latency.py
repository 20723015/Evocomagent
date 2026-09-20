"""延迟探针（P3-2）纯逻辑测试：分位数、分层抽样、预算建议（无网络）。

探针本身要真实 LLM 调用（属人工复核入口），但其口径计算必须可确定性验证——
分位数算错会让「校准依据」本身失真。
"""

from __future__ import annotations

from app.evaluation.dataset import EvalCase
from app.scripts.probe_turn_latency import _percentile, _stats, recommend, sample_cases


def test_percentile_matches_linear_interpolation():
    values = [1.0, 2.0, 3.0, 4.0]
    assert _percentile(values, 0) == 1.0
    assert _percentile(values, 100) == 4.0
    assert _percentile(values, 50) == 2.5          # (4-1)*0.5 = 1.5 → 2 + 0.5*(3-2)
    assert _percentile([5.0], 95) == 5.0           # 单点
    assert _percentile([], 95) == 0.0              # 空集不炸


def test_percentile_p95_within_range():
    values = [float(i) for i in range(1, 101)]
    p95 = _percentile(values, 95)
    assert 95.0 <= p95 <= 96.0


def test_stats_reports_n_and_extremes():
    stat = _stats([10.0, 20.0, 30.0])
    assert stat["n"] == 3
    assert stat["p50_ms"] == 20.0
    assert stat["max_ms"] == 30.0
    assert stat["mean_ms"] == 20.0
    assert _stats([]) == {"n": 0}


def _case(case_id: str, intent: str) -> EvalCase:
    return EvalCase(id=case_id, description="", turns=["x"], expected_intent=intent)


def test_sample_cases_is_stratified_and_deterministic():
    cases = [
        _case("a1", "order_query"), _case("a2", "order_query"),
        _case("a3", "order_query"), _case("b1", "complaint"),
        _case("b2", "complaint"), _case("c1", "greeting"),
    ]
    picked = sample_cases(cases, per_intent=2)
    ids = [c.id for c in picked]
    # 每层最多 per_intent 条；层间按意图名排序 → 确定性
    assert ids == ["b1", "b2", "c1", "a1", "a2"]
    assert sample_cases(cases, per_intent=2) == picked  # 可复现
    # 层内不足时取全部
    assert [c.id for c in sample_cases(cases, per_intent=1)] == ["b1", "c1", "a1"]


def test_sample_cases_buckets_unspecified_intent():
    cases = [_case("x1", None), _case("x2", None)]
    picked = sample_cases(cases, per_intent=5)
    assert [c.id for c in picked] == ["x1", "x2"]


def test_recommend_uses_formula():
    report = {
        "overall": {"p95_ms": 10000.0},
        "react_steps": {"max_ms": 3.0},
    }
    rec = recommend(report, max_react_steps=8)
    assert rec["llm_call_p95_seconds"] == 10.0
    assert rec["required_budget_seconds"] == 80.0
    assert rec["observed_max_react_steps"] == 3.0
    assert "max_react_steps" in rec["formula"]


def test_recommend_handles_empty_sample():
    rec = recommend({"overall": {}, "react_steps": {}}, max_react_steps=8)
    assert rec["llm_call_p95_seconds"] == 0.0
    assert rec["required_budget_seconds"] == 0.0
