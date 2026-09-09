"""3.3 A/B 评测脚本逻辑单测（不调真实 LLM）。

覆盖：关键子集抽样、paired delta、分类通过率、Judge 不一致率、门禁判定、
配置应用（globals + retriever 重置）。
"""

from __future__ import annotations

import json

import pytest

from app.evaluation.dataset import EvalCase
from app.scripts.run_ab_eval import (
    BASELINE_CFG,
    CANDIDATE_CFG,
    KEY_CATEGORIES,
    AbConfig,
    _category_pass_rates,
    _evaluate_gates,
    _judge_disagreement_rate,
    _key_subset,
    _paired_delta,
    _security_regressions,
)


def _case(cid: str) -> EvalCase:
    return EvalCase(id=cid, description="d", turns=["t"])


def _report(passed: list[tuple[str, bool]], *, scores=None, tokens=None,
            aq=None) -> dict:
    """构造与 evaluator 输出同构的报告。"""
    cases = []
    for i, (cid, ok) in enumerate(passed):
        score = (scores or {}).get(cid, 0.8)
        token = (tokens or {}).get(cid, 100)
        cases.append({
            "case_id": cid, "passed": ok,
            "process": {"token_cost": token, "token_pass": True,
                        "process_score": 0.8, "tool_accuracy": 1.0},
            "result": {"result_score": score, "answer_quality": (aq or {}).get(cid)},
            "trace": {"num_tool_calls": 2},
            "security": {"critical_gate_pass": None},
            "error": None,
        })
    return {
        "summary": {
            "pass_rate": (sum(1 for _, ok in passed if ok) / len(passed)
                          if passed else 0),
            "distributions": {
                "tool_calls": {"mean": 2.0, "median": 2, "p90": 2, "p95": 2, "n": len(passed)},
                "tokens": {"mean": 100.0, "median": 100, "p90": 100, "p95": 100, "n": len(passed)},
            },
        },
        "cases": cases,
    }


def test_key_subset_all_key_categories_plus_sample():
    cases = [_case(f"inject_{i}") for i in range(2)] \
        + [_case(f"abuse_{i}") for i in range(13)] \
        + [_case(f"complaint_{i}") for i in range(6)] \
        + [_case(f"rag_no_hit_{i}") for i in range(5)] \
        + [_case(f"kb_cross_{i}") for i in range(5)] \
        + [_case(f"foo_{i}") for i in range(10)] \
        + [_case(f"bar_{i}") for i in range(10)]
    sub = _key_subset(cases)
    ids = [c.id for c in sub]
    for cat in KEY_CATEGORIES:
        assert any(i.startswith(cat) for i in ids), cat  # rag→rag_no_hit* 前缀匹配
    foos = [i for i in ids if i.startswith("foo_")]
    bars = [i for i in ids if i.startswith("bar_")]
    rag = [i for i in ids if i.startswith("rag_")]
    kb = [i for i in ids if i.startswith("kb_")]
    assert len(foos) == 3 and len(bars) == 3  # 非关键类别固定种子抽 3
    assert len(rag) == 5 and len(kb) == 5  # 关键类别全部保留
    # 固定种子确定性
    assert [c.id for c in _key_subset(cases)] == [c.id for c in sub]


def test_paired_delta():
    before = _report([("a", True), ("b", False)], scores={"a": 0.8, "b": 0.5})
    after = _report([("a", False), ("b", True)], scores={"a": 0.4, "b": 0.9})
    deltas = _paired_delta(before, after)
    by_id = {d["case_id"]: d for d in deltas}
    assert by_id["a"]["passed_before"] is True
    assert by_id["a"]["passed_after"] is False
    assert by_id["a"]["result_score_delta"] == -0.4
    assert by_id["b"]["passed_before"] is False
    assert by_id["b"]["passed_after"] is True


def test_category_pass_rates():
    rep = _report([("inject_1", True), ("inject_2", False),
                   ("abuse_1", True), ("foo_1", False)])
    rates = _category_pass_rates(rep)
    assert rates["inject"] == 0.5
    assert rates["abuse"] == 1.0
    assert rates["foo"] == 0.0


def test_judge_disagreement_rate():
    rep = _report([("a", True), ("b", True), ("c", False)],
                  aq={"a": 0.9, "b": 0.2, "c": 0.8})
    d = _judge_disagreement_rate(rep)
    assert d["n"] == 3
    assert d["disagreement_rate"] == pytest.approx(0.6667, abs=1e-3)  # b/c 与规则判定不一致


def test_security_regressions_flags_critical_failures():
    rep = _report([("abuse_1", True)])
    rep["cases"][0]["security"] = {"critical_gate_pass": False}
    bad = _security_regressions(rep, KEY_CATEGORIES)
    assert "abuse_1:critical_gate" in bad


def test_security_regressions_compare_key_categories_to_baseline():
    baseline = _report([
        ("inject_1", True), ("inject_2", True),
        ("abuse_1", True), ("complaint_1", True),
    ])
    candidate = _report([
        ("inject_1", True), ("inject_2", False),
        ("abuse_1", True), ("complaint_1", True),
    ])
    bad = _security_regressions(
        candidate, ("inject", "abuse", "complaint"), baseline,
    )
    assert any(item.startswith("inject:security_rate_regression") for item in bad)
    # key_cat is authoritative; an unrelated category must not create a result.
    assert not any(item.startswith("foo:") for item in bad)


def test_gates_require_candidate_security_rate_not_below_baseline():
    baseline = _report([
        ("inject_1", True), ("inject_2", True),
        ("abuse_1", True), ("complaint_1", True),
        ("foo_1", True), ("foo_2", True), ("foo_3", True), ("foo_4", True),
        ("foo_5", True), ("foo_6", True),
    ])
    candidate = _report([
        ("inject_1", True), ("inject_2", False),
        ("abuse_1", True), ("complaint_1", True),
        ("foo_1", True), ("foo_2", True), ("foo_3", True), ("foo_4", True),
        ("foo_5", True), ("foo_6", True),
    ])
    g = _evaluate_gates(baseline, candidate, sec_fails=[])
    assert g["categories_ok"] is False
    assert g["pass"] is False


def test_gates_require_critical_security_rate_100_percent():
    report = _report([
        ("inject_1", True), ("abuse_1", True), ("complaint_1", True),
        ("foo_1", True), ("foo_2", True), ("foo_3", True),
        ("foo_4", True), ("foo_5", True), ("foo_6", True), ("foo_7", True),
    ])
    report["summary"]["security"] = {
        "critical_total": 2, "critical_passed": 1,
    }
    g = _evaluate_gates(report, report, sec_fails=[])
    assert g["critical_rate"] == 0.5
    assert g["critical_ok"] is False
    assert g["pass"] is False


def test_gates_evaluation():
    b = _report([("inject_1", True), ("abuse_1", True), ("complaint_1", True),
                 ("foo_1", True), ("foo_2", True), ("foo_3", True),
                 ("foo_4", True), ("foo_5", True), ("foo_6", True), ("foo_7", True)])
    # 10 条：overall 0.9，关键类别 injection 1.0/abuse 1.0/complaint 1.0
    g = _evaluate_gates(b, b, sec_fails=[])
    assert g["pass"] is True
    # 安全回退 → 不过
    g2 = _evaluate_gates(b, b, sec_fails=["abuse_1:critical_gate"])
    assert g2["pass"] is False and g2["security_ok"] is False
    # 综合过低 → 不过（构造 pass_rate 0.5；无关键类别时 categories_ok 仍需整体判定）
    low = _report([("a", True), ("b", False)])
    g3 = _evaluate_gates(low, low, sec_fails=[])
    assert g3["overall_ok"] is False


def test_ab_config_apply_globals(monkeypatch, reset_settings):
    from app.agent.tools import knowledge as knowledge_tool
    from app.config.settings import settings

    BASELINE_CFG.apply_globals()
    assert settings.rag_hybrid is False
    assert settings.rag_rerank == "none"
    assert settings.tool_call_guard_enabled is False
    assert settings.guardrails_enabled is False
    assert settings.enforce_order_ownership is True  # 安全底线不消融

    CANDIDATE_CFG.apply_globals()
    assert settings.rag_hybrid is True
    assert settings.rag_rerank == "bge-reranker-v2-m3"
    assert settings.tool_call_guard_enabled is True
    assert settings.guardrails_enabled is True
    assert settings.enforce_order_ownership is True


def test_ab_configs_frozen_shape():
    # min_score 为 dev 校准后冻结的检索阈值（run_ab_eval 顶部常量），属配置指纹的一部分
    assert BASELINE_CFG.to_dict() == {
        "hybrid": False, "rerank": "none",
        "tool_guard_enabled": False, "guardrails_enabled": False,
        "min_score": BASELINE_CFG.min_score,
        "enforce_order_ownership": True,
    }
    assert CANDIDATE_CFG.to_dict() == {
        "hybrid": True, "rerank": "bge-reranker-v2-m3",
        "tool_guard_enabled": True, "guardrails_enabled": True,
        "min_score": CANDIDATE_CFG.min_score,
        "enforce_order_ownership": True,
    }
    assert BASELINE_CFG.min_score is not None
    assert CANDIDATE_CFG.min_score is not None
