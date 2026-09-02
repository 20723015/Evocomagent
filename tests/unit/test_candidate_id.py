"""candidate ID：只依赖已落盘数据，跨 LLM 输出稳定。"""

from __future__ import annotations

from app.evolution.sanitizer import candidate_id_for_turn, legacy_candidate_id


def test_turn_candidate_id_deterministic():
    assert candidate_id_for_turn("t1") == candidate_id_for_turn("t1")
    assert candidate_id_for_turn("t1") != candidate_id_for_turn("t2")


def test_legacy_candidate_id_stable_across_runs():
    args = ("session", 3, "退货可以吗", "可以，运费自理", ["退货政策.md"])
    assert legacy_candidate_id(*args) == legacy_candidate_id(*args)


def test_legacy_candidate_id_sensitive_to_inputs():
    base = ("session", 3, "退货可以吗", "可以，运费自理", ["退货政策.md"])
    assert legacy_candidate_id(*base) != legacy_candidate_id(
        "session", 3, "换货可以吗", "可以，运费自理", ["退货政策.md"]
    )
    assert legacy_candidate_id(*base) != legacy_candidate_id(
        "session", 4, "退货可以吗", "可以，运费自理", ["退货政策.md"]
    )
    assert legacy_candidate_id(*base) != legacy_candidate_id(
        "session", 3, "退货可以吗", "可以，运费自理", ["退货政策.md", "配送说明.md"]
    )


def test_legacy_source_order_independent():
    """source_ids 排序后拼接 → 顺序无关。"""
    a = legacy_candidate_id("s", 1, "q", "a", ["b.md", "a.md"])
    b = legacy_candidate_id("s", 1, "q", "a", ["a.md", "b.md"])
    assert a == b