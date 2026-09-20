"""修复计划·五：拒答校准脚本 CLI 测试（--help / dev 冻结 / holdout 缺文件 / 成功路径）。"""

from __future__ import annotations

import json

import pytest

from app.scripts import calibrate_rejection as cal


_SIGNALS = [
    {"id": "n1", "negative": True, "top1": 0.1, "gap": 0.0, "coverage": 0.0, "rerank": None},
    {"id": "n2", "negative": True, "top1": 0.2, "gap": 0.0, "coverage": 0.0, "rerank": None},
    {"id": "p1", "negative": False, "top1": 0.9, "gap": 0.3, "coverage": 0.9, "rerank": None},
    {"id": "p2", "negative": False, "top1": 0.8, "gap": 0.2, "coverage": 0.8, "rerank": None},
]

_CASES = [
    {"id": "n1", "query": "q1", "k": 5},
    {"id": "n2", "query": "q2", "k": 5},
    {"id": "p1", "query": "q3", "k": 5, "expected": ["doc1"]},
    {"id": "p2", "query": "q4", "k": 5, "expected": ["doc2"]},
]


def _patch(monkeypatch):
    monkeypatch.setattr(cal, "load_cases", lambda path: list(_CASES))
    monkeypatch.setattr(cal, "_collect_signals", lambda split: list(_SIGNALS))


def test_cli_help_exits_zero():
    with pytest.raises(SystemExit) as e:
        cal.main(["--help"])
    assert e.value.code == 0


def test_dev_freezes_params(monkeypatch, tmp_path):
    _patch(monkeypatch)
    out = tmp_path / "params.json"
    assert cal.main(["--dev", "--out", str(out)]) == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["holdout_used"] is False
    assert "params" in record and "dev_metrics" in record
    assert record["dataset_hash"]


def test_holdout_missing_freeze_file_returns_2(monkeypatch, tmp_path):
    _patch(monkeypatch)
    out = tmp_path / "missing.json"
    assert cal.main(["--holdout", "--out", str(out)]) == 2


def test_holdout_success_path(monkeypatch, tmp_path):
    _patch(monkeypatch)
    out = tmp_path / "params.json"
    assert cal.main(["--dev", "--out", str(out)]) == 0
    rc = cal.main(["--holdout", "--out", str(out)])
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["holdout_used"] is True
    assert "holdout_metrics" in record
    assert rc == 0


def test_holdout_rerun_blocked_without_force(monkeypatch, tmp_path):
    _patch(monkeypatch)
    out = tmp_path / "params.json"
    cal.main(["--dev", "--out", str(out)])
    cal.main(["--holdout", "--out", str(out)])
    assert cal.main(["--holdout", "--out", str(out)]) == 3  # 冻结纪律：拒绝复跑


# ============================================================
# A5：RRF 守卫（分数无语义时拒绝校准）
# ============================================================
def test_rrf_mode_guard_blocks_calibration(monkeypatch, tmp_path):
    """hybrid 且未挂精排 → 拒绝校准并非 0 退出，且不写冻结文件。

    回归背景：RRF 只依赖排名，分数量纲与余弦/精排分不可比；在其上网格搜索
    会静默冻结一批「看起来合理、实则无意义」的阈值，并被同步进 settings
    默认值。
    """
    from app.config.settings import settings

    _patch(monkeypatch)
    monkeypatch.setattr(settings, "rag_hybrid", True)
    monkeypatch.setattr(settings, "rag_rerank", "none")
    out = tmp_path / "params.json"

    rc = cal.main(["--dev", "--out", str(out)])

    assert rc == 2
    assert not out.exists(), "守卫未生效：RRF 模式下仍写出了冻结参数"


def test_guard_allows_calibration_with_reranker(monkeypatch, tmp_path):
    """挂了精排 → 分数有语义，校准照常进行。"""
    from app.config.settings import settings

    _patch(monkeypatch)
    monkeypatch.setattr(settings, "rag_hybrid", True)
    monkeypatch.setattr(settings, "rag_rerank", "bge-reranker-v2-m3")
    out = tmp_path / "params.json"

    assert cal.main(["--dev", "--out", str(out)]) == 0
    assert out.exists()


def test_guard_allows_pure_vector_calibration(monkeypatch, tmp_path):
    """纯向量（hybrid=false）分数有语义，不受守卫影响。"""
    from app.config.settings import settings

    _patch(monkeypatch)
    monkeypatch.setattr(settings, "rag_hybrid", False)
    monkeypatch.setattr(settings, "rag_rerank", "none")
    out = tmp_path / "params.json"

    assert cal.main(["--dev", "--out", str(out)]) == 0


def test_rrf_mode_single_source():
    """rrf_mode 判定单一来源：评测脚本与拒绝模块结果一致。"""
    from app.agent.rag.rejection import rrf_mode as prod_rrf_mode
    from app.scripts.run_retrieval_eval import rrf_mode as eval_rrf_mode

    assert eval_rrf_mode() == prod_rrf_mode()
