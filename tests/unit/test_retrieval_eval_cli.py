"""run_retrieval_eval CLI 纯函数测试：--exclude-id 过滤与校准约束留档。

review 修复（v3 收尾）：
- filter_cases_by_exclude：复现 v2 冻结 535 口径（排除 evolved 增补用例）
- build_retrieval_thresholds：校准约束/exclude ID 必须随报告与 manifest 留档
- archive_calibration_failure：校准失败也留档（gate 轮 dev-cal 引用无产物）
"""
from argparse import Namespace
from pathlib import Path

from app.scripts.run_retrieval_eval import (
    archive_calibration_failure,
    build_retrieval_thresholds,
    filter_cases_by_exclude,
)


def _case(cid: str) -> dict:
    return {"id": cid, "query": cid, "expected": ["x.md"], "k": 5, "tags": ["easy"]}


def test_multi_query_overlay_arg_parsed():
    """回归：main() 读 args.multi_query_overlay，解析器必须定义该参数。

    修复前 build_arg_parser 缺 --multi-query-overlay → 任何一次 CLI 运行都会在
    加载 overlay 处 AttributeError；此处锁定默认值与显式传参两种形态。
    """
    from app.scripts.run_retrieval_eval import build_arg_parser

    parser = build_arg_parser()
    default = parser.parse_args([])
    assert default.multi_query_overlay == ""  # 默认不启用（falsy 才走既有分支）
    explicit = parser.parse_args([
        "--multi-query-overlay",
        "app/evaluation/retrieval_cases_v3_multihop_queries.json",
    ])
    assert explicit.multi_query_overlay == (
        "app/evaluation/retrieval_cases_v3_multihop_queries.json"
    )


def test_filter_exclude_basic():
    cases = [_case(f"c{i}") for i in range(5)]
    kept, missing = filter_cases_by_exclude(cases, ["c1", "c3"])
    assert [c["id"] for c in kept] == ["c0", "c2", "c4"]
    assert missing == set()


def test_filter_exclude_reports_missing_id():
    cases = [_case("c0"), _case("c1")]
    kept, missing = filter_cases_by_exclude(cases, ["c0", "no_such_id"])
    assert [c["id"] for c in kept] == ["c1"]
    assert missing == {"no_such_id"}


def test_filter_exclude_noop_without_args():
    cases = [_case("c0"), _case("c1")]
    kept, missing = filter_cases_by_exclude(cases, [])
    assert kept == cases and missing == set()
    kept2, _ = filter_cases_by_exclude(cases, None)
    assert kept2 == cases


def test_thresholds_plain_without_calibrate():
    args = Namespace(
        min_positive_recall=0.95, min_easy_recall=0.98, min_hard_recall=0.80,
        min_mrr=0.90, min_ndcg=0.90, min_negative_rejection=0.90,
        calibrate=False, calibrate_min_positive_recall=0.98, exclude_id=None,
    )
    t = build_retrieval_thresholds(args)
    assert t["min_positive_recall"] == 0.95
    assert "calibrate_min_positive_recall" not in t
    assert "excluded_case_ids" not in t


def test_thresholds_calibration_persisted():
    args = Namespace(
        min_positive_recall=0.95, min_easy_recall=0.98, min_hard_recall=0.80,
        min_mrr=0.90, min_ndcg=0.90, min_negative_rejection=0.90,
        calibrate=True, calibrate_min_positive_recall=0.90,
        exclude_id=["retrieval_evolved_slo_01", "retrieval_evolved_measure_install_01"],
    )
    t = build_retrieval_thresholds(args)
    assert t["calibrate_min_positive_recall"] == 0.90
    assert t["calibrate_min_negative_rejection"] == 0.80
    assert t["excluded_case_ids"] == [
        "retrieval_evolved_measure_install_01", "retrieval_evolved_slo_01",
    ]


def test_thresholds_calibration_without_exclude():
    args = Namespace(
        min_positive_recall=0.95, min_easy_recall=0.98, min_hard_recall=0.80,
        min_mrr=0.90, min_ndcg=0.90, min_negative_rejection=0.90,
        calibrate=True, calibrate_min_positive_recall=0.92, exclude_id=None,
    )
    t = build_retrieval_thresholds(args)
    assert t["calibrate_min_positive_recall"] == 0.92
    assert "excluded_case_ids" not in t


def _cal_args(json_out, cal_min=0.90):
    return Namespace(
        min_positive_recall=0.95, min_easy_recall=0.98, min_hard_recall=0.80,
        min_mrr=0.90, min_ndcg=0.90, min_negative_rejection=0.90,
        calibrate=True, calibrate_min_positive_recall=cal_min, exclude_id=None,
        json_out=json_out, variant="hybrid-rerank", top_k=5,
    )


def test_archive_calibration_failure_writes_payload(tmp_path):
    """review 修复：校准失败必须留最小失败记录（约束/SHA/原因）。"""
    dataset = tmp_path / "cases.json"
    dataset.write_text('{"cases": []}', encoding="utf-8")
    out = tmp_path / "runs" / "dev-cal" / "report.json"
    args = _cal_args(str(out))

    path = archive_calibration_failure(
        args, dataset, [], ValueError("正负例分数不可分：负例拒绝率仅 66.9%")
    )

    assert path == out and out.exists()
    payload = __import__("json").loads(out.read_text(encoding="utf-8"))
    assert payload["calibration_failed"] is True
    assert payload["calibrate_min_positive_recall"] == 0.90
    assert payload["calibrate_min_negative_rejection"] == 0.80
    assert "正负例分数不可分" in payload["error"]
    assert payload["dataset"]["num_cases"] == 0
    assert len(payload["dataset"]["sha256"]) == 64


def test_archive_calibration_failure_noop_without_json_out():
    """未配置 --json-out 时不写任何文件（保持旧行为）。"""
    args = _cal_args(None)
    path = archive_calibration_failure(
        args, Path("whatever.json"), [], ValueError("x")
    )
    assert path is None
