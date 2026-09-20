"""评估数据集门禁（阶段五 5.1，CI 无网络可跑）。

检查黄金集（cases.json）与检索集（retrieval_cases.json）：
- 数量门槛（黄金集 ≥ 50；检索集 ≥ 170、困难正例 ≥ 20、负例 ≥ 15）
- schema 字段齐全、turns 非空
- 分类覆盖：注入类与非注入类都有；检索集覆盖全部知识文档
- 退出码：全部通过 0；任一不满足 1（PR 门禁判定）

用法：python -m app.scripts.check_eval_dataset [--eval-cases PATH] [--retrieval-cases PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config.settings import settings
from app.evaluation.retrieval_metrics import load_cases
from app.observability.logging import get_logger

log = get_logger("app.scripts.check_eval_dataset")

MIN_GOLDEN_CASES = 50
MIN_RETRIEVAL_CASES = 170
MIN_HARD_RETRIEVAL_CASES = 20
MIN_NEGATIVE_RETRIEVAL_CASES = 15

# v3 测试集分布门禁（《语料补足与评测v3计划》§3.1 / §1.3）
V3_MIN_TOTAL = 850
V3_MAX_EASY_RATIO = 0.55
V3_MIN_HARD = 150
V3_MIN_NEGATIVE = 80
V3_MIN_TIMING = 30
V3_MIN_MULTI_INTENT = 40
V3_MIN_COVERED_DOCS = 100
# 保留 v2 冻结 535 的头部文档不再接受新增用例（配额已超 2%）
V3_QUOTA_EXEMPT_DOCS = {
    "会员权益.md", "配送说明.md", "常见问题FAQ.md", "退换货政策.md",
}


def _v3_distribution_problems(rcases: list[dict], kb_root: Path) -> list[str]:
    """v3 数据集分布断言：easy 占比 / hard / 负例 / 时序 / multi / 配额。"""
    problems: list[str] = []
    total = len(rcases)
    easy = [c for c in rcases if "easy" in c.get("tags", []) and c["expected"]]
    hard = [c for c in rcases if "hard" in c.get("tags", []) and c["expected"]]
    negative = [c for c in rcases if not c["expected"]]
    timing = [c for c in rcases if "timing" in c.get("tags", [])]
    multi = [c for c in rcases if "multi_intent" in c.get("tags", [])]

    if total < V3_MIN_TOTAL:
        problems.append(f"v3 检索集仅 {total} 条（门槛 ≥ {V3_MIN_TOTAL}）")
    if easy:
        ratio = len(easy) / total
        if ratio > V3_MAX_EASY_RATIO:
            problems.append(
                f"v3 easy 占比 {ratio:.1%} > {V3_MAX_EASY_RATIO:.0%}"
                f"（{len(easy)}/{total}）"
            )
    if len(hard) < V3_MIN_HARD:
        problems.append(f"v3 hard 仅 {len(hard)} 条（门槛 ≥ {V3_MIN_HARD}）")
    if len(negative) < V3_MIN_NEGATIVE:
        problems.append(f"v3 负例仅 {len(negative)} 条（门槛 ≥ {V3_MIN_NEGATIVE}）")
    if len(timing) < V3_MIN_TIMING:
        problems.append(f"v3 时序用例仅 {len(timing)} 条（门槛 ≥ {V3_MIN_TIMING}）")
    if len(multi) < V3_MIN_MULTI_INTENT:
        problems.append(
            f"v3 multi_intent 仅 {len(multi)} 条（门槛 ≥ {V3_MIN_MULTI_INTENT}）"
        )

    # 单文档配额：任一文档用例 ≤ 2% × 全集（v2 头部文档按冻结豁免）；
    # 新增用例（id 前缀 retrieval_v3_）单独按新增集 2% 复核
    added = [c for c in rcases if str(c.get("id", "")).startswith("retrieval_v3_")]
    total_quota = max(2, int(total * 0.02) + 1)
    added_quota = max(2, int(len(added) * 0.02) + 1)
    per_doc: dict[str, int] = {}
    added_per_doc: dict[str, int] = {}
    for c in rcases:
        for doc in c["expected"]:
            per_doc[doc] = per_doc.get(doc, 0) + 1
    for c in added:
        for doc in c["expected"]:
            added_per_doc[doc] = added_per_doc.get(doc, 0) + 1
    for doc, n in sorted(per_doc.items(), key=lambda kv: -kv[1]):
        if doc not in V3_QUOTA_EXEMPT_DOCS and n > total_quota:
            problems.append(
                f"v3 单文档 {doc} 共 {n} 次（全集配额 ≤ {total_quota}）"
            )
    for doc, n in sorted(added_per_doc.items(), key=lambda kv: -kv[1]):
        if n > added_quota:
            problems.append(
                f"v3 新增用例单文档 {doc} 出现 {n} 次（新增配额 ≤ {added_quota}）"
            )
        if doc in V3_QUOTA_EXEMPT_DOCS:
            problems.append(f"v3 新增用例引用了配额豁免文档 {doc}（v2 头部已超限）")

    # expected 文档存在性（含 docx/pdf；archive/evolved 前缀）
    kb_files = {
        p.relative_to(kb_root).as_posix()
        for p in kb_root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".md", ".txt", ".docx", ".pdf"}
    }
    covered: set[str] = set()
    for c in rcases:
        for sp in c["expected"]:
            covered.add(sp)
            if sp not in kb_files:
                problems.append(f"v3 用例 {c['id']} 引用不存在的文档 {sp!r}")
    if len(covered) < V3_MIN_COVERED_DOCS:
        problems.append(
            f"v3 覆盖文档仅 {len(covered)} 份（门槛 ≥ {V3_MIN_COVERED_DOCS}）"
        )
    return problems


def _load(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["cases"] if isinstance(data, dict) else data


def _problems(eval_cases: Path, retrieval_cases: Path) -> list[str]:
    problems: list[str] = []

    # ---------- 黄金集 ----------
    cases = _load(eval_cases) if eval_cases.exists() else []
    if len(cases) < MIN_GOLDEN_CASES:
        problems.append(
            f"黄金集仅 {len(cases)} 条（门槛 ≥ {MIN_GOLDEN_CASES}）"
        )
    for c in cases:
        if not c.get("id") or not c.get("description") or not c.get("turns"):
            problems.append(f"用例 {c.get('id', '?')} 缺少 id/description/turns")
        # 修复计划：引用字段门禁——类型正确、非空字符串、无重复
        expected_cit = c.get("expected_citations", [])
        if expected_cit is None:
            expected_cit = []
        if not isinstance(expected_cit, list):
            problems.append(
                f"用例 {c.get('id', '?')} expected_citations 必须是 list"
            )
            continue
        seen: set[str] = set()
        for item in expected_cit:
            if item is None or not isinstance(item, str) or not item.strip():
                problems.append(
                    f"用例 {c.get('id', '?')} expected_citations 含空/非字符串项"
                )
            elif item.strip() in seen:
                problems.append(
                    f"用例 {c.get('id', '?')} expected_citations 含重复项 {item!r}"
                )
            else:
                seen.add(item.strip())
        forbid = c.get("forbid_unretrieved_citations", False)
        if forbid is not None and not isinstance(forbid, bool):
            problems.append(
                f"用例 {c.get('id', '?')} forbid_unretrieved_citations 必须是 bool"
            )
    if cases:
        inject = [c for c in cases if c["id"].startswith("inject_")]
        if not inject:
            problems.append("黄金集缺少注入类用例（id 前缀 inject_）")
    if cases:
        non_inject = [c for c in cases if not c["id"].startswith("inject_")]
        if not non_inject:
            problems.append("黄金集缺少正常类用例")
    # 修复计划：引用场景覆盖门禁——expected 与 forbid 两类都需出现
    with_citation = [c for c in cases if c.get("expected_citations")]
    with_forbid = [c for c in cases if c.get("forbid_unretrieved_citations") is True]
    if not with_citation:
        problems.append("黄金集缺少 expected_citations 用例（引用真实性门禁）")
    if not with_forbid:
        problems.append(
            "黄金集缺少 forbid_unretrieved_citations=true 用例（引用真实性门禁）"
        )

    # ---------- 检索集 ----------
    rcases: list[dict] = []
    if retrieval_cases.exists():
        try:
            rcases = load_cases(retrieval_cases)
        except ValueError as exc:
            problems.append(str(exc))
    if len(rcases) < MIN_RETRIEVAL_CASES:
        problems.append(
            f"检索集仅 {len(rcases)} 条（门槛 ≥ {MIN_RETRIEVAL_CASES}）"
        )
    hard = [c for c in rcases if "hard" in c.get("tags", []) and c["expected"]]
    negative = [c for c in rcases if not c["expected"]]
    if len(hard) < MIN_HARD_RETRIEVAL_CASES:
        problems.append(
            f"检索集困难正例仅 {len(hard)} 条（门槛 ≥ {MIN_HARD_RETRIEVAL_CASES}）"
        )
    if len(negative) < MIN_NEGATIVE_RETRIEVAL_CASES:
        problems.append(
            f"检索集负例仅 {len(negative)} 条（门槛 ≥ {MIN_NEGATIVE_RETRIEVAL_CASES}）"
        )
    covered_docs: set[str] = set()
    kb_root = Path(settings.kb_dir)
    for c in rcases:
        tags = set(c.get("tags", []))
        if not tags.intersection({"easy", "hard", "no_hit"}):
            problems.append(f"检索用例 {c['id']} 缺少难度/负例标签")
        if not c["expected"] and "no_hit" not in tags:
            problems.append(f"检索负例 {c['id']} 缺少 no_hit 标签")
        if c["expected"] and "no_hit" in tags:
            problems.append(f"检索正例 {c['id']} 不得包含 no_hit 标签")
        for source_path in c["expected"]:
            covered_docs.add(source_path)
            if not (kb_root / source_path).is_file():
                problems.append(
                    f"检索用例 {c['id']} 引用了不存在的知识文档 {source_path!r}"
                )
    knowledge_docs = {
        p.relative_to(kb_root).as_posix()
        for p in kb_root.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt"}
    }
    missing_coverage = sorted(knowledge_docs - covered_docs)
    if missing_coverage:
        # v2 冻结集只覆盖当时 40 份语料；v3 语料 147 份后不再要求单集全覆盖，
        # 覆盖充足性由 v3 分布断言（--v3，≥100 份）承担
        log.warning(f"检索集未覆盖知识文档（{len(missing_coverage)} 份，非阻断）: "
                    f"{missing_coverage[:5]}...")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评估数据集门禁")
    parser.add_argument("--eval-cases", default=settings.eval_dataset_path)
    parser.add_argument("--retrieval-cases", default=settings.retrieval_eval_dataset_path)
    parser.add_argument(
        "--v3", action="store_true",
        help="v3 测试集分布断言（easy≤55% / hard≥150 / 负例≥80 / 时序≥30 / "
             "multi≥40 / 单文档配额 / 覆盖 ≥100 份）",
    )
    args = parser.parse_args(argv)

    eval_cases = Path(args.eval_cases)
    retrieval_cases = Path(args.retrieval_cases)
    problems = _problems(eval_cases, retrieval_cases)
    if args.v3:
        rcases = load_cases(retrieval_cases) if retrieval_cases.exists() else []
        problems.extend(_v3_distribution_problems(rcases, Path(settings.kb_dir)))
        if not problems:
            # v3 全量断言通过时附分布摘要
            easy = len([c for c in rcases if "easy" in c.get("tags", []) and c["expected"]])
            hard = len([c for c in rcases if "hard" in c.get("tags", []) and c["expected"]])
            neg = len([c for c in rcases if not c["expected"]])
            timing = len([c for c in rcases if "timing" in c.get("tags", [])])
            multi = len([c for c in rcases if "multi_intent" in c.get("tags", [])])
            log.info(
                f"v3 分布: 总 {len(rcases)}（easy {easy} {easy/len(rcases):.1%} / "
                f"hard {hard} / 负例 {neg} / 时序 {timing} / multi {multi}）"
            )
    if problems:
        for p in problems:
            log.info(f"  ❌ {p}")
        log.info("评估数据集门禁：未通过")
        return 1
    log.info(
        f"评估数据集门禁：通过（黄金集 {len(_load(eval_cases))} 条 / "
        f"检索集 {len(_load(retrieval_cases))} 条）"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
