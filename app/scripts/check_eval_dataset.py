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
        problems.append(f"检索集未覆盖知识文档: {missing_coverage}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评估数据集门禁")
    parser.add_argument("--eval-cases", default=settings.eval_dataset_path)
    parser.add_argument("--retrieval-cases", default=settings.retrieval_eval_dataset_path)
    args = parser.parse_args(argv)

    eval_cases = Path(args.eval_cases)
    retrieval_cases = Path(args.retrieval_cases)
    problems = _problems(eval_cases, retrieval_cases)
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
