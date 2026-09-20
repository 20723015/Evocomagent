#!/usr/bin/env python
"""切分分布报告（RAG 切分与文档处理优化方案 P0 验收工具）。

不依赖 embedding / 向量后端：只跑「解析 → 切分」并统计块分布，因此可在索引
重建前离线判断切分策略的效果（真检索质量仍须走 run_retrieval_eval.py 的
959 例配置级 A/B 门禁）。

用法：
  # 当前配置的块分布
  python app/scripts/report_chunk_distribution.py

  # 优化前后并排对比（关掉 rag_parent_merge / rag_prefix_dedup 作为基线）
  python app/scripts/report_chunk_distribution.py --compare

  # 留档（artifacts/ 或 CI 产物目录）
  python app/scripts/report_chunk_distribution.py --compare --json-out artifacts/eval/chunk_dist.json

口径：
- 「检索块长」= Chunk.text 全长（含前缀），「正文长」= 去掉前缀行后的正文；
- 「父块覆盖率」= 有 parent_text 的块占比（P0-1 前现网为 1%）；
- 「前缀」= 检索块首行长度（P0-2 前缀去重的直接指标）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.rag.parsers import chunk_kb_dir  # noqa: E402
from app.agent.rag.chunker import chunk_body  # noqa: E402
from app.config.settings import settings  # noqa: E402
from app.observability.logging import get_logger  # noqa: E402

log = get_logger("app.scripts.report_chunk_distribution")


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return float(ordered[idx])


def collect(kb_dir: Path) -> dict:
    """按当前 settings 切分并汇总分布指标。"""
    chunks = chunk_kb_dir(kb_dir, strict=False)
    texts = [c.text for c in chunks]
    bodies = [chunk_body(t) for t in texts]
    prefixes = [len(t.split("\n", 1)[0]) + 1 for t in texts]  # +1 = 前缀换行
    parents = [c.parent_text for c in chunks if c.parent_text]
    units = {c.parent_id for c in chunks if c.parent_id}
    docs = {c.source_path for c in chunks}
    return {
        "chunks": len(chunks),
        "docs": len(docs),
        "chunk_chars_mean": round(statistics.mean([len(t) for t in texts]), 1),
        "chunk_chars_median": _percentile([len(t) for t in texts], 0.5),
        "body_chars_mean": round(statistics.mean([len(b) for b in bodies]), 1),
        "body_chars_p90": _percentile([len(b) for b in bodies], 0.9),
        "body_lt_200_ratio": round(
            sum(1 for b in bodies if len(b) < 200) / len(bodies), 4
        ) if bodies else 0.0,
        "prefix_chars_mean": round(statistics.mean(prefixes), 1) if prefixes else 0.0,
        "prefix_chars_median": _percentile(prefixes, 0.5),
        "parent_coverage": round(len(parents) / len(chunks), 4) if chunks else 0.0,
        "parent_chars_median": _percentile([len(p) for p in parents], 0.5),
        "parent_chars_max": max((len(p) for p in parents), default=0),
        "generation_units": len(units),
        "units_per_doc_mean": round(len(units) / len(docs), 2) if docs else 0.0,
        "chunks_with_index_text": sum(1 for c in chunks if c.index_text),
        "table_chunks": sum(1 for t in texts if "| --- " in t or "\n| " in t),
    }


def _override(**kwargs):
    """临时覆盖切分开关（返回恢复函数）——settings 变更不影响进程外。"""
    saved = {k: getattr(settings, k) for k in kwargs}
    for key, value in kwargs.items():
        setattr(settings, key, value)

    def restore() -> None:
        for key, value in saved.items():
            setattr(settings, key, value)

    return restore


_KEYS = [
    ("chunks", "块数"),
    ("docs", "文档数"),
    ("chunk_chars_mean", "检索块均长(含前缀)"),
    ("chunk_chars_median", "检索块中位长"),
    ("body_chars_mean", "正文均长"),
    ("body_chars_p90", "正文 p90"),
    ("body_lt_200_ratio", "<200字块占比"),
    ("prefix_chars_mean", "前缀均长"),
    ("prefix_chars_median", "前缀中位长"),
    ("parent_coverage", "父块覆盖率"),
    ("parent_chars_median", "父块中位长"),
    ("parent_chars_max", "父块最长"),
    ("generation_units", "生成单元数"),
    ("units_per_doc_mean", "单元/文档"),
    ("chunks_with_index_text", "带索引输入块数"),
    ("table_chunks", "含表格块数"),
]


def _print_single(stats: dict) -> None:
    width = max(len(label) for _, label in _KEYS)
    for key, label in _KEYS:
        log.info(f"  {label:<{width}} : {stats[key]}")


def _print_compare(baseline: dict, current: dict) -> None:
    width = max(len(label) for _, label in _KEYS)
    log.info(f"  {'指标':<{width}} | {'优化前':>12} | {'优化后':>12} | 变化")
    log.info("  " + "-" * (width + 38))
    for key, label in _KEYS:
        before, after = baseline[key], current[key]
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            delta = after - before
            sign = "+" if delta > 0 else ""
            change = f"{sign}{round(delta, 3)}"
        else:
            change = ""
        log.info(f"  {label:<{width}} | {before:>12} | {after:>12} | {change}")


def main() -> int:
    parser = argparse.ArgumentParser(description="切分分布报告（离线，无 embedding）")
    parser.add_argument("--kb-dir", default="", help="知识库目录（默认 settings.kb_dir）")
    parser.add_argument("--compare", action="store_true",
                        help="并排对比：关闭 rag_parent_merge/rag_prefix_dedup 作为优化前基线")
    parser.add_argument("--json-out", default="", help="把指标写入 JSON（留档）")
    args = parser.parse_args()

    kb_dir = Path(args.kb_dir) if args.kb_dir else ROOT / settings.kb_dir
    if not kb_dir.exists():
        log.info(f"❌ 知识库目录不存在: {kb_dir}")
        return 1

    log.info("=" * 68)
    log.info("  切分分布报告")
    log.info(f"  源目录: {kb_dir}")
    log.info(f"  配置  : parent_merge={settings.rag_parent_merge} "
             f"prefix_dedup={settings.rag_prefix_dedup} "
             f"contextual={settings.rag_contextual_index}")
    log.info("=" * 68)

    baseline = None
    if args.compare:
        restore = _override(rag_parent_merge=False, rag_prefix_dedup=False)
        try:
            baseline = collect(kb_dir)
        finally:
            restore()
    current = collect(kb_dir)

    if baseline is not None:
        _print_compare(baseline, current)
    else:
        _print_single(current)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"kb_dir": str(kb_dir), "baseline": baseline, "current": current},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        log.info(f"  指标已写入: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
