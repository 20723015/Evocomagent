"""知识时效治理（阶段六 6.4）：文档 frontmatter 扫描。

统一元数据规范（7.1）：source/owner/effective_date/version/provenance。
本模块扫描知识库 markdown，收集 effective_date 早于阈值（或缺失）的文档，
供审核后台展示与知识清理决策。
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional


def parse_frontmatter(text: str) -> dict:
    """极简 frontmatter 解析（key: value，与 skills loader 思路一致）。

    publisher 输出：provenance/owner；人工文档可含 effective_date/version。
    """
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return {}
    out: dict = {}
    for line in match.group(1).splitlines():
        if ":" in line and not line.startswith(" "):
            key, _, value = line.partition(":")
            value = value.strip().strip("'\"")
            if value:
                out[key.strip()] = value
    return out


def _parse_date(value: str) -> Optional[date]:
    try:
        return datetime.fromisoformat(value.strip()).date()
    except (ValueError, TypeError):
        try:
            return datetime.strptime(value.strip()[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return None


def scan_expired_knowledge(kb_dir: Path, aging_days: int) -> list[dict]:
    """返回时效标注列表（expired / valid / unknown）。

    - 有 effective_date 且早于「今天 - aging_days」→ expired；
    - 在有效期内 → valid；
    - 无 effective_date → unknown（人工文档默认标注缺失，提示补全）。
    """
    from datetime import timedelta

    threshold = date.today()
    cutoff = threshold - timedelta(days=aging_days)

    out: list[dict] = []
    for f in sorted(kb_dir.rglob("*.md")):
        try:
            meta = parse_frontmatter(f.read_text(encoding="utf-8"))
        except OSError:
            continue
        effective = _parse_date(meta.get("effective_date", ""))
        if effective is None:
            out.append({
                "path": str(f.relative_to(kb_dir)), "effective_date": "",
                "owner": meta.get("owner", ""), "status": "unknown",
            })
        elif effective < cutoff:
            out.append({
                "path": str(f.relative_to(kb_dir)),
                "effective_date": effective.isoformat(),
                "owner": meta.get("owner", ""), "status": "expired",
            })
        else:
            out.append({
                "path": str(f.relative_to(kb_dir)),
                "effective_date": effective.isoformat(),
                "owner": meta.get("owner", ""), "status": "valid",
            })
    return out
