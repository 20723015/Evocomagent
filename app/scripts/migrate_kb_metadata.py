"""KB 文档元数据迁移（RAG 修复计划·4）。

为可索引的 md/txt 文档补齐 `status` / `authority` / `effective_date`
frontmatter；已存在的键保留，只补缺失项。非 md/txt 由上传链路写 sidecar。

用法：
    python -m app.scripts.migrate_kb_metadata --dry-run
    python -m app.scripts.migrate_kb_metadata --date 2026-09-12
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.agent.rag import governance
from app.agent.rag.loader import parse_frontmatter, serialize_frontmatter
from app.config.settings import settings
from app.observability.logging import get_logger

log = get_logger("app.scripts.migrate_kb_metadata")

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_STATUS = "active"
DEFAULT_AUTHORITY = "platform"


def _migrate_text(raw: str, effective_date: str) -> tuple[str, bool]:
    meta, body = parse_frontmatter(raw)
    changed = False
    if not str(meta.get("status", "") or "").strip():
        meta["status"] = DEFAULT_STATUS
        changed = True
    if not str(meta.get("authority", "") or "").strip():
        meta["authority"] = DEFAULT_AUTHORITY
        changed = True
    if not str(meta.get("effective_date", "") or "").strip():
        meta["effective_date"] = effective_date
        changed = True
    if not changed:
        return raw, False
    # frontmatter 块 + 正文（原无 frontmatter 时 body == raw）
    return serialize_frontmatter(meta) + body.lstrip("\n"), True


def migrate(kb_dir: Path, effective_date: str, dry_run: bool = False) -> dict:
    migrated: list[str] = []
    skipped: list[str] = []
    for path in sorted(kb_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(kb_dir).as_posix()
        if not governance.is_indexable(rel):
            continue
        if path.suffix.lower() not in (".md", ".txt"):
            continue
        raw = path.read_text(encoding="utf-8")
        updated, changed = _migrate_text(raw, effective_date)
        if not changed:
            skipped.append(rel)
            continue
        migrated.append(rel)
        if not dry_run:
            path.write_text(updated, encoding="utf-8")
    return {"migrated": migrated, "skipped": skipped}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KB 文档元数据迁移")
    parser.add_argument("--kb-dir", default=None)
    parser.add_argument("--date", default="2026-09-12",
                        help="effective_date（本轮审核发布日期，YYYY-MM-DD）")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    kb_dir = Path(args.kb_dir) if args.kb_dir else ROOT / settings.kb_dir
    result = migrate(kb_dir, args.date, dry_run=args.dry_run)
    log.info(
        f"{'[dry-run] ' if args.dry_run else ''}"
        f"迁移 {len(result['migrated'])} 篇，跳过 {len(result['skipped'])} 篇"
    )
    for rel in result["migrated"][:50]:
        log.info(f"  + {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
