"""仓库级知识文档治理检查（RAG 修复计划·4）：CI 阻断新增文档缺元数据。

扫描可索引范围的 md/txt（frontmatter）与其它支持格式（sidecar），
逐项校验 status/authority/effective_date；任一违规 → 退出码 2。

用法：
    python -m app.scripts.check_kb_governance
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.agent.rag import governance
from app.agent.rag.loader import read_text
from app.agent.rag.parsers import SUPPORTED_SUFFIXES
from app.config.settings import settings
from app.observability.logging import get_logger

log = get_logger("app.scripts.check_kb_governance")

ROOT = Path(__file__).resolve().parent.parent.parent


def check(kb_dir: Path) -> list[str]:
    problems: list[str] = []
    for path in sorted(kb_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(kb_dir).as_posix()
        if not governance.is_indexable(rel):
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        text = ""
        if suffix in (".md", ".txt"):
            text = read_text(path)
        meta = governance.metadata_for(path, rel, text)
        try:
            governance.validate_metadata(rel, meta)
        except governance.DocumentGovernanceError as e:
            problems.append(str(e))
    return problems


def main(argv: list[str] | None = None) -> int:
    kb_dir = ROOT / settings.kb_dir
    if not kb_dir.exists():
        log.error(f"知识库目录不存在: {kb_dir}")
        return 2
    problems = check(kb_dir)
    if problems:
        log.error(
            "知识文档治理检查失败（新增可索引文档必须带 "
            "status/authority/effective_date，或同名 .meta.yaml sidecar）："
        )
        for p in problems:
            log.error(f"  - {p}")
        return 2
    log.info("知识文档治理检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
