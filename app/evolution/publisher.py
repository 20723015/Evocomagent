"""publisher.py：Markdown 渲染、最终复扫、文件名、发布、unpublish（第10期）。

- render：固定模板（# 自进化知识 + ## 问题 + 回答）。
- 对渲染后全文再跑 sanitizer 复扫，命中 PII/注入 → 整条拒绝（不落原文）。
- 文件名：YYYYMMDD-<candidate_id[:12]>-<slug>.md。
- 两阶段发布：write_staging（写入 state/staging）→ pipeline 写 journal →
  move_into_kb（移入 knowledge/evolved/）；事务性由 pipeline 的 journal 保证。
- unpublish：移 trash + 返回相对路径（索引重建与 ledger 标记由调用方负责）。
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from app.config.settings import settings
from app.evolution.models import CandidateQA
from app.evolution.sanitizer import has_injection, has_pii, make_slug


class Publisher:
    def __init__(self, kb_dir, staging_dir, clock=None):
        self._kb_dir = Path(kb_dir)
        self._staging = Path(staging_dir)
        self._clock = clock

    def _now(self) -> datetime:
        return self._clock.now() if self._clock else datetime.now()

    @staticmethod
    def render(question: str, answer: str) -> str:
        """固定模板：# 自进化知识 + ## <规范化问题> + 回答。

        章节标题用规范化问题本身，chunk 的 section 元数据即问题，可检索性更好。
        """
        return f"# 自进化知识\n\n## {question}\n\n{answer}\n"

    @staticmethod
    def rescan(rendered: str) -> bool:
        """渲染后全文复扫：命中 PII/注入 → True（整条拒绝，不落原文）。"""
        return has_pii(rendered) or has_injection(rendered)

    def filename_for(self, candidate: CandidateQA, ts: Optional[datetime] = None) -> str:
        now = ts or self._now()
        day = now.strftime("%Y%m%d")
        return f"{day}-{candidate.candidate_id[:12]}-{make_slug(candidate.question)}.md"

    @staticmethod
    def _wrap_frontmatter(candidate: CandidateQA, ts: Optional[datetime] = None) -> str:
        """沉淀文档的 frontmatter 块（7.7 provenance/owner + 淘汰半边元数据）。

        - provenance/owner：来源 turn + system 标识（7.7）。
        - effective_date：发布日 + EVOLVE_EFFECTIVE_DAYS（默认 180），
          scan.py 通用 frontmatter 解析自动生效，用于知识时效治理。
        - grounded_on：沉淀时人工证据（根目录 + uploads/）的 source_path 列表。
        - quality_score / last_validated：前者供 P2-1 近重复替换决策，
          后者由 P1-2 重接地刷新（frontmatter 不进 chunk，无需重建索引）。
        """
        grounded = [
            s.source_path for s in candidate.sources
            if s.source_path and not s.source_path.startswith("evolved/")
        ]
        now = ts or datetime.now()
        effective = (now + timedelta(days=settings.evolve_effective_days)).strftime("%Y-%m-%d")
        quality = f"{candidate.quality_score:.2f}"
        return (
            "---\n"
            f"provenance: {candidate.turn_id}\n"
            "owner: system\n"
            f"effective_date: {effective}\n"
            f"quality_score: {quality}\n"
            f"last_validated: {now.strftime('%Y-%m-%d')}\n"
            f"grounded_on: {', '.join(grounded)}\n"
            "---\n"
        )

    def write_staging(self, candidate: CandidateQA, ts: Optional[datetime] = None) -> Optional[str]:
        """渲染 → 拼 frontmatter → 复扫（完整文档）→ 写入 staging 目录。

        7.7 沉淀元数据补强：完整文档 = frontmatter 块（provenance=
        turn_id / owner=system）+ render() 正文；沉淀文档可溯源到 turn，
        owner=system 供 7.1 统一元数据规范识别自进化文档。
        复扫作用于完整文档（含 frontmatter），命中 PII/注入返回 None（整条拒绝）。

        返回文件名；不产生任何知识库可见变化。
        """
        rendered = self.render(candidate.question, candidate.answer)
        full_doc = self._wrap_frontmatter(candidate, ts) + rendered
        if self.rescan(full_doc):
            return None
        filename = self.filename_for(candidate, ts)
        self._staging.mkdir(parents=True, exist_ok=True)
        (self._staging / filename).write_text(full_doc, encoding="utf-8")
        return filename

    def move_into_kb(self, filename: str) -> str:
        """staging → knowledge/evolved/ 移动；返回相对 kb_dir 的 posix 路径。"""
        evolved = self._kb_dir / "evolved"
        evolved.mkdir(parents=True, exist_ok=True)
        os.replace(self._staging / filename, evolved / filename)
        return (evolved / filename).relative_to(self._kb_dir).as_posix()

    def remove(self, filename: str) -> None:
        """删除 staging 与 evolved/ 下的同名文档（journal 恢复 / 阻断回滚用）。"""
        (self._staging / filename).unlink(missing_ok=True)
        (self._kb_dir / "evolved" / filename).unlink(missing_ok=True)

    def unpublish(self, filename: str, trash_dir) -> Optional[str]:
        """把 evolved/ 下的文档移到 trash；返回相对 kb_dir 路径，不存在返回 None。"""
        src = self._kb_dir / "evolved" / filename
        if not src.exists():
            return None
        trash = Path(trash_dir)
        trash.mkdir(parents=True, exist_ok=True)
        dst = trash / filename
        os.replace(src, dst)
        return src.relative_to(self._kb_dir).as_posix()

    def restore(self, filename: str, trash_dir) -> Optional[str]:
        """trash → evolved/ 还原（journal 未切换恢复用）；返回相对 kb_dir 路径，不存在返回 None。"""
        src = Path(trash_dir) / filename
        if not src.exists():
            return None
        evolved = self._kb_dir / "evolved"
        evolved.mkdir(parents=True, exist_ok=True)
        os.replace(src, evolved / filename)
        return (evolved / filename).relative_to(self._kb_dir).as_posix()

    def clean_staging(self, keep: set[str]) -> int:
        """删除 staging 下未被 keep 引用的 *.md / *.tmp（孤儿回收）；返回删除数。"""
        if not self._staging.is_dir():
            return 0
        removed = 0
        for f in list(self._staging.glob("*.md")) + list(self._staging.glob("*.tmp")):
            if f.name not in keep:
                f.unlink(missing_ok=True)
                removed += 1
        return removed