#!/usr/bin/env python
"""把两份 E 类长文档转换为 .docx / .pdf 原生格式入库（计划 §7.5）。

- 商家入驻与保证金管理规范.md → .docx（python-docx，保留标题层级）
- 平台治理与处罚总则.md → .pdf（reportlab，保留标题层级）

转换后原 .md 保留（md 与 docx/pdf 并存入库，用于多格式解析验收对照）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
KB = ROOT / "app/agent/rag/knowledge"


def md_to_blocks(text: str) -> list[tuple[str, str]]:
    """把 markdown 行转为 (类型, 内容) 块。类型: h1-h6 / p / bullet / num"""
    blocks = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            blocks.append((f"h{len(m.group(1))}", m.group(2).strip()))
            continue
        m = re.match(r"^\|.*\|$", stripped)
        if m:
            blocks.append(("table", stripped))
            continue
        if stripped.startswith("- "):
            blocks.append(("bullet", stripped[2:].strip()))
            continue
        if re.match(r"^\d+[.、]\s*", stripped):
            blocks.append(("num", re.sub(r"^\d+[.、]\s*", "", stripped).strip()))
            continue
        blocks.append(("p", stripped))
    return blocks


def convert_docx(src: Path, dst: Path) -> None:
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    for kind, content in md_to_blocks(src.read_text(encoding="utf-8")):
        if kind == "h1":
            doc.add_heading(content, level=1)
        elif kind == "h2":
            doc.add_heading(content, level=2)
        elif kind == "h3":
            doc.add_heading(content, level=3)
        elif kind == "h4":
            doc.add_heading(content, level=4)
        elif kind in ("h5", "h6"):
            doc.add_heading(content, level=5)
        elif kind == "table":
            cells = [c.strip() for c in content.strip("|").split("|")]
            doc.add_paragraph(" | ".join(cells), style="Normal")
        elif kind in ("bullet", "num"):
            p = doc.add_paragraph(content)
            p.style = doc.styles["List Bullet"] if kind == "bullet" else doc.styles["List Number"]
        else:
            doc.add_paragraph(content)
    doc.save(dst)
    print(f"docx: {dst.name} ({dst.stat().st_size} B)")


def convert_pdf(src: Path, dst: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    # 中文字体：尝试系统微软雅黑/宋体
    font = None
    for cand in (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ):
        if Path(cand).exists():
            try:
                pdfmetrics.registerFont(TTFont("CJK", cand))
                font = "CJK"
                break
            except Exception:
                continue
    if font is None:
        print("警告: 未找到可用中文字体，PDF 中文可能无法渲染", file=sys.stderr)

    styles = {
        "h1": ParagraphStyle("h1", fontName=font or "Helvetica", fontSize=16, leading=22, spaceAfter=8),
        "h2": ParagraphStyle("h2", fontName=font or "Helvetica", fontSize=13, leading=18, spaceBefore=8, spaceAfter=4),
        "h3": ParagraphStyle("h3", fontName=font or "Helvetica", fontSize=11, leading=15, spaceBefore=6, spaceAfter=3),
        "h4": ParagraphStyle("h4", fontName=font or "Helvetica", fontSize=10, leading=14, spaceBefore=4),
        "p": ParagraphStyle("p", fontName=font or "Helvetica", fontSize=9.5, leading=14, spaceAfter=2),
    }
    doc = SimpleDocTemplate(str(dst), pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm)
    story = []
    for kind, content in md_to_blocks(src.read_text(encoding="utf-8")):
        style = styles.get(kind if kind in styles else "p", styles["p"])
        if kind == "table":
            story.append(Paragraph(content.replace("|", "｜"), style))
        elif kind in ("bullet", "num"):
            prefix = "• " if kind == "bullet" else "· "
            story.append(Paragraph(prefix + content, style))
        elif kind in ("h5", "h6"):
            story.append(Paragraph(content, styles["h4"]))
        else:
            story.append(Paragraph(content, style))
        if kind.startswith("h"):
            story.append(Spacer(1, 3))
    doc.build(story)
    print(f"pdf : {dst.name} ({dst.stat().st_size} B)")


def main() -> int:
    jobs = [
        (KB / "商家入驻与保证金管理规范.md", KB / "商家入驻与保证金管理规范.docx", convert_docx),
        (KB / "平台治理与处罚总则.md", KB / "平台治理与处罚总则.pdf", convert_pdf),
    ]
    for src, dst, fn in jobs:
        if not src.exists():
            print(f"❌ 源文件缺失: {src.name}", file=sys.stderr)
            continue
        fn(src, dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
