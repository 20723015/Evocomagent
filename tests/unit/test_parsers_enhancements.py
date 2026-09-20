"""解析器增强（P1-3 PDF 版式 / P1-4 docx 编号 / P1-5 xlsx）。

覆盖点：
- P1-4：样式继承编号（本库语料形态）、直接 numPr、被打断后重新计数、
  多级 lvlText、非十进制不生成标记、无 numbering.xml 不炸；
- P1-3：字号 → 标题级别提升、跨页重复行剔除、无提示时输出逐字节不变、
  页眉即唯一大字行时保留页码标题、Docling 未安装时静默回退；
- P1-5：xlsx sheet → pipe 表（纯函数 + 真实 openpyxl）、空工作簿报错、
  strict 构建接受 xlsx。
"""

from __future__ import annotations

import pytest

from app.agent.rag.parsers import (
    _PdfLayoutHints,
    _apply_layout_hints,
    _heading_pages,
    _parse_docx,
    _parse_xlsx,
    _pdf_layout_hints,
    _workbook_to_markdown,
    parse_document,
)


# ============================================================
# P1-4 docx 自动编号
# ============================================================
def _docx_with_numbering(tmp_path, name: str = "n.docx"):
    """构造含命名空间的最小 docx（List Number 样式承载编号，同本库语料）。"""
    docx = pytest.importorskip("docx")
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    doc = docx.Document()
    doc.add_heading("政策", level=1)

    styles = doc.styles.element
    style = OxmlElement("w:style")
    style.set(qn("w:type"), "paragraph")
    style.set(qn("w:styleId"), "ListNumber")
    name_el = OxmlElement("w:name")
    name_el.set(qn("w:val"), "List Number")
    style.append(name_el)
    ppr = OxmlElement("w:pPr")
    numpr = OxmlElement("w:numPr")
    num_id = OxmlElement("w:numId")
    num_id.set(qn("w:val"), "5")
    numpr.append(num_id)
    ppr.append(numpr)
    style.append(ppr)
    styles.append(style)

    doc.add_heading("一节", level=2)
    for text in ("第一条内容。", "第二条内容。"):
        para = doc.add_paragraph(text)
        para.style = doc.styles["List Number"]
    doc.add_heading("二节", level=2)
    for text in ("新组第一条。", "新组第二条。", "新组第三条。"):
        doc.add_paragraph(text).style = doc.styles["List Number"]

    path = tmp_path / name
    doc.save(path)
    return path


def test_docx_style_inherited_numbering_restored(tmp_path):
    """样式继承的编号（段落无 numPr）必须还原，且被标题打断后重新计数。"""
    path = _docx_with_numbering(tmp_path)
    text = _parse_docx(path)
    assert "1. 第一条内容。" in text
    assert "2. 第二条内容。" in text
    # 新小节重新从 1 开始（与源 Markdown 的观感一致）
    assert "1. 新组第一条。" in text
    assert "3. 新组第三条。" in text
    assert "4. 新组第三条。" not in text


def test_docx_direct_num_pr_and_non_decimal(tmp_path):
    docx = pytest.importorskip("docx")
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    path = tmp_path / "d.docx"
    doc = docx.Document()

    numbering = doc.part.numbering_part.element
    abstract = OxmlElement("w:abstractNum")
    abstract.set(qn("w:abstractNumId"), "9")
    lvl = OxmlElement("w:lvl")
    lvl.set(qn("w:ilvl"), "0")
    for tag, val in (("w:start", "1"), ("w:numFmt", "decimal"),
                     ("w:lvlText", "%1.")):
        el = OxmlElement(tag)
        el.set(qn("w:val"), val)
        lvl.append(el)
    abstract.append(lvl)
    numbering.insert(0, abstract)
    num = OxmlElement("w:num")
    num.set(qn("w:numId"), "42")
    abs_ref = OxmlElement("w:abstractNumId")
    abs_ref.set(qn("w:val"), "9")
    num.append(abs_ref)
    numbering.append(num)

    for text in ("甲条款。", "乙条款。"):
        para = doc.add_paragraph(text)
        ppr = para._p.get_or_add_pPr()
        numpr = OxmlElement("w:numPr")
        nid = OxmlElement("w:numId")
        nid.set(qn("w:val"), "42")
        numpr.append(nid)
        ppr.append(numpr)
    doc.save(path)

    text = _parse_docx(path)
    assert "1. 甲条款。" in text and "2. 乙条款。" in text


def test_docx_typed_ordinal_is_not_double_numbered(tmp_path):
    """段落正文已手打序号（且挂了列表样式）→ 不再前置自动编号（避免「1. 1.」）。"""
    path = _docx_with_numbering(tmp_path, "typed.docx")
    docx = pytest.importorskip("docx")
    doc = docx.Document(str(path))
    para = doc.add_paragraph("3. 作者已手打的第三条。")
    para.style = doc.styles["List Number"]
    doc.save(path)

    text = _parse_docx(path)
    assert "3. 作者已手打的第三条。" in text
    assert "4. 3. 作者已手打的第三条。" not in text


def test_docx_bullet_lists_are_not_numbered(tmp_path):
    """非十进制（项目符号）不生成标记——宁简勿错，不臆造序号。"""
    docx = pytest.importorskip("docx")

    path = tmp_path / "b.docx"
    doc = docx.Document()
    doc.add_paragraph("要点甲").style = doc.styles["List Bullet"]
    doc.add_paragraph("要点乙").style = doc.styles["List Bullet"]
    doc.save(path)

    text = _parse_docx(path)
    assert "要点甲" in text and "要点乙" in text
    assert "1. 要点甲" not in text


# ============================================================
# P1-3 PDF 版式提示
# ============================================================
def _hints(headings=None, margins=()):
    return _PdfLayoutHints(headings=headings or {}, margins=frozenset(margins))


def test_apply_layout_hints_promotes_headings_in_order():
    pages = ["公司名\n一、总则\n正文第一段。\n1.1 细则\n细则正文。"]
    out = _apply_layout_hints(
        pages,
        _hints({0: [("一、总则", 2), ("1.1细则", 3)]}),
    )[0]
    lines = out.split("\n")
    assert lines[0] == "公司名"  # 未命中的行原样保留
    assert lines[1] == "## 一、总则"
    assert lines[2] == "正文第一段。"
    assert lines[3] == "### 1.1 细则"
    # 行数不变（只提升，不重排）
    assert len(out.split("\n")) == len(pages[0].split("\n"))


def test_apply_layout_hints_strips_repeated_margins():
    pages = ["页眉\n正文甲。", "页眉\n正文乙。"]
    out = _apply_layout_hints(pages, _hints(margins=("页眉",)))
    assert out == ["正文甲。", "正文乙。"]


def test_apply_layout_hints_is_noop_without_hints():
    pages = ["原样\n内容"]
    assert _apply_layout_hints(pages, None) == pages
    assert _apply_layout_hints(pages, _hints()) == pages


def test_heading_pages_keeps_marker_when_only_running_header():
    """页内大字行全是被剔除的页眉 → 未恢复结构，保留「## 第 N 页」分块边界。"""
    pages_marker = _heading_pages(
        _hints({0: [("公司名", 2)]}, margins=("公司名",))
    )
    assert pages_marker == frozenset()
    # 含真实结构标题的页则不再套页码标题
    assert _heading_pages(
        _hints({0: [("公司名", 2), ("一、总则", 3)]}, margins=("公司名",))
    ) == frozenset({0})


def test_pdf_layout_hints_none_without_large_text(tmp_path):
    reportlab = pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas

    path = tmp_path / "flat.pdf"
    c = canvas.Canvas(str(path))
    for i in range(3):
        c.setFont("Helvetica", 10)
        c.drawString(50, 700, f"body line {i}")
        c.showPage()
    c.save()
    assert _pdf_layout_hints(path) is None  # 无提示 → 输出与历史实现一致


def test_pdf_docling_missing_falls_back(tmp_path, monkeypatch):
    """Docling 未安装时静默回退（可选依赖不是错误）。"""
    from app.agent.rag import parsers

    assert parsers._parse_docling(tmp_path / "x.pdf") is None


def test_pdf_docling_toggle_off(tmp_path, monkeypatch):
    from app.agent.rag import parsers
    from app.config.settings import settings

    monkeypatch.setattr(settings, "rag_pdf_docling", False, raising=False)
    assert parsers._parse_docling(tmp_path / "x.pdf") is None


# ============================================================
# P1-5 xlsx
# ============================================================
def test_xlsx_sheets_become_tables():
    openpyxl = pytest.importorskip("openpyxl")

    class _Sheet:
        def __init__(self, title, rows):
            self.title = title
            self._rows = rows

        def iter_rows(self, values_only=True):
            return iter(self._rows)

    class _Book:
        worksheets = [
            _Sheet("分期费率", [["期数", "费率"], [3, "1.5%"], ["", None]]),
            _Sheet("空表", [[None, ""]]),
        ]

    text = _workbook_to_markdown("费率表", _Book())
    assert text.startswith("# 费率表")
    assert "## 分期费率" in text
    assert "| 期数 | 费率 |" in text and "| 3 | 1.5% |" in text
    assert "空表" not in text  # 空工作表跳过


def test_xlsx_real_file_roundtrip(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")

    path = tmp_path / "价保除外类目清单.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "除外类目"
    ws.append(["类目", "说明"])
    ws.append(["生鲜", "含 A|B 单元格"])
    wb.save(path)

    text = _parse_xlsx(path)
    assert "## 除外类目" in text
    assert "含 A\\|B 单元格" in text  # 单元格竖线转义（防列错位）


def test_xlsx_empty_workbook_errors(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")

    path = tmp_path / "empty.xlsx"
    openpyxl.Workbook().save(path)
    with pytest.raises(ValueError, match="无可提取内容"):
        _parse_xlsx(path)


def test_strict_build_accepts_xlsx(tmp_path):
    """xlsx 进入受支持格式：strict 构建不再报「不支持的格式」。"""
    openpyxl = pytest.importorskip("openpyxl")
    from app.agent.rag.parsers import chunk_kb_dir

    kb = tmp_path / "kb"
    kb.mkdir()
    path = kb / "费率.xlsx"
    wb = openpyxl.Workbook()
    wb.active.append(["期数", "费率"])
    wb.active.append([3, "1.5%"])
    wb.save(path)

    chunks = chunk_kb_dir(kb, strict=True)
    assert chunks
    assert any("1.5%" in c.text for c in chunks)


def test_parse_document_routes_xlsx(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")

    path = tmp_path / "a.xlsx"
    wb = openpyxl.Workbook()
    wb.active.append(["列"])
    wb.active.append(["值"])
    wb.save(path)
    assert "| 值 |" in parse_document(path)
