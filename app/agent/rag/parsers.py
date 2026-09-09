"""文档格式处理流水线（多格式接入：解析 → 结构保留 → 切分 → parent-child）。

生产知识源是 PDF/Word/HTML/飞书文档/截图。本模块提供：
- parse_document(path)：按后缀走解析器（.md/.txt 直通；.pdf 走 pypdf；
  .docx 走 python-docx；.html 走内置 html.parser）→ 统一 Markdown 化文本；
  各格式的标题/段落/表格结构尽量保留成 Markdown（供标题递归切分用）；
- chunk_kb_dir(kb_dir)：目录级入口（替代原 chunk_markdown_dir），对所有
  支持后缀统一解析、切分、frontmatter 透传、parent-child 装配；
- attach_parents：parent-child 兼容装配件（新流水线已在 chunker 内联装配，
  父块 = 切分前章节原文；本函数仅供手工构造的 Chunk 列表使用）。

OCR（扫描件/图片型政策文档）为独立可选件：PaddleOCR 等重依赖不进
requirements，接入点在 parsers.parse_document 的 .pdf/.jpg 分支，
预留 parse_image_ocr(path) 钩子。旧版 .doc 不再支持（提示转 .docx），
本轮不引入 LibreOffice。
"""

from __future__ import annotations

import re
from pathlib import Path

from app.agent.rag.chunker import Chunk, chunk_markdown_dir, _chunk_text
from app.agent.rag.loader import normalize_document
from app.observability.logging import get_logger

log = get_logger("app.agent.rag.parsers")

SUPPORTED_SUFFIXES = (".md", ".txt", ".pdf", ".docx", ".html", ".htm")


def parse_document(path: Path) -> str:
    """按后缀解析为统一文本（失败抛 ValueError，由调用方跳过该文件）。

    第三方库的解析异常（pypdf PdfStreamError、zipfile BadZipFile、docx 包异常等）
    统一包装为 ValueError——上层（strict 构建）只认这一种「解析失败」语义。
    """
    suffix = path.suffix.lower()
    try:
        if suffix in (".md", ".txt"):
            return path.read_text(encoding="utf-8")
        if suffix == ".pdf":
            return _parse_pdf(path)
        if suffix == ".docx":
            return _parse_docx(path)
        if suffix == ".doc":
            raise ValueError("旧版 .doc 格式不支持，请先转换为 .docx 后再上传")
        if suffix in (".html", ".htm"):
            return _parse_html(path)
        raise ValueError(f"不支持的文件格式: {suffix}（支持 {SUPPORTED_SUFFIXES}）")
    except ValueError:
        raise
    except OSError:
        raise
    except Exception as e:  # noqa: BLE001 —— 第三方解析异常统一归为「解析失败」
        raise ValueError(f"{suffix} 解析失败: {e}") from e


def _parse_pdf(path: Path) -> str:
    """逐页抽取 → 文件名一级标题 + 每页「## 第 N 页」章节（空页跳过）。

    页级章节让标题递归切分天然按页分块；全页无文本 = 扫描件，明确报
    「不支持 OCR」错误（OCR 为独立可选件）。
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ValueError("解析 PDF 需要 pypdf（pip install pypdf）") from e
    reader = PdfReader(str(path))
    parts = [f"# {path.stem}"]
    for num, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue
        parts.append(f"## 第 {num} 页\n\n{page_text}")
    if len(parts) <= 1:
        raise ValueError("PDF 无可提取文本（扫描件走 OCR 分支）")
    return "\n\n".join(parts)


def _parse_docx(path: Path) -> str:
    """按 XML 原始顺序处理段落与表格；Heading 1-6 → Markdown 标题。

    修复旧实现「先输出所有段落、再追加所有表格」的顺序错乱：直接遍历
    body 的 w:p / w:tbl 子元素，保持文档原序。表格转 Markdown 行。
    """
    try:
        from docx import Document
        from docx.oxml.ns import qn
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as e:
        raise ValueError("解析 Word 需要 python-docx（pip install python-docx）") from e

    doc = Document(str(path))
    parts: list[str] = []
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            para = Paragraph(child, doc)
            text = para.text.strip()
            if not text:
                continue
            level = _heading_level(para)
            parts.append(f"{'#' * level} {text}" if level else text)
        elif child.tag == qn("w:tbl"):
            table = Table(child, doc)
            rows: list[str] = []
            for row in table.rows:
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                if any(cells):
                    rows.append("| " + " | ".join(cells) + " |")
            if rows:
                parts.append("\n".join(rows))
    return "\n\n".join(parts)


def _heading_level(para) -> int:
    """Heading 1-6 段落样式 → Markdown 标题级别；非标题返回 0。

    python-docx 读 styles.xml 的样式名，内置标题在各语言文档中通常都是
    "Heading N"；个别本地化命名不匹配时按正文处理（宁缺勿错）。
    """
    try:
        name = (para.style.name or "").strip()
    except Exception:  # noqa: BLE001 —— 样式缺失/损坏按正文处理
        return 0
    m = re.match(r"^heading\s*(\d)$", name, re.IGNORECASE)
    if not m:
        return 0
    return min(int(m.group(1)), 6)


def _parse_html(path: Path) -> str:
    """HTML → Markdown 化正文：h1-h6 → 标题，li → 列表项，块级标签分段。

    script/style/head 等非正文内容整体忽略；块内空白按 HTML 语义折叠。
    """
    from html.parser import HTMLParser

    _SKIP = {"script", "style", "noscript", "template", "head", "iframe", "svg"}
    _HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
    _BLOCKS = {"p", "div", "section", "article", "aside", "header", "footer",
               "main", "nav", "table", "tr", "ul", "ol", "blockquote", "pre", "dl"}

    class _HTML2Markdown(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self._parts: list[str] = []
            self._buf: list[str] = []
            self._prefix = ""
            self._skip: dict[str, int] = {}

        def flush(self) -> None:
            text = " ".join("".join(self._buf).split())
            if text:
                self._parts.append(self._prefix + text)
            self._buf = []

        def _skipping(self) -> bool:
            return any(self._skip.values())

        def handle_starttag(self, tag, attrs):
            if tag in _SKIP:
                self._skip[tag] = self._skip.get(tag, 0) + 1
                return
            if self._skipping():
                return
            if tag in _HEADINGS:
                self.flush()
                self._prefix = "#" * int(tag[1]) + " "
            elif tag == "li":
                self.flush()
                self._prefix = "- "
            elif tag == "br" or tag in _BLOCKS:
                self.flush()

        def handle_endtag(self, tag):
            if tag in _SKIP:
                if self._skip.get(tag):
                    self._skip[tag] -= 1
                return
            if self._skipping():
                return
            if tag in _HEADINGS or tag == "li":
                self.flush()
                self._prefix = ""
            elif tag == "br" or tag in _BLOCKS:
                self.flush()

        def handle_startendtag(self, tag, attrs):
            if tag == "br" and not self._skipping():
                self.flush()

        def handle_data(self, data):
            if self._skipping() or not data.strip():
                return
            self._buf.append(data)

    parser = _HTML2Markdown()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    parser.flush()
    return "\n\n".join(parser._parts)


def parse_image_ocr(path: Path) -> str:
    """OCR 钩子（7.1）：扫描件/图片型政策文档。

    默认不实现（PaddleOCR 为可选重依赖）：接入实现放到
    requirements-ocr.txt 并在此返回识别文本；未接入抛 ValueError。
    """
    raise ValueError(
        "OCR 未接入：扫描件请安装 PaddleOCR/云服务后实现 parse_image_ocr"
    )


def attach_parents(chunks: list[Chunk], max_parent_chars: int = 4000) -> list[Chunk]:
    """parent-child 兼容装配件（手工 Chunk 列表用）。

    正常构建路径的父块已在 chunker._chunk_text 内联装配（父块 = 命中子块
    附近的章节原文窗口）。本函数只对「同 (doc, section) 多块且全部
    parent_text 为空」的旧式输入生效；拼接结果超过 max_parent_chars 时
    无法保证完整包含每个子块 → 不装配（回退命中子块），绝不截断了事。
    """
    groups: dict[tuple[str, str], list[Chunk]] = {}
    for c in chunks:
        groups.setdefault((c.doc, c.section), []).append(c)

    for group in groups.values():
        if len(group) <= 1 or any(c.parent_text for c in group):
            continue
        parent = "\n\n".join(c.text for c in group)
        if len(parent) > max_parent_chars:
            continue
        for c in group:
            c.parent_text = parent
    return chunks


def chunk_kb_dir(kb_dir: Path, *, parent_child: bool = True, strict: bool = False) -> list[Chunk]:
    """目录级多格式接入入口：所有支持后缀 → 解析 → 切分 → parent-child。

    - 隐藏目录（路径段以 `.` 开头：.staging/.trash/.git 等）一律跳过——
      KB 上传链路的暂存/隔离区绝不进入索引（v7 评审 R3）；
    - strict=True（上传/下架等**会改变正式 alias 的构建**）：任一源文件解析
      失败 → 整体抛错（含路径），杜绝「一次重建静默丢失已有知识」；
      strict=False 仅用于扫描/报告类工具（坏文件跳过，进报告人工处理）；
    - 旧版 .doc 不属于可索引格式：strict 立即失败并提示转换，非 strict
      跳过并记录明确 warning（绝不静默丢失）；其它未知后缀维持忽略。

    .md/.txt 直接复用既有切分（含 evolved/ 沉淀 frontmatter 透传）。
    """
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(kb_dir).as_posix()
        if any(part.startswith(".") for part in Path(rel).parts):
            continue  # 隐藏目录（.staging/.trash）不进索引
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            if suffix == ".doc":
                msg = f"{rel} 为旧版 .doc 格式，已跳过：请先转换为 .docx 后重新上传"
                if strict:
                    raise ValueError(f"strict 构建中止：{msg}")
                log.warning(msg)
            continue
        try:
            text = parse_document(path)
            normalized = normalize_document(text)
            if suffix in (".md", ".txt"):
                chunks.extend(
                    _chunk_text(path.read_text(encoding="utf-8"), rel, kb_dir,
                                parent_child=parent_child)
                )
            elif normalized:
                chunks.extend(
                    _chunk_text(normalized, rel, kb_dir, parent_child=parent_child)
                )
        except (ValueError, OSError) as e:
            if strict:
                raise ValueError(
                    f"strict 构建中止：{rel} 解析失败（{e}）"
                ) from e
            # 非 strict：坏文件跳过，进报告由审核侧处理
            continue
    return chunks
