"""文档格式处理流水线（阶段七 7.1 剩余大头：解析 → 清洗 → 元数据）。

生产知识源是 PDF/Word/HTML/飞书文档/截图。本模块提供：
- parse_document(path)：按后缀走解析器（.md/.txt 直通；.pdf 走 pypdf；
  .docx 走 python-docx；.html 走内置 html.parser 抽文本）→ 统一纯文本；
- chunk_kb_dir(kb_dir)：目录级入口（替代原 chunk_markdown_dir），对所有
  支持后缀统一解析、切分、frontmatter 透传、parent-child 装配；
- attach_parents：parent-child 分块——小块（检索单元）命中后返回父块
  （章节级）给 LLM，兼顾精度与上下文完整。

OCR（扫描件/图片型政策文档）为独立可选件：PaddleOCR 等重依赖不进
requirements，接入点在 parsers.parse_document 的 .pdf/.jpg 分支，
预留 parse_image_ocr(path) 钩子。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.agent.rag.chunker import Chunk, chunk_markdown_dir, _chunk_text
from app.agent.rag.loader import normalize_document

SUPPORTED_SUFFIXES = (".md", ".txt", ".pdf", ".docx", ".doc", ".html", ".htm")


def parse_document(path: Path) -> str:
    """按后缀解析为统一纯文本（失败抛 ValueError，由调用方跳过该文件）。

    第三方库的解析异常（pypdf PdfStreamError、zipfile BadZipFile、docx 包异常等）
    统一包装为 ValueError——上层（strict 构建）只认这一种「解析失败」语义。
    """
    suffix = path.suffix.lower()
    try:
        if suffix in (".md", ".txt"):
            return path.read_text(encoding="utf-8")
        if suffix == ".pdf":
            return _parse_pdf(path)
        if suffix in (".docx", ".doc"):
            return _parse_docx(path)
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
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ValueError("解析 PDF 需要 pypdf（pip install pypdf）") from e
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages)
    if not text.strip():
        raise ValueError("PDF 无可提取文本（扫描件走 OCR 分支）")
    return text


def _parse_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as e:
        raise ValueError("解析 Word 需要 python-docx（pip install python-docx）") from e
    doc = Document(str(path))
    parts: list[str] = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def _parse_html(path: Path) -> str:
    from html.parser import HTMLParser

    class _TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self._blocks: list[str] = []
            self._buf: list[str] = []

        def handle_data(self, data):
            if data.strip():
                self._buf.append(data.strip())

        def flush(self):
            if self._buf:
                self._blocks.append(" ".join(self._buf))
                self._buf = []

        def handle_starttag(self, tag, attrs):
            if tag in ("p", "div", "li", "h1", "h2", "h3", "tr", "br"):
                self.flush()

        def handle_endtag(self, tag):
            if tag in ("p", "div", "li", "h1", "h2", "h3", "tr"):
                self.flush()

    extractor = _TextExtractor()
    extractor.feed(path.read_text(encoding="utf-8", errors="replace"))
    extractor.flush()
    return "\n\n".join(extractor._blocks)


def parse_image_ocr(path: Path) -> str:
    """OCR 钩子（7.1）：扫描件/图片型政策文档。

    默认不实现（PaddleOCR 为可选重依赖）：接入实现放到
    requirements-ocr.txt 并在此返回识别文本；未接入抛 ValueError。
    """
    raise ValueError(
        "OCR 未接入：扫描件请安装 PaddleOCR/云服务后实现 parse_image_ocr"
    )


def attach_parents(chunks: list[Chunk], max_parent_chars: int = 4000) -> list[Chunk]:
    """parent-child 装配：同 (doc, section) 的块共用一个章节级父块。

    父块 = 同章节所有块文本拼接（截断到 max_parent_chars），命中任意子块
    时由上层返回父块给 LLM（见 chunk_kb_dir 的 parent-child 语义注释）。
    """
    groups: dict[tuple[str, str], list[Chunk]] = {}
    for c in chunks:
        groups.setdefault((c.doc, c.section), []).append(c)

    for key, group in groups.items():
        if len(group) <= 1:
            continue
        parent = "\n\n".join(c.text for c in group)
        if len(parent) > max_parent_chars:
            parent = parent[:max_parent_chars]
        for c in group:
            c.parent_text = parent
    return chunks


def chunk_kb_dir(kb_dir: Path, *, parent_child: bool = True, strict: bool = False) -> list[Chunk]:
    """目录级多格式接入入口：所有支持后缀 → 解析 → 切分 → parent-child。

    - 隐藏目录（路径段以 `.` 开头：.staging/.trash/.git 等）一律跳过——
      KB 上传链路的暂存/隔离区绝不进入索引（v7 评审 R3）；
    - strict=True（上传/下架等**会改变正式 alias 的构建**）：任一源文件解析
      失败 → 整体抛错（含路径），杜绝「一次重建静默丢失已有知识」；
      strict=False 仅用于扫描/报告类工具（坏文件跳过，进报告人工处理）。

    .md/.txt 直接复用既有切分（含 evolved/ 沉淀 frontmatter 透传）。
    """
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        rel = path.relative_to(kb_dir).as_posix()
        if any(part.startswith(".") for part in Path(rel).parts):
            continue  # 隐藏目录（.staging/.trash）不进索引
        try:
            text = parse_document(path)
            normalized = normalize_document(text)
            if not normalized:
                continue
            if path.suffix.lower() in (".md", ".txt"):
                chunks.extend(
                    _chunk_text(path.read_text(encoding="utf-8"), rel, kb_dir)
                )
            else:
                chunks.extend(_chunk_text(normalized, rel, kb_dir))
        except (ValueError, OSError) as e:
            if strict:
                raise ValueError(
                    f"strict 构建中止：{rel} 解析失败（{e}）"
                ) from e
            # 非 strict：坏文件跳过，进报告由审核侧处理
            continue
    if parent_child:
        attach_parents(chunks)
    return chunks
