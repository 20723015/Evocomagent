"""按二级标题切分知识库文档（.md / .txt，7.1 多格式接入）。

切分策略：
- 以 `## ` 二级标题为切分边界，每个二级章节作为一个 chunk。
- 二级章节内的三级/四级小节保留在同一 chunk 中（保持语义完整）。
- 文档开头到第一个二级标题之间的内容（含一级标题和导语）作为 "概览" chunk。
- chunk 长度超过 ~1200 字时按段落进一步切分，避免单个 chunk 过长。

每个 chunk 保留：
- doc：文档名（不含扩展名），如 "退换货政策"；evolved/ 子目录下统一为 "自进化知识"
- section：章节标题，如 "二、质量问题退换货"
- text：chunk 全文（含小节结构）
- chunk_id：稳定的字符串 id，便于增量更新
- source_path：相对 kb_dir 的 posix 路径，用于来源溯源（第10期）
- provenance：来源溯源（如沉淀文档的 turn_id，7.7 写入 frontmatter）
- owner：归属（system 表示自进化生成，供 7.1 统一元数据规范识别）

文档接入流程：raw → normalize_document → parse_frontmatter（frontmatter
不进入 chunk text）→ 其余为正文走既有切分。
"""

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.agent.rag.loader import normalize_document, parse_frontmatter

MAX_CHUNK_CHARS = 1200

# evolved/ 子目录（第10期 QA 自动沉淀）的统一展示名
EVOLVED_DOC_NAME = "自进化知识"


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    section: str
    text: str
    source_path: str = ""  # 相对 kb_dir 的 posix 路径；默认空保证旧索引反序列化兼容
    provenance: str = ""  # 来源溯源（如沉淀文档的 turn_id）；默认空兼容旧索引
    owner: str = ""  # 归属（system 表示自进化生成）；默认空兼容旧索引
    parent_text: str = ""  # 7.1 parent-child：命中子块时返回章节级父块；默认空兼容旧索引

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_markdown_dir(kb_dir: Path) -> list[Chunk]:
    """递归扫描目录下所有 .md 与 .txt 文档（含 evolved/ 子目录），逐一切分并汇总。

    7.1 多格式接入第一步：在 Markdown 基础上叠加 .txt，按后缀合并排序扫描。
    """
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.rglob("*.md")) + sorted(kb_dir.rglob("*.txt")):
        if path.is_dir():
            continue
        chunks.extend(_chunk_one_file(path, kb_dir))
    return chunks


def _chunk_one_file(path: Path, kb_dir: Path) -> list[Chunk]:
    """切分单个文档文件：normalize → 解析 frontmatter → 正文切分。

    frontmatter 中的 provenance / owner 透传到每个 chunk（缺省 ""），
    frontmatter 本身不进入 chunk text。
    """
    raw = path.read_text(encoding="utf-8")
    rel = path.relative_to(kb_dir).as_posix()
    return _chunk_text(raw, rel, kb_dir)


def _chunk_text(raw: str, rel: str, kb_dir: Path) -> list[Chunk]:
    """对已读入文本做 frontmatter 解析与正文切分（7.1 解析流水线复用）。"""
    path = Path(rel)
    is_evolved = rel.startswith("evolved/") or rel == "evolved.md"
    if is_evolved:
        doc_name = EVOLVED_DOC_NAME
        id_prefix = path.with_suffix("").as_posix()
    else:
        doc_name = path.stem
        id_prefix = doc_name
    meta, body = parse_frontmatter(normalize_document(raw))
    provenance = meta.get("provenance", "")
    owner = meta.get("owner", "")
    sections = _split_by_h2(body)

    out: list[Chunk] = []
    for idx, (section_title, section_body) in enumerate(sections):
        text = section_body.strip()
        if not text:
            continue

        if len(text) <= MAX_CHUNK_CHARS:
            out.append(_make_chunk(doc_name, section_title, text, idx, 0, id_prefix, rel,
                                   provenance, owner))
            continue

        for sub_idx, piece in enumerate(_split_long(text)):
            out.append(_make_chunk(doc_name, section_title, piece, idx, sub_idx, id_prefix, rel,
                                   provenance, owner))
    return out


def _split_by_h2(raw: str) -> list[tuple[str, str]]:
    """返回 [(section_title, section_body), ...]。

    第一个 section 是文档头部（H1 + 导语），title 取 H1 文本。
    """
    lines = raw.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_body: list[str] = []

    for line in lines:
        if line.startswith("# ") and not current_body and not sections:
            current_title = line[2:].strip() + " · 概览"
            continue
        if line.startswith("## "):
            if current_body:
                sections.append((current_title or "概览", current_body))
            current_title = line[3:].strip()
            current_body = []
            continue
        current_body.append(line)

    if current_body:
        sections.append((current_title or "概览", current_body))

    return [(t, "\n".join(b).strip()) for t, b in sections]


def _split_long(text: str) -> list[str]:
    """按段落贪心打包到 MAX_CHUNK_CHARS。"""
    paragraphs = re.split(r"\n\s*\n", text)
    pieces: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if buf and buf_len + len(p) + 2 > MAX_CHUNK_CHARS:
            pieces.append("\n\n".join(buf))
            buf = [p]
            buf_len = len(p)
        else:
            buf.append(p)
            buf_len += len(p) + 2
    if buf:
        pieces.append("\n\n".join(buf))
    return pieces


def _make_chunk(
    doc: str,
    section: str,
    text: str,
    idx: int,
    sub_idx: int,
    id_prefix: str,
    source_path: str,
    provenance: str = "",
    owner: str = "",
) -> Chunk:
    chunk_id = f"{id_prefix}#{idx:02d}-{sub_idx:02d}"
    body = f"【{doc} · {section}】\n{text}"
    return Chunk(
        chunk_id=chunk_id, doc=doc, section=section, text=body,
        source_path=source_path, provenance=provenance, owner=owner,
    )
