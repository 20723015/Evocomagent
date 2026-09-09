"""文档格式与递归分块补全：多级标题路径、长文本兜底、多格式解析、父子块检索。

覆盖点：
- H1-H6 递归标题路径、缺级标题、同名子标题、fenced code 内形似标题忽略；
- 长文本兜底：2500 字无空行文本多块、块长 ≤ 硬上限、相邻块 overlap；
- Markdown 图片/链接/代码块原子性（不从语法内部截断）；
- PDF 页级章节、扫描 PDF 明确失败；DOCX 文档顺序与标题；HTML 结构保留；
- .doc 明确提示转换；旧 Chunk 数据反序列化兼容；
- 检索结果 parent_id 去重与 parent/matched 上下文；三种后端新字段持久化。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.agent.rag.backends.base import RetrievedChunk
from app.agent.rag.chunker import (
    MAX_CHUNK_CHARS,
    OVERLAP_CHARS,
    Chunk,
    _chunk_text,
    _split_long,
    chunk_body,
    chunk_markdown_dir,
)
from app.agent.rag.parsers import chunk_kb_dir, parse_document
from app.agent.rag.retriever import collapse_by_parent


# ============================================================
# 1. 递归标题切片
# ============================================================
def _write(kb, name: str, content: str) -> None:
    (kb / name).write_text(content, encoding="utf-8")


def test_h1_to_h6_heading_paths(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    _write(kb, "policy.md", (
        "# 退款政策\n\n总则内容。\n\n"
        "## 特殊商品\n\n特殊商品通用规则。\n\n"
        "### 生鲜商品\n\n生鲜不支持七天无理由。\n\n"
        "#### 冷冻海鲜\n\n冷冻海鲜48小时内反馈。\n\n"
        "##### 进口虾类\n\n进口虾类需检疫证明。\n\n"
        "###### 其他\n\n其他特殊情形。\n"
    ))
    chunks = chunk_markdown_dir(kb)
    by_path = {c.heading_path: c for c in chunks}
    assert "退款政策" in by_path
    assert "退款政策 > 特殊商品" in by_path
    assert "退款政策 > 特殊商品 > 生鲜商品" in by_path
    assert "退款政策 > 特殊商品 > 生鲜商品 > 冷冻海鲜 > 进口虾类" in by_path
    # 每个标题节点的直属正文独立成块：子标题正文不混入父标题正文
    assert "生鲜不支持" not in by_path["退款政策 > 特殊商品"].text
    assert "冷冻海鲜" not in by_path["退款政策 > 特殊商品 > 生鲜商品"].text
    # 检索文本带完整标题路径前缀
    leaf = by_path["退款政策 > 特殊商品 > 生鲜商品"]
    assert leaf.text.startswith("【policy · 退款政策 > 特殊商品 > 生鲜商品】")
    assert leaf.parent_id  # 每个章节都有稳定父标识


def test_skipped_level_heading_path(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    _write(kb, "skip.md", "# 总则\n\n### 突降三级\n\n内容。\n")
    chunks = chunk_markdown_dir(kb)
    paths = {c.heading_path for c in chunks}
    # H1 无直属正文 → 不产生独立块；缺级跳转路径仍按栈拼接
    assert paths == {"总则 > 突降三级"}
    _write(kb, "skip2.md", "# 总则\n\n导语。\n\n### 突降三级\n\n内容。\n")
    paths2 = {c.heading_path for c in chunk_markdown_dir(kb) if c.doc == "skip2"}
    assert paths2 == {"总则", "总则 > 突降三级"}


def test_duplicate_sibling_headings_unique_paths_and_ids(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    _write(kb, "dup.md", (
        "# 政策\n\n## 情形\n\n第一种。\n\n### 说明\n\n甲说明。\n\n"
        "## 情形\n\n第二种。\n\n### 说明\n\n乙说明。\n"
    ))
    chunks = chunk_markdown_dir(kb)
    paths = [c.heading_path for c in chunks]
    assert paths.count("政策 > 情形") == 2
    assert paths.count("政策 > 情形 > 说明") == 2
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)  # 同名子标题 chunk_id 仍唯一


def test_heading_inside_code_fence_ignored(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    _write(kb, "code.md", (
        "# 政策\n\n## 用法\n\n示例：\n\n"
        "```\n### 这不是标题\n## 也不是\n```\n\n"
        "## 下一个章节\n\n内容。\n"
    ))
    chunks = chunk_markdown_dir(kb)
    paths = {c.heading_path for c in chunks}
    assert "政策 > 这不是标题" not in paths
    assert "政策 > 也不是" not in paths
    # 代码块本体保留在章节正文中
    usage = [c for c in chunks if c.heading_path == "政策 > 用法"][0]
    assert "### 这不是标题" in usage.text
    assert "政策 > 下一个章节" in paths


def test_recursive_scan_and_hidden_dirs(tmp_path):
    kb = tmp_path / "kb"
    (kb / "sub" / "deep").mkdir(parents=True)
    (kb / ".trash").mkdir()
    _write(kb / "sub" / "deep", "a.md", "# 文档A\n\n## 章节\n\n内容A\n")
    _write(kb / ".trash", "b.md", "# 垃圾\n\n## 章节\n\n内容B\n")
    chunks = chunk_kb_dir(kb)
    docs = {c.doc for c in chunks}
    assert "a" in docs
    assert "垃圾" not in docs
    assert all(c.heading_path for c in chunks)


def test_old_chunk_dict_without_new_fields_deserializes():
    """旧索引（无 heading_path/parent_id/parent_text 键）Chunk(**c) 不炸。"""
    old = {"chunk_id": "退货政策#00-00", "doc": "退货政策",
           "section": "七天无理由", "text": "内容"}
    chunk = Chunk(**old)
    assert chunk.heading_path == ""
    assert chunk.parent_id == ""
    assert chunk.parent_text == ""


# ============================================================
# 2. 长文本强制分块
# ============================================================
def test_2500_char_text_multi_blocks_cap_and_overlap():
    text = "这是一段用于测试的连续文本，包含多个完整句子。" * 114  # ~2500 字，无空行
    pieces = _split_long(text)
    texts = [p[0] for p in pieces]
    assert len(texts) >= 2
    assert all(len(t) <= MAX_CHUNK_CHARS for t in texts)
    # 相邻块重叠：下一块以上一块尾部开头
    for prev, nxt in zip(texts, texts[1:]):
        assert nxt.startswith(prev[-OVERLAP_CHARS:])
    # 位置可定位且覆盖连续区间
    for t, start, end in pieces:
        assert start >= 0 and text[start:end] == t


def test_paragraph_packing_respects_target_and_cap():
    para_a = "段落甲的内容。" * 80  # ~480 字
    para_b = "段落乙的内容。" * 80
    para_c = "段落丙的内容。" * 80
    texts = [p[0] for p in _split_long("\n\n".join([para_a, para_b, para_c]))]
    assert len(texts) >= 2
    assert all(len(t) <= MAX_CHUNK_CHARS for t in texts)
    # 段落边界处切分（自然边界，不要求重叠）
    assert texts[0].startswith("段落甲")


def test_long_section_parent_windows_contain_children(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    para = "支持七天无理由退货。" * 100  # ~1000 字
    body = "\n\n".join(para for _ in range(12))
    _write(kb, "long.md", f"# 政策\n\n## 七天无理由\n\n{body}\n")
    chunks = chunk_kb_dir(kb, parent_child=True)
    section = [c for c in chunks if c.section == "七天无理由"]
    assert len(section) > 1  # 章节被拆成多个子块
    for c in section:
        # 父块窗口必须完整包含命中子块正文，且 ≤4000、不含前缀
        assert c.parent_text
        assert len(c.parent_text) <= 4000
        assert chunk_body(c.text) in c.parent_text
        assert not c.parent_text.startswith("【")
    # 同章节子块共用 parent_id；不同章节不同
    pids = {c.parent_id for c in section}
    assert len(pids) == 1
    others = [c for c in chunks if c.section != "七天无理由"]
    assert all(c.parent_id not in pids for c in others)


def test_parent_window_covers_tail_hit_for_10k_section(tmp_path):
    """10k 字章节：每个窗口都包含对应子块，末尾命中的窗口贴着章节末尾。"""
    kb = tmp_path / "kb"
    kb.mkdir()
    para = "这是长章节中的一段测试内容，用于验证父块窗口的包含性。"
    body = "\n\n".join(para for _ in range(160))
    _write(kb, "huge.md", f"# 政策\n\n## 长章节\n\n{body}\n")
    section = [c for c in chunk_kb_dir(kb) if c.section == "长章节"]
    assert len(section) >= 5
    for c in section:
        assert chunk_body(c.text) in c.parent_text
        assert len(c.parent_text) <= 4000
    # 末尾子块的窗口延伸到章节末尾（body 末尾 == 章节末尾）
    assert section[-1].parent_text.endswith(chunk_body(section[-1].text))


def test_split_long_spans_increase_for_identical_paragraphs():
    """重复段落各归其位：start 严格递增且 piece == text[start:end]。

    回归：旧实现事后 find 定位时游标不前移，相同内容的段落全部命中
    第一次出现的位置（start 均为 0），父窗口错位。
    """
    para = "重复段落内容测试。" * 100  # 900 字
    text = "\n\n".join([para] * 4)  # ~3600 字，逐段打包成块
    pieces = _split_long(text)
    assert len(pieces) == 4
    starts = [p[1] for p in pieces]
    assert len(set(starts)) == len(starts) and starts == sorted(starts)
    assert starts[1] > starts[0]  # 第二块在第二段的位置，而非首段
    for t, s, e in pieces:
        assert text[s:e] == t


def test_identical_paragraph_windows_contain_own_children(tmp_path):
    """内容完全相同的长段落：每个父窗口都包含自己的子块（而非首段窗口）。"""
    kb = tmp_path / "kb"
    kb.mkdir()
    para = "公共填充内容，用于验证重复段落定位的正确性。" * 50  # ~760 字
    body = "\n\n".join([para] * 6)  # ~4600 字，超过父窗口上限
    _write(kb, "dup.md", f"# 政策\n\n## 重复章节\n\n{body}\n")
    section = [c for c in chunk_kb_dir(kb) if c.section == "重复章节"]
    assert len(section) >= 5
    for c in section:
        # 旧缺陷下，后段落定位到首段位置 → 窗口不含末段正文 → parent 被清空
        assert c.parent_text, "重复段落定位失败：父窗口未包含对应子块"
        assert chunk_body(c.text) in c.parent_text
    # 窗口随子块位置滑动：首末窗口不重合
    assert section[0].parent_text != section[-1].parent_text


def test_atomic_image_and_link_not_cut():
    img = "![商品图](https://example.com/a.png)"
    link = "[退换货政策](https://example.com/policy)"
    filler = "这是一段足够长的填充文本，用于把内容推过切分边界。"
    para = "。".join([filler] * 40) + img + "。" \
        + "。".join([filler] * 40) + link + "。"
    texts = [p[0] for p in _split_long(para)]
    assert len(texts) >= 2
    # 图片/链接完整出现在某一块中，不被从语法内部截断
    assert any(img in t for t in texts)
    assert any(link in t for t in texts)


def test_code_block_atomic_unless_over_hard_cap():
    body = "\n".join(f"line_{i} = 'value_{i}'" for i in range(30))
    code = f"```python\n{body}\n```"
    filler = "这是一段足够长的填充文本。" * 60
    para = filler + "。" + code + "。" + filler
    texts = [p[0] for p in _split_long(para)]
    assert any(code in t for t in texts)  # 代码块整体保留


# ============================================================
# 3. 多格式结构保留
# ============================================================
def _build_pdf(page_texts: list[str]) -> bytes:
    """构造最小合法多页 PDF（每页一个文本流，含正确 xref）。"""
    header = b"%PDF-1.4\n"
    n = len(page_texts)
    body: list[bytes] = [
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n",
        f"2 0 obj<</Type/Pages/Kids[{' '.join(f'{4 + 2 * i} 0 R' for i in range(n))}]"
        f"/Count {n}>>endobj\n".encode(),
        b"3 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n",
    ]
    for i, t in enumerate(page_texts):
        page_num, content_num = 4 + 2 * i, 5 + 2 * i
        stream = f"BT /F1 12 Tf 72 720 Td ({t}) Tj ET".encode("latin-1", "replace")
        body.append(
            f"{page_num} 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
            f"/Contents {content_num} 0 R/Resources<</Font<</F1 3 0 R>>>>>>endobj\n".encode()
        )
        body.append(
            f"{content_num} 0 obj<</Length {len(stream)}>>stream\n".encode()
            + stream + b"\nendstream\nendobj\n"
        )
    xref_pos = len(header) + sum(len(b) for b in body)
    xref = [f"xref\n0 {3 + 2 * n + 1}\n".encode(), b"0000000000 65535 f \n"]
    pos = len(header)
    for obj in body:
        xref.append(f"{pos:010d} 00000 n \n".encode())
        pos += len(obj)
    trailer = (
        f"trailer<</Size {3 + 2 * n + 1}/Root 1 0 R>>\nstartxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return header + b"".join(body) + b"".join(xref) + trailer


def test_pdf_pages_become_sections(tmp_path):
    pytest.importorskip("pypdf")
    f = tmp_path / "多页政策.pdf"
    f.write_bytes(_build_pdf([
        "Page one about return policy.",
        "Page two about shipping.",
    ]))
    text = parse_document(f)
    assert text.startswith("# 多页政策")
    assert "## 第 1 页" in text and "## 第 2 页" in text
    assert "Page one" in text and "Page two" in text

    chunks = chunk_kb_dir(tmp_path)
    paths = {c.heading_path for c in chunks if c.doc == "多页政策"}
    assert "多页政策 > 第 1 页" in paths
    assert "多页政策 > 第 2 页" in paths


def test_scanned_pdf_clear_error(tmp_path):
    pytest.importorskip("pypdf")
    f = tmp_path / "scan.pdf"
    f.write_bytes(_build_pdf(["", ""]))
    with pytest.raises(ValueError, match="扫描件|无可提取文本"):
        parse_document(f)


def test_pdf_empty_pages_skipped(tmp_path):
    pytest.importorskip("pypdf")
    f = tmp_path / "含空页.pdf"
    f.write_bytes(_build_pdf(["First page.", "", "Third page."]))
    text = parse_document(f)
    assert "## 第 1 页" in text and "## 第 3 页" in text
    assert "## 第 2 页" not in text


def test_docx_headings_tables_document_order(tmp_path):
    docx = pytest.importorskip("docx")
    from docx import Document

    doc = Document()
    doc.add_heading("退换货政策", level=1)
    doc.add_paragraph("支持七天无理由退货。")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "时效"
    table.rows[0].cells[1].text = "7天"
    doc.add_heading("特殊商品", level=2)
    doc.add_paragraph("生鲜不支持七天无理由。")
    f = tmp_path / "政策.docx"
    doc.save(str(f))

    text = parse_document(f)
    assert "# 退换货政策" in text and "## 特殊商品" in text
    assert "| 时效 | 7天 |" in text
    # 文档顺序：标题 → 正文 → 表格 → 标题 → 正文（修复旧实现的段落/表格分离）
    assert text.index("# 退换货政策") < text.index("支持七天无理由退货") \
        < text.index("| 时效") < text.index("## 特殊商品") < text.index("生鲜不支持")

    # 落进 KB 后标题路径生效
    kb = tmp_path / "kb"
    kb.mkdir()
    doc.save(str(kb / "政策.docx"))
    chunks = chunk_kb_dir(kb)
    paths = {c.heading_path for c in chunks if c.doc == "政策"}
    assert "退换货政策" in paths
    assert "退换货政策 > 特殊商品" in paths


def test_html_headings_lists_and_noise_filtered(tmp_path):
    f = tmp_path / "p.html"
    f.write_text(
        "<html><head><title>页面标题</title>"
        "<style>.a { color: red; }</style></head><body>"
        "<h1>配送政策</h1><script>alert(1)</script>"
        "<h2>偏远地区</h2><p>偏远地区不包邮</p>"
        "<ul><li>北京次日达</li><li>上海次日达</li></ul>"
        "<h6>附则</h6><p>最终解释权归平台</p>"
        "</body></html>",
        encoding="utf-8",
    )
    text = parse_document(f)
    assert "# 配送政策" in text
    assert "## 偏远地区" in text
    assert "- 北京次日达" in text and "- 上海次日达" in text
    assert "###### 附则" in text
    assert "alert" not in text and "color: red" not in text and "页面标题" not in text


def test_doc_rejected_with_conversion_hint(tmp_path):
    f = tmp_path / "旧文档.doc"
    f.write_bytes(b"legacy doc bytes")
    with pytest.raises(ValueError, match=r"\.docx"):
        parse_document(f)


def test_doc_rejected_at_upload_format_gate():
    from app.agent.rag.upload_service import UploadError, _format_of

    with pytest.raises(UploadError, match="docx"):
        _format_of("旧文档.doc")


# ============================================================
# 4. 父子块进入检索结果
# ============================================================
def _hit(cid: str, pid: str = "", ptext: str = "", score: float = 0.5,
         text: str = "t") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=cid, doc="d", section="s", text=text,
                    heading_path="d > s", parent_id=pid, parent_text=ptext,
                    source_path="x.md"),
        score=score,
    )


def test_collapse_by_parent_keeps_first_and_truncates():
    hits = [
        _hit("c1", "p1", "章节全文", 0.9),
        _hit("c2", "p1", "章节全文", 0.8),  # 同父块 → 丢弃
        _hit("c3", "p2", "", 0.7),
        _hit("c4", "", "", 0.6),
    ]
    out = collapse_by_parent(hits, top_k=3)
    assert [h.chunk.chunk_id for h in out] == ["c1", "c3", "c4"]


def test_collapse_by_parent_without_pid_old_index():
    hits = [_hit("c1", "", "", 0.9), _hit("c2", "", "", 0.8), _hit("c3", "", "", 0.7)]
    out = collapse_by_parent(hits, top_k=2)
    assert [h.chunk.chunk_id for h in out] == ["c1", "c2"]


def test_search_knowledge_parent_context_and_dedup(reset_settings, monkeypatch):
    """search_knowledge：父块去重 + matched/text/context_type 输出。"""
    from app.agent.tools import knowledge as knowledge_tool
    from app.config.settings import settings

    class _FakeRetriever:
        def search(self, query, top_k=3, timeout=None):
            return [
                _hit("c1", "p1", "章节全文含答案：命中子块甲，以及命中子块乙。", 0.9,
                     text="命中子块甲"),
                _hit("c2", "p1", "章节全文含答案：命中子块甲，以及命中子块乙。", 0.8,
                     text="命中子块乙"),
                _hit("c3", "", "", 0.7, text="独立子块"),
            ]

        def load(self):
            pass

        @property
        def size(self):
            return 3

        @property
        def backend(self):
            return None

    knowledge_tool.reset_retriever()
    knowledge_tool.push_retriever_override(_FakeRetriever())
    monkeypatch.setattr(settings, "retrieval_fence_enabled", False)
    monkeypatch.setattr(settings, "rag_min_relevance_score", None)
    try:
        out = knowledge_tool.search_knowledge("七天无理由", top_k=2)
    finally:
        knowledge_tool.pop_retriever_override()
        knowledge_tool.reset_retriever()

    assert out["success"]
    r1, r2 = out["results"]
    assert r1["parent_id"] == "p1" and r1["context_type"] == "parent"
    assert "命中子块甲" in r1["text"]        # 发给模型的是包含命中内容的父块
    assert r1["matched_text"] == "命中子块甲"  # 命中的是子块
    assert r2["context_type"] == "self"
    assert r2["text"] == "独立子块"           # 无父块 → 回退命中子块
    assert r2["heading_path"] == "d > s"


def test_search_knowledge_falls_back_when_parent_missing_child(reset_settings, monkeypatch):
    """父块不含命中子块正文（旧索引/异常数据）→ 回退子块，context_type=self。"""
    from app.agent.tools import knowledge as knowledge_tool
    from app.config.settings import settings

    class _BadParentRetriever:
        def search(self, query, top_k=3, timeout=None):
            return [_hit("c1", "p1", "与命中内容完全无关的窗口", 0.9, text="命中子块")]

        def load(self):
            pass

        @property
        def size(self):
            return 1

        @property
        def backend(self):
            return None

    knowledge_tool.reset_retriever()
    knowledge_tool.push_retriever_override(_BadParentRetriever())
    monkeypatch.setattr(settings, "retrieval_fence_enabled", False)
    monkeypatch.setattr(settings, "rag_min_relevance_score", None)
    try:
        out = knowledge_tool.search_knowledge("任意", top_k=1)
    finally:
        knowledge_tool.pop_retriever_override()
        knowledge_tool.reset_retriever()

    r = out["results"][0]
    assert r["context_type"] == "self"
    assert r["text"] == "命中子块"  # 回退命中子块而非缺失内容的父块


# ============================================================
# 5. 三种后端的新字段持久化
# ============================================================
def _fielded_chunk() -> Chunk:
    return Chunk(chunk_id="c1", doc="d", section="s", text="t",
                 source_path="p.md", provenance="up:1", owner="ops",
                 parent_text="章节原文", heading_path="d > s", parent_id="d#p00")


def test_numpy_backend_new_fields_roundtrip(tmp_path):
    from app.agent.rag.backends.numpy_backend import NumpyBackend

    backend = NumpyBackend(tmp_path / "idx.json")
    backend.upsert([_fielded_chunk()], [[1.0, 0.0]], "m")
    fresh = NumpyBackend(tmp_path / "idx.json")
    fresh.load()
    got = fresh.chunks()[0]
    assert got.heading_path == "d > s"
    assert got.parent_id == "d#p00"
    assert got.parent_text == "章节原文"
    assert got.provenance == "up:1"


def test_chroma_backend_new_fields_roundtrip(tmp_path, chromadb_usable):
    from app.agent.rag.backends.chroma_backend import ChromaBackend

    backend = ChromaBackend(tmp_path / "chroma", collection_name="fields_t")
    backend.upsert([_fielded_chunk()], [[0.1, 0.2, 0.3]], "m")
    fresh = ChromaBackend(tmp_path / "chroma", collection_name="fields_t")
    fresh.load()
    got = fresh.chunks()[0]
    assert got.heading_path == "d > s"
    assert got.parent_id == "d#p00"
    assert got.parent_text == "章节原文"
    hits = fresh.search([0.1, 0.2, 0.3], top_k=1)
    assert hits[0].chunk.parent_id == "d#p00"


class _MiniES:
    """最小 ES 替身：记录 mapping 与 upsert 文档，检索按字段白名单过滤 _source。"""

    _FIELDS = ("chunk_id", "doc", "section", "text", "source_path",
               "provenance", "owner", "parent_text", "heading_path", "parent_id")

    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.last_mapping: dict = {}

        class _Indices:
            def create(_self, index, settings=None, mappings=None):
                self.last_mapping = mappings

            def refresh(_self, index):
                pass

        self.indices = _Indices()

    def bulk(self, operations, index=None, refresh=False):
        pending = None
        for op in operations:
            if "create" in op:
                pending = op["create"]["_id"]
            else:
                self.docs[pending] = op

    def search(self, index, **kw):
        hits = [
            {"_id": d, "_score": 1.0,
             "_source": {f: doc.get(f, "") for f in self._FIELDS}}
            for d, doc in self.docs.items()
        ]
        return {"hits": {"hits": hits}}


def test_es_backend_new_fields_roundtrip():
    from app.agent.rag.backends.es_backend import ESBackend

    backend = ESBackend(_MiniES(), index_name="ecom-kb-g9")
    backend.upsert([_fielded_chunk()], [[1.0, 0.0]], "m")
    hits = backend.search([1.0, 0.0], top_k=1)
    got = hits[0].chunk
    assert got.heading_path == "d > s"
    assert got.parent_id == "d#p00"
    assert got.parent_text == "章节原文"
    props = backend._es.last_mapping["properties"]
    assert "heading_path" in props and "parent_id" in props


def test_numpy_backend_old_index_json_without_new_fields(tmp_path):
    """旧 JSON 索引（无新字段键）加载不炸且新字段为空。"""
    from app.agent.rag.backends.numpy_backend import NumpyBackend

    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "embedding_model": "m",
        "chunks": [{"chunk_id": "c0", "doc": "d", "section": "s", "text": "旧块"}],
        "vectors": [[1.0, 0.0]],
    }, ensure_ascii=False), encoding="utf-8")
    backend = NumpyBackend(path)
    backend.load()
    got = backend.chunks()[0]
    assert got.heading_path == "" and got.parent_id == "" and got.parent_text == ""


# ============================================================
# Review 修正：.doc 扫描、动态预算、统一检索口径
# ============================================================
def test_doc_strict_build_fails_and_non_strict_warns(tmp_path, monkeypatch):
    kb = tmp_path / "kb"
    kb.mkdir()
    _write(kb, "好.md", "# 好\n\n内容\n")
    (kb / "旧.doc").write_bytes(b"legacy")

    warnings: list[str] = []
    monkeypatch.setattr(
        "app.agent.rag.parsers.log.warning", lambda m: warnings.append(m)
    )

    with pytest.raises(ValueError, match="strict 构建中止.*旧\\.doc.*docx"):
        chunk_kb_dir(kb, strict=True)

    chunks = chunk_kb_dir(kb, strict=False)
    assert {c.doc for c in chunks} == {"好"}
    assert len(warnings) == 1
    assert "旧.doc" in warnings[0] and ".docx" in warnings[0]


def test_final_text_cap_with_long_doc_name_and_six_level_path(tmp_path):
    """超长文档名 + 六级标题路径：所有最终 Chunk.text ≤ 1200，正文不丢。"""
    kb = tmp_path / "kb"
    kb.mkdir()
    paras = [f"第{i}段内容：" + "正文测试。" * 50 for i in range(20)]  # ~280字/段
    _write(kb, "doc.md", (
        "# 根标题\n\n## 二级\n\n### 三级\n\n#### 四级\n\n##### 五级\n\n"
        "###### 六级\n\n" + "\n\n".join(paras) + "\n"
    ))
    chunks = chunk_markdown_dir(kb)
    assert chunks
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)
    # 完整标题路径保留在元数据（未因展示截断而丢失）
    deep = [c for c in chunks if c.heading_path.count(" > ") == 5]
    assert deep
    assert all(c.heading_path.startswith("根标题") for c in deep)
    assert all(len(c.text.split("\n", 1)[0]) <= 260 for c in chunks)
    # 正文不丢：每个原始段落完整出现在某个 chunk 中
    texts = "\n".join(c.text for c in chunks)
    assert all(p in texts for p in paras)

    # 超长文档名（超 Windows MAX_PATH 无法落盘）→ 合成 rel 直接驱动切分：
    # 前缀挤占预算，正文按 body_limit 重切，最终文本仍 ≤ 1200 且不丢段落
    rel = "超长文档名称" * 30 + ".md"
    long_named = _chunk_text("# 根标题\n\n" + "\n\n".join(paras) + "\n", rel, tmp_path)
    assert long_named
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in long_named)
    long_texts = "\n".join(c.text for c in long_named)
    assert all(p in long_texts for p in paras)


def test_dynamic_budget_with_long_prefix(tmp_path):
    """前缀挤占预算：body_limit = 1200 - prefix，正文按预算重切而非截断。"""
    rel = "非常长的文档名称" * 40 + ".md"
    raw = "# 根\n\n## 章节\n\n" + "句子内容，这是一句测试。" * 200
    chunks = _chunk_text(raw, rel, tmp_path)
    assert len(chunks) >= 2
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)
    # 每块仍有实质正文（不是被截断的残片）
    assert all(chunk_body(c.text).strip() for c in chunks)


def test_rejects_document_when_prefix_leaves_no_body_room():
    rel = "名" * (MAX_CHUNK_CHARS - 100) + ".md"
    with pytest.raises(ValueError, match="标题/文档名过长"):
        _chunk_text("# 根\n\n正文", rel, Path(tempfile.gettempdir()))


def test_display_label_keeps_root_and_leaf():
    from app.agent.rag.chunker import _display_label

    assert _display_label("A > B") == "A > B"
    long_path = " > ".join(f"标题{i}" for i in range(60))  # ~357 字符
    assert len(long_path) > 240
    label = _display_label(long_path)
    assert label == "标题0 > … > 标题59"  # 根标题 + 末级标题，中间省略
    assert len(label) <= 240


def test_online_tool_and_offline_eval_same_pipeline(reset_settings, monkeypatch):
    """同一假 retriever：线上 search_knowledge 与离线 evaluate 结果同口径。"""
    from app.agent.tools import knowledge as knowledge_tool
    from app.config.settings import settings
    from app.evaluation.retrieval_metrics import evaluate

    def _mk(cid, pid, score, ptext, text):
        return RetrievedChunk(
            chunk=Chunk(chunk_id=cid, doc="d", section="s", text=text,
                        heading_path="d > s", parent_id=pid, parent_text=ptext,
                        source_path="d.md"),
            score=score,
        )

    class _FakeRetriever:
        def search(self, query, top_k=3, timeout=None):
            return [
                _mk("c1", "p1", 0.9, "父窗口：命中子块甲在这里。", "命中子块甲"),
                _mk("c2", "p1", 0.8, "父窗口：命中子块甲在这里。", "命中子块乙"),
                _mk("c3", "p2", 0.3, "", "低分子块"),
            ]

        def load(self):
            pass

        @property
        def size(self):
            return 3

        @property
        def backend(self):
            return None

    retriever = _FakeRetriever()
    monkeypatch.setattr(settings, "retrieval_fence_enabled", False)
    monkeypatch.setattr(settings, "rag_min_relevance_score", 0.5)

    from app.agent.rag.retriever_factory import final_search

    # 线上工具
    knowledge_tool.reset_retriever()
    knowledge_tool.push_retriever_override(retriever)
    try:
        out = knowledge_tool.search_knowledge("问题", top_k=2)
    finally:
        knowledge_tool.pop_retriever_override()
        knowledge_tool.reset_retriever()

    # 离线评测（同阈值、同 top_k 口径）与统一口径直查
    report = evaluate(
        [{"id": "x", "query": "问题", "expected": ["d.md"], "k": 2}],
        retriever, min_score=0.5,
    )
    case = report["cases"][0]

    # 0.5 阈值过滤掉 c3，父块 p1 去重折叠 c2 → 三处同为 [c1]
    assert [r["matched_text"] for r in out["results"]] == ["命中子块甲"]
    direct = final_search(retriever, "问题", 2, min_score=0.5)
    assert [h.chunk.chunk_id for h in direct.hits] == ["c1"]
    assert case["collapsed_hit_keys"] == ["p1"]
    assert case["collapsed_parents"] == 1
    assert case["raw_candidates"] == 3
    assert case["gated_candidates"] == 2
    assert case["accepted_hits"] == 1
    assert case["threshold_applied"] == 0.5
    assert case["recall_at_k"] == 1.0
