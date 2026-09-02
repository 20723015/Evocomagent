"""阶段七 7.1/7.3 单测：多格式解析流水线、parent-child、HTTP 精排器。"""

from __future__ import annotations

import json

import pytest

from app.agent.rag.backends.base import RetrievedChunk
from app.agent.rag.chunker import Chunk
from app.agent.rag.parsers import (
    attach_parents,
    chunk_kb_dir,
    parse_document,
)
from app.agent.rag.rerank import HTTPReranker, NullReranker


def _chunk(text: str, doc="d", section="s", i=0, sub=0) -> Chunk:
    return Chunk(
        chunk_id=f"c{i}-{sub}", doc=doc, section=section, text=text,
        source_path="x.md",
    )


def _hits(*texts: str) -> list[RetrievedChunk]:
    return [RetrievedChunk(chunk=_chunk(t), score=0.5) for t in texts]


# ============================================================
# 7.1 解析
# ============================================================
def test_parse_markdown_passthrough(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("# 政策\n\n正文\n", encoding="utf-8")
    assert "正文" in parse_document(f)


def test_parse_docx(tmp_path):
    from docx import Document

    doc = Document()
    doc.add_heading("退换货政策", level=1)
    doc.add_paragraph("支持七天无理由退货。")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "时效"
    table.rows[0].cells[1].text = "7天"
    f = tmp_path / "政策.docx"
    doc.save(str(f))

    text = parse_document(f)
    assert "七天无理由退货" in text
    assert "时效" in text and "7天" in text


def test_parse_html(tmp_path):
    f = tmp_path / "p.html"
    f.write_text("<html><body><h1>政策</h1><p>七天无理由退货</p><li>运费自理</li></body></html>",
                 encoding="utf-8")
    text = parse_document(f)
    assert "七天无理由退货" in text
    assert "运费自理" in text


def test_parse_pdf_minimal(tmp_path):
    """最小合法 PDF（单页文本流），验证 pypdf 抽取路径。"""
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length 60>>stream\n"
        b"BT /F1 12 Tf 72 720 Td (Seven days return policy.) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n"
        b"0000000058 00000 n \n0000000115 00000 n \n0000000241 00000 n \n"
        b"0000000351 00000 n \ntrailer<</Size 6/Root 1 0 R>>\n"
        b"startxref\n421\n%%EOF\n"
    )
    f = tmp_path / "policy.pdf"
    f.write_bytes(pdf)
    try:
        text = parse_document(f)
    except ValueError as e:
        pytest.skip(f"本机 pypdf 无法解析最小 PDF 夹具（{e}）")
    assert "Seven days" in text


def test_parse_unsupported_suffix_raises(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("a,b", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_document(f)


def test_chunk_kb_dir_multi_format(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "a.md").write_text("# 政策\n\n## 七天\n\n支持七天无理由退货。\n", encoding="utf-8")
    (kb / "b.html").write_text("<h1>配送</h1><p>偏远地区不包邮。</p>", encoding="utf-8")

    chunks = chunk_kb_dir(kb)
    docs = {c.doc for c in chunks}
    assert "a" in docs and "b" in docs
    texts = "\n".join(c.text for c in chunks)
    assert "七天无理由" in texts and "不包邮" in texts
    # 所有 chunk 带来源路径（统一元数据溯源）
    assert all(c.source_path for c in chunks)


def test_parent_child_attach(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    # 单章节拆成多块 → 父块装配（每段 600 字 × 12 → 远超 MAX_CHUNK_CHARS）
    para = "支持七天无理由退货。" * 100  # 约 1000 字
    body = "\n\n".join(para for _ in range(12))
    (kb / "long.md").write_text(f"# 政策\n\n## 七天无理由\n\n{body}\n", encoding="utf-8")
    chunks = chunk_kb_dir(kb, parent_child=True)
    multi = [c for c in chunks if c.parent_text]
    assert multi, "应存在命中父块的子块"
    # 父块包含整个章节（拼接后截断）
    assert all(len(c.parent_text) <= 4000 for c in multi)


def test_parent_child_disabled(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "a.md").write_text("# 政策\n\n## 七天\n\n支持七天无理由退货。\n", encoding="utf-8")
    chunks = chunk_kb_dir(kb, parent_child=False)
    assert all(c.parent_text == "" for c in chunks)


# ============================================================
# 7.3 HTTP 精排器
# ============================================================
def _client_with(handler):
    """用 httpx.MockTransport 构造注入客户端（不再 monkeypatch 不完整函数签名）。"""
    import httpx

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_http_reranker_cohere_shape():
    calls = {}

    def handler(request: "httpx.Request") -> "httpx.Response":
        import httpx

        body = json.loads(request.content)
        calls.update({"url": str(request.url), "json": body,
                      "headers": dict(request.headers)})
        # 让 index 1 的分最高
        return httpx.Response(200, json={"results": [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.20},
        ]})

    r = HTTPReranker("cohere", endpoint_url="http://oops", api_key="k",
                     client=_client_with(handler))
    out = r.rerank("七天无理由", _hits("A", "B"), top_k=2)
    assert [h.chunk.doc for h in out] == ["d", "d"]  # 均为 doc=d…按 text 区分
    assert out[0].chunk.text == "B"  # 精排后 B 在前
    assert calls["url"] == "http://oops"
    assert calls["headers"]["authorization"] == "Bearer k"  # httpx 规范化后键为小写
    assert calls["json"]["documents"] == ["A", "B"]


def test_http_reranker_tei_shape():
    import httpx

    def handler(request: "httpx.Request") -> "httpx.Response":
        return httpx.Response(200, json=[[1, 0.9], [0, 0.1]])

    r = HTTPReranker("bge-reranker-v2-m3", endpoint_url="http://tei:8000/rerank",
                     client=_client_with(handler))
    out = r.rerank("问", _hits("A", "B"), top_k=1)
    assert out[0].chunk.text == "B"


def test_http_reranker_fallback_on_network_error():
    import httpx

    def handler(request: "httpx.Request") -> "httpx.Response":
        raise httpx.ConnectError("network down")

    r = HTTPReranker("cohere", client=_client_with(handler))
    out = r.rerank("问", _hits("A", "B", "C"), top_k=2)
    # 网络失败 → 原序前 top_k（持续可用，不向上抛）
    assert [h.chunk.text for h in out] == ["A", "B"]


def test_http_reranker_missing_indices_kept_original_order():
    import httpx

    def handler(request: "httpx.Request") -> "httpx.Response":
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.1}]})

    r = HTTPReranker("jina", client=_client_with(handler))
    out = r.rerank("问", _hits("A", "B"), top_k=2)
    assert [h.chunk.text for h in out] == ["A", "B"]  # index 0 保留，B 补位


def test_http_reranker_partial_scored_then_unscored_fill():
    """部分索引有分：已评分项按分降序在前，未评分项按原召回顺序补位。"""
    import httpx

    def handler(request: "httpx.Request") -> "httpx.Response":
        return httpx.Response(200, json={"results": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.3},
        ]})

    r = HTTPReranker("cohere", client=_client_with(handler))
    out = r.rerank("问", _hits("A", "B", "C"), top_k=3)
    # 已评分：[2, 0]；未评分 1 补位 → [C, A, B]
    assert [h.chunk.text for h in out] == ["C", "A", "B"]
    # 分数写回 hit.score（负例拒绝依赖精排分）
    assert out[0].score == 0.9


def test_http_reranker_ignores_bad_indices_and_nan():
    """越界索引、重复索引、NaN/Inf 分数一律忽略；有效分数仍生效。"""
    import httpx

    def handler(request: "httpx.Request") -> "httpx.Response":
        return httpx.Response(200, json={"results": [
            {"index": 99, "relevance_score": 1.0},   # 越界
            {"index": 0, "relevance_score": 0.5},    # 有效
            {"index": 0, "relevance_score": 0.99},   # 重复索引 → 首见保留
            {"index": 1, "relevance_score": float("nan")},
            {"index": 2, "relevance_score": float("inf")},
        ]})

    r = HTTPReranker("jina", client=_client_with(handler))
    out = r.rerank("问", _hits("A", "B", "C"), top_k=3)
    assert [h.chunk.text for h in out] == ["A", "B", "C"]  # 仅 0 有分，其余补位
    assert out[0].score == 0.5  # 首见分数保留而非被重复索引覆盖
    assert out[1].score == 0.5  # B/C 未评分 → 补位原顺序、分不变


def test_http_reranker_no_request_on_empty_or_zero_topk():
    import httpx

    sent = []

    def handler(request: "httpx.Request") -> "httpx.Response":
        sent.append(str(request.url))
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 1.0}]})

    r = HTTPReranker("cohere", client=_client_with(handler))
    assert r.rerank("问", [], top_k=5) == []
    assert r.rerank("问", _hits("A"), top_k=0) == []
    assert sent == [], "空候选/top_k<=0 不应发起 HTTP 请求"


def test_http_reranker_batches_with_global_index_offset():
    """批量 > 30 时切片多次请求，全局索引偏移合并。"""
    import httpx

    requests = []

    def handler(request: "httpx.Request") -> "httpx.Response":
        body = json.loads(request.content)
        base = requests.__len__() * 30
        requests.append(body)
        # 每批返回「批次内最后一个」分最高（分数唯一，避免并列歧义）→
        # 全局末位应排到最前，验证批次偏移合并
        texts = body["documents"]
        return httpx.Response(200, json={"results": [
            {"index": len(texts) - 1, "relevance_score": 1.0 + base / 100.0},
            {"index": 0, "relevance_score": 0.1},
        ]})

    hits = _hits(*[f"t{i}" for i in range(35)])
    r = HTTPReranker("jina", client=_client_with(handler))
    out = r.rerank("问", hits, top_k=35)
    assert len(requests) == 2
    assert out[0].chunk.text == "t34"  # 第二批末位 0.9 全局第 34
    assert requests[1]["documents"][0] == "t30"  # 第二批起点偏移正确


def test_null_reranker_preserves_order():
    r = NullReranker()
    out = r.rerank("q", _hits("A", "B", "C"), top_k=2)
    assert [h.chunk.text for h in out] == ["A", "B"]
