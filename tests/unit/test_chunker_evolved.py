"""chunker：rglob 递归、evolved 显示名、source_path、旧索引反序列化兼容。"""

from __future__ import annotations

from app.agent.rag.chunker import (
    EVOLVED_DOC_NAME,
    Chunk,
    chunk_markdown_dir,
)


def test_rglob_includes_evolved_subdir(tmp_kb_dir):
    chunks = chunk_markdown_dir(tmp_kb_dir)
    docs = {c.doc for c in chunks}
    assert "退货政策" in docs  # 根文档 doc=stem（展示名不变）
    assert "配送说明" in docs
    assert EVOLVED_DOC_NAME in docs  # evolved/ 下统一展示名


def test_root_chunk_ids_and_source_path(tmp_kb_dir):
    chunks = chunk_markdown_dir(tmp_kb_dir)
    root = [c for c in chunks if c.doc == "退货政策"]
    assert root
    for c in root:
        # chunk_id 保持旧格式：{stem}#{idx:02d}-{sub_idx:02d}
        assert c.chunk_id.startswith("退货政策#")
        # 根文档 source_path 无 "/"（GroundingJudge 的证据集边界）
        assert "/" not in c.source_path
        assert c.source_path == "退货政策.md"


def test_evolved_chunk_id_unique_and_path(tmp_kb_dir):
    chunks = chunk_markdown_dir(tmp_kb_dir)
    evolved = [c for c in chunks if c.doc == EVOLVED_DOC_NAME]
    assert evolved
    ids = [c.chunk_id for c in evolved]
    assert len(set(ids)) == len(ids)  # 文件间不重复
    for c in evolved:
        assert c.source_path.startswith("evolved/")
        assert "/" in c.source_path
        # chunk_id 用无扩展名相对路径保证唯一
        assert c.chunk_id.startswith(c.source_path[: -len(".md")])


def test_old_index_dict_deserializes(tmp_kb_dir):
    """旧索引（无 source_path 键）Chunk(**c) 反序列化不炸。"""
    old = {"chunk_id": "退货政策#00-00", "doc": "退货政策",
           "section": "七天无理由", "text": "内容"}
    chunk = Chunk(**old)
    assert chunk.source_path == ""  # 默认空
    assert chunk.chunk_id == "退货政策#00-00"


def test_new_chunk_dumps_source_path(tmp_kb_dir):
    chunk = chunk_markdown_dir(tmp_kb_dir)[0]
    dumped = chunk.to_dict()
    assert "source_path" in dumped
    assert dumped["source_path"] == chunk.source_path


# ============================================================
# 7.1 frontmatter 元数据 + .txt 接入
# ============================================================
def test_frontmatter_meta_lands_on_chunks_and_not_in_text(tmp_kb_dir):
    """frontmatter 的 provenance/owner 透传进 chunk；frontmatter 不进正文。"""
    (tmp_kb_dir / "售后政策.md").write_text(
        "---\nprovenance: turn-abc123\nowner: system\n---\n"
        "# 售后政策\n\n## 质量问题\n\n支持换货，运费由顾客承担。\n",
        encoding="utf-8",
    )
    chunks = chunk_markdown_dir(tmp_kb_dir)
    with_fm = [c for c in chunks if c.doc == "售后政策"]
    assert with_fm
    for c in with_fm:
        assert c.provenance == "turn-abc123"
        assert c.owner == "system"
        # frontmatter 行不得出现在 chunk text 里
        assert "provenance" not in c.text
        assert "owner" not in c.text
        assert not c.text.startswith("---")


def test_no_frontmatter_docs_default_meta(tmp_kb_dir):
    """无 frontmatter 的文档 chunk.provenance / owner 为 ""（缺省）。"""
    chunks = chunk_markdown_dir(tmp_kb_dir)
    assert chunks
    for c in chunks:
        assert c.provenance == ""
        assert c.owner == ""


def test_txt_file_is_chunked(tmp_kb_dir):
    """.txt 接入：doc=stem、source_path 带 .txt 后缀、正常切分。"""
    (tmp_kb_dir / "常见问题.txt").write_text(
        "# 常见问题\n\n## 发货时效\n\n付款后 48 小时内发货。\n",
        encoding="utf-8",
    )
    chunks = chunk_markdown_dir(tmp_kb_dir)
    txt = [c for c in chunks if c.doc == "常见问题"]
    assert txt
    for c in txt:
        assert c.source_path == "常见问题.txt"
        assert "发货时效" in c.text