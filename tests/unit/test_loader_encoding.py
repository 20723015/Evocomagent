"""批次4 单测：文本编码回退（utf-8 → gb18030）与二进制拒绝。

覆盖计划点名的 5 处调用点中的用户输入路径：md/txt 直读（parse_document、
chunk_kb_dir）、html、以及 CI 门禁 check_kb_governance。chunker._chunk_one_file
经 chunk_kb_dir 的 md/txt 分支间接覆盖。
"""

from __future__ import annotations

import pytest

from app.agent.rag.loader import decode_text
from app.agent.rag.parsers import chunk_kb_dir, parse_document

GBK_MD = (
    "---\n"
    "status: active\n"
    "authority: platform\n"
    "effective_date: 2026-01-01\n"
    "---\n"
    "# 退换货政策\n\n"
    "支持七天无理由退货。\n"
)


# ============================================================
# 1. decode_text 语义
# ============================================================
def test_decode_text_utf8_priority():
    assert decode_text("中文内容".encode("utf-8")) == "中文内容"


def test_decode_text_gb18030_fallback():
    raw = "支持七天无理由退货。".encode("gbk")
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")  # 先证明确非 utf-8
    assert decode_text(raw) == "支持七天无理由退货。"


def test_decode_text_rejects_binary():
    with pytest.raises(ValueError, match="解码失败"):
        decode_text(b"\x00\x01\x02\x03" * 64)


def test_decode_text_rejects_nul_free_binary():
    """无 NUL 的二进制：gb18030 能解出内容，靠控制字符占比守卫拒绝。"""
    raw = bytes(range(1, 256)) * 4
    with pytest.raises(ValueError, match="解码失败"):
        decode_text(raw)


def test_decode_text_empty_is_text():
    assert decode_text(b"") == ""


# ============================================================
# 2. 各格式 GBK 文件正确解析
# ============================================================
def test_gbk_markdown_with_frontmatter(tmp_path):
    f = tmp_path / "退换货政策.md"
    f.write_bytes(GBK_MD.encode("gbk"))

    assert "支持七天无理由退货。" in parse_document(f)

    chunks = chunk_kb_dir(tmp_path)
    assert any(c.doc == "退换货政策" for c in chunks)
    assert any("支持七天无理由退货。" in c.text for c in chunks)


def test_gbk_txt_file(tmp_path):
    f = tmp_path / "说明.txt"
    f.write_bytes("配送范围仅限大陆地区。".encode("gbk"))

    assert "配送范围仅限大陆地区。" in parse_document(f)


def test_gbk_html_not_garbled(tmp_path):
    """原实现 errors="replace" 会让 GBK 页面静默乱码入库。"""
    f = tmp_path / "p.html"
    f.write_bytes("<h1>配送政策</h1><p>偏远地区不包邮</p>".encode("gbk"))

    text = parse_document(f)

    assert "# 配送政策" in text
    assert "偏远地区不包邮" in text
    assert "\ufffd" not in text  # 替换字符不得出现


# ============================================================
# 3. 二进制仍失败（strict 语义不变）
# ============================================================
def test_binary_txt_still_fails(tmp_path):
    f = tmp_path / "bogus.txt"
    f.write_bytes(bytes(range(256)) * 4)

    with pytest.raises(ValueError, match="解码失败|二进制"):
        parse_document(f)


def test_binary_txt_in_strict_build_aborts(tmp_path):
    f = tmp_path / "bogus.txt"
    f.write_bytes(b"\x00\x01\x02" * 128)

    with pytest.raises(ValueError, match="strict 构建中止"):
        chunk_kb_dir(tmp_path, strict=True)


# ============================================================
# 4. CI 门禁不再因 GBK 崩溃
# ============================================================
def test_gbk_md_passes_governance_check(tmp_path):
    from app.scripts.check_kb_governance import check

    f = tmp_path / "退换货政策.md"
    f.write_bytes(GBK_MD.encode("gbk"))

    assert check(tmp_path) == []
