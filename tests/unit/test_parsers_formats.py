"""批次6 单测：strict 下未知后缀报错，且不误伤辅助文件。

背景：原实现对不在 SUPPORTED_SUFFIXES 的后缀直接 continue（仅 .doc 特殊），
`.xlsx` 传了既不报错也不入库。本批次改为 strict 报错、非 strict 记 warning。

承重墙：生产构建 strict_build=True，而 knowledge/ 下真实存在
`x.pdf.meta.yaml` / `x.docx.meta.yaml` sidecar——sidecar 排除必须先于后缀
判定（且不能用 Path.suffix，它对 `x.pdf.meta.yaml` 得到 `.yaml`），否则
strict 构建直接失败。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.rag.parsers import chunk_kb_dir

KB_DIR = Path(__file__).resolve().parent.parent.parent / "app" / "agent" / "rag" / "knowledge"

DOC = (
    "---\n"
    "status: active\n"
    "authority: platform\n"
    "effective_date: 2026-01-01\n"
    "---\n"
    "# 退换货政策\n\n支持七天无理由退货。\n"
)


def _write_md(kb: Path, name: str = "政策.md") -> None:
    (kb / name).write_text(DOC, encoding="utf-8")


# ============================================================
# 1. 未知后缀
# ============================================================
def test_unknown_suffix_raises_in_strict(tmp_path):
    _write_md(tmp_path)
    (tmp_path / "表.xlsx").write_bytes(b"PK\x03\x04not-really")

    with pytest.raises(ValueError, match="strict 构建中止") as exc:
        chunk_kb_dir(tmp_path, strict=True)

    assert "表.xlsx" in str(exc.value)  # 报错必须带路径，否则无法定位


def test_unknown_suffix_skipped_when_not_strict(tmp_path):
    _write_md(tmp_path)
    (tmp_path / "表.xlsx").write_bytes(b"PK\x03\x04not-really")

    chunks = chunk_kb_dir(tmp_path, strict=False)

    assert chunks  # 正常文档仍被切分
    assert all("表.xlsx" not in c.source_path for c in chunks)


def test_doc_suffix_hint_kept(tmp_path):
    _write_md(tmp_path)
    (tmp_path / "旧.doc").write_bytes(b"legacy")

    with pytest.raises(ValueError, match=r"\.docx"):
        chunk_kb_dir(tmp_path, strict=True)


# ============================================================
# 2. 辅助文件不误伤
# ============================================================
def test_sidecar_not_treated_as_unknown_suffix(tmp_path):
    """`.meta.yaml` 必须被排除，否则 strict 构建直接失败。"""
    _write_md(tmp_path)
    (tmp_path / "政策.md.meta.yaml").write_text(
        "status: active\nauthority: platform\neffective_date: 2026-01-01\n",
        encoding="utf-8",
    )

    chunks = chunk_kb_dir(tmp_path, strict=True)

    assert chunks
    assert all(not c.source_path.endswith(".meta.yaml") for c in chunks)


def test_os_junk_not_treated_as_unknown_suffix(tmp_path):
    _write_md(tmp_path)
    (tmp_path / "Thumbs.db").write_bytes(b"\x00junk")
    (tmp_path / "desktop.ini").write_bytes(b"[.ShellClassInfo]")

    chunks = chunk_kb_dir(tmp_path, strict=True)

    assert chunks


def test_junk_without_valid_doc_still_yields_nothing(tmp_path):
    """仅辅助文件时不应报未知后缀错（非 strict 与 strict 都不应）。"""
    (tmp_path / "Thumbs.db").write_bytes(b"\x00junk")

    assert chunk_kb_dir(tmp_path, strict=False) == []
    assert chunk_kb_dir(tmp_path, strict=True) == []


# ============================================================
# 3. 真实 KB：strict 构建必须通过（承重墙回归）
# ============================================================
def test_repo_kb_strict_build_passes():
    from app.config.settings import settings

    if not KB_DIR.is_dir():
        pytest.skip("知识库目录不存在")

    chunks = chunk_kb_dir(KB_DIR, strict=True)

    assert chunks
    assert settings.rag_doc_metadata_required in (True, False)  # 显式声明依赖配置
    # sidecar 绝不进入索引
    assert all(not c.source_path.endswith(".meta.yaml") for c in chunks)
