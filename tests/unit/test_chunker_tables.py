"""批次5 单测：表格行感知切分 + 小数点保护。

背景（评审实证盲区）：``_SENTENCE_END`` 不含 ``\\n`` 但含 ``.`` 与 ``；``，
所以「找不到句末标点才回退行边界」的原设计永不触发——含 ``0.5%`` 的费率行
会被 ``.`` 命中、在数字中间切开；纯文字表格行则落到 ``_hard_spans`` 按字符
硬切。改为表格行原子化（行内禁用标点断点、只在行边界分段）。

回归口径：非表格文本的**切分输出**（``_split_long``）与批次5 前逐字节一致，
唯一例外是小数点保护。对比方式是把 ``_sentence_spans`` 换回旧实现，其余
（``_force_split`` / ``_split_long``）保持不动——直接验证可观察契约。
"""

from __future__ import annotations

from app.agent.rag import chunker
from app.agent.rag.chunker import (
    _SENTENCE_END,
    _atomic_spans,
    _hard_spans,
    _in_atomic,
    _sentence_spans,
    _split_long,
)

LIMIT = 1200


def _legacy_sentence_spans(p: str, limit: int) -> list[tuple[int, int]]:
    """批次5 前的 ``_sentence_spans``（仅用于回归对照）。"""
    atomic = _atomic_spans(p)
    cuts = [i + 1 for i, ch in enumerate(p)
            if ch in _SENTENCE_END and not _in_atomic(i + 1, atomic)]
    bounds = [0] + cuts + [len(p)]
    out: list[tuple[int, int]] = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a > limit:
            out.extend(_hard_spans(p, a, b, limit))
        else:
            out.append((a, b))
    return out


def _rate_rows(n: int, line: str = "| 分期手续费 | 0.5% | 3期 |") -> str:
    return "\n".join([line] * n)


# ============================================================
# 1. 表格行不被拦腰切开
# ============================================================
def test_rate_table_not_split_mid_number():
    """含 ``0.5%`` 的费率行：不得出现「0.」与「5%」分处两块的切断。"""
    pieces = _split_long(_rate_rows(60), LIMIT)
    texts = [p[0] for p in pieces]

    assert len(texts) >= 2
    assert all(len(t) <= LIMIT for t in texts)
    for t in texts:
        # 每一块都以完整表格行边界结尾（行尾换行或文末）
        assert t.endswith("|\n") or t.endswith("|")
        # 小数点两侧未被切开：出现 "0." 必然紧跟 "5%"
        idx = t.find("0.")
        while idx != -1:
            assert t[idx:idx + 4] == "0.5%", f"数字被切断: {t[idx:idx + 6]!r}"
            idx = t.find("0.", idx + 1)


def test_long_table_block_splits_on_line_boundaries():
    """无标点的长表格块：按行边界切，而不是按字符数硬切。"""
    pieces = _split_long(_rate_rows(40), LIMIT)

    for piece, _, _ in pieces:
        first_line = piece.split("\n", 1)[0]
        assert first_line.startswith("|"), f"块起始不是行边界: {first_line!r}"
        # 每行都是完整行（行内不含换行截断残留）
        assert all(line.startswith("|") for line in piece.rstrip("\n").split("\n"))


def test_table_row_bounds_cover_all_pipe_lines():
    text = "说明如下：\n| a | b |\n| c | d |\n以上。"
    bounds = chunker._table_row_bounds(text)

    assert len(bounds) == 2
    for start, cut in bounds:
        assert text[start] == "|"
        assert text[start:cut].endswith("\n")


# ============================================================
# 2. 小数点保护
# ============================================================
def test_decimal_dot_is_not_a_break():
    text = "费率0.5%分期"
    spans = _sentence_spans(text, 100)

    assert spans == [(0, len(text))]  # 未在 0 与 5 之间切开
    # 对照：点号前不是数字（非小数）时仍按旧语义断句
    assert len(_sentence_spans("费率A.5%分期", 100)) > 1


def test_numbered_list_dot_still_breaks():
    """「数字 + . + 空格」是编号列表，行为与批次5 前一致。"""
    text = "1. 编号说明"
    spans = _sentence_spans(text, 100)

    assert spans == [(0, 2), (2, len(text))]


def test_section_number_not_split_across_chunks():
    """prose 中的章节号（如 PDF 正文里的 2.1.1 / 3.4）不得从数字中间切开。

    回归来源（Review 实证）：基线对比中 `平台治理与处罚总则.pdf` 的 4 个
    chunk 因小数点保护变化——旧实现把「2.1.1 售假」切成 `…\n2.1.` +
    `1 售假`。本用例把该真实触发模式钉死。
    """
    text = "4 治理原则教育与处罚相结合。" + "二、商品类违规2.1 售假2.1.1 认定流程" * 80
    pieces = _split_long(text, LIMIT)
    texts = [p[0] for p in pieces]

    assert len(texts) >= 2
    # 章节号整体出现在单块内，不被切到两块
    assert any("2.1.1 认定流程" in t for t in texts)
    for t in texts:
        assert not t.endswith("2.1") and not t.endswith("2.1.")
        assert not t.startswith("1 售假") and not t.startswith("1 认定")


# ============================================================
# 3. 非表格文本切分与批次5 前逐字节一致
# ============================================================
def test_non_table_splitting_byte_identical(monkeypatch):
    texts = [
        "这是一段用于测试的连续文本，包含多个完整句子。" * 114,
        "第一段。\n\n第二段。\n\n第三段。",
        "无标点的超长文本" * 300,
        "行内代码 `a.b.c` 与链接 [政策](https://example.com/p.md)。" * 40,
        "",
        "短句。",
    ]
    new = [_split_long(t, LIMIT) for t in texts]

    monkeypatch.setattr(chunker, "_sentence_spans", _legacy_sentence_spans)
    old = [_split_long(t, LIMIT) for t in texts]

    assert new == old


def test_decimal_protection_actually_changes_spans():
    """含小数的文本必须与旧实现不同，证明保护生效（否则测试无意义）。"""
    text = "首期费率0.5%，尾期1.5%。"

    assert len(_legacy_sentence_spans(text, LIMIT)) > 1  # 旧实现每处 "." 都断
    assert _sentence_spans(text, LIMIT) == [(0, len(text))]


# ============================================================
# 4. 边界：单行超限仍按字符硬切（最后兜底，已知边界）
# ============================================================
def test_single_overlong_row_still_hard_split():
    row = "| " + "x" * 3000 + " |"
    pieces = _split_long(row, LIMIT)

    assert len(pieces) > 1
    assert all(len(p[0]) <= LIMIT for p in pieces)


# ============================================================
# 5. 表格行命中判定不吃掉正文
# ============================================================
def test_prose_line_starting_with_pipe_not_mistaken():
    """正文里的 ``|`` 只作为行首才算表格行——行中竖线不影响标点切分。"""
    text = "运费|时效如下。"
    spans = _sentence_spans(text, 100)

    assert spans == [(0, len(text))]
    assert chunker._table_row_bounds(text) == []
    assert chunker._table_row_bounds("| a | b |") != []
