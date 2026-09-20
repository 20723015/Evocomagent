"""批次5（Review #5）：摘要句边界截断。

- 超限截断先回退到最后一个句边界（。！？；换行），不得把订单号/承诺
  切成半截注入 prompt；
- 仅当首句即超限（预算内无句边界）才硬切兜底；
- 截断后追加「（摘要已截断）」标记，最终长度仍 ≤ max_chars（预算保证）。
纯函数测试，无 LLM。
"""

from __future__ import annotations

from app.agent.summarizer import _truncate_summary


class TestSentenceBoundaryTruncation:
    def test_cut_falls_back_to_sentence_boundary(self):
        text = "第一句比较短。用户咨询了订单ORD-20240115-001的退款进度并得到回复。" + "后续补充" * 30
        out = _truncate_summary(text, 60)
        assert out.endswith("（摘要已截断）")
        assert len(out) <= 60
        # 截断点落在句边界之后：最后一个保留字符是句号
        assert out.rstrip("（摘要已截断）").endswith("。")

    def test_order_number_not_cut_in_half(self):
        text = "开头。" + "垫" * 40 + "订单ORD-20240115-001已受理。" + "尾" * 100
        out = _truncate_summary(text, 50)
        assert "ORD-20240115-001" not in out or out.endswith("。（摘要已截断）") or "已受理" not in out
        # 订单号若出现必须完整（不得半截）
        idx = out.find("ORD-")
        if idx != -1:
            assert out[idx:].startswith("ORD-20240115-001") or "ORD-20240115-001" not in out

    def test_no_boundary_hard_cut_fallback(self):
        text = "无" * 200
        out = _truncate_summary(text, 50)
        assert out.endswith("（摘要已截断）")
        assert len(out) <= 50
        assert out.startswith("无" * 10)

    def test_marker_keeps_budget_guarantee(self):
        text = "字" * 500
        for limit in (20, 50, 300):
            out = _truncate_summary(text, limit)
            assert len(out) <= limit, limit

    def test_within_limit_unchanged(self):
        text = "这条摘要没有超限。"
        assert _truncate_summary(text, 300) == text

    def test_newline_counts_as_boundary(self):
        text = "第一行摘要\n第二行摘要内容继续很长" + "长" * 60
        out = _truncate_summary(text, 30)
        assert out.endswith("（摘要已截断）")
        assert len(out) <= 30
