"""摘要转录层评审修复的无网络单测（P1-P4）。

- P1：SUMMARY_PROMPT 显式提取「进行中的退款申请/工单」（申请号 + 状态）；
- P2：assistant 块列表 content（required_signed 画像回传形态）只取 text 块，
  不再渲染成 Python repr（summarize 与 extraction._build_transcript 共用）；
- P3：tool_calls.arguments 转录截断（结果侧有 digest 预算，参数侧对齐）；
- P4：摘要调用 max_tokens 按字数预算推导，不再沿用 llm_max_tokens（8192）。
"""

from __future__ import annotations

from types import SimpleNamespace

from app.agent.memory.extraction import _build_transcript
from app.agent.summarizer import _ARGS_CUT, _format_tool_args, summarize
from app.agent.tools.digest import visible_assistant_text
from app.config.settings import settings
from app.prompts.summarizer import SUMMARY_PROMPT


class _FakeClient:
    """记录 create 入参、返回固定摘要文本的最小 OpenAI client 替身。"""

    def __init__(self, response_text: str = "用户咨询订单退款的摘要。"):
        captured: dict = {}

        def create(**kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(content=response_text)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
        self.captured = captured


# required_signed 画像窗口消息的真实形态：thinking 块 + text 块
_THINKING_BLOCKS = [
    {"type": "thinking", "thinking": "用户可能想退款，先查订单", "signature": "sig-1"},
    {"type": "text", "text": "已为您查询到订单 ORD-20240115-001，正在核实物流状态。"},
]


def _assistant_msg(content, tool_calls=None) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return msg


def _tool_call(call_id: str, name: str, arguments) -> list[dict]:
    return [{
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": arguments},
    }]


# ============================================================
# P2：块列表 content 只取 text 块
# ============================================================
class TestVisibleAssistantText:
    def test_str_passthrough(self):
        assert visible_assistant_text("普通回复") == "普通回复"

    def test_blocks_only_text_kept(self):
        out = visible_assistant_text(_THINKING_BLOCKS)
        assert out == "已为您查询到订单 ORD-20240115-001，正在核实物流状态。"

    def test_thinking_only_blocks_empty(self):
        assert visible_assistant_text(
            [{"type": "thinking", "thinking": "内部独白", "signature": "s"}],
        ) == ""

    def test_none_and_other_types_empty(self):
        assert visible_assistant_text(None) == ""
        assert visible_assistant_text(123) == ""


class TestSummarizeTranscript:
    def test_thinking_blocks_no_repr_leak(self):
        client = _FakeClient()
        summarize(client, "test-model", [_assistant_msg(_THINKING_BLOCKS)], None)
        user_content = client.captured["messages"][1]["content"]
        assert "客服：已为您查询到订单" in user_content
        # thinking 块绝不以 Python repr 形态进转录
        assert "signature" not in user_content
        assert "内部独白" not in user_content
        assert "[{" not in user_content

    def test_extraction_transcript_no_repr_leak(self):
        out = _build_transcript([_assistant_msg(_THINKING_BLOCKS)])
        assert "客服：已为您查询到订单" in out
        assert "signature" not in out
        assert "[{" not in out

    def test_string_content_unchanged(self):
        client = _FakeClient()
        summarize(client, "test-model", [_assistant_msg("普通文本回复")], None)
        assert "客服：普通文本回复" in client.captured["messages"][1]["content"]


# ============================================================
# P3：tool_calls.arguments 截断
# ============================================================
class TestToolArgsClamp:
    def test_short_args_verbatim(self):
        args = '{"order_id": "ORD-20240115-001"}'
        client = _FakeClient()
        summarize(
            client, "test-model",
            [_assistant_msg(None, _tool_call("c1", "query_order", args))],
            None,
        )
        user_content = client.captured["messages"][1]["content"]
        assert f"客服：[调用工具 query_order({args})]" in user_content

    def test_long_args_truncated(self):
        long_reason = "退" * 500
        args = '{"reason": "%s"}' % long_reason
        client = _FakeClient()
        summarize(
            client, "test-model",
            [_assistant_msg(None, _tool_call("c1", "submit_refund_application", args))],
            None,
        )
        user_content = client.captured["messages"][1]["content"]
        assert long_reason not in user_content
        line = next(l for l in user_content.splitlines() if "调用工具" in l)
        assert "…" in line
        assert len(line) <= len("客服：[调用工具 submit_refund_application()]") + _ARGS_CUT

    def test_dict_args_serialized(self):
        assert _format_tool_args({"order_id": "X"}) == '{"order_id": "X"}'

    def test_over_cut_args_end_with_ellipsis(self):
        out = _format_tool_args("a" * (_ARGS_CUT + 50))
        assert out.endswith("…")
        assert len(out) == _ARGS_CUT


# ============================================================
# P4：max_tokens 按字数预算推导
# ============================================================
class TestSummaryMaxTokens:
    def test_max_tokens_derived_from_char_budget(self):
        client = _FakeClient()
        summarize(client, "test-model", [{"role": "user", "content": "hi"}], None)
        expected = min(
            int(settings.llm_max_tokens),
            max(int(settings.summary_max_chars), 100) * 2 + 256,
        )
        assert client.captured["max_tokens"] == expected
        # 默认配置（500 字 / 8192）下显著小于旧值的 8192
        assert client.captured["max_tokens"] < int(settings.llm_max_tokens)


# ============================================================
# P1：prompt 显式覆盖进行中的退款申请
# ============================================================
class TestPromptCoversRefundApplication:
    def test_prompt_mentions_in_flight_application(self):
        assert "退款申请" in SUMMARY_PROMPT
        assert "申请号" in SUMMARY_PROMPT
