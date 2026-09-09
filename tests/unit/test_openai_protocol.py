"""OpenAI Chat Completions 请求形态回归测试。

这些断言锁定模型边界：持久化层的 metadata 只能留在 raw history，
强制终答的 tool_choice 必须使用 Chat Completions 的嵌套 function 形态。
"""

from __future__ import annotations

from app.agent.chat import EcomAgent
from app.agent.context_builder import fold_history
from app.config.settings import settings
from tests.unit.conftest import FakeChatClient


def _agent(tmp_path, client):
    agent = EcomAgent(
        session_path=str(tmp_path / "openai-protocol.json"),
        client=client,
        memory_enabled=False,
        use_mcp=False,
    )
    agent.context_builder._window = 8192
    return agent


def test_fold_history_strips_persistence_metadata_for_every_message():
    raw = [
        {
            "role": "user",
            "content": "历史问题",
            "metadata": {"internal": "user-secret"},
        },
        {
            "role": "assistant",
            "content": "历史答复",
            "metadata": {
                "turn_id": "turn-1",
                "pending_writes": [{"token": "refund-secret"}],
            },
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "工具结果",
            "metadata": {"audit": "internal-only"},
        },
    ]

    folded = fold_history(raw)

    assert all("metadata" not in message for message in folded)
    assert folded[1]["content"] == "历史答复"
    assert raw[1]["metadata"]["pending_writes"][0]["token"] == "refund-secret"


def test_chat_request_has_no_persistence_metadata(tmp_path, reset_settings):
    client = FakeChatClient().enqueue_final_response("已处理。", intent="other")
    agent = _agent(tmp_path, client)
    agent.raw_messages.extend([
        {"role": "user", "content": "历史问题", "metadata": {"secret": "u"}},
        {
            "role": "assistant",
            "content": "历史答复",
            "metadata": {"turn_id": "turn-1", "secret": "a"},
        },
    ])

    agent.chat("继续处理")

    request_messages = client.calls[0][1]["messages"]
    assert all("metadata" not in message for message in request_messages)
    assert agent.raw_messages[1]["metadata"]["secret"] == "a"


def test_forced_final_response_uses_nested_function_tool_choice(
    tmp_path, reset_settings, monkeypatch,
):
    monkeypatch.setattr(settings, "max_react_steps", 1)
    from app.agent.tools import registry

    monkeypatch.setitem(
        registry._TOOL_MAP,
        "query_product",
        lambda keyword, ctx=None: {"success": True, "products": []},
    )
    client = (
        FakeChatClient()
        .enqueue_tool_call("call-1", "query_product", {"keyword": "耳机"})
        .enqueue_final_response("已为您查询。", intent="product_consult")
    )
    agent = _agent(tmp_path, client)

    result = agent.chat("查一下耳机")

    assert result.reply == "已为您查询。"
    assert client.calls[1][1]["tool_choice"] == {
        "type": "function",
        "function": {"name": "final_response"},
    }
