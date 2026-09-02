"""阶段四 4.1~4.4 单测：print 清零、韧性客户端、SSE 帧、指标端点。"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest
from openai import APIConnectionError, BadRequestError

from app.config.settings import settings
from app.server.main import _sse
from app.observability import metrics


APP_ROOT = Path(__file__).resolve().parents[2] / "app"


def _iter_py_files(root: Path):
    for p in root.rglob("*.py"):
        if "__pycache__" not in p.parts:
            yield p


# ============================================================
# 4.1 print 清零（结构化日志替换）
# ============================================================
def test_no_print_in_core_modules():
    offenders = []
    for p in _iter_py_files(APP_ROOT):
        text = p.read_text(encoding="utf-8")
        # 排除注释/字符串里的字面量太低效——直接数裸 print( 调用
        for m in re.finditer(r"(?<![\w.])print\s*\(", text):
            line = text[:m.start()].count("\n") + 1
            offenders.append(f"{p.relative_to(APP_ROOT)}:{line}")
    assert offenders == [], f"以下位置仍使用 print（应改为 structlog）: {offenders}"


def test_no_print_in_main_cli():
    main_py = Path(__file__).resolve().parents[2] / "main.py"
    text = main_py.read_text(encoding="utf-8")
    assert not re.search(r"(?<![\w.])print\s*\(", text), "main.py 仍有 print"


# ============================================================
# 4.4 韧性 LLM 客户端
# ============================================================
def _install_internal_error(iterations):
    """fake create：前 iterations 次抛可重试异常，随后返回 chat_response。"""
    def make():
        state = {"n": 0}

        def fn(kind, kwargs):
            from conftest import chat_response

            state["n"] += 1
            if state["n"] <= iterations:
                raise APIConnectionError(request=httpx.Request("POST", "http://x"))
            return chat_response(f"ok-{state['n']}")

        return fn
    return make()


def test_retry_then_success():
    from conftest import FakeChatClient
    from app.llm.client import install_resilience

    client = FakeChatClient()
    # 每次尝试消耗一个剧本条目（共享闭包计数：前 2 次抛、第 3 次成功）
    fn = _install_internal_error(2)
    client.enqueue_callable(fn).enqueue_callable(fn).enqueue_callable(fn)
    install_resilience(client, "m1", max_retries=3, max_concurrent=4,
                       fallback_model="")
    resp = client.chat.completions.create(model="m1", messages=[{"role": "user", "content": "hi"}])
    assert resp.choices[0].message.content == "ok-3"
    # 重试前会真实重调（3 次尝试）
    assert len(client.calls) == 3


def test_retry_exhausted_then_fallback_model():
    from conftest import FakeChatClient
    from app.llm.client import install_resilience

    client = FakeChatClient()
    fn = _install_internal_error(99)
    client.enqueue_callable(fn).enqueue_callable(fn)  # 2 次尝试全失败（retries=1）
    client.enqueue("fallback-ok")  # 降级链备用模型成功
    install_resilience(client, "m1", max_retries=1, fallback_model="m2")
    resp = client.chat.completions.create(model="m1", messages=[])
    assert resp.choices[0].message.content == "fallback-ok"
    assert client.calls[-1][1]["model"] == "m2"


def test_bad_request_not_retried():
    from conftest import FakeChatClient
    from app.llm.client import install_resilience

    client = FakeChatClient()
    def _bad(kind, kwargs):
        raise BadRequestError(
            "bad",
            response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
            body={"error": "bad"},
        )

    client.enqueue_callable(_bad)
    install_resilience(client, "m1", max_retries=3, fallback_model="m2")
    with pytest.raises(BadRequestError):
        client.chat.completions.create(model="m1", messages=[])
    assert len(client.calls) == 1  # 未重试


def test_cheap_model_routing_for_extract(reset_settings):
    settings.extraction_model = "cheap-model"
    from app.llm.client import _purpose_model

    assert _purpose_model("extract", "gpt-4o-mini") == "cheap-model"
    assert _purpose_model("summarize", "gpt-4o-mini") == "cheap-model"
    assert _purpose_model("react", "gpt-4o-mini") == "gpt-4o-mini"  # 主任务不走廉价


# ============================================================
# 4.5 SSE 帧 & 指标
# ============================================================
def test_sse_frame_format():
    frame = _sse("thought", {"text": "先查一下订单"})
    assert frame.startswith("event: thought\ndata: ")
    payload = json.loads(frame.split("data: ", 1)[1])
    assert payload == {"text": "先查一下订单"}
    assert frame.endswith("\n\n")


def test_metrics_counters_increment():
    from prometheus_client import REGISTRY

    # 同一进程内计数器对象复用；验证可观测递增
    before = metrics.TOOL_CALLS.labels(tool="query_order", result="ok")._value.get()
    metrics.record_tool_call("query_order", "ok")
    after = metrics.TOOL_CALLS.labels(tool="query_order", result="ok")._value.get()
    assert after == before + 1

    metrics.record_conflict()
    assert metrics.SESSION_CONFLICT._value.get() >= 1


def test_sse_stream_event_sequence(monkeypatch):
    """4.5：流式端到端——meta → thought → tool_call → tool_result → reply → end。"""
    from fastapi.testclient import TestClient
    import app.server.main as main_mod

    async def fake_turn(agent, message, on_start=None):
        if on_start is not None:
            on_start()
        cb = agent.event_callback
        cb("thought", {"text": "思考中"})
        cb("tool_call", {"name": "query_order", "arguments": {"order_id": "1"}})
        cb("tool_result", {"result": "订单已找到"})
        from conftest import sample_response

        return sample_response(reply="最终答案")

    from test_server_api import _FakeComponents, _ScriptedAgent

    monkeypatch.setattr(main_mod.runtime, "run_agent_turn", fake_turn)
    monkeypatch.setattr(main_mod, "build_pod_components", lambda: _FakeComponents())

    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda user_id, session_id="", components=None, **_kw: _ScriptedAgent(user_id, session_id),
    )
    with TestClient(main_mod.create_app()) as client:
        resp = client.get(
            "/v1/chat/stream",
            params={"user_id": "u1", "session_id": "s9", "message": "帮我查订单"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

    events = []
    for line in resp.text.splitlines():
        if line.startswith("event: "):
            events.append(line[len("event: "):])
    # 0.3.0：GET 流弃用标记——流首 deprecation 事件（老客户端未知事件类型，
    # 按 SSE 规范安全忽略），其后事件序列不变
    assert events[0] == "deprecation"
    assert events[1:3] == ["meta", "thought"]
    assert "tool_call" in events and "tool_result" in events
    assert events[-2] == "reply"
    assert events[-1] == "end"
    # reply 内容
    reply_line = [l for l in resp.text.splitlines() if l.startswith("data: ")][-2]
    assert "最终答案" in reply_line
    # 弃用响应头
    assert resp.headers.get("deprecation") == "true"
    assert "sunset" in {k.lower() for k in resp.headers.keys()}


def test_sse_post_stream_no_deprecation_event(monkeypatch):
    """0.3.0 新增继任端点 POST /v1/chat/stream：无 deprecation 事件/头。"""
    from fastapi.testclient import TestClient
    import app.server.main as main_mod

    async def fake_turn(agent, message, on_start=None):
        if on_start is not None:
            on_start()
        agent.event_callback("thought", {"text": "思考中"})
        from conftest import sample_response

        return sample_response(reply="最终答案")

    from test_server_api import _FakeComponents, _ScriptedAgent

    monkeypatch.setattr(main_mod.runtime, "run_agent_turn", fake_turn)
    monkeypatch.setattr(main_mod, "build_pod_components", lambda: _FakeComponents())
    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda user_id, session_id="", components=None, **_kw: _ScriptedAgent(user_id, session_id),
    )
    with TestClient(main_mod.create_app()) as client:
        resp = client.post(
            "/v1/chat/stream",
            json={"user_id": "u1", "session_id": "s9", "message": "帮我查订单"},
        )
        assert resp.status_code == 200

    events = [
        line[len("event: "):]
        for line in resp.text.splitlines() if line.startswith("event: ")
    ]
    assert events[0] == "meta"
    assert "deprecation" not in events
    assert events[-1] == "end"
    assert resp.headers.get("deprecation") is None


def test_metrics_endpoint_serves_prom_text(monkeypatch):
    from fastapi.testclient import TestClient
    import app.server.main as main_mod
    from test_server_api import _FakeComponents, _ScriptedAgent
    from test_server_api import _RECORDED_CALLS

    def factory(user_id, session_id="", components=None, **_kw):
        return _ScriptedAgent(user_id, session_id)

    monkeypatch.setattr(main_mod, "build_agent", factory)
    monkeypatch.setattr(main_mod, "build_pod_components", lambda: _FakeComponents())
    with TestClient(main_mod.create_app()) as client:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "chat_latency_seconds" in resp.text
        assert "tool_calls_total" in resp.text
