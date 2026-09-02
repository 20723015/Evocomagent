"""recorder：轮次切片解析、原子落盘、两类 Agent 挂载点、压缩不丢、失败不阻断。"""

from __future__ import annotations

import json

import pytest

from conftest import sample_response, search_tool_messages
from app.evolution.recorder import TurnRecorder, parse_turn_slice


# ============================================================
# parse_turn_slice
# ============================================================
def test_parse_extracts_search_results():
    messages = search_tool_messages("退货可以吗", [
        {"doc": "退货政策", "section": "七天无理由", "score": 0.93, "text": "支持七天无理由"},
        {"doc": "退货政策", "section": "运费", "score": 0.81, "text": "运费顾客承担"},
    ])
    sources = parse_turn_slice(messages, 0)
    assert len(sources) == 2
    assert sources[0].doc == "退货政策"
    assert sources[0].score == 0.93
    assert sources[0].section == "七天无理由"


def test_parse_ignores_failed_tool_payloads():
    base = search_tool_messages("问", [{"doc": "A", "score": 0.9, "text": "x" * 40}])
    # 把 tool 结果改成 success=False 与其它工具
    messages = [
        {"role": "user", "content": "问"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "search_knowledge", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "check_order", "arguments": "{}"}},
            {"id": "c3", "type": "function", "function": {"name": "search_knowledge", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"success": False, "results": []})},
        {"role": "tool", "tool_call_id": "c2", "content": json.dumps({"success": True, "results": [{"doc": "订单", "score": 0.9, "text": "订单信息"}]})},
        {"role": "tool", "tool_call_id": "c3", "content": json.dumps({"success": True, "results": [{"doc": "退货政策", "score": 0.9, "text": "政策内容"}]})},
    ]
    sources = parse_turn_slice(messages, 0)
    assert len(sources) == 1
    assert sources[0].doc == "退货政策"


def test_parse_works_without_start_idx():
    """assistant(tool_calls) 在 tool 消息之前 → 两遍扫描仍正确。"""
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "search_knowledge", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps(
            {"success": True, "results": [{"doc": "配送说明", "score": 0.8, "text": "偏远地区"}]})},
    ]
    sources = parse_turn_slice(messages, 0)
    assert len(sources) == 1


def test_parse_sanitizes_source_text():
    messages = search_tool_messages("问", [
        {"doc": "A", "score": 0.9, "text": "联系 13812345678 询问"},
    ])
    sources = parse_turn_slice(messages, 0)
    assert "【手机号】" in sources[0].text
    assert "13812345678" not in sources[0].text


# ============================================================
# record：原子落盘
# ============================================================
def test_record_writes_turn_file(tmp_state_dir, frozen_clock):
    recorder = TurnRecorder(tmp_state_dir["turns"], clock=frozen_clock)
    messages = search_tool_messages("退货可以吗", [
        {"doc": "退货政策", "score": 0.9, "text": "支持七天无理由"},
    ])
    turn_id = recorder.record(
        session_id="sess-1",
        mode="single",
        question="退货可以吗 13812345678",
        structured_reply=sample_response(
            reply="可以退货，运费由您承担。联系电话 13812345678",
            intent="return_request",
            confidence=0.85,
            follow_up_question="请问需要上门取件吗？",
        ),
        turn_slice=messages,
    )
    assert turn_id and len(turn_id) == 32
    day_dir = tmp_state_dir["turns"] / "20260828"
    files = list(day_dir.glob("*.json"))
    assert len(files) == 1
    assert files[0].stem == turn_id
    assert not list(day_dir.glob("*.tmp"))

    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["session_id"] == "sess-1"
    assert data["mode"] == "single"
    assert data["status"] == "captured"
    assert data["intent"] == "return_request"
    assert data["confidence"] == 0.85
    assert data["follow_up"] == "请问需要上门取件吗？"
    assert data["ts"].startswith("2026-08-28")
    # PII 已脱敏
    assert "【手机号】" in data["question"]
    assert "13812345678" not in data["question"]
    assert "【手机号】" in data["reply"]
    assert len(data["sources"]) == 1


def test_record_failure_returns_none_and_does_not_raise(tmp_state_dir):
    recorder = TurnRecorder(tmp_state_dir["turns"])
    # structured_reply 缺字段 → 内部异常被捕获，返回 None 且不抛
    result = recorder.record(
        session_id="s", mode="single", question="q", structured_reply=object(),
        turn_slice=[],
    )
    assert result is None


# ============================================================
# 两类 Agent 挂载点
# ============================================================
def test_ecom_agent_mount_records_turn(monkeypatch, tmp_state_dir, tmp_path,
                                       reset_settings):
    from app.config.settings import settings as s
    from app.agent.chat import EcomAgent

    s.evolve_capture_enabled = True
    s.evolve_turns_dir = str(tmp_state_dir["turns"])
    session_path = str(tmp_path / "session.json")

    class OfflineAgent(EcomAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._react_loop = lambda state, budget: "这是客服回复内容。"
            self._extract_structured_response = lambda text: sample_response(
                reply="这是客服回复内容。"
            )

    agent = OfflineAgent(session_path=session_path)
    agent.memory_manager.update_short_term = lambda *a, **k: None
    agent.history_threshold = 1000  # 不触发压缩
    resp = agent.chat("七天无理由退货可以吗？")

    assert agent.session_id
    files = list(tmp_state_dir["turns"].rglob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["mode"] == "single"
    assert data["session_id"] == agent.session_id

    # 会话文件 v2：session_id 贯通
    loaded = __import__("app.agent.storage", fromlist=["load_session"]).load_session(session_path)
    assert loaded["session_id"] == agent.session_id


def test_multi_agent_mount_records_turn(monkeypatch, tmp_state_dir, tmp_path,
                                        reset_settings):
    from app.config.settings import settings as s
    from app.multi_agent.orchestrator import MultiAgentOrchestrator

    s.evolve_capture_enabled = True
    s.evolve_turns_dir = str(tmp_state_dir["turns"])
    session_path = str(tmp_path / "session.json")

    agent = MultiAgentOrchestrator(session_path=session_path)
    agent.router.route = lambda user_input, messages: list(agent.agents.keys())[0]
    for sub in agent.agents.values():
        sub.handle = lambda messages, ctx=None, max_steps=5, executor=None, state=None, budget=None: (
            "这是客服回复内容。", [], 1,
        )
    agent._extract_structured_response = lambda text: sample_response(
        reply="这是客服回复内容。"
    )
    agent.memory_manager.update_short_term = lambda *a, **k: None
    agent.history_threshold = 1000

    agent.chat("退货运费谁出？")
    files = list(tmp_state_dir["turns"].rglob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["mode"] == "multi"
    assert data["session_id"] == agent.session_id


def test_compress_history_does_not_lose_candidates(monkeypatch, tmp_state_dir,
                                                   tmp_path, reset_settings):
    """压缩发生在 record 之后：turn 已落盘，历史压缩不丢候选。"""
    from app.config.settings import settings as s
    from app.agent.chat import EcomAgent

    s.evolve_capture_enabled = True
    s.evolve_turns_dir = str(tmp_state_dir["turns"])
    # chat.py 里是 `from app.agent.summarizer import summarize`，要 patch 它的引用
    monkeypatch.setattr("app.agent.chat.summarize",
                        lambda **kw: "历史摘要（已压缩）")

    class OfflineAgent(EcomAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._react_loop = lambda state, budget: "这是客服回复内容。"
            self._extract_structured_response = lambda text: sample_response(
                reply="这是客服回复内容。"
            )

    agent = OfflineAgent(session_path=str(tmp_path / "session.json"))
    agent.memory_manager.update_short_term = lambda *a, **k: None
    agent.history_threshold = 3  # 两轮后（4 条消息）触发压缩
    for _ in range(2):
        agent.chat("问一句")
    assert agent.summary == "历史摘要（已压缩）"
    assert len(list(tmp_state_dir["turns"].rglob("*.json"))) == 2


def test_recorder_failure_does_not_break_chat(monkeypatch, tmp_state_dir,
                                              tmp_path, reset_settings):
    from app.config.settings import settings as s
    from app.agent.chat import EcomAgent

    s.evolve_capture_enabled = True
    # turns 目录不可写：父路径是文件 → 落盘失败但主流程照常
    blocker = tmp_path / "is_a_file"
    blocker.write_text("x", encoding="utf-8")
    s.evolve_turns_dir = str(blocker / "turns")

    class OfflineAgent(EcomAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._react_loop = lambda state, budget: "这是客服回复内容。"
            self._extract_structured_response = lambda text: sample_response(
                reply="这是客服回复内容。"
            )

    agent = OfflineAgent(session_path=str(tmp_path / "session.json"))
    agent.memory_manager.update_short_term = lambda *a, **k: None
    agent.history_threshold = 1000
    resp = agent.chat("问一句")
    assert resp.reply == "这是客服回复内容。"