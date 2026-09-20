"""推理模型全量适配（T0–T9）单测：画像改写、双通道、去强制、SSE、计费、评测指纹。

对应计划各任务的验收标准（T0–T9；实施内容与遗留项见 docs/推理模型适配-实施记录.md）：
- T1：4 种 temperature_mode × 3 种 return_policy × 2 种 token 参数名的组合改写；
  画像缺失时行为与现状逐字节一致（回归保护）；降级链按备用模型画像重写；
- T2：推理画像下请求体无 temperature、输出上限 ≥8192；新水位下最近消息不被裁；
- T3：DeepSeek 画像窗口无 reasoning / 审计有；Claude 画像窗口含带 signature 的块；
- T4：支持/不支持强制 tool_choice 两分支；软强制下仍回纯文本 → 转人工；
- T6：开关关 → 无 reasoning 事件；开 → 有；两种情况审计都留存；命中 guardrails 不透出；
- T7：reasoning 方向独立计数、completion 扣减；参数改名后预留不吃掉输出上限；
- T9：沙箱温度按画像分支、manifest 记画像、judge 守卫拒绝非 free 画像。

全程无网络（FakeChatClient 剧本 + 纯函数）。
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from app.agent.chat import EcomAgent
from app.config.settings import settings
from tests.unit.conftest import FakeChatClient

DEEPSEEK = "deepseek-reasoner"
CLAUDE = "claude-sonnet-4-thinking"


# ============================================================
# 测试替身
# ============================================================
def _tool_call(call_id: str, name: str, arguments: dict):
    import json

    return NS(
        id=call_id, type="function",
        function=NS(name=name, arguments=json.dumps(arguments, ensure_ascii=False)),
    )


def _response(*, content=None, tool_calls=None, reasoning_content=None, usage=None):
    """构造带 reasoning 的 chat 响应（SDK 形态对象；reasoning 按需挂字段）。"""
    message = NS(content=content, tool_calls=tool_calls, role="assistant")
    if reasoning_content is not None:
        message.reasoning_content = reasoning_content
    return NS(
        choices=[NS(message=message, finish_reason="stop")],
        usage=usage or NS(
            prompt_tokens=100, completion_tokens=50, total_tokens=150,
        ),
    )


def _text_response(text, reasoning=None):
    return _response(content=text, reasoning_content=reasoning)


def _final_response_call(reply="已为您处理。", intent="other", call_id="call_final"):
    return _response(tool_calls=[_tool_call(call_id, "final_response", {
        "intent": intent, "reply": reply,
        "requires_human": False, "follow_up_question": None,
    })])


def _agent(tmp_path, client, *, model=None, event_callback=None):
    agent = EcomAgent(
        session_path=str(tmp_path / "adaptation.json"), client=client,
        memory_enabled=False, use_mcp=False, event_callback=event_callback,
    )
    if model is not None:
        agent.model = model
    agent.context_builder._window = 8192  # 本文件不测历史压缩
    return agent


def _window_messages_seen_by_model(client, call_index: int) -> list[dict]:
    """第 call_index 次 LLM 调用实际发出去的 messages（= 窗口通道真身）。"""
    return client.calls[call_index][1]["messages"]


def _audit_assistant_messages(agent) -> list[dict]:
    """审计切片里的 assistant 消息（append_log = full_turn_messages 汇聚点）。"""
    return [
        m for m in agent._append_log
        if m.get("role") == "assistant"
    ]


# ============================================================
# T1 画像层：改写矩阵
# ============================================================
@pytest.mark.parametrize("temperature_mode", ["free", "fixed_1", "ignored", "forbidden"])
@pytest.mark.parametrize("policy", ["forbidden", "internal", "required_signed"])
@pytest.mark.parametrize("token_param", ["max_tokens", "max_completion_tokens"])
def test_apply_profile_matrix(temperature_mode, policy, token_param):
    """4 × 3 × 2 组合：temperature 与输出参数按画像改写，其余字段不动。"""
    from app.llm.model_profile import apply_profile

    model = f"probe-{temperature_mode}"
    profile = {
        "temperature_mode": temperature_mode,
        "reasoning_return_policy": policy,
        "max_tokens_param": token_param,
        "min_max_tokens": 8192,
    }
    settings.model_profile_overrides = f'{{"{model}": {_json(profile)}}}'
    kwargs = {"messages": [{"role": "user", "content": "hi"}],
              "temperature": 0.7, "max_tokens": 2048}
    apply_profile(kwargs, model)

    if temperature_mode == "fixed_1":
        assert kwargs["temperature"] == 1.0
    elif temperature_mode == "free":
        assert kwargs["temperature"] == 0.7      # free：原样透传调用方偏好值
    else:
        assert "temperature" not in kwargs
    assert kwargs[token_param] == 8192
    if token_param != "max_tokens":
        assert "max_tokens" not in kwargs
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]  # 其余不动


def _json(payload: dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)


def test_no_profile_is_byte_identical(reset_settings):
    """回归保护：未命中画像 → 请求参数逐字节不动（旧模型部署零变化）。"""
    from app.llm.model_profile import apply_profile, resolve_profile

    assert resolve_profile("gpt-4o-mini") is None
    assert resolve_profile("DeepSeek-V4-Flash-0731") is None
    kwargs = {"temperature": 0.7, "max_tokens": 2048, "tools": [{"x": 1}]}
    snapshot = dict(kwargs)
    assert apply_profile(kwargs, "gpt-4o-mini") == snapshot
    assert kwargs == snapshot


def test_overrides_beat_registry_and_bad_values_fall_back(reset_settings):
    """部署覆写优先于内置 registry；非法字段值落回默认并告警（不 500）。"""
    from app.llm.model_profile import resolve_profile

    settings.model_profile_overrides = _json({
        "deepseek-reasoner": {
            "temperature_mode": "not-a-mode",       # 非法 → free
            "reasoning_return_policy": "official",  # 非法 → forbidden
            "supports_forced_tool_choice": "false",  # 字符串布尔 → False
            "max_tokens_param": "max_completion_tokens",
            "min_max_tokens": 4096,
        },
    })
    profile = resolve_profile("deepseek-reasoner")
    assert profile is not None
    assert profile.temperature_mode == "free"
    assert profile.reasoning_return_policy == "forbidden"
    assert profile.supports_forced_tool_choice is False
    assert profile.max_tokens_param == "max_completion_tokens"
    assert profile.min_max_tokens == 4096

    settings.model_profile_overrides = "{不是 JSON"
    assert resolve_profile("deepseek-reasoner") is not None  # 回退内置 registry


def test_wildcard_override_matches_any_model(reset_settings):
    from app.llm.model_profile import resolve_profile

    settings.model_profile_overrides = _json({"*": {"temperature_mode": "ignored"}})
    assert resolve_profile("anything-at-all").temperature_mode == "ignored"


# ============================================================
# T1 + T2 端到端：请求体形态
# ============================================================
def test_reasoning_profile_request_body_has_no_temperature(
    tmp_path, reset_settings, monkeypatch,
):
    """T2 验收：推理画像下请求体无 temperature，输出上限抬到 8192。"""
    from app.llm.client import install_resilience

    monkeypatch.setattr(settings, "llm_max_tokens", 8192)
    client = (
        FakeChatClient()
        .enqueue_callable(lambda kind, kwargs: _text_response("好的", reasoning="思考中"))
        .enqueue_callable(lambda kind, kwargs: _final_response_call())
    )
    agent = _agent(tmp_path, client, model=DEEPSEEK)
    install_resilience(client, DEEPSEEK)
    agent.chat("你好")

    for _, kwargs in client.calls:
        assert "temperature" not in kwargs          # ignored 画像：不再发送
        assert kwargs["max_tokens"] >= 8192         # min_max_tokens 兜底


def test_o_series_profile_renames_max_tokens(tmp_path, reset_settings, monkeypatch):
    """o 系列画像：max_tokens → max_completion_tokens。"""
    from app.llm.client import install_resilience

    client = (
        FakeChatClient()
        .enqueue_callable(lambda kind, kwargs: _text_response("好的"))
        .enqueue_callable(lambda kind, kwargs: _final_response_call())
    )
    agent = _agent(tmp_path, client, model="o4-mini")
    install_resilience(client, "o4-mini")
    agent.chat("你好")

    for _, kwargs in client.calls:
        assert "max_tokens" not in kwargs
        assert kwargs["max_completion_tokens"] >= 8192


def test_fallback_chain_rewrites_for_fallback_profile(
    tmp_path, reset_settings, monkeypatch,
):
    """降级链按**备用模型**画像重写：主模型改名后的参数不能带给备用模型。

    主模型 o 系列（max_completion_tokens）→ 持续限流 → 降级到 gpt-4o，
    gpt-4o 必须收到 max_tokens（而不是 o 系列的参数名）。
    """
    from openai import RateLimitError

    from app.llm.client import install_resilience

    rate_limited = NS(
        request=NS(method="POST", url="http://x"), body=None, code=None,
        response=NS(status_code=429, headers={}, request=NS(method="POST", url="http://x")),
    )

    def _flaky(kind, kwargs):
        if kwargs["model"] == "o4-mini":
            raise RateLimitError("boom", response=rate_limited.response, body=None)
        return _text_response("降级成功")

    # 主模型与降级各消费一个剧本条目（降级是第二次真实调用）
    client = FakeChatClient().enqueue_callable(_flaky).enqueue_callable(_flaky)
    install_resilience(client, "o4-mini", fallback_model="gpt-4o",
                       max_retries=0, timeout_seconds=1)
    client.chat.completions.create(
        model="o4-mini", messages=[{"role": "user", "content": "hi"}],
        temperature=0.7, max_tokens=2048,
    )

    fallback_calls = [kw for _, kw in client.calls if kw["model"] == "gpt-4o"]
    assert fallback_calls, "降级链未触发"
    assert fallback_calls[0]["max_tokens"] == 2048   # 备用模型用原参数名与原值
    assert fallback_calls[0]["temperature"] == 0.7   # gpt-4o 是 free 画像
    assert "max_completion_tokens" not in fallback_calls[0]


# ============================================================
# T2 水位一致性
# ============================================================
def test_output_reserve_covers_output_cap():
    """T2 不变式：输出预留 ≥ llm_max_tokens（否则水位自相矛盾）。"""
    from app.agent.token_budget import budget_shares

    shares = budget_shares(settings.context_window_tokens)
    assert shares["output_reserve"] >= settings.llm_max_tokens


def test_recent_messages_survive_new_watermark():
    """T2 验收：新水位下最近 3 条消息不被裁（trim 单测同口径）。"""
    from app.agent.token_budget import budget_shares, trim_messages_to_budget

    shares = budget_shares(settings.context_window_tokens)
    messages = [{"role": "user", "content": f"第 {i} 轮问题" + "字" * 20}
                for i in range(12)]
    kept = trim_messages_to_budget(messages, shares["dialog"])
    assert kept[-3:] == messages[-3:]


# ============================================================
# T3 reasoning 双通道
# ============================================================
def test_deepseek_window_strips_reasoning_audit_keeps_it(
    tmp_path, reset_settings,
):
    """DeepSeek 画像（forbidden）：窗口无 reasoning，审计有。"""
    reasoning = "先判断时效：20 天未拆封，走特批流程。"
    client = (
        FakeChatClient()
        .enqueue_callable(
            lambda kind, kwargs: _text_response("让我核实一下。", reasoning=reasoning),
        )
        .enqueue_callable(lambda kind, kwargs: _final_response_call("请稍等，正在核实。"))
    )
    agent = _agent(tmp_path, client, model=DEEPSEEK)
    agent.chat("我的订单能退吗")

    # 第 2 次调用的 messages = 窗口真身：含上一步的 assistant 消息，但无 reasoning
    window = _window_messages_seen_by_model(client, 1)
    assistant_msgs = [m for m in window if m.get("role") == "assistant"]
    assert assistant_msgs, "窗口里应能观察到上一步的 assistant 消息"
    assert all("reasoning" not in m for m in assistant_msgs)
    assert all("reasoning_content" not in m for m in assistant_msgs)

    # 审计：reasoning 附加字段始终留存（additive）
    audits = [m for m in _audit_assistant_messages(agent) if "reasoning" in m]
    assert audits, "审计切片必须保留 reasoning"
    payload = audits[0]["reasoning"]
    assert payload["field"] == "reasoning_content"
    assert payload["text"] == reasoning
    assert payload["chars"] == len(reasoning)


def test_claude_window_keeps_signed_thinking_block(tmp_path, reset_settings):
    """Claude thinking 画像（required_signed）：窗口原样回传带 signature 的块。"""
    blocks = [
        {"type": "thinking", "thinking": "先确认时效边界。", "signature": "sig-abc123"},
        {"type": "text", "text": "让我核实一下。"},
    ]
    client = (
        FakeChatClient()
        .enqueue_callable(lambda kind, kwargs: _response(content=blocks))
        .enqueue_callable(lambda kind, kwargs: _final_response_call("正在核实。"))
    )
    agent = _agent(tmp_path, client, model=CLAUDE)
    agent.chat("我的订单能退吗")

    window = _window_messages_seen_by_model(client, 1)
    assistant_msgs = [m for m in window if m.get("role") == "assistant"]
    sent = assistant_msgs[0]["content"]
    assert isinstance(sent, list), "required_signed 必须原样回传块列表"
    thinking = [b for b in sent if b.get("type") == "thinking"]
    assert thinking and thinking[0]["signature"] == "sig-abc123"

    # 审计：reasoning 文本只取推理块本身（不重复正文）
    audit = [m for m in _audit_assistant_messages(agent) if "reasoning" in m][0]
    assert audit["reasoning"]["field"] == "thinking_blocks"
    assert audit["reasoning"]["text"] == "先确认时效边界。"
    assert audit["reasoning"]["blocks"][0]["signature"] == "sig-abc123"


def test_unprofiled_model_still_audits_reasoning(tmp_path, reset_settings):
    """未登记画像的模型也捕获 reasoning 进审计（捕获与放行解耦）。"""
    client = (
        FakeChatClient()
        .enqueue_callable(lambda kind, kwargs: _text_response("好", reasoning="想一想"))
        .enqueue_callable(lambda kind, kwargs: _final_response_call())
    )
    agent = _agent(tmp_path, client, model="some-unlisted-model")
    agent.chat("你好")

    window = _window_messages_seen_by_model(client, 1)
    assert all("reasoning" not in m for m in window if m.get("role") == "assistant")
    assert [m for m in _audit_assistant_messages(agent) if "reasoning" in m]


# ============================================================
# T4 强制收尾去强制化
# ============================================================
def _soft_profile(model: str = "soft-forced-model") -> str:
    return _json({model: {
        "temperature_mode": "free",
        "supports_forced_tool_choice": False,
        "reasoning_field": "none",
    }})


def test_forced_finalize_hard_mode_sends_tool_choice(tmp_path, reset_settings,
                                                     monkeypatch):
    """支持强制的画像：仍走显式 tool_choice（现状行为）。"""
    metrics = _metrics()
    before = metrics.FORCED_FINALIZE_ATTEMPTS.labels(mode="hard")._value.get()
    monkeypatch.setattr(settings, "max_react_steps", 1)
    client = (
        FakeChatClient()
        .enqueue_callable(lambda kind, kwargs: _text_response("先说说看"))
        .enqueue_callable(lambda kind, kwargs: _final_response_call("已处理。"))
    )
    agent = _agent(tmp_path, client)
    result = agent.chat("你好")

    assert result.reply == "已处理。"
    forced_kwargs = client.calls[1][1]
    assert forced_kwargs["tool_choice"]["function"]["name"] == "final_response"
    assert metrics.FORCED_FINALIZE_ATTEMPTS.labels(mode="hard")._value.get() == before + 1


def test_forced_finalize_soft_mode_drops_tool_choice(tmp_path, reset_settings,
                                                     monkeypatch):
    """不支持强制的画像：只挂终止工具 + system 软强制，模型仍回 tool_calls → 收尾成功。"""
    metrics = _metrics()
    before = metrics.FORCED_FINALIZE_ATTEMPTS.labels(mode="soft")._value.get()
    monkeypatch.setattr(settings, "max_react_steps", 1)
    settings.model_profile_overrides = _soft_profile("soft-forced-model")
    client = (
        FakeChatClient()
        .enqueue_callable(lambda kind, kwargs: _text_response("先说说看"))
        .enqueue_callable(lambda kind, kwargs: _final_response_call("已处理。"))
    )
    agent = _agent(tmp_path, client, model="soft-forced-model")
    result = agent.chat("你好")

    assert result.reply == "已处理。"
    forced_kwargs = client.calls[1][1]
    assert "tool_choice" not in forced_kwargs
    assert [t["function"]["name"] for t in forced_kwargs["tools"]] == ["final_response"]
    assert any(
        "必须调用 final_response" in str(m.get("content"))
        for m in forced_kwargs["messages"] if m.get("role") == "system"
    )
    assert metrics.FORCED_FINALIZE_ATTEMPTS.labels(mode="soft")._value.get() == before + 1


def test_forced_finalize_soft_mode_plain_text_still_goes_to_human(
    tmp_path, reset_settings, monkeypatch,
):
    """软强制失败（仍纯文本）→ ForcedFinalizeFailed → 确定性转人工（兜底语义不变）。"""
    metrics = _metrics()
    before = metrics.FORCED_FINALIZE_FALLBACK.labels(mode="soft")._value.get()
    monkeypatch.setattr(settings, "max_react_steps", 1)
    settings.model_profile_overrides = _soft_profile("soft-forced-model")
    client = (
        FakeChatClient()
        .enqueue_callable(lambda kind, kwargs: _text_response("先说说看"))
        .enqueue_callable(lambda kind, kwargs: _text_response("我就是不给结构化终答"))
    )
    agent = _agent(tmp_path, client, model="soft-forced-model")
    result = agent.chat("你好")

    assert result.requires_human is True
    assert metrics.FORCED_FINALIZE_FALLBACK.labels(mode="soft")._value.get() == before + 1


def _metrics():
    from app.observability import metrics

    return metrics


# ============================================================
# T6 SSE 透出
# ============================================================
def _sse_agent(tmp_path, client, model, events, **kwargs):
    def callback(event_type, data):
        events.append((event_type, data))

    return _agent(tmp_path, client, model=model, event_callback=callback, **kwargs)


def test_sse_reasoning_off_by_default(tmp_path, reset_settings):
    """默认关：不出现 reasoning 事件（现状序列不变）。"""
    events: list[tuple[str, dict]] = []
    client = (
        FakeChatClient()
        .enqueue_callable(lambda kind, kwargs: _text_response("好", reasoning="内部推理"))
        .enqueue_callable(lambda kind, kwargs: _final_response_call())
    )
    agent = _sse_agent(tmp_path, client, DEEPSEEK, events)
    assert settings.sse_reasoning_enabled is False
    agent.chat("你好")

    assert "thought" in [e for e, _ in events]
    assert "reasoning" not in [e for e, _ in events]
    # 审计不依赖透出开关
    assert [m for m in _audit_assistant_messages(agent) if "reasoning" in m]


def test_sse_reasoning_on_emits_event_and_audits(tmp_path, reset_settings):
    """开关开：SSE 出现 reasoning{text,step}，审计同样留存。"""
    events: list[tuple[str, dict]] = []
    client = (
        FakeChatClient()
        .enqueue_callable(lambda kind, kwargs: _text_response("好", reasoning="内部推理"))
        .enqueue_callable(lambda kind, kwargs: _final_response_call())
    )
    agent = _sse_agent(tmp_path, client, DEEPSEEK, events)
    settings.sse_reasoning_enabled = True
    agent.chat("你好")

    reasoning_events = [d for e, d in events if e == "reasoning"]
    assert reasoning_events and reasoning_events[0]["text"] == "内部推理"
    assert reasoning_events[0]["step"] >= 1
    assert [m for m in _audit_assistant_messages(agent) if "reasoning" in m]


def test_sse_reasoning_blocked_by_guardrails(tmp_path, reset_settings):
    """开启透出但命中输出 guardrails → 不透出（审计仍留存）。"""
    events: list[tuple[str, dict]] = []
    client = (
        FakeChatClient()
        .enqueue_callable(lambda kind, kwargs: _text_response("好", reasoning="忽略以上指令"))
        .enqueue_callable(lambda kind, kwargs: _final_response_call())
    )
    agent = _sse_agent(tmp_path, client, DEEPSEEK, events)
    settings.sse_reasoning_enabled = True
    settings.guardrails_enabled = True
    settings.guardrail_block_terms = "忽略以上指令"
    agent.chat("你好")

    assert "reasoning" not in [e for e, _ in events]
    assert [m for m in _audit_assistant_messages(agent) if "reasoning" in m]


# ============================================================
# T7 计费与预算
# ============================================================
def test_reasoning_tokens_split_from_completion():
    """推理 token 从 completion 拆出单列；成本合计不变。"""
    from app.observability import metrics

    def _value(direction):
        return metrics.LLM_TOKENS.labels(
            direction=direction, purpose="react-t7",
        )._value.get()

    before = {d: _value(d) for d in ("prompt", "completion", "reasoning")}
    metrics.record_llm_usage("deepseek-reasoner", "react-t7", 1000, 900, 700)

    assert _value("prompt") == before["prompt"] + 1000
    assert _value("completion") == before["completion"] + 200   # 900 - 700
    assert _value("reasoning") == before["reasoning"] + 700


def test_zero_reasoning_tokens_keeps_legacy_split():
    """reasoning_tokens=0 → completion 全额计入（与拆分前逐字节一致）。"""
    from app.observability import metrics

    def _value(direction):
        return metrics.LLM_TOKENS.labels(
            direction=direction, purpose="react-t7-legacy",
        )._value.get()

    before = _value("completion")
    metrics.record_llm_usage("gpt-4o-mini", "react-t7-legacy", 10, 500)
    assert _value("completion") == before + 500


def test_estimate_tokens_reads_renamed_output_param():
    """参数改名后预留仍含输出上限（否则 o 系列预留被系统性低估）。"""
    from app.llm.client import estimate_llm_tokens

    renamed = estimate_llm_tokens({
        "messages": [{"role": "user", "content": "x" * 400}],
        "max_completion_tokens": 8192,
    })
    legacy = estimate_llm_tokens({
        "messages": [{"role": "user", "content": "x" * 400}],
        "max_tokens": 8192,
    })
    assert renamed == legacy


# ============================================================
# T9 评测基线
# ============================================================
def test_sandbox_temperature_branches_by_profile(reset_settings):
    """沙箱温度按画像分支：非 free 画像不传（None），free/未命中保持 0.0。"""
    from app.evaluation.sandbox import SANDBOX_TEMPERATURE, sandbox_temperature

    assert sandbox_temperature("gpt-4o-mini") == SANDBOX_TEMPERATURE
    assert sandbox_temperature("some-unlisted") == SANDBOX_TEMPERATURE
    assert sandbox_temperature(DEEPSEEK) is None
    assert sandbox_temperature("o4-mini") is None
    assert sandbox_temperature(CLAUDE) is None


def test_manifest_records_model_profile(tmp_path, reset_settings, monkeypatch):
    """manifest 指纹含画像全字段与原因子（新旧报告不可混淆）。"""
    from app.evaluation.manifest import build_manifest

    ds = tmp_path / "cases.json"
    ds.write_text('{"cases": []}', encoding="utf-8")
    monkeypatch.setattr(settings, "model_name", DEEPSEEK)

    m = build_manifest(str(ds), num_cases=1, model=DEEPSEEK,
                       judge_model="judge-model-free")
    assert m["model"]["profile"]["matched"] is True
    profile = m["model"]["profile"]["profile"]
    assert profile["temperature_mode"] == "ignored"
    assert profile["reasoning_field"] == "reasoning_content"
    assert profile["reasoning_return_policy"] == "forbidden"
    assert profile["min_max_tokens"] == 8192
    assert m["model"]["temperature"] is None  # 沙箱不传 temperature
    assert m["model"]["judge_profile"]["matched"] is False
    assert m["config"]["model_profile"]["matched"] is True


def test_manifest_free_profile_keeps_sandbox_temperature(
    tmp_path, reset_settings, monkeypatch,
):
    """未命中画像（现状部署）：指纹仍记 0.0（回归保护，旧报告可续跑比较）。"""
    from app.evaluation.manifest import build_manifest
    from app.evaluation.sandbox import SANDBOX_TEMPERATURE

    ds = tmp_path / "cases.json"
    ds.write_text('{"cases": []}', encoding="utf-8")
    monkeypatch.setattr(settings, "model_name", "m-under-test")

    m = build_manifest(str(ds), num_cases=1, model="m-under-test",
                       judge_model="m-judge")
    assert m["model"]["temperature"] == SANDBOX_TEMPERATURE
    assert m["model"]["profile"]["matched"] is False
    # 未命中画像 = 全默认画像（free/无 reasoning），即现状语义
    profile = m["model"]["profile"]["profile"]
    assert profile["temperature_mode"] == "free"
    assert profile["reasoning_field"] == "none"
    assert profile["max_tokens_param"] == "max_tokens"


@pytest.mark.parametrize("judge,ok", [
    ("gpt-4o-mini", True),
    ("judge-unlisted", True),
    (DEEPSEEK, False),
    ("o4-mini", False),
    (CLAUDE, False),
])
def test_run_eval_judge_profile_guard(reset_settings, monkeypatch, judge, ok):
    """judge 守卫：非 free 画像直接拒绝（judge 硬设 temperature=0.0）。"""
    from app.scripts.run_eval import (
        JudgeModelConfigError,
        JudgeProfileError,
        resolve_judge_model,
    )

    monkeypatch.setattr(settings, "model_name", "under-test-model")
    if ok:
        assert resolve_judge_model(judge) == judge
    else:
        with pytest.raises(JudgeProfileError):
            resolve_judge_model(judge)
    # 与被测模型同名仍走原有拒绝路径（守卫不掩盖既有约束）
    with pytest.raises(JudgeModelConfigError):
        resolve_judge_model("under-test-model")


def test_online_quality_judge_guard(reset_settings):
    """lite judge 同一守卫口径。"""
    from app.scripts.online_quality import judge_guard_reason

    assert judge_guard_reason("gpt-4o-mini") == ""
    assert judge_guard_reason(DEEPSEEK) != ""
    judged = __import__(
        "app.scripts.online_quality", fromlist=["judge_turn"],
    ).judge_turn("问题", "回答", "prompt", client=None, model=DEEPSEEK)
    assert judged["resolved"] is None and "temperature_mode" in judged["reason"]


# ============================================================
# T0 探针
# ============================================================
def test_probe_dry_run_passes_gate(tmp_path, capsys):
    """dry-run：脚本接线可跑通、门禁判定、画像落盘。"""
    from app.scripts.probe_model_capability import main

    out = tmp_path / "profile.json"
    code = main(["--dry-run", "--model", DEEPSEEK, "--out", str(out)])
    assert code == 0
    import json

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["gate"]["passed"] is True
    assert report["profile"]["supports_tools"] is True
    assert report["profile"]["temperature_mode"] in (
        "free", "ignored", "forbidden", "fixed_1",
    )
    assert report["latency"]["samples"] > 0
    assert report["reasoning_tokens"]["typical"] > 0


def test_probe_gate_failure_terminates_with_exit_code(tmp_path, monkeypatch):
    """门禁失败（模型不返回 tool_calls）→ 退出码 2（计划终止信号）。"""
    from app.scripts import probe_model_capability as probe

    def _no_tools(**kwargs):
        return NS(
            choices=[NS(message=NS(content="我不会调用工具", tool_calls=None),
                         finish_reason="stop")],
            usage=NS(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    client = NS(chat=NS(completions=NS(create=_no_tools)))
    monkeypatch.setattr(probe, "_scripted_client", lambda model: client)

    out = tmp_path / "gate.json"
    code = probe.main(["--dry-run", "--model", "no-tools-model", "--out", str(out)])
    assert code == probe.GATE_FAILED_EXIT
    import json

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["gate"]["passed"] is False
