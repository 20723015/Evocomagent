"""sandbox 配置恢复 + with-eval 两条端到端（通过 / 阻断）。"""

from __future__ import annotations

import json

import pytest

from app.config.settings import settings
from app.evaluation.dataset import EvalCase
from app.evaluation.sandbox import Sandbox


# ============================================================
# 配置恢复（第10期修复）
# ============================================================
def _offline_chat(monkeypatch):
    """把被测 Agent 的 chat 替换为立即抛本地异常：沙箱 finally 依旧会走配置恢复，
    且不会向 api.openai.com 发出任何请求（CI「单测全程无网络」约束）。"""
    monkeypatch.setattr(
        "app.agent.chat.EcomAgent.chat",
        lambda self, user_input: (_ for _ in ()).throw(RuntimeError("offline")),
    )


def test_sandbox_uses_explicit_overrides_and_leaves_settings_untouched(
        tmp_path, reset_settings, monkeypatch):
    """阶段一 1.4：沙箱以显式 override 参数隔离被测 Agent，不再 monkey-patch 全局 settings。"""
    _offline_chat(monkeypatch)
    settings.evolve_capture_enabled = False  # 避免把 turn 写进仓库
    settings.memory_enabled = True
    settings.mcp_enabled = False
    settings.temperature = 0.7

    captured = {}

    class Probe(Sandbox):
        def _build_agent(self, session_path, case=None):
            agent = super()._build_agent(session_path, case)
            captured["agent"] = agent
            return agent

    sandbox = Probe(tmp_root=str(tmp_path / "sbx"))
    trace = sandbox.run(EvalCase(id="c1", description="离线用例", turns=["你好"]))

    assert trace.case_id == "c1"
    assert trace.error  # 离线异常是预期（chat 被替换为即抛）

    # Agent 实例拿到的是显式 override：记忆/路由关闭、温度归零
    agent = captured["agent"]
    assert agent.memory_manager.memory_enabled is False
    assert agent.temperature == 0.0
    assert agent.tool_manager._tool_source.get("query_order") == "local"
    # 全局 settings 全程未被改动
    assert settings.memory_enabled is True
    assert settings.mcp_enabled is False
    assert settings.temperature == 0.7
    assert getattr(sandbox, "_settings_snapshot", None) is None


def test_sandbox_freezes_temperature_during_run(tmp_path, reset_settings,
                                                monkeypatch):
    _offline_chat(monkeypatch)
    settings.evolve_capture_enabled = False
    settings.temperature = 0.7

    captured = {}

    class Probe(Sandbox):
        def _build_agent(self, session_path, case=None):
            agent = super()._build_agent(session_path, case)
            captured["temperature"] = agent.temperature  # 构造完成后的 Agent 温度
            return agent

    sandbox = Probe(tmp_root=str(tmp_path / "sbx"))
    sandbox.run(EvalCase(id="c2", description="d", turns=["你好"]))
    assert captured["temperature"] == 0.0  # 评测期间归零（行为变更）
    assert settings.temperature == 0.7  # 全局不受影响


def test_sandbox_collects_react_protocol_fields(tmp_path, reset_settings,
                                                monkeypatch):
    """步数余量感知（修改5）：每轮后从 _last_turn_ctx 采集协议级观测。

    步数/纠错按轮累计（求和）；预告与强制终答取「任一轮触发」。
    """
    from types import SimpleNamespace

    from app.schemas.response import IntentType

    settings.evolve_capture_enabled = False

    def fake_chat(self, user_input):
        self._last_turn_ctx = SimpleNamespace(
            react_steps=3, steps_margin_hint=True,
            forced_finalize=False, protocol_corrections=1,
        )
        return SimpleNamespace(
            reply="ok", intent=IntentType.GREETING, requires_human=False,
        )

    monkeypatch.setattr("app.agent.chat.EcomAgent.chat", fake_chat)
    sandbox = Sandbox(tmp_root=str(tmp_path / "sbx"))
    trace = sandbox.run(EvalCase(id="rp1", description="d", turns=["第一轮", "第二轮"]))

    assert trace.error is None
    assert trace.react_steps == 6  # 2 轮 × 3 步
    assert trace.steps_margin_hint is True
    assert trace.forced_finalize is False
    assert trace.protocol_corrections == 2
    snapshot = trace.to_dict()
    assert snapshot["react_steps"] == 6
    assert snapshot["steps_margin_hint"] is True


def test_sandbox_collects_citation_verdict_from_turn_ctx(
    tmp_path, reset_settings, monkeypatch,
):
    """回归（P0）：引用 verdict 从 _last_turn_ctx.citation_verdict 采集。

    单 Agent 重构后 verdict 落在 AgentTurnContext 而非实例属性——
    沙箱读 _last_citation_verdict 恒 None，引用用例被全数判 0（假阴性）。
    同时验证 sources 并集与全轮回复留痕（泄露检查覆盖每一轮）。
    """
    from types import SimpleNamespace

    from app.schemas.response import IntentType

    settings.evolve_capture_enabled = False

    def fake_chat(self, user_input):
        # 与真实 AgentTurnContext 同形（含 citation_verdict/sources）
        self._last_turn_ctx = SimpleNamespace(
            react_steps=1, steps_margin_hint=False,
            forced_finalize=False, protocol_corrections=0,
            citation_verdict={"cited": ["退货政策.md"], "matched": ["退货政策.md"], "missing": []},
            sources={"退货政策"},
        )
        return SimpleNamespace(
            reply=f"回复[{user_input}]", intent=IntentType.GREETING,
            requires_human=False,
        )

    monkeypatch.setattr("app.agent.chat.EcomAgent.chat", fake_chat)
    sandbox = Sandbox(tmp_root=str(tmp_path / "sbx"))
    trace = sandbox.run(EvalCase(id="cit1", description="d", turns=["第一轮", "第二轮"]))

    assert trace.error is None
    assert trace.citation_verdict == {
        "cited": ["退货政策.md"], "matched": ["退货政策.md"], "missing": [],
    }
    assert trace.retrieved_sources == ["退货政策"]
    assert trace.turn_replies == ["回复[第一轮]", "回复[第二轮]"]


def test_sandbox_report_drops_raw_tool_outputs(tmp_path):
    """P1：报告快照默认不携带原始工具结果（2.2 隐私口径）；
    include_tool_outputs=True 时（离线诊断脚本）才落原文。"""
    from app.evaluation.trace import ToolObservation, RunTrace

    trace = RunTrace(case_id="c", turns=["t"])
    trace.tool_observations.append(ToolObservation(
        name="query_order", arguments={"order_id": "ORD-1"},
        result='{"success": true, "order": {"amount": 4697.00}}',
        outcome={"success": True},
    ))

    default_snapshot = trace.to_dict()
    assert "tool_outputs" not in default_snapshot
    # 结构化摘要在（可判定字段），金额等敏感明细不在
    assert default_snapshot["tool_outcomes"][0]["outcome"] == {"success": True}

    full = trace.to_dict(include_tool_outputs=True)
    assert "4697" in full["tool_outputs"][0]["result"]


def test_sandbox_resets_stale_session_file(tmp_path, reset_settings, monkeypatch):
    """同一 Sandbox 实例重跑同 case：上一次会话/播种目录被清理，不串轮。"""
    from types import SimpleNamespace

    from app.schemas.response import IntentType

    settings.evolve_capture_enabled = False

    def fake_chat(self, user_input):
        # 侧写真实会话文件的写入（LocalFileSessionStore 落盘）
        from pathlib import Path

        session_file = Path(self.session_path)
        session_file.write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            reply="ok", intent=IntentType.GREETING, requires_human=False,
        )

    monkeypatch.setattr("app.agent.chat.EcomAgent.chat", fake_chat)
    sandbox = Sandbox(tmp_root=str(tmp_path / "sbx"))
    case = EvalCase(id="dup", description="d", turns=["第一轮"])

    sandbox.run(case)
    assert sandbox.session_path_for("dup").endswith("dup.json")
    assert (tmp_path / "sbx" / "dup.json").exists()

    # 第二次运行：残留会话文件先被删除（重跑可复现，不带上轮历史）
    trace = sandbox.run(case)
    assert trace.error is None


# ============================================================
# with-eval：通过 / 阻断 两条端到端
# ============================================================
class ScriptedEvaluator:
    """按顺序弹出预设评测报告的 Evaluator 替身。"""

    def __init__(self, before_report, after_report):
        self._queue = [before_report, after_report]

    def run_all(self, cases):
        return self._queue.pop(0)


def _report(pass_rate, result_score, passed):
    return {
        "summary": {
            "total": 1, "passed": passed, "pass_rate": pass_rate,
            "avg_process_score": 0.9, "avg_result_score": result_score,
            "total_tokens": 10, "avg_tokens_per_case": 10,
        },
        "cases": [{
            "case_id": "case-1", "description": "退货咨询", "passed": passed,
            "process": {}, "result": {}, "judge_reasons": {},
            "trace": {}, "error": None,
        }],
    }


FAKE_CASES = [EvalCase(id="case-1", description="退货咨询", turns=["支持退货吗"])]


def _make_svc(tmp_path, before, after):
    from test_pipeline import make_services

    # 单例 evaluator：factory 每次调用都返回同一实例，before/after 按序弹出
    evaluator = ScriptedEvaluator(before, after)
    return make_services(
        tmp_path,
        evaluator_factory=lambda: evaluator,
        eval_cases=FAKE_CASES,
    )


def make_services_from_evaluator(tmp_path, evaluator):
    from test_pipeline import make_services

    return make_services(
        tmp_path,
        evaluator_factory=lambda: evaluator,
        eval_cases=FAKE_CASES,
    )


def test_with_eval_blocked_publishes_nothing_and_exit_2(tmp_path, reset_settings):
    settings.self_evolve_enabled = True
    from test_pipeline import write_turn

    # 队列 3 个报告：第一次 run（baseline+after）+ CLI 二次 run（baseline）。
    # 第一次走「通过→回归」剧本 → 阻断；第二次候选被 pending 精确去重零成本跳过。
    before = _report(1.0, 0.9, passed=True)
    after = _report(0.6, 0.7, passed=False)
    evaluator = ScriptedEvaluator(before, after)
    evaluator._queue.extend([_report(1.0, 0.9, passed=True)])
    svc = make_services_from_evaluator(tmp_path, evaluator)
    write_turn(svc["turns_dir"], "t1", question="七天无理由退货可以吗")

    report = svc["pipeline"].run(with_eval=True)
    detail = svc["pipeline"].eval_detail
    assert detail is not None and detail["blocked"] is True
    assert any(r.startswith("regression:") or r.startswith("pass_rate_down")
               for r in detail["reasons"])
    # staging 已清理：无文档、无指针切换、journal 归档；候选进 pending 供人工审核
    assert report.sedimented == 0
    assert report.pending == 1
    assert not list((svc["kb_dir"] / "evolved").glob("*.md"))
    assert svc["pipeline"]._generation_store.active("numpy") is None
    assert (svc["state_dir"] / "journal.json.archived").exists()
    reason = svc["ledger"].list_pending(aging_days=30)[0][1]["reason"]
    assert reason.startswith("eval_blocked:")

    # 第二次运行：pending 命中 → 零成本跳过（不再重复烧 Judge），CLI 正常退出 0
    from app.scripts.run_evolution import main

    assert main(["--with-eval"], services=svc) == 0
    detail2 = svc["pipeline"].eval_detail
    assert detail2 is None or not detail2["blocked"]


def test_with_eval_passes_publishes_and_exit_0(tmp_path, reset_settings):
    settings.self_evolve_enabled = True
    from test_pipeline import write_turn

    svc = _make_svc(
        tmp_path,
        _report(0.6, 0.7, passed=False),  # baseline 差
        _report(1.0, 0.9, passed=True),   # after 变好 → 不阻断
    )
    write_turn(svc["turns_dir"], "t1", question="七天无理由退货可以吗")

    report = svc["pipeline"].run(with_eval=True)
    detail = svc["pipeline"].eval_detail
    assert detail is not None and detail["blocked"] is False
    assert detail["probe_failures"] == []  # 原问题/规范问题均命中
    assert report.sedimented == 1
    assert len(list((svc["kb_dir"] / "evolved").glob("*.md"))) == 1
    assert svc["pipeline"]._generation_store.active("numpy") is not None

    from app.scripts.run_evolution import main

    assert main(["--dry-run"], services=svc) == 0  # 只读子命令不触发运行