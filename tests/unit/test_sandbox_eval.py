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

    sandbox = Probe(mode="single", tmp_root=str(tmp_path / "sbx"))
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

    sandbox = Probe(mode="single", tmp_root=str(tmp_path / "sbx"))
    sandbox.run(EvalCase(id="c2", description="d", turns=["你好"]))
    assert captured["temperature"] == 0.0  # 评测期间归零（行为变更）
    assert settings.temperature == 0.7  # 全局不受影响


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