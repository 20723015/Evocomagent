"""3.1/3.2 冻结协议测试：manifest 指纹、--check 冻结、不同模型 Judge、通过规则分离。

覆盖：
- manifest 内容完整（数据集 SHA/commit/prompt SHA/模型/Judge/阈值/依赖锁）；
- verify_manifest_unchanged：数据集或配置指纹漂移 → False（拒绝合并）；
- generate_eval_data --check：生成结果与冻结集一致通过；不一致失败；
- 3.2 通过规则：tool_efficiency/token_pass 不参与 pass（观测分离）；
- 不同模型 Judge：resolve_judge_model 强制不同模型。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.config.settings import settings


# ============================================================
# manifest
# ============================================================
def test_manifest_contains_full_fingerprint(tmp_path, reset_settings, monkeypatch):
    ds = tmp_path / "cases.json"
    ds.write_text('{"cases": []}', encoding="utf-8")
    monkeypatch.setattr(settings, "rag_backend", "es")
    monkeypatch.setattr(settings, "rag_hybrid", True)
    monkeypatch.setattr(settings, "rag_rerank", "bge-reranker-v2-m3")

    from app.evaluation.manifest import build_manifest

    m = build_manifest(str(ds), num_cases=317, model="m-under-test",
                       judge_model="m-judge")
    assert m["protocol"] == "eval-v2"
    assert m["frozen"] is True
    assert m["dataset"]["sha256"] and m["dataset"]["num_cases"] == 317
    assert m["git"]["commit"]  # 非空（unknown 也可接受）
    assert m["prompts"]["sha256"]
    assert m["model"]["under_test"] == "m-under-test"
    assert m["model"]["judge"] == "m-judge"
    assert m["model"]["temperature"] == settings.temperature
    assert m["model"]["max_tokens"] == settings.llm_max_tokens
    assert m["retrieval"]["backend"] == "es"
    assert m["retrieval"]["hybrid"] is True
    assert m["retrieval"]["rerank"] == "bge-reranker-v2-m3"
    assert m["threshold"]["pass"] == settings.eval_pass_threshold
    assert m["runtime"]["python"]
    assert m["config_hash"]


def test_verify_manifest_unchanged_detects_dataset_drift(tmp_path, reset_settings):
    ds = tmp_path / "cases.json"
    ds.write_text('{"cases": []}', encoding="utf-8")

    from app.evaluation.manifest import build_manifest, verify_manifest_unchanged

    m = build_manifest(str(ds), 1, "m1", "judge1")
    assert verify_manifest_unchanged(
        m, dataset_path=str(ds), model="m1", judge_model="judge1",
    ) is True
    # 数据集内容变化 → 哈希漂移 → False
    ds.write_text('{"cases": [1]}', encoding="utf-8")
    assert verify_manifest_unchanged(
        m, dataset_path=str(ds), model="m1", judge_model="judge1",
    ) is False


def test_verify_manifest_rejects_legacy_report(tmp_path, reset_settings):
    """旧格式报告（无 manifest 字段）→ 拒绝合并。"""
    from app.evaluation.manifest import verify_manifest_unchanged

    assert verify_manifest_unchanged(
        {"summary": {"passed": 1}}, dataset_path="x", model="m", judge_model="j",
    ) is False
    assert verify_manifest_unchanged(
        None, dataset_path="x", model="m", judge_model="j",
    ) is False


def test_manifest_binds_actual_cli_mode_judge_and_runtime_config(
    tmp_path, reset_settings, monkeypatch,
):
    """续跑校验不得把 mode/use_judge 或 guardrail 变化当成同一实验。"""
    ds = tmp_path / "cases.json"
    ds.write_text('{"cases": []}', encoding="utf-8")
    monkeypatch.setattr(settings, "model_name", "settings-model")
    monkeypatch.setattr(settings, "guardrails_enabled", True)

    from app.evaluation.manifest import build_manifest, verify_manifest_unchanged

    manifest = build_manifest(
        str(ds), 1, "cli-model", "cli-judge", mode="multi", use_judge=True,
    )
    assert manifest["execution"]["mode"] == "multi"
    assert manifest["execution"]["use_judge"] is True
    assert manifest["config"]["model"] == "cli-model"
    assert manifest["config"]["judge_model"] == "cli-judge"
    assert verify_manifest_unchanged(
        manifest, dataset_path=str(ds), model="cli-model", judge_model="cli-judge",
        mode="multi", use_judge=True,
    ) is True
    assert verify_manifest_unchanged(
        manifest, dataset_path=str(ds), model="cli-model", judge_model="cli-judge",
        mode="single", use_judge=True,
    ) is False
    assert verify_manifest_unchanged(
        manifest, dataset_path=str(ds), model="cli-model", judge_model="cli-judge",
        mode="multi", use_judge=False,
    ) is False
    monkeypatch.setattr(settings, "guardrails_enabled", False)
    assert verify_manifest_unchanged(
        manifest, dataset_path=str(ds), model="cli-model", judge_model="cli-judge",
        mode="multi", use_judge=True,
    ) is False


def test_verify_manifest_binds_prompt_dependency_and_git_context(
    tmp_path, reset_settings,
):
    """评测代码/依赖/提交变化及旧缺字段均 fail-closed。"""
    ds = tmp_path / "cases.json"
    ds.write_text('{"cases": []}', encoding="utf-8")
    from app.evaluation.manifest import build_manifest, verify_manifest_unchanged

    manifest = build_manifest(str(ds), 1, "m1", "judge1")
    for section, field in (("prompts", "sha256"), ("runtime", "deps_sha256"), ("git", "commit")):
        changed = json.loads(json.dumps(manifest))
        changed[section][field] = "drifted"
        assert verify_manifest_unchanged(
            changed, dataset_path=str(ds), model="m1", judge_model="judge1",
        ) is False

    missing = json.loads(json.dumps(manifest))
    missing.pop("prompts")
    assert verify_manifest_unchanged(
        missing, dataset_path=str(ds), model="m1", judge_model="judge1",
    ) is False
    # Any bypass is explicit and opt-in, never the default resume behavior.
    changed = json.loads(json.dumps(manifest))
    changed["git"]["commit"] = "migration-source"
    assert verify_manifest_unchanged(
        changed, dataset_path=str(ds), model="m1", judge_model="judge1",
        allow_context_mismatch=True,
    ) is True


def test_manifest_redacts_endpoint_credentials_and_queries(
    tmp_path, reset_settings, monkeypatch,
):
    ds = tmp_path / "cases.json"
    ds.write_text('{"cases": []}', encoding="utf-8")
    monkeypatch.setattr(
        settings, "openai_base_url",
        "https://alice:openai-secret@example.test/v1?api_key=query-secret#fragment",
    )
    monkeypatch.setattr(
        settings, "sophnet_embedding_url",
        "https://embed-user:embed-secret@example.test/embeddings?token=embed-query",
    )
    monkeypatch.setattr(
        settings, "rerank_endpoint_url",
        "https://rerank-user:rerank-secret@example.test/rerank?key=rerank-query",
    )
    from app.evaluation.manifest import build_manifest

    manifest = build_manifest(str(ds), 1, "m1", "judge1")
    serialized = json.dumps(manifest, ensure_ascii=False)
    for secret in ("openai-secret", "query-secret", "embed-secret", "embed-query",
                   "rerank-secret", "rerank-query", "alice:"):
        assert secret not in serialized
    assert manifest["execution"]["base_url"] == "https://example.test/v1"
    assert manifest["execution"]["base_url_sha256"]
    assert manifest["retrieval"]["embedding_endpoint"] == "https://example.test/embeddings"
    assert manifest["retrieval"]["embedding_endpoint_sha256"]


# ============================================================
# generate_eval_data --check（冻结保护）
# ============================================================
def test_generate_check_pass_when_consistent(tmp_path, monkeypatch):
    """--check：生成结果与磁盘一致 → 退出 0。"""
    from app.scripts.generate_eval_data import build_golden_cases

    golden = build_golden_cases()
    out = tmp_path / "cases.json"
    out.write_text(
        json.dumps({"cases": golden}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # 直接验证 _dump 语义：重新生成后 == 磁盘内容
    from app.scripts.generate_eval_data import _dump

    _dump(out, golden)
    assert out.read_text(encoding="utf-8") == (
        json.dumps({"cases": golden}, ensure_ascii=False, indent=2) + "\n"
    )


def test_generate_check_fails_on_frozen_change(tmp_path, monkeypatch):
    """冻结集被改动 → --check 拒绝（模拟 CI 门禁）。"""
    from app.scripts.generate_eval_data import _dump, build_golden_cases
    from app.scripts.generate_eval_data import main as gen_main

    golden = build_golden_cases()
    out = tmp_path / "cases.json"
    _dump(out, golden)
    # 人为篡改一条
    mutated = list(golden)
    mutated[0] = {**mutated[0], "turns": ["被改写的输入"]}
    _dump(out, mutated)

    with pytest.raises(SystemExit) as exc:
        gen_main([
            "--out-eval", str(out),
            "--out-multi", str(tmp_path / "multi.json"),
            "--out-retrieval", str(tmp_path / "retrieval.json"),
            "--check",
        ])
    assert exc.value.code == 1


# ============================================================
# 3.2 通过规则：观测指标分离
# ============================================================
def test_pass_ignores_tool_efficiency_and_token(tmp_path, monkeypatch):
    """v2：tool_efficiency 低、token 超预算 → 只要业务维度达标仍通过（观测分离）。"""
    from app.evaluation.dataset import EvalCase
    from app.evaluation.evaluator import EvalResult, Evaluator

    case = EvalCase(
        id="c1", description="d", turns=["t"],
        expected_intent="greeting", expected_tools=["query_order"],
        min_tool_calls=1, max_tokens=100,  # token 预算极小
    )
    eval_ = Evaluator.__new__(Evaluator)
    eval_.pass_threshold = 0.6
    res = EvalResult(case_id="c1", description="d", trace={})
    res.tool_accuracy = 1.0
    res.intent_match = 1.0
    res.keyword_coverage = 1.0
    res.requires_human_match = 1.0
    res.tool_efficiency = 0.2   # 效率差（观测）
    res.token_pass = False      # 超预算（观测）
    assert eval_._decide_pass(case, res) is True

    # 业务维度不达标 → 仍失败
    res2 = EvalResult(case_id="c2", description="d", trace={})
    res2.tool_accuracy = 0.0
    res2.intent_match = 1.0
    assert eval_._decide_pass(case, res2) is False


# ============================================================
# 3.2 不同模型 Judge
# ============================================================
def test_resolve_judge_model_requires_different_model(reset_settings, monkeypatch):
    monkeypatch.setattr(settings, "model_name", "model-A")
    monkeypatch.setattr(settings, "eval_judge_model", "")

    from app.scripts.run_eval import JudgeModelConfigError, resolve_judge_model

    # 显式 --judge-model 不同 → 通过
    assert resolve_judge_model("model-B") == "model-B"
    # settings 提供不同模型的 Judge → 通过
    monkeypatch.setattr(settings, "eval_judge_model", "model-C")
    assert resolve_judge_model("") == "model-C"
    # 与被测相同 → 拒绝
    monkeypatch.setattr(settings, "eval_judge_model", "model-A")
    with pytest.raises(JudgeModelConfigError):
        resolve_judge_model("")
    # 未配置 → 拒绝
    monkeypatch.setattr(settings, "eval_judge_model", "")
    with pytest.raises(JudgeModelConfigError):
        resolve_judge_model("")
