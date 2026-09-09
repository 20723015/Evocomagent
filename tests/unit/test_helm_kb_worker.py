"""Helm chart checks for the multi-instance KB worker deployment.

The tests execute the real Helm renderer when the binary is available (CI/kind).
On developer machines without Helm they are skipped; the value/template checks
still document the contract and keep the expected profile names explicit.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
CHART = ROOT / "deploy" / "helm" / "ecom-agent"


def _render(profile: str) -> list[dict]:
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm executable unavailable")
    result = subprocess.run(
        [helm, "template", "kb-render-test", str(CHART), "-f", str(CHART / profile)],
        check=True,
        capture_output=True,
        encoding="utf-8",
        text=True,
    )
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def _by_kind(items: list[dict], kind: str) -> list[dict]:
    return [item for item in items if item.get("kind") == kind]


@pytest.mark.parametrize("profile", ["values-production.yaml", "values-kind.yaml"])
def test_kb_worker_profile_renders_with_single_concurrency(profile):
    docs = _render(profile)
    deployments = [
        d for d in _by_kind(docs, "Deployment")
        if d["metadata"]["name"] == "kb-render-test-kb-worker"
    ]
    assert len(deployments) == 1
    worker = deployments[0]
    assert worker["spec"]["replicas"] == 2

    pod = worker["spec"]["template"]
    affinity = pod["spec"]["affinity"]["podAntiAffinity"]
    if profile == "values-production.yaml":
        assert affinity["requiredDuringSchedulingIgnoredDuringExecution"]
    else:
        assert affinity["preferredDuringSchedulingIgnoredDuringExecution"]

    env = {
        item["name"]: item["value"]
        for item in pod["spec"]["containers"][0]["env"]
    }
    assert env["KB_WORKER_CONCURRENCY"] == "1"
    assert env["KB_WORKER_ENABLED"] == "1"


def test_production_has_worker_pdb_and_migration_hook():
    docs = _render("values-production.yaml")
    pdbs = [
        d for d in _by_kind(docs, "PodDisruptionBudget")
        if d["metadata"]["name"] == "kb-render-test-kb-worker"
    ]
    assert len(pdbs) == 1
    assert pdbs[0]["spec"]["minAvailable"] == 1
    assert pdbs[0]["spec"]["selector"]["matchLabels"]["app"] == "ecom-agent-kb-worker"

    migration = [
        d for d in _by_kind(docs, "Job")
        if d["metadata"]["name"] == "kb-render-test-db-migrate"
    ]
    assert len(migration) == 1
    hooks = migration[0]["metadata"]["annotations"]["helm.sh/hook"]
    assert hooks == "pre-install,pre-upgrade"


def test_kind_worker_has_independent_pdb():
    docs = _render("values-kind.yaml")
    pdbs = [
        d for d in _by_kind(docs, "PodDisruptionBudget")
        if d["metadata"]["name"] == "kb-render-test-kb-worker"
    ]
    assert len(pdbs) == 1
    assert pdbs[0]["spec"]["minAvailable"] == 1


def test_base_worker_is_disabled_and_profiles_enable_two_replicas():
    values = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))
    production = yaml.safe_load(
        (CHART / "values-production.yaml").read_text(encoding="utf-8")
    )
    kind = yaml.safe_load((CHART / "values-kind.yaml").read_text(encoding="utf-8"))
    assert values["kbWorker"]["enabled"] is False
    assert production["kbWorker"]["enabled"] is True
    assert kind["kbWorker"]["enabled"] is True
    assert production["kbWorker"]["replicaCount"] == 2
    assert kind["kbWorker"]["replicaCount"] == 2
    assert production["kbWorker"]["podDisruptionBudget"]["enabled"] is True
    assert kind["kbWorker"]["podDisruptionBudget"]["enabled"] is True


def test_human_qa_capture_and_import_states_are_explicit_per_profile():
    """kind 的采集开关必须和导入 CronJob 同时打开；默认/生产保持关闭。"""
    values = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))
    production = yaml.safe_load(
        (CHART / "values-production.yaml").read_text(encoding="utf-8")
    )
    kind = yaml.safe_load((CHART / "values-kind.yaml").read_text(encoding="utf-8"))

    assert values["humanEvalCron"]["enabled"] is False
    assert values["humanPublishWorker"]["enabled"] is False
    assert production.get("humanEvalCron", values["humanEvalCron"])["enabled"] is False
    assert kind["env"]["HUMAN_QA_EVOLUTION_ENABLED"] == "true"
    assert kind["humanEvalCron"]["enabled"] is True
    assert kind["humanPublishWorker"]["enabled"] is True
    assert kind["kbSharedVolume"]["enabled"] is True
    assert kind["env"]["REDIS_URL"]


def test_kind_human_eval_cron_timezone_and_shared_volume():
    """kind：评审 Cron 02:00 Asia/Shanghai + 共享卷 + 开关；发布 Worker 开启。"""
    docs = _render("values-kind.yaml")
    crons = [
        d for d in _by_kind(docs, "CronJob")
        if d["metadata"]["name"] == "kb-render-test-human-eval"
    ]
    assert len(crons) == 1
    spec = crons[0]["spec"]
    assert spec["schedule"] == "0 2 * * *"
    assert spec.get("timeZone") == "Asia/Shanghai"
    assert spec["concurrencyPolicy"] == "Forbid"
    container = spec["jobTemplate"]["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item["value"] for item in container["env"]}
    assert env["HUMAN_QA_EVOLUTION_ENABLED"] == "true"
    assert env["REDIS_URL"] == "redis://redis-master:6379/0"
    assert container["args"] == ["--human-eval"]
    assert container["volumeMounts"] == [
        {"name": "kb-shared", "mountPath": "/kb-shared"},
    ]

    workers = [
        d for d in _by_kind(docs, "Deployment")
        if d["metadata"]["name"] == "kb-render-test-human-publish"
    ]
    assert len(workers) == 1
    assert workers[0]["spec"]["replicas"] == 2
    wk_container = workers[0]["spec"]["template"]["spec"]["containers"][0]
    wk_args = wk_container.get("args") or []
    assert "--human-publish-worker" in wk_args
