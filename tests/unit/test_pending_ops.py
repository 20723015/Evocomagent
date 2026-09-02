"""pending 操作：aging / approve / reject / unpublish / prune（含 CLI 子命令）。"""

from __future__ import annotations

import json
import os
import time

import pytest

from app.config.settings import settings
from app.evolution.models import CandidateQA

pytestmark = pytest.mark.usefixtures("reset_settings")


def _candidate(cid="cid1", **overrides):
    base = dict(
        candidate_id=cid, turn_id=f"turn-{cid}",
        question="退款多久到账？", answer="一般 3 个工作日内原路退回，请留意到账通知。",
        intent="after_sale", confidence=0.9, filter_state="pending",
    )
    base.update(overrides)
    return CandidateQA(**base)


def _make_services(tmp_path, clock=None):
    from test_pipeline import make_services

    return make_services(tmp_path, clock=clock)


# ============================================================
# aging
# ============================================================
def test_pending_aging_marking(tmp_state_dir, frozen_clock):
    from app.evolution.ledger import Ledger

    ledger = Ledger(tmp_state_dir["state"], clock=frozen_clock)
    ledger.add_pending(_candidate(), reason="judge_failed")
    entries = ledger.list_pending(aging_days=30)
    assert entries[0][1]["status"] == "pending"

    frozen_clock.advance(days=31)
    entries = ledger.list_pending(aging_days=30)
    assert entries[0][1]["status"] == "aging"


def test_reject_permanent_skip(tmp_state_dir):
    from app.evolution.ledger import Ledger

    ledger = Ledger(tmp_state_dir["state"])
    c = _candidate()
    ledger.add_pending(c)
    ledger.reject(c.candidate_id)
    assert not ledger.pending_entry(c.candidate_id)
    assert ledger.contains(c.candidate_id)  # rejected → 永久跳过


def test_prune_pending_only_aging(tmp_state_dir, frozen_clock):
    from app.evolution.ledger import Ledger

    ledger = Ledger(tmp_state_dir["state"], clock=frozen_clock)
    ledger.add_pending(_candidate("old"), reason="judge_failed")
    frozen_clock.advance(days=31)
    ledger.add_pending(_candidate("fresh"), reason="capacity")  # 新条目未超龄
    ledger.list_pending(aging_days=30)  # 只把 old 标记 aging
    assert ledger.prune_pending(only_aging=True) == 1
    assert ledger.pending_entry("fresh") is not None
    assert ledger.pending_entry("old") is None
    assert ledger.contains("old")  # 被清理的 aging 条目记入 rejected（永久跳过）


# ============================================================
# unpublish：移 trash + 不再命中
# ============================================================
def test_unpublish_moves_to_trash_and_marks(tmp_path, tmp_kb_dir):
    from app.evolution.ledger import Ledger
    from app.evolution.publisher import Publisher

    staging = tmp_path / "staging"
    staging.mkdir()
    publisher = Publisher(kb_dir=tmp_kb_dir, staging_dir=staging)
    ledger = Ledger(tmp_path / "state")

    c = _candidate()
    ledger.add_pending(c)
    filename = publisher.write_staging(c)
    assert filename is not None
    publisher.move_into_kb(filename)
    ledger.mark_published(c.candidate_id, filename)
    src = tmp_kb_dir / "evolved" / filename
    assert src.exists()

    trash = tmp_path / "trash"
    rel = publisher.unpublish(filename, trash)
    assert rel == f"evolved/{filename}"
    assert not src.exists()
    assert (tmp_path / "trash" / filename).exists()

    ledger.move_to_trash(filename, c.candidate_id)
    ledger.unpublish_mark(c.candidate_id)
    assert c.candidate_id not in ledger.published()
    assert ledger.contains(c.candidate_id)  # 不再命中（rejected）
    assert ledger.trash_entry(filename)["candidate_id"] == c.candidate_id


def test_unpublish_missing_file_returns_none(tmp_path, tmp_kb_dir):
    from app.evolution.publisher import Publisher

    staging = tmp_path / "staging"
    staging.mkdir()
    publisher = Publisher(kb_dir=tmp_kb_dir, staging_dir=staging)
    assert publisher.unpublish("不存在.md", tmp_path / "trash") is None


# ============================================================
# CLI 子命令（DI services）
# ============================================================
def _cli_services(tmp_path, clock=None):
    svc = _make_services(tmp_path, clock=clock)
    from app.scripts.run_evolution import main

    return svc, main


def test_cli_list_pending(tmp_path):
    svc, main = _cli_services(tmp_path)
    svc["ledger"].add_pending(_candidate(), reason="judge_failed")
    assert main(["--list-pending"], services=svc) == 0


def test_cli_reject(tmp_path):
    settings.self_evolve_enabled = True
    svc, main = _cli_services(tmp_path)
    svc["ledger"].add_pending(_candidate(), reason="judge_failed")
    assert main(["--reject", "cid1"], services=svc) == 0
    assert not svc["ledger"].pending_entry("cid1")


def test_cli_approve_publishes(tmp_path):
    settings.self_evolve_enabled = True
    svc, main = _cli_services(tmp_path)
    svc["ledger"].add_pending(_candidate(), reason="judge_failed")
    assert main(["--approve", "cid1"], services=svc) == 0
    assert svc["ledger"].published().get("cid1")
    assert len(list((svc["kb_dir"] / "evolved").glob("*.md"))) == 1


def test_cli_prune_turns(tmp_path):
    settings.self_evolve_enabled = True
    from test_pipeline import write_turn

    svc, main = _cli_services(tmp_path)
    write_turn(svc["turns_dir"], "t-processed")
    write_turn(svc["turns_dir"], "t-fresh")
    svc["ledger"].mark_processed(["t-processed", "t-fresh"])

    # 把 t-processed 的 mtime 改为 100 天前
    old = time.time() - 100 * 86400
    for name in ("t-processed.json", "t-fresh.json"):
        path = svc["turns_dir"] / "20260828" / name
        os.utime(path, (old, old))

    assert main(["--prune-turns", "--older-than", "90"], services=svc) == 0
    remains = [p.stem for p in (svc["turns_dir"] / "20260828").glob("*.json")]
    assert remains == []


def test_cli_prune_turns_protects_pending(tmp_path):
    settings.self_evolve_enabled = True
    from test_pipeline import write_turn

    svc, main = _cli_services(tmp_path)
    # pending 候选的 turn_id 是 turn-cid1 → 对应文件受保护
    write_turn(svc["turns_dir"], "turn-cid1")
    write_turn(svc["turns_dir"], "t-free")
    svc["ledger"].mark_processed(["turn-cid1", "t-free"])
    svc["ledger"].add_pending(_candidate(), reason="capacity")

    old = time.time() - 100 * 86400
    for name in ("turn-cid1.json", "t-free.json"):
        path = svc["turns_dir"] / "20260828" / name
        os.utime(path, (old, old))

    assert main(["--prune-turns", "--older-than", "90"], services=svc) == 0
    remains = [p.stem for p in (svc["turns_dir"] / "20260828").glob("*.json")]
    assert remains == ["turn-cid1"]  # 关联 pending 的记录受保护


def test_cli_prune_pending(tmp_path, frozen_clock):
    settings.self_evolve_enabled = True
    svc, main = _cli_services(tmp_path, clock=frozen_clock)
    svc["ledger"].add_pending(_candidate("old"), reason="judge_failed")
    frozen_clock.advance(days=31)
    svc["ledger"].list_pending(aging_days=30)  # 标记 aging
    assert main(["--prune-pending"], services=svc) == 0
    assert svc["ledger"].pending_entry("old") is None


def test_cli_force_unlock(tmp_path):
    svc, main = _cli_services(tmp_path)
    svc["lock"].acquire(phase="run")
    assert svc["lock"].path.exists()
    assert main(["--force-unlock"], services=svc) == 0
    assert not svc["lock"].path.exists()


def test_cli_write_ops_gated(tmp_path):
    settings.self_evolve_enabled = False
    svc, main = _cli_services(tmp_path)
    assert main(["--reject", "cid1"], services=svc) == 1
    assert main([], services=svc) == 1
    assert main(["--dry-run"], services=svc) == 0  # 只读操作不受限