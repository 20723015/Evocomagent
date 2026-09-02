"""ledger 批量写（P2-4）：mark_published_many / drop_pending_many / commit_publish /
batch_cleanup_published 单次落盘；drop_processed 供 prune-turns 同步收缩。"""

from __future__ import annotations

from app.evolution.ledger import Ledger
from app.evolution.models import CandidateQA


def _save_counter(ledger):
    counter = {"n": 0}
    orig = ledger._save

    def counting():
        counter["n"] += 1
        return orig()

    ledger._save = counting
    return counter


def _candidate(cid, turn_id="t1"):
    return CandidateQA(candidate_id=cid, turn_id=turn_id,
                       question="退款多久到账？" * 2, answer="一般 3 个工作日内原路退回。")


def test_mark_published_many_single_save(tmp_state_dir):
    ledger = Ledger(tmp_state_dir["state"])
    counter = _save_counter(ledger)
    ledger.mark_published_many([("c1", "f1.md"), ("c2", "f2.md")])
    assert counter["n"] == 1
    assert ledger.published() == {"c1": "f1.md", "c2": "f2.md"}
    ledger.mark_published_many([])  # 空批量不落盘
    assert counter["n"] == 1


def test_drop_pending_many_single_save(tmp_state_dir):
    ledger = Ledger(tmp_state_dir["state"])
    ledger.add_pending(_candidate("c1"), reason="capacity")
    ledger.add_pending(_candidate("c2"), reason="judge_failed")
    counter = _save_counter(ledger)
    ledger.drop_pending_many(["c1", "c3"])  # c3 无 pending → 无变化也不写
    assert counter["n"] == 1
    assert ledger.pending_entry("c1") is None
    assert ledger.pending_entry("c2") is not None
    ledger.drop_pending_many(["c2"])
    assert counter["n"] == 2


def test_commit_publish_single_save(tmp_state_dir):
    """发布提交：mark_published + drop_pending 合并为一次落盘（原逐条全量重写）。"""
    ledger = Ledger(tmp_state_dir["state"])
    ledger.add_pending(_candidate("c1"), reason="ungrounded:x")
    counter = _save_counter(ledger)
    ledger.commit_publish([("c1", "f1.md"), ("c2", "f2.md")])
    assert counter["n"] == 1
    assert ledger.published() == {"c1": "f1.md", "c2": "f2.md"}
    assert ledger.pending_entry("c1") is None  # 同步 drop pending
    assert ledger.pending_entry("c2") is None


def test_batch_cleanup_published_single_save(tmp_state_dir):
    """批量 unpublish 清账（P2-1 替换 / revalidate 复用）：trash + rejected 一次落盘。"""
    ledger = Ledger(tmp_state_dir["state"])
    ledger.mark_published("cid-old", "old.md")
    ledger.mark_published("cid-keep", "keep.md")
    counter = _save_counter(ledger)
    assert ledger.batch_cleanup_published([("old.md", "cid-old")]) == 1
    assert counter["n"] == 1
    assert "cid-old" not in ledger.published()
    assert ledger.contains("cid-old")  # rejected → 不再命中
    assert ledger.trash_entry("old.md")["candidate_id"] == "cid-old"
    assert ledger.published()["cid-keep"] == "keep.md"


def test_drop_processed_shrinks_cursor(tmp_state_dir):
    ledger = Ledger(tmp_state_dir["state"])
    ledger.mark_processed(["t1", "t2", "t3"])
    assert ledger.drop_processed(["t2", "t99"]) == 1
    assert ledger.processed_set() == {"t1", "t3"}
    assert ledger.drop_processed([]) == 0


def test_pending_quality_score_roundtrip(tmp_state_dir):
    """L5：pending 持久化 quality_score，approve 重建候选时带回
    （人工通过的候选才能参与 P2-1 近重复替换的质量比较）。"""
    ledger = Ledger(tmp_state_dir["state"])
    ledger.add_pending(_candidate("c1"), reason="capacity")
    c2 = _candidate("c2")
    c2.quality_score = 0.88
    ledger.add_pending(c2, reason="capacity")
    assert ledger.pending_entry("c1")["quality_score"] == 0.0  # 默认 0
    assert ledger.pending_entry("c2")["quality_score"] == 0.88
    back = ledger.approve("c2")
    assert back.quality_score == 0.88
    assert ledger.approve("c1").quality_score == 0.0
