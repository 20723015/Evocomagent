"""ledger.py：processed（cursor）、published、pending、trash、运行报告（第10期）。

ledger.json（{evolve_state_dir} 下）：
{
  "processed": ["turn_id", ...],          # 已挖掘（cursor）
  "published": {"candidate_id": "filename"},
  "rejected":  ["candidate_id", ...],
  "pending":   {"candidate_id": {"turn_id", "question", "answer", "created_at",
                                 "reason", "status": "pending"|"aging",
                                 "human_trusted": bool}},
  "trash":     {"filename": {"candidate_id", "moved_at"}}
}
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.evolution.models import CandidateQA, EvolutionReport


def _parse_ts(text) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(text)
    except (ValueError, TypeError, AttributeError):
        return None


class Ledger:
    """沉淀全过程的持久化账本（单进程写，原子落盘）。"""

    def __init__(self, state_dir, clock=None):
        self._path = Path(state_dir) / "ledger.json"
        self._clock = clock
        self._data = self._load()

    def _now(self) -> datetime:
        return self._clock.now() if self._clock else datetime.now()

    # ---------- 持久化 ----------
    def _load(self) -> dict:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return {
                "processed": list(data.get("processed", [])),
                "published": dict(data.get("published", {})),
                "rejected": list(data.get("rejected", [])),
                "pending": dict(data.get("pending", {})),
                "trash": dict(data.get("trash", {})),
            }
        except (json.JSONDecodeError, OSError, TypeError):
            return {
                "processed": [],
                "published": {},
                "rejected": [],
                "pending": {},
                "trash": {},
            }

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp, self._path)

    # ---------- processed（cursor） ----------
    def processed_set(self) -> set[str]:
        return set(self._data["processed"])

    def mark_processed(self, turn_ids) -> None:
        changed = False
        for t in turn_ids:
            if t not in self._data["processed"]:
                self._data["processed"].append(t)
                changed = True
        if changed:
            self._save()

    def drop_processed(self, turn_ids) -> int:
        """从 processed 移除（prune-turns 删文件后同步收缩游标）；返回移除条数。"""
        to_remove = set(turn_ids)
        if not to_remove:
            return 0
        before = len(self._data["processed"])
        self._data["processed"] = [t for t in self._data["processed"] if t not in to_remove]
        removed = before - len(self._data["processed"])
        if removed:
            self._save()
        return removed

    # ---------- 精确去重判断 ----------
    def contains(self, candidate_id: str) -> bool:
        """candidate 是否已处理过（published/rejected/pending 任一）。

        processed 存的是 turn_id（与 candidate_id 类型不同），不参与比对。
        """
        return (
            candidate_id in self._data["published"]
            or candidate_id in self._data["rejected"]
            or candidate_id in self._data["pending"]
        )

    # ---------- published ----------
    def published(self) -> dict[str, str]:
        return dict(self._data["published"])

    def mark_published(self, candidate_id: str, filename: str) -> None:
        self._data["published"][candidate_id] = filename
        self._save()

    def mark_published_many(self, entries) -> None:
        """批量 mark_published（单次落盘；entries = [(candidate_id, filename), ...]）。"""
        if not entries:
            return
        for cid, filename in entries:
            self._data["published"][cid] = filename
        self._save()

    def commit_publish(self, entries) -> None:
        """发布提交批量版：mark_published + drop_pending 一次落盘（pipeline._publish 用）。"""
        if not entries:
            return
        for cid, filename in entries:
            self._data["published"][cid] = filename
            self._data["pending"].pop(cid, None)
        self._save()

    def unpublish_mark(self, candidate_id: str) -> None:
        """unpublish 后：从 published 移除并记入 rejected（不再命中）。"""
        self._data["published"].pop(candidate_id, None)
        if candidate_id not in self._data["rejected"]:
            self._data["rejected"].append(candidate_id)
        self._save()

    def batch_cleanup_published(self, filename_cids) -> int:
        """批量 unpublish 清账（P2-1 替换 / revalidate 复用）：trash + 移除 published + 记 rejected，
        单次落盘。filename_cids = [(filename, candidate_id), ...]；返回处理条数。"""
        changed = 0
        for filename, cid in filename_cids:
            self._data["trash"][filename] = {
                "candidate_id": cid,
                "moved_at": self._now().isoformat(timespec="seconds"),
            }
            self._data["published"].pop(cid, None)
            if cid not in self._data["rejected"]:
                self._data["rejected"].append(cid)
            changed += 1
        if changed:
            self._save()
        return changed

    # ---------- pending ----------
    def add_pending(self, candidate: CandidateQA, reason: str = "") -> None:
        existing = self._data["pending"].get(candidate.candidate_id)
        if existing:
            existing["reason"] = reason
        else:
            self._data["pending"][candidate.candidate_id] = {
                "turn_id": candidate.turn_id,
                "question": candidate.question,
                "answer": candidate.answer,
                "created_at": self._now().isoformat(timespec="seconds"),
                "reason": reason,
                "status": "pending",
                "human_trusted": False,
                # 持久化质量分：approve 重建候选时带回，人工通过的候选
                # 才能参与 P2-1 近重复替换的质量比较
                "quality_score": float(candidate.quality_score or 0.0),
            }
        self._save()

    def list_pending(self, aging_days: int) -> list[tuple[str, dict]]:
        """返回 (candidate_id, entry) 列表；超龄未审条目标记 aging。"""
        now = self._now()
        changed = False
        for cid, entry in self._data["pending"].items():
            if entry.get("status") == "pending":
                created = _parse_ts(entry.get("created_at"))
                if created is not None and (now - created).days >= aging_days:
                    entry["status"] = "aging"
                    changed = True
        if changed:
            self._save()
        return list(self._data["pending"].items())

    def pending_entry(self, candidate_id: str) -> Optional[dict]:
        return self._data["pending"].get(candidate_id)

    def pending_turn_ids(self) -> set[str]:
        """pending 条目引用的 turn_id 集合（prune-turns 保护用）。"""
        return {
            entry.get("turn_id") for entry in self._data["pending"].values()
            if entry.get("turn_id")
        }

    def approve(self, candidate_id: str) -> Optional[CandidateQA]:
        """人工通过：标记 human_trusted 并取回候选（走发布链路）。"""
        entry = self._data["pending"].get(candidate_id)
        if not entry:
            return None
        entry["human_trusted"] = True
        self._save()
        return CandidateQA(
            candidate_id=candidate_id,
            turn_id=entry.get("turn_id", ""),
            question=entry.get("question", ""),
            answer=entry.get("answer", ""),
            intent="",
            confidence=1.0,
            sources=[],
            filter_state="pending",
            quality_score=float(entry.get("quality_score") or 0.0),
        )

    def reject(self, candidate_id: str) -> None:
        """人工拒绝：永久跳过。"""
        if self._data["pending"].pop(candidate_id, None) is None:
            return
        if candidate_id not in self._data["rejected"]:
            self._data["rejected"].append(candidate_id)
        self._save()

    def drop_pending(self, candidate_id: str) -> None:
        """发布成功后从 pending 移除。"""
        if self._data["pending"].pop(candidate_id, None) is not None:
            self._save()

    def drop_pending_many(self, cids) -> None:
        """批量 drop_pending（单次落盘）。"""
        changed = False
        for cid in cids:
            if self._data["pending"].pop(cid, None) is not None:
                changed = True
        if changed:
            self._save()

    def prune_pending(self, only_aging: bool = True) -> int:
        """清理 pending；prune-pending 只动已 aging 的。清理的条目记入 rejected
        （rejected_expired 语义：永久跳过，杜绝同候选复活）。返回清理条数。"""
        removed = [
            cid for cid, entry in self._data["pending"].items()
            if (not only_aging) or entry.get("status") == "aging"
        ]
        for cid in removed:
            self._data["pending"].pop(cid, None)
            if cid not in self._data["rejected"]:
                self._data["rejected"].append(cid)
        if removed:
            self._save()
        return len(removed)

    # ---------- trash ----------
    def trash_entry(self, filename: str) -> Optional[dict]:
        return self._data["trash"].get(filename)

    def move_to_trash(self, filename: str, candidate_id: str) -> None:
        self._data["trash"][filename] = {
            "candidate_id": candidate_id,
            "moved_at": self._now().isoformat(timespec="seconds"),
        }
        self._save()

    # ---------- 运行报告 ----------
    def write_report(self, report_dir, report: EvolutionReport, run_id: str) -> Path:
        out_dir = Path(report_dir) / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{run_id}-report.json"
        path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path