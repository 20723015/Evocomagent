"""lifecycle.py：知识生命周期协调器（发布后治理路由，010/PR-3）。

按文档 frontmatter 的 ``candidate_id`` 选择治理正本：
- ``human-{id}`` → MySQL 人工候选（human_knowledge_candidates）；
- 其余（自动沉淀）→ Ledger（ledger.json）。

消费者：
- 发布替换结算（settle_replacement）：human 旧候选 → superseded/replaced_by；
  自动沉淀旧文档 → ledger trash/published 清账；
- 重接地隔离（settle_revalidation_failure）：human 文档 → 候选回 pending_review
  重审；自动沉淀 → ledger.add_pending（现状）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.agent.rag.loader import parse_frontmatter
from app.config.settings import settings
from app.evolution.human_store import HumanKnowledgeStore
from app.observability.logging import get_logger

log = get_logger("app.evolution.lifecycle")


@dataclass
class DocRef:
    """文档治理正本引用：kind = mysql | ledger；candidate_id 为 int（mysql）或 str。"""

    kind: str
    candidate_id: int | str | None
    filename: str


class KnowledgeLifecycleCoordinator:
    """按 frontmatter candidate_id/source_kind 路由文档治理动作。"""

    def __init__(self, store: HumanKnowledgeStore, ledger, *, kb_dir=None):
        self._store = store
        self._ledger = ledger
        self._kb_dir = Path(kb_dir) if kb_dir is not None else Path(settings.kb_dir)

    # ---------- 路由 ----------
    def resolve(self, filename: str, *, metadata: dict | None = None) -> DocRef:
        """文档 → 治理正本引用（frontmatter 不可读 → ledger 兜底）。

        ``metadata`` 供文件已移入 trash 的恢复路径使用；不能只依赖
        ``evolved/`` 现存文件，否则移动文件与 MySQL 结算之间崩溃会把人工
        文档误路由到 Ledger。
        """
        meta = metadata if metadata is not None else self._frontmatter(filename)
        raw = str((meta or {}).get("candidate_id") or "")
        if raw.startswith("human-") and raw[len("human-") :].isdigit():
            return DocRef(
                kind="mysql",
                candidate_id=int(raw[len("human-") :]),
                filename=filename,
            )
        return DocRef(
            kind="ledger", candidate_id=self._ledger_candidate_id(filename), filename=filename
        )

    def _frontmatter(self, filename: str) -> dict | None:
        path = self._kb_dir / "evolved" / filename
        try:
            meta, _body = parse_frontmatter(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None
        return meta or {}

    def _ledger_candidate_id(self, filename: str) -> str | None:
        """ledger.published 的反向映射（robot 文档 frontmatter 无 candidate_id）。"""
        try:
            for cid, fname in self._ledger.published().items():
                if fname == filename:
                    return cid
        except Exception:  # noqa: BLE001 - 路由失败按未知正本处理
            return None
        return None

    # ---------- 替换结算 ----------
    def settle_replacement(
        self, old_filename: str, new_candidate_id: int | None, *,
        metadata: dict | None = None,
    ) -> bool:
        """旧文档被新发布替换后的清账（按正本路由）。

        返回是否找到并处理了正本；找不到（文档缺失/无归属）返回 False。
        ``metadata``：旧文档已移 trash（发布在阶段 1/5 移文件、阶段 5/5 才
        结算）时显式携带其 frontmatter，否则 resolve 只看 evolved/ 现存
        文件会把人工文档误路由到 Ledger。
        """
        ref = self.resolve(old_filename, metadata=metadata)
        if ref.kind == "mysql" and isinstance(ref.candidate_id, int):
            return self._store.mark_candidate_superseded_by(
                ref.candidate_id, int(new_candidate_id or 0)
            )
        if ref.candidate_id is not None:
            self._ledger.batch_cleanup_published(
                [(old_filename, ref.candidate_id)]
            )
            return True
        # 无归属（robot 文档不在 published / frontmatter 缺失）：trash 记账兜底
        try:
            self._ledger.move_to_trash(old_filename, "")
        except Exception as exc:  # noqa: BLE001 - 清账失败不回滚发布
            log.warning(
                "lifecycle.ledger_cleanup_failed file_prefix=%s err=%s",
                old_filename[:8],
                type(exc).__name__,
            )
        return False

    # ---------- 重接地隔离 ----------
    def settle_revalidation_failure(
        self,
        filename: str,
        reason: str,
        *,
        metadata: dict | None = None,
    ) -> bool:
        """重接地失败隔离：human 候选回 pending_review 重审；robot → False。

        返回值表示“是否属于 MySQL 人工正本”，而不是本次 UPDATE 是否改行。
        因此恢复时遇到已完成结算的 ``pending_review`` 候选仍返回 True，绝不
        因幂等重放而把同一文档再写进 Ledger。
        """
        ref = self.resolve(filename, metadata=metadata)
        if ref.kind == "mysql" and isinstance(ref.candidate_id, int):
            ok = self._store.reset_candidate_revalidation_failed(
                ref.candidate_id, reason
            )
            if ok:
                return True
            current = self._store.get_candidate(ref.candidate_id)
            if current is not None and current.get("status") == "pending_review":
                return True
            raise RuntimeError(
                "人工知识重接地结算状态冲突: "
                f"candidate={ref.candidate_id} "
                f"status={(current or {}).get('status', 'missing')}"
            )
        return False  # 调用方回退到既有 ledger.add_pending 路径
