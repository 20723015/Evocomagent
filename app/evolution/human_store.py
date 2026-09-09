"""human_store.py：人工客服知识链路的 MySQL 正本（迁移 008）。

四张表、一条链路：
    human_conversations（脱敏会话正本，版本/摘要幂等）
    → human_evaluation_jobs（SKIP LOCKED + lease token；退避 5m→24h，6 次 blocked）
    → human_knowledge_candidates（抽取/评分/分类结果；编辑→revision+1→重评）
    → human_publish_batches/items（批量批准 202 → Worker 租约领取 → 一个批次
      一个 staging generation → 原子切代 → MySQL 事务结算）

并发契约：
- 领取即生成新 lease_token——旧持有者续租/提交/写回一律 rowcount=0 失败；
- LLM/embedding/RAG 评审不占 KB 写锁，只在短事务内提交结果；
- 失败不写「已完成评审」：complete 与结果写入同事务，任何一步失败整体回滚，
  任务留在 retry_wait 自动重试。

时间口径与 kb_index_jobs 相同：MySQL 用服务端时间（NOW()），sqlite（测试）
用进程时间——同一方言内读写一致。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta

from sqlalchemy import and_, func, or_, select, text, update

from app.observability.logging import get_logger
from app.stores.base import StorageUnavailableError
from app.stores.sql.schema import (
    human_conversations,
    human_evaluation_jobs,
    human_knowledge_candidates,
    human_publish_batches,
    human_publish_items,
)

log = get_logger("app.evolution.human_store")

# ---- 会话 ----
CONV_PENDING = "pending"
CONV_EVALUATED = "evaluated"

# ---- 评审任务 ----
JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_RETRY_WAIT = "retry_wait"
JOB_COMPLETED = "completed"
JOB_BLOCKED = "blocked"
JOB_TYPE_CONVERSATION = "conversation"
JOB_TYPE_CANDIDATE = "candidate"

# 重试退避（秒）：5m、30m、2h、6h、12h、24h；第 6 次失败 → blocked
RETRY_BACKOFF_SECONDS = (300, 1800, 7200, 21600, 43200, 86400)
MAX_EVAL_ATTEMPTS = len(RETRY_BACKOFF_SECONDS)

# ---- 候选 ----
CAND_PENDING_REVIEW = "pending_review"
CAND_REJECTED = "rejected"
CAND_SUPERSEDED = "superseded"
CAND_PUBLISH_QUEUED = "publish_queued"
CAND_PUBLISHED = "published"
CAND_RETIRED = "retired"

CLASS_NEW = "new"
CLASS_DUPLICATE = "duplicate"
CLASS_UPDATE = "update"

# 证据链状态（010）：ok = 双快照齐备可批准；legacy_evidence_missing = 历史行
# 未回填（禁止批准，审核台可见原因）
EVIDENCE_OK = "ok"
EVIDENCE_LEGACY_MISSING = "legacy_evidence_missing"

# ---- 发布批次 ----
BATCH_QUEUED = "queued"
BATCH_RUNNING = "running"
BATCH_RETRY_WAIT = "retry_wait"
BATCH_COMPLETED = "completed"  # 终态（原 published 重命名；兼容读容忍遗留行）
BATCH_BLOCKED = "blocked"

BATCH_OP_PUBLISH = "publish"
BATCH_OP_RETIRE = "retire"

PUBLISH_RETRY_BACKOFF_SECONDS = (30, 120, 600, 1800, 3600, 21600)
MAX_PUBLISH_ATTEMPTS = len(PUBLISH_RETRY_BACKOFF_SECONDS)

ITEM_QUEUED = "queued"
ITEM_PUBLISHED = "published"
ITEM_REJECTED = "rejected"
ITEM_FAILED = "failed"

_MAX_ERROR_CHARS = 500

# 审批快照参与摘要的字段（canonical JSON → SHA-256；发布前逐项复核）
_APPROVAL_DIGEST_FIELDS = (
    "candidate_id",
    "candidate_revision",
    "source_version",
    "question",
    "answer",
    "value_score",
    "classification",
    "dedup_target_path",
    "approved_by",
    "approved_at",
)


def approval_digest(snap: dict) -> str:
    """审批快照摘要（canonical JSON 的 SHA-256；建批写入、发布前复核）。"""
    import hashlib

    canonical = json.dumps(
        {k: snap.get(k) for k in _APPROVAL_DIGEST_FIELDS},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _local_now() -> datetime:
    """SQLite test/dev follows its local naive DATETIME convention."""
    return datetime.now()  # noqa: DTZ005


def _sanitize_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}".splitlines()[0][:_MAX_ERROR_CHARS]


class HumanKnowledgeConflict(RuntimeError):
    """批量/版本冲突（映射 409）；detail 面向审核员。"""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class HumanLeaseLost(RuntimeError):
    """租约/所有权丢失（旧 token 不得续租或写回）。"""


def conversation_digest(messages: list[dict]) -> str:
    """脱敏后消息正本的 SHA-256（幂等/冲突判据；canonical JSON）。"""
    import hashlib

    canonical = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_or_none(value) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return None


def _load_json_dict(raw) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


class HumanKnowledgeStore:
    """SQLAlchemy 实现（生产 mysql+pymysql；本地/测试 sqlite 同表）。"""

    def __init__(self, engine, *, lease_seconds: int = 1800):
        self._engine = engine
        self._lease = int(lease_seconds)
        self._server_time = engine.dialect.name == "mysql"

    @property
    def engine(self):
        """底层 SQLAlchemy engine（会话栅栏等需要专用连接的组件用）。"""
        return self._engine

    # ---------- 时间口径 ----------
    def _now(self):
        return func.now() if self._server_time else _local_now()

    def _now_plus(self, seconds: int):
        if self._server_time:
            return text(f"NOW() + INTERVAL {int(max(seconds, 0))} SECOND")
        return _local_now() + timedelta(seconds=max(seconds, 0))

    def _lease_is_live(self, column):
        """SQL fencing predicate shared by every lease-owned mutation."""
        return and_(column.isnot(None), column > self._now())

    def _daily_cutoff(self, now_date=None):
        """每日截止线：上海当日 00:00 对应的 UTC 时刻（D4 时区口径）。

        MySQL 服务端时钟为 UTC，ended_at 落库为 UTC-naive——按上海日历切日，
        而不是 UTC 日历（00:00 上海 = 前一日 16:00 UTC）。sqlite（测试）沿用
        本地 naive 午夜。租约/NOW() 列保持服务端时钟自洽（只与彼此比较）。
        """
        if not self._server_time:
            base = now_date or _local_now()
            return base.replace(hour=0, minute=0, second=0, microsecond=0)
        from datetime import timezone

        try:
            from zoneinfo import ZoneInfo

            shanghai = ZoneInfo("Asia/Shanghai")
        except Exception:  # noqa: BLE001 - tz 数据缺失时退回 UTC 日历
            base = now_date or datetime.now(timezone.utc)
            base = base.replace(tzinfo=None)
            return base.replace(hour=0, minute=0, second=0, microsecond=0)
        base = now_date
        if base is None:
            base = datetime.now(timezone.utc)
        elif base.tzinfo is None:
            base = base.replace(tzinfo=timezone.utc)
        sh_now = base.astimezone(shanghai)
        return (
            sh_now.replace(hour=0, minute=0, second=0, microsecond=0)
            .astimezone(timezone.utc)
            .replace(tzinfo=None)
        )

    # ============================================================
    # 会话接入（幂等/冲突/修订）
    # ============================================================
    def ingest_conversation(
        self,
        *,
        source: str,
        external_conversation_id: str,
        source_version: int,
        agent_id: str,
        started_at,
        ended_at,
        messages: list[dict],
        enqueue_evaluation: bool = True,
    ) -> tuple[dict, str]:
        """登记单段会话；会话与评审任务在同一事务提交。"""
        rows = self.ingest_conversations(
            [
                {
                    "source": source,
                    "external_conversation_id": external_conversation_id,
                    "source_version": source_version,
                    "agent_id": agent_id,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "messages": messages,
                    "enqueue_evaluation": enqueue_evaluation,
                }
            ]
        )
        return rows[0]

    def ingest_conversations(self, conversations: list[dict]) -> list[tuple[dict, str]]:
        """原子登记一批会话；任一冲突或存储错误使整批回滚。"""
        try:
            with self._engine.begin() as conn:
                return [
                    self._ingest_conversation_tx(conn, **item) for item in conversations
                ]
        except HumanKnowledgeConflict:
            raise
        except Exception as e:
            raise StorageUnavailableError(f"人工会话写入失败: {e}") from e

    def _ingest_conversation_tx(
        self,
        conn,
        *,
        source: str,
        external_conversation_id: str,
        source_version: int,
        agent_id: str,
        started_at,
        ended_at,
        messages: list[dict],
        enqueue_evaluation: bool = True,
    ) -> tuple[dict, str]:
        digest = conversation_digest(messages)
        versions = (
            conn.execute(
                select(human_conversations)
                .where(
                    human_conversations.c.source == source,
                    human_conversations.c.external_conversation_id
                    == external_conversation_id,
                )
                .order_by(human_conversations.c.source_version.desc())
                .with_for_update()
            )
            .mappings()
            .all()
        )
        latest = int(versions[0]["source_version"]) if versions else None
        if latest is not None and int(source_version) < latest:
            raise HumanKnowledgeConflict(
                f"会话 {source}/{external_conversation_id} v{source_version} "
                f"已过期（当前最高版本 v{latest}）"
            )
        same = next(
            (
                row
                for row in versions
                if int(row["source_version"]) == int(source_version)
            ),
            None,
        )
        if same is not None:
            if same["conversation_digest"] == digest:
                return dict(same), "duplicate"
            raise HumanKnowledgeConflict(
                f"会话 {source}/{external_conversation_id} v{source_version} "
                "已存在且内容不同（摘要不一致）"
            )
        insert = conn.execute(
            human_conversations.insert().values(
                source=source,
                external_conversation_id=external_conversation_id,
                source_version=source_version,
                agent_id=agent_id,
                started_at=started_at,
                ended_at=ended_at,
                message_count=len(messages),
                conversation_digest=digest,
                transcript_json=json.dumps(messages, ensure_ascii=False),
                eval_status=CONV_PENDING,
            )
        )
        conv_id = insert.inserted_primary_key[0]
        if enqueue_evaluation:
            conn.execute(
                human_evaluation_jobs.insert().values(
                    job_type=JOB_TYPE_CONVERSATION,
                    conversation_id=conv_id,
                    status=JOB_QUEUED,
                    error="",
                )
            )
        else:
            conn.execute(
                update(human_conversations)
                .where(
                    human_conversations.c.id == conv_id,
                )
                .values(eval_status=CONV_EVALUATED, evaluation_finished_at=self._now())
            )
        conn.execute(
            update(human_knowledge_candidates)
            .where(
                human_knowledge_candidates.c.conversation_id.in_(
                    select(human_conversations.c.id).where(
                        human_conversations.c.source == source,
                        human_conversations.c.external_conversation_id
                        == external_conversation_id,
                        human_conversations.c.source_version < source_version,
                    )
                ),
                human_knowledge_candidates.c.status.in_(
                    (
                        CAND_PENDING_REVIEW,
                        CAND_PUBLISH_QUEUED,
                    )
                ),
            )
            .values(status=CAND_SUPERSEDED, updated_at=self._now())
        )
        created = (
            conn.execute(
                select(human_conversations).where(
                    human_conversations.c.id == conv_id,
                )
            )
            .mappings()
            .first()
        )
        return dict(created), "created"

    def get_conversation(self, conversation_id: int) -> dict | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    select(human_conversations).where(
                        human_conversations.c.id == conversation_id
                    )
                )
                .mappings()
                .first()
            )
        return dict(row) if row is not None else None

    def conversation_messages(self, conversation_id: int) -> list[dict]:
        conv = self.get_conversation(conversation_id)
        if conv is None:
            return []
        return json.loads(conv["transcript_json"])

    # ============================================================
    # 评审任务（SKIP LOCKED + lease token + 退避）
    # ============================================================
    def claim_evaluation_job(self, worker_id: str, *, now_date=None) -> dict | None:
        """原子领取一个到期任务（queued 或租约过期的 running）。

        会话类任务只处理 ended_at < 当日 00:00 的已结束会话（每日 Cron 语义，
        迟到数据次日自动补收）；候选重评任务不受该过滤限制。
        """
        now = self._now()
        due = or_(
            human_evaluation_jobs.c.next_run_at.is_(None),
            human_evaluation_jobs.c.next_run_at <= now,
        )
        lease_expired = and_(
            human_evaluation_jobs.c.status == JOB_RUNNING,
            human_evaluation_jobs.c.lease_until.isnot(None),
            human_evaluation_jobs.c.lease_until < now,
        )
        today_start = self._daily_cutoff(now_date)
        claimable = or_(
            and_(
                human_evaluation_jobs.c.job_type == JOB_TYPE_CANDIDATE,
                human_evaluation_jobs.c.status.in_((JOB_QUEUED, JOB_RETRY_WAIT)),
                due,
            ),
            and_(
                human_evaluation_jobs.c.job_type == JOB_TYPE_CANDIDATE,
                lease_expired,
            ),
            and_(
                human_evaluation_jobs.c.job_type == JOB_TYPE_CONVERSATION,
                human_conversations.c.ended_at.isnot(None),
                human_conversations.c.ended_at < today_start,
                human_evaluation_jobs.c.status.in_((JOB_QUEUED, JOB_RETRY_WAIT)),
                due,
            ),
            and_(
                human_evaluation_jobs.c.job_type == JOB_TYPE_CONVERSATION,
                lease_expired,
            ),
        )
        try:
            with self._engine.begin() as conn:
                rows = (
                    conn.execute(
                        select(human_evaluation_jobs)
                        .outerjoin(
                            human_conversations,
                            human_conversations.c.id
                            == human_evaluation_jobs.c.conversation_id,
                        )
                        .where(claimable)
                        .order_by(human_evaluation_jobs.c.id)
                        .limit(1)
                        .with_for_update(skip_locked=True)
                    )
                    .mappings()
                    .all()
                )
                for row in rows:
                    token = uuid.uuid4().hex
                    conn.execute(
                        update(human_evaluation_jobs)
                        .where(human_evaluation_jobs.c.id == row["id"])
                        .values(
                            status=JOB_RUNNING,
                            lease_owner=worker_id,
                            lease_token=token,
                            lease_until=self._now_plus(self._lease),
                            attempts=row["attempts"] + 1,
                            error="",
                            updated_at=self._now(),
                        )
                    )
                    job = dict(row)
                    job["lease_token"] = token
                    job["attempts"] = int(row["attempts"] or 0) + 1
                    return job
            return None
        except Exception as e:  # noqa: BLE001 - queue claim fails closed
            log.warning("human_eval.claim_failed err=%s", type(e).__name__)
            return None

    def heartbeat_evaluation(self, job_id: int, lease_token: str) -> bool:
        try:
            with self._engine.begin() as conn:
                updated = conn.execute(
                    update(human_evaluation_jobs)
                    .where(
                        human_evaluation_jobs.c.id == job_id,
                        human_evaluation_jobs.c.lease_token == lease_token,
                        human_evaluation_jobs.c.status == JOB_RUNNING,
                        self._lease_is_live(human_evaluation_jobs.c.lease_until),
                    )
                    .values(
                        lease_until=self._now_plus(self._lease), updated_at=self._now()
                    )
                )
        except Exception:  # noqa: BLE001 - heartbeat must fail closed
            return False
        return updated.rowcount == 1

    def complete_evaluation(
        self,
        job: dict,
        *,
        candidates: list[dict],
        eval_meta: dict,
    ) -> None:
        """短事务结算：候选行 + 任务 completed + 会话 evaluated（同事务）。

        candidates 元素：question/answer/evidence_message_ids/value_score/
        worth_saving/value_reason/max_similarity/novelty_score/composite_score/
        rag_hit_path/rag_hit_kind/classification/status/reject_reason。
        失败整体回滚（任务留在 running 由租约接管重试）。
        """
        try:
            with self._engine.begin() as conn:
                owned = conn.execute(
                    update(human_evaluation_jobs)
                    .where(
                        human_evaluation_jobs.c.id == job["id"],
                        human_evaluation_jobs.c.lease_token == job["lease_token"],
                        human_evaluation_jobs.c.status == JOB_RUNNING,
                        self._lease_is_live(human_evaluation_jobs.c.lease_until),
                    )
                    .values(
                        status=JOB_COMPLETED,
                        error="",
                        finished_at=self._now(),
                        updated_at=self._now(),
                    )
                )
                if owned.rowcount != 1:
                    raise HumanLeaseLost(f"评审任务 {job['id']} 所有权已丢失，结果作废")
                stale_conversation = False
                if job.get("conversation_id"):
                    conversation = (
                        conn.execute(
                            select(human_conversations).where(
                                human_conversations.c.id == job["conversation_id"],
                            )
                        )
                        .mappings()
                        .first()
                    )
                    if conversation is None:
                        stale_conversation = True
                    else:
                        latest_version = conn.execute(
                            select(
                                func.max(human_conversations.c.source_version)
                            ).where(
                                human_conversations.c.source == conversation["source"],
                                human_conversations.c.external_conversation_id
                                == conversation["external_conversation_id"],
                            )
                        ).scalar()
                        stale_conversation = int(latest_version or 0) > int(
                            conversation["source_version"],
                        )
                if stale_conversation:
                    candidates = []
                for c in candidates:
                    source_snap = c.get("source_snapshot")
                    evidence_snap = c.get("evidence_snapshot")
                    conn.execute(
                        human_knowledge_candidates.insert().values(
                            conversation_id=job["conversation_id"],
                            status=c.get("status", CAND_PENDING_REVIEW),
                            reject_reason=c.get("reject_reason", ""),
                            question=c["question"],
                            answer=c["answer"],
                            evidence_message_ids=json.dumps(
                                c.get("evidence_message_ids", []), ensure_ascii=False
                            ),
                            value_score=c.get("value_score"),
                            worth_saving=int(c.get("worth_saving") or 0),
                            value_reason=c.get("value_reason", ""),
                            max_similarity=c.get("max_similarity"),
                            novelty_score=c.get("novelty_score"),
                            composite_score=c.get("composite_score"),
                            rag_hit_path=c.get("rag_hit_path", ""),
                            rag_hit_kind=c.get("rag_hit_kind", ""),
                            classification=c.get("classification", ""),
                            # 证据链快照（010）：双快照齐备 → ok（可批准）
                            source_snapshot_json=_json_or_none(source_snap),
                            evidence_snapshot_json=_json_or_none(evidence_snap),
                            evidence_state=(
                                EVIDENCE_OK
                                if source_snap is not None
                                and evidence_snap is not None
                                else EVIDENCE_LEGACY_MISSING
                            ),
                            dedup_snapshot_json=_json_or_none(
                                c.get("dedup_snapshot")
                            ),
                            eval_model=eval_meta.get("model", ""),
                            eval_prompt_version=eval_meta.get("prompt_version", ""),
                            eval_embedding_version=eval_meta.get(
                                "embedding_version", ""
                            ),
                            eval_kb_generation=eval_meta.get("kb_generation", ""),
                        )
                    )
                if job.get("conversation_id"):
                    conn.execute(
                        update(human_conversations)
                        .where(human_conversations.c.id == job["conversation_id"])
                        .values(
                            eval_status=CONV_EVALUATED,
                            evaluation_finished_at=self._now(),
                            updated_at=self._now(),
                        )
                    )
        except HumanLeaseLost:
            raise
        except Exception as e:
            raise StorageUnavailableError(f"评审结果写入失败: {e}") from e

    def fail_evaluation(self, job: dict, exc: Exception) -> str:
        """失败分流：固定退避表；MAX 次后 blocked（人工可重试）。"""
        attempts = int(job.get("attempts", 0) or 0)
        if attempts >= MAX_EVAL_ATTEMPTS:
            new_status, delay = JOB_BLOCKED, 0
        else:
            new_status = JOB_RETRY_WAIT
            delay = RETRY_BACKOFF_SECONDS[min(attempts, len(RETRY_BACKOFF_SECONDS)) - 1]
        values = {
            "status": new_status,
            "error": _sanitize_error(exc),
            "lease_owner": "",
            "lease_token": "",
            "lease_until": None,
            "updated_at": self._now(),
        }
        values["next_run_at"] = self._now_plus(delay) if delay else None
        if new_status == JOB_BLOCKED:
            values["finished_at"] = self._now()
        try:
            with self._engine.begin() as conn:
                updated = conn.execute(
                    update(human_evaluation_jobs)
                    .where(
                        human_evaluation_jobs.c.id == job["id"],
                        human_evaluation_jobs.c.lease_token == job["lease_token"],
                        human_evaluation_jobs.c.status == JOB_RUNNING,
                        self._lease_is_live(human_evaluation_jobs.c.lease_until),
                    )
                    .values(**values)
                )
        except Exception as e:
            raise StorageUnavailableError(f"评审失败落库失败: {e}") from e
        if updated.rowcount != 1:
            raise HumanLeaseLost(f"评审任务 {job['id']} 所有权已丢失")
        return new_status

    def retry_blocked_job(self, job_id: int) -> dict | None:
        """人工重试 blocked 任务（清租约/计数回零重新排队）。"""
        try:
            with self._engine.begin() as conn:
                updated = conn.execute(
                    update(human_evaluation_jobs)
                    .where(
                        human_evaluation_jobs.c.id == job_id,
                        human_evaluation_jobs.c.status == JOB_BLOCKED,
                    )
                    .values(
                        status=JOB_QUEUED,
                        attempts=0,
                        next_run_at=None,
                        error="",
                        lease_owner="",
                        lease_token="",
                        lease_until=None,
                        finished_at=None,
                        updated_at=self._now(),
                    )
                )
        except Exception as e:
            raise StorageUnavailableError(f"blocked 任务重试失败: {e}") from e
        return self.get_evaluation_job(job_id) if updated.rowcount == 1 else None

    def get_evaluation_job(self, job_id: int) -> dict | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    select(human_evaluation_jobs).where(
                        human_evaluation_jobs.c.id == job_id
                    )
                )
                .mappings()
                .first()
            )
        return dict(row) if row is not None else None

    # ============================================================
    # 候选（编辑/拒绝/列表/重评）
    # ============================================================
    def get_candidate(self, candidate_id: int) -> dict | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    select(human_knowledge_candidates).where(
                        human_knowledge_candidates.c.id == candidate_id
                    )
                )
                .mappings()
                .first()
            )
        return dict(row) if row is not None else None

    def list_candidates(
        self, status: str = "", limit: int = 50, offset: int = 0
    ) -> tuple[list[dict], int]:
        limit = min(max(int(limit), 1), 200)
        offset = max(int(offset), 0)
        conds = []
        if status:
            conds.append(human_knowledge_candidates.c.status == status)
        try:
            with self._engine.connect() as conn:
                total = int(
                    conn.execute(
                        select(func.count())
                        .select_from(human_knowledge_candidates)
                        .where(*conds)
                    ).scalar()
                    or 0
                )
                rows = (
                    conn.execute(
                        select(human_knowledge_candidates)
                        .where(*conds)
                        .order_by(human_knowledge_candidates.c.id.desc())
                        .limit(limit)
                        .offset(offset)
                    )
                    .mappings()
                    .all()
                )
        except Exception as e:
            raise StorageUnavailableError(f"候选查询失败: {e}") from e
        return [dict(r) for r in rows], total

    def edit_candidate(
        self, candidate_id: int, question: str, answer: str, expected_revision: int
    ) -> tuple[dict | None, int]:
        """审核员编辑：锁行校验 revision；revision+1 + 旧评分过期 + 重评入队。

        返回 (更新后行或 None, 当前 revision)；状态非 pending_review → None。
        同事务完成（编辑与重评入队原子）。
        """
        from app.evolution.sanitizer import (
            has_injection,
            has_pii,
            normalize_answer,
            normalize_question,
        )

        question = normalize_question(question)
        answer = normalize_answer(answer)
        if not question or not answer:
            raise HumanKnowledgeConflict("问题或答案格式/长度不合法")
        if has_pii(question + "\n" + answer) or has_injection(
            question + "\n" + answer,
        ):
            raise HumanKnowledgeConflict("问题或答案含敏感信息或提示注入")
        try:
            with self._engine.begin() as conn:
                row = (
                    conn.execute(
                        select(human_knowledge_candidates)
                        .where(human_knowledge_candidates.c.id == candidate_id)
                        .with_for_update()
                    )
                    .mappings()
                    .first()
                )
                if row is None or row["status"] != CAND_PENDING_REVIEW:
                    return None, int((row or {}).get("revision", -1) or -1)
                current = int(row["revision"] or 0)
                if int(expected_revision) != current:
                    return None, current
                conn.execute(
                    update(human_knowledge_candidates)
                    .where(human_knowledge_candidates.c.id == candidate_id)
                    .values(
                        question=question,
                        answer=answer,
                        revision=current + 1,
                        score_stale=1,
                        updated_at=self._now(),
                    )
                )
                conn.execute(
                    human_evaluation_jobs.insert().values(
                        job_type=JOB_TYPE_CANDIDATE,
                        candidate_id=candidate_id,
                        status=JOB_QUEUED,
                        error="",
                    )
                )
                fresh = (
                    conn.execute(
                        select(human_knowledge_candidates).where(
                            human_knowledge_candidates.c.id == candidate_id
                        )
                    )
                    .mappings()
                    .first()
                )
                return dict(fresh), current + 1
        except Exception as e:
            raise StorageUnavailableError(f"候选编辑失败: {e}") from e

    def settle_candidate_evaluation(
        self, job: dict, scores: dict, eval_meta: dict
    ) -> None:
        """候选评分与评审任务完成在同一事务内提交（租约 fencing）。"""
        try:
            with self._engine.begin() as conn:
                owned = conn.execute(
                    update(human_evaluation_jobs)
                    .where(
                        human_evaluation_jobs.c.id == job["id"],
                        human_evaluation_jobs.c.candidate_id == job["candidate_id"],
                        human_evaluation_jobs.c.lease_token == job["lease_token"],
                        human_evaluation_jobs.c.status == JOB_RUNNING,
                        self._lease_is_live(human_evaluation_jobs.c.lease_until),
                    )
                    .values(
                        status=JOB_COMPLETED,
                        error="",
                        finished_at=self._now(),
                        updated_at=self._now(),
                    )
                )
                if owned.rowcount != 1:
                    raise HumanLeaseLost(f"评审任务 {job['id']} 所有权或租约已丢失")
                updated = conn.execute(
                    update(human_knowledge_candidates)
                    .where(
                        human_knowledge_candidates.c.id == job["candidate_id"],
                        human_knowledge_candidates.c.revision
                        == scores.get("expected_revision"),
                    )
                    .values(
                        value_score=scores.get("value_score"),
                        worth_saving=int(scores.get("worth_saving") or 0),
                        value_reason=scores.get("value_reason", ""),
                        max_similarity=scores.get("max_similarity"),
                        novelty_score=scores.get("novelty_score"),
                        composite_score=scores.get("composite_score"),
                        rag_hit_path=scores.get("rag_hit_path", ""),
                        rag_hit_kind=scores.get("rag_hit_kind", ""),
                        classification=scores.get("classification", ""),
                        status=scores.get("status", CAND_PENDING_REVIEW),
                        reject_reason=scores.get("reject_reason", ""),
                        score_stale=0,
                        dedup_snapshot_json=(
                            _json_or_none(scores.get("dedup_snapshot"))
                            if scores.get("dedup_snapshot") is not None
                            else human_knowledge_candidates.c.dedup_snapshot_json
                        ),
                        eval_model=eval_meta.get("model", ""),
                        eval_prompt_version=eval_meta.get("prompt_version", ""),
                        eval_embedding_version=eval_meta.get("embedding_version", ""),
                        eval_kb_generation=eval_meta.get("kb_generation", ""),
                        updated_at=self._now(),
                    )
                )
                if updated.rowcount != 1:
                    raise HumanLeaseLost(
                        f"候选 {job['candidate_id']} revision 已漂移，结果作废"
                    )
        except HumanLeaseLost:
            raise
        except Exception as e:
            raise StorageUnavailableError(f"候选重评结算失败: {e}") from e

    def mark_scored(self, candidate_id: int, lease_token: str, scores: dict) -> None:
        """兼容入口：必须找到仍有效的对应重评租约，再原子完成任务。"""
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    select(human_evaluation_jobs).where(
                        human_evaluation_jobs.c.candidate_id == candidate_id,
                        human_evaluation_jobs.c.lease_token == lease_token,
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            raise HumanLeaseLost(f"候选 {candidate_id} 无有效评分租约")
        self.settle_candidate_evaluation(dict(row), scores, {})

    def reject_candidate(self, candidate_id: int, reason: str) -> dict | None:
        try:
            with self._engine.begin() as conn:
                updated = conn.execute(
                    update(human_knowledge_candidates)
                    .where(
                        human_knowledge_candidates.c.id == candidate_id,
                        human_knowledge_candidates.c.status == CAND_PENDING_REVIEW,
                    )
                    .values(
                        status=CAND_REJECTED,
                        reject_reason=reason[:64],
                        updated_at=self._now(),
                    )
                )
        except Exception as e:
            raise StorageUnavailableError(f"候选拒绝失败: {e}") from e
        return self.get_candidate(candidate_id) if updated.rowcount == 1 else None

    def migrate_candidate(
        self,
        *,
        conversation_id: int,
        question: str,
        answer: str,
        value_score: float,
        submitted_by: str,
    ) -> int | None:
        """Ledger 一次性迁移：幂等（同会话同问题已存在 → 返回 None）。"""
        try:
            with self._engine.begin() as conn:
                exists = conn.execute(
                    select(human_knowledge_candidates.c.id).where(
                        human_knowledge_candidates.c.conversation_id == conversation_id,
                        human_knowledge_candidates.c.question == question[:200],
                    )
                ).first()
                if exists is not None:
                    return None
                insert = conn.execute(
                    human_knowledge_candidates.insert().values(
                        conversation_id=conversation_id,
                        status=CAND_PENDING_REVIEW,
                        question=question[:200],
                        answer=answer,
                        evidence_message_ids="[]",
                        value_score=value_score,
                        worth_saving=1,
                        value_reason="migrated_from_ledger",
                        classification=CLASS_NEW,
                        eval_model="ledger-migration",
                    )
                )
                return insert.inserted_primary_key[0]
        except Exception as e:
            raise StorageUnavailableError(f"候选迁移失败: {e}") from e

    def cleanup_expired_conversations(self, days: int, limit: int = 500) -> int:
        """清理过期会话正文，但永久保留会话版本头与摘要。

        NOT EXISTS 门禁：该会话下存在证据链缺失且仍在活跃生命周期
        （pending_review/publish_queued/published）的候选 → 跳过本轮——
        正文清空后 legacy 候选永远无法回填证据，必须先人工处理候选。

        保留 source/external_conversation_id/source_version 是发布正确性要求：
        审批、会话 fence 与激活前复核都必须在 180 天后仍能识别更高版本。
        """
        if days <= 0:
            return 0
        cutoff = (
            text(f"NOW() - INTERVAL {int(days * 86400)} SECOND")
            if self._server_time
            else _local_now() - timedelta(seconds=days * 86400)
        )
        active_legacy_candidate = (
            select(human_knowledge_candidates.c.id)
            .where(
                human_knowledge_candidates.c.conversation_id
                == human_conversations.c.id,
                human_knowledge_candidates.c.evidence_state != EVIDENCE_OK,
                human_knowledge_candidates.c.status.in_(
                    (CAND_PENDING_REVIEW, CAND_PUBLISH_QUEUED, CAND_PUBLISHED)
                ),
            )
            .exists()
        )
        try:
            with self._engine.begin() as conn:
                ids = [
                    r[0]
                    for r in conn.execute(
                        select(human_conversations.c.id)
                        .where(
                            human_conversations.c.evaluation_finished_at.isnot(None),
                            human_conversations.c.evaluation_finished_at < cutoff,
                            human_conversations.c.transcript_purged_at.is_(None),
                            ~active_legacy_candidate,
                        )
                        .limit(limit)
                    ).fetchall()
                ]
                if ids:
                    conn.execute(
                        update(human_conversations)
                        .where(
                            human_conversations.c.id.in_(ids)
                        )
                        .values(
                            transcript_json="[]",
                            message_count=0,
                            transcript_purged_at=self._now(),
                            updated_at=self._now(),
                        )
                    )
        except Exception as e:
            raise StorageUnavailableError(f"会话保留期清理失败: {e}") from e
        return len(ids)

    # ============================================================
    # 发布批次（先整体校验，同事务建批次并置 publish_queued）
    # ============================================================
    def create_publish_batch(
        self,
        items: list[dict],
        requested_by: str,
    ) -> tuple[dict, list[dict]]:
        """items = [{"candidate_id", "revision"}]；任一冲突整体 409。

        校验：候选存在、status=pending_review、score_stale=0、revision 匹配、
        classification ∈ {new, update}（duplicate/低分已被自动终结）、
        evidence_state=ok（证据链快照齐备）、会话无更高 source_version。
        同事务：INSERT 批次/items（含不可变审批快照 + digest）+ 候选 →
        publish_queued。发布内容从此取自快照——批准什么就发什么。
        """
        if not items:
            raise HumanKnowledgeConflict("发布批次不能为空")
        if len(items) > 100:
            raise HumanKnowledgeConflict("单个发布批次最多 100 条候选")
        candidate_ids = [int(it["candidate_id"]) for it in items]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise HumanKnowledgeConflict("发布批次包含重复 candidate_id")
        try:
            with self._engine.begin() as conn:
                rows = (
                    conn.execute(
                        select(human_knowledge_candidates)
                        .where(human_knowledge_candidates.c.id.in_(candidate_ids))
                        .with_for_update()
                    )
                    .mappings()
                    .all()
                )
                by_id = {int(r["id"]): dict(r) for r in rows}
                conversation_ids = {
                    int(r["conversation_id"]) for r in rows if r["conversation_id"]
                }
                conversations = {}
                if conversation_ids:
                    conversations = {
                        int(c["id"]): dict(c)
                        for c in conn.execute(
                            select(
                                human_conversations.c.id,
                                human_conversations.c.source,
                                human_conversations.c.external_conversation_id,
                                human_conversations.c.source_version,
                            ).where(human_conversations.c.id.in_(conversation_ids))
                        )
                        .mappings()
                        .all()
                    }
                snapshots: dict[int, dict] = {}
                for it in items:
                    cid = int(it["candidate_id"])
                    row = by_id.get(cid)
                    if row is None:
                        raise HumanKnowledgeConflict(f"候选 {cid} 不存在")
                    if row["status"] != CAND_PENDING_REVIEW:
                        raise HumanKnowledgeConflict(
                            f"候选 {cid} 状态为 {row['status']}，不可批准发布"
                        )
                    if int(row["score_stale"] or 0):
                        raise HumanKnowledgeConflict(
                            f"候选 {cid} 编辑后评分已过期，需等待重评完成"
                        )
                    if int(it.get("revision", -1)) != int(row["revision"] or 0):
                        raise HumanKnowledgeConflict(
                            f"候选 {cid} revision 不匹配（当前 {row['revision']}）"
                        )
                    if row["classification"] not in (CLASS_NEW, CLASS_UPDATE):
                        raise HumanKnowledgeConflict(
                            f"候选 {cid} 分类为 {row['classification'] or '未评审'}"
                            "，不可批准发布"
                        )
                    if row["evidence_state"] != EVIDENCE_OK:
                        raise HumanKnowledgeConflict(
                            f"候选 {cid} 证据链快照缺失"
                            f"（{row['evidence_state'] or '未知'}），禁止批准",
                        )
                    conv = conversations.get(int(row["conversation_id"] or 0))
                    source_version = int(conv["source_version"]) if conv else 0
                    if conv is not None:
                        latest = conn.execute(
                            select(func.max(human_conversations.c.source_version)).where(
                                human_conversations.c.source == conv["source"],
                                human_conversations.c.external_conversation_id
                                == conv["external_conversation_id"],
                            )
                        ).scalar()
                        if int(latest or 0) > int(conv["source_version"]):
                            raise HumanKnowledgeConflict(
                                f"候选 {cid} 所属会话已接入更高版本"
                                f"（v{latest} > v{conv['source_version']}）",
                            )
                    dedup = _load_json_dict(row.get("dedup_snapshot_json"))
                    q_hit = (dedup.get("question") or {}) if dedup else {}
                    # 审批时间用 Python UTC-naive（快照 digest 需与落库列一致）。
                    # 微秒必须清零：MySQL DATETIME(0) 对小数秒四舍五入，列值会与
                    # 摘要串差 1 秒 → 发布端 digest 复核必失败（2026-09 实测）。
                    from datetime import timezone as _tz

                    approved_at = datetime.now(_tz.utc).replace(
                        tzinfo=None, microsecond=0
                    )
                    approved_at_iso = approved_at.isoformat(sep=" ", timespec="seconds")
                    snap = {
                        "candidate_id": cid,
                        "candidate_revision": int(row["revision"] or 0),
                        "source_version": source_version,
                        "question": row["question"],
                        "answer": row["answer"],
                        "value_score": row.get("value_score"),
                        "classification": row["classification"],
                        "dedup_target_path": str(q_hit.get("path", "") or ""),
                        "approved_by": requested_by,
                        "approved_at": approved_at_iso,
                    }
                    snapshots[cid] = snap
                batch = conn.execute(
                    human_publish_batches.insert().values(
                        requested_by=requested_by,
                        item_count=len(items),
                        operation=BATCH_OP_PUBLISH,
                        error="",
                    )
                )
                batch_id = batch.inserted_primary_key[0]
                for it in items:
                    cid = int(it["candidate_id"])
                    snap = snapshots[cid]
                    conn.execute(
                        human_publish_items.insert().values(
                            batch_id=batch_id,
                            candidate_id=cid,
                            candidate_revision=snap["candidate_revision"],
                            source_version=snap["source_version"],
                            question=snap["question"],
                            answer=snap["answer"],
                            value_score=snap["value_score"],
                            classification=snap["classification"],
                            dedup_target_path=snap["dedup_target_path"],
                            approved_by=snap["approved_by"],
                            approved_at=datetime.fromisoformat(snap["approved_at"]),
                            approval_digest=approval_digest(snap),
                        )
                    )
                    conn.execute(
                        update(human_knowledge_candidates)
                        .where(human_knowledge_candidates.c.id == cid)
                        .values(
                            status=CAND_PUBLISH_QUEUED,
                            publish_batch_id=batch_id,
                            updated_at=self._now(),
                        )
                    )
                batch_row = (
                    conn.execute(
                        select(human_publish_batches).where(
                            human_publish_batches.c.id == batch_id
                        )
                    )
                    .mappings()
                    .first()
                )
                item_rows = (
                    conn.execute(
                        select(human_publish_items).where(
                            human_publish_items.c.batch_id == batch_id
                        )
                    )
                    .mappings()
                    .all()
                )
                return dict(batch_row), [dict(r) for r in item_rows]
        except HumanKnowledgeConflict:
            raise
        except Exception as e:
            raise StorageUnavailableError(f"发布批次创建失败: {e}") from e

    def claim_publish_batch(
        self, worker_id: str, *, batch_id: int | None = None
    ) -> dict | None:
        """租约领取待发布批次（queued 或租约过期的 running）。

        排序 operation DESC：'retire' > 'publish' 字典序——补偿/人工下架
        优先于新发布（避免新发布反复重建索引饿死下架）。
        """
        now = self._now()
        claimable = or_(
            and_(
                human_publish_batches.c.status.in_((BATCH_QUEUED, BATCH_RETRY_WAIT)),
                or_(
                    human_publish_batches.c.next_run_at.is_(None),
                    human_publish_batches.c.next_run_at <= now,
                ),
            ),
            and_(
                human_publish_batches.c.status == BATCH_RUNNING,
                human_publish_batches.c.lease_until.isnot(None),
                human_publish_batches.c.lease_until < now,
            ),
        )
        try:
            with self._engine.begin() as conn:
                rows = (
                    conn.execute(
                        select(human_publish_batches)
                        .where(claimable)
                        .where(
                            human_publish_batches.c.id == batch_id
                            if batch_id is not None
                            else text("1=1")
                        )
                        .order_by(
                            human_publish_batches.c.operation.desc(),
                            human_publish_batches.c.id,
                        )
                        .limit(1)
                        .with_for_update(skip_locked=True)
                    )
                    .mappings()
                    .all()
                )
                for row in rows:
                    token = uuid.uuid4().hex
                    conn.execute(
                        update(human_publish_batches)
                        .where(human_publish_batches.c.id == row["id"])
                        .values(
                            status=BATCH_RUNNING,
                            lease_owner=worker_id,
                            lease_token=token,
                            lease_until=self._now_plus(self._lease),
                            attempts=row["attempts"] + 1,
                            updated_at=self._now(),
                        )
                    )
                    batch = dict(row)
                    batch["lease_token"] = token
                    batch["attempts"] = int(row["attempts"] or 0) + 1
                    batch["items"] = [
                        dict(r)
                        for r in conn.execute(
                            select(human_publish_items)
                            .where(human_publish_items.c.batch_id == row["id"])
                            .order_by(human_publish_items.c.id)
                        )
                        .mappings()
                        .all()
                    ]
                    return batch
            return None
        except Exception as e:  # noqa: BLE001 - queue claim fails closed
            log.warning("human_publish.claim_failed err=%s", type(e).__name__)
            return None

    def heartbeat_batch(self, batch_id: int, lease_token: str) -> bool:
        try:
            with self._engine.begin() as conn:
                updated = conn.execute(
                    update(human_publish_batches)
                    .where(
                        human_publish_batches.c.id == batch_id,
                        human_publish_batches.c.lease_token == lease_token,
                        human_publish_batches.c.status == BATCH_RUNNING,
                        self._lease_is_live(human_publish_batches.c.lease_until),
                    )
                    .values(
                        lease_until=self._now_plus(self._lease), updated_at=self._now()
                    )
                )
        except Exception:  # noqa: BLE001 - heartbeat must fail closed
            return False
        return updated.rowcount == 1

    def check_batch_owner(self, batch_id: int, lease_token: str) -> bool:
        """Read-only fencing check used immediately before shared KB side effects."""
        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    select(human_publish_batches.c.id).where(
                        human_publish_batches.c.id == batch_id,
                        human_publish_batches.c.lease_token == lease_token,
                        human_publish_batches.c.status == BATCH_RUNNING,
                        self._lease_is_live(human_publish_batches.c.lease_until),
                    )
                ).first()
        except Exception as e:
            raise StorageUnavailableError(f"发布批次所有权检查失败: {e}") from e
        return row is not None

    def fail_batch(
        self, batch: dict, exc: Exception, *, post_commit: bool = False
    ) -> None:
        """发布失败退避；提交点后永久重试，不得进入 blocked。"""
        attempts = int(batch.get("attempts", 0) or 0)
        blocked = not post_commit and attempts >= MAX_PUBLISH_ATTEMPTS
        status = BATCH_BLOCKED if blocked else BATCH_RETRY_WAIT
        delay = (
            0
            if blocked
            else PUBLISH_RETRY_BACKOFF_SECONDS[
                min(max(attempts, 1), len(PUBLISH_RETRY_BACKOFF_SECONDS)) - 1
            ]
        )
        try:
            with self._engine.begin() as conn:
                updated = conn.execute(
                    update(human_publish_batches)
                    .where(
                        human_publish_batches.c.id == batch["id"],
                        human_publish_batches.c.lease_token == batch["lease_token"],
                        human_publish_batches.c.status == BATCH_RUNNING,
                        self._lease_is_live(human_publish_batches.c.lease_until),
                    )
                    .values(
                        status=status,
                        error=_sanitize_error(exc),
                        next_run_at=None if blocked else self._now_plus(delay),
                        lease_owner="",
                        lease_token="",
                        lease_until=None,
                        finished_at=self._now() if blocked else None,
                        updated_at=self._now(),
                    )
                )
        except Exception as e:
            raise StorageUnavailableError(f"批次失败落库失败: {e}") from e
        if updated.rowcount != 1:
            raise HumanLeaseLost(f"发布批次 {batch['id']} 所有权已丢失")

    def settle_batch(
        self, batch: dict, *, generation_id: str, results: list[dict]
    ) -> list[dict]:
        """发布/下架结算（同事务）：批次 completed + items 终态 + 候选 CAS 终态。

        results 元素：candidate_id/status/filename/detail；发布成功项另带
        candidate_revision/lifecycle_revision（CAS 条件）与可选
        replaced_candidate_id；下架成功项 status=retired 且带
        lifecycle_revision（人工下架）或 filename（补偿下架按文件名兜底）。

        CAS 条件更新：候选已漂移 → rowcount=0 → 记入返回的 cas-miss 清单，
        **绝不覆盖**（superseded/rejected 在任何阶段都不被改回 published）。
        旧 journal（results 无 candidate_revision 字段）按「publish_queued +
        本批次」窄条件兼容结算。调用方负责把 cas-miss 接补偿下架（PR-3）。
        """
        op = batch.get("operation") or BATCH_OP_PUBLISH
        cas_misses: list[dict] = []
        try:
            with self._engine.begin() as conn:
                owned = conn.execute(
                    update(human_publish_batches)
                    .where(
                        human_publish_batches.c.id == batch["id"],
                        human_publish_batches.c.lease_token == batch["lease_token"],
                        human_publish_batches.c.status == BATCH_RUNNING,
                        self._lease_is_live(human_publish_batches.c.lease_until),
                    )
                    .values(
                        status=BATCH_COMPLETED,
                        error="",
                        generation_id=generation_id,
                        lease_owner="",
                        lease_token="",
                        lease_until=None,
                        next_run_at=None,
                        finished_at=self._now(),
                        updated_at=self._now(),
                    )
                )
                if owned.rowcount != 1:
                    raise HumanLeaseLost(
                        f"发布批次 {batch['id']} 所有权已丢失，结果作废"
                    )
                for r in results:
                    conn.execute(
                        update(human_publish_items)
                        .where(
                            human_publish_items.c.batch_id == batch["id"],
                            human_publish_items.c.candidate_id == r["candidate_id"],
                        )
                        .values(
                            status=r["status"],
                            filename=r.get("filename", ""),
                            detail=r.get("detail", "")[:500],
                        )
                    )
                    cas_misses.extend(
                        self._settle_result_candidate(
                            conn, batch, op, r, generation_id
                        )
                    )
        except HumanLeaseLost:
            raise
        except Exception as e:
            raise StorageUnavailableError(f"发布批次结算失败: {e}") from e
        return cas_misses

    def _settle_result_candidate(
        self, conn, batch: dict, op: str, r: dict, generation_id: str
    ) -> list[dict]:
        """单条结果的候选终态（CAS）；返回 cas-miss 清单。"""
        from app.observability.metrics import record_human_lifecycle_inconsistency

        cid = int(r["candidate_id"])
        misses: list[dict] = []
        now = self._now()

        def _miss(row_candidate: int, kind: str, detail: str = "") -> None:
            misses.append(
                {
                    "candidate_id": int(row_candidate),
                    "reason": "settle_cas_miss",
                    "kind": kind,
                    "detail": detail,
                    "filename": r.get("filename", ""),
                }
            )
            record_human_lifecycle_inconsistency("settle_cas_miss")

        if op == BATCH_OP_RETIRE:
            if r["status"] != "retired":
                # 下架无效项（document_missing / candidate_{status}）：候选已非
                # published 时不覆盖；仍是 published 时终结为 retired（人工处理）
                if r.get("candidate_status") == CAND_RETIRED:
                    conn.execute(
                        update(human_knowledge_candidates)
                        .where(
                            human_knowledge_candidates.c.id == cid,
                            human_knowledge_candidates.c.status == CAND_PUBLISHED,
                        )
                        .values(
                            status=CAND_RETIRED,
                            retired_at=now,
                            retire_reason=r.get("detail", "")[:255] or "retire_invalid",
                            updated_at=now,
                        )
                    )
                return misses
            if r.get("lifecycle_revision") is not None:
                updated = conn.execute(
                    update(human_knowledge_candidates)
                    .where(
                        human_knowledge_candidates.c.id == cid,
                        human_knowledge_candidates.c.status == CAND_PUBLISHED,
                        human_knowledge_candidates.c.lifecycle_revision
                        == int(r["lifecycle_revision"]),
                    )
                    .values(
                        status=CAND_RETIRED,
                        retired_at=now,
                        retire_reason=batch.get("reason", "")[:255],
                        lifecycle_revision=human_knowledge_candidates.c.lifecycle_revision
                        + 1,
                        updated_at=now,
                    )
                )
            else:
                # 补偿下架：候选行可能已被并发改写（superseded），按文件名兜底；
                # 仍不匹配则保持行现状（绝不覆盖非 published 状态）。
                updated = conn.execute(
                    update(human_knowledge_candidates)
                    .where(
                        human_knowledge_candidates.c.id == cid,
                        human_knowledge_candidates.c.status == CAND_PUBLISHED,
                        human_knowledge_candidates.c.published_filename
                        == r.get("filename", ""),
                    )
                    .values(
                        status=CAND_RETIRED,
                        retired_at=now,
                        retire_reason=batch.get("reason", "")[:255],
                        lifecycle_revision=human_knowledge_candidates.c.lifecycle_revision
                        + 1,
                        updated_at=now,
                    )
                )
            if updated.rowcount != 1:
                _miss(cid, "retire")
            return misses

        # ---- publish ----
        if r["status"] == ITEM_PUBLISHED:
            common = {
                "status": CAND_PUBLISHED,
                "published_filename": r.get("filename", ""),
                "updated_at": now,
            }
            legacy = "candidate_revision" not in r
            if legacy:
                updated = conn.execute(
                    update(human_knowledge_candidates)
                    .where(
                        human_knowledge_candidates.c.id == cid,
                        human_knowledge_candidates.c.status == CAND_PUBLISH_QUEUED,
                        human_knowledge_candidates.c.publish_batch_id == batch["id"],
                    )
                    .values(
                        published_at=now,
                        published_generation=generation_id,
                        lifecycle_revision=human_knowledge_candidates.c.lifecycle_revision
                        + 1,
                        **common,
                    )
                )
            else:
                updated = conn.execute(
                    update(human_knowledge_candidates)
                    .where(
                        human_knowledge_candidates.c.id == cid,
                        human_knowledge_candidates.c.status == CAND_PUBLISH_QUEUED,
                        human_knowledge_candidates.c.publish_batch_id == batch["id"],
                        human_knowledge_candidates.c.revision
                        == int(r["candidate_revision"]),
                        human_knowledge_candidates.c.lifecycle_revision
                        == int(r["lifecycle_revision"]),
                    )
                    .values(
                        published_at=now,
                        published_generation=generation_id,
                        lifecycle_revision=human_knowledge_candidates.c.lifecycle_revision
                        + 1,
                        **common,
                    )
                )
            if updated.rowcount != 1:
                _miss(cid, "publish")
        else:
            # 拒绝/失败：只收敛仍属于本批次的 publish_queued 候选；
            # 已漂移（superseded/rejected）的候选绝不覆盖。
            target = r.get("candidate_status") or (
                CAND_REJECTED
                if r["status"] in (ITEM_REJECTED, ITEM_FAILED)
                else r["status"]
            )
            conn.execute(
                update(human_knowledge_candidates)
                .where(
                    human_knowledge_candidates.c.id == cid,
                    human_knowledge_candidates.c.status == CAND_PUBLISH_QUEUED,
                    human_knowledge_candidates.c.publish_batch_id == batch["id"],
                )
                .values(status=target, updated_at=now)
            )
        replaced = r.get("replaced_candidate_id")
        if replaced:
            rep = conn.execute(
                update(human_knowledge_candidates)
                .where(
                    human_knowledge_candidates.c.published_filename
                    == r.get("replaced_filename", ""),
                    human_knowledge_candidates.c.status == CAND_PUBLISHED,
                )
                .values(
                    status=CAND_SUPERSEDED,
                    replaced_by_candidate_id=cid,
                    updated_at=now,
                )
            )
            if rep.rowcount != 1:
                _miss(int(replaced), "replacement")
        return misses

    def fence_keys_for_candidates(self, candidate_ids: list[int]) -> list[str]:
        """批内候选 → 去重后的会话栅栏 key（字典序；会话行缺失则跳过）。"""
        from app.evolution.fence import conversation_fence_key

        ids = [int(c) for c in candidate_ids if c]
        if not ids:
            return []
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    select(
                        human_conversations.c.source,
                        human_conversations.c.external_conversation_id,
                    )
                    .select_from(human_knowledge_candidates)
                    .outerjoin(
                        human_conversations,
                        human_conversations.c.id
                        == human_knowledge_candidates.c.conversation_id,
                    )
                    .where(human_knowledge_candidates.c.id.in_(ids))
                ).all()
        except Exception as e:
            raise StorageUnavailableError(f"会话栅栏 key 查询失败: {e}") from e
        keys = {
            conversation_fence_key(str(src), str(ext))
            for src, ext in rows
            if src and ext
        }
        return sorted(keys)

    def conversation_has_higher_version(
        self, source: str, external_conversation_id: str, source_version: int
    ) -> bool:
        """该逻辑会话是否已接入更高 source_version（补偿下架检测用）。"""
        try:
            with self._engine.connect() as conn:
                latest = conn.execute(
                    select(func.max(human_conversations.c.source_version)).where(
                        human_conversations.c.source == source,
                        human_conversations.c.external_conversation_id
                        == external_conversation_id,
                    )
                ).scalar()
        except Exception:  # noqa: BLE001 - 检测失败按无更高版本（观测兜底）
            return False
        return int(latest or 0) > int(source_version)

    def revalidate_batch_items(self, batch_id: int) -> list[dict]:
        """激活前复核（发布 Worker 在 alias 动手前调用）：逐项条件查询。

        - 候选仍 publish_queued 且属于本批次；
        - 候选 revision 与 item 审批快照一致；
        - item 审批快照 digest 复核一致（item 行未被篡改）；
        - 会话无更高 source_version（会话行已被保留期清理 → 视为无更高版本）。

        返回漂移清单（空 = 通过）：发布侧任何漂移 → fail_batch(retry_wait)，
        journal 停在 INDEX_BUILT，下次领取走既有 rollback 后剔除漂移项重发。
        """
        try:
            with self._engine.connect() as conn:
                batch = conn.execute(
                    select(human_publish_batches).where(
                        human_publish_batches.c.id == batch_id
                    )
                ).mappings().first()
                if batch is None:
                    return [{"candidate_id": 0, "reason": "batch_missing"}]
                items = conn.execute(
                    select(human_publish_items).where(
                        human_publish_items.c.batch_id == batch_id
                    )
                ).mappings().all()
                drift: list[dict] = []
                for item in items:
                    cid = int(item["candidate_id"])
                    cand = conn.execute(
                        select(human_knowledge_candidates).where(
                            human_knowledge_candidates.c.id == cid
                        )
                    ).mappings().first()
                    if cand is None:
                        drift.append({"candidate_id": cid, "reason": "candidate_missing"})
                        continue
                    if (
                        cand["status"] != CAND_PUBLISH_QUEUED
                        or int(cand["publish_batch_id"] or 0) != int(batch_id)
                    ):
                        drift.append(
                            {
                                "candidate_id": cid,
                                "reason": f"candidate_{cand['status']}",
                            }
                        )
                        continue
                    if int(cand["revision"] or 0) != (
                        -1 if item["candidate_revision"] is None
                        else int(item["candidate_revision"])
                    ):
                        drift.append({"candidate_id": cid, "reason": "revision_drift"})
                        continue
                    snap = {
                        "candidate_id": cid,
                        "candidate_revision": int(item["candidate_revision"] or 0),
                        "source_version": int(item["source_version"] or 0),
                        "question": item["question"],
                        "answer": item["answer"],
                        "value_score": item["value_score"],
                        "classification": item["classification"],
                        "dedup_target_path": item["dedup_target_path"],
                        "approved_by": item["approved_by"],
                        "approved_at": (
                            item["approved_at"].isoformat(sep=" ", timespec="seconds")
                            if getattr(item["approved_at"], "isoformat", None)
                            else str(item["approved_at"] or "")
                        ),
                    }
                    if approval_digest(snap) != (item["approval_digest"] or ""):
                        drift.append({"candidate_id": cid, "reason": "digest_mismatch"})
                        continue
                    conv = conn.execute(
                        select(human_conversations).where(
                            human_conversations.c.id
                            == int(cand["conversation_id"] or 0)
                        )
                    ).mappings().first()
                    if conv is not None:
                        latest = conn.execute(
                            select(func.max(human_conversations.c.source_version)).where(
                                human_conversations.c.source == conv["source"],
                                human_conversations.c.external_conversation_id
                                == conv["external_conversation_id"],
                            )
                        ).scalar()
                        if int(latest or 0) > int(conv["source_version"]):
                            drift.append(
                                {"candidate_id": cid, "reason": "source_version_drift"}
                            )
                return drift
        except Exception as e:
            raise StorageUnavailableError(f"激活前复核失败: {e}") from e

    def create_retire_batch(
        self,
        items: list[dict],
        *,
        reason: str,
        requested_by: str,
    ) -> tuple[dict, list[dict]]:
        """人工下架批次：items = [{"candidate_id", "expected_lifecycle_revision"}]。

        校验：候选 status=published 且 lifecycle_revision 匹配（不设中间态；
        双下架由结算 CAS 兜底为 lifecycle cas-miss）；item 快照冻结
        published_filename。任一冲突整体 409。
        """
        if not items:
            raise HumanKnowledgeConflict("下架批次不能为空")
        if len(items) > 100:
            raise HumanKnowledgeConflict("单个下架批次最多 100 条候选")
        candidate_ids = [int(it["candidate_id"]) for it in items]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise HumanKnowledgeConflict("下架批次包含重复 candidate_id")
        try:
            with self._engine.begin() as conn:
                rows = (
                    conn.execute(
                        select(human_knowledge_candidates)
                        .where(human_knowledge_candidates.c.id.in_(candidate_ids))
                        .with_for_update()
                    )
                    .mappings()
                    .all()
                )
                by_id = {int(r["id"]): dict(r) for r in rows}
                for it in items:
                    cid = int(it["candidate_id"])
                    row = by_id.get(cid)
                    if row is None:
                        raise HumanKnowledgeConflict(f"候选 {cid} 不存在")
                    if row["status"] != CAND_PUBLISHED:
                        raise HumanKnowledgeConflict(
                            f"候选 {cid} 状态为 {row['status']}，仅已发布候选可下架"
                        )
                    expected = int(it.get("expected_lifecycle_revision", -1))
                    if expected != int(row["lifecycle_revision"] or 0):
                        raise HumanKnowledgeConflict(
                            f"候选 {cid} lifecycle_revision 不匹配"
                            f"（当前 {row['lifecycle_revision']}）",
                        )
                    if not row["published_filename"]:
                        raise HumanKnowledgeConflict(
                            f"候选 {cid} 缺少已发布文件名，无法下架"
                        )
                batch = conn.execute(
                    human_publish_batches.insert().values(
                        requested_by=requested_by,
                        item_count=len(items),
                        operation=BATCH_OP_RETIRE,
                        reason=str(reason or "")[:255],
                        error="",
                    )
                )
                batch_id = batch.inserted_primary_key[0]
                for it in items:
                    cid = int(it["candidate_id"])
                    row = by_id[cid]
                    conn.execute(
                        human_publish_items.insert().values(
                            batch_id=batch_id,
                            candidate_id=cid,
                            candidate_revision=int(row["revision"] or 0),
                            question=row["question"],
                            approved_by=requested_by,
                            approval_digest="retire",
                        )
                    )
                batch_row = conn.execute(
                    select(human_publish_batches).where(
                        human_publish_batches.c.id == batch_id
                    )
                ).mappings().first()
                item_rows = conn.execute(
                    select(human_publish_items).where(
                        human_publish_items.c.batch_id == batch_id
                    )
                ).mappings().all()
                return dict(batch_row), [dict(r) for r in item_rows]
        except HumanKnowledgeConflict:
            raise
        except Exception as e:
            raise StorageUnavailableError(f"下架批次创建失败: {e}") from e

    def enqueue_retire_batch(
        self,
        items: list[dict],
        *,
        reason: str,
        requested_by: str = "system:compensation",
    ) -> int | None:
        """补偿下架入队（系统触发；幂等：已有同候选未完结下架批次则跳过）。

        items = [{"candidate_id", "filename"?}]；filename 缺省读行上
        published_filename。候选行无已发布文件名（如发布中被并发 superseded
        且结果未回写）→ 跳过并记不一致指标，由调用方日志留痕。
        """
        resolved: list[dict] = []
        with self._engine.connect() as conn:
            open_batch_ids = (
                select(human_publish_batches.c.id).where(
                    human_publish_batches.c.operation == BATCH_OP_RETIRE,
                    human_publish_batches.c.status.in_(
                        (BATCH_QUEUED, BATCH_RUNNING, BATCH_RETRY_WAIT)
                    ),
                )
            )
            protected = {
                int(r[0])
                for r in conn.execute(
                    select(human_publish_items.c.candidate_id).where(
                        human_publish_items.c.batch_id.in_(open_batch_ids)
                    )
                ).all()
            }
            for it in items:
                cid = int(it["candidate_id"])
                if cid in protected:
                    continue
                row = conn.execute(
                    select(
                        human_knowledge_candidates.c.published_filename,
                        human_knowledge_candidates.c.revision,
                    ).where(human_knowledge_candidates.c.id == cid)
                ).first()
                if row is None:
                    continue
                filename = str(it.get("filename") or row[0] or "")
                if not filename:
                    record_missing = True
                else:
                    record_missing = False
                resolved.append(
                    {
                        "candidate_id": cid,
                        "filename": filename,
                        "revision": int(row[1] or 0),
                        "no_filename": record_missing,
                    }
                )
        resolvable = [it for it in resolved if not it["no_filename"]]
        if not resolvable:
            from app.observability.metrics import record_human_lifecycle_inconsistency

            for it in resolved:
                if it["no_filename"]:
                    record_human_lifecycle_inconsistency("document_missing")
            return None
        try:
            with self._engine.begin() as conn:
                batch = conn.execute(
                    human_publish_batches.insert().values(
                        requested_by=requested_by,
                        item_count=len(resolvable),
                        operation=BATCH_OP_RETIRE,
                        reason=str(reason or "")[:255],
                        error="",
                    )
                )
                batch_id = batch.inserted_primary_key[0]
                for it in resolvable:
                    conn.execute(
                        human_publish_items.insert().values(
                            batch_id=batch_id,
                            candidate_id=it["candidate_id"],
                            candidate_revision=it["revision"],
                            # 冻结文件名：候选行可能在批次执行前继续漂移
                            # （published_filename 清空），下架以入队时为准。
                            filename=it["filename"],
                            approved_by=requested_by,
                            approval_digest="compensation",
                        )
                    )
                return batch_id
        except Exception as e:
            raise StorageUnavailableError(f"补偿下架入队失败: {e}") from e

    def get_batch(self, batch_id: int) -> dict | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    select(human_publish_batches).where(
                        human_publish_batches.c.id == batch_id
                    )
                )
                .mappings()
                .first()
            )
            if row is None:
                return None
            batch = dict(row)
            batch["items"] = [
                dict(r)
                for r in conn.execute(
                    select(human_publish_items)
                    .where(human_publish_items.c.batch_id == batch_id)
                    .order_by(human_publish_items.c.id)
                )
                .mappings()
                .all()
            ]
        return batch

    # ============================================================
    # 生命周期协调 / 审核详情（010）
    # ============================================================
    def backfill_candidate_evidence(
        self,
        candidate_id: int,
        *,
        source_snapshot: dict,
        evidence_snapshot: list[dict],
    ) -> bool:
        """证据链回填（010 上线一次性脚本）：legacy 候选 → 双快照 + ok。

        只对 evidence_state != ok 的候选生效；并发安全（条件更新）。
        """
        try:
            with self._engine.begin() as conn:
                updated = conn.execute(
                    update(human_knowledge_candidates)
                    .where(
                        human_knowledge_candidates.c.id == candidate_id,
                        human_knowledge_candidates.c.evidence_state != EVIDENCE_OK,
                    )
                    .values(
                        source_snapshot_json=_json_or_none(source_snapshot),
                        evidence_snapshot_json=_json_or_none(evidence_snapshot),
                        evidence_state=EVIDENCE_OK,
                        updated_at=self._now(),
                    )
                )
        except Exception as e:
            raise StorageUnavailableError(f"证据链回填失败: {e}") from e
        return updated.rowcount == 1

    def mark_candidate_superseded_by(
        self, candidate_id: int, replaced_by: int
    ) -> bool:
        """替换结算（协调器路由 human 正本）：published → superseded。"""
        try:
            with self._engine.begin() as conn:
                updated = conn.execute(
                    update(human_knowledge_candidates)
                    .where(
                        human_knowledge_candidates.c.id == candidate_id,
                        human_knowledge_candidates.c.status == CAND_PUBLISHED,
                    )
                    .values(
                        status=CAND_SUPERSEDED,
                        replaced_by_candidate_id=replaced_by,
                        updated_at=self._now(),
                    )
                )
        except Exception as e:
            raise StorageUnavailableError(f"替换结算失败: {e}") from e
        return updated.rowcount == 1

    def reset_candidate_revalidation_failed(self, candidate_id: int, reason: str) -> bool:
        """重接地隔离（协调器路由 human 正本）：published → pending_review 重审。"""
        try:
            with self._engine.begin() as conn:
                updated = conn.execute(
                    update(human_knowledge_candidates)
                    .where(
                        human_knowledge_candidates.c.id == candidate_id,
                        human_knowledge_candidates.c.status == CAND_PUBLISHED,
                    )
                    .values(
                        status=CAND_PENDING_REVIEW,
                        score_stale=1,
                        reject_reason=str(reason or "")[:64],
                        published_filename="",
                        publish_batch_id=None,
                        updated_at=self._now(),
                    )
                )
                if updated.rowcount == 1:
                    conn.execute(
                        human_evaluation_jobs.insert().values(
                            job_type=JOB_TYPE_CANDIDATE,
                            candidate_id=candidate_id,
                            status=JOB_QUEUED,
                            error="",
                        )
                    )
        except Exception as e:
            raise StorageUnavailableError(f"重接地隔离结算失败: {e}") from e
        return updated.rowcount == 1

    def latest_blocked_job_for_candidate(self, candidate_id: int) -> dict | None:
        """候选最近一条 blocked 评审任务（候选级人工重试入口）。"""
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    select(human_evaluation_jobs)
                    .where(
                        human_evaluation_jobs.c.candidate_id == candidate_id,
                        human_evaluation_jobs.c.job_type == JOB_TYPE_CANDIDATE,
                        human_evaluation_jobs.c.status == JOB_BLOCKED,
                    )
                    .order_by(human_evaluation_jobs.c.id.desc())
                    .limit(1)
                )
                .mappings()
                .first()
            )
        return dict(row) if row is not None else None

    def get_candidate_detail(self, candidate_id: int) -> dict | None:
        """候选详情：行 + 最近 5 条评审任务（审核台详情/证据展开用）。"""
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    select(human_knowledge_candidates).where(
                        human_knowledge_candidates.c.id == candidate_id
                    )
                )
                .mappings()
                .first()
            )
            if row is None:
                return None
            detail = dict(row)
            detail["recent_evaluation_jobs"] = [
                dict(j)
                for j in conn.execute(
                    select(
                        human_evaluation_jobs.c.id,
                        human_evaluation_jobs.c.status,
                        human_evaluation_jobs.c.attempts,
                        human_evaluation_jobs.c.error,
                        human_evaluation_jobs.c.created_at,
                    )
                    .where(
                        or_(
                            human_evaluation_jobs.c.candidate_id == candidate_id,
                            and_(
                                human_evaluation_jobs.c.job_type
                                == JOB_TYPE_CONVERSATION,
                                human_evaluation_jobs.c.conversation_id
                                == detail["conversation_id"],
                            ),
                        )
                    )
                    .order_by(human_evaluation_jobs.c.id.desc())
                    .limit(5)
                )
                .mappings()
                .all()
            ]
        return detail

    def next_evaluation_delay(self, *, poll_seconds: float = 30.0) -> float:
        """常驻评审 Worker 的空闲睡眠时长：到期任务 → 0；否则到最近
        next_run_at 的秒数；无排队任务 → poll_seconds。"""
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    select(
                        human_evaluation_jobs.c.next_run_at,
                        human_evaluation_jobs.c.status,
                    ).where(
                        human_evaluation_jobs.c.status.in_((JOB_QUEUED, JOB_RETRY_WAIT))
                    )
                ).all()
        except Exception:  # noqa: BLE001 - metrics must not break workers
            return poll_seconds
        now = _local_now()
        due = any(
            next_run is None or next_run <= now
            for next_run, _status in rows
        )
        if due:
            return 0.0
        future = [next_run for next_run, _s in rows if next_run is not None]
        if not future:
            return poll_seconds
        return max((min(future) - now).total_seconds(), 0.0) or poll_seconds

    # ============================================================
    # 观测（队列深度/最老任务年龄/blocked）
    # ============================================================
    def stats(self) -> dict:
        try:
            with self._engine.connect() as conn:
                rows = (
                    conn.execute(
                        select(
                            human_evaluation_jobs.c.status,
                            func.count().label("n"),
                            func.min(human_evaluation_jobs.c.created_at).label(
                                "oldest"
                            ),
                        ).group_by(human_evaluation_jobs.c.status)
                    )
                    .mappings()
                    .all()
                )
        except Exception:  # noqa: BLE001 - metrics must not break workers
            return {}
        now = _local_now() if not self._server_time else None
        by_status: dict[str, int] = {}
        oldest_age = 0.0
        for r in rows:
            by_status[r["status"]] = int(r["n"])
            if (
                r["status"] in (JOB_QUEUED, JOB_RETRY_WAIT, JOB_RUNNING)
                and r["oldest"] is not None
            ):
                created = r["oldest"]
                base = now if now is not None else _local_now()
                age = max((base - created).total_seconds(), 0.0)
                oldest_age = max(oldest_age, age)
        return {
            "by_status": by_status,
            "oldest_age_seconds": oldest_age,
            "blocked": by_status.get(JOB_BLOCKED, 0),
            "queue_depth": by_status.get(JOB_QUEUED, 0)
            + by_status.get(JOB_RETRY_WAIT, 0),
        }
