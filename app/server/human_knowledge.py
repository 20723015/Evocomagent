"""人工客服知识自进化 API（迁移 008/010）：会话批量接入 + 人工审核/批量发布/下架。

端点：
    POST /v1/human-conversations/batch            外部客服系统批量推送已结束会话
                                                  （scope: human_chat_ingest）
    GET    /v1/human-knowledge/candidates         候选列表（ops）
    GET    /v1/human-knowledge/candidates/{id}    候选详情（证据/去重快照/评审任务）
    POST   /v1/human-knowledge/candidates/{id}/edit   编辑（ops；revision+1→重评）
    POST   /v1/human-knowledge/candidates/{id}/reject 拒绝（ops）
    POST   /v1/human-knowledge/candidates/{id}/retry-evaluation blocked 重评（ops）
    POST   /v1/human-knowledge/candidates/{id}/retire  下架已发布候选（ops）
    POST   /v1/human-knowledge/evaluation-jobs/{id}/retry  blocked 任务人工重试（ops）
    POST   /v1/human-knowledge/publish-batches    批量批准（ops；SELF_EVOLVE 门禁）
    GET    /v1/human-knowledge/publish-batches/{id}  批次状态（ops）

开关门禁：human_qa_evolution_enabled 门禁整条人工链路的**变更**入口
（接入/编辑/重试/发布/下架，503）；查询只读保留。self_evolve_enabled 保持为
最终发布安全开关（发布 API 两者同时检查）。

接入契约（服务端强制）：
- 单批 ≤100 会话，每会话 ≤500 条消息，单条 ≤8000 字；
- 至少一条 customer + 一条 human_agent 消息；message_id 缺省补齐后查重；
- PII 在写库前脱敏（sanitize_text），原文不落库、不进日志；
- (source, external_conversation_id, source_version) 唯一：同摘要幂等返回
  原记录，同版本不同内容 409，更高版本作为修订（旧未发布候选 superseded）。
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from app.config.settings import settings
from app.evolution.human_store import (
    HumanKnowledgeConflict,
    conversation_digest,
)
from app.evolution.sanitizer import sanitize_text
from app.security.principal import SCOPE_HUMAN_INGEST, SCOPE_OPS, authorize_scopes
from app.server.schema import (
    HumanCandidateRetireRequest,
    HumanConversationBatch,
    HumanConversationItem,
    HumanKnowledgeEditRequest,
    HumanPublishBatchRequest,
)

router = APIRouter(tags=["human-knowledge"])

MAX_BATCH_CONVERSATIONS = 100
MAX_MESSAGES = 500
MAX_MESSAGE_CHARS = 8000
_VALID_ACTORS = ("customer", "human_agent", "bot", "system")


def _storage_unavailable(exc: Exception) -> HTTPException:
    from app.stores.base import StorageUnavailableError

    if isinstance(exc, StorageUnavailableError):
        return HTTPException(status_code=503, detail="存储暂不可用，请稍后重试")
    raise exc


def _require_human_enabled() -> None:
    """人工链路总开关（D1）：关闭 → 503。仅作用于变更入口。"""
    if not settings.human_qa_evolution_enabled:
        raise HTTPException(
            status_code=503,
            detail="人工知识链路被开关禁止（HUMAN_QA_EVOLUTION_ENABLED=false）",
        )


def _store(request: Request):
    components = getattr(request.app.state, "components", None)
    store = getattr(components, "human_knowledge_store", None) if components else None
    if store is None:
        raise HTTPException(status_code=503, detail="人工知识链路未启用（需要 MySQL）")
    return store


def _parse_ts(value: str, field: str) -> datetime:
    """ISO 8601 → UTC-naive（D4 时区口径：aware 输入统一转 UTC 后去 tzinfo）。"""
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=422,
            detail=f"{field} 不是合法时间（ISO 8601）",
        ) from None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _validate_conversation(item: HumanConversationItem) -> dict:
    """服务端强制校验 + PII 预脱敏（原文不落库）。"""
    if not item.source.strip() or not item.external_conversation_id.strip():
        raise HTTPException(
            status_code=422, detail="source / external_conversation_id 必填"
        )
    if len(item.messages) == 0:
        raise HTTPException(status_code=422, detail="messages 不能为空")
    if len(item.messages) > MAX_MESSAGES:
        raise HTTPException(
            status_code=422,
            detail=f"单会话消息数超限（{len(item.messages)} > {MAX_MESSAGES}）",
        )
    has_customer = has_agent = False
    sanitized: list[dict] = []
    seen_message_ids: set[str] = set()
    for idx, msg in enumerate(item.messages):
        if msg.actor_type not in _VALID_ACTORS:
            raise HTTPException(
                status_code=422,
                detail=f"第 {idx} 条消息 actor_type 非法: {msg.actor_type}",
            )
        content = msg.content
        if len(content) > MAX_MESSAGE_CHARS:
            raise HTTPException(
                status_code=422,
                detail=f"第 {idx} 条消息超过 {MAX_MESSAGE_CHARS} 字上限",
            )
        if not content.strip():
            raise HTTPException(status_code=422, detail=f"第 {idx} 条消息内容为空")
        if msg.actor_type == "customer":
            has_customer = True
        if msg.actor_type == "human_agent":
            has_agent = True
        # PII 在写库前脱敏：原文不落库、不进日志
        message_id = msg.message_id or f"m{idx:04d}"
        if message_id in seen_message_ids:
            raise HTTPException(
                status_code=422,
                detail=f"第 {idx} 条消息 message_id 重复: {message_id}",
            )
        seen_message_ids.add(message_id)
        sanitized.append(
            {
                "message_id": message_id,
                "actor_type": msg.actor_type,
                "content": sanitize_text(content),
                "sent_at": msg.sent_at or "",
            }
        )
    if not has_customer or not has_agent:
        raise HTTPException(
            status_code=422,
            detail="会话必须至少包含一条 customer 消息和一条 human_agent 消息",
        )
    if not item.ended_at:
        raise HTTPException(status_code=422, detail="ended_at 必填（仅接收已结束会话）")
    started_at = _parse_ts(item.started_at, "started_at") if item.started_at else None
    ended_at = _parse_ts(item.ended_at, "ended_at")
    if started_at is not None and started_at > ended_at:
        raise HTTPException(status_code=422, detail="started_at 不得晚于 ended_at")
    return {
        "source": item.source.strip()[:64],
        "external_conversation_id": item.external_conversation_id.strip()[:128],
        "source_version": int(item.source_version),
        "agent_id": item.agent_id.strip()[:64],
        "started_at": started_at,
        "ended_at": ended_at,
        "messages": sanitized,
    }


def _acquire_ingest_fence(store, validated: list[dict]):
    """接入侧会话栅栏（D1）：与发布共用同名锁，收敛接入/发布竞态窗口。

    超时（human_fence_ingest_timeout_seconds，缺省 3s）→ 503 让外部系统重试；
    sqlite（测试）方言 no-op。
    """
    from app.evolution.fence import (
        ConversationFence,
        FenceTimeout,
        conversation_fence_key,
    )

    keys = sorted(
        {
            conversation_fence_key(v["source"], v["external_conversation_id"])
            for v in validated
            if v.get("source") and v.get("external_conversation_id")
        }
    )
    if not keys:
        return None
    fence = ConversationFence(
        store.engine,
        timeout_seconds=settings.human_fence_ingest_timeout_seconds,
    )
    try:
        fence.acquire(keys)
    except FenceTimeout:
        raise HTTPException(
            status_code=503,
            detail="会话正在接入或发布中（会话栅栏占用），请稍后重试",
        ) from None
    return fence


@router.post("/v1/human-conversations/batch")
async def ingest_human_conversations(request: Request, body: HumanConversationBatch):
    """外部客服系统批量推送已结束的完整会话（幂等/冲突/修订见模块 docstring）。

    写库前对批内全部 (source, external_conversation_id) 取会话栅栏——
    与发布 Worker 互斥，避免「旧知识上线后再补偿下架」的竞态窗口。
    """
    authorize_scopes(request, SCOPE_HUMAN_INGEST)
    _require_human_enabled()
    if len(body.conversations) == 0:
        raise HTTPException(status_code=422, detail="conversations 不能为空")
    if len(body.conversations) > MAX_BATCH_CONVERSATIONS:
        raise HTTPException(
            status_code=422,
            detail=f"单批会话数超限（{len(body.conversations)} > "
            f"{MAX_BATCH_CONVERSATIONS}）",
        )
    validated = [_validate_conversation(item) for item in body.conversations]
    store = _store(request)
    fence = _acquire_ingest_fence(store, validated)
    try:
        try:
            stored = store.ingest_conversations(validated)
        except HumanKnowledgeConflict as e:
            raise HTTPException(status_code=409, detail=e.detail) from e
        except Exception as exc:
            raise _storage_unavailable(exc) from exc
    finally:
        if fence is not None:
            fence.release()
    results = []
    for item, (record, outcome) in zip(body.conversations, stored, strict=True):
        results.append(
            {
                "source": item.source,
                "external_conversation_id": item.external_conversation_id,
                "source_version": int(item.source_version),
                "conversation_id": record["id"],
                "result": outcome,  # created | duplicate
            }
        )
    return {"results": results}


# ============================================================
# 人工审核与批量发布（ops 面）
# ============================================================
@router.get("/v1/human-knowledge/candidates")
async def list_candidates(
    request: Request, status: str = "", limit: int = 50, offset: int = 0
):
    authorize_scopes(request, SCOPE_OPS)
    try:
        rows, total = _store(request).list_candidates(
            status=status, limit=limit, offset=offset
        )
    except Exception as exc:
        raise _storage_unavailable(exc) from exc
    return {
        "candidates": [_candidate_dto(r) for r in rows],
        "total": total,
        "limit": int(limit),
        "offset": int(offset),
    }


@router.post("/v1/human-knowledge/candidates/{candidate_id}/edit")
async def edit_candidate(
    candidate_id: int, request: Request, body: HumanKnowledgeEditRequest
):
    """编辑规范问题/标准答案：revision+1 + 旧评分过期 + 重评入队（同事务）。"""
    authorize_scopes(request, SCOPE_OPS)
    _require_human_enabled()
    try:
        row, current = _store(request).edit_candidate(
            candidate_id,
            body.question.strip(),
            body.answer.strip(),
            int(body.expected_revision),
        )
    except HumanKnowledgeConflict as exc:
        raise HTTPException(status_code=422, detail=exc.detail) from exc
    except Exception as exc:
        raise _storage_unavailable(exc) from exc
    if row is None:
        raise HTTPException(
            status_code=409 if current >= 0 else 404,
            detail=(
                f"候选已被其他审核员修改（当前 revision {current}）"
                if current >= 0
                else "候选不存在"
            ),
        )
    return {
        "candidate_id": candidate_id,
        "status": row["status"],
        "revision": int(row["revision"]),
        "score_stale": True,
    }


@router.post("/v1/human-knowledge/candidates/{candidate_id}/reject")
async def reject_candidate(candidate_id: int, request: Request, body: dict):
    authorize_scopes(request, SCOPE_OPS)
    _require_human_enabled()
    reason = str((body or {}).get("reason", "manual_review_reject"))
    try:
        row = _store(request).reject_candidate(candidate_id, reason)
    except Exception as exc:
        raise _storage_unavailable(exc) from exc
    if row is None:
        raise HTTPException(status_code=404, detail="候选不存在或不在待审核状态")
    return {"candidate_id": candidate_id, "status": row["status"]}


@router.get("/v1/human-knowledge/candidates/{candidate_id}")
async def get_candidate_detail(candidate_id: int, request: Request):
    """候选详情：基础 DTO + 证据/去重快照 + 最近 5 条评审任务 + 生命周期字段。"""
    authorize_scopes(request, SCOPE_OPS)
    try:
        row = _store(request).get_candidate_detail(candidate_id)
    except Exception as exc:
        raise _storage_unavailable(exc) from exc
    if row is None:
        raise HTTPException(status_code=404, detail="候选不存在")
    return _candidate_detail_dto(row)


@router.post("/v1/human-knowledge/candidates/{candidate_id}/retry-evaluation")
async def retry_candidate_evaluation(candidate_id: int, request: Request):
    """候选级 blocked 重评：定位该候选最新 blocked 任务 → 清计数重新排队。"""
    authorize_scopes(request, SCOPE_OPS)
    _require_human_enabled()
    store = _store(request)
    try:
        job = store.latest_blocked_job_for_candidate(candidate_id)
        if job is None:
            raise HTTPException(
                status_code=404, detail="候选不存在或无 blocked 评审任务"
            )
        retried = store.retry_blocked_job(int(job["id"]))
    except HTTPException:
        raise
    except Exception as exc:
        raise _storage_unavailable(exc) from exc
    if retried is None:
        raise HTTPException(status_code=404, detail="任务不存在或未处于 blocked")
    return {"candidate_id": candidate_id, "job_id": int(job["id"]), "status": retried["status"]}


@router.post("/v1/human-knowledge/candidates/{candidate_id}/retire", status_code=202)
async def retire_candidate(candidate_id: int, request: Request, body: HumanCandidateRetireRequest):
    """下架已发布候选：建 retire 批次（Worker 异步执行，lifecycle_revision 乐观锁）。"""
    principal = authorize_scopes(request, SCOPE_OPS)
    _require_human_enabled()
    try:
        batch, items = _store(request).create_retire_batch(
            [
                {
                    "candidate_id": candidate_id,
                    "expected_lifecycle_revision": int(
                        body.expected_lifecycle_revision,
                    ),
                }
            ],
            reason=body.reason.strip(),
            requested_by=principal.sub,
        )
    except HumanKnowledgeConflict as e:
        raise HTTPException(status_code=409, detail=e.detail) from e
    except Exception as exc:
        raise _storage_unavailable(exc) from exc
    return {
        "batch_id": batch["id"],
        "status": batch["status"],
        "operation": batch["operation"],
        "item_count": batch["item_count"],
        "candidate_ids": [it["candidate_id"] for it in items],
    }


@router.post("/v1/human-knowledge/evaluation-jobs/{job_id}/retry")
async def retry_evaluation_job(job_id: int, request: Request):
    """人工重试 blocked 评审任务（清计数重新排队）。"""
    authorize_scopes(request, SCOPE_OPS)
    _require_human_enabled()
    try:
        job = _store(request).retry_blocked_job(job_id)
    except Exception as exc:
        raise _storage_unavailable(exc) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在或未处于 blocked")
    return {"job_id": job_id, "status": job["status"]}


@router.post("/v1/human-knowledge/publish-batches", status_code=202)
async def create_publish_batch(request: Request, body: HumanPublishBatchRequest):
    """批量批准：先整体校验（revision/评分/状态/证据链），任一冲突整体 409。

    同事务创建发布批次（含不可变审批快照）并将候选置 publish_queued，HTTP 202；
    发布由 Worker 异步执行（SELF_EVOLVE_ENABLED 仍控制最终发布）。
    """
    principal = authorize_scopes(request, SCOPE_OPS)
    _require_human_enabled()
    if not settings.self_evolve_enabled:
        raise HTTPException(
            status_code=409,
            detail="最终发布被安全开关禁止（SELF_EVOLVE_ENABLED=false）",
        )
    store = _store(request)
    try:
        batch, items = store.create_publish_batch(
            [
                {"candidate_id": it.candidate_id, "revision": it.revision}
                for it in body.items
            ],
            requested_by=principal.sub,
        )
    except HumanKnowledgeConflict as e:
        raise HTTPException(status_code=409, detail=e.detail) from e
    except Exception as exc:
        raise _storage_unavailable(exc) from exc
    return {
        "batch_id": batch["id"],
        "status": batch["status"],
        "operation": batch["operation"],
        "item_count": batch["item_count"],
        "candidate_ids": [it["candidate_id"] for it in items],
    }


@router.get("/v1/human-knowledge/publish-batches/{batch_id}")
async def get_publish_batch(batch_id: int, request: Request):
    authorize_scopes(request, SCOPE_OPS)
    try:
        batch = _store(request).get_batch(batch_id)
    except Exception as exc:
        raise _storage_unavailable(exc) from exc
    if batch is None:
        raise HTTPException(status_code=404, detail="批次不存在")
    terminal_counts: dict[str, int] = {
        "published": 0,
        "rejected": 0,
        "failed": 0,
        "superseded": 0,
    }
    for it in batch["items"]:
        status = str(it["status"] or "")
        if status in terminal_counts:
            terminal_counts[status] += 1
    return {
        "batch_id": batch["id"],
        "status": batch["status"],
        "operation": batch.get("operation", "publish"),
        "reason": batch.get("reason", ""),
        "item_count": batch["item_count"],
        "generation_id": batch["generation_id"],
        "error": batch["error"],
        "attempts": batch["attempts"],
        "created_at": _iso(batch.get("created_at")),
        "finished_at": _iso(batch.get("finished_at")),
        "terminal_counts": terminal_counts,
        "items": [
            {
                "candidate_id": it["candidate_id"],
                "status": it["status"],
                "filename": it["filename"],
                "detail": it["detail"],
                "approved_by": it.get("approved_by", ""),
                "approval_digest": it.get("approval_digest", ""),
            }
            for it in batch["items"]
        ],
    }


def _candidate_dto(row: dict) -> dict:
    dedup = _json_or(row.get("dedup_snapshot_json"), {})
    return {
        "id": row["id"],
        "conversation_id": row["conversation_id"],
        "status": row["status"],
        "reject_reason": row.get("reject_reason", ""),
        "question": row["question"],
        "answer": row["answer"],
        "evidence_message_ids": _json_or(row.get("evidence_message_ids"), []),
        "value_score": row.get("value_score"),
        "worth_saving": bool(row.get("worth_saving")),
        "value_reason": row.get("value_reason", ""),
        "max_similarity": row.get("max_similarity"),
        "novelty_score": row.get("novelty_score"),
        "composite_score": row.get("composite_score"),
        "classification": row.get("classification", ""),
        "rag_hit_path": row.get("rag_hit_path", ""),
        "dedup_target_path": ((dedup.get("question") or {}) or {}).get("path", ""),
        "evidence_state": row.get("evidence_state", ""),
        "score_stale": bool(int(row.get("score_stale") or 0)),
        "revision": int(row.get("revision") or 0),
        "lifecycle_revision": int(row.get("lifecycle_revision") or 0),
        "eval_model": row.get("eval_model", ""),
        "eval_kb_generation": row.get("eval_kb_generation", ""),
        "published_filename": row.get("published_filename", ""),
        "published_at": _iso(row.get("published_at")),
        "retired_at": _iso(row.get("retired_at")),
        "retire_reason": row.get("retire_reason", ""),
        "replaced_by_candidate_id": row.get("replaced_by_candidate_id"),
        "publish_batch_id": row.get("publish_batch_id"),
        "updated_at": _iso(row.get("updated_at")),
    }


def _candidate_detail_dto(row: dict) -> dict:
    dto = _candidate_dto(row)
    dto.update(
        {
            "source_snapshot": _json_or(row.get("source_snapshot_json"), {}),
            "evidence_snapshot": _json_or(row.get("evidence_snapshot_json"), []),
            "dedup_snapshot": _json_or(row.get("dedup_snapshot_json"), {}),
            "published_generation": row.get("published_generation", ""),
            "recent_evaluation_jobs": [
                {
                    "id": int(j["id"]),
                    "status": j["status"],
                    "attempts": int(j.get("attempts") or 0),
                    "error": j.get("error", ""),
                    "created_at": _iso(j.get("created_at")),
                }
                for j in row.get("recent_evaluation_jobs", [])
            ],
        }
    )
    return dto


def _json_or(raw, default):
    import json

    try:
        return json.loads(raw) if raw else default
    except (TypeError, ValueError):
        return default


def _iso(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    return str(value)


__all__ = ["conversation_digest", "router"]
