"""KB 文档上传 API（v7 端点 + 多实例异步改造的 jobs 管理端点，全部 SCOPE_OPS）。

端点：
    POST   /v1/kb/uploads                              创建断点会话（upload_id 幂等锚点）
    PUT    /v1/kb/uploads/{upload_id}/chunks/{seq}      上传分片（octet-stream；重传幂等）
    GET    /v1/kb/uploads/{upload_id}                   断点状态（received[] 供续传；终态以 SQL 为正本）
    POST   /v1/kb/uploads/{upload_id}/complete          异步入库：202 + job 地址（同步模式兼容 200）
    DELETE /v1/kb/uploads/{upload_id}                   取消上传（委托任务取消；queued 才可取消）
    GET    /v1/kb/documents                             列表（全量，不做本人过滤）
    GET    /v1/kb/documents/{doc_id}                    详情
    DELETE /v1/kb/documents/{doc_id}                    异步下架：202 + job 地址（已 deleted 200）
    GET    /v1/kb/jobs/{job_id}                         任务详情（状态/阶段/进度/尝试/错误）
    GET    /v1/kb/jobs                                  任务分页查询（status/operation 过滤）
    POST   /v1/kb/jobs/{job_id}/retry                   人工重试（仅可重试的 failed/blocked）
    DELETE /v1/kb/jobs/{job_id}                         取消任务（仅 queued/retry_wait；运行中 409）

异步语义（kb_control.kb_async_enabled 启用后）：
    complete/delete 入队即返回 202（Location: 任务地址，Retry-After: 2）；
    客户端轮询 GET /v1/kb/jobs/{job_id} 到终态；已 indexed/deleted 幂等 200；
    永久失败 409 并附任务地址。uploader 语义不变：认证身份优先（auth 关闭时
    回退 body/query uploader），complete/delete 行归属校验（不一致 403）。

service 是同步代码：async 端点一律经 asyncio.to_thread（执行模型 1.7）。
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config.settings import settings
from app.security.identifiers import InvalidIdentifier, validate_identifier
from app.security.principal import SCOPE_OPS, authorize_scopes
from app.server.deps import authenticate_user
from app.stores.base import StorageUnavailableError
from app.stores.kb_write_lock import KbWriteLockBackendError, KbWriteLockError

router = APIRouter(prefix="/v1/kb", tags=["kb-documents"])


def _service(request: Request):
    components = getattr(request.app.state, "components", None)
    if components is None:
        raise HTTPException(status_code=503, detail="组件未就绪")
    svc = getattr(components, "upload_service", None)
    if svc is None:
        raise HTTPException(status_code=503,
                            detail="KB 上传服务未启用（kb_upload_enabled 或 DB/Redis 不可用）")
    return svc


def _job_store(request: Request):
    store = getattr(_service(request), "_jobs", None)
    if store is None:
        raise HTTPException(status_code=503, detail="KB 任务队列未启用（需要 DB）")
    return store


def _uploader_of(request: Request, body: dict) -> str:
    """认证身份优先；开发（auth 关闭）回退 body.uploader。"""
    return authenticate_user(request, str(body.get("uploader", "")))


def _check_id(value: str, field: str) -> str:
    try:
        return validate_identifier(value, field)
    except InvalidIdentifier as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


def _map_exc(exc: Exception) -> JSONResponse:
    """异常 → HTTP 状态（上传错误自带 status_code；锁定/存储 → 503）。"""
    from app.agent.rag.job_store import JobConflictError
    from app.agent.rag.upload_service import UploadError
    from app.stores.upload_state import UploadStateError

    if isinstance(exc, UploadError):
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})
    if isinstance(exc, JobConflictError):
        # 文档 CAS 竞争是客户端可重试的业务冲突，不应落成 500。
        return JSONResponse(status_code=409, content={"detail": str(exc)})
    if isinstance(exc, (UploadStateError, StorageUnavailableError,
                        KbWriteLockError, KbWriteLockBackendError)):
        return JSONResponse(status_code=503, content={"detail": str(exc)})
    # 其余异常交全局 handler（500 脱敏 + trace_id）
    raise exc


async def _run(service, method, *args, **kwargs):
    """线程池执行同步 service 方法（与 runtime.run_agent_turn 同一执行模型）。"""
    try:
        return await asyncio.to_thread(getattr(service, method), *args, **kwargs)
    except Exception as e:  # noqa: BLE001
        return _map_exc(e)


def _job_response(payload: dict) -> JSONResponse:
    """任务负载 → 202（处理中）/409（永久失败）+ Location/Retry-After。"""
    status = str(payload.get("status", ""))
    headers = {"Location": str(payload.get("status_url", ""))}
    if status == "failed":
        # 永久失败/重试耗尽：409 并附任务地址（人工 POST .../retry 可重排）
        return JSONResponse(status_code=409, content=payload, headers=headers)
    return JSONResponse(
        status_code=202, content=payload,
        headers={**headers, "Retry-After": str(max(int(settings.kb_job_poll_seconds), 1))},
    )


def _is_job_payload(result) -> bool:
    return isinstance(result, dict) and "job_id" in result


def _job_dto(row: dict) -> dict:
    """任务行 → API 视图（内部租约字段不外露；时间 isoformat）。"""
    def _iso(value):
        return value.isoformat(sep=" ", timespec="seconds") if value is not None else None

    return {
        "job_id": row["job_id"],
        "operation": row["operation"],
        "doc_id": row["doc_id"],
        "upload_id": row.get("upload_id", ""),
        "requested_by": row.get("requested_by", ""),
        "status": row["status"],
        "stage": row.get("stage", ""),
        "progress": int(row.get("progress") or 0),
        "attempts": int(row.get("attempts") or 0),
        "manual_retry_count": int(row.get("manual_retry_count") or 0),
        "retryable": bool(int(row.get("retryable") or 0)),
        "error": row.get("error", ""),
        "status_url": f"/v1/kb/jobs/{row['job_id']}",
        "created_at": _iso(row.get("created_at")),
        "started_at": _iso(row.get("started_at")),
        "finished_at": _iso(row.get("finished_at")),
        "updated_at": _iso(row.get("updated_at")),
    }


@router.post("/uploads")
async def create_upload(request: Request):
    authorize_scopes(request, SCOPE_OPS)
    body = await request.json()
    uploader = _uploader_of(request, body)
    client_upload_id = str(body.get("upload_id", "") or "")
    if client_upload_id:
        _check_id(client_upload_id, "upload_id")  # 标识符白名单 422（与 P2 一致）
    return await _run(_service(request), "create_upload",
                      uploader=uploader,
                      filename=str(body.get("filename", "")),
                      size_bytes=int(body.get("size_bytes", 0) or 0),
                      content_type=str(body.get("content_type", "")),
                      chunk_size=int(body.get("chunk_size", 0) or 0),
                      sha256=str(body.get("sha256", "")),
                      upload_id=str(body.get("upload_id", "")))


@router.put("/uploads/{upload_id}/chunks/{seq}")
async def put_chunk(upload_id: str, seq: int, request: Request):
    authorize_scopes(request, SCOPE_OPS)
    _check_id(upload_id, "upload_id")
    data = await request.body()
    if len(data) > settings.kb_upload_max_chunk_size + 4096:
        return JSONResponse(status_code=413, content={"detail": "分片超过大小上限"})
    return await _run(_service(request), "put_chunk", upload_id, seq, data)


@router.get("/uploads/{upload_id}")
async def get_status(upload_id: str, request: Request):
    authorize_scopes(request, SCOPE_OPS)
    _check_id(upload_id, "upload_id")
    return await _run(_service(request), "get_status", upload_id)


@router.post("/uploads/{upload_id}/complete")
async def complete(upload_id: str, request: Request):
    """执行入库（异步：202 + 任务地址；同步兼容：200 终态）。"""
    authorize_scopes(request, SCOPE_OPS)
    _check_id(upload_id, "upload_id")
    body = {}
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 —— 允许空 body
        body = {}
    uploader = _uploader_of(request, body)
    result = await _run(_service(request), "complete", upload_id, uploader)
    if isinstance(result, JSONResponse):  # 存储层异常映射
        return result
    if _is_job_payload(result):
        return _job_response(result)
    return result  # 已 indexed：200 幂等终态


@router.delete("/uploads/{upload_id}")
async def cancel(upload_id: str, request: Request):
    authorize_scopes(request, SCOPE_OPS)
    _check_id(upload_id, "upload_id")
    uploader = _uploader_of(request, {"uploader": request.query_params.get("uploader", "")})
    return await _run(_service(request), "cancel_upload", upload_id, uploader)


@router.get("/documents")
async def list_documents(request: Request, status: str = "", limit: int = 100):
    authorize_scopes(request, SCOPE_OPS)
    docs = await _run(_service(request)._doc, "list", status=status, limit=limit)
    if isinstance(docs, JSONResponse):  # 存储层异常映射
        return docs
    return {"documents": [d.to_dict() for d in docs]}


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str, request: Request):
    authorize_scopes(request, SCOPE_OPS)
    _check_id(doc_id, "doc_id")
    rec = await _run(_service(request)._doc, "get", doc_id)
    if rec is None:
        return JSONResponse(status_code=404, content={"detail": "文档不存在"})
    if isinstance(rec, JSONResponse):  # 存储层异常映射
        return rec
    return rec.to_dict()


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, request: Request):
    """下架（异步：202 + 任务地址；已 deleted 200；同步兼容：200 终态）。"""
    authorize_scopes(request, SCOPE_OPS)
    _check_id(doc_id, "doc_id")
    uploader = _uploader_of(request, {"uploader": request.query_params.get("uploader", "")})
    result = await _run(_service(request), "delete_document", doc_id, uploader)
    if isinstance(result, JSONResponse):  # 存储层异常映射
        return result
    if _is_job_payload(result):
        return _job_response(result)
    return result  # 已 deleted / 同步终态：200


# ============================================================
# 任务管理（GET 详情/列表、POST 重试、DELETE 取消）
# ============================================================
@router.get("/jobs")
async def list_jobs(request: Request, status: str = "", operation: str = "",
                    limit: int = 50, offset: int = 0):
    """任务分页查询（按状态/操作类型过滤；created_at 倒序）。"""
    authorize_scopes(request, SCOPE_OPS)
    if status or operation:
        from app.agent.rag.job_store import (
            JOB_ACTIVE,
            JOB_TERMINAL,
            OP_DELETE,
            OP_UPLOAD,
        )

        if status and status not in (*JOB_ACTIVE, *JOB_TERMINAL):
            raise HTTPException(status_code=422, detail=f"未知任务状态: {status}")
        if operation and operation not in (OP_UPLOAD, OP_DELETE):
            raise HTTPException(status_code=422, detail=f"未知操作类型: {operation}")
    result = await _run(_job_store(request), "list",
                        status=status, operation=operation,
                        limit=limit, offset=offset)
    if isinstance(result, JSONResponse):  # 存储层异常映射
        return result
    rows, total = result
    return {"jobs": [_job_dto(r) for r in rows], "total": total,
            "limit": int(limit), "offset": int(offset)}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    """任务详情：状态、阶段、进度、尝试次数、错误及时间。"""
    authorize_scopes(request, SCOPE_OPS)
    _check_id(job_id, "job_id")
    job = await _run(_job_store(request), "get", job_id)
    if isinstance(job, JSONResponse):  # 存储层异常映射
        return job
    if job is None:
        return JSONResponse(status_code=404, content={"detail": "任务不存在"})
    return _job_dto(job)


@router.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str, request: Request):
    """人工重试：仅可重试的 failed/blocked；重置本轮 attempts（审计数只增）。"""
    authorize_scopes(request, SCOPE_OPS)
    _check_id(job_id, "job_id")
    job = await _run(_job_store(request), "retry", job_id)
    if isinstance(job, JSONResponse):  # 存储层异常映射
        return job
    if job is None:
        return JSONResponse(
            status_code=409,
            content={"detail": "仅可重试的 failed/blocked 任务允许人工重试"},
        )
    return _job_response(_job_dto(job))


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str, request: Request):
    """取消任务：仅 queued/retry_wait；运行中/终态 409；文档同事务恢复。"""
    authorize_scopes(request, SCOPE_OPS)
    _check_id(job_id, "job_id")
    _job_store(request)  # 与其它 jobs 端点一致：队列未装配时返回 503
    # service 先让 JobStore 原子提交「任务 + 文档」取消，再清理 Redis/分片；
    # 不能直接调用 store.cancel 后遗漏上传对象清理。
    job = await _run(_service(request), "cancel_job", job_id)
    if isinstance(job, JSONResponse):  # 存储层异常映射
        return job
    if job is None:
        return JSONResponse(
            status_code=409,
            content={"detail": "仅 queued/retry_wait 任务可取消（运行中不可取消）"},
        )
    dto = _job_dto(job)
    return {"job_id": dto["job_id"], "status": dto["status"],
            "doc_id": dto["doc_id"], "operation": dto["operation"]}
