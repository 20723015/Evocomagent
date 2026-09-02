"""KB 文档上传 API（v7 冻结的 8 端点，全部 SCOPE_OPS）。

端点：
    POST   /v1/kb/uploads                              创建断点会话（upload_id 幂等锚点）
    PUT    /v1/kb/uploads/{upload_id}/chunks/{seq}      上传分片（octet-stream；重传幂等）
    GET    /v1/kb/uploads/{upload_id}                   断点状态（received[] 供续传）
    POST   /v1/kb/uploads/{upload_id}/complete          执行入库（幂等；同步最长几十秒）
    DELETE /v1/kb/uploads/{upload_id}                   取消上传（封口→清对象→cancelled）
    GET    /v1/kb/documents                             列表（全量，不做本人过滤）
    GET    /v1/kb/documents/{doc_id}                    详情
    DELETE /v1/kb/documents/{doc_id}                    下架（trash→重建→deleted；失败回滚）

uploader 语义：来自认证身份（auth 关闭时回退 body uploader）——create 幂等比对、
complete/delete 行归属校验（不一致 403，防会话碰撞/接管）；列表语义为 ops
全量可见。service 是同步代码：async 端点一律经 asyncio.to_thread（执行模型 1.7）。
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
    from app.agent.rag.upload_service import UploadError
    from app.stores.upload_state import UploadStateError

    if isinstance(exc, UploadError):
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})
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
    authorize_scopes(request, SCOPE_OPS)
    _check_id(upload_id, "upload_id")
    body = {}
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 —— 允许空 body
        body = {}
    uploader = _uploader_of(request, body)
    return await _run(_service(request), "complete", upload_id, uploader)


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
    authorize_scopes(request, SCOPE_OPS)
    _check_id(doc_id, "doc_id")
    uploader = _uploader_of(request, {"uploader": request.query_params.get("uploader", "")})
    return await _run(_service(request), "delete_document", doc_id, uploader)
