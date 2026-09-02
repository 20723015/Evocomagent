"""HTTP API 请求/响应模型（阶段一 1.1）。

阶段三后 user_id 从 JWT 解出、请求体不再接受——届时调整 ChatRequest。
安全修复 P2：user_id/session_id 字符集白名单（路径穿越防护的第一层，
存储层另有兜底校验）。
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.security.identifiers import InvalidIdentifier, validate_identifier


def _check_user_id(v: str) -> str:
    try:
        return validate_identifier(v, "user_id")
    except InvalidIdentifier as e:
        raise ValueError(str(e)) from e


def _check_session_id(v: str) -> str:
    try:
        return validate_identifier(v, "session_id", allow_empty=True)
    except InvalidIdentifier as e:
        raise ValueError(str(e)) from e


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64, description="用户标识（阶段三起改从 JWT 解出）")
    session_id: str = Field("", max_length=128, description="会话标识；留空则服务端新建/续用该用户默认会话")
    message: str = Field(..., min_length=1, max_length=8000, description="用户消息")

    _user_id = field_validator("user_id")(_check_user_id)
    _session_id = field_validator("session_id")(_check_session_id)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: str
    confidence: float
    requires_human: bool
    follow_up_question: Optional[str] = None


class SessionResetRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    session_id: str = Field("", max_length=128)

    _user_id = field_validator("user_id")(_check_user_id)
    _session_id = field_validator("session_id")(_check_session_id)


class SessionResetResponse(BaseModel):
    session_id: str
    reset: bool = True


class HealthResponse(BaseModel):
    """4.1：状态兼容保留 status；components 为依赖明细（readyz 用）。"""

    status: str
    components: dict = {}  # {"redis": "ok"|"degraded"|"error: ...", ...}
