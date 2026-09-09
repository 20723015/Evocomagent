"""HTTP API 请求/响应模型（阶段一 1.1）。

阶段三后 user_id 从 JWT 解出、请求体不再接受——届时调整 ChatRequest。
安全修复 P2：user_id/session_id 字符集白名单（路径穿越防护的第一层，
存储层另有兜底校验）。
"""

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
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="用户标识（阶段三起改从 JWT 解出）",
    )
    session_id: str = Field(
        "", max_length=128, description="会话标识；留空则服务端新建/续用该用户默认会话"
    )
    message: str = Field(..., min_length=1, max_length=8000, description="用户消息")

    _user_id = field_validator("user_id")(_check_user_id)
    _session_id = field_validator("session_id")(_check_session_id)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: str
    confidence: float
    requires_human: bool
    follow_up_question: str | None = None


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


class HandoffResolveRequest(BaseModel):
    """坐席回写工单（人工客服问答沉淀）。

    - resolution 保持 dict：普通结论的扩展字段兼容（note/by/...）；
      knowledge_candidate 相关字段由服务端联动校验（422）；
    - resolution_version 必须为正整数（幂等锚点，客户端重试用同一版本）；
    - 不接受客户端 resolved_by：主体一律取 authorize_scopes(...).sub。
    """

    model_config = {"extra": "allow"}  # 扩展字段兼容（老客户端多传的字段不拒收）

    user_id: str = ""
    reclaim: bool = True
    resolution_version: int = Field(default=1, ge=1)
    resolution: dict = Field(default_factory=dict)


class HumanConversationMessage(BaseModel):
    """人工会话消息（外部客服系统推送；PII 由服务端写库前脱敏）。"""

    message_id: str = ""
    actor_type: str  # customer | human_agent | bot | system
    content: str = Field(max_length=8000)
    sent_at: str = ""


class HumanConversationItem(BaseModel):
    """一段已结束的完整会话。"""

    source: str = Field(min_length=1, max_length=64)
    external_conversation_id: str = Field(min_length=1, max_length=128)
    source_version: int = Field(default=1, ge=1)
    agent_id: str = ""
    started_at: str = ""
    ended_at: str = Field(min_length=1)
    messages: list[HumanConversationMessage]


class HumanConversationBatch(BaseModel):
    conversations: list[HumanConversationItem]


class HumanKnowledgeEditRequest(BaseModel):
    question: str = Field(min_length=1, max_length=200)
    answer: str = Field(min_length=1)
    expected_revision: int = Field(ge=0)


class HumanPublishBatchItem(BaseModel):
    candidate_id: int
    revision: int = Field(ge=0)


class HumanPublishBatchRequest(BaseModel):
    items: list[HumanPublishBatchItem] = Field(min_length=1, max_length=100)


class HumanCandidateRetireRequest(BaseModel):
    """人工下架已发布候选（乐观锁：expected_lifecycle_revision 不匹配 → 409）。"""

    reason: str = Field(min_length=1, max_length=255)
    expected_lifecycle_revision: int = Field(ge=0)
