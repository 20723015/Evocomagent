"""HTTP API 请求/响应模型（阶段一 1.1）。

阶段三后 user_id 从 JWT 解出、请求体不再接受——届时调整 ChatRequest。
安全修复 P2：user_id/session_id 字符集白名单（路径穿越防护的第一层，
存储层另有兜底校验）。
"""

from pydantic import AliasChoices, BaseModel, Field, field_validator

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


def _check_optional_id(field_name: str):
    def _check(v: str) -> str:
        try:
            return validate_identifier(v, field_name, allow_empty=True)
        except InvalidIdentifier as e:
            raise ValueError(str(e)) from e

    return _check


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
    pending_turn: dict | None = None
    # 掉线恢复（记忆系统重构·Step6）：非空 = 进入本轮前检测到上次回复未完成
    # （{turn_id, user_message, started_at}）；客户端可据此提示「上次回复未完成」
    # 并提供重发入口。本轮正常收尾后该标记已在服务端清除。
    # P2-3：requires_human 时携带用户可见状态（ticket_id/status/message），
    # 会话 meta 与工单号对账（SSE 同名字段走 handoff 事件）。
    handoff: dict | None = None


class ChannelMessageRequest(BaseModel):
    """渠道 webhook 入站消息（P2-2，通用信封；不接真实第三方渠道）。

    - external_user_id：渠道侧用户标识（必填；服务端映射为内部 user_id，
      渠道不能直接指定内部 user_id）；兼容 user_id 别名；
    - message：文本内容；兼容 text/content 别名（渠道适配器字段差异）；
    - session_id 留空 → 该外部用户的默认会话（会话路由确定性）；
    - message_id 可选（渠道重试对账用；留空服务端生成）。
    """

    model_config = {"populate_by_name": True}

    external_user_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        validation_alias=AliasChoices("external_user_id", "user_id"),
        description="渠道侧用户标识（服务端映射为内部 user_id）",
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        validation_alias=AliasChoices("message", "text", "content"),
    )
    session_id: str = Field("", max_length=128)
    message_id: str = Field("", max_length=128)
    metadata: dict = Field(default_factory=dict)

    _external_user_id = field_validator("external_user_id")(_check_user_id)
    _session_id = field_validator("session_id")(_check_session_id)
    _message_id = field_validator("message_id")(_check_optional_id("message_id"))


class ChannelMessageResponse(BaseModel):
    """webhook 同步受理结果（出站走 GET /v1/channels/{channel}/outbound 轮询）。"""

    channel: str
    message_id: str
    user_id: str          # 映射后的内部用户
    session_id: str
    status: str = "replied"
    outbound_seq: int     # 出站队列序号（轮询 cursor 起点）
    requires_human: bool
    handoff: dict | None = None


class ChannelOutboundResponse(BaseModel):
    """渠道出站轮询结果：cursor 之后的出站消息（至少一次投递）。"""

    channel: str
    messages: list[dict]
    next_cursor: int


class HandoffNoteRequest(BaseModel):
    """坐席处理备注（P2-3：append-only 留痕，不改变工单状态）。"""

    note: str = Field(..., min_length=1, max_length=4000)


class SessionResetRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    session_id: str = Field("", max_length=128)

    _user_id = field_validator("user_id")(_check_user_id)
    _session_id = field_validator("session_id")(_check_session_id)


class SessionResetResponse(BaseModel):
    session_id: str
    reset: bool = True


class HealthResponse(BaseModel):
    """4.1：状态兼容保留 status；components 为依赖明细（readyz 用）。

    修复计划·三：新增 capabilities（能力分级）；status ∈ ready|degraded|not_ready。
    """

    status: str
    components: dict = {}  # {"redis": "ok"|"not_configured"|"unavailable", ...}
    capabilities: dict = {}  # {"chat": "ok"|"not_configured"|"unavailable", ...}


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
