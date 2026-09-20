"""渠道适配层（P2-2）：generic webhook 渠道入口的归一 / 路由 / 出站。

定位：**渠道就绪性演示**——不接任何真实第三方渠道（无商家资质，接了也是
空转），只证明「消息入口与 Agent 运行时已解耦」：任何渠道把消息按通用
webhook 格式推进来，服务端做三件事后复用现有 chat 管线（build_agent →
SessionLease → runtime.run_agent_turn，含写确认协议）：

1. 消息格式归一：渠道无关的信封（external_user_id / message / session_id /
   message_id / metadata）→ InboundMessage；第三方字段差异由渠道适配器在
   推送到 webhook 前完成（或在此扩展 alias 映射）。
2. 会话路由（user_id 映射）：`channel_user_id(channel, external_user_id)`
   把外部身份映射为内部 user_id（确定性 + 不可歧义 + 字符集合法），
   同一外部用户始终落到同一会话/记忆/限流桶；渠道不能直接指定内部 user_id。
3. 出站：**轮询**（见下）。

出站为什么选轮询而不是回调：
- 回调需要服务端主动出网（回调 URL 校验、SSRF 防护、重试/退避/死信、签名），
  这套机制在「没有真实渠道」的前提下是纯空转代码，且引入出网风险面；
- 轮询把第三方连接留给渠道适配器自己（真实渠道适配器本就持有 WebSocket/
  长连接），服务端只维护一个 append-only 出站队列（Redis 有则持久，无则
  进程内），语义与仓库既有 outbox（MySQL→ES）一致；
- 渠道适配器用 cursor 拉取，至少一次投递、可重放，测试可确定性断言。

队列语义：append-only，cursor = 出站序号；保留最近 MAX_OUTBOUND_RETAINED
条（Redis ZSET + TTL）。不实现 ack（适配器用 cursor 自行推进）。
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import dataclass, field

from app.handoff.board import _now
from app.security.identifiers import InvalidIdentifier, validate_identifier

# 渠道标识：小写字母/数字开头，[a-z0-9_-]，≤32（避免大小写造成同一渠道两套队列）
CHANNEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")

# 出站队列保留条数（每渠道）与 Redis TTL
MAX_OUTBOUND_RETAINED = 1000
OUTBOUND_TTL_SECONDS = 7 * 24 * 3600

# 内部 user_id 映射前缀：`ch-{channel}-{external}-{digest8}`。
# digest 后缀保证 (channel, external_user_id) 二元组不歧义——channel 与
# external 都允许 '-'，仅靠分隔符拼接会出现 a-b/c 与 a/b-c 撞同一 user_id。
USER_PREFIX = "ch"
_EXTERNAL_ID_MAX = 64  # 与 user_id 上限一致（schema 同步约束）


class InvalidChannel(ValueError):
    """渠道标识非法（API 映射 422）。"""


def validate_channel_id(channel: str) -> str:
    """渠道标识白名单校验；非法抛 InvalidChannel。"""
    if not isinstance(channel, str) or not CHANNEL_ID_RE.match(channel):
        raise InvalidChannel(
            "渠道标识含非法字符（仅允许小写字母/数字开头，[a-z0-9_-]，≤32 字符）"
        )
    return channel


def channel_user_id(channel: str, external_user_id: str) -> str:
    """外部身份 → 内部 user_id（确定性映射，同一外部用户始终同一内部用户）。

    形如 `ch-web-ext-1-1a2b3c4d`：前缀 + 可读片段 + 8 位摘要后缀（消歧义）。
    映射结果满足标识符白名单（会进会话文件路径/Redis key）。
    """
    import hashlib

    digest = hashlib.sha256(
        f"{channel}\x00{external_user_id}".encode("utf-8"),
    ).hexdigest()[:8]
    mapped = f"{USER_PREFIX}-{channel}-{external_user_id}-{digest}"
    # 长度兜底：external ≤64 + channel ≤32 + 前缀/后缀 → ≤ 109，仍防御性校验
    if len(mapped) > 128:
        mapped = f"{USER_PREFIX}-{channel}-{digest}"
    validate_identifier(mapped, "user_id")
    return mapped


def default_session_id(channel: str, external_user_id: str) -> str:
    """未显式给 session_id 时的默认会话（同一外部用户续用同一会话）。"""
    return validate_identifier(f"{channel}-{external_user_id}", "session_id")


@dataclass(frozen=True)
class InboundMessage:
    """归一后的渠道入站消息（渠道无关信封）。"""

    channel: str
    external_user_id: str
    message: str
    session_id: str
    message_id: str
    received_at: str
    metadata: dict = field(default_factory=dict)


def normalize_inbound(channel: str, body) -> InboundMessage:
    """归一：渠道无关字段 + 默认值（message_id / session_id / 接收时间）。

    body 为 ChannelMessageRequest（或同字段对象）；外部标识先过字符集白名单，
    避免把渠道侧任意字符串带进路径/键空间。
    """
    external_user_id = validate_identifier(
        body.external_user_id, "external_user_id"
    )
    if len(external_user_id) > _EXTERNAL_ID_MAX:
        raise InvalidIdentifier(
            f"external_user_id 超长（≤{_EXTERNAL_ID_MAX} 字符）"
        )
    session_id = body.session_id or default_session_id(channel, external_user_id)
    validate_identifier(session_id, "session_id")
    message_id = (body.message_id or "").strip() or uuid.uuid4().hex
    validate_identifier(message_id, "message_id")
    return InboundMessage(
        channel=channel,
        external_user_id=external_user_id,
        message=body.message,
        session_id=session_id,
        message_id=message_id,
        received_at=_now(),
        metadata=dict(getattr(body, "metadata", None) or {}),
    )


def outbound_payload(
    *,
    inbound: InboundMessage,
    user_id: str,
    reply: str,
    intent: str,
    confidence: float,
    requires_human: bool,
    handoff: dict | None = None,
    pending_turn: dict | None = None,
) -> dict:
    """构造出站消息（渠道适配器轮询消费；seq 由 outbox.append 填充）。"""
    return {
        "seq": None,
        "direction": "outbound",
        "channel": inbound.channel,
        "message_id": inbound.message_id,          # 入站消息 id（对账/去重用）
        "external_user_id": inbound.external_user_id,
        "user_id": user_id,
        "session_id": inbound.session_id,
        "reply": reply,
        "intent": intent,
        "confidence": confidence,
        "requires_human": requires_human,
        "handoff": handoff,
        "pending_turn": pending_turn,
        "created_at": _now(),
    }


class InProcessChannelOutbox:
    """进程内出站队列（测试/单机开发；多 Pod 必须 Redis 实现）。"""

    durable = False

    def __init__(self, max_retained: int = MAX_OUTBOUND_RETAINED):
        self._lock = threading.Lock()
        self._messages: dict[str, list[dict]] = {}
        self._seq: dict[str, int] = {}
        self._max = max_retained

    def append(self, channel: str, payload: dict) -> int:
        with self._lock:
            seq = self._seq.get(channel, 0) + 1
            self._seq[channel] = seq
            item = dict(payload)
            item["seq"] = seq
            bucket = self._messages.setdefault(channel, [])
            bucket.append(item)
            if len(bucket) > self._max:
                del bucket[: len(bucket) - self._max]
            return seq

    def list_since(self, channel: str, cursor: int = 0,
                   limit: int = 50) -> dict:
        with self._lock:
            bucket = self._messages.get(channel, [])
            items = [dict(m) for m in bucket if int(m.get("seq", 0)) > cursor]
        items = items[:limit]
        next_cursor = items[-1]["seq"] if items else cursor
        return {"messages": items, "next_cursor": next_cursor}


class RedisChannelOutbox:
    """Redis 出站队列：`channel:out:{channel}` ZSET（score=序号，append-only）。

    序号由 INCR 生成（多 Pod 不重号）；ZADD 单条原子追加；ZREMRANGEBYRANK
    裁剪最旧条目（保留最近 MAX 条）并刷新 TTL。
    """

    durable = True
    SEQ_KEY_PREFIX = "channel:out:seq:"
    ZSET_KEY_PREFIX = "channel:out:"

    def __init__(self, redis, max_retained: int = MAX_OUTBOUND_RETAINED):
        self._redis = redis
        self._max = max_retained

    def append(self, channel: str, payload: dict) -> int:
        item = dict(payload)
        seq = int(self._redis.incr(self.SEQ_KEY_PREFIX + channel))
        item["seq"] = seq
        zset_key = self.ZSET_KEY_PREFIX + channel
        self._redis.zadd(
            zset_key, {json.dumps(item, ensure_ascii=False): seq}
        )
        # 只保留最近 max 条（rank 0 = 最旧）；裁剪与追加不在同一事务内，
        # 极端并发下多裁一条不影响 cursor 语义（序号单调、不重号）
        try:
            self._redis.zremrangebyrank(zset_key, 0, -(self._max + 1))
        except Exception:  # noqa: BLE001 —— 裁剪失败不影响投递
            pass
        self._redis.expire(zset_key, OUTBOUND_TTL_SECONDS)
        self._redis.expire(self.SEQ_KEY_PREFIX + channel, OUTBOUND_TTL_SECONDS)
        return seq

    def list_since(self, channel: str, cursor: int = 0,
                   limit: int = 50) -> dict:
        zset_key = self.ZSET_KEY_PREFIX + channel
        raw = self._redis.zrangebyscore(
            zset_key, f"({int(cursor)}", "+inf", start=0, num=limit,
        )
        messages = []
        for item in raw or []:
            if isinstance(item, bytes):
                item = item.decode("utf-8")
            try:
                parsed = json.loads(item)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict):
                messages.append(parsed)
        next_cursor = messages[-1].get("seq", cursor) if messages else cursor
        return {"messages": messages, "next_cursor": next_cursor}


def build_channel_outbox(redis=None):
    """按 Redis 可用性选择出站队列实现（与 handoff board 同模式）。"""
    if redis is not None:
        return RedisChannelOutbox(redis)
    return InProcessChannelOutbox()
