"""转人工协议（阶段六 6.1）。

requires_human=true 时生成 handoff 包（对话摘要 + 结构化状态 + 建议），
推坐席系统；坐席完成回写结果（resolve），Agent 可继续同一 session
（reclaim 语义 = 会话键不变，下一轮直接续上）。

存储：Redis（生产）/ 本地目录（开发）。一种实现即可满足双方。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class HandoffTicket:
    """一份转人工工单。"""

    ticket_id: str
    user_id: str
    session_id: str
    intent: str = ""
    summary: str = ""          # 对话摘要（agent.summary）
    question: str = ""         # 触发转人工的用户消息
    reply: str = ""            # agent 给出的话术
    suggested_actions: list[str] = field(default_factory=list)
    created_at: str = ""
    status: str = "pending"    # pending | resolved
    resolution: Optional[dict] = None  # 坐席回写结果（含人工结论）

    def to_dict(self) -> dict:
        return asdict(self)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class InProcessHandoffBoard:
    """进程内实现（测试/单机）。"""

    def __init__(self):
        self._tickets: dict[str, HandoffTicket] = {}

    def create(self, ticket: HandoffTicket) -> HandoffTicket:
        self._tickets[ticket.ticket_id] = ticket
        return ticket

    def get(self, ticket_id: str) -> Optional[HandoffTicket]:
        return self._tickets.get(ticket_id)

    def list(self, status: str = "pending") -> list[HandoffTicket]:
        return [t for t in self._tickets.values()
                if status in ("", "all") or t.status == status]

    def resolve(self, ticket_id: str, resolution: dict) -> Optional[HandoffTicket]:
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            return None
        ticket.status = "resolved"
        ticket.resolution = resolution
        return ticket


class RedisHandoffBoard:
    """Redis 实现：handoff:{ticket_id} = JSON；list 用 hash 索引（pending set）。"""

    KEY_PREFIX = "handoff:"
    PENDING_SET = "handoff:pending"

    def __init__(self, redis):
        self._redis = redis

    def create(self, ticket: HandoffTicket) -> HandoffTicket:
        self._redis.set(self.KEY_PREFIX + ticket.ticket_id,
                        json.dumps(ticket.to_dict(), ensure_ascii=False))
        self._redis.sadd(self.PENDING_SET, ticket.ticket_id)
        return ticket

    def get(self, ticket_id: str) -> Optional[HandoffTicket]:
        raw = self._redis.get(self.KEY_PREFIX + ticket_id)
        if raw is None:
            return None
        return HandoffTicket(**json.loads(raw))

    def list(self, status: str = "pending") -> list[HandoffTicket]:
        ids = self._redis.smembers(self.PENDING_SET) if status == "pending" else []
        out = []
        for tid in list(ids):
            tick = self.get(tid.decode() if isinstance(tid, bytes) else tid)
            if tick is not None:
                out.append(tick)
        return out

    def resolve(self, ticket_id: str, resolution: dict) -> Optional[HandoffTicket]:
        ticket = self.get(ticket_id)
        if ticket is None:
            return None
        ticket.status = "resolved"
        ticket.resolution = resolution
        self._redis.set(self.KEY_PREFIX + ticket_id,
                        json.dumps(ticket.to_dict(), ensure_ascii=False))
        self._redis.srem(self.PENDING_SET, ticket_id)
        return ticket


def build_handoff_ticket(agent, result) -> HandoffTicket:
    """从 Agent 与结构化响应构建工单（摘要 + 状态 + 建议）。"""
    suggested = [a for a in result.follow_up_question.split("\n")
                 if a.strip()] if result.follow_up_question else []
    # 修复计划：写结果未知（indeterminate）→ 建议人工按幂等键对账（不记录 token）
    for w in getattr(agent, "_indeterminate_writes", None) or []:
        suggested.append(
            f"工具结果未知需人工对账: {w.get('tool')}, 订单号 {w.get('order_id')}, "
            f"幂等键 {w.get('idempotency_key')}"
        )
    return HandoffTicket(
        ticket_id=uuid.uuid4().hex,
        user_id=getattr(agent, "user_id", ""),
        session_id=agent.session_id,
        intent=result.intent.value if hasattr(result.intent, "value") else str(result.intent),
        summary=getattr(agent, "summary", "") or "",
        question=(getattr(agent, "raw_messages", []) or [{}])[:-1][-1].get("content", "")
        if getattr(agent, "raw_messages", None) else "",
        reply=result.reply,
        suggested_actions=suggested,
        created_at=_now(),
    )


def get_board(redis=None):
    from app.stores.redis_client import get_redis as _get_redis

    redis = redis if redis is not None else _get_redis()
    if redis is not None:
        return RedisHandoffBoard(redis)
    return InProcessHandoffBoard()
