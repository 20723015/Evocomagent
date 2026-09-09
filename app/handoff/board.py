"""转人工协议（阶段六 6.1 + 人工客服问答沉淀）。

requires_human=true 时生成 handoff 包（对话摘要 + 结构化状态 + 建议），
推坐席系统；坐席完成回写结果（resolve），Agent 可继续同一 session
（reclaim 语义 = 会话键不变，下一轮直接续上）。

存储：Redis（生产）/ 本地目录（开发）。一种实现即可满足双方。

人工知识支路（人工客服问答沉淀）：resolve 勾选 knowledge_candidate 时，
用 Lua 原子完成「工单转 resolved + 保存 resolution 事件 + 移出 pending +
加入 handoff:evolution:pending」——Evolution 定时导入为 pending 候选。
勾选只是加急通道：导入任务还会扫描全部 resolved 工单自动收集（客服零操作）。

- 幂等键 = SHA-256("human-handoff:v1:" + ticket_id + ":" + resolution_version)；
  相同请求（同幂等键或同内容指纹）重复提交返回同一事件；
- 已解决工单提交不同内容 → 冲突（API 映射 409）；
- 内容指纹 = canonical JSON（note/规范问题/答案/依据/证据）的 SHA-256，
  与幂等键互为补充：换 version 重发相同内容仍是幂等；
- 保留原 Redis key（handoff:{id} / handoff:pending），新增 handoff:resolved
  与 handoff:evolution:pending 索引；旧工单 JSON 缺新字段 → dataclass 默认值。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


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
    resolution: dict | None = None  # 坐席回写结果（含人工结论）
    # 人工知识支路（旧工单 JSON 缺省 → 默认值兼容，无需迁移）
    resolved_by: str = ""          # 解决人（ops principal，来自认证）
    resolution_id: str = ""        # 幂等键（SHA-256）
    content_digest: str = ""       # resolution 内容指纹（SHA-256）
    knowledge_candidate: bool = False  # 是否勾选沉淀为知识候选

    def to_dict(self) -> dict:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().replace(
        tzinfo=None,
    ).isoformat(timespec="seconds")


class HandoffNotFound(LookupError):
    """工单不存在。"""


class HandoffConflict(RuntimeError):
    """工单已解决且提交内容不同（幂等键与内容指纹均不匹配）。"""


def human_idempotency_key(ticket_id: str, resolution_version) -> str:
    """幂等键：SHA-256("human-handoff:v1:" + ticket_id + ":" + version)。"""
    return hashlib.sha256(
        f"human-handoff:v1:{ticket_id}:{int(resolution_version)}".encode(),
    ).hexdigest()


def human_content_digest(resolution: dict) -> str:
    """resolution 内容指纹：canonical JSON 的 SHA-256（键排序，稳定）。"""
    payload = {
        "knowledge_candidate": bool(resolution.get("knowledge_candidate", False)),
        "note": str(resolution.get("note", "")),
        "canonical_question": str(resolution.get("canonical_question", "")),
        "canonical_answer": str(resolution.get("canonical_answer", "")),
        "knowledge_basis": str(resolution.get("knowledge_basis", "")),
        "evidence_source_paths": sorted(
            str(p) for p in resolution.get("evidence_source_paths") or []
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
    ).hexdigest()


def build_resolution_event(ticket_id: str, resolution: dict, resolved_by: str,
                           idempotency_key: str) -> dict:
    """resolution 事件（幂等重放返回同一事件；Evolution 导入的数据源）。"""
    return {
        "ticket_id": ticket_id,
        "resolved_by": resolved_by,
        "knowledge_candidate": bool(resolution.get("knowledge_candidate", False)),
        "note": str(resolution.get("note", "")),
        "canonical_question": str(resolution.get("canonical_question", "")),
        "canonical_answer": str(resolution.get("canonical_answer", "")),
        "knowledge_basis": str(resolution.get("knowledge_basis", "")),
        "evidence_source_paths": [
            str(p) for p in resolution.get("evidence_source_paths") or []
        ],
        "idempotency_key": idempotency_key,
        "resolved_at": _now(),
    }


class InProcessHandoffBoard:
    """进程内实现（测试/单机）。候选队列不可恢复（durable=False）——
    人工沉淀请求必须 503 拒绝，绝不写入进程内队列。"""

    durable = False

    def __init__(self):
        self._tickets: dict[str, HandoffTicket] = {}
        self.resolved_set: set[str] = set()
        self.evolution_pending: set[str] = set()
        self._resolution_events: dict[str, dict] = {}

    def create(self, ticket: HandoffTicket) -> HandoffTicket:
        self._tickets[ticket.ticket_id] = ticket
        return ticket

    def get(self, ticket_id: str) -> HandoffTicket | None:
        return self._tickets.get(ticket_id)

    def list(self, status: str = "pending") -> list[HandoffTicket]:
        if status == "all":
            return list(self._tickets.values())
        return [t for t in self._tickets.values() if t.status == status]

    def resolve(self, ticket_id: str, resolution: dict) -> HandoffTicket | None:
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            return None
        ticket.status = "resolved"
        ticket.resolution = resolution
        return ticket

    def resolve_atomic(self, ticket_id: str, resolution: dict, resolved_by: str,
                       resolution_version: int = 1,
                       *, now: str | None = None) -> tuple[dict, bool]:
        """原子 resolve（与 Redis Lua 同语义）：返回 (事件, 是否幂等重放)。"""
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            raise HandoffNotFound(ticket_id)
        idem = human_idempotency_key(ticket_id, resolution_version)
        digest = human_content_digest(resolution)
        if ticket.status == "resolved":
            # 内容指纹是唯一冲突判据：相同内容（无论 version）→ 同一事件；
            # 不同内容（即使撞幂等键，如同 version 改字段）→ 409
            if ticket.content_digest == digest:
                stored = self._resolution_events.get(ticket_id)
                if stored is not None:
                    return dict(stored), True
            raise HandoffConflict(
                f"工单 {ticket_id} 已解决且提交内容不同（内容指纹不匹配）"
            )
        ticket.status = "resolved"
        ticket.resolution = dict(resolution)
        ticket.resolved_by = resolved_by
        ticket.resolution_id = idem
        ticket.content_digest = digest
        ticket.knowledge_candidate = bool(resolution.get("knowledge_candidate", False))
        event = build_resolution_event(ticket_id, resolution, resolved_by, idem)
        if now is not None:  # 测试时钟注入
            event["resolved_at"] = now
        self._resolution_events[ticket_id] = event
        self.resolved_set.add(ticket_id)
        self.evolution_pending.discard(ticket_id)
        if ticket.knowledge_candidate:
            self.evolution_pending.add(ticket_id)
        return event, False

    def evolution_pending_ids(self) -> list[str]:
        """人工候选导入队列快照（与 RedisHandoffBoard 同接口）。"""
        return sorted(self.evolution_pending)

    def drop_evolution_pending(self, ticket_id: str) -> None:
        self.evolution_pending.discard(ticket_id)

    def resolved_ids(self) -> list[str]:
        return sorted(self.resolved_set)

    def get_resolution_event(self, ticket_id: str) -> dict | None:
        """只读读取原子 resolve 保存的 resolution 事件。"""
        event = self._resolution_events.get(ticket_id)
        return deepcopy(event) if event is not None else None

    def resolution_event(self, ticket_id: str) -> dict | None:
        """``get_resolution_event`` 的兼容别名（仍为只读）。"""
        return self.get_resolution_event(ticket_id)


# 原子 resolve 的 Lua：工单转 resolved + 保存事件 + 移出 pending +
# 加入 resolved/evolution pending 索引（全部成功或全部不执行）。
# KEYS: 1=handoff:{id} 2=handoff:resolution:{id} 3=handoff:pending
#       4=handoff:resolved 5=handoff:evolution:pending
# ARGV: 1=新工单 JSON 2=事件 JSON 3=内容指纹 4=幂等键 5=ticket_id
#       6=knowledge_candidate(0/1)
# 返回: {OK|DUPLICATE|CONFLICT|NOT_FOUND, 事件 JSON?}
_RESOLVE_LUA = """
local raw = redis.call('GET', KEYS[1])
if not raw then return {'NOT_FOUND', ''} end
local t = cjson.decode(raw)
if t['status'] == 'resolved' then
  -- 内容指纹是唯一冲突判据：相同内容（无论 version）→ 同一事件；不同内容 → 冲突
  if t['content_digest'] == ARGV[3] then
    local ev = redis.call('GET', KEYS[2])
    if ev then return {'DUPLICATE', ev} end
  end
  return {'CONFLICT', ''}
end
redis.call('SET', KEYS[1], ARGV[1])
redis.call('SET', KEYS[2], ARGV[2])
redis.call('SREM', KEYS[3], ARGV[5])
redis.call('SADD', KEYS[4], ARGV[5])
if ARGV[6] == '1' then redis.call('SADD', KEYS[5], ARGV[5]) end
return {'OK', ARGV[2]}
"""


class RedisHandoffBoard:
    """Redis 实现：handoff:{ticket_id} = JSON；list 用 set 索引（pending/resolved）。"""

    KEY_PREFIX = "handoff:"
    PENDING_SET = "handoff:pending"
    RESOLVED_SET = "handoff:resolved"            # 新增：已解决索引（新工单开始维护，不回扫）
    EVOLUTION_PENDING_SET = "handoff:evolution:pending"  # 新增：人工候选导入队列
    RESOLUTION_KEY_PREFIX = "handoff:resolution:"  # 新增：resolution 事件（幂等重放正本）

    durable = True

    def __init__(self, redis):
        self._redis = redis

    def create(self, ticket: HandoffTicket) -> HandoffTicket:
        self._redis.set(self.KEY_PREFIX + ticket.ticket_id,
                        json.dumps(ticket.to_dict(), ensure_ascii=False))
        self._redis.sadd(self.PENDING_SET, ticket.ticket_id)
        return ticket

    def get(self, ticket_id: str) -> HandoffTicket | None:
        raw = self._redis.get(self.KEY_PREFIX + ticket_id)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return HandoffTicket(**json.loads(raw))

    def list(self, status: str = "pending") -> list[HandoffTicket]:
        """pending / resolved / all（合并去重；状态校验在 API 层 422）。"""
        if status == "pending":
            ids = self._redis.smembers(self.PENDING_SET)
        elif status == "resolved":
            ids = self._redis.smembers(self.RESOLVED_SET)
        elif status == "all":
            ids = (self._redis.smembers(self.PENDING_SET)
                   | self._redis.smembers(self.RESOLVED_SET))
        else:
            ids = []
        out = []
        seen: set[str] = set()
        for tid in ids:
            tid = tid.decode() if isinstance(tid, bytes) else tid
            if tid in seen:
                continue
            seen.add(tid)
            tick = self.get(tid)
            if tick is not None:
                out.append(tick)
        return out

    def resolve(self, ticket_id: str, resolution: dict) -> HandoffTicket | None:
        """旧接口（直接置 resolved，不维护 resolved/evolution 索引）——仅存量兼容。"""
        ticket = self.get(ticket_id)
        if ticket is None:
            return None
        ticket.status = "resolved"
        ticket.resolution = resolution
        self._redis.set(self.KEY_PREFIX + ticket_id,
                        json.dumps(ticket.to_dict(), ensure_ascii=False))
        self._redis.srem(self.PENDING_SET, ticket_id)
        return ticket

    def resolve_atomic(self, ticket_id: str, resolution: dict, resolved_by: str,
                       resolution_version: int = 1,
                       *, now: str | None = None) -> tuple[dict, bool]:
        """Lua 原子 resolve：工单/事件/pending/索引一次提交，防双实例竞态。

        返回 (事件, 是否幂等重放)；不存在/冲突抛 HandoffNotFound/HandoffConflict。
        """
        ticket = self.get(ticket_id)
        if ticket is None:
            raise HandoffNotFound(ticket_id)
        idem = human_idempotency_key(ticket_id, resolution_version)
        digest = human_content_digest(resolution)  # 唯一冲突判据（Lua 同口径）
        event = build_resolution_event(ticket_id, resolution, resolved_by, idem)
        if now is not None:  # 测试时钟注入
            event["resolved_at"] = now
        new_ticket = dict(ticket.to_dict())
        new_ticket.update({
            "status": "resolved",
            "resolution": dict(resolution),
            "resolved_by": resolved_by,
            "resolution_id": idem,
            "content_digest": digest,
            "knowledge_candidate": bool(resolution.get("knowledge_candidate", False)),
        })
        result = self._redis.eval(
            _RESOLVE_LUA, 5,
            self.KEY_PREFIX + ticket_id,
            self.RESOLUTION_KEY_PREFIX + ticket_id,
            self.PENDING_SET,
            self.RESOLVED_SET,
            self.EVOLUTION_PENDING_SET,
            json.dumps(new_ticket, ensure_ascii=False),
            json.dumps(event, ensure_ascii=False),
            digest, idem, ticket_id,
            "1" if new_ticket["knowledge_candidate"] else "0",
        )
        status = result[0].decode() if isinstance(result[0], bytes) else result[0]
        if status == "NOT_FOUND":  # 竞态：get 后被删除（罕见）
            raise HandoffNotFound(ticket_id)
        if status == "CONFLICT":
            raise HandoffConflict(
                f"工单 {ticket_id} 已解决且提交内容不同（幂等键/内容指纹不匹配）"
            )
        payload = result[1]
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        return json.loads(payload), status == "DUPLICATE"

    def evolution_pending_ids(self) -> list[str]:
        """人工候选导入队列快照（Evolution 导入器消费）。"""
        ids = self._redis.smembers(self.EVOLUTION_PENDING_SET)
        return [
            i.decode() if isinstance(i, bytes) else i for i in ids
        ]

    def drop_evolution_pending(self, ticket_id: str) -> None:
        """导入完成/终结后移出导入队列（先持久化 ledger 再调用）。"""
        self._redis.srem(self.EVOLUTION_PENDING_SET, ticket_id)

    def resolved_ids(self) -> list[str]:
        """全部 resolved 工单快照（cronjob 自动收集扫描用；与
        InProcessHandoffBoard.resolved_ids 同接口）。

        注意：RESOLVED_SET 只维护索引引入后的新工单，存量旧工单不在其中。
        """
        ids = self._redis.smembers(self.RESOLVED_SET)
        return [i.decode() if isinstance(i, bytes) else i for i in ids]

    def get_resolution_event(self, ticket_id: str) -> dict | None:
        """只读读取 Redis 中的 resolution 事件正本。"""
        raw = self._redis.get(self.RESOLUTION_KEY_PREFIX + ticket_id)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            event = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None
        return event if isinstance(event, dict) else None

    def resolution_event(self, ticket_id: str) -> dict | None:
        """``get_resolution_event`` 的兼容别名（仍为只读）。"""
        return self.get_resolution_event(ticket_id)


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
