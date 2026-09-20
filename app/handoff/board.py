"""转人工协议（阶段六 6.1 + 人工客服问答沉淀 + P2-3 最小坐席工作台）。

requires_human=true 时生成 handoff 包（对话摘要 + 结构化状态 + 建议），
推坐席系统；坐席完成回写结果（resolve），Agent 可继续同一 session
（reclaim 语义 = 会话键不变，下一轮直接续上）。

存储：Redis（生产）/ 本地目录（开发）。一种实现即可满足双方。

P2-3 坐席工作台（领取 / SLA / 处理留痕）：
- 领取（claim）：工单可被一名坐席领取（assignee/claimed_at）；重复领取同一
  坐席幂等，他人已领取/已解决 → HandoffConflict（API 409）。Redis 侧用 Lua
  完成「读当前状态 → 比较 → 写入」的原子操作（与 resolve 同一模式）；进程内
  实现用 RLock 保护同一临界区（单进程语义等价）。
- SLA：从 created_at 起算 HANDOFF_SLA_SECONDS（模块级常量，不加 settings
  开关）；已解决工单按 resolved_at 判定，未解决按当前时间判定；超时由
  ops API 在 to_ops_dict 中透出（UI 标红）。
- 处理留痕：events 为 append-only 事件流（claimed / note / resolved），
  记录 actor（认证主体，不接受请求体伪造）与时间。Redis 侧事件存
  handoff:trail:{id} 列表（RPUSH，与工单状态变更同一 Lua 提交，原子且不
  受 cjson 空数组编码影响）；进程内实现直接追加到 ticket.events。

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
import threading
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

# P2-3：工单 SLA 时长（创建时间起算）。模块级常量而非 settings 字段——
# 能力默认启用，不引入新开关；调整口径直接改这里。
HANDOFF_SLA_SECONDS = 30 * 60

# _now() 输出本地 naive ISO；比较时把带时区的时间统一折算到本地 naive。
_LOCAL_TZ = datetime.now().astimezone().tzinfo


def _parse_dt(value) -> datetime | None:
    """宽松解析 ISO 时间 → 本地 naive datetime；空值/坏值返回 None。"""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(_LOCAL_TZ).replace(tzinfo=None)
    return dt


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
    # P2-3 坐席工作台（旧工单 JSON 缺省 → 默认值兼容，无需迁移）
    assignee: str = ""             # 领取人（ops principal sub；空 = 未领取）
    claimed_at: str = ""           # 领取时间
    resolved_at: str = ""          # 解决时间（与 resolution 事件的 resolved_at 同源）
    events: list[dict] = field(default_factory=list)  # append-only 处理留痕

    def to_dict(self) -> dict:
        return asdict(self)

    # ---------- P2-3：SLA 判定（只读计算，不落库）----------
    def sla_due(self) -> datetime | None:
        """SLA 截止时间（created_at + HANDOFF_SLA_SECONDS）；坏时间返回 None。"""
        created = _parse_dt(self.created_at)
        if created is None:
            return None
        return created + timedelta(seconds=HANDOFF_SLA_SECONDS)

    def sla_breached(self, now=None) -> bool:
        """是否超时：已解决按 resolved_at，未解决按 now（缺省当前时间）。

        时间字段缺失/非法 → 不判超时（fail-open，避免脏数据把看板整片标红）。
        """
        due = self.sla_due()
        if due is None:
            return False
        reference = _parse_dt(self.resolved_at) if self.resolved_at else _parse_dt(now)
        if reference is None:
            reference = datetime.now()
        return reference > due

    def to_ops_dict(self, now=None) -> dict:
        """工单 + SLA 计算字段（ops API/工作台 UI 用；存储仍走 to_dict）。"""
        data = self.to_dict()
        due = self.sla_due()
        reference = (
            _parse_dt(self.resolved_at) if self.resolved_at else _parse_dt(now)
        )
        if reference is None:
            reference = datetime.now()
        remaining = None
        if due is not None:
            remaining = int((due - reference).total_seconds())
        data.update({
            "sla_seconds": HANDOFF_SLA_SECONDS,
            "sla_due_at": due.isoformat(timespec="seconds") if due else "",
            "sla_breached": self.sla_breached(now=now),
            "sla_remaining_seconds": remaining,
        })
        return data


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
    人工沉淀请求必须 503 拒绝，绝不写入进程内队列。

    P2-3：claim/note/resolve 用 RLock 保护「读-比较-写」临界区，单进程语义
    与 Redis Lua 一致（多进程部署必须走 Redis 实现）。
    """

    durable = False

    def __init__(self):
        self._tickets: dict[str, HandoffTicket] = {}
        self.resolved_set: set[str] = set()
        self.evolution_pending: set[str] = set()
        self._resolution_events: dict[str, dict] = {}
        self._sla_breached: set[str] = set()
        self._lock = threading.RLock()

    def create(self, ticket: HandoffTicket) -> HandoffTicket:
        with self._lock:
            self._tickets[ticket.ticket_id] = ticket
            return ticket

    def get(self, ticket_id: str) -> HandoffTicket | None:
        return self._tickets.get(ticket_id)

    def list(self, status: str = "pending", *, assignee: str = "") -> list[HandoffTicket]:
        """status ∈ pending/resolved/all；assignee 非空时只返回该坐席领取的工单。"""
        with self._lock:
            if status == "all":
                out = list(self._tickets.values())
            else:
                out = [t for t in self._tickets.values() if t.status == status]
            if assignee:
                out = [t for t in out if t.assignee == assignee]
            return out

    def claim(self, ticket_id: str, actor: str, *,
              now: str | None = None) -> tuple[HandoffTicket, bool]:
        """领取工单：返回 (工单, 是否已由本人领取)。

        他人已领取 / 已解决 → HandoffConflict（API 409）；不存在 → HandoffNotFound。
        """
        with self._lock:
            ticket = self._tickets.get(ticket_id)
            if ticket is None:
                raise HandoffNotFound(ticket_id)
            if ticket.status == "resolved":
                raise HandoffConflict(f"工单 {ticket_id} 已解决，无法领取")
            if ticket.assignee and ticket.assignee != actor:
                raise HandoffConflict(
                    f"工单 {ticket_id} 已被 {ticket.assignee} 领取"
                )
            if ticket.assignee == actor:
                return ticket, True
            claimed_at = now or _now()
            ticket.assignee = actor
            ticket.claimed_at = claimed_at
            ticket.events.append(
                {"action": "claimed", "actor": actor, "at": claimed_at, "note": ""}
            )
            return ticket, False

    def add_note(self, ticket_id: str, actor: str, note: str, *,
                 now: str | None = None) -> HandoffTicket:
        """追加处理备注（append-only 留痕，不改变工单状态）。"""
        with self._lock:
            ticket = self._tickets.get(ticket_id)
            if ticket is None:
                raise HandoffNotFound(ticket_id)
            ticket.events.append({
                "action": "note", "actor": actor, "at": now or _now(),
                "note": str(note),
            })
            return ticket

    def mark_sla_breaches(self, now=None) -> list[str]:
        """标记「首次观测到已超时」的工单，返回本次新增的 ticket_id 列表。

        幂等：同一工单只返回一次（进程内集合 / Redis SET 去重），调用方据此
        累加 handoff_sla_breach_total，避免每次轮询重复计数。
        """
        with self._lock:
            newly: list[str] = []
            for ticket in self._tickets.values():
                if not ticket.sla_breached(now=now):
                    continue
                if ticket.ticket_id in self._sla_breached:
                    continue
                self._sla_breached.add(ticket.ticket_id)
                newly.append(ticket.ticket_id)
            return newly

    def resolve(self, ticket_id: str, resolution: dict) -> HandoffTicket | None:
        with self._lock:
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
        with self._lock:
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
            ticket.resolved_at = event["resolved_at"]
            if not ticket.assignee:
                # 未领取直接解决：处理人即解决人（留痕不出现空 actor）
                ticket.assignee = resolved_by
                ticket.claimed_at = event["resolved_at"]
            ticket.events.append({
                "action": "resolved", "actor": resolved_by,
                "at": event["resolved_at"], "note": str(resolution.get("note", "")),
            })
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


# 原子 resolve 的 Lua：工单转 resolved + 保存事件 + 追加处理留痕 +
# 移出 pending + 加入 resolved/evolution pending 索引（全部成功或全部不执行）。
# KEYS: 1=handoff:{id} 2=handoff:resolution:{id} 3=handoff:pending
#       4=handoff:resolved 5=handoff:evolution:pending 6=handoff:trail:{id}
# ARGV: 1=新工单 JSON 2=事件 JSON 3=内容指纹 4=幂等键 5=ticket_id
#       6=knowledge_candidate(0/1) 7=留痕事件 JSON
# 返回: {OK|DUPLICATE|CONFLICT|RETRY|NOT_FOUND, 事件 JSON?}
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
-- 并发领取保护：Python 侧快照之后若工单已被他人领取，直接覆盖会丢领取状态
-- （留痕与工单不一致）→ 返回 RETRY，调用方重读最新状态后重试。
local newt = cjson.decode(ARGV[1])
local cur = t['assignee'] or ''
local newa = newt['assignee'] or ''
if cur ~= '' and cur ~= newa then return {'RETRY', cur} end
redis.call('SET', KEYS[1], ARGV[1])
redis.call('SET', KEYS[2], ARGV[2])
redis.call('RPUSH', KEYS[6], ARGV[7])
redis.call('SREM', KEYS[3], ARGV[5])
redis.call('SADD', KEYS[4], ARGV[5])
if ARGV[6] == '1' then redis.call('SADD', KEYS[5], ARGV[5]) end
return {'OK', ARGV[2]}
"""

# 原子领取（P2-3）：读当前工单 → 比较 assignee → 写入领取状态 + 追加留痕。
# 并发两方同时领取时，Lua 内「读-比较-写」不可分割，只有一个能成功；
# 另一方拿到 CONFLICT（附当前 assignee）。
# KEYS: 1=handoff:{id} 2=handoff:trail:{id}
# ARGV: 1=新工单 JSON 2=actor 3=now 4=留痕事件 JSON
# 返回: {OK|ALREADY|CONFLICT|RESOLVED|NOT_FOUND, 当前 assignee}
_CLAIM_LUA = """
local raw = redis.call('GET', KEYS[1])
if not raw then return {'NOT_FOUND', ''} end
local t = cjson.decode(raw)
if t['status'] == 'resolved' then return {'RESOLVED', ''} end
local cur = t['assignee'] or ''
if cur ~= '' and cur ~= ARGV[2] then return {'CONFLICT', cur} end
if cur == ARGV[2] then return {'ALREADY', cur} end
redis.call('SET', KEYS[1], ARGV[1])
redis.call('RPUSH', KEYS[2], ARGV[4])
return {'OK', ARGV[2]}
"""

# 原子追加处理备注：存在性检查 + RPUSH 留痕（append-only，不覆盖工单 JSON）。
# KEYS: 1=handoff:{id} 2=handoff:trail:{id}
# ARGV: 1=留痕事件 JSON
# 返回: {OK|NOT_FOUND}
_NOTE_LUA = """
if redis.call('EXISTS', KEYS[1]) == 0 then return {'NOT_FOUND'} end
redis.call('RPUSH', KEYS[2], ARGV[1])
return {'OK'}
"""


class RedisHandoffBoard:
    """Redis 实现：handoff:{ticket_id} = JSON；list 用 set 索引（pending/resolved）。

    P2-3：处理留痕存 handoff:trail:{id}（RPUSH append-only）；claim/note/resolve
    的状态变更与留痕写入在同一 Lua 内提交（原子，多 Pod 安全）。
    """

    KEY_PREFIX = "handoff:"
    PENDING_SET = "handoff:pending"
    RESOLVED_SET = "handoff:resolved"            # 新增：已解决索引（新工单开始维护，不回扫）
    EVOLUTION_PENDING_SET = "handoff:evolution:pending"  # 新增：人工候选导入队列
    RESOLUTION_KEY_PREFIX = "handoff:resolution:"  # 新增：resolution 事件（幂等重放正本）
    TRAIL_KEY_PREFIX = "handoff:trail:"          # P2-3：处理留痕（append-only 列表）
    SLA_BREACHED_SET = "handoff:sla:breached"    # P2-3：SLA 超时首次观测去重

    durable = True

    def __init__(self, redis):
        self._redis = redis

    def create(self, ticket: HandoffTicket) -> HandoffTicket:
        self._redis.set(self.KEY_PREFIX + ticket.ticket_id,
                        json.dumps(ticket.to_dict(), ensure_ascii=False))
        self._redis.sadd(self.PENDING_SET, ticket.ticket_id)
        return ticket

    def _trail(self, ticket_id: str) -> list[dict]:
        raw = self._redis.lrange(self.TRAIL_KEY_PREFIX + ticket_id, 0, -1)
        events = []
        for item in raw or []:
            if isinstance(item, bytes):
                item = item.decode("utf-8")
            try:
                event = json.loads(item)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    def get(self, ticket_id: str) -> HandoffTicket | None:
        raw = self._redis.get(self.KEY_PREFIX + ticket_id)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        ticket = HandoffTicket(**json.loads(raw))
        trail = self._trail(ticket_id)
        if trail:
            ticket.events = trail  # 留痕正本在 trail 列表；工单 JSON 内的 events 不参与
        return ticket

    def list(self, status: str = "pending", *, assignee: str = "") -> list[HandoffTicket]:
        """pending / resolved / all（合并去重；状态校验在 API 层 422）。

        assignee 非空时只返回该坐席领取的工单（工作台「我的」列表）。
        """
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
            if tick is None:
                continue
            if assignee and tick.assignee != assignee:
                continue
            out.append(tick)
        return out

    def claim(self, ticket_id: str, actor: str, *,
              now: str | None = None) -> tuple[HandoffTicket, bool]:
        """Lua 原子领取：并发两方同时领取只有一个成功（另一方 409）。

        返回 (工单, 是否已由本人领取)；不存在/已解决/他人已领取分别抛
        HandoffNotFound / HandoffConflict。
        """
        ticket = self.get(ticket_id)
        if ticket is None:
            raise HandoffNotFound(ticket_id)
        claimed_at = now or _now()
        new_ticket = dict(ticket.to_dict())
        new_ticket.update({"assignee": actor, "claimed_at": claimed_at})
        # 留痕存 trail 列表（正本），工单 JSON 内 events 不参与，避免双重记录
        new_ticket["events"] = []
        trail_event = {
            "action": "claimed", "actor": actor, "at": claimed_at, "note": "",
        }
        result = self._redis.eval(
            _CLAIM_LUA, 2,
            self.KEY_PREFIX + ticket_id,
            self.TRAIL_KEY_PREFIX + ticket_id,
            json.dumps(new_ticket, ensure_ascii=False),
            actor, claimed_at,
            json.dumps(trail_event, ensure_ascii=False),
        )
        status = result[0].decode() if isinstance(result[0], bytes) else result[0]
        if status == "NOT_FOUND":
            raise HandoffNotFound(ticket_id)
        if status == "RESOLVED":
            raise HandoffConflict(f"工单 {ticket_id} 已解决，无法领取")
        if status == "CONFLICT":
            holder = result[1]
            if isinstance(holder, bytes):
                holder = holder.decode("utf-8")
            raise HandoffConflict(f"工单 {ticket_id} 已被 {holder} 领取")
        updated = self.get(ticket_id)
        if updated is None:  # 极端竞态：Lua 提交后被删除
            raise HandoffNotFound(ticket_id)
        return updated, status == "ALREADY"

    def add_note(self, ticket_id: str, actor: str, note: str, *,
                 now: str | None = None) -> HandoffTicket:
        """Lua 原子追加处理备注（append-only，不改变工单状态）。"""
        event = {"action": "note", "actor": actor, "at": now or _now(),
                 "note": str(note)}
        result = self._redis.eval(
            _NOTE_LUA, 2,
            self.KEY_PREFIX + ticket_id,
            self.TRAIL_KEY_PREFIX + ticket_id,
            json.dumps(event, ensure_ascii=False),
        )
        status = result[0].decode() if isinstance(result[0], bytes) else result[0]
        if status == "NOT_FOUND":
            raise HandoffNotFound(ticket_id)
        updated = self.get(ticket_id)
        if updated is None:
            raise HandoffNotFound(ticket_id)
        return updated

    def mark_sla_breaches(self, now=None) -> list[str]:
        """标记首次观测到超时的工单（SADD 去重），返回本次新增 id 列表。"""
        newly: list[str] = []
        for ticket in self.list("all"):
            if not ticket.sla_breached(now=now):
                continue
            if self._redis.sadd(self.SLA_BREACHED_SET, ticket.ticket_id):
                newly.append(ticket.ticket_id)
        return newly

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
                       *, now: str | None = None,
                       _retry: int = 0) -> tuple[dict, bool]:
        """Lua 原子 resolve：工单/事件/留痕/pending/索引一次提交，防双实例竞态。

        返回 (事件, 是否幂等重放)；不存在/冲突抛 HandoffNotFound/HandoffConflict。
        并发领取与解决交错（Lua 返回 RETRY）时重读最新状态重试（有界）。
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
            "resolved_at": event["resolved_at"],
        })
        if not new_ticket.get("assignee"):
            # 未领取直接解决：处理人即解决人（留痕不出现空 actor）
            new_ticket["assignee"] = resolved_by
            new_ticket["claimed_at"] = event["resolved_at"]
        new_ticket["events"] = []  # 留痕正本在 trail 列表，工单 JSON 不重复存
        trail_event = {
            "action": "resolved", "actor": resolved_by,
            "at": event["resolved_at"], "note": str(resolution.get("note", "")),
        }
        result = self._redis.eval(
            _RESOLVE_LUA, 6,
            self.KEY_PREFIX + ticket_id,
            self.RESOLUTION_KEY_PREFIX + ticket_id,
            self.PENDING_SET,
            self.RESOLVED_SET,
            self.EVOLUTION_PENDING_SET,
            self.TRAIL_KEY_PREFIX + ticket_id,
            json.dumps(new_ticket, ensure_ascii=False),
            json.dumps(event, ensure_ascii=False),
            digest, idem, ticket_id,
            "1" if new_ticket["knowledge_candidate"] else "0",
            json.dumps(trail_event, ensure_ascii=False),
        )
        status = result[0].decode() if isinstance(result[0], bytes) else result[0]
        if status == "NOT_FOUND":  # 竞态：get 后被删除（罕见）
            raise HandoffNotFound(ticket_id)
        if status == "RETRY":
            # 并发领取：重读最新工单（此时 assignee 已非空，不再被解决覆盖）
            if _retry >= 2:
                raise HandoffConflict(f"工单 {ticket_id} 并发状态变更，请重试")
            return self.resolve_atomic(
                ticket_id, resolution, resolved_by, resolution_version,
                now=now, _retry=_retry + 1,
            )
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
    # 修复计划：写结果未知（indeterminate）→ 建议人工按目标标识对账（不记录凭证）；
    # 工具层自生成的请求标识在结果载荷中回写，外层硬超时时可能缺失 → 按需附加
    for w in getattr(agent, "_indeterminate_writes", None) or []:
        target = w.get("order_id") or w.get("application_id") or ""
        action = f"工具结果未知需人工对账: {w.get('tool')}, 目标 {target}"
        request_id = w.get("client_request_id")
        if request_id:
            action += f", 请求标识 {request_id}"
        suggested.append(action)
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
