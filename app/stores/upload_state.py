"""上传断点状态：Redis 实现（生产）+ 进程内实现（仅开发/单测）。

协议（v7 冻结，评审修正后）：
- 会话三态 session_state：accepting | sealed | cancelled（写在 upload:{id} hash）；
- 分片记录（upload:{id}:chunks hash，field=seq → JSON）：
    {state: publishing|ready, token, publishing_at, size, sha256, object_key}
- PUT 原子声明（Lua）：session 必须 accepting；无记录 → 创建 publishing（新 token）；
  已存在 → ready 且 size/sha256 相同 → duplicate；hash 不同 → conflict；
  publishing 且 token 相同 → same_request（同请求重试）；publishing 且超时且
  size/hash 相同 → takeover（token 换新）；publishing 且未超时 → pending（409 稍后重试）；
- finalize（Lua）：session 仍 accepting 且 record 是 publishing 且 token 匹配 → ready；
- seal（Lua 单脚本）：accepting 且无 publishing 且已收够 total 且全部 ready → sealed；
- ready-but-missing 解封（Lua）：sealed 且指定 seq 仍为原 ready 值 → HDEL 该 seq → sealed→accepting；
  客户端重传走全新声明流程（**不预生成 token**）；
- cancel：先封口（accepting→sealed→cancelled）再删对象（PUT 并发时 sealed 拒绝新声明）；
- publishing_at 用 Redis TIME（避免 Pod 时钟漂移）；进程内实现用本地时钟（单进程无碍）。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Optional

from app.config.settings import settings

SESSION_ACCEPTING = "accepting"
SESSION_SEALED = "sealed"
SESSION_CANCELLED = "cancelled"

STATE_PUBLISHING = "publishing"
STATE_READY = "ready"

# Lua 脚本：单次执行完成「会话校验 + 读旧记录 + 分支判定」——无 TOCTOU
_DECLARE_LUA = """
local session_key = KEYS[1]
local chunks_key = KEYS[2]
local seq = ARGV[1]
local token = ARGV[2]
local size = tonumber(ARGV[3])
local sha256 = ARGV[4]
local timeout = tonumber(ARGV[5])
local now = tonumber(redis.call('TIME')[1])

local s = redis.call('HGET', session_key, 'session_state')
if s == false then
    return 'NOSESSION'
end
if s ~= 'accepting' then
    return 'CLOSED'
end
local cur = redis.call('HGET', chunks_key, seq)
if cur == false then
    local meta = cjson.encode({
        state = 'publishing', token = token, publishing_at = now,
        size = size, sha256 = sha256, object_key = ''
    })
    redis.call('HSET', chunks_key, seq, meta)
    redis.call('EXPIRE', chunks_key, ARGV[6])
    redis.call('EXPIRE', session_key, ARGV[6])
    return 'NEW'
end
local m = cjson.decode(cur)
if m.state == 'ready' then
    if tonumber(m.size) == size and m.sha256 == sha256 then
        return 'DUPLICATE'
    end
    return 'CONFLICT'
end
-- publishing
if m.token == token then
    return 'SAME_REQUEST'
end
if tonumber(m.size) ~= size or m.sha256 ~= sha256 then
    return 'CONFLICT'
end
if (now - tonumber(m.publishing_at)) >= timeout then
    -- 超时接管：旧 token 此后不能 finalize（token 已换）
    m.token = token
    m.publishing_at = now
    redis.call('HSET', chunks_key, seq, cjson.encode(m))
    return 'TAKEOVER'
end
return 'PENDING'
"""

_FINALIZE_LUA = """
local session_key = KEYS[1]
local chunks_key = KEYS[2]
local seq = ARGV[1]
local token = ARGV[2]
local object_key = ARGV[3]
local size = ARGV[4]
local sha256 = ARGV[5]
local ttl = ARGV[6]

local s = redis.call('HGET', session_key, 'session_state')
if s == false then return 'NOSESSION' end
if s ~= 'accepting' then return 'CLOSED' end
local cur = redis.call('HGET', chunks_key, seq)
if cur == false then return 'MISSING' end
local m = cjson.decode(cur)
if m.state ~= 'publishing' then return 'NOT_PUBLISHING' end
if m.token ~= token then return 'TOKEN_MISMATCH' end
m.state = 'ready'
if object_key ~= '' then m.object_key = object_key end
m.size = size
m.sha256 = sha256
redis.call('HSET', chunks_key, seq, cjson.encode(m))
redis.call('EXPIRE', chunks_key, ttl)
return 'READY'
"""

_SEAL_LUA = """
local session_key = KEYS[1]
local chunks_key = KEYS[2]
local total = tonumber(ARGV[1])
local ttl = ARGV[2]

local s = redis.call('HGET', session_key, 'session_state')
if s == false then return 'NOSESSION' end
if s ~= 'accepting' then return 'NOT_ACCEPTING' end
local entries = redis.call('HGETALL', chunks_key)
local count = 0
for i = 1, #entries, 2 do
    local m = cjson.decode(entries[i + 1])
    if m.state ~= 'ready' then return 'HAS_PUBLISHING' end
    count = count + 1
end
if count ~= total then return 'INCOMPLETE' end
redis.call('HSET', session_key, 'session_state', 'sealed')
redis.call('EXPIRE', session_key, ttl)
return 'SEALED'
"""

_UNSEAL_LUA = """
local session_key = KEYS[1]
local chunks_key = KEYS[2]
local seq = ARGV[1]
local size = ARGV[2]
local sha256 = ARGV[3]
local ttl = ARGV[4]

local s = redis.call('HGET', session_key, 'session_state')
if s == false then return 'NOSESSION' end
if s ~= 'sealed' then return 'NOT_SEALED' end
local cur = redis.call('HGET', chunks_key, seq)
if cur == false then return 'MISSING' end
local m = cjson.decode(cur)
if m.state ~= 'ready' then return 'NOT_READY' end
if tonumber(m.size) ~= tonumber(size) or m.sha256 ~= sha256 then return 'META_MISMATCH' end
redis.call('HDEL', chunks_key, seq)
redis.call('HSET', session_key, 'session_state', 'accepting')
redis.call('EXPIRE', chunks_key, ttl)
redis.call('EXPIRE', session_key, ttl)
return 'UNSEALED'
"""


@dataclass
class UploadSession:
    """一个断点上传会话的元信息。"""

    upload_id: str
    filename: str
    size_bytes: int
    chunk_size: int
    total_chunks: int
    uploader: str = ""
    content_type: str = ""
    format: str = ""
    sha256: str = ""  # 客户端可选声明；complete 时服务端计算比对
    session_state: str = SESSION_ACCEPTING

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "UploadSession":
        return cls(**data)


class UploadStateError(ValueError):
    """会话不存在/参数非法（调用方映射 422/404 语义）。"""


class UploadClosed(UploadStateError):
    """会话非 accepting（sealed/cancelled）：PUT 或 finalize 被拒（409 UPLOAD_CLOSED）。"""


class UploadConflict(UploadStateError):
    """同 seq 重传不同内容（分片被覆盖过）——保留原分片（409）。"""


class RedisUploadStateStore:
    """Redis 断点状态（生产）：会话 hash + 分片 hash + Lua 原子协议。"""

    KEY_PREFIX = "upload:"

    def __init__(self, redis, ttl_seconds: Optional[int] = None,
                 publish_timeout: Optional[int] = None):
        self._redis = redis
        self._ttl = int(ttl_seconds if ttl_seconds is not None else settings.kb_upload_session_ttl)
        self._publish_timeout = int(
            publish_timeout if publish_timeout is not None else settings.kb_upload_publish_timeout
        )

    def _key(self, upload_id: str) -> str:
        return f"{self.KEY_PREFIX}{upload_id}"

    def _chunks_key(self, upload_id: str) -> str:
        return f"{self.KEY_PREFIX}{upload_id}:chunks"

    # ---------- 会话 ----------
    def create(self, session: UploadSession) -> None:
        """创建会话（幂等：同 upload_id 已存在 → 原样保留，不覆盖）。"""
        key = self._key(session.upload_id)
        try:
            if self._redis.exists(key):
                return
            self._redis.hset(key, mapping=dict(
                session_state=session.session_state,
                meta=json.dumps(session.to_dict(), ensure_ascii=False),
            ))
            self._redis.expire(key, self._ttl)
        except Exception as e:  # noqa: BLE001 —— 断点语义依赖 Redis：明确失败
            raise UploadStateError(f"上传会话创建失败（Redis 不可用）: {e}") from e

    def get(self, upload_id: str) -> Optional[UploadSession]:
        try:
            raw = self._redis.hget(self._key(upload_id), "meta")
            state = self._redis.hget(self._key(upload_id), "session_state")
        except Exception as e:  # noqa: BLE001
            raise UploadStateError(f"上传会话读取失败（Redis 不可用）: {e}") from e
        if raw is None:
            return None
        try:
            session = UploadSession.from_dict(json.loads(raw))
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            raise UploadStateError(f"上传会话数据损坏: {e}") from e
        if state is not None:
            session.session_state = (
                state.decode() if isinstance(state, bytes) else str(state)
            )
        return session

    def session_state(self, upload_id: str) -> str:
        s = self.get(upload_id)
        return s.session_state if s is not None else ""

    # ---------- 分片声明（Lua 原子） ----------
    def declare_chunk(self, upload_id: str, seq: int, size: int, sha256: str,
                      token: str) -> str:
        """声明一片：NEW | DUPLICATE | CONFLICT | PENDING | TAKEOVER | SAME_REQUEST |
        CLOSED | NOSESSION。对象尚未发布前调用。"""
        try:
            out = self._redis.eval(
                _DECLARE_LUA, 2,
                self._key(upload_id), self._chunks_key(upload_id),
                str(seq), token, size, sha256,
                self._publish_timeout, self._ttl,
            )
        except Exception as e:  # noqa: BLE001
            raise UploadStateError(f"分片声明失败（Redis 不可用）: {e}") from e
        return out.decode() if isinstance(out, bytes) else str(out)

    def _mark_extra(self, code: str, upload_id: str, seq: int) -> None:
        """声明后动作由调用方按 code 决定；这里只暴露语义化异常。"""
        if code in ("CONFLICT",):
            raise UploadConflict(f"分片 {seq} 已存在且内容不一致")
        if code == "CLOSED":
            raise UploadClosed(f"会话已关闭（sealed/cancelled），拒绝分片 {seq}")

    def finalize_chunk(self, upload_id: str, seq: int, token: str,
                       size: int, sha256: str, object_key: str) -> str:
        """发布完成：READY | CLOSED | MISSING | NOT_PUBLISHING | TOKEN_MISMATCH | NOSESSION。"""
        try:
            out = self._redis.eval(
                _FINALIZE_LUA, 2,
                self._key(upload_id), self._chunks_key(upload_id),
                str(seq), token, object_key, size, sha256, self._ttl,
            )
        except Exception as e:  # noqa: BLE001
            raise UploadStateError(f"分片发布确认失败（Redis 不可用）: {e}") from e
        return out.decode() if isinstance(out, bytes) else str(out)

    def seal_if_ready(self, upload_id: str) -> str:
        """封口（单 Lua 全校验）：SEALED | NOT_ACCEPTING | HAS_PUBLISHING | INCOMPLETE | NOSESSION。"""
        session = self.get(upload_id)
        if session is None:
            return "NOSESSION"
        try:
            out = self._redis.eval(
                _SEAL_LUA, 2,
                self._key(upload_id), self._chunks_key(upload_id),
                session.total_chunks, self._ttl,
            )
        except Exception as e:  # noqa: BLE001
            raise UploadStateError(f"会话封口失败（Redis 不可用）: {e}") from e
        return out.decode() if isinstance(out, bytes) else str(out)

    def unseal_and_drop(self, upload_id: str, seq: int, size: int, sha256: str) -> str:
        """ready-but-missing 修复：UNSEALED | NOT_SEALED | MISSING | NOT_READY |
        META_MISMATCH | NOSESSION（解封后客户端重传走全新声明，不预生成 token）。"""
        try:
            out = self._redis.eval(
                _UNSEAL_LUA, 2,
                self._key(upload_id), self._chunks_key(upload_id),
                str(seq), size, sha256, self._ttl,
            )
        except Exception as e:  # noqa: BLE001
            raise UploadStateError(f"分片解封失败（Redis 不可用）: {e}") from e
        return out.decode() if isinstance(out, bytes) else str(out)

    def mark_closed(self, upload_id: str, state: str = SESSION_SEALED) -> bool:
        """封口/取消切换（无并发 PUT 时直接用；并发安全靠 declare/finalize 的
        会话校验兜底）。返回是否已封口。"""
        try:
            cur = self.session_state(upload_id)
            if cur not in (SESSION_ACCEPTING, SESSION_SEALED):
                return False
            self._redis.hset(self._key(upload_id), "session_state", state)
            return True
        except Exception:  # noqa: BLE001
            return False

    def drop_chunk_record(self, upload_id: str, seq: int, token: str) -> bool:
        """声明后发布失败的撤回：仅同 token compare-and-delete（Lua 快路径用
        HSETNX 语义简化：值里 token 匹配才 HDEL）。"""
        try:
            out = self._redis.eval(
                "local m = redis.call('HGET', KEYS[2], ARGV[1]);"
                "if m == false then return 0 end;"
                "local d = cjson.decode(m);"
                "if d.token ~= ARGV[2] or d.state ~= 'publishing' then return 0 end;"
                "redis.call('HDEL', KEYS[2], ARGV[1]); return 1",
                2, self._key(upload_id), self._chunks_key(upload_id),
                str(seq), token,
            )
            return bool(out)
        except Exception:  # noqa: BLE001 —— 撤回失败：残留 publishing 由接管/超时兜底
            return False

    # ---------- 查询/清理 ----------
    def received(self, upload_id: str) -> dict[int, dict]:
        """已收分片（含 publishing；complete 只统计 ready——由调用方过滤）。"""
        try:
            raw = self._redis.hgetall(self._chunks_key(upload_id))
        except Exception as e:  # noqa: BLE001
            raise UploadStateError(f"分片状态读取失败（Redis 不可用）: {e}") from e
        out: dict[int, dict] = {}
        for k, v in raw.items():
            try:
                out[int(k)] = json.loads(v)
            except (ValueError, json.JSONDecodeError, TypeError):
                continue
        return out

    def ready_count(self, upload_id: str) -> int:
        return sum(1 for m in self.received(upload_id).values() if m.get("state") == STATE_READY)

    def delete(self, upload_id: str) -> None:
        try:
            self._redis.delete(self._key(upload_id), self._chunks_key(upload_id))
        except Exception:  # noqa: BLE001 —— 清理失败靠 TTL 兜底
            return


class InProcessUploadStateStore:
    """进程内实现（仅开发/单测）：与 Redis 版接口一致，单进程语义。

    … 不实现 Lua 原子性：线程锁保证本进程一致性；token/publishing_at 语义保留。
    """

    def __init__(self, ttl_seconds: Optional[int] = None,
                 publish_timeout: Optional[int] = None):
        self._ttl = int(ttl_seconds if ttl_seconds is not None else settings.kb_upload_session_ttl)
        self._publish_timeout = int(
            publish_timeout if publish_timeout is not None else settings.kb_upload_publish_timeout
        )
        self._sessions: dict[str, UploadSession] = {}
        self._chunks: dict[str, dict[int, dict]] = {}
        self._lock = threading.Lock()

    def create(self, session: UploadSession) -> None:
        with self._lock:
            if session.upload_id in self._sessions:
                return
            self._sessions[session.upload_id] = session
            self._chunks.setdefault(session.upload_id, {})

    def get(self, upload_id: str) -> Optional[UploadSession]:
        with self._lock:
            return self._sessions.get(upload_id)

    def session_state(self, upload_id: str) -> str:
        s = self.get(upload_id)
        return s.session_state if s is not None else ""

    def declare_chunk(self, upload_id: str, seq: int, size: int, sha256: str,
                      token: str) -> str:
        with self._lock:
            session = self._sessions.get(upload_id)
            if session is None:
                return "NOSESSION"
            if session.session_state != SESSION_ACCEPTING:
                return "CLOSED"
            chunks = self._chunks[upload_id]
            cur = chunks.get(seq)
            if cur is None:
                chunks[seq] = {
                    "state": STATE_PUBLISHING, "token": token,
                    "publishing_at": time.time(),
                    "size": size, "sha256": sha256, "object_key": "",
                }
                return "NEW"
            if cur["state"] == STATE_READY:
                if cur["size"] == size and cur["sha256"] == sha256:
                    return "DUPLICATE"
                return "CONFLICT"
            if cur["token"] == token:
                return "SAME_REQUEST"
            if cur["size"] != size or cur["sha256"] != sha256:
                return "CONFLICT"
            if time.time() - cur["publishing_at"] >= self._publish_timeout:
                cur["token"] = token
                cur["publishing_at"] = time.time()
                return "TAKEOVER"
            return "PENDING"

    def finalize_chunk(self, upload_id: str, seq: int, token: str,
                       size: int, sha256: str, object_key: str) -> str:
        with self._lock:
            session = self._sessions.get(upload_id)
            if session is None:
                return "NOSESSION"
            if session.session_state != SESSION_ACCEPTING:
                return "CLOSED"
            cur = self._chunks[upload_id].get(seq)
            if cur is None:
                return "MISSING"
            if cur["state"] != STATE_PUBLISHING:
                return "NOT_PUBLISHING"
            if cur["token"] != token:
                return "TOKEN_MISMATCH"
            cur["state"] = STATE_READY
            cur["object_key"] = object_key or cur["object_key"]
            cur["size"] = size
            cur["sha256"] = sha256
            return "READY"

    def seal_if_ready(self, upload_id: str) -> str:
        with self._lock:
            session = self._sessions.get(upload_id)
            if session is None:
                return "NOSESSION"
            if session.session_state != SESSION_ACCEPTING:
                return "NOT_ACCEPTING"
            chunks = self._chunks[upload_id]
            # 与 Redis 版 Lua 顺序一致：先检出 publishing，再校验总数（INCOMPLETE 可能同时双因）
            for m in chunks.values():
                if m["state"] != STATE_READY:
                    return "HAS_PUBLISHING"
            if len(chunks) != session.total_chunks:
                return "INCOMPLETE"
            session.session_state = SESSION_SEALED
            return "SEALED"

    def unseal_and_drop(self, upload_id: str, seq: int, size: int, sha256: str) -> str:
        with self._lock:
            session = self._sessions.get(upload_id)
            if session is None:
                return "NOSESSION"
            if session.session_state != SESSION_SEALED:
                return "NOT_SEALED"
            cur = self._chunks[upload_id].get(seq)
            if cur is None:
                return "MISSING"
            if cur["state"] != STATE_READY:
                return "NOT_READY"
            if cur["size"] != size or cur["sha256"] != sha256:
                return "META_MISMATCH"
            del self._chunks[upload_id][seq]
            session.session_state = SESSION_ACCEPTING
            return "UNSEALED"

    def mark_closed(self, upload_id: str, state: str = SESSION_SEALED) -> bool:
        with self._lock:
            session = self._sessions.get(upload_id)
            if session is None or session.session_state == SESSION_CANCELLED:
                return False
            session.session_state = state
            return True

    def drop_chunk_record(self, upload_id: str, seq: int, token: str) -> bool:
        with self._lock:
            cur = self._chunks.get(upload_id, {}).get(seq)
            if cur is None or cur["state"] != STATE_PUBLISHING or cur["token"] != token:
                return False
            del self._chunks[upload_id][seq]
            return True

    def received(self, upload_id: str) -> dict[int, dict]:
        with self._lock:
            return {k: dict(v) for k, v in self._chunks.get(upload_id, {}).items()}

    def ready_count(self, upload_id: str) -> int:
        with self._lock:
            return sum(
                1 for m in self._chunks.get(upload_id, {}).values()
                if m.get("state") == STATE_READY
            )

    def delete(self, upload_id: str) -> None:
        with self._lock:
            self._sessions.pop(upload_id, None)
            self._chunks.pop(upload_id, None)


def build_upload_state_store(redis=None):
    """工厂：Redis 可用 → Redis 实现；否则进程内（仅开发/单测）。"""
    if redis is not None:
        return RedisUploadStateStore(redis)
    return InProcessUploadStateStore()
