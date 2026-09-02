"""阶段八 单测：SQL 正本（消息追加/CAS/热缓存/outbox）+ ES 后端（alias/kNN）。"""

from __future__ import annotations

import math

import fakeredis
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.pool import StaticPool

from app.agent.rag.backends.es_backend import ESBackend
from app.agent.rag.backends.base import VectorBackend
from app.agent.rag.chunker import Chunk
from app.stores.base import SessionConflictError, SessionState
from app.stores.sql.memory_store import SqlLTMStore
from app.stores.sql.outbox import ensure_message_index, sync_outbox_to_es
from app.stores.sql.session_store import SqlSessionStore
from app.stores.sql.schema import chat_messages, metadata, outbox_rows, sessions


def _engine():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    metadata.create_all(engine)
    return engine


def _state(version=0, summary=None, messages=None, **overrides):
    base = dict(
        session_id="s-1", user_id="u1", summary=summary,
        messages=messages if messages is not None else [],
        version=version, updated_at="",
    )
    base.update(overrides)
    return SessionState(**base)


def _msg(role, content, **extra):
    return {"role": role, "content": content, **extra}


# ============================================================
# SqlSessionStore：CAS / 追加 / 删除 / iter_all
# ============================================================
def test_sql_session_roundtrip_and_cas():
    store = SqlSessionStore(_engine())
    saved = store.save("u1", "s1", _state(messages=[_msg("user", "hi")]))
    assert saved.version == 1
    loaded = store.load("u1", "s1")
    assert loaded.messages == [_msg("user", "hi")]
    assert loaded.user_id == "u1"

    with pytest.raises(SessionConflictError):
        store.save("u1", "s1", _state(version=0))
    ok = store.save("u1", "s1", _state(version=1, messages=loaded.messages))
    assert ok.version == 2


def test_sql_session_append_only_not_lost_on_compression():
    """压缩只裁窗口：save 带 new_messages 时，历史消息行式保留。"""
    store = SqlSessionStore(_engine())
    store.save("u1", "s1", _state(messages=[_msg("user", "旧")]))
    # 模拟压缩后：state.messages 只剩窗口内 1 条，但新消息 2 条
    store.save(
        "u1", "s1",
        _state(version=1, summary="压缩摘要",
               messages=[_msg("assistant", "窗口内")]),
        new_messages=[_msg("assistant", "窗口内"), _msg("user", "新问")],
    )
    loaded = store.load("u1", "s1")
    assert len(loaded.messages) == 3  # 正本永续：旧+窗口内+新问
    assert loaded.messages[0]["content"] == "旧"
    assert loaded.messages[-1]["content"] == "新问"
    assert loaded.summary == "压缩摘要"


def test_sql_session_delete_and_iter_all():
    store = SqlSessionStore(_engine())
    store.save("u1", "s1", _state(messages=[_msg("user", "a")]))
    store.save("u2", "", _state(session_id="s2", messages=[_msg("user", "b")]))
    assert sorted(store.iter_all()) == [("u1", "s1"), ("u2", "session")]
    store.delete("u1", "s1")
    assert store.load("u1", "s1") is None
    assert store.iter_all() == [("u2", "session")]  # u1 的会话已删


# ============================================================
# Redis 热缓存（保留并发角色：write-through）
# ============================================================
def test_sql_session_hot_cache_write_through():
    engine = _engine()
    redis = fakeredis.FakeRedis(server=fakeredis.FakeServer())
    store = SqlSessionStore(engine, redis=redis, hot_ttl=60)
    store.save("u1", "s1", _state(messages=[_msg("user", "hi")]))
    assert redis.exists("session:u1/s1") == 1  # 保存即写缓存

    # 直连库删除后，缓存仍可命中（热读）；缓存失效后回源 MySQL
    with engine.begin() as conn:
        conn.execute(chat_messages.delete())
        conn.execute(sessions.delete())
    cached = store.load("u1", "s1")
    assert cached is not None and cached.messages == [_msg("user", "hi")]
    redis.delete("session:u1/s1")
    assert store.load("u1", "s1") is None  # 缓存失效回源：正本已删 → 无会话


# ============================================================
# SqlLTMStore
# ============================================================
def test_sql_ltm_store_roundtrip_and_replace():
    store = SqlLTMStore(_engine())
    store.save("u1", {"facts": [{"content": "住深圳", "category": "preference",
                                 "created_at": "", "source_session": ""}],
                      "interaction_summaries": [{"summary": "老客", "timestamp": ""}]})
    payload = store.load("u1")
    assert payload["facts"][0]["content"] == "住深圳"
    assert payload["interaction_summaries"][0]["summary"] == "老客"
    # 全量替换：再次 save 只保留新内容（与 LongTermMemory 内存态一致）
    store.save("u1", {"facts": [], "interaction_summaries": []})
    assert store.load("u1")["facts"] == []


def test_long_term_memory_uses_sql_store():
    from app.agent.memory.long_term import LongTermMemory

    engine = _engine()
    ltm = LongTermMemory(user_id="u1", memory_dir="/never", store=SqlLTMStore(engine))
    ltm.add_interaction_summary("偏好顺丰")
    ltm.save()
    reloaded = LongTermMemory(user_id="u1", memory_dir="/never", store=SqlLTMStore(engine))
    reloaded.load()
    assert reloaded.interaction_summaries[0]["summary"] == "偏好顺丰"
    assert reloaded.memory_path.exists() is False  # 行式化后不再写文件


# ============================================================
# Outbox → ES
# ============================================================
class _ESish:
    """Outbox 使用的 ES 客户替换身（bulk + indices.exists/create）。"""

    def __init__(self, fail=False):
        self.indexes = set()
        self.bulk_calls = []
        self.fail = fail

    def __getattr__(self, name):
        if name == "bulk":
            def bulk(operations=None, index=None, **kwargs):
                self.bulk_calls.append((index, operations))
                self.indexes.add(index)
                if self.fail:
                    return {"errors": True, "items": [
                        {"index": {"status": 500, "error": {"reason": "boom"}}}
                        for _ in range(len(operations) // 2)
                    ]}
                return {"errors": False, "items": [{"index": {}} for _ in range(len(operations) // 2)]}
            return bulk
        if name == "indices":
            class _Idx:
                def __init__(self, owner):
                    self._o = owner

                def exists(self, index):
                    return index in self._o.indexes

                def create(self, index, **kwargs):
                    self._o.indexes.add(index)
                    return {"acknowledged": True}

            return _Idx(self)
        raise AttributeError(name)


def test_outbox_flow_to_es():
    engine = _engine()
    store = SqlSessionStore(engine, outbox_enabled=True)
    store.save("u1", "s1", _state(messages=[_msg("user", "你好")]))
    with engine.connect() as conn:
        rows = conn.execute(outbox_rows.select()).mappings().all()
    assert len(rows) == 1  # 消息与 outbox 同事务

    es = _ESish()
    ensure_message_index(es, "ecom-messages")
    assert "ecom-messages" in es.indexes
    synced = sync_outbox_to_es(engine, es, "ecom-messages")
    assert synced == 1
    assert es.bulk_calls[0][0] == "ecom-messages"
    doc = es.bulk_calls[0][1][1]
    assert doc["content"] == "你好"
    assert doc["session_key"] == "u1/s1"
    assert doc["user_id"] == "u1"  # 归属围栏字段（读取端点过滤用）
    assert sync_outbox_to_es(engine, es, "ecom-messages") == 0  # 幂等：已同步


# ============================================================
# /v1/messages/search（阶段八读取端）
# ============================================================
class _MsgSearchES:
    """消息索引替身：term 过滤 + match 子串 + ts 倒序。"""

    def __init__(self, docs=None):
        self.docs = docs or []

    def search(self, index=None, query=None, sort=None, size=10, source=None, **kwargs):
        b = (query or {}).get("bool", {})
        must = b.get("must", [])
        filters = b.get("filter", [])
        term = filters[0].get("term", {}) if filters else {}
        user_filter = term.get("user_id") if "user_id" in term else None
        match_text = must[0].get("match", {}).get("content", "") if must else ""
        hits = []
        for d in self.docs:
            if user_filter and d.get("user_id") != user_filter:
                continue
            if match_text and match_text not in d.get("content", ""):
                continue
            hits.append(d)
        hits.sort(key=lambda d: d.get("ts", ""), reverse=True)
        return {"hits": {"hits": [
            {"_id": f"{d['session_key']}:{d['seq']}", "_source": d} for d in hits[:size]
        ]}}


def test_message_search_endpoint_owner_scoped(monkeypatch):
    from fastapi.testclient import TestClient
    import app.server.main as main_mod
    from test_server_api import _FakeComponents, _ScriptedAgent

    docs = [
        {"session_key": "u1/trip", "user_id": "u1", "seq": 1, "role": "user",
         "content": "我的订单到哪了", "ts": "2026-08-28T10:00:01"},
        {"session_key": "u1/trip", "user_id": "u1", "seq": 2, "role": "assistant",
         "content": "订单已发出", "ts": "2026-08-28T10:00:05"},
        {"session_key": "u2/other", "user_id": "u2", "seq": 1, "role": "user",
         "content": "我的订单到哪了", "ts": "2026-08-28T10:00:02"},
    ]
    msges = _MsgSearchES(docs)

    comps = _FakeComponents()
    comps.es_client = msges
    comps.message_index = "ecom-messages"
    monkeypatch.setattr(main_mod, "build_pod_components", lambda: comps)
    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda uid, sid="", comps=None, **_k: _ScriptedAgent(uid, sid),
    )
    with TestClient(main_mod.create_app()) as client:
        resp = client.get("/v1/messages/search", params={
            "user_id": "u1", "q": "订单", "session_id": "trip",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["degraded"] is False
        assert len(data["hits"]) == 2  # 归属围栏：只搜到 u1 自己的
        assert all(h["session_id"] == "trip" for h in data["hits"])
        assert data["hits"][0]["ts"] > data["hits"][1]["ts"]  # ts 倒序


def test_message_search_degrades_without_es(monkeypatch):
    from fastapi.testclient import TestClient
    import app.server.main as main_mod
    from test_server_api import _FakeComponents

    monkeypatch.setattr(main_mod, "build_pod_components", lambda: _FakeComponents())
    with TestClient(main_mod.create_app()) as client:
        resp = client.get("/v1/messages/search", params={"user_id": "u1", "q": "订单"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["degraded"] is True and data["hits"] == []


# ============================================================
# ESBackend（迷你假 ES 内存实现）
# ============================================================
def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class _MiniES:
    """支持 ESBackend 调用面的内存实现：create/refresh/update_aliases/
    get_alias/get_mapping/search(knn)/bulk/count。"""

    def __init__(self, dim=8):
        self._indices: dict[str, dict] = {}
        self._aliases: dict[str, str] = {}
        self._dim = dim

    # ---- indices 命名空间 ----
    @property
    def indices(self):  # noqa: A003
        return self

    def create(self, index, settings=None, mappings=None):
        self._indices[index] = {
            "meta": ((mappings or {}).get("_meta") or {}),
            "docs": {},
            "vectors": {},
        }
        return {"acknowledged": True}

    def refresh(self, index):
        return {"_shards": {"successful": 1}}

    def exists(self, index):
        return index in self._indices

    def get_alias(self, name):
        targets = {i for i, a in self._aliases.items() if a == name}
        if not targets:
            raise _NotFoundError
        return {i: {"aliases": {name: {}}} for i in targets}

    def get_mapping(self, index):
        return {index: {"mappings": {"_meta": self._indices[index]["meta"]}}}

    def update_aliases(self, actions=None):
        for action in actions:
            if "remove" in action:
                self._aliases.pop(action["remove"]["index"], None)
            elif "add" in action:
                self._aliases[action["add"]["index"]] = action["add"]["alias"]
        return {"acknowledged": True}

    # ---- 顶层 ----
    def bulk(self, operations=None, index=None, refresh=None):
        store = self._indices[index]
        i = 0
        while i < len(operations):
            op = operations[i]
            doc = operations[i + 1]
            doc_id = list(op.values())[0]["_id"]
            store["docs"][doc_id] = doc
            store["vectors"][doc_id] = doc.pop("vector", [])
            i += 2
        return {"errors": False, "items": []}

    @staticmethod
    def _rrf_merge(ranked_lists, size, rank_constant=60):
        fused = {}
        for ranked in ranked_lists:
            for pos, doc_id in enumerate(ranked, start=1):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rank_constant + pos)
        return sorted(fused, key=lambda d: -fused[d])[:size]

    def search(self, index=None, knn=None, size=10, source=None, query=None, sort=None,
               **kwargs):
        store = self._indices[index]
        if knn is not None and query:  # ES 原生 hybrid：BM25(via match) + kNN + RRF
            query_text = query.get("match", {}).get("text", "")
            vec = knn["query_vector"]
            knn_rank = sorted(
                store["docs"], key=lambda d: -(_cosine(vec, store["vectors"].get(d, []))),
            )
            bm25_rank = sorted(
                store["docs"],
                key=lambda d: -(store["docs"][d].get("text", "").count(query_text)),
            )
            merged = self._rrf_merge([knn_rank, bm25_rank], size)
            return {"hits": {"hits": [
                {"_id": d, "_score": 0.0,
                 "_source": {f: store["docs"][d].get(f, "") for f in
                             ("chunk_id", "doc", "section", "text", "source_path",
                              "provenance", "owner", "parent_text")}}
                for d in merged
            ]}}
        if knn is None:  # match_all（chunks() 用）
            docs = sorted(store["docs"])
            return {"hits": {"hits": [
                {"_id": d, "_score": 0.0,
                 "_source": {f: store["docs"][d].get(f, "") for f in
                             ("chunk_id", "doc", "section", "text", "source_path",
                              "provenance", "owner", "parent_text")}}
                for d in docs[:size]
            ]}}
        query_vector = knn["query_vector"]
        k = knn["k"]
        scored = []
        for doc_id, doc in store["docs"].items():
            vec = store["vectors"].get(doc_id) or []
            if not vec:
                continue
            sim = _cosine(query_vector, vec)
            scored.append((sim, doc_id, doc))
        scored.sort(key=lambda x: -x[0])
        hits = []
        for sim, doc_id, doc in scored[:k]:
            hits.append({
                "_id": doc_id, "_score": sim,
                "_source": {f: doc.get(f, "") for f in
                            ("chunk_id", "doc", "section", "text", "source_path",
                             "provenance", "owner", "parent_text")},
            })
        return {"hits": {"hits": hits}}

    def count(self, index=None):
        return {"count": len(self._indices[index]["docs"])}


class _NotFoundError(Exception):
    pass


def _backed_chunks():
    c1 = Chunk(chunk_id="c1", doc="退货", section="七天", text="七天无理由可退",
               source_path="policy.md", parent_text="章节全文")
    c2 = Chunk(chunk_id="c2", doc="配送", section="偏远", text="新疆不包邮",
               source_path="ship.md")
    return [c1, c2]


def test_es_backend_upsert_search_activate():
    es = _MiniES()
    backend = ESBackend(es, index_prefix="ecom")
    chunks = _backed_chunks()
    vectors = [[1.0, 0, 0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0, 0, 0]]
    backend.upsert(chunks, vectors, "fake-model", index_name="ecom-kb-g1")

    # 激活前 alias 不存在（验证通过前不生效）
    with pytest.raises(Exception):
        es.get_alias("ecom-kb-active")
    hits = backend.search(vectors[0], top_k=1)
    assert hits[0].chunk.chunk_id == "c1"  # pending 索引直接查

    backend.activate()
    assert es._aliases.get("ecom-kb-g1") == "ecom-kb-active"
    # 新实例经 alias 解析 + 模型校验
    runtime = ESBackend(es, index_prefix="ecom")
    runtime.load()
    assert runtime.expected_embedding_model() == "fake-model"
    assert runtime.size() == 2
    assert runtime.chunks()[0].parent_text == "章节全文"
    assert VectorBackend is not None  # 协议标记


def test_es_backend_verify_probe_matches_index_service_semantics():
    """index_service.build 流程：upsert → verify(load+model+size+probe) → activate。"""
    es = _MiniES()
    backend = ESBackend(es, index_name="ecom-kb-g2", index_prefix="ecom")
    chunks = _backed_chunks()
    vectors = [[1.0, 0, 0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0, 0, 0]]
    backend.upsert(chunks, vectors, "fake-model")
    backend.load()
    assert backend.expected_embedding_model() == "fake-model"
    assert backend.size() == 2
    probe = vectors[0]
    top1 = backend.search(probe, top_k=1)[0]
    assert top1.chunk.chunk_id == "c1"
    backend.activate()


class _RecordingES(_MiniES):
    def __init__(self):
        super().__init__(dim=8)
        self.search_calls = []

    def search(self, *args, **kwargs):
        self.search_calls.append(kwargs)
        return super().search(*args, **kwargs)


def test_es_native_hybrid_payload_and_retriever():
    from app.agent.rag.hybrid import ESHybridRetriever

    es = _RecordingES()
    backend = ESBackend(es, index_prefix="ecom")
    chunks = _backed_chunks()
    vectors = [[1.0, 0, 0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0, 0, 0]]
    backend.upsert(chunks, vectors, "fake-model", index_name="ecom-kb-h1")

    # 一条查询：match + knn + rank.rrf（窗口=recall_k，k=60 与 Python RRF 同常数）；
    # hybrid_search 返回 RRF 候选窗口（size=recall_k），top_k 截断交由调用方精排后执行
    hits = backend.hybrid_search("七天", vectors[0], top_k=2, recall_k=5)
    call = es.search_calls[-1]
    assert call["size"] == 5  # 候选窗口：精排前至少取 recall_k 条，不能只回 top_k
    assert call["rank"] == {"rrf": {"window_size": 5, "rank_constant": 60}}
    assert call["knn"]["k"] == 5
    assert call["query"] == {"match": {"text": "七天"}}

    # ESHybridRetriever：嵌入 → 原生混合查询 → （可选）rerank
    class _Embed:
        def encode_one(self, text, timeout=None):
            return vectors[0]

    hybrid = ESHybridRetriever(_Embed(), backend, recall_k=5)
    hybrid.load()
    results = hybrid.search("七天", top_k=2)
    assert len(results) == 2


def test_es_backend_missing_alias_raises_filenotfound():
    es = _MiniES()
    backend = ESBackend(es, index_prefix="ecom")
    with pytest.raises(FileNotFoundError):
        backend.load()


# ============================================================
# Review 修复回归（Bug A/B + 4 小问题）
# ============================================================
def test_get_engine_builds_when_db_url_configured(tmp_path, reset_settings):
    """Bug A：初始值 None ≠ _UNSET 曾导致 db_url 永远不生效——此探针防回归。"""
    from app.config.settings import settings
    from app.stores.sql import engine as eng

    eng.reset_engine()
    settings.db_url = f"sqlite:///{tmp_path / 'ecom.sqlite'}"
    try:
        engine_built = eng.get_engine()
        assert engine_built is not None, "get_engine() 应按 db_url 构建引擎"
    finally:
        eng.reset_engine()


def test_elasticsearch_client_api_guard():
    """Bug B：9.x 客户端把 search(source) 改成布尔开关——必须钉 8.x 并守住签名。"""
    import inspect
    from importlib.metadata import version

    from elasticsearch import Elasticsearch

    major = int(version("elasticsearch").split(".")[0])
    assert major < 9, "elasticsearch-py >=9 修改了 search(source) 语义（字段过滤失效），必须钉 <9"
    assert "source" in inspect.signature(Elasticsearch.search).parameters


def test_sql_messages_turn_id_per_save():
    """turn_id 按轮次粒度：两次 save（两轮）应两组不同 turn_id。"""
    engine = _engine()
    store = SqlSessionStore(engine)
    store.save("u1", "s1", _state(messages=[_msg("user", "第一轮")]))
    store.save("u1", "s1", _state(version=1, messages=[_msg("user", "第一轮")]),
               new_messages=[_msg("assistant", "第二轮回答")])
    with engine.connect() as conn:
        turn_ids = conn.execute(
            select(chat_messages.c.seq, chat_messages.c.turn_id)
            .where(chat_messages.c.session_key == "u1/s1")
        ).all()
    assert len({t for _, t in turn_ids}) == 2  # 两轮两个 turn_id
    assert {seq for seq, _ in turn_ids} == {1, 2}


def test_sql_ltm_preserves_original_created_at():
    """全量替换不得把「一条记忆多久前形成」刷成 now。"""
    engine = _engine()
    store = SqlLTMStore(engine)
    store.save("u1", {
        "facts": [{"content": "旧记忆", "category": "preference",
                   "created_at": "2024-01-01T00:00:00", "source_session": ""}],
        "interaction_summaries": [{"summary": "老客", "timestamp": "2024-01-01"}],
    })
    payload = store.load("u1")
    assert payload["facts"][0]["created_at"].startswith("2024-01-01")
    assert payload["interaction_summaries"][0]["timestamp"].startswith("2024-01-01")


def test_outbox_failure_records_sync_error():
    """ES 批量失败：synced_at 留空 + sync_error 落列（不再只有日志）。"""
    from app.stores.sql.outbox import sync_outbox_to_es

    engine = _engine()
    store = SqlSessionStore(engine, outbox_enabled=True)
    store.save("u1", "s1", _state(messages=[_msg("user", "你好")]))

    es = _ESish(fail=True)
    assert sync_outbox_to_es(engine, es, "ecom-messages") == 0
    with engine.connect() as conn:
        row = conn.execute(outbox_rows.select()).mappings().one()
    assert row["synced_at"] is None
    assert "bulk errors=true" in (row["sync_error"] or "")  # 失败原因落列（不再只有日志）
