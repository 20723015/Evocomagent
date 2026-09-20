"""记忆系统重构·阶段0/1/2 测试：注入埋点、双时态、混合检索与降级等价。

全程无网络：嵌入用确定性假向量/假 embedder；LLM 用 FakeChatClient。
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine

from app.agent.memory.embeddings import (
    FileMemoryEmbeddingStore,
    SqlMemoryEmbeddingStore,
    backfill_embeddings,
)
from app.agent.memory.long_term import LongTermMemory, _cosine, dice_score, _token_set
from app.agent.memory.models import MemoryFact, apply_memory_mutations, MemoryMutation
from app.config.settings import settings
from app.llm.embeddings import MemoryEmbeddingClient, get_memory_embedder, reset_memory_embedder
from app.observability import metrics
from app.stores.sql.schema import metadata
from tests.unit.conftest import FakeChatClient, FakeEmbedder


def _fact(content: str, category: str = "preference", days_old: float = 10,
          fact_key: str = "preference.color") -> MemoryFact:
    created = datetime.now(timezone.utc) - timedelta(days=days_old)
    return MemoryFact(
        content=content, category=category,
        created_at=created.isoformat(timespec="seconds"), fact_key=fact_key,
    )


class _DictEmbeddingStore:
    """内存版嵌入存储（检索侧融合测试用）。"""

    def __init__(self, mapping: dict[str, list[float]] | None = None):
        self._m = dict(mapping or {})

    def get(self, user_id, fact_id, model):
        return self._m.get(fact_id)

    def put(self, user_id, fact_id, model, content_hash, vector):
        self._m[fact_id] = vector


def _semantic_ltm(tmp_path, facts, embedding_map=None,
                  monkeypatch=None) -> LongTermMemory:
    """构造语义开启的 LTM（假嵌入存储 + 指定 settings）。"""
    monkeypatch.setattr(settings, "memory_semantic_enabled", True)
    monkeypatch.setattr(settings, "memory_embedding_model", "fake-embedder")
    ltm = LongTermMemory(user_id="u_sem", memory_dir=str(tmp_path))
    ltm.facts = facts
    ltm._embedding_store = _DictEmbeddingStore(embedding_map)
    return ltm


# ---------- 融合打分排序（计划 4.6）----------
def test_semantic_high_zero_lexical_selected(tmp_path, monkeypatch):
    """语义高分 + 词面零重叠 → 入选（这正是混合检索的目的）。"""
    fact = _fact("用户偏好退款时原路退回付款账户", fact_key="preference.delivery")
    ltm = _semantic_ltm(tmp_path, [fact], monkeypatch=monkeypatch,
                        embedding_map={fact.fact_id: [1.0, 0.0, 0.0]})
    query = "如果我退货，钱会怎么打给我？"
    assert dice_score(_token_set(query), _token_set(fact.content)) == 0.0
    selected = ltm.select_facts_for_prompt(query, query_embedding=[0.9, 0.1, 0.0])
    assert [f.fact_id for f in selected] == [fact.fact_id]


def test_lexical_high_zero_semantic_selected(tmp_path, monkeypatch):
    """词面高分 + 语义缺失（无嵌入）→ 仍入选（词面路保留为兜底）。"""
    fact = _fact("用户最喜欢蓝色的运动鞋")
    ltm = _semantic_ltm(tmp_path, [fact], monkeypatch=monkeypatch)  # 无嵌入
    selected = ltm.select_facts_for_prompt(
        "我想买双运动鞋，推荐个颜色呗？", query_embedding=[0.1, 0.9, 0.0],
    )
    assert [f.fact_id for f in selected] == [fact.fact_id]


def test_both_low_excluded(tmp_path, monkeypatch):
    """双低（词面零重叠 + 语义正交）→ 排除。

    用非 identity 类别验证门槛本身：identity 受控单值键每轮必注入
    （记忆系统重构·Step3），不再受此门槛约束。
    """
    fact = _fact("用户常在晚间下单", category="behavior",
                 fact_key="behavior.payment")
    ltm = _semantic_ltm(tmp_path, [fact], monkeypatch=monkeypatch,
                        embedding_map={fact.fact_id: [0.0, 0.0, 1.0]})
    selected = ltm.select_facts_for_prompt(
        "我想买双运动鞋，推荐个颜色呗？", query_embedding=[1.0, 0.0, 0.0],
    )
    assert selected == []


def test_identity_exempt_from_semantic_gate(tmp_path, monkeypatch):
    """identity 单值键即使双低也注入（Step3：身份每轮必注入）。"""
    fact = _fact("用户是健身教练", category="identity",
                 fact_key="identity.occupation")
    ltm = _semantic_ltm(tmp_path, [fact], monkeypatch=monkeypatch,
                        embedding_map={fact.fact_id: [0.0, 0.0, 1.0]})
    selected = ltm.select_facts_for_prompt(
        "我想买双运动鞋，推荐个颜色呗？", query_embedding=[1.0, 0.0, 0.0],
    )
    assert [f.fact_id for f in selected] == [fact.fact_id]


def test_semantic_rank_orders_by_fused_score(tmp_path, monkeypatch):
    """融合排序：语义高分事实排在仅类别/新近度加权的事实之前。"""
    target = _fact("用户偏好退款时原路退回付款账户", days_old=100,
                   fact_key="preference.delivery")
    other = _fact("用户是钻石会员", category="identity", days_old=1,
                  fact_key="identity.membership_level")
    ltm = _semantic_ltm(tmp_path, [target, other], monkeypatch=monkeypatch,
                        embedding_map={target.fact_id: [1.0, 0.0]})
    selected = ltm.select_facts_for_prompt(
        "退货后钱多久到账？", query_embedding=[1.0, 0.0],
    )
    assert selected and selected[0].fact_id == target.fact_id


# ---------- 降级路径逐字节等价（计划 4.6）----------
def test_degraded_path_byte_equivalent(tmp_path, monkeypatch):
    """语义关闭/嵌入缺失：选择结果与纯词面旧实现完全一致。"""
    facts = [
        _fact("用户最喜欢蓝色的运动鞋", days_old=5),
        _fact("用户是钻石会员", category="identity", days_old=30,
              fact_key="identity.membership_level"),
        _fact("用户反馈订单少发了一件", category="issue", days_old=10,
              fact_key="issue.current"),
    ]
    query = "我想买双运动鞋，推荐个颜色呗？"

    monkeypatch.setattr(settings, "memory_semantic_enabled", False)
    plain = LongTermMemory(user_id="u_plain", memory_dir=str(tmp_path))
    plain.facts = list(facts)
    expected = plain.select_facts_for_prompt(query)

    # 语义开启但嵌入全缺 → 逐字节等价
    sem = _semantic_ltm(tmp_path, list(facts), monkeypatch=monkeypatch)
    got = sem.select_facts_for_prompt(query, query_embedding=[0.5, 0.5])
    assert [f.fact_id for f in got] == [f.fact_id for f in expected]

    # 语义关闭且带 query_embedding 入参 → 仍等价（参数被忽略）
    got2 = plain.select_facts_for_prompt(query, query_embedding=[1.0, 0.0])
    assert [f.fact_id for f in got2] == [f.fact_id for f in expected]


def test_empty_query_semantic_no_threshold(tmp_path, monkeypatch):
    """空 query：不做门槛过滤，按权重+新近度取 top-N（语义路同旧行为）。"""
    facts = [_fact("用户是钻石会员", category="identity",
                   fact_key="identity.membership_level")]
    ltm = _semantic_ltm(tmp_path, facts, monkeypatch=monkeypatch,
                        embedding_map={facts[0].fact_id: [1.0]})
    selected = ltm.select_facts_for_prompt("", query_embedding=[0.5, 0.5])
    assert len(selected) == 1


# ---------- 阶段0：注入埋点 ----------
def test_inject_metrics_recorded(tmp_path, monkeypatch):
    facts = [
        _fact("用户最喜欢蓝色的运动鞋"),
        _fact("用户反馈订单少发了一件", category="issue", days_old=4000,
              fact_key="issue.current"),  # 超 TTL
        _fact("用户常在晚间下单", category="behavior",
              fact_key="behavior.payment"),  # 词面不相关（非 identity）
        _fact("用户是健身教练", category="identity",
              fact_key="identity.occupation"),  # 词面不相关但身份必注入
    ]
    ltm = LongTermMemory(user_id="u_metrics", memory_dir=str(tmp_path))
    ltm.facts = facts

    before_candidates = metrics.MEMORY_INJECT_CANDIDATES._sum.get()
    before_filtered_ttl = metrics.MEMORY_INJECT_FILTERED.labels(
        reason="ttl")._value.get()
    before_filtered_thr = metrics.MEMORY_INJECT_FILTERED.labels(
        reason="below_threshold")._value.get()

    selected = ltm.select_facts_for_prompt("我想买双运动鞋，推荐个颜色呗？")

    assert [f.content for f in selected] == [
        "用户最喜欢蓝色的运动鞋", "用户是健身教练",
    ]
    assert metrics.MEMORY_INJECT_CANDIDATES._sum.get() > before_candidates
    assert (metrics.MEMORY_INJECT_FILTERED.labels(reason="ttl")._value.get()
            > before_filtered_ttl)
    assert (metrics.MEMORY_INJECT_FILTERED.labels(reason="below_threshold")._value.get()
            > before_filtered_thr)
    # 末次注入快照（漏注度量依据）
    assert set(ltm._last_injected) == {facts[0].fact_id, facts[3].fact_id}


def test_inject_debug_log_has_no_content(tmp_path, caplog):
    fact = _fact("用户最喜欢蓝色的运动鞋")
    ltm = LongTermMemory(user_id="u_log", memory_dir=str(tmp_path))
    ltm.facts = [fact]
    import logging

    with caplog.at_level(logging.DEBUG, logger="app.agent.memory.long_term"):
        ltm.select_facts_for_prompt("运动鞋推荐")
    text = caplog.text
    assert "memory.inject" in text
    assert fact.content not in text  # 防 PII：日志只有 fact_id+score
    assert fact.fact_id in text


# ---------- 阶段0：recall 漏注度量 + 阶段1.2 时态字段 ----------
def test_recall_miss_metric_and_temporal_fields(tmp_path, monkeypatch):
    from app.agent.memory.manager import MemoryManager
    from app.agent.tools.memory_tool import recall_user_memory
    from app.agent.context import ToolContext

    fact = _fact("用户最喜欢蓝色的运动鞋")
    manager = MemoryManager(client=None, model="m", user_id="u_recall",
                            memory_dir=str(tmp_path), memory_enabled=True)
    manager.ltm.facts = [fact]
    ctx = ToolContext(user_id="u_recall", memory=manager)

    before = metrics.MEMORY_RECALL_MISS._value.get()
    result = recall_user_memory("运动鞋", ctx=ctx)
    assert result["success"] is True
    entry = result["long_term_facts"][0]
    assert entry["valid_from"] == fact.created_at
    assert entry["invalid_at"] == ""
    assert "fact_key" in entry
    # 未发生自动注入（_last_injected 空）→ 召回命中 = 漏注
    assert metrics.MEMORY_RECALL_MISS._value.get() > before

    # 注入快照覆盖后不再计漏注
    manager.ltm._last_injected = {fact.fact_id: 0.5}
    after = metrics.MEMORY_RECALL_MISS._value.get()
    recall_user_memory("运动鞋", ctx=ctx)
    assert metrics.MEMORY_RECALL_MISS._value.get() == after


# ---------- 阶段1.2：双时态 facts_active_at ----------
def test_facts_active_at_supersede_chain(tmp_path):
    """supersede 链：旧事实在 invalid_at 前有效，新事实在 created_at 后有效。"""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    old = MemoryFact(content="用户喜欢红色", category="preference",
                     created_at=t0.isoformat(timespec="seconds"))
    records = apply_memory_mutations(
        [old],
        [MemoryMutation(
            operation="upsert", fact_key="preference.color",
            content="用户现在喜欢蓝色", category="preference",
            confidence=0.95, explicit=True, evidence="我现在喜欢蓝色",
            target_fact_id=old.fact_id,  # legacy 键迁移到受控键并 supersede 旧版
        )],
        max_active=50, now=t1.isoformat(timespec="seconds"),
    )
    ltm = LongTermMemory(user_id="u_temporal", memory_dir=str(tmp_path))
    ltm.facts = records

    before = ltm.facts_active_at(t0 + timedelta(days=30))
    assert [f.content for f in before] == ["用户喜欢红色"]

    after = ltm.facts_active_at(t1 + timedelta(days=1))
    assert [f.content for f in after] == ["用户现在喜欢蓝色"]

    # 字符串入参 + 不可解析时间戳 fail-safe（按当前 active 返回）
    assert [f.content for f in ltm.facts_active_at("2026-03-01T00:00:00+00:00")] == ["用户喜欢红色"]
    assert [f.content for f in ltm.facts_active_at("not-a-date")] == ["用户现在喜欢蓝色"]


# ---------- 阶段2.1：嵌入客户端 ----------
def test_embedder_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "memory_semantic_enabled", False)
    reset_memory_embedder()
    assert get_memory_embedder() is None

    monkeypatch.setattr(settings, "memory_semantic_enabled", True)
    monkeypatch.setattr(settings, "memory_embedding_model", "fake-embedder")
    monkeypatch.setattr(settings, "openai_api_key", "")
    reset_memory_embedder()
    assert get_memory_embedder() is None


def test_embedder_lru_cache_avoids_duplicate_calls(monkeypatch):
    client = MemoryEmbeddingClient("fake-embedder")
    fake = FakeEmbedder()
    calls = {"n": 0}
    orig = fake.encode_one

    def counting(text, timeout=None):
        calls["n"] += 1
        return orig(text, timeout)

    fake.encode_one = counting
    client._embedder = fake
    v1 = client.encode("用户喜欢蓝色", stage="query")
    v2 = client.encode("用户喜欢蓝色", stage="query")
    assert v1 == v2 and calls["n"] == 1
    assert client.encode("", stage="query") is None


def test_embedder_failure_returns_none_and_counts(monkeypatch):
    client = MemoryEmbeddingClient("fake-embedder")

    class _Boom:
        def encode_one(self, text, timeout=None):
            raise RuntimeError("boom")

        def encode(self, texts, timeout=None):
            raise RuntimeError("boom")

    client._embedder = _Boom()
    before = metrics.MEMORY_EMBEDDING_FAILURES.labels(stage="query")._value.get()
    assert client.encode("文本", stage="query") is None
    assert (metrics.MEMORY_EMBEDDING_FAILURES.labels(stage="query")._value.get()
            == before + 1)
    assert client.encode_batch(["a", "b"], stage="write") == [None, None]


# ---------- 阶段2.2：嵌入存储 ----------
def test_file_embedding_store_roundtrip(tmp_path):
    store = FileMemoryEmbeddingStore(str(tmp_path))
    store.put("u1", "f1", "m1", "h1", [0.1, 0.2])
    assert store.get("u1", "f1", "m1") == pytest.approx([0.1, 0.2])
    assert store.get("u1", "f1", "m2") is None  # 模型不匹配 → 未命中
    assert store.get("u1", "missing", "m1") is None
    store.delete_user("u1")
    assert store.get("u1", "f1", "m1") is None


def test_sql_embedding_store_roundtrip(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/emb.sqlite")
    metadata.create_all(engine)
    store = SqlMemoryEmbeddingStore(engine)
    store.put("u1", "f1", "m1", "h1", [0.1, 0.2, 0.3])
    assert store.get("u1", "f1", "m1") == pytest.approx([0.1, 0.2, 0.3])
    # upsert：同 (user, fact, model) 覆盖
    store.put("u1", "f1", "m1", "h2", [9.0, 9.0, 9.0])
    assert store.get("u1", "f1", "m1") == pytest.approx([9.0, 9.0, 9.0])
    assert store.get("u1", "f1", "other-model") is None
    store.delete_user("u1")
    assert store.get("u1", "f1", "m1") is None


def test_backfill_skips_content_addressed_hits(tmp_path):
    store = FileMemoryEmbeddingStore(str(tmp_path))
    embedder = MemoryEmbeddingClient("fake-embedder")
    fake = FakeEmbedder()
    calls = {"n": 0}
    orig = fake.encode

    def counting(texts, timeout=None):
        calls["n"] += len(list(texts))
        return orig(texts, timeout=timeout)

    fake.encode = counting
    embedder._embedder = fake

    facts = [_fact("用户喜欢蓝色"), _fact("用户是钻石会员", category="identity",
                                        fact_key="identity.membership_level")]
    n1 = backfill_embeddings("u1", facts, embedder, store)
    assert n1 == 2 and calls["n"] == 2
    # 第二次回填：内容未变 → 全部跳过（不再计费）
    n2 = backfill_embeddings("u1", facts, embedder, store)
    assert n2 == 0 and calls["n"] == 2
    # 非 active 事实不补算
    facts[0].status = "superseded"
    n3 = backfill_embeddings("u1", facts, embedder, store)
    assert n3 == 0 and calls["n"] == 2


class _TopicEmbedder:
    """主题维度假 embedder：同义词族共享维度 → 同义改述高余弦（模拟真实语义空间）。

    FakeEmbedder 是字符哈希（同义词零共享 → 正交），无法模拟语义检索场景；
    这里按词族投维：query 与事实命中同族即高余弦，无命中即零向量。
    """

    GROUPS = (
        ("退款", "退货", "退回"),   # dim 0
        ("钱", "账户", "打给", "付款"),  # dim 1
        ("运动鞋", "颜色"),          # dim 2
    )

    def __init__(self, dim: int = 8, model: str = "fake-embedder"):
        self._dim = dim
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def encode(self, texts, timeout=None):
        return [self._vector(t) for t in texts]

    def encode_one(self, text, timeout=None):
        return self._vector(text)

    def _vector(self, text):
        v = [0.0] * self._dim
        for i, words in enumerate(self.GROUPS):
            if any(w in str(text) for w in words):
                v[i] = 1.0
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else v


# ---------- 阶段2 集成：worker 巩固后嵌入落库 + 下一轮注入命中同义改述 ----------
def test_worker_embeds_after_consolidation_then_semantic_hit(tmp_path, monkeypatch):
    from app.agent.memory.jobs import FileMemoryJobStore, MemoryJobWorker
    from app.agent.memory.manager import _derive_embedding_store

    monkeypatch.setattr(settings, "memory_semantic_enabled", True)
    monkeypatch.setattr(settings, "memory_embedding_model", "fake-embedder")
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    reset_memory_embedder()
    # 进程单例换成主题假 embedder（同义词族共享维度 → 高余弦）
    embedder = get_memory_embedder()
    assert embedder is not None
    embedder._embedder = _TopicEmbedder(model="fake-embedder")

    # 巩固提取的 LLM 输出：一条退款偏好事实
    ltm_payload = json.dumps({
        "mutations": [{
            "operation": "upsert", "fact_key": "preference.delivery",
            "content": "用户偏好退款时原路退回付款账户", "category": "preference",
            "confidence": 0.95, "target_fact_id": "", "explicit": True,
            "evidence": "退款麻烦原路退回",
        }],
        "interaction_summary": "用户咨询退款方式",
    }, ensure_ascii=False)
    client = FakeChatClient().enqueue_chat(ltm_payload)

    queue_dir = tmp_path / "jobs"
    job_store = FileMemoryJobStore(str(queue_dir))
    memory_dir = str(tmp_path / "mem")
    embedding_store = _derive_embedding_store(None, memory_dir)

    from app.agent.memory.long_term import LongTermMemory as _LTM

    def ltm_factory(user_id: str, session_id: str):
        return _LTM(user_id=user_id, memory_dir=memory_dir,
                    embedding_store=embedding_store, source_session=session_id)

    worker = MemoryJobWorker(job_store, ltm_factory, client, "m")
    job_store.enqueue(
        "u1/s1", "u1", 2, session_uuid="",
        messages=[
            {"role": "user", "content": "退款麻烦原路退回"},
            {"role": "assistant", "content": "好的，已为您记录"},
        ],
        turn_id="t1",
    )
    assert worker.process_once(limit=1) == 1

    # 嵌入落库（派生数据）
    ltm = ltm_factory("u1", "s1")
    ltm.load()
    assert len(ltm.active_facts) == 1
    fact = ltm.active_facts[0]
    assert embedding_store.get("u1", fact.fact_id, "fake-embedder") is not None

    # 下一轮：同义改述 query（词面零重叠）经语义路命中
    query = "如果我退货，钱会怎么打给我？"
    assert dice_score(_token_set(query), _token_set(fact.content)) == 0.0
    query_embedding = embedder.encode(query, stage="query")
    selected = ltm.select_facts_for_prompt(query, query_embedding=query_embedding)
    assert [f.fact_id for f in selected] == [fact.fact_id]

    # 降级对照：同一 query 无嵌入 → 纯词面不命中（漏注基线）
    plain = ltm.select_facts_for_prompt(query)
    assert plain == []


def test_cosine_edge_cases():
    assert _cosine([], [1.0]) == 0.0
    assert _cosine([1.0], [1.0, 2.0]) == 0.0  # 维度不一致
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0  # 零向量
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
