"""Review 问题全量修复·验收测试（无网络；SQL 用 sqlite 内存库）。

覆盖《Review 问题全量修复计划》测试计划：
- SQL 端到端：无工具/单工具/混合工具持久化重载；完整审计 + 最终 assistant
  进库；模型历史不含中间 tool 消息；outbox 不含 confirmation token；
- 终答协议：纯文本纠错恢复；混合/多 final/非法参数消息序；强制终答失败
  → 确定性 fallback；
- 记忆任务：水位单调（并发不回退）/ reset 后 obsolete / Redis 缓存失效 /
  文件队列真实 worker（完成、失败重试、租约接管、幂等）；
- 事实校验：同主题 7天/15天 冲突发现；不同主题不误报；单一证据通过。
"""

from __future__ import annotations

import json

import pytest
from fakeredis import FakeRedis, FakeServer
from sqlalchemy import create_engine, select
from sqlalchemy.pool import StaticPool

from app.agent.chat import EcomAgent
from app.agent.fact_guard import ground_reply
from app.agent.memory.jobs import (
    FileMemoryJobStore,
    MemoryJobWorker,
    SqlMemoryJobStore,
)
from app.config.settings import settings
from app.stores.base import SessionState
from app.stores.sql.schema import chat_messages, memory_jobs, metadata
from app.stores.sql.session_store import SqlSessionStore
from tests.unit.conftest import FakeChatClient


@pytest.fixture(autouse=True)
def _isolated_tool_registry():
    """工具注册表快照（避免跨测试泄漏）。"""
    from app.agent.tools import registry as _registry

    snapshot = dict(_registry._TOOL_MAP)
    yield
    _registry._TOOL_MAP.clear()
    _registry._TOOL_MAP.update(snapshot)


def _sql_engine():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    metadata.create_all(engine)
    return engine


def _sql_agent(engine, tmp_path, client, user_id="u1", session_id="s1"):
    store = SqlSessionStore(engine)
    agent = EcomAgent(
        user_id=user_id, session_id=session_id, session_store=store,
        client=client, memory_enabled=False, use_mcp=False,
    )
    # 修复计划·二轮 1：写工具租约门禁在 ToolManager；测试直构 Agent 时
    # 模拟路由已绑定租约（生产由 /v1/chat 注入）
    agent.bind_lease_guard(lambda: None)
    return agent, store


# ============================================================
# SQL 端到端：持久化 → 重载归一化
# ============================================================
def test_sql_e2e_tool_turn_reload_has_no_intermediate_tool_messages():
    """工具轮持久化重载：模型历史只含 user + 最终 assistant；审计含 tool。"""
    from app.agent.tools import registry

    registry._TOOL_MAP["query_order"] = (
        lambda order_id, ctx=None: {
            "success": True, "code": "ORDER_FOUND",
            "order": {"order_id": order_id, "status": "已发货"},
        }
    )
    engine = _sql_engine()
    client = (
        FakeChatClient()
        .enqueue_tool_call("c1", "query_order", {"order_id": "ORD-1"})
        .enqueue_final_response("您的订单 ORD-1 已发货。", intent="order_query")
    )
    agent, store = _sql_agent(engine, None, client)
    result = agent.chat("查订单 ORD-1")
    assert "已发货" in result.reply

    loaded = store.load("u1", "s1")
    roles = [m["role"] for m in loaded.messages]
    assert roles == ["user", "assistant"]  # 模型历史不含中间 tool 消息
    assert loaded.messages[-1]["content"] == "您的订单 ORD-1 已发货。"
    assert loaded.messages[-1].get("metadata", {}).get("schema") == 2

    # 数据库审计正本：完整消息序保留（业务 tool 结果 + final 接受结果）
    with engine.connect() as conn:
        rows = conn.execute(
            select(chat_messages.c.role).where(chat_messages.c.session_key == "u1/s1")
        ).scalars().all()
    assert rows == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    # tool 数=2（query_order 结果 + final 接受结果）；
    # assistant 数=3（两个 tool-call 步骤 + 最终答复，均在审计增量中）
    assert rows.count("tool") == 2
    assert rows.count("assistant") == 3
    assert rows[-1] == "assistant"  # 最终 assistant 在审计增量中（Review 修复）


def test_sql_mixed_calls_persist_one_assistant_and_exact_tool_results():
    """混合业务+final 调用：一条 assistant tool-call 消息 + 每个 call 恰好一个结果。"""
    from app.agent.tools import registry

    registry._TOOL_MAP["query_product"] = (
        lambda keyword, ctx=None: {"success": True, "products": []}
    )
    engine = _sql_engine()
    client = (
        FakeChatClient()
        .enqueue_callable(lambda kind, kwargs: _response_mixed())
        .enqueue_final_response("为您找到耳机商品。", intent="product_consult")
    )
    agent, store = _sql_agent(engine, None, client)
    result = agent.chat("有什么耳机")
    assert result.reply == "为您找到耳机商品。"

    with engine.connect() as conn:
        rows = conn.execute(
            select(chat_messages.c.role, chat_messages.c.content)
            .where(chat_messages.c.session_key == "u1/s1")
            .order_by(chat_messages.c.seq)
        ).all()
    contents = [json.loads(r.content) for r in rows]
    tool_call_msgs = [m for m in contents if m.get("role") == "assistant" and m.get("tool_calls")]
    # 混合步骤只写一条 assistant tool-call 消息（含全部 3 个 call）；
    # 第二步的 final_response 是独立一条（合法消息序）
    mixed = next(m for m in tool_call_msgs if len(m["tool_calls"]) == 3)
    assert [tc["id"] for tc in mixed["tool_calls"]] == ["cb1", "cf1", "cf2"]
    tool_results = {
        m["tool_call_id"]: m["content"] for m in contents if m.get("role") == "tool"
    }
    # 每个 call ID 全程恰好一个结果（无重复、无缺失）
    all_ids = [tc["id"] for m in tool_call_msgs for tc in m["tool_calls"]]
    assert sorted(tool_results) == sorted(all_ids)
    assert len(tool_results) == len(set(tool_results))
    # 混合场景（含业务工具）：final 调用退回 PREMATURE（观察业务结果后重新终答）
    assert "FINAL_RESPONSE_PREMATURE" in tool_results["cf1"]
    assert "FINAL_RESPONSE_PREMATURE" in tool_results["cf2"]


def _response_mixed():
    """混合响应：query_product + 两个 final_response 调用。"""
    from tests.unit.conftest import _Choice, _Response, _ToolCall, _ToolCallMessage

    return _Response([_Choice(_ToolCallMessage("", [
        _ToolCall("cb1", "query_product", '{"keyword": "耳机"}'),
        _ToolCall("cf1", "final_response", '{"intent": "product_consult", "reply": "x", "requires_human": false}'),
        _ToolCall("cf2", "final_response", '{"intent": "other", "reply": "y", "requires_human": false}'),
    ]))])


# ============================================================
# 终答协议
# ============================================================
def test_plain_text_output_gets_protocol_correction_then_recovers():
    """纯文本输出 → 协议纠错重试 → final_response 成功。"""
    client = (
        FakeChatClient()
        .enqueue_chat("我直接输出文本。")
        .enqueue_final_response("好的，这是最终答复。", intent="after_sale")
    )
    engine = _sql_engine()
    agent, _ = _sql_agent(engine, None, client)
    result = agent.chat("确认一下")
    assert result.reply == "好的，这是最终答复。"
    # 第一次调用后窗口出现协议纠错 system 说明
    second_messages = client.calls[1][1]["messages"]
    assert any("final_response 工具提交最终答复" in str(m.get("content", ""))
               for m in second_messages)


def test_mixed_call_message_order_and_multi_final_errors():
    """混合调用 + 多 final：合法消息序 + 多个 final 全部协议错误。"""
    client = (
        FakeChatClient()
        .enqueue_callable(lambda kind, kwargs: _response_mixed())
        .enqueue_final_response("为您找到耳机商品。", intent="product_consult")
    )
    agent, _ = _sql_agent(_sql_engine(), None, client)
    result = agent.chat("有什么耳机")
    assert result.reply == "为您找到耳机商品。"
    second_messages = client.calls[1][1]["messages"]
    # 一条 assistant tool-call（3 个 call）→ 3 个 tool 结果（cf1/cf2 为协议错误）
    tool_msgs = {m["tool_call_id"]: m["content"] for m in second_messages
                 if m.get("role") == "tool"}
    assert set(tool_msgs) == {"cb1", "cf1", "cf2"}
    assert "FINAL_RESPONSE_PREMATURE" in tool_msgs["cf1"]
    assert "FINAL_RESPONSE_PREMATURE" in tool_msgs["cf2"]
    assert "query_product" in tool_msgs["cb1"] or "success" in tool_msgs["cb1"]


def test_forced_finalize_failure_goes_deterministic_fallback(monkeypatch):
    """强制终答失败 → ForcedFinalizeFailed → 确定性转人工 fallback。"""

    monkeypatch.setattr(settings, "max_react_steps", 1)
    from app.agent.tools import registry

    monkeypatch.setitem(registry._TOOL_MAP, "query_product",
                        lambda keyword, ctx=None: {"success": True, "products": []})
    # 步骤1：业务工具；强制收尾：返回无 tool_calls 的纯文本 → 失败
    client = (
        FakeChatClient()
        .enqueue_tool_call("c1", "query_product", {"keyword": "耳机"})
        .enqueue_chat("还是纯文本")
    )
    agent, _ = _sql_agent(_sql_engine(), None, client)
    result = agent.chat("有什么耳机")
    assert result.requires_human is True
    assert result.confidence == 0.0
    assert result.reply  # 确定性话术


def test_validate_final_response_rejects_bad_args():
    args, error = validate_args({"reply": "缺 intent"})
    assert args is None and "FINAL_RESPONSE_INVALID" in error


def validate_args(payload):
    from app.agent.final_response import validate_final_response

    return validate_final_response(payload)


# ============================================================
# 记忆任务
# ============================================================
def test_sql_watermark_never_regresses_with_stale_agent_state():
    """worker 推进水位后，旧 Agent 状态保存不得写小（数据库内单调 clamp）。"""
    engine = _sql_engine()
    store = SqlSessionStore(engine)
    store.save("u1", "s1", SessionState(
        session_id="s-uuid-1", user_id="u1",
        messages=[{"role": "user", "content": "hi"}],
        version=0,
    ), new_messages=[{"role": "user", "content": "hi"}],
        enqueue_memory_job=True,
    )
    jobs = SqlMemoryJobStore(engine)
    claimed = jobs.claim("w1")
    assert claimed and claimed[0]["session_uuid"] == "s-uuid-1"
    jobs.advance_watermark("u1/s1", claimed[0]["through_seq"])

    # 旧 Agent（version 已推进，consolidated_len=0）再保存 → 水位不得回退
    loaded = store.load("u1", "s1")
    store.save("u1", "s1", SessionState(
        session_id="s-uuid-1", user_id="u1", messages=loaded.messages,
        version=loaded.version, consolidated_len=0,
    ), new_messages=[{"role": "assistant", "content": "答"}])

    with engine.connect() as conn:
        value = conn.execute(
            select(memory_jobs.c.through_seq).limit(1)
        ).scalar_one()
        from sqlalchemy import text

        cl = conn.execute(
            text("SELECT consolidated_len FROM sessions WHERE session_key='u1/s1'")
        ).scalar()
    assert cl >= value  # 水位单调不回退


def test_sql_worker_marks_stale_job_obsolete_after_reset():
    """reset 后旧任务按 obsolete 处理，绝不读取新会话消息。"""
    engine = _sql_engine()
    store = SqlSessionStore(engine)
    store.save("u1", "s1", SessionState(
        session_id="uuid-A", user_id="u1",
        messages=[{"role": "user", "content": "旧"}],
    ), new_messages=[{"role": "user", "content": "旧"}],
        enqueue_memory_job=True,
    )
    jobs = SqlMemoryJobStore(engine)
    claimed = jobs.claim("w1")
    assert claimed[0]["session_uuid"] == "uuid-A"

    # 模拟 reset：同 session_id 重建（新 uuid）；多 Pod 场景下旧 pod 手里
    # 还持有已领取的 job 行——uuid 比对是唯一防线
    from sqlalchemy import text as _text

    with engine.begin() as conn:
        conn.execute(_text(
            "UPDATE sessions SET session_uuid='uuid-B' WHERE session_key='u1/s1'"
        ))
    called = []
    worker = MemoryJobWorker(
        jobs,
        lambda uid, sid: (_ for _ in ()).throw(AssertionError("obsolete 任务不得读取消息")),
        None, "fake", worker_id="w1",
    )
    # 手动标记 claimed 为 pending 以便 process_once 领取（模拟崩溃前已入队）
    from sqlalchemy import text as _t

    with engine.begin() as conn:
        conn.execute(_t(
            "UPDATE memory_jobs SET status='pending', lease_until=NULL"
        ))
    worker.process_once()
    with engine.connect() as conn:
        status = conn.execute(
            select(memory_jobs.c.status).limit(1)
        ).scalar_one()
    assert status == "obsolete"
    assert called == []  # 新会话消息从未被读取


def test_sql_worker_invalidates_redis_cache_on_advance():
    """worker 推进水位后同步失效会话热缓存。"""
    engine = _sql_engine()
    redis = FakeRedis(server=FakeServer())
    store = SqlSessionStore(engine, redis=redis, hot_ttl=60)
    store.save("u1", "s1", SessionState(
        session_id="uuid-1", user_id="u1",
        messages=[{"role": "user", "content": "hi"}],
    ), new_messages=[{"role": "user", "content": "hi"}],
        enqueue_memory_job=True,
    )
    assert redis.exists("session:u1/s1") == 1
    jobs = SqlMemoryJobStore(engine, redis=redis)
    claimed = jobs.claim("w1")
    jobs.advance_watermark("u1/s1", claimed[0]["through_seq"])
    assert redis.exists("session:u1/s1") == 0  # 缓存已失效


def test_file_queue_real_worker_complete_fail_lease_idempotent(tmp_path, monkeypatch):
    """Windows 文件队列：真实 worker 完成失败重试租约接管幂等。"""
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path / "memory"))
    queue_dir = tmp_path / "memory" / "jobs"
    store = FileMemoryJobStore(str(queue_dir))

    payload = [
        {"role": "user", "content": "我叫雷腾，喜欢红色"},
        {"role": "assistant", "content": "已记住您的偏好。"},
    ]
    store.enqueue("u1/s1", "u1", 2, session_uuid="uuid-1", messages=payload)
    store.enqueue("u1/s1", "u1", 2, session_uuid="uuid-1", messages=payload)  # 幂等
    assert len(store.jobs_snapshot()) == 1

    # 真实 worker：完成
    captured = []

    class FakeLTM:
        def extract_and_save(self, client, model, messages, summary):
            captured.append(messages)

    worker = MemoryJobWorker(store, lambda uid, sid: FakeLTM(), None, "fake", "w1")
    done = worker.process_once()
    assert done == 1 and len(captured) == 1
    assert store.jobs_snapshot() == []  # 完成即出队
    done_log = (queue_dir / "memory_jobs.done.jsonl").read_text(encoding="utf-8")
    assert "u1/s1" in done_log  # 完成记录

    # 失败重试：LTM 工厂抛错 → attempts+1 → 未到期不再领取
    store.enqueue("u2/s2", "u2", 3, session_uuid="uuid-2", messages=payload)

    class BoomLTM:
        def extract_and_save(self, *a, **k):
            raise RuntimeError("boom")

    worker2 = MemoryJobWorker(store, lambda uid, sid: BoomLTM(), None, "fake", "w2")
    assert worker2.process_once() == 0
    entry = store.jobs_snapshot()[0]
    assert entry["status"] == "pending" and entry["attempts"] == 1
    assert entry["next_run_at"]  # 退避时间已设置
    assert store.claim("w2") == []  # 未到期不领取

    # 租约接管：过期 processing 可被其他 worker 领取
    store._upsert({**entry, "next_run_at": ""})  # 清退避，便于领取
    first = store.claim("w-a")[0]
    assert first["status"] == "processing"
    assert store.claim("w-b") == []  # 租约未过期
    store._upsert({**first, "lease_until": "2000-01-01T00:00:00"})  # 过期
    taken = store.claim("w-b")
    assert taken and taken[0]["id"] == first["id"]

    # reset 清理
    store.enqueue("u3/s3", "u3", 1, session_uuid="u", messages=payload)
    assert store.purge_session("u3/s3") == 1
    assert all(j["session_key"] != "u3/s3" for j in store.jobs_snapshot())


def test_memory_job_payload_skips_plain_text_correction_turns():
    """低危修复 A4：纯文本协议纠错的 assistant 稿（后跟 PLAIN_TEXT_CORRECTION
    system 消息）不进 memory job 巩固负载——负载只含用户消息与最终答复
    （修复前中间纠错文本被当作独立轮次巩固进 LTM）。"""
    from app.agent.chat import _memory_job_payload
    from app.agent.react_runner import PLAIN_TEXT_CORRECTION

    folded = [
        {"role": "user", "content": "退货政策是什么"},
        {"role": "assistant", "content": "七天无理由可退。"},  # 中间纠错稿
        {"role": "system", "content": PLAIN_TEXT_CORRECTION},
        {"role": "assistant", "content": "七天无理由退货，运费由商家承担。"},
    ]
    payload = _memory_job_payload(folded)
    assert [m["content"] for m in payload] == [
        "退货政策是什么", "七天无理由退货，运费由商家承担。",
    ]

    # 无纠错轮的普通负载不受影响
    plain = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "您好，请问有什么可以帮您？"},
    ]
    assert [m["content"] for m in _memory_job_payload(plain)] == [
        "你好", "您好，请问有什么可以帮您？",
    ]


# ============================================================
# 事实校验（主题锚点 + 单位）
# ============================================================
def test_fact_conflict_same_topic_detected():
    evidence = [
        "退款将在7天内原路退回。",
        "退款周期为15天，具体以银行处理为准。",
    ]
    cleaned, verdict = ground_reply("退款将在7天内原路退回。", evidence)
    assert verdict.conflicts  # 同主题（退款）同单位（天）多值 → 冲突
    assert "7天" not in cleaned  # 冲突句删除并转核实话术
    assert "核实" in cleaned


def test_fact_conflict_ignores_leading_discourse_modifier():
    evidence = [
        "本次退款7天内原路退回。",
        "退款周期为15天，具体以银行处理为准。",
    ]
    cleaned, verdict = ground_reply("本次退款7天内原路退回。", evidence)
    assert verdict.conflicts
    assert "7天" not in cleaned


def test_fact_no_conflict_across_different_topics():
    evidence = [
        "退货支持7天无理由。",
        "到货周期为15天。",
    ]
    cleaned, verdict = ground_reply("退货支持7天无理由。到货周期为15天。", evidence)
    assert not verdict.conflicts  # 不同主题同单位不误报
    assert "7天" in cleaned and "15天" in cleaned


def test_fact_conflict_does_not_remove_same_value_from_other_topic():
    evidence = [
        "退款将在7天内原路退回。",
        "退款周期为15天。",
        "退货支持7天无理由。",
    ]
    cleaned, verdict = ground_reply(
        "退款将在7天内原路退回。退货支持7天无理由。", evidence,
    )
    assert verdict.conflicts == ["7天"]
    assert "退款将在7天" not in cleaned
    assert "退货支持7天" in cleaned


def test_fact_single_supporting_evidence_passes():
    evidence = ["钻石会员专属客服的响应时效SLO是30秒内接入。"]
    cleaned, verdict = ground_reply("专属客服30秒内接入。", evidence)
    assert not verdict.conflicts and verdict.removed_sentences == 0
    assert cleaned == "专属客服30秒内接入。"
