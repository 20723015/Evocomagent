"""Regression tests for the human-QA surface fixes owned by this task."""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.handoff.board import HandoffTicket, InProcessHandoffBoard, RedisHandoffBoard
from app.stores.sql.schema import metadata

ROOT = Path(__file__).parents[2]


def _resolution(question: str) -> dict:
    return {
        "note": "已按政策处理",
        "knowledge_candidate": True,
        "canonical_question": question,
        "canonical_answer": "标准答案，按政策办理并保留相关凭证；如有疑问可联系人工客服。",
        "knowledge_basis": "客服确认的政策条款",
    }


def test_resolution_event_is_public_and_read_only():
    """两种看板都从 resolution 事件正本读取，而非工单 resolution 快照。"""
    import fakeredis

    for board in (
        InProcessHandoffBoard(),
        RedisHandoffBoard(fakeredis.FakeStrictRedis()),
    ):
        board.create(HandoffTicket(ticket_id="event-1", user_id="u", session_id="s"))
        event, _ = board.resolve_atomic(
            "event-1", _resolution("问题"), "ops", now="2026-08-01T00:00:00+00:00",
        )
        read = board.get_resolution_event("event-1")
        assert read == event
        read["resolved_at"] = "tampered"
        assert board.get_resolution_event("event-1")["resolved_at"] == event["resolved_at"]


def test_resolve_semantics_idempotent_and_conflict():
    """工单 resolve 语义保留：同内容幂等返回同一事件，不同内容 409。"""
    board = InProcessHandoffBoard()
    board.create(HandoffTicket(ticket_id="a", user_id="u", session_id="s"))
    event1, dup1 = board.resolve_atomic(
        "a", _resolution("拆封后的耳机支持七天无理由退货吗"), "ops")
    event2, dup2 = board.resolve_atomic(
        "a", _resolution("拆封后的耳机支持七天无理由退货吗"), "ops")
    assert (dup1, dup2) == (False, True)
    assert event1 == event2
    with pytest.raises(Exception):
        board.resolve_atomic(
            "a", {**_resolution("拆封后的耳机支持七天无理由退货吗"),
                  "note": "另一种结论"}, "ops")


def test_human_eval_stats_feed_metrics(tmp_path, monkeypatch):
    """新链路指标：评审队列深度/最老年龄/blocked 由 store.stats() 驱动。"""
    from app.evolution.human_store import HumanKnowledgeStore
    from app.observability import metrics

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    metadata.create_all(engine)
    store = HumanKnowledgeStore(engine)
    store.ingest_conversation(
        source="cs", external_conversation_id="c1", source_version=1,
        agent_id="", started_at=None,
        ended_at=datetime.now(timezone.utc).replace(tzinfo=None)
        - timedelta(days=1),
        messages=_messages(),
    )
    observed: dict = {}
    monkeypatch.setattr(
        metrics, "set_human_eval_stats",
        lambda stats: observed.update(**stats),
    )
    from app.observability.metrics import set_human_eval_stats as _set

    _set(store.stats())
    assert observed["queue_depth"] >= 1
    assert observed["oldest_age_seconds"] >= 0


def _messages():
    return [
        {"message_id": "c0", "actor_type": "customer", "content": "问题"},
        {"message_id": "a0", "actor_type": "human_agent", "content": "回答"},
    ]


def test_ops_resolve_form_has_no_deposit_and_reads_reclaim():
    """真实 DOM 回归：沉淀 checkbox 已下线；reclaim 仍由独立 checkbox 读取。"""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node executable unavailable")

    source_path = (ROOT / "app/server/static/js/ops.js").as_posix()
    script = "(async () => {\n" + textwrap.dedent(
        f"""
        const fs = require("fs");
        const vm = require("vm");
        const source = fs.readFileSync({json.dumps(source_path)}, "utf8");
        const start = source.indexOf("function renderTicket");
        const end = source.indexOf(String.fromCharCode(10) + "/* ---------- 消息检索", start);
        if (source.slice(start, end).includes("deposit-toggle")) {{
          throw new Error("deposit checkbox should be removed");
        }}
        if (source.slice(start, end).includes("knowledge_candidate")) {{
          throw new Error("resolve body must not send knowledge_candidate");
        }}
        const calls = [];

        class Element {{
          constructor() {{
            this.innerHTML = "";
            this.children = [];
            this.fields = {{
              reclaim: {{ checked: false }},
              note: {{ value: "", focus() {{}} }},
              toggle: {{ addEventListener: (k, fn) => this.toggleFn = fn,
                         setAttribute() {{}} }},
              resolve: {{ addEventListener: (k, fn) => this.resolveFn = fn,
                          disabled: false }},
            }};
            this.style = {{}};
          }}
          appendChild(child) {{ this.children.push(child); }}
          insertAdjacentHTML(where, html) {{ this.innerHTML += html; }}
          querySelector(selector) {{
            if (selector === ".reclaim-toggle") return this.fields.reclaim;
            if (selector === ".toggle-resolve") return this.fields.toggle;
            if (selector === ".do-resolve") return this.fields.resolve;
            if (selector === "textarea") return this.fields.note;
            return null;
          }}
        }}

        const context = {{
          STATUS_LABELS: {{ pending: "待处理" }}, INTENT_LABELS: {{}},
          document: {{ createElement: () => new Element() }},
          escapeHtml: (x) => String(x), renderMd: (x) => String(x),
          prettyJson: (x) => JSON.stringify(x), fmtTime: () => "",
          toast: () => {{}}, announce: () => {{}}, loadTickets: () => {{}},
          api: async (url, options) => {{ calls.push({{ url, options }}); return {{}}; }},
        }};
        vm.runInNewContext(source.slice(start, end), context);
        const card = context.renderTicket({{ ticket_id: "t", user_id: "u", status: "pending", question: "q" }});
        const form = card.children[0];
        form.fields.note.value = "人工结论";
        form.fields.reclaim.checked = false;
        await form.resolveFn();
        const body = JSON.parse(calls[0].options.body);
        if (body.reclaim !== false) throw new Error("reclaim checkbox was not read independently");
        if (body.resolution.knowledge_candidate) throw new Error("deprecated field sent");
        """
    ) + "\n})().catch((error) => { console.error(error); process.exit(1); });\n"
    result = subprocess.run(
        [node, "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_ops_reclaim_checkbox_is_named_explicitly():
    source = (ROOT / "app/server/static/js/ops.js").read_text(encoding="utf-8")
    assert 'class="reclaim-toggle"' in source
    assert 'form.querySelector(".reclaim-toggle").checked' in source
    assert 'form.querySelector("input[type=checkbox]")' not in source


# ============================================================
# P3：ended_at 未来时间校验（评审按当日 00:00 过滤，未来时间会静默积压）
# ============================================================
def _conv_item(ended_at: str):
    from app.server.schema import HumanConversationItem

    return HumanConversationItem(
        source="cs",
        external_conversation_id="c9",
        ended_at=ended_at,
        messages=[
            {"message_id": "c0", "actor_type": "customer", "content": "问题"},
            {"message_id": "a0", "actor_type": "human_agent", "content": "回答"},
        ],
    )


def test_ended_at_in_future_rejected_with_422():
    from datetime import datetime, timedelta

    from fastapi import HTTPException

    from app.server.human_knowledge import _validate_conversation

    future = (datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=3)).isoformat()
    with pytest.raises(HTTPException) as ei:
        _validate_conversation(_conv_item(future))
    assert ei.value.status_code == 422


def test_ended_at_within_tolerance_accepted():
    from datetime import datetime, timedelta

    from app.server.human_knowledge import _validate_conversation

    near = (datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=2)).isoformat()
    parsed = _validate_conversation(_conv_item(near))
    assert parsed["ended_at"] is not None


# ============================================================
# P2-2：常驻评审 Worker 循环（异常不退出 / stats 上报 / cleanup 节流）
# ============================================================
class _FakeEvalWorkerStore:
    def __init__(self, *, fail_stats=False):
        self.cleanup_calls: list[int] = []
        self._fail_stats = fail_stats

    def stats(self):
        if self._fail_stats:
            raise RuntimeError("stats boom")
        return {
            "queue_depth": 1,
            "oldest_age_seconds": 1.0,
            "blocked": 0,
            "stale_published": 0,
        }

    def cleanup_expired_conversations(self, days, limit=500):
        self.cleanup_calls.append(days)
        return 0

    def next_evaluation_delay(self, poll_seconds=30):
        return 0.0


def _run_eval_worker(monkeypatch, *, process_once_script, store, cleanup_interval):
    """驱动 _cmd_human_eval_worker 至可控退出；返回 (stats 调用, store)。"""
    import time as _time

    from app.config.settings import settings as settings_mod
    from app.scripts import run_evolution

    class _FakeEvaluator:
        def __init__(self, *a, **k):
            self._script = list(process_once_script)

        def process_once(self):
            step = self._script.pop(0)
            if isinstance(step, BaseException):
                raise step  # KeyboardInterrupt 也必须抛出而非返回
            return step

    from app.evolution import human_evaluator as he_mod

    monkeypatch.setattr(he_mod, "HumanKnowledgeEvaluator", _FakeEvaluator)
    monkeypatch.setattr(settings_mod, "human_eval_worker_cleanup_interval_seconds", cleanup_interval)
    observed: list[dict] = []
    monkeypatch.setattr(
        "app.observability.metrics.set_human_eval_stats",
        lambda stats: observed.append(stats),
    )
    monkeypatch.setattr(_time, "sleep", lambda s: None)
    svc = {"store": store, "extractor": None, "scorer": None, "generation_store": None}
    rc = run_evolution._cmd_human_eval_worker(svc, poll_seconds=0, max_jobs=0)
    assert rc == 0
    return observed, store


def test_eval_worker_survives_store_exception_and_reports_stats(monkeypatch):
    """store.stats() 抛异常 → 循环仅记日志继续；后续空闲周期恢复上报与清理。"""

    class _FlakyStatsStore(_FakeEvalWorkerStore):
        def __init__(self):
            super().__init__()
            self._stats_calls = 0

        def stats(self):
            self._stats_calls += 1
            if self._stats_calls == 1:
                raise RuntimeError("stats boom")
            return super().stats()

    store = _FlakyStatsStore()
    # 第1轮空闲 stats 抛错 → 捕获；第2轮空闲正常上报；第3轮 KeyboardInterrupt 退出
    observed, store = _run_eval_worker(
        monkeypatch,
        process_once_script=[False, False, KeyboardInterrupt()],
        store=store,
        cleanup_interval=10**9,
    )
    assert len(observed) >= 1
    assert observed[-1]["queue_depth"] == 1
    assert store.cleanup_calls  # 异常后的空闲周期恢复执行过清理


def test_eval_worker_cleanup_throttled_by_interval(monkeypatch):
    """cleanup 只按 monotonic 间隔触发：默认日频下多轮空闲仅清理一次。"""
    store = _FakeEvalWorkerStore()
    _observed, store = _run_eval_worker(
        monkeypatch,
        process_once_script=[False, False, False, KeyboardInterrupt()],
        store=store,
        cleanup_interval=10**9,  # 远大于测试时长 → 只有启动那次
    )
    assert len(store.cleanup_calls) == 1


def test_eval_worker_cleanup_every_idle_when_interval_zero(monkeypatch):
    """间隔 0 → 每个空闲周期都清理（校准/压测用极端配置）。"""
    store = _FakeEvalWorkerStore()
    _observed, store = _run_eval_worker(
        monkeypatch,
        process_once_script=[False, False, KeyboardInterrupt()],
        store=store,
        cleanup_interval=0,
    )
    assert len(store.cleanup_calls) >= 2


def test_eval_worker_processes_jobs_between_failures(monkeypatch):
    """异常轮与正常轮交错：任务照常计数（对齐发布 Worker 容错口径）。"""
    store = _FakeEvalWorkerStore()
    # 第1轮抛异常、第2轮处理成功、第3轮空闲、第4轮退出
    observed, _store = _run_eval_worker(
        monkeypatch,
        process_once_script=[RuntimeError("boom"), True, False, KeyboardInterrupt()],
        store=store,
        cleanup_interval=10**9,
    )
    assert observed  # 异常后循环未死，空闲分支恢复了指标上报
