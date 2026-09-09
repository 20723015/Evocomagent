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
