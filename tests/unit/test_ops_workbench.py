"""P2-3 坐席最小工作台验收：领取（并发安全）/ SLA 超时 / 处理留痕 / UI 契约。

- 领取：Redis Lua 与进程内 RLock 两条实现都验证「两方同时领取只有一个成功」；
- SLA：创建时间起算 HANDOFF_SLA_SECONDS，超时判定 + 首次观测去重 + 指标累加；
- 留痕：claim / note / resolve 全部进 events（actor + 时间），可查；
- 端点契约：claim / notes / list(mine) / resolve 返回工单与留痕；
- UI：ops.html / ops.js 模板能渲染（静态资源可服务 + JS 语法检查）。
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path

import fakeredis
import pytest
from fastapi.testclient import TestClient

from app.handoff.board import (
    HANDOFF_SLA_SECONDS,
    HandoffConflict,
    HandoffNotFound,
    HandoffTicket,
    InProcessHandoffBoard,
    RedisHandoffBoard,
    _now,
)
from app.observability.metrics import HANDOFF_SLA_BREACH
from app.security.ratelimit import UsageTracker, UserLimiter
from app.stores.locks import SessionLockManager
from test_server_api import _FakeComponents, _ScriptedAgent

ROOT = Path(__file__).parents[2]


def _old_time(seconds_ago: int = HANDOFF_SLA_SECONDS + 60) -> str:
    return (
        datetime.now() - timedelta(seconds=seconds_ago)
    ).isoformat(timespec="seconds")


def _board_factories():
    return [
        ("in-process", lambda: InProcessHandoffBoard()),
        ("redis", lambda: RedisHandoffBoard(fakeredis.FakeStrictRedis())),
    ]


def _components():
    comps = _FakeComponents()
    comps.handoff_board = InProcessHandoffBoard()
    comps.locks = SessionLockManager(None, ttl_seconds=60)
    comps.limiter = UserLimiter(None, max_rps=0, daily_token_budget=0)
    comps.usage_tracker = UsageTracker()
    return comps


def _make_client(monkeypatch, comps=None):
    import app.server.main as main_mod

    comps = comps or _components()
    monkeypatch.setattr(main_mod, "build_pod_components", lambda: comps)
    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda uid, sid="", components=None, **_k: _ScriptedAgent(uid, sid),
    )
    return TestClient(main_mod.create_app()), comps


def _ticket(ticket_id="t-1", **overrides):
    base = dict(
        ticket_id=ticket_id, user_id="u1", session_id="s1", intent="complaint",
        question="怎么还没到", reply="已为您转接人工", created_at=_now(),
    )
    base.update(overrides)
    return HandoffTicket(**base)


# ============================================================
# 领取：并发安全（两方同时领取只有一个成功）
# ============================================================
@pytest.mark.parametrize("name,factory", _board_factories())
def test_claim_concurrent_only_one_wins(name, factory):
    board = factory()
    board.create(_ticket())
    barrier = threading.Barrier(2)
    results: list[tuple[str, str]] = []
    lock = threading.Lock()

    def worker(actor: str):
        barrier.wait()  # 两线程尽量同时进入 claim
        try:
            board.claim("t-1", actor)
            outcome = "ok"
        except HandoffConflict:
            outcome = "conflict"
        with lock:
            results.append((actor, outcome))

    threads = [threading.Thread(target=worker, args=(a,)) for a in ("alice", "bob")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    outcomes = sorted(o for _, o in results)
    assert outcomes == ["conflict", "ok"], f"{name}: {results}"
    ticket = board.get("t-1")
    assert ticket.assignee in ("alice", "bob")
    claimed_events = [e for e in ticket.events if e["action"] == "claimed"]
    assert len(claimed_events) == 1  # 失败方不写留痕
    assert claimed_events[0]["actor"] == ticket.assignee


@pytest.mark.parametrize("name,factory", _board_factories())
def test_claim_idempotent_and_conflicts(name, factory):
    board = factory()
    board.create(_ticket())
    ticket, already = board.claim("t-1", "alice")
    assert (ticket.assignee, already) == ("alice", False)
    assert ticket.claimed_at
    # 同一坐席重复领取：幂等，不重复追加留痕
    ticket2, already2 = board.claim("t-1", "alice")
    assert (ticket2.assignee, already2) == ("alice", True)
    assert [e["action"] for e in ticket2.events].count("claimed") == 1
    # 他人领取 → 409
    with pytest.raises(HandoffConflict) as ei:
        board.claim("t-1", "bob")
    assert "alice" in str(ei.value)
    # 不存在 → 404
    with pytest.raises(HandoffNotFound):
        board.claim("missing", "alice")
    # 已解决 → 冲突
    board.resolve_atomic("t-1", {"note": "done"}, "alice")
    with pytest.raises(HandoffConflict):
        board.claim("t-1", "bob")


# ============================================================
# SLA：创建时间起算 / 超时判定 / 首次观测去重
# ============================================================
@pytest.mark.parametrize("name,factory", _board_factories())
def test_sla_breach_detection_and_dedupe(name, factory):
    board = factory()
    board.create(_ticket("old", created_at=_old_time()))
    board.create(_ticket("fresh"))
    # 首次观测：只有旧工单超时
    assert board.mark_sla_breaches() == ["old"]
    # 重复观测不重复计数
    assert board.mark_sla_breaches() == []

    old = board.get("old").to_ops_dict()
    assert old["sla_breached"] is True
    assert old["sla_seconds"] == HANDOFF_SLA_SECONDS
    assert old["sla_due_at"]
    assert old["sla_remaining_seconds"] < 0
    fresh = board.get("fresh").to_ops_dict()
    assert fresh["sla_breached"] is False
    assert fresh["sla_remaining_seconds"] > 0


@pytest.mark.parametrize("name,factory", _board_factories())
def test_sla_resolved_late_breached_resolved_in_time_not(name, factory):
    board = factory()
    # 迟到解决：创建于 SLA 之前，解决时间 = 现在 → 超时
    board.create(_ticket("late", created_at=_old_time()))
    board.resolve_atomic("late", {"note": "迟处理"}, "alice")
    assert board.get("late").sla_breached() is True
    assert board.mark_sla_breaches() == ["late"]
    # 及时解决：创建/解决都在窗口内 → 不超时
    board.create(_ticket("intime"))
    board.resolve_atomic("intime", {"note": "及时"}, "alice")
    assert board.get("intime").sla_breached() is False
    assert board.mark_sla_breaches() == []


# ============================================================
# 处理留痕：谁在何时领取 / 备注 / 解决
# ============================================================
@pytest.mark.parametrize("name,factory", _board_factories())
def test_ticket_trail_records_claim_note_resolve(name, factory):
    board = factory()
    board.create(_ticket())
    board.claim("t-1", "alice", now="2026-09-01T10:00:00")
    board.add_note("t-1", "bob", "已电话联系用户", now="2026-09-01T10:05:00")
    board.resolve_atomic(
        "t-1", {"note": "补偿 10 元券"}, "alice", now="2026-09-01T10:10:00",
    )
    ticket = board.get("t-1")
    assert [e["action"] for e in ticket.events] == ["claimed", "note", "resolved"]
    assert ticket.events[0] == {
        "action": "claimed", "actor": "alice", "at": "2026-09-01T10:00:00", "note": "",
    }
    assert ticket.events[1]["actor"] == "bob"
    assert ticket.events[1]["note"] == "已电话联系用户"
    assert ticket.events[2]["actor"] == "alice"
    assert ticket.events[2]["note"] == "补偿 10 元券"
    assert ticket.resolved_at == "2026-09-01T10:10:00"
    assert ticket.assignee == "alice"  # 领取人保留
    # 备注不改变状态
    assert ticket.status == "resolved"


@pytest.mark.parametrize("name,factory", _board_factories())
def test_unclaimed_resolve_records_actor_as_assignee(name, factory):
    board = factory()
    board.create(_ticket())
    board.resolve_atomic("t-1", {"note": "直接处理"}, "carol")
    ticket = board.get("t-1")
    assert ticket.assignee == "carol"       # 未领取直接解决：处理人即解决人
    assert ticket.claimed_at == ticket.resolved_at
    assert ticket.events[-1]["action"] == "resolved"


def test_note_on_missing_ticket_raises():
    for name, factory in _board_factories():
        board = factory()
        with pytest.raises(HandoffNotFound):
            board.add_note("missing", "alice", "x")


def test_resolve_retries_when_claim_races(monkeypatch):
    """解决与领取交错：Lua 返回 RETRY，重读后领取状态不被覆盖（Redis）。"""
    board = RedisHandoffBoard(fakeredis.FakeStrictRedis())
    board.create(_ticket())
    board.claim("t-1", "alice")
    real_get = board.get
    calls = {"n": 0}

    def stale_get(ticket_id):
        calls["n"] += 1
        if calls["n"] == 1:
            stale = real_get(ticket_id)  # 模拟 resolve 快照发生在 claim 之前
            stale.assignee = ""
            stale.claimed_at = ""
            stale.events = []
            return stale
        return real_get(ticket_id)

    monkeypatch.setattr(board, "get", stale_get)
    event, duplicate = board.resolve_atomic("t-1", {"note": "done"}, "bob")
    assert calls["n"] >= 2            # 触发 RETRY 后重读最新状态
    assert duplicate is False
    ticket = real_get("t-1")
    assert ticket.assignee == "alice"  # 并发领取未被解决覆盖
    assert ticket.resolved_by == "bob"
    assert [e["action"] for e in ticket.events] == ["claimed", "resolved"]
    assert event["resolved_by"] == "bob"


# ============================================================
# 端点契约：claim / notes / list(mine) / resolve
# ============================================================
def test_ops_claim_and_mine_endpoints(monkeypatch):
    client, comps = _make_client(monkeypatch)
    with client:
        comps.handoff_board.create(_ticket("t-1"))
        comps.handoff_board.create(_ticket("t-2"))  # 未领取

        resp = client.post("/v1/handoffs/t-1/claim")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["assignee"] == "anonymous"  # auth 关闭 → 匿名主体（不取请求体）
        assert body["already"] is False
        assert body["ticket"]["assignee"] == "anonymous"
        assert body["ticket"]["claimed_at"]

        # 幂等重复领取
        again = client.post("/v1/handoffs/t-1/claim").json()
        assert again["already"] is True

        # 我的列表只含已领取工单
        mine = client.get("/v1/handoffs", params={"mine": "true"}).json()
        assert [t["ticket_id"] for t in mine["tickets"]] == ["t-1"]
        assert mine["assignee"] == "anonymous"
        all_pending = client.get("/v1/handoffs").json()
        assert {t["ticket_id"] for t in all_pending["tickets"]} == {"t-1", "t-2"}

        # 他人已领取 → 409；不存在 → 404
        comps.handoff_board.get("t-1").assignee = "someone-else"
        assert client.post("/v1/handoffs/t-1/claim").status_code == 409
        assert client.post("/v1/handoffs/missing/claim").status_code == 404


def test_ops_note_endpoint_appends_trail(monkeypatch):
    client, comps = _make_client(monkeypatch)
    with client:
        comps.handoff_board.create(_ticket("t-1"))
        client.post("/v1/handoffs/t-1/claim")

        resp = client.post("/v1/handoffs/t-1/notes", json={"note": "已电话联系用户"})
        assert resp.status_code == 200, resp.text
        event = resp.json()["event"]
        assert event["action"] == "note"
        assert event["actor"] == "anonymous"
        assert event["note"] == "已电话联系用户"
        assert [e["action"] for e in resp.json()["events"]] == ["claimed", "note"]

        # 空备注 422；工单不存在 404
        assert client.post(
            "/v1/handoffs/t-1/notes", json={"note": ""}
        ).status_code == 422
        assert client.post(
            "/v1/handoffs/missing/notes", json={"note": "x"}
        ).status_code == 404


def test_ops_list_exposes_sla_fields_and_metric_counts_once(monkeypatch):
    client, comps = _make_client(monkeypatch)
    with client:
        comps.handoff_board.create(_ticket("t-old", created_at=_old_time()))
        before = HANDOFF_SLA_BREACH._value.get()

        first = client.get("/v1/handoffs").json()
        ticket = first["tickets"][0]
        assert ticket["sla_breached"] is True
        assert ticket["sla_due_at"] and ticket["sla_remaining_seconds"] < 0
        assert ticket["sla_seconds"] == HANDOFF_SLA_SECONDS
        assert HANDOFF_SLA_BREACH._value.get() == before + 1

        # 再次轮询：不重复计数
        client.get("/v1/handoffs")
        assert HANDOFF_SLA_BREACH._value.get() == before + 1


def test_ops_resolve_returns_trail_and_assignee(monkeypatch):
    client, comps = _make_client(monkeypatch)
    with client:
        comps.handoff_board.create(_ticket("t-1"))
        client.post("/v1/handoffs/t-1/claim")
        client.post("/v1/handoffs/t-1/notes", json={"note": "处理中"})
        resp = client.post(
            "/v1/handoffs/t-1/resolve",
            json={"resolution": {"note": "补偿 10 元券"}, "reclaim": True},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "resolved"
        assert body["assignee"] == "anonymous"
        assert [e["action"] for e in body["ticket"]["events"]] == [
            "claimed", "note", "resolved",
        ]
        assert body["ticket"]["resolved_at"]
        # 已解决工单不再出现在 pending
        assert client.get("/v1/handoffs").json()["tickets"] == []
        resolved = client.get(
            "/v1/handoffs", params={"status": "resolved"}
        ).json()["tickets"]
        assert [t["ticket_id"] for t in resolved] == ["t-1"]


# ============================================================
# UI 模板契约（无法手工点击：至少保证可服务 + 语法可解析 + 关键交互存在）
# ============================================================
def test_ops_workbench_ui_contract(monkeypatch):
    client, _comps = _make_client(monkeypatch)
    with client:
        html = client.get("/ops.html").text
        js = client.get("/js/ops.js").text
        assert 'id="ticketScope"' in html        # 全部/我的 过滤
        assert 'id="ticketBox"' in html
        assert "data-claim" in js                # 领取按钮（事件委托）
        assert "/claim`" in js or "/claim\"" in js
        assert "/notes" in js                    # 处理备注
        assert "sla_breached" in js              # 超时标红
        assert "已转人工" in js or "工单" in js
        # 模板仍是 4 个 tab（既有结构契约不回归）
        assert html.count('role="tab"') == 4


def test_ops_js_syntax_check():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node executable unavailable")
    result = subprocess.run(
        [node, "--check", str(ROOT / "app/server/static/js/ops.js")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
