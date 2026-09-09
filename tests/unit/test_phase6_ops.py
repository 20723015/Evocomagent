"""阶段六 6.1~6.4 单测：Handoff 板、Redis evolution 锁、知识时效扫描、review 端点。"""

from __future__ import annotations

from typing import ClassVar

import fakeredis
import pytest

from app.handoff.board import (
    HandoffTicket,
    InProcessHandoffBoard,
    RedisHandoffBoard,
    build_handoff_ticket,
)
from app.review.scan import parse_frontmatter, scan_expired_knowledge


def _make_redis():
    return fakeredis.FakeRedis(server=fakeredis.FakeServer())


def _ticket(tid="t-1", uid="u1", sid="s1"):
    return HandoffTicket(
        ticket_id=tid, user_id=uid, session_id=sid,
        intent="complaint", summary="用户投诉物流慢", question="怎么还没到",
        reply="已为您转接人工",
    )


# ============================================================
# 6.1 Handoff 板
# ============================================================
def test_handoff_board_roundtrip_in_process():
    board = InProcessHandoffBoard()
    board.create(_ticket())
    assert board.get("t-1").status == "pending"
    assert [t.ticket_id for t in board.list("pending")] == ["t-1"]
    resolved = board.resolve("t-1", {"action": "补偿优惠券"})
    assert resolved.status == "resolved"
    assert resolved.resolution == {"action": "补偿优惠券"}
    assert board.list("pending") == []


def test_handoff_board_roundtrip_redis():
    board = RedisHandoffBoard(_make_redis())
    board.create(_ticket())
    assert board.get("t-1").user_id == "u1"
    assert [t.ticket_id for t in board.list("pending")] == ["t-1"]
    board.resolve("t-1", {"action": "退款"})
    assert board.list("pending") == []
    assert board.get("t-1").status == "resolved"


def test_build_ticket_from_agent_like_object():
    class Agent:
        user_id = "u1"
        session_id = "s1"
        summary = "摘要"
        raw_messages: ClassVar[list[dict[str, str]]] = [
            {"role": "user", "content": "我要投诉"},
            {"role": "assistant", "content": "..."},
        ]

    class Result:
        intent = "complaint"
        reply = "已转人工"
        follow_up_question = "请描述问题\n请补充订单号"

    ticket = build_handoff_ticket(Agent(), Result())
    assert ticket.user_id == "u1"
    assert ticket.suggested_actions == ["请描述问题", "请补充订单号"]


def test_handoff_create_resolve_endpoints(monkeypatch):
    from fastapi.testclient import TestClient
    from test_server_api import _FakeComponents, _ScriptedAgent

    import app.server.main as main_mod

    monkeypatch.setattr(main_mod, "build_pod_components", lambda: _FakeComponents())
    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda uid, sid="", comps=None, **_k: _ScriptedAgent(uid, sid),
    )
    with TestClient(main_mod.create_app()) as client:
        resp = client.post("/v1/handoffs", json={"user_id": "u1", "session_id": "s1"})
        assert resp.status_code == 200
        ticket_id = resp.json()["ticket_id"]

        listed = client.get("/v1/handoffs", params={"status": "pending"}).json()
        assert any(t["ticket_id"] == ticket_id for t in listed["tickets"])

        resolved = client.post(
            f"/v1/handoffs/{ticket_id}/resolve",
            json={"user_id": "u1", "resolution": {"action": "退款"}, "reclaim": True},
        )
        assert resolved.json()["status"] == "resolved"
        assert client.get("/v1/handoffs", params={"status": "pending"}).json()["tickets"] == []


def test_chat_requires_human_creates_handoff_ticket(monkeypatch):
    from conftest import sample_response
    from fastapi.testclient import TestClient
    from test_server_api import _FakeComponents, _ScriptedAgent

    import app.server.main as main_mod

    class _HumanAgent(_ScriptedAgent):
        def chat(self, message: str):
            return sample_response(reply="已转人工", requires_human=True)

    monkeypatch.setattr(main_mod, "build_pod_components", lambda: _FakeComponents())
    monkeypatch.setattr(
        main_mod, "build_agent",
        lambda uid, sid="", comps=None, **_k: _HumanAgent(uid, sid),
    )
    with TestClient(main_mod.create_app()) as client:
        resp = client.post("/v1/chat", json={"user_id": "u1", "session_id": "s1", "message": "我要投诉"})
        assert resp.status_code == 200
        assert resp.json()["requires_human"] is True
        tickets = client.get("/v1/handoffs", params={"status": "pending"}).json()["tickets"]
        assert len(tickets) == 1
        assert tickets[0]["user_id"] == "u1"
        assert tickets[0]["intent"] == "order_query"  # sample_response 默认 intent


# ============================================================
# 6.2 Redis evolution 锁
# ============================================================
def test_redis_evolution_lock_mutual_exclusion():
    from app.evolution.lock import LockHeldError, RedisLockGuard

    redis = _make_redis()
    g1 = RedisLockGuard(redis, key="run_evolution", ttl_seconds=3600)
    g2 = RedisLockGuard(redis, key="run_evolution", ttl_seconds=3600)
    g1.acquire(phase="run")
    with pytest.raises(LockHeldError):
        g2.acquire(phase="run")  # 另一 pod 在跑 → 本轮跳过
    g1.release()
    g2.acquire(phase="approve")  # 释放后可获取
    g2.release()


def test_redis_evolution_lock_wrong_token_cannot_release_others():
    from app.evolution.lock import LockHeldError, RedisLockGuard

    redis = _make_redis()
    g1 = RedisLockGuard(redis, key="k", ttl_seconds=3600)
    g2 = RedisLockGuard(redis, key="k", ttl_seconds=3600)
    g1.acquire()
    g2.release()  # 从未持有 → no-op（token 也必然不匹配）
    with pytest.raises(LockHeldError):
        RedisLockGuard(redis, key="k", ttl_seconds=3600).acquire()
    g1.release()


# ============================================================
# 6.4 知识时效扫描
# ============================================================
def test_parse_frontmatter_from_publisher():
    text = """---
provenance: abcd1234
owner: system
---

# 自进化知识
"""
    meta = parse_frontmatter(text)
    assert meta["provenance"] == "abcd1234"
    assert meta["owner"] == "system"


def test_scan_expired_knowledge(tmp_path):
    kb = tmp_path / "kb"
    (kb / "md").mkdir(parents=True)
    (kb / "md" / "旧政策.md").write_text(
        "---\neffective_date: 2020-01-01\nowner: ops\n---\n# 旧政策\n", encoding="utf-8",
    )
    (kb / "md" / "新政策.md").write_text(
        "---\neffective_date: 2026-01-01\n---\n# 新政策\n", encoding="utf-8",
    )
    (kb / "md" / "无标注.md").write_text("# 无标注\n", encoding="utf-8")

    scanned = scan_expired_knowledge(kb / "md", aging_days=365)
    by_status = {d["path"]: d["status"] for d in scanned}
    assert by_status["旧政策.md"] == "expired"
    assert by_status["新政策.md"] == "valid"  # 距今 <365 天 → 有效
    assert by_status["无标注.md"] == "unknown"  # 缺 effective_date → 提示补全


# ============================================================
# 6.3 审核后台端点（安全修复 P1：认证 + CSRF + XSS 转义）
# ============================================================
def _review_login(client, monkeypatch, token="admin-token-0123456789abcdef"):
    """配置管理令牌并完成登录，返回（csrf token）以及登录后的 client。"""
    from app.config.settings import settings

    monkeypatch.setattr(settings, "review_admin_token", token, raising=False)
    resp = client.post("/login", data={"token": token}, follow_redirects=False)
    assert resp.status_code == 303
    # 从首页表单取 CSRF token
    index_html = client.get("/").text
    marker = "name='csrf' value='"
    assert marker in index_html
    csrf = index_html.split(marker, 1)[1].split("'", 1)[0]
    return csrf


def test_review_webapp_fail_closed_without_token(monkeypatch):
    """安全修复 P1：REVIEW_ADMIN_TOKEN 未配置 → 整站 503（公开写入口关闭）。"""
    from fastapi.testclient import TestClient

    from app.config.settings import settings
    from app.review import webapp as wmod

    monkeypatch.setattr(settings, "review_admin_token", "", raising=False)
    with TestClient(wmod.create_review_app()) as client:
        assert client.get("/").status_code == 503
        assert client.get("/aging").status_code == 503
        assert client.post("/approve/cid-1").status_code == 503
        assert client.post("/login", data={"token": "x"}).status_code == 503


def test_review_webapp_requires_login(monkeypatch):
    from fastapi.testclient import TestClient

    from app.config.settings import settings
    from app.review import webapp as wmod

    monkeypatch.setattr(settings, "review_admin_token", "admin-token-0123456789abcdef",
                        raising=False)
    with TestClient(wmod.create_review_app()) as client:
        assert client.get("/").status_code == 401
        assert client.get("/aging").status_code == 401
        assert client.post("/approve/cid-1").status_code == 401


def test_review_webapp_approve_reject(monkeypatch):
    """审核后台走 ReviewService（不再借道 CLI 返回码）；结构化状态返回。"""
    from fastapi.testclient import TestClient

    from app.review import webapp as wmod

    calls = []

    class _StubService:
        def approve(self, cid, revision):
            calls.append(("approve", cid, revision))
            if cid == "cid-dup":
                return {"status": "duplicate_rejected",
                        "detail": "审核时发现重复，候选已终结（rejected）"}
            return {"status": "published", "filename": "x.md"}

        def reject(self, cid, revision=None):
            calls.append(("reject", cid, revision))
            return {"status": "rejected"}

        def edit(self, cid, question, answer, revision):
            calls.append(("edit", cid, revision))
            return {"status": "edited", "revision": revision + 1}

    monkeypatch.setattr(wmod, "_pending_entries", list)
    monkeypatch.setattr(wmod, "_review_service", lambda: _StubService())
    with TestClient(wmod.create_review_app()) as client:
        csrf = _review_login(client, monkeypatch)
        assert client.get("/").status_code == 200
        # 缺 CSRF token 的 POST 一律 403（跨站表单防护）
        resp = client.post("/approve/cid-1")
        assert resp.status_code == 403
        resp = client.post("/approve/cid-1", data={"csrf": csrf,
                                                   "expected_revision": "0"})
        assert resp.status_code == 200
        assert resp.json()["published"] is True
        resp = client.post("/approve/cid-dup", data={"csrf": csrf,
                                                     "expected_revision": "0"})
        assert resp.status_code == 200
        assert resp.json()["published"] is False  # 重复绝不虚报发布成功
        resp = client.post("/reject/cid-2", data={"csrf": csrf})
        assert resp.status_code == 200
        assert resp.json()["rejected"] is True
        # 编辑成功 → 303 回列表页
        resp = client.post("/edit/cid-1", data={"csrf": csrf,
                                                "expected_revision": "2",
                                                "question": "q2", "answer": "a2"},
                           follow_redirects=False)
        assert resp.status_code == 303  # 编辑成功 → 重定向回列表页
        # CSRF token 错误 → 403
        assert client.post("/reject/cid-3", data={"csrf": "forged"}).status_code == 403
        assert calls[0] == ("approve", "cid-1", 0)


def test_review_webapp_xss_escaped(monkeypatch):
    """安全修复 P1：候选问题/答案（来自用户对话）必须 HTML 转义。"""
    from fastapi.testclient import TestClient

    from app.review import webapp as wmod

    payload = "<script>alert(1)</script>"
    monkeypatch.setattr(
        wmod, "_pending_entries",
        lambda: [("cid-1", {"question": payload, "answer": payload, "status": "pending"})],
    )
    with TestClient(wmod.create_review_app()) as client:
        _review_login(client, monkeypatch)
        html = client.get("/").text
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "<script>alert(1)</script>" not in html
