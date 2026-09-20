"""P1-3 查询改写（查询侧双路召回）：改写器 fail-open + 子查询归因。

覆盖：
- 改写器正常路径（去前缀/取单行）与全部 fail-open 分支（异常/空输出/等价
  输出/不可信输出/零预算），任何失败都必须返回 None 而不是抛异常；
- 未配置 API key → 不建客户端（零网络）；
- knowledge 接线：原 query 必居首、改写追加、≤3 不挤掉模型子查询、
  去重；双路合流后子查询归因与 hits 严格平行。
"""

from __future__ import annotations

import pytest

from app.agent.rag.query_rewrite import (
    QueryRewriter,
    create_query_rewriter,
    get_query_rewriter,
    reset_query_rewriter,
    rewrite_query,
)
from app.config.settings import settings


class _FakeRewriter:
    """脚本化改写器替身（直接替换 knowledge 的改写入口）。"""

    def __init__(self, result: str | None):
        self.result = result
        self.calls: list[str] = []

    def rewrite(self, query, *, timeout=None):
        self.calls.append(query)
        return self.result


# ------------------------------------------------------------
# 改写器：正常路径
# ------------------------------------------------------------
def test_rewrite_ok_and_prompt_carries_original():
    from conftest import FakeChatClient

    client = FakeChatClient()
    client.enqueue("七天无理由退货的运费由谁承担")
    rewriter = QueryRewriter(client, "cheap-model")
    out = rewriter.rewrite("这个退货运费谁出")
    assert out == "七天无理由退货的运费由谁承担"
    kind, kwargs = client.calls[0]
    assert kind == "chat"
    assert kwargs["model"] == "cheap-model"
    assert kwargs["temperature"] == 0.0
    assert "这个退货运费谁出" in kwargs["messages"][1]["content"]
    # 首句含「提取结构化」→ llm.client 归入 extract 用途（辅助调用治理口径）
    assert "提取结构化" in kwargs["messages"][0]["content"]


def test_rewrite_strips_prefix_and_takes_first_line():
    from conftest import FakeChatClient

    client = FakeChatClient()
    client.enqueue("改写：退货运费承担方\n补充说明不该出现")
    rewriter = QueryRewriter(client, "m")
    assert rewriter.rewrite("运费谁出") == "退货运费承担方"


# ------------------------------------------------------------
# 改写器：fail-open（异常/空/等价/不可信/零预算）
# ------------------------------------------------------------
def test_rewrite_exception_fail_open():
    from conftest import FakeChatClient

    client = FakeChatClient()
    client.enqueue_error(RuntimeError("gateway down"))
    rewriter = QueryRewriter(client, "m")
    assert rewriter.rewrite("这个能退吗") is None


def test_rewrite_empty_output_fail_open():
    from conftest import FakeChatClient

    for payload in ("", "   ", "\n\n"):
        client = FakeChatClient()
        client.enqueue(payload)
        assert QueryRewriter(client, "m").rewrite("这个能退吗") is None


def test_rewrite_identical_output_fail_open():
    from conftest import FakeChatClient

    client = FakeChatClient()
    client.enqueue("这个能退吗")
    assert QueryRewriter(client, "m").rewrite("这个能退吗") is None


def test_rewrite_meta_commentary_leak_fail_open():
    """模型把思考/指令复述写进可见输出 → 判为未改写（绝不把说明文字当 query）。"""
    from conftest import FakeChatClient

    for payload in (
        "根据指令，原问题已自足，原样输出。今天天气怎么样",
        "无需改写，保持原样",
        "作为一个AI，我无法改写该查询",
    ):
        client = FakeChatClient()
        client.enqueue(payload)
        assert QueryRewriter(client, "m").rewrite("今天天气怎么样") is None


def test_rewrite_implausible_output_fail_open():
    """模型跑偏成长回答（超长）→ 丢弃，绝不当 query 用。"""
    from conftest import FakeChatClient

    client = FakeChatClient()
    client.enqueue("很抱歉，" + "退" * 500)
    assert QueryRewriter(client, "m").rewrite("这个能退吗") is None


def test_rewrite_zero_budget_skips_without_calling_llm():
    from conftest import FakeChatClient

    client = FakeChatClient()
    rewriter = QueryRewriter(client, "m")
    assert rewriter.rewrite("这个能退吗", timeout=0.0) is None
    assert client.calls == []  # 预算为 0 不发起调用


def test_rewrite_query_entry_fail_open_on_factory_error(monkeypatch):
    """单例装配异常也必须 fail-open（返回 None），不冒泡进检索热路径。"""
    import app.agent.rag.query_rewrite as qr

    monkeypatch.setattr(qr, "get_query_rewriter", lambda: (_ for _ in ()).throw(
        RuntimeError("boom")))
    assert rewrite_query("这个能退吗") is None


def test_create_query_rewriter_without_api_key_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "")
    reset_query_rewriter()
    assert create_query_rewriter() is None
    assert get_query_rewriter() is None


# ------------------------------------------------------------
# knowledge 接线：子查询规整契约 + 双路归因
# ------------------------------------------------------------
def _scripted_retriever(script):
    from app.agent.rag.retriever import SCORE_SOURCE_VECTOR, RetrievalResult

    class _R:
        model = "fake"
        size = 0

        def load(self):
            pass

        def search_with_status(self, query, top_k=3, timeout=None):
            return RetrievalResult(
                hits=list(script.get(query, []))[:top_k],
                score_source=SCORE_SOURCE_VECTOR,
            )

    return _R()


def _hit(chunk_id, doc, parent_id, score=0.9):
    from app.agent.rag.backends.base import RetrievedChunk
    from app.agent.rag.chunker import Chunk

    return RetrievedChunk(
        chunk=Chunk(chunk_id=chunk_id, doc=doc, section="s", text="正文",
                    parent_id=parent_id, source_path=f"{doc}.md"),
        score=score,
    )


def test_resolve_subqueries_appends_rewrite(monkeypatch):
    from app.agent.tools import knowledge as knowledge_mod

    monkeypatch.setattr(
        "app.agent.rag.query_rewrite.rewrite_query",
        lambda q, timeout=None: "改写后的检索问题",
    )
    assert knowledge_mod._resolve_subqueries("这个能退吗", None, None) == [
        "这个能退吗", "改写后的检索问题",
    ]


def test_resolve_subqueries_does_not_evict_model_subqueries(monkeypatch):
    """模型已给满 ≤3 条 → 不改写（不挤掉模型显式提交的子查询）。"""
    from app.agent.tools import knowledge as knowledge_mod

    called = []
    monkeypatch.setattr(
        "app.agent.rag.query_rewrite.rewrite_query",
        lambda q, timeout=None: called.append(q) or "改写",
    )
    assert knowledge_mod._resolve_subqueries("主查询", ["a", "b"], None) == [
        "主查询", "a", "b",
    ]
    assert called == []  # 无空位：连改写调用都不发起
    # 还有空位：改写追加在末尾，原 query 仍居首
    assert knowledge_mod._resolve_subqueries("主查询", ["a"], None) == [
        "主查询", "a", "改写",
    ]


def test_resolve_subqueries_fail_open_keeps_list(monkeypatch):
    from app.agent.tools import knowledge as knowledge_mod

    monkeypatch.setattr(
        "app.agent.rag.query_rewrite.rewrite_query",
        lambda q, timeout=None: None,  # 未配置/失败/等价
    )
    assert knowledge_mod._resolve_subqueries("这个能退吗", None, None) == [
        "这个能退吗",
    ]


def test_search_knowledge_dual_route_attribution(monkeypatch):
    """双路合流：原 query + 改写路并行召回，归因与 hits 严格平行。"""
    from app.agent.tools import knowledge as knowledge_mod

    retriever = _scripted_retriever({
        "这个能退吗": [_hit("m1", "会员权益", "pm")],
        "七天无理由退货政策": [_hit("r1", "退换货政策", "pr")],
    })
    monkeypatch.setattr(
        "app.agent.rag.query_rewrite.rewrite_query",
        lambda q, timeout=None: "七天无理由退货政策",
    )
    knowledge_mod.push_retriever_override(retriever)
    try:
        result = knowledge_mod.search_knowledge("这个能退吗", top_k=5)
    finally:
        knowledge_mod.pop_retriever_override()
    assert result["success"] is True
    assert result["subqueries"] == ["这个能退吗", "七天无理由退货政策"]
    assert {r["doc"] for r in result["results"]} == {"会员权益", "退换货政策"}


def test_multi_query_search_bundle_attributes_hits_to_subquery(monkeypatch):
    """pack.items 的 query 归因 = hit_queries（折叠后同步截断，严格平行）。"""
    from app.agent.tools import knowledge as knowledge_mod

    retriever = _scripted_retriever({
        "这个能退吗": [_hit("m1", "会员权益", "pm")],
        "七天无理由退货政策": [_hit("r1", "退换货政策", "pr")],
    })
    bundle = knowledge_mod._multi_query_search_bundle(
        retriever, ["这个能退吗", "七天无理由退货政策"], top_k=5, timeout=None,
    )
    assert [item.query for item in bundle.pack.items] == \
        bundle.outcome.hit_queries
    assert [item.query for item in bundle.pack.items] == \
        ["这个能退吗", "七天无理由退货政策"]


def test_search_knowledge_rewrite_skipped_when_disabled(monkeypatch):
    """未配置改写（无 key）→ 子查询保持原样，工具结果不出现改写路。"""
    from app.agent.tools import knowledge as knowledge_mod

    retriever = _scripted_retriever({"这个能退吗": [_hit("m1", "会员权益", "pm")]})
    knowledge_mod.push_retriever_override(retriever)
    try:
        result = knowledge_mod.search_knowledge("这个能退吗", top_k=5)
    finally:
        knowledge_mod.pop_retriever_override()
    assert result["subqueries"] == ["这个能退吗"]
