"""RAG 端到端集成测试（原 tests/test_rag.py 的 pytest 化）。

分层（RAG 修复计划·1）：
- 普通 pytest / PR CI 只跑 tests/unit；本文件由专属 RAG E2E job 显式执行，
  且必须先构建索引并注入真实 embedding/LLM 配置；
- 是否“必须通过”由显式 `RAG_E2E_REQUIRED=true` 决定（不再依赖通用 CI=true）。

断言（加强）：
- search_knowledge.success 必须为 true；
- 必须返回预期知识文档（而非仅非空）；
- Agent 测试通过调用记录确认实际调用了 search_knowledge。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config.settings import settings

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_DOCS = {"退换货政策", "配送说明", "会员权益", "常见问题FAQ"}


def _backend_name() -> str:
    return os.environ.get("TEST_RAG_BACKEND", "numpy").lower()


def _required() -> bool:
    """显式开关：只有 RAG_E2E_REQUIRED=true 时才把前置条件缺失当失败。"""
    return os.environ.get("RAG_E2E_REQUIRED", "").strip().lower() == "true"


def _skip_or_fail(msg: str):
    if _required():
        pytest.fail(msg)
    pytest.skip(msg)


def _needs_embedding():
    if not (settings.openai_api_key or settings.sophnet_api_key):
        _skip_or_fail("缺少 embedding key（OPENAI_API_KEY / SOPHNET_API_KEY）")


def test_chunking_covers_expected_docs():
    from app.agent.rag.parsers import chunk_kb_dir

    chunks = chunk_kb_dir(ROOT / settings.kb_dir)
    assert chunks, "未切到任何 chunk"
    docs = {c.doc for c in chunks}
    missing = EXPECTED_DOCS - docs
    if missing:
        _skip_or_fail(f"当前知识库缺少期望文档（内容演进）：{missing}")
    assert all(c.heading_path for c in chunks)


def test_retriever_returns_expected_documents_and_tool_success():
    _needs_embedding()
    from app.agent.rag.retriever_factory import (
        open_retriever,
        retrieval_config_from_settings,
    )
    from app.agent.tools import knowledge as knowledge_tool

    name = _backend_name()
    settings.rag_backend = name
    knowledge_tool.reset_retriever()
    try:
        if name == "numpy" and not Path(ROOT / settings.kb_index_path).exists():
            _skip_or_fail("numpy 索引不存在：请先运行 build_kb_index")
        try:
            open_retriever(retrieval_config_from_settings())
        except FileNotFoundError as e:
            _skip_or_fail(f"索引不可用: {e}")

        result = knowledge_tool.search_knowledge("七天无理由退货的运费怎么算", top_k=3)
        assert result.get("success") is True, f"search_knowledge 失败: {result.get('error')}"
        assert result.get("backend") == name
        docs = {r.get("doc") for r in result["results"]}
        assert docs, "结果为空"
        # 必须命中预期知识文档，而不是仅返回非空
        assert docs & EXPECTED_DOCS, f"未命中预期文档：{docs}"
    finally:
        knowledge_tool.reset_retriever()


def test_agent_actually_calls_search_knowledge():
    _needs_embedding()
    from openai import AuthenticationError

    from app.agent.chat import EcomAgent
    from app.agent.tools import knowledge as knowledge_tool

    settings.rag_backend = _backend_name()
    knowledge_tool.reset_retriever()
    session_path = ROOT / "app" / "sessions" / "test_rag_e2e.json"
    agent = EcomAgent(memory_enabled=False, use_mcp=False,
                      session_path=str(session_path))
    try:
        try:
            resp = agent.chat("七天无理由退货的运费怎么算？")
        except AuthenticationError as e:
            _skip_or_fail(f"embedding/LLM key 无效: {e}")
        assert resp.reply
        # 通过调用记录确认实际调用了 search_knowledge（不能只看回复非空）
        called = any(
            m.get("role") == "assistant" and m.get("tool_calls")
            and any(
                tc.get("function", {}).get("name") == "search_knowledge"
                for tc in m["tool_calls"]
            )
            for m in agent.raw_messages
        )
        assert called, "Agent 未实际调用 search_knowledge（调用记录缺失）"
    finally:
        agent.close()
        session_path.unlink(missing_ok=True)
        knowledge_tool.reset_retriever()
