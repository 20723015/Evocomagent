"""第10期单元测试基建：全部 fakes 与 fixtures，单测全程无网络。

- FakeEmbedder：对文本做确定性哈希 → 32 维向量，共享字符多的文本余弦相似度高。
- FakeChatClient：脚本化 chat.completions.create / beta.chat.completions.parse，
  可注入超时/网络/坏 JSON 异常，并可记录每次调用的 kwargs。
- FakeBackend / FakeRetriever：内存版 VectorBackend / 检索器替身。
- 目录与时钟 fixture：tmp_state_dir / tmp_kb_dir / frozen_clock。
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta

import pytest

from app.agent.rag.backends.base import RetrievedChunk, VectorBackend
from app.agent.rag.chunker import Chunk
from app.config.settings import settings
from app.schemas.response import CustomerServiceResponse


# ============================================================
# FakeEmbedder
# ============================================================
class FakeEmbedder:
    """确定性哈希 embedder：相似文本（共享字符多）→ 高余弦相似度。"""

    def __init__(self, dim: int = 32, model: str = "fake-embedder"):
        self._model = model
        self._dim = dim

    @property
    def model(self) -> str:
        return self._model

    def encode(self, texts, timeout=None) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def encode_one(self, text: str, timeout=None) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * self._dim
        for ch in str(text):
            for d in range(self._dim):
                seed = hashlib.sha256(f"{ch}:{d}".encode()).digest()
                v[d] += (seed[0] / 255.0) - 0.5
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0:
            return [0.0] * self._dim
        return [x / norm for x in v]


# ============================================================
# FakeChatClient（脚本化 OpenAI 客户端替身）
# ============================================================
class _Choice:
    def __init__(self, message):
        self.message = message


class _Response:
    def __init__(self, choices):
        self.choices = choices


class _ParsedMessage:
    def __init__(self, parsed):
        self.parsed = parsed


class _TextMessage:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None  # 与真实 SDK 一致：无工具调用时为 None


class _FunctionRef:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.type = "function"
        self.function = _FunctionRef(name, arguments)


class _ToolCallMessage:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls


class _CompletionsEndpoint:
    """kind: "chat"（create 返回文本）| "parse"（返回 .parsed）"""

    def __init__(self, owner, kind):
        self._owner = owner
        self._kind = kind

    def create(self, **kwargs):
        return self._owner._dispatch(self._kind, kwargs)

    def parse(self, **kwargs):
        return self._owner._dispatch(self._kind, kwargs)


class _Chat:
    """模拟 openai 结构：client.chat.completions.create(...)。"""

    def __init__(self, owner):
        self.completions = _CompletionsEndpoint(owner, "chat")


class _BetaCompletions:
    def __init__(self, owner):
        self.completions = _CompletionsEndpoint(owner, "parse")


class _Beta:
    def __init__(self, owner):
        self.chat = _BetaCompletions(owner)


class FakeChatClient:
    """脚本化 client：逐条消费剧本（enqueue），记录调用快照。"""

    def __init__(self):
        self.chat = _Chat(self)
        self.beta = _Beta(self)
        self.calls: list[tuple[str, dict]] = []  # (kind, kwargs)
        self._script: list[dict] = []

    # ---------- 剧本 ----------
    def enqueue(self, result=None, error=None) -> FakeChatClient:
        """追加一个剧本条目。

        result 为 str → chat.create 的 content；为其他对象 → parse 的 .parsed；
        为可调用对象 → fn(kind, kwargs) 返回原始 response 或抛异常。
        """
        self._script.append({"result": result, "error": error})
        return self

    def enqueue_tool_call(self, call_id: str, name: str,
                          arguments) -> FakeChatClient:
        """追加一条工具调用响应（ReAct 步骤用；arguments 为 str 或 dict）。"""
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        return self.enqueue(chat_response(
            "",
            tool_calls=[_ToolCall(call_id, name, arguments)],
        ))

    def enqueue_final_response(self, reply: str, intent: str = "other",
                               requires_human: bool = False,
                               follow_up_question=None,
                               call_id: str = "call_final") -> FakeChatClient:
        """追加一条 final_response 终止调用（新结构化终答协议）。"""

        return self.enqueue_tool_call(call_id, "final_response", {
            "intent": intent,
            "reply": reply,
            "requires_human": requires_human,
            "follow_up_question": follow_up_question,
        })

    def enqueue_chat(self, text: str) -> FakeChatClient:
        return self.enqueue(text)

    def enqueue_parse(self, parsed) -> FakeChatClient:
        return self.enqueue(parsed)

    def enqueue_error(self, exc: Exception) -> FakeChatClient:
        return self.enqueue(error=exc)

    def enqueue_callable(self, fn) -> FakeChatClient:
        return self.enqueue(fn)

    def clear_script(self) -> None:
        self._script.clear()

    # ---------- 内部 ----------
    def _dispatch(self, kind: str, kwargs: dict):
        self.calls.append((kind, kwargs))
        if not self._script:
            raise AssertionError(f"FakeChatClient 剧本已耗尽（kind={kind}）")
        item = self._script.pop(0)
        if item["error"] is not None:
            raise item["error"]
        result = item["result"]
        if callable(result):
            return result(kind, kwargs)
        if isinstance(result, _Response):
            return result  # 完整响应对象（工具调用剧本等）直接返回
        if kind == "parse":
            return _Response([_Choice(_ParsedMessage(result))])
        content = result if result is not None else ""
        return _Response([_Choice(_TextMessage(content))])


# ============================================================
# FakeBackend / FakeRetriever
# ============================================================
def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class FakeBackend(VectorBackend):
    """内存版 VectorBackend：chunk_id → (chunk, vector)。"""

    def __init__(self, embedding_model: str = "fake-embedder"):
        self._data: dict[str, tuple[Chunk, list[float]]] = {}
        self._embedding_model = embedding_model

    def upsert(self, chunks, vectors, embedding_model) -> None:
        self._data = {
            c.chunk_id: (c, list(vec))
            for c, vec in zip(chunks, vectors)
        }
        self._embedding_model = embedding_model

    def search(self, query_vector, top_k, timeout=None):
        scored = [
            (chunk, _cosine(query_vector, vec))
            for chunk, vec in self._data.values()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [RetrievedChunk(chunk=c, score=s) for c, s in scored[:top_k]]

    def chunks(self):
        """全部已索引 chunk（混合检索 BM25 建索引用）。"""
        return [chunk for chunk, _ in self._data.values()]

    def size(self) -> int:
        return len(self._data)

    def load(self) -> None:
        pass  # 内存版无需加载

    def expected_embedding_model(self) -> str:
        return self._embedding_model


class FakeRetriever:
    """KnowledgeRetriever 的内存版替身：query → embed → backend.search。"""

    def __init__(self, embedder: FakeEmbedder, backend: FakeBackend):
        self._embedder = embedder
        self._backend = backend

    def search(self, query: str, top_k: int = 3, timeout=None):
        q_vec = self._embedder.encode_one(query)
        return self._backend.search(q_vec, top_k=top_k)


# ============================================================
# 样本构造辅助
# ============================================================
def chat_response(text: str, tool_calls=None):
    """构造一个 chat 响应对象（供 enqueue_callable/enqueue 使用）。"""
    message = (
        _TextMessage(text) if tool_calls is None
        else _ToolCallMessage(text, tool_calls)
    )
    return _Response([_Choice(message)])


def sample_response(**overrides) -> CustomerServiceResponse:
    """构造一条样例结构化回复（pydantic 纯构造，无网络）。"""
    base = dict(
        intent="order_query",
        confidence=0.9,
        reply="您的订单已发出，预计 3 天内送达。",
        requires_human=False,
        follow_up_question=None,
    )
    base.update(overrides)
    return CustomerServiceResponse(**base)


def search_tool_messages(question: str, results: list[dict], call_id: str = "call_001") -> list[dict]:
    """构造 raw_messages 中一轮 search_knowledge 调用链的切片。"""
    return [
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "search_knowledge", "arguments": '{"query": "%s"}' % question},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps({"success": True, "results": results}, ensure_ascii=False),
        },
    ]


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def reset_settings():
    """测试内改动全局 settings 后自动还原（含删除新增键）。"""
    snapshot = dict(settings.__dict__)
    yield
    for k in list(settings.__dict__.keys()):
        if k not in snapshot:
            delattr(settings, k)
    settings.__dict__.update(snapshot)


@pytest.fixture(autouse=True)
def _default_embedding_settings(monkeypatch):
    """embedding 相关设置固定为默认值：单测不依赖开发机 .env。

    CI 无 .env（默认 openai 提供方）；本地 .env 可能已切到
    EMBEDDING_PROVIDER=sophnet，直接读会让 create_embedder 走 sophnet
    分支并因 URL 为空报错。需要测 sophnet 分支的用例自行覆盖
    （见 test_sophnet_embedder.py）。
    """
    monkeypatch.setattr(settings, "embedding_provider", "openai")
    monkeypatch.setattr(settings, "sophnet_embedding_url", "")
    monkeypatch.setattr(settings, "sophnet_api_key", "")
    monkeypatch.setattr(settings, "sophnet_easyllm_id", "")
    monkeypatch.setattr(settings, "embedding_model", "text-embedding-3-small")


@pytest.fixture
def tmp_state_dir(tmp_path):
    """evolution 状态目录：turns / state / output。"""
    dirs = {name: tmp_path / name for name in ("turns", "state", "output")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


@pytest.fixture
def tmp_kb_dir(tmp_path):
    """迷你知识库：根目录两份文档 + evolved/ 子目录一份。"""
    kb = tmp_path / "knowledge"
    (kb / "evolved").mkdir(parents=True, exist_ok=True)
    (kb / "退货政策.md").write_text(
        "# 退货政策\n\n## 七天无理由\n\n支持七天无理由退货，运费由顾客承担。\n",
        encoding="utf-8",
    )
    (kb / "配送说明.md").write_text(
        "# 配送说明\n\n## 偏远地区\n\n偏远地区不包邮。\n",
        encoding="utf-8",
    )
    (kb / "evolved" / "20260801-abc123-自定义问答.md").write_text(
        "# 自进化知识\n\n## 问题\n\n饼干保质期一年内可退吗？\n\n## 回答\n\n可以，未拆封可退。\n",
        encoding="utf-8",
    )
    return kb


@pytest.fixture
def frozen_clock():
    """冻结时钟：now() 固定，可 advance() 推进（供 pipeline/ledger 注入）。"""
    state = {"now": datetime(2026, 8, 28, 12, 0, 0)}

    class Clock:
        def now(self):
            return state["now"]

        def advance(self, **kwargs):
            state["now"] += timedelta(**kwargs)

    return Clock()


# ============================================================
# chromadb 可用性守卫
# ============================================================
_CHROMA_PROBE = """
import tempfile, pathlib, chromadb
p = pathlib.Path(tempfile.mkdtemp()) / "c"
c = chromadb.PersistentClient(path=str(p))
col = c.create_collection("probe_col")
col.add(ids=["a"], embeddings=[[0.1, 0.2, 0.3]], documents=["x"])
print("PROBE_OK")
"""


@pytest.fixture
def chromadb_usable():
    """chromadb 可导入 + Rust 核心可运行才通过。

    chromadb 某版本在部分 Windows 环境的 Rust 后端会以 0xC0000005
    （access violation）直接崩溃整个进程——这不是 catch 得住的异常，
    只能在子进程里探测：探针都跑不通就 skip，断言逻辑照旧在健康环境执行。
    """
    pytest.importorskip("chromadb")
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", _CHROMA_PROBE],
        capture_output=True, text=True, timeout=90,
    )
    if result.returncode != 0 or "PROBE_OK" not in result.stdout:
        pytest.skip(
            "chromadb Rust 后端在本平台不可用"
            f"（子进程退出码 {result.returncode}）；"
            "若需运行 chroma 用例，请安装 VC++ 运行时或改用 numpy 后端"
        )
    return True