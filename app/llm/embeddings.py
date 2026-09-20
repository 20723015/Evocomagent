"""记忆嵌入客户端（记忆系统重构·阶段2.1）。

OpenAI 兼容 embeddings 调用，复用 RAG Embedder 的连接/超时模式：
- 全部调用由调用方保证在 ``budget_user_scope`` 内（与 memory job worker
  一致，禁止匿名计费；本模块不自行建作用域——worker 已建，热路径
  chat() 请求已绑定请求用户）；
- 失败/超时/未配置 → 返回 None，调用方降级纯词面（**绝不抛进热路径**）；
- 内容寻址进程内 LRU（key = sha256(model + content)），避免重复计费；
  SQL 侧持久缓存见 app/agent/memory/embeddings.py。
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

from app.observability.logging import get_logger
from app.observability.metrics import record_memory_embedding_failure

log = get_logger("app.llm.embeddings")

_LRU_CAPACITY = 512


class MemoryEmbeddingClient:
    """记忆专用嵌入客户端：单条/批量编码，失败返回 None（降级词面）。"""

    def __init__(self, model: str, *, timeout: float = 5.0):
        self._model = model
        self._timeout = timeout
        self._embedder = None
        self._lru: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return self._model

    @staticmethod
    def content_hash(model: str, content: str) -> str:
        """内容寻址缓存键：sha256(model + content)（模型换版自动失效）。"""
        return hashlib.sha256(
            f"{model}{content}".encode("utf-8"),
        ).hexdigest()

    def _cached(self, key: str) -> list[float] | None:
        with self._lock:
            value = self._lru.get(key)
            if value is not None:
                self._lru.move_to_end(key)
            return value

    def _store(self, key: str, vector: list[float]) -> None:
        with self._lock:
            self._lru[key] = vector
            self._lru.move_to_end(key)
            while len(self._lru) > _LRU_CAPACITY:
                self._lru.popitem(last=False)

    def _ensure_embedder(self):
        if self._embedder is None:
            # 复用 RAG 嵌入器工厂（openai/sophnet 双提供方），短超时：
            # 热路径每轮至多一次调用，P95 预算 << turn 预算余量。
            # model 必须透传——缓存键/存储标签用 self._model，实际向量
            # 来源不一致会让标签说谎（配置 memory_embedding_model 时）
            from app.agent.rag.embedder import create_embedder

            self._embedder = create_embedder(
                timeout=self._timeout, model=self._model,
            )
        return self._embedder

    def encode(self, text: str, *, stage: str = "query") -> list[float] | None:
        """单条编码；未配置/失败 → None（调用方降级纯词面，绝不抛出）。"""
        text = (text or "").strip()
        if not text:
            return None
        key = self.content_hash(self._model, text)
        hit = self._cached(key)
        if hit is not None:
            return hit
        try:
            vector = self._ensure_embedder().encode_one(
                text, timeout=self._timeout,
            )
        except Exception as e:  # noqa: BLE001 —— 嵌入是增强，失败不阻塞
            record_memory_embedding_failure(stage)
            log.info(
                "memory.embedding_failed stage=%s err=%s", stage, type(e).__name__,
            )
            return None
        if not vector:
            return None
        vector = [float(x) for x in vector]
        self._store(key, vector)
        return vector

    def encode_batch(
        self, texts: list[str], *, stage: str = "write",
    ) -> list[list[float] | None]:
        """批量编码（写侧回填/sweep）；单项失败该位为 None，不拖垮整批。"""
        out: list[list[float] | None] = [None] * len(texts)
        pending: list[tuple[int, str, str]] = []  # (idx, key, text)
        for i, text in enumerate(texts):
            text = (text or "").strip()
            if not text:
                continue
            key = self.content_hash(self._model, text)
            hit = self._cached(key)
            if hit is not None:
                out[i] = hit
            else:
                pending.append((i, key, text))
        if not pending:
            return out
        try:
            vectors = self._ensure_embedder().encode(
                [t for _, _, t in pending], timeout=self._timeout,
            )
        except Exception as e:  # noqa: BLE001
            record_memory_embedding_failure(stage)
            log.info(
                "memory.embedding_batch_failed stage=%s n=%d err=%s",
                stage, len(pending), type(e).__name__,
            )
            return out
        for (i, key, _), vector in zip(pending, vectors):
            if not vector:
                continue
            vector = [float(x) for x in vector]
            self._store(key, vector)
            out[i] = vector
        return out


_MEMORY_EMBEDDER: MemoryEmbeddingClient | None = None
_MEMORY_EMBEDDER_LOCK = threading.Lock()


def get_memory_embedder() -> MemoryEmbeddingClient | None:
    """进程级记忆嵌入客户端（pod 单例）；未配置语义检索 → None。

    条件（阶段1.3 配置骨架）：
    - ``memory_semantic_enabled=false``（默认）→ None，全链路纯词面；
    - ``memory_embedding_model`` 空 → 回退 ``embedding_model``（RAG 同款）；
    - ``openai_api_key`` 未配置 → None（离线/测试环境零外呼）。
    """
    from app.config.settings import settings

    if not settings.memory_semantic_enabled:
        return None
    model = settings.memory_embedding_model or settings.embedding_model
    if not model or not settings.openai_api_key:
        return None
    global _MEMORY_EMBEDDER
    with _MEMORY_EMBEDDER_LOCK:
        if _MEMORY_EMBEDDER is None or _MEMORY_EMBEDDER.model != model:
            _MEMORY_EMBEDDER = MemoryEmbeddingClient(model)
        return _MEMORY_EMBEDDER


def reset_memory_embedder() -> None:
    """测试用：重置进程单例（settings 变更后重建）。"""
    global _MEMORY_EMBEDDER
    with _MEMORY_EMBEDDER_LOCK:
        _MEMORY_EMBEDDER = None
