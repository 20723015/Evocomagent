"""向量化封装：调用 OpenAI Embeddings 接口。

- 支持批量编码（list[str] → list[list[float]]）。
- 可配置 model 与 base_url（与 chat 模型共用一套 OpenAI 客户端配置）。
- 返回原始 list[float]，由调用方决定如何持久化（这里用 json，不引入 numpy 依赖）。
- 修复计划：构造时显式 timeout（不再用 SDK 600s 默认），调用时可传 remaining
  覆盖（检索链路受轮次预算约束）。
"""

from typing import Iterable, Optional

import httpx
from openai import OpenAI


class Embedder:
    """OpenAI Embeddings 同步封装。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "text-embedding-3-small",
        batch_size: int = 64,
        timeout: float = 60.0,
    ):
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self._model = model
        self._batch_size = batch_size
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    def encode(self, texts: Iterable[str], timeout: Optional[float] = None) -> list[list[float]]:
        """批量编码，自动按 batch_size 分批请求。

        timeout：单次请求超时（remaining 传入时取 min(remaining, 构造超时)）。
        """
        texts = list(texts)
        if not texts:
            return []
        attempt_timeout = self._attempt_timeout(timeout)
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = self._client.embeddings.create(
                model=self._model, input=batch, timeout=attempt_timeout,
            )
            out.extend(item.embedding for item in resp.data)
        return out

    def encode_one(self, text: str, timeout: Optional[float] = None) -> list[float]:
        return self.encode([text], timeout=timeout)[0]

    def _attempt_timeout(self, timeout: Optional[float]) -> float:
        if timeout is None or timeout <= 0:
            return self._timeout
        return min(self._timeout, timeout)


def _extract_vectors(obj, expected: int) -> "list[list[float]] | None":
    """从响应对象中提取向量列表；形状不符返回 None（交给下一候选尝试）。

    支持：data[].embedding（OpenAI）/ data[].embeddings（嵌套组，展开）/
    顶层 embeddings / 单条输入的顶层 embedding。
    """
    if not isinstance(obj, dict):
        if isinstance(obj, list) and len(obj) == expected and _is_vector_list(obj):
            return obj
        return None

    data = obj.get("data")
    if isinstance(data, list) and data and all(isinstance(d, dict) for d in data):
        items = [
            d.get("embedding") if isinstance(d.get("embedding"), list)
            else (d.get("embeddings") if isinstance(d.get("embeddings"), list) else None)
            for d in data
        ]
        # 嵌套形态：data[].embeddings 每项再含一组向量（按输入顺序展开）
        if items and all(i is not None and i and isinstance(i[0], list) for i in items):
            flattened = [v for group in items for v in group]
            if len(flattened) == expected:
                return flattened
        if len(items) == expected and all(i is not None for i in items):
            return items

    if isinstance(obj.get("embeddings"), list) and len(obj["embeddings"]) == expected:
        return obj["embeddings"]
    if expected == 1 and isinstance(obj.get("embedding"), list):
        return [obj["embedding"]]
    return None


def _is_vector_list(obj) -> bool:
    return isinstance(obj, list) and all(
        isinstance(v, list) and all(isinstance(x, (int, float)) for x in v)
        for v in obj
    )


class SophnetEmbedder:
    """SophNet EasyLLM Embeddings 适配（如 bge-m3）。

    SophNet 的 /easyllms/embeddings 不是 OpenAI 兼容格式：请求体用
    input_texts / easyllm_id / dimensions（OpenAI 是 input），因此不走
    OpenAI SDK，直接 HTTP（httpx，随 openai 依赖已可用）。

    响应格式官方文档未给出稳定 schema，这里做宽容解析，依次尝试：
    OpenAI 风格 data[].embedding → 顶层 embeddings → 顶层 embedding
    （仅单条输入时）。解析不出与输入等长的向量即报错，不静默返回错误数据。
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        easyllm_id: str,
        model: str = "bge-m3",
        dimensions: int = 1024,
        batch_size: int = 16,
        timeout: float = 60.0,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        if not url or not easyllm_id:
            raise ValueError(
                "SophnetEmbedder 需要 sophnet_embedding_url 与 sophnet_easyllm_id"
            )
        self._url = url
        self._easyllm_id = easyllm_id
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._timeout = timeout
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            transport=transport,
        )

    @property
    def model(self) -> str:
        return self._model

    def encode(self, texts: Iterable[str], timeout: Optional[float] = None) -> list[list[float]]:
        """批量编码，自动按 batch_size 分批请求（与 Embedder.encode 语义一致）。"""
        texts = list(texts)
        if not texts:
            return []
        attempt_timeout = self._attempt_timeout(timeout)
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            out.extend(self._post(batch, attempt_timeout))
        return out

    def encode_one(self, text: str, timeout: Optional[float] = None) -> list[float]:
        return self.encode([text], timeout=timeout)[0]

    def _post(self, batch: list[str], timeout: float) -> list[list[float]]:
        resp = self._client.post(
            self._url,
            json={
                "model": self._model,
                "easyllm_id": self._easyllm_id,
                "input_texts": batch,
                "dimensions": self._dimensions,
            },
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"SophNet embedding 请求失败 HTTP {resp.status_code}: {resp.text[:300]}"
            )
        vectors = self._parse(resp.json(), len(batch))
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"SophNet embedding 响应向量数({len(vectors)})与输入数({len(batch)})不一致"
            )
        bad = [i for i, v in enumerate(vectors) if not isinstance(v, list) or not v]
        if bad:
            raise RuntimeError(f"SophNet embedding 响应中第 {bad} 条向量缺失或为空")
        return vectors

    @staticmethod
    def _parse(payload: dict, expected: int) -> list[list[float]]:
        """宽容解析：兼容 SophNet 信封结构与 OpenAI 风格。

        SophNet 响应形如 {"status":20000,"message":...,"result":{...}}（错误码
        20012=ApiKey 无效等）；向量可能在 result 内，也可能直接是顶层 data /
        embeddings（OpenAI 风格）。依次尝试：result 信封 → OpenAI 风格
        data[].embedding → 顶层 embeddings → 顶层 embedding（仅单条输入时）。
        """
        status = payload.get("status")
        if status is not None and status not in (20000, 200, 0):
            raise RuntimeError(
                f"SophNet embedding 业务错误 status={status}: "
                f"{payload.get('message', '')[:200]}"
            )
        candidates = [payload]
        if isinstance(payload.get("result"), (dict, list)):
            candidates.insert(0, payload["result"])
        for candidate in candidates:
            vectors = _extract_vectors(candidate, expected)
            if vectors is not None:
                return vectors
        raise RuntimeError(
            f"无法解析 SophNet embedding 响应（顶层键: {sorted(payload)}），"
            f"期望 {expected} 条向量；请把实际响应发给开发者补充解析"
        )

    def _attempt_timeout(self, timeout: Optional[float]) -> float:
        if timeout is None or timeout <= 0:
            return self._timeout
        return min(self._timeout, timeout)


def create_embedder(timeout: float = 60.0):
    """按 settings.embedding_provider 构建嵌入器。

    openai（默认）→ OpenAI /embeddings；sophnet → SophNet EasyLLM 适配。
    供 knowledge / build_kb_index / run_evolution 等统一入口使用。
    """
    from app.config.settings import settings  # 局部导入：embedder 属底层模块

    if settings.embedding_provider == "sophnet":
        return SophnetEmbedder(
            url=settings.sophnet_embedding_url,
            api_key=settings.sophnet_api_key or settings.openai_api_key,
            easyllm_id=settings.sophnet_easyllm_id,
            model=settings.embedding_model,
            dimensions=settings.sophnet_embedding_dimensions,
            timeout=timeout,
        )
    return Embedder(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.embedding_model,
        timeout=timeout,
    )
