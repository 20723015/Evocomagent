"""Reranker 精排扩展点（7.3）。

阶段 7.3 计划接入 bge-reranker-v2-m3 做语义精排：混合检索（hybrid.py）融合出
recall_k 个候选后，再交给精排模型出更细的序。本阶段先落扩展点、不引依赖：

- Reranker：抽象基类，rerank(query, hits, top_k) 定义精排契约；
- NullReranker：原样返回前 top_k 个（顺序不变），保证默认行为与旧管线一致；
- create_reranker(name)：工厂函数，按配置名创建实现。

bge-reranker-v2-m3 需要自部署（本地 GPU 服务化）或接第三方 API，
拿到模型端点/密钥后，在 create_reranker 里补一个分支实现即可，上层无需改动。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from app.agent.rag.backends.base import RetrievedChunk


class Reranker(ABC):
    """精排器抽象：对融合后的候选列表重新排序，取前 top_k 个。"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        hits: list[RetrievedChunk],
        top_k: int,
        timeout: float | None = None,
    ) -> list[RetrievedChunk]:
        """按语义相关性精排，返回前 top_k 个命中。

        timeout：轮次预算剩余（修复计划）；远端实现取 min(self.timeout, timeout)。
        """


class NullReranker(Reranker):
    """空实现：保持输入顺序，返回前 top_k 个（未配置精排时的默认行为）。"""

    def rerank(
        self,
        query: str,
        hits: list[RetrievedChunk],
        top_k: int,
        timeout: float | None = None,
    ) -> list[RetrievedChunk]:
        return list(hits[:top_k])


class HTTPReranker(Reranker):
    """HTTP 精排器（7.3 真模型接入）：cohere/jina API 或自部署 bge-reranker-v2-m3。

    请求形状按 provider 区分：
    - cohere：POST /v2/rerank，body {model, query, documents:[...]}，
      resp {"results": [{"index", "relevance_score"}]}；
    - jina：POST /v1/rerank（body 与 cohere 同形，resp 同构）；
    - tei / bge-reranker-v2-m3（自部署，Text-Embeddings-Inference）：
      POST /rerank，body {query, texts:[...]}，resp [[index, score], ...]。
    任一网络失败 → 返回原序前 top_k（持续可用优先，不向 Agent 抛错）。
    """

    PROVIDERS = ("cohere", "jina", "tei", "bge-reranker-v2-m3")

    def __init__(self, provider: str, endpoint_url: str = "", api_key: str = "",
                 model: str = "", timeout: float = 10.0, client=None):
        self._provider = provider
        self._endpoint = endpoint_url
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        # 持久化 client（trust_env=False）：rerank 为内网/自部署服务，不走系统代理
        #（Windows 上 httpx 拾取系统代理会把 localhost 请求转发到代理导致 502）。
        # 单测注入 httpx.MockTransport client；未注入时懒创建并自行持有。
        self._client = client
        self._owns_client = client is None

    def _get_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(trust_env=False)
        return self._client

    def close(self) -> None:
        """释放自持的持久化连接（注入的 client 由调用方管理）。"""
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def _payload(self, query: str, texts: list[str]) -> tuple[str, dict]:
        if self._provider in ("tei", "bge-reranker-v2-m3"):
            return self._endpoint or "http://127.0.0.1:8000/rerank", {
                "query": query, "texts": texts,
            }
        endpoint = self._endpoint or (
            "https://api.cohere.com/v2/rerank" if self._provider == "cohere"
            else "https://api.jina.ai/v1/rerank"
        )
        return endpoint, {"model": self._model or "rerank-multilingual-v3.0",
                          "query": query, "documents": texts}

    def _parse(self, data: dict) -> list[tuple[int, float]]:
        """把响应解析为 (全局索引, 分数) 列表；跳过畸形条目，顶层结构不可识别才抛错。

        - cohere/jina：{"results": [{"index", "relevance_score"}]}
        - TEI/bge-reranker-v2-m3：[[index, score], ...] 或 [{"index", "score"}, ...]
        - 单个条目畸形（缺字段/类型错）→ 忽略该条目；整体不可解析 → ValueError
          （调用方据此回退原序，不因个别坏条目丢弃全部精排结果）。
        """
        if not isinstance(data, (dict, list)):
            raise ValueError(f"不可识别的 rerank 响应: {str(data)[:120]}")

        def _pair(item) -> tuple[int, float] | None:
            try:
                if isinstance(item, dict):
                    idx = item.get("index")
                    score = item.get("relevance_score", item.get("score"))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    idx, score = item[0], item[1]
                else:
                    return None
                if idx is None or score is None:
                    return None
                return int(idx), float(score)
            except (TypeError, ValueError, KeyError):
                return None

        if isinstance(data, dict):
            results = data.get("results")
            if results is None:
                raise ValueError(f"不可识别的 rerank 响应: {str(data)[:120]}")
            items = results if isinstance(results, list) else [results]
        else:
            items = data if isinstance(data, list) else [data]

        out: list[tuple[int, float]] = []
        for item in items:
            pair = _pair(item)
            if pair is not None:
                out.append(pair)
        return out

    def rerank(self, query: str, hits: list[RetrievedChunk], top_k: int,
               timeout: float | None = None) -> list[RetrievedChunk]:
        # 空候选 / top_k<=0：无意义，不发 HTTP 请求
        if not hits or top_k <= 0:
            return []
        from app.observability.metrics import record_reranker_fallback
        # 分批调用（TEI/cohere 有单请求批量上限，如 TEI 默认 32）：候选超过
        # 上限时切片多次请求，index 偏移合并——recall_k 放大后不依赖批次大小。
        batch_size = 30
        attempt = min(self._timeout, timeout) if timeout is not None else self._timeout
        scored: dict[int, float] = {}
        any_scored = False
        try:
            client = self._get_client()
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            for start in range(0, len(hits), batch_size):
                chunk = hits[start:start + batch_size]
                endpoint, body = self._payload(query, [h.chunk.text for h in chunk])
                resp = client.post(endpoint, json=body, headers=headers,
                                   timeout=attempt)
                resp.raise_for_status()
                for index, score in self._parse(resp.json()):
                    gi = start + index
                    # 忽略越界、重复（首见保留）与 NaN/Inf 分数
                    if not 0 <= gi < len(hits):
                        continue
                    if gi in scored:
                        continue
                    if not math.isfinite(score):
                        continue
                    scored[gi] = float(score)
                    any_scored = True
        except Exception:  # noqa: BLE001 —— 精排失败退回原序（可用性优先）
            record_reranker_fallback()
            return list(hits[:top_k])

        if not any_scored:
            # 响应可解析但没有任何可用分数（全部畸形/越界）→ 也按原序
            record_reranker_fallback()
            return list(hits[:top_k])

        # 已评分项按精排分数降序在前，未评分项按原召回顺序补位；分数写回
        # hit.score——下游 filter_hits_by_score（负例拒绝）依赖精排后的分数。
        for gi, s in scored.items():
            hits[gi].score = s
        unscored = [i for i in range(len(hits)) if i not in scored]
        ordered = sorted(scored, key=lambda gi: scored[gi], reverse=True) + unscored
        return [hits[i] for i in ordered[:top_k]]


def create_reranker(name: str) -> Reranker:
    """按配置名创建精排器；未知名称报错，避免静默降级。

    7.3：cohere/jina 为托管 API；bge-reranker-v2-m3 为自部署
    （endpoint 由 settings.rerank_endpoint_url 配置）。
    """
    if name in ("", "none"):
        return NullReranker()
    if name in HTTPReranker.PROVIDERS:
        from app.config.settings import settings

        return HTTPReranker(
            provider=name,
            endpoint_url=settings.rerank_endpoint_url or "",
            api_key=settings.rerank_api_key or "",
            model=settings.rerank_model or "",
        )
    raise NotImplementedError(
        f"暂未实现 reranker: {name!r}（可选: none / cohere / jina / bge-reranker-v2-m3）"
    )