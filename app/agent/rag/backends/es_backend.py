"""ES 向量检索后端（阶段八）：alias 版本化 + kNN + 建模校验。

设计要点（对齐阶段二/七）：
- generation 版本化 → ES alias 原子切换：build 建 `kb-<gen>` 索引（pending，
  验证通过前不生效）→ activate() 把 `{prefix}-kb-active` 别名原子切到新索引
  → 旧索引保留 N 天可秒级回滚（与「staging 验证失败不切指针」语义一致）；
- embedding 模型校验：建索引时写 mapping._meta.embedding_model，加载时校验，
  不一致拒绝启动（与 chroma 语义一致）；
- upsert(chunks, vectors, model, index_name)：全量重建；search 走 kNN
  （ES cosine 语义：`_score = (1+cosine)/2`，与 numpy 后端的原始余弦不同，
  跨后端比较阈值前需逆变换 `cosine = 2*_score - 1`）。
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from app.agent.rag.backends.base import RetrievedChunk, VectorBackend
from app.agent.rag.chunker import Chunk

_EMBEDDING_MODEL_KEY = "embedding_model"
_KB_ALIAS_SUFFIX = "-kb-active"


class ESBackend(VectorBackend):
    """Elasticsearch 8.x 向量后端（dense_vector + cosine）。"""

    def __init__(self, es=None, index_name: str = "", alias: str = "",
                 index_prefix: str = "", es_provider=None):
        self._es = es
        # 修复计划·二轮 5：按操作获取当前客户端（可恢复 provider），
        # 不长期持有失效实例；provider 优先于固定 es。
        self._es_provider = es_provider
        self._alias = alias or f"{index_prefix or 'ecom'}{_KB_ALIAS_SUFFIX}"
        self._index = index_name or ""  # 显式索引（构建目标）或经 alias 解析
        self._embedding_model = ""

    def _client(self):
        """当前 ES 客户端：provider 动态获取；失败时 invalidate（由 provider 负责）。"""
        es = self._es_provider() if self._es_provider is not None else self._es
        if es is None:
            raise RuntimeError("ES 客户端不可用（provider 返回 None）")
        return es

    # ---------- 索引解析 ----------
    def _alias_target(self) -> str:
        resp = self._client().indices.get_alias(name=self._alias)
        return next(iter(resp.keys()))

    def _resolve_index(self) -> str:
        if self._index:
            return self._index
        try:
            self._index = self._alias_target()
            return self._index
        except Exception as e:  # noqa: BLE001 —— alias 缺失 = 索引未构建
            raise FileNotFoundError(
                f"ES alias '{self._alias}' 不存在：请先运行 "
                f"`python -m app.scripts.build_kb_index --backend es` 构建索引。"
            ) from e

    def _read_model(self, index: str) -> str:
        return self._read_meta(index).get(_EMBEDDING_MODEL_KEY, "")

    def _read_meta(self, index: str) -> dict:
        mapping = self._client().indices.get_mapping(index=index)
        return (mapping.get(index, {}).get("mappings", {}).get("_meta", {}) or {})

    def expected_embedding_meta(self) -> dict:
        """索引 `_meta`（provider/model/dimensions/config_fingerprint）。"""
        index = self._resolve_index()
        return self._read_meta(index)

    # ---------- 构建 / 激活 ----------
    @staticmethod
    def _mapping(embedding_model: str, dim: int, *, embedding_provider: str = "",
                 dimensions: int = 0, config_fingerprint: str = "") -> dict:
        """KB 索引 mapping（upsert / ensure_empty 共用）。

        RAG 修复计划·3：`_meta` 额外记录 embedding provider/dimensions 与配置
        指纹，运行时逐项比对；旧索引缺字段 → 判为需重建。
        """
        meta: dict = {_EMBEDDING_MODEL_KEY: embedding_model}
        if embedding_provider:
            meta["embedding_provider"] = embedding_provider
        if dimensions:
            meta["embedding_dimensions"] = int(dimensions)
        if config_fingerprint:
            meta["config_fingerprint"] = config_fingerprint
        return {
            "_meta": meta,
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc": {"type": "keyword"},
                "section": {"type": "text"},
                "text": {"type": "text", "analyzer": "standard"},
                "source_path": {"type": "keyword"},
                "provenance": {"type": "keyword"},
                "owner": {"type": "keyword"},
                "parent_text": {"type": "text"},
                "heading_path": {"type": "keyword"},
                "parent_id": {"type": "keyword"},
                "status": {"type": "keyword"},
                "authority": {"type": "keyword"},
                "effective_date": {"type": "keyword"},
                # 索引输入（P1-1/P1-2）：BM25 匹配该字段，text 仍是证据原文
                "index_text": {"type": "text", "analyzer": "standard"},
                "vector": {"type": "dense_vector", "dims": dim,
                           "index": True, "similarity": "cosine"},
            },
        }

    def _index_settings(self) -> dict:
        return {"number_of_shards": 1, "number_of_replicas": 0}

    def _target_index(self) -> str:
        """显式 index_name → 构造时 index_name → 自造（generation target）。"""
        return self._index or (
            f"{self._alias[:-len(_KB_ALIAS_SUFFIX)]}-kb-{uuid.uuid4().hex[:8]}"
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]],
               embedding_model: str, index_name: str = "", *,
               embedding_provider: str = "", dimensions: int = 0,
               config_fingerprint: str = "") -> str:
        """全量重建：创建新索引并写入（不切 alias——验证通过后由 activate() 切）。"""
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks 与 vectors 长度不一致: {len(chunks)} vs {len(vectors)}")
        dim = len(vectors[0]) if vectors else 0
        # 优先显式 index_name，其次构造时 index_name（generation target），最后自造
        index = index_name or self._target_index()

        self._client().indices.create(
            index=index,
            settings=self._index_settings(),
            mappings=self._mapping(
                embedding_model, dim,
                embedding_provider=embedding_provider,
                dimensions=dimensions or dim,
                config_fingerprint=config_fingerprint,
            ),
        )
        actions = []
        for c, v in zip(chunks, vectors):
            actions.append({"create": {"_index": index, "_id": c.chunk_id}})
            actions.append({
                "chunk_id": c.chunk_id, "doc": c.doc, "section": c.section,
                "text": c.text, "source_path": c.source_path or "",
                "provenance": c.provenance or "", "owner": c.owner or "",
                "parent_text": c.parent_text or "",
                "heading_path": c.heading_path or "",
                "parent_id": c.parent_id or "",
                "status": c.status or "",
                "authority": c.authority or "",
                "effective_date": c.effective_date or "",
                "index_text": c.index_input(),
                "vector": list(v),
            })
        # 分批 bulk（每批 200 文档）
        for i in range(0, len(actions), 400):
            resp = self._client().bulk(operations=actions[i:i + 400], index=index,
                                 refresh=False)
            failed = resp.get("errors", False)
            if failed:
                # review：bulk 逐项检查，静默丢 chunk 必须显式失败（chunk_id
                # 冲突/文本超限等），与 outbox 同步同款判定（status >= 300）
                n_failed = sum(
                    1 for item in resp.get("items", [])
                    if any(v.get("status", 200) >= 300 for v in item.values())
                )
                raise RuntimeError(
                    f"ES bulk 写入失败 {n_failed} 项（errors=true）；"
                    f"已中止，generation 未切换"
                )
        self._client().indices.refresh(index=index)
        self._index = index
        self._embedding_model = embedding_model
        return index

    def ensure_empty(self, embedding_model: str, dims: int, *,
                     embedding_provider: str = "",
                     config_fingerprint: str = "") -> str:
        """空索引（allow_empty 下架终态）：只建 mapping 不写 bulk。

        dense_vector dims=0 非法，必须用 embedder 的维度常量建空 mapping。
        """
        index = self._target_index()
        self._client().indices.create(
            index=index,
            settings=self._index_settings(),
            mappings=self._mapping(
                embedding_model, dims,
                embedding_provider=embedding_provider,
                dimensions=dims,
                config_fingerprint=config_fingerprint,
            ),
        )
        self._client().indices.refresh(index=index)
        self._index = index
        self._embedding_model = embedding_model
        return index

    def activate(self) -> None:
        """原子切换 alias 到当前索引（验证通过后调用）；旧索引保留（回滚/清理另行处理）。"""
        index = self._resolve_index()
        actions = []
        try:
            old = self._alias_target()
            if old != index:
                actions.append({"remove": {"index": old, "alias": self._alias}})
        except Exception:  # noqa: BLE001 —— alias 尚不存在（首次激活）
            pass
        actions.append({"add": {"index": index, "alias": self._alias}})
        self._client().indices.update_aliases(actions=actions)

    # ---------- 检索 ----------
    def hybrid_search(self, query_text: str, query_vector: list[float],
                      top_k: int, recall_k: int = 30,
                      timeout: Optional[float] = None) -> list[RetrievedChunk]:
        """ES 原生混合检索（阶段八）：一条查询同时出 BM25 + kNN 并做 RRF。

        rank 参数自 ES 8.8 起可用（rank_constant=60 与 Python 侧 RRF 语义一致）；
        查询级融合、单次网络往返，Python 侧 hybrid 层在 ES 路径下退役。

        返回 RRF 融合后的候选窗口（size=recall_k 而非 top_k）：精排器需要
        「融合前 N 名」才有机会把排位靠后的相关 chunk 拉回 Top-K——若只回
        top_k 条，rerank 候选与最终结果同源，排位在 top_k 之外的期望文档
        永远不可达（与 HybridRetriever 先取 recall_k 候选再精排的语义对齐）。
        top_k 截断由调用方（ESHybridRetriever）在精排后执行。

        timeout：轮次预算剩余（修复计划）→ 请求级 request_timeout。
        """
        index = self._resolve_index()
        query_kwargs: dict = {}
        if timeout is not None and timeout > 0:
            query_kwargs["request_timeout"] = timeout
        resp = self._client().search(
            index=index,
            # BM25 吃索引输入字段（P1-1 生成上下文/P1-2 evolved 问题在此生效）；
            # 旧索引无该字段时 BM25 路空召回，kNN 路仍返回，RRF 降级但可用
            query={"match": {"index_text": query_text}},
            knn={
                "field": "vector", "query_vector": query_vector,
                "k": recall_k, "num_candidates": max(recall_k * 5, 50),
            },
            rank={"rrf": {"window_size": recall_k, "rank_constant": 60}},
            size=recall_k,
            source=["chunk_id", "doc", "section", "text", "source_path",
                    "provenance", "owner", "parent_text", "heading_path",
                    "parent_id", "status", "authority", "effective_date",
                    "index_text"],
            **query_kwargs,
        )
        return self._hits_to_results(resp)

    def search(self, query_vector: list[float], top_k: int,
               timeout: Optional[float] = None) -> list[RetrievedChunk]:
        index = self._resolve_index()
        query_kwargs: dict = {}
        if timeout is not None and timeout > 0:
            query_kwargs["request_timeout"] = timeout
        resp = self._client().search(
            index=index,
            knn={
                "field": "vector", "query_vector": query_vector, "k": top_k,
                "num_candidates": max(50, top_k * 5),
            },
            size=top_k,
            source=["chunk_id", "doc", "section", "text", "source_path",
                    "provenance", "owner", "parent_text", "heading_path",
                    "parent_id", "status", "authority", "effective_date",
                    "index_text"],
            **query_kwargs,
        )
        return self._hits_to_results(resp)

    @staticmethod
    def _hits_to_results(resp: dict) -> list[RetrievedChunk]:
        out: list[RetrievedChunk] = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            raw_score = hit.get("_score")
            # RRF 融合（rank 加权）后 _score 可能为 None：记 0.0——
            # 排序由 ES 保证，分数仅用于相关性门控（未配置阈值时原序返回）
            out.append(RetrievedChunk(
                chunk=Chunk(
                    chunk_id=src.get("chunk_id", hit["_id"]),
                    doc=src.get("doc", ""), section=src.get("section", ""),
                    text=src.get("text", ""), source_path=src.get("source_path", ""),
                    provenance=src.get("provenance", ""), owner=src.get("owner", ""),
                    parent_text=src.get("parent_text", ""),
                    heading_path=src.get("heading_path", ""),
                    parent_id=src.get("parent_id", ""),
                    status=src.get("status", ""),
                    authority=src.get("authority", ""),
                    effective_date=src.get("effective_date", ""),
                    index_text=src.get("index_text", ""),
                ),
                score=float(raw_score) if raw_score is not None else 0.0,
            ))
        return out

    def chunks(self) -> list[Chunk]:
        """全量 chunk（BM25 建索引用 / 重建导出）。"""
        index = self._resolve_index()
        docs: list[Chunk] = []
        after: Optional[bytes] = None
        for _ in range(1000):
            body = {"query": {"match_all": {}}, "size": 500}
            if after is not None:
                body["search_after"] = [after]
            resp = self._client().search(index=index, sort=["_id"], **body)
            hits = resp["hits"]["hits"]
            if not hits:
                break
            for h in hits:
                src = h["_source"]
                docs.append(Chunk(
                    chunk_id=src.get("chunk_id", h["_id"]),
                    doc=src.get("doc", ""), section=src.get("section", ""),
                    text=src.get("text", ""), source_path=src.get("source_path", ""),
                    provenance=src.get("provenance", ""), owner=src.get("owner", ""),
                    parent_text=src.get("parent_text", ""),
                    heading_path=src.get("heading_path", ""),
                    parent_id=src.get("parent_id", ""),
                    status=src.get("status", ""),
                    authority=src.get("authority", ""),
                    effective_date=src.get("effective_date", ""),
                    index_text=src.get("index_text", ""),
                ))
            after = hits[-1].get("sort", [None])[0]
            if after is None or len(hits) < 500:
                break
        return docs

    def size(self) -> int:
        index = self._resolve_index()
        return int(self._client().count(index=index).get("count", 0))

    # ---------- VectorBackend 协议 ----------
    def load(self) -> None:
        index = self._resolve_index()
        self._embedding_model = self._read_model(index)

    def expected_embedding_model(self) -> str:
        if not self._embedding_model:
            self.load()
        return self._embedding_model
