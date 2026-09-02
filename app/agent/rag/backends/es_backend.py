"""ES 向量检索后端（阶段八）：alias 版本化 + kNN + 建模校验。

设计要点（对齐阶段二/七）：
- generation 版本化 → ES alias 原子切换：build 建 `kb-<gen>` 索引（pending，
  验证通过前不生效）→ activate() 把 `{prefix}-kb-active` 别名原子切到新索引
  → 旧索引保留 N 天可秒级回滚（与「staging 验证失败不切指针」语义一致）；
- embedding 模型校验：建索引时写 mapping._meta.embedding_model，加载时校验，
  不一致拒绝启动（与 chroma 语义一致）；
- upsert(chunks, vectors, model, index_name)：全量重建；search 走 kNN
  （余弦相似度，score = _score 直接可用）。
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

    def __init__(self, es, index_name: str = "", alias: str = "",
                 index_prefix: str = ""):
        self._es = es
        self._alias = alias or f"{index_prefix or 'ecom'}{_KB_ALIAS_SUFFIX}"
        self._index = index_name or ""  # 显式索引（构建目标）或经 alias 解析
        self._embedding_model = ""

    # ---------- 索引解析 ----------
    def _alias_target(self) -> str:
        resp = self._es.indices.get_alias(name=self._alias)
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
        mapping = self._es.indices.get_mapping(index=index)
        meta = (mapping.get(index, {}).get("mappings", {}).get("_meta", {}) or {})
        return meta.get(_EMBEDDING_MODEL_KEY, "")

    # ---------- 构建 / 激活 ----------
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]],
               embedding_model: str, index_name: str = "") -> str:
        """全量重建：创建新索引并写入（不切 alias——验证通过后由 activate() 切）。"""
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks 与 vectors 长度不一致: {len(chunks)} vs {len(vectors)}")
        dim = len(vectors[0]) if vectors else 0
        # 优先显式 index_name，其次构造时 index_name（generation target），最后自造
        index = index_name or self._index or (
            f"{self._alias[:-len(_KB_ALIAS_SUFFIX)]}-kb-{uuid.uuid4().hex[:8]}"
        )

        self._es.indices.create(
            index=index,
            settings={"number_of_shards": 1, "number_of_replicas": 0},
            mappings={
                "_meta": {_EMBEDDING_MODEL_KEY: embedding_model},
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc": {"type": "keyword"},
                    "section": {"type": "text"},
                    "text": {"type": "text", "analyzer": "standard"},
                    "source_path": {"type": "keyword"},
                    "provenance": {"type": "keyword"},
                    "owner": {"type": "keyword"},
                    "parent_text": {"type": "text"},
                    "vector": {"type": "dense_vector", "dims": dim,
                               "index": True, "similarity": "cosine"},
                },
            },
        )
        actions = []
        for c, v in zip(chunks, vectors):
            actions.append({"create": {"_index": index, "_id": c.chunk_id}})
            actions.append({
                "chunk_id": c.chunk_id, "doc": c.doc, "section": c.section,
                "text": c.text, "source_path": c.source_path or "",
                "provenance": c.provenance or "", "owner": c.owner or "",
                "parent_text": c.parent_text or "",
                "vector": list(v),
            })
        # 分批 bulk（每批 200 文档）
        for i in range(0, len(actions), 400):
            self._es.bulk(operations=actions[i:i + 400], index=index, refresh=False)
        self._es.indices.refresh(index=index)
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
        self._es.indices.update_aliases(actions=actions)

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
        resp = self._es.search(
            index=index,
            query={"match": {"text": query_text}},
            knn={
                "field": "vector", "query_vector": query_vector,
                "k": recall_k, "num_candidates": max(recall_k * 5, 50),
            },
            rank={"rrf": {"window_size": recall_k, "rank_constant": 60}},
            size=recall_k,
            source=["chunk_id", "doc", "section", "text", "source_path",
                    "provenance", "owner", "parent_text"],
            **query_kwargs,
        )
        return self._hits_to_results(resp)

    def search(self, query_vector: list[float], top_k: int,
               timeout: Optional[float] = None) -> list[RetrievedChunk]:
        index = self._resolve_index()
        query_kwargs: dict = {}
        if timeout is not None and timeout > 0:
            query_kwargs["request_timeout"] = timeout
        resp = self._es.search(
            index=index,
            knn={
                "field": "vector", "query_vector": query_vector, "k": top_k,
                "num_candidates": max(50, top_k * 5),
            },
            size=top_k,
            source=["chunk_id", "doc", "section", "text", "source_path",
                    "provenance", "owner", "parent_text"],
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
            resp = self._es.search(index=index, sort=["_id"], **body)
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
                ))
            after = hits[-1].get("sort", [None])[0]
            if after is None or len(hits) < 500:
                break
        return docs

    def size(self) -> int:
        index = self._resolve_index()
        return int(self._es.count(index=index).get("count", 0))

    # ---------- VectorBackend 协议 ----------
    def load(self) -> None:
        index = self._resolve_index()
        self._embedding_model = self._read_model(index)

    def expected_embedding_model(self) -> str:
        if not self._embedding_model:
            self.load()
        return self._embedding_model
