"""向量后端：抽象接口 + 两套实现（Numpy 手写 / Chroma 向量数据库）。

教学目的：同一份知识库可在两种后端间切换，便于直观对比
- 手写余弦（教学透明，零依赖，适合 < 1k chunks）
- 向量数据库（生产标配，HNSW/IVF 索引，支持亿级规模 + 元数据过滤 + 并发写入）

通过 settings.rag_backend 选择，对上层 KnowledgeRetriever / search_knowledge 透明。
"""

from app.agent.rag.backends.base import RetrievedChunk, VectorBackend
from app.agent.rag.backends.numpy_backend import NumpyBackend

__all__ = ["RetrievedChunk", "VectorBackend", "NumpyBackend", "create_backend"]


def create_backend(name: str, **kwargs) -> VectorBackend:
    """工厂方法：根据名称创建对应后端。

    name: "numpy" | "chroma" | "es"（阶段八：alias 版本化 + kNN + 建模校验）
    kwargs: 后端特定参数（index_path / persist_dir / collection / index_name 等）
    """
    name = (name or "numpy").lower()
    if name == "numpy":
        return NumpyBackend(index_path=kwargs["index_path"])
    if name == "chroma":
        from app.agent.rag.backends.chroma_backend import ChromaBackend

        return ChromaBackend(
            persist_dir=kwargs["persist_dir"],
            collection_name=kwargs.get("collection_name", "ecom_kb"),
        )
    if name == "es":
        from app.agent.rag.backends.es_backend import ESBackend
        from app.agent.rag.es_util import get_es_client
        from app.config.settings import settings

        # 修复计划·二轮 5：不在此处固定客户端——注入可恢复 provider，
        # 每次操作动态获取（ES 恢复无需重建后端）。
        es = kwargs.pop("es", None)
        es_provider = kwargs.pop("es_provider", None)
        if es is None and es_provider is None:
            es_provider = get_es_client
        return ESBackend(
            es=es,
            es_provider=es_provider,
            index_name=kwargs.get("index_name", ""),
            alias=kwargs.get("collection_name", ""),  # 缺省按 prefix 派生 alias
            index_prefix=settings.es_index_prefix,
        )
    raise ValueError(f"未知的 RAG 后端: {name}（可选: numpy / chroma / es）")
