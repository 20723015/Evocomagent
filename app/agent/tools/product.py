"""商品检索工具：商品 ID 直查 + 商品知识库检索（独立命名空间）。

数据源（P2-1 扩容，追加不改写）：
- 商品目录来自 ``app.agent.tools.mock_data.get_dataset()["products"]``——
  内置 6 款种子 + 生成器扩容的 100+ 款（生成文件缺失时即种子，行为与扩容前一致）；
- 检索走 ``app.agent.rag.product_kb.search_products``：独立命名空间
  （authority=product / 独立索引文件），**绝不与政策库 search_knowledge 混检**；
  向量索引未构建或不可用时回落确定性词法检索（零网络、零全局随机）。

未命中任何商品 → 保留既有 mock 兜底（product_id 前缀 MOCK-），技能提示词据此
提示用户「无该商品」而不是编造。
"""

from __future__ import annotations

import copy
import random
from typing import Optional

from app.agent.context import ToolContext
from app.agent.rag import product_kb
from app.agent.tools.mock_data import get_dataset

# 单次返回上限：扩容后同一类目可能命中数十款，全量返回会撑爆工具输出与
# 上下文预算；按相关性取前 N 款（技能层再挑 1–3 款推荐）。
PRODUCT_RESULT_LIMIT = 8

# 商品向量检索超时（仅在商品索引已构建时才会走到；失败即词法回落）
PRODUCT_KB_TIMEOUT_SECONDS = 5.0


def _catalog() -> dict:
    """商品目录视图（种子 + 扩容条目；缓存由 mock_data 按文件指纹管理）。"""
    return get_dataset()["products"]


def _match_score(product: dict, keywords: list[str]) -> int:
    """（兼容保留）商品与关键词列表的匹配度；口径见 product_kb.match_score。"""
    query = " ".join(keywords)
    strong, category_overlap, overlap = product_kb.match_score(product, keywords, query)
    return strong * 100 + category_overlap * 10 + overlap


def _generate_mock_product(keyword: str) -> dict:
    """未命中任何商品时，生成一个 mock 商品兜底。"""
    price = round(random.uniform(99, 2999), 2)
    return {
        "product_id": f"MOCK-{random.randint(1000,9999)}",
        "name": f"{keyword}（热销款）",
        "category": keyword,
        "price": price,
        "stock": random.randint(10, 200),
        "description": f"并夕夕精选{keyword}，品质保证，支持七天无理由退换",
        "specs": {"备注": "模拟商品数据"},
    }


def _kb_embedder():
    """商品向量索引已构建时才构造 embedder（惰性；失败 → None 走词法）。"""
    if not product_kb.product_index_path().exists():
        return None
    try:
        from app.agent.rag.embedder import create_embedder

        return create_embedder(timeout=PRODUCT_KB_TIMEOUT_SECONDS)
    except Exception:  # noqa: BLE001 —— 配置缺失/构造失败不影响商品检索
        return None


def query_product(keyword: str, ctx: Optional[ToolContext] = None) -> dict:
    """根据商品名称关键词或商品ID查询商品信息，包括价格、库存、规格等。

    检索顺序：商品 ID 直查 → 商品知识库（独立命名空间）→ mock 兜底。
    """
    catalog = _catalog()
    if keyword in catalog:
        return {"success": True, "products": [copy.deepcopy(catalog[keyword])]}

    results = product_kb.search_products(
        keyword, catalog,
        top_k=PRODUCT_RESULT_LIMIT,
        embedder=_kb_embedder(),
    )

    if not results:
        return {"success": True, "products": [_generate_mock_product(keyword)]}
    return {"success": True, "products": [copy.deepcopy(p) for p in results]}
