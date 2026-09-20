"""P2-1 商品知识库：商品文档 + 独立命名空间检索（对标 JC0v0 双知识库）。

与政策库（``settings.kb_dir`` → ``kb_index_path`` / chroma collection / ES alias）
**物理隔离**，四道保险：

1. 商品 md 文档写在 ``app/agent/tools/data/product_docs/``——在政策库扫描根之外，
   且相对路径 ``app/...`` 不在 ``governance.ALLOWED_ROOTS``（"" / evolved / uploads）
   允许范围内，误扫也不会进政策索引；
2. 商品索引是独立文件 ``app/agent/tools/data/product_kb_index.json``，与政策库索引
   路径/collection 无交集（本模块从不构造指向政策索引的后端）；
3. 载入商品索引时做**命名空间断言**：任一 chunk 的 ``authority != "product"`` 或
   来源不是商品文档目录的平铺 md → ``ProductNamespaceViolation``（fail-closed，
   绝不把政策/外部文档当商品返回，也绝不把商品文档混进政策检索）；
4. 政策检索侧（``search_knowledge`` / ``retriever_factory``）对本模块零依赖——
   商品文档不进政策索引，政策门禁（负例压力）口径不被污染。

检索实现（本仓库为离线/无外部依赖默认）：
- 默认**词法检索**（确定性、零网络）：名称/类目/描述/规格/编号 精确子串 +
  中文 bigram 兜底，阈值门控，无命中即返回空（由工具层回落到 mock 兜底）；
- 可选**向量检索**：``build_product_index`` 用注入的 embedder（真实场景经
  ``seed_commerce_data.py --build-index``）建独立 numpy 索引，``search_products``
  在显式传入 embedder 且索引存在时优先走向量路，异常/模型不一致 → 词法回落
  （fail-open，工具层永不因商品库不可用而失败）。
"""

from __future__ import annotations

import re
from pathlib import Path

from app.agent.rag.backends import create_backend
from app.agent.rag.chunker import Chunk, chunk_markdown_dir
from app.observability.logging import get_logger

log = get_logger("app.agent.rag.product_kb")

# 商品命名空间标记（写入文档 frontmatter，随 chunk 下沉到索引）
PRODUCT_AUTHORITY = "product"
PRODUCT_NAMESPACE = "product"
# 商品文档生效日期：常量而非当前时钟——保证生成器输出逐字节可复现
PRODUCT_DOC_EFFECTIVE_DATE = "2024-06-01"

_DATA_DIR = Path(__file__).resolve().parents[2] / "agent" / "tools" / "data"
PRODUCT_DOCS_DIR = _DATA_DIR / "product_docs"
PRODUCT_INDEX_PATH = _DATA_DIR / "product_kb_index.json"

DEFAULT_TOP_K = 5


class ProductNamespaceViolation(RuntimeError):
    """商品索引中出现非商品命名空间条目（fail-closed，拒绝检索）。"""


def product_docs_dir() -> Path:
    return PRODUCT_DOCS_DIR


def product_index_path() -> Path:
    return PRODUCT_INDEX_PATH


# ============================================================
# 文档渲染 / 落盘
# ============================================================
def render_product_doc(product: dict) -> str:
    """商品详情 → 独立命名空间 md 文档（frontmatter 带 authority/namespace）。"""
    specs = product.get("specs") or {}
    price = float(product.get("price", 0) or 0)
    stock = int(product.get("stock", 0) or 0)
    lines = [
        "---",
        "status: active",
        f"authority: {PRODUCT_AUTHORITY}",
        f"namespace: {PRODUCT_NAMESPACE}",
        f"product_id: {product['product_id']}",
        f"category: {product.get('category', '')}",
        f"price: {price}",
        f"effective_date: {PRODUCT_DOC_EFFECTIVE_DATE}",
        "---",
        "",
        f"# {product.get('name', product['product_id'])}",
        "",
        "## 商品信息",
        f"- 商品编号：{product['product_id']}",
        f"- 类目：{product.get('category', '')}",
        f"- 价格：¥{price:.2f}",
        f"- 库存：{stock} 件（{'有货' if stock > 0 else '缺货'}）",
        "",
        "## 商品卖点",
        str(product.get("description", "") or ""),
        "",
        "## 规格参数",
        "| 参数 | 值 |",
        "| --- | --- |",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in specs.items())
    lines.extend([
        "",
        "## 售后与配送",
        "支持七天无理由退换（以平台售后政策为准）；下单后 48 小时内发货。",
        "",
    ])
    return "\n".join(lines)


def product_doc_name(product_id: str) -> str:
    return f"{product_id}.md"


def write_product_docs(products, docs_dir: str | Path | None = None) -> int:
    """把商品目录渲染为 md 文档；返回写出的文档数。

    幂等：同输入两次写出逐字节一致；清理只针对「本命名空间且已不在目录中」的
    陈旧文档（带 ``authority: product`` 标记），不误删其它文件。
    """
    target = Path(docs_dir) if docs_dir is not None else PRODUCT_DOCS_DIR
    target.mkdir(parents=True, exist_ok=True)
    items = products.values() if isinstance(products, dict) else products
    written: set[str] = set()
    for product in items:
        name = product_doc_name(product["product_id"])
        written.add(name)
        (target / name).write_text(
            render_product_doc(product), encoding="utf-8", newline="\n",
        )
    for stale in sorted(target.glob("*.md")):
        if stale.name in written:
            continue
        try:
            head = stale.read_text(encoding="utf-8")[:400]
        except OSError:
            continue
        if f"authority: {PRODUCT_AUTHORITY}" in head:
            stale.unlink()
    return len(written)


def iter_product_doc_ids(docs_dir: str | Path | None = None) -> list[str]:
    """商品文档目录中的 product_id 列表（文件名即商品编号，排序确定）。

    只认带 ``authority: product`` 标记的文档——同目录下的其它 md（若有人手工
    放入）不会被当成商品。
    """
    target = Path(docs_dir) if docs_dir is not None else PRODUCT_DOCS_DIR
    if not target.exists():
        return []
    ids: list[str] = []
    for path in sorted(target.glob("*.md")):
        try:
            head = path.read_text(encoding="utf-8")[:400]
        except OSError:
            continue
        if f"authority: {PRODUCT_AUTHORITY}" in head:
            ids.append(path.stem)
    return ids


# ============================================================
# 索引构建（独立命名空间）
# ============================================================
def assert_product_namespace(chunks) -> None:
    """命名空间断言：全部 chunk 必须是商品文档目录的平铺 md + authority=product。"""
    for chunk in chunks:
        authority = str(getattr(chunk, "authority", "") or "").strip().lower()
        rel = str(getattr(chunk, "source_path", "") or "").replace("\\", "/")
        if authority != PRODUCT_AUTHORITY:
            raise ProductNamespaceViolation(
                f"{chunk.chunk_id}: authority={authority!r} 非商品命名空间"
                f"（{PRODUCT_AUTHORITY}），拒绝载入商品索引"
            )
        if not rel or "/" in rel or not rel.endswith(".md"):
            raise ProductNamespaceViolation(
                f"{chunk.chunk_id}: source_path={rel!r} 不在商品文档目录内，"
                "拒绝载入商品索引"
            )


def product_doc_chunks(docs_dir: str | Path | None = None) -> list[Chunk]:
    """切分商品文档目录（复用 chunker 的 md 切分），并做命名空间断言。"""
    target = Path(docs_dir) if docs_dir is not None else PRODUCT_DOCS_DIR
    if not target.exists():
        return []
    chunks = chunk_markdown_dir(target)
    assert_product_namespace(chunks)
    return chunks


def build_product_index(products, embedder, *,
                        index_path: str | Path | None = None,
                        docs_dir: str | Path | None = None) -> int:
    """构建商品向量索引（独立文件，绝不写政策库索引）；返回 chunk 数。

    embedder 由调用方注入（真实场景：``seed_commerce_data.py --build-index``
    用 settings 的 embedding 提供方；测试用确定性替身）。文档按目录当前内容
    重建（幂等），索引文件名固定为商品索引路径——与政策库索引无交集。
    """
    write_product_docs(products, docs_dir)
    chunks = product_doc_chunks(docs_dir)
    if not chunks:
        raise FileNotFoundError(
            f"商品文档目录为空: {docs_dir or PRODUCT_DOCS_DIR}"
            "（先运行 python -m app.scripts.seed_commerce_data）"
        )
    vectors = embedder.encode([chunk.index_input() for chunk in chunks])
    backend = create_backend(
        "numpy", index_path=Path(index_path) if index_path else PRODUCT_INDEX_PATH,
    )
    backend.upsert(chunks, vectors, embedder.model)
    return len(chunks)


# ============================================================
# 检索
# ============================================================
def searchable_text(product: dict) -> str:
    specs = product.get("specs") or {}
    return " ".join([
        str(product.get("product_id", "")),
        str(product.get("name", "")),
        str(product.get("category", "")),
        str(product.get("description", "")),
        " ".join(str(v) for v in specs.values()),
    ]).lower()


def _bigrams(text: str) -> set[str]:
    cleaned = re.sub(r"\s+", "", str(text or "").lower())
    if len(cleaned) < 2:
        return set()
    return {cleaned[i:i + 2] for i in range(len(cleaned) - 1)}


def match_score(product: dict, terms: list[str], query: str) -> tuple[int, int, int]:
    """商品相关性 (strong, category_overlap, overlap)。

    - strong 档位：类目精确命中 4 > 名称子串命中 3 > 描述/规格子串命中 2——
      类目查询（"耳机"/"手机"）优先返回该类目商品，而不是名字里恰好带词的配件；
    - category_overlap：查询 bigram 与**类目名**的重合数——「有手机卖吗」这类
      口语化品类问法能命中类目，把真正的手机排在手机壳前面；
    - overlap：查询 bigram 与商品全文的重合数（短语兜底召回的排序依据）。
    """
    text = searchable_text(product)
    name = str(product.get("name", "")).lower()
    category = str(product.get("category", "")).lower()
    strong = 0
    for term in terms:
        if len(term) < 2:
            continue
        if category and term == category:
            strong = max(strong, 4)
        elif term in name:
            strong = max(strong, 3)
        elif term in text:
            strong = max(strong, 2)
    query_bigrams = _bigrams(query)
    overlap = len(query_bigrams & _bigrams(text))
    category_overlap = len(query_bigrams & _bigrams(category)) if category else 0
    return strong, category_overlap, overlap


def _terms(query: str) -> list[str]:
    raw = str(query or "").strip().lower()
    parts = [raw]
    parts.extend(t for t in re.split(r"[\s,，、/|]+", raw) if t)
    out: list[str] = []
    for part in parts:
        if part and part not in out:
            out.append(part)
    return out


def is_relevant(strong: int, overlap: int, query: str) -> bool:
    """相关性门控：子串/类目命中，或 bigram 重合达到查询长度自适应阈值。

    阈值 ``max(1, bigram 数 // 3)``：短查询（「有耳机卖吗」）至少一个 bigram
    重合即召回，长查询要求更多重合，避免噪声。
    """
    if strong > 0:
        return True
    return overlap >= max(1, len(_bigrams(query)) // 3)


def lexical_search(query: str, catalog: dict, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """确定性词法检索（无网络、无全局随机）：返回商品记录（按相关性降序）。"""
    query = str(query or "").strip()
    if not query or not catalog:
        return []
    terms = _terms(query)
    scored: list[tuple[int, str, dict]] = []
    for product_id in sorted(catalog):
        product = catalog[product_id]
        strong, category_overlap, overlap = match_score(product, terms, query)
        if not is_relevant(strong, overlap, query):
            continue
        score = strong * 100 + category_overlap * 10 + overlap
        scored.append((score, product_id, product))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:max(1, int(top_k))]]


def _vector_search(query: str, catalog: dict, embedder, top_k: int,
                   index_path: str | Path | None) -> list[dict]:
    """向量路：载入独立商品索引 → 命名空间断言 → 检索 → 映射回商品记录。"""
    from app.agent.rag.retriever import KnowledgeRetriever

    path = Path(index_path) if index_path else PRODUCT_INDEX_PATH
    if not path.exists():
        return []
    backend = create_backend("numpy", index_path=path)
    backend.load()
    chunks = backend.chunks()
    assert_product_namespace(chunks)
    retriever = KnowledgeRetriever(embedder=embedder, backend=backend)
    retriever.load()
    hits = retriever.search(query, top_k=max(1, int(top_k)) * 2)
    out: list[dict] = []
    seen: set[str] = set()
    for hit in hits:
        product_id = Path(str(hit.chunk.source_path or "")).stem
        product = catalog.get(product_id)
        if product is None or product_id in seen:
            continue
        seen.add(product_id)
        out.append(product)
        if len(out) >= max(1, int(top_k)):
            break
    return out


def search_products(query: str, catalog: dict, *, top_k: int = DEFAULT_TOP_K,
                    embedder=None,
                    index_path: str | Path | None = None) -> list[dict]:
    """商品知识库检索：向量索引可用（且显式给了 embedder）走向量，否则词法。

    任何向量路异常（索引损坏 / 模型不一致 / 命名空间越界）都回落词法——
    调用方只需处理「有结果 / 无结果」，不需要处理异常。
    """
    query = str(query or "").strip()
    if not query or not catalog:
        return []
    if embedder is not None:
        try:
            hits = _vector_search(query, catalog, embedder, top_k, index_path)
            if hits:
                return hits
        except ProductNamespaceViolation:
            # 命名空间越界是配置错误：绝不返回可疑结果，回落词法并留痕
            log.info("⚠️  商品索引命名空间校验失败，回落词法检索")
        except Exception as e:  # noqa: BLE001 —— 商品库不可用不得影响工具可用性
            log.info(f"⚠️  商品向量检索不可用（{type(e).__name__}），回落词法检索")
    return lexical_search(query, catalog, top_k=top_k)
