"""P2-1 商品知识库：独立命名空间 / 与政策库物理隔离 / 检索召回 / 向量索引路。

覆盖验收：
- 商品文档渲染带 ``authority: product`` + ``namespace: product``；
- 命名空间隔离：文档目录在政策库扫描根之外、路径不在 governance 允许范围内、
  政策校验器拒绝 product authority（误入政策索引即 fail-closed）、
  索引文件与政策库索引不同、政策检索模块零商品依赖；
- 检索召回：种子商品与扩容商品都能按关键词命中，无关查询不误召回；
- 向量索引路（独立文件、命名空间断言、非商品 chunk 拒绝）；
- ``query_product`` 接入：ID 直查 / 关键词 / 无关查询回落 mock 兜底 / 返回副本。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.agent.rag import governance, product_kb
from app.agent.rag.chunker import Chunk
from app.agent.tools import mock_data
from app.agent.tools.product import query_product
from app.config.settings import settings
from app.scripts import seed_commerce_data as gen

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = (
    REPO_ROOT / "app" / "agent" / "skills" / "definitions"
    / "product-recommend" / "SKILL.md"
)


def _catalog() -> dict:
    return gen.generate_dataset()["products"]


# ============================================================
# 文档渲染
# ============================================================
def test_render_product_doc_has_product_namespace_frontmatter():
    from app.agent.rag.loader import parse_frontmatter

    product = mock_data.SEED_PRODUCTS["SHOE-270-BK-42"]
    meta, body = parse_frontmatter(product_kb.render_product_doc(product))
    assert meta["authority"] == product_kb.PRODUCT_AUTHORITY == "product"
    assert meta["namespace"] == product_kb.PRODUCT_NAMESPACE
    assert meta["status"] == "active"
    assert meta["product_id"] == product["product_id"]
    assert meta["effective_date"] == product_kb.PRODUCT_DOC_EFFECTIVE_DATE
    assert product["name"] in body
    assert "¥899.00" in body
    for value in product["specs"].values():
        assert str(value) in body


def test_write_product_docs_is_idempotent_and_cleans_stale(tmp_path: Path):
    docs_dir = tmp_path / "product_docs"
    catalog = {
        pid: copy.deepcopy(mock_data.SEED_PRODUCTS[pid])
        for pid in ("SHOE-270-BK-42", "ELEC-APP-002")
    }
    assert product_kb.write_product_docs(catalog, docs_dir) == 2
    first = {p.name: p.read_bytes() for p in docs_dir.glob("*.md")}
    assert set(first) == {"SHOE-270-BK-42.md", "ELEC-APP-002.md"}

    assert product_kb.write_product_docs(catalog, docs_dir) == 2
    assert {p.name: p.read_bytes() for p in docs_dir.glob("*.md")} == first

    # 下架商品 → 陈旧商品文档被清理；非本命名空间文件不受影响
    (docs_dir / "keep.md").write_text("---\nauthority: platform\n---\n政策\n", encoding="utf-8")
    assert product_kb.write_product_docs({"SHOE-270-BK-42": catalog["SHOE-270-BK-42"]}, docs_dir) == 1
    assert product_kb.iter_product_doc_ids(docs_dir) == ["SHOE-270-BK-42"]
    assert (docs_dir / "keep.md").exists()


def test_product_doc_chunks_are_namespaced(tmp_path: Path):
    docs_dir = tmp_path / "product_docs"
    product_kb.write_product_docs(_catalog(), docs_dir)
    chunks = product_kb.product_doc_chunks(docs_dir)
    assert chunks
    assert {chunk.authority for chunk in chunks} == {product_kb.PRODUCT_AUTHORITY}
    assert {Path(chunk.source_path).stem for chunk in chunks} == set(
        product_kb.iter_product_doc_ids(docs_dir)
    )
    product_kb.assert_product_namespace(chunks)  # 不抛即通过


def test_assert_product_namespace_rejects_foreign_chunks():
    policy_chunk = Chunk(
        chunk_id="七天无理由商品清单#00-00",
        doc="七天无理由商品清单",
        section="适用范围",
        text="部分商品不支持七天无理由退货。",
        source_path="七天无理由商品清单.md",
        authority="platform",
    )
    with pytest.raises(product_kb.ProductNamespaceViolation):
        product_kb.assert_product_namespace([policy_chunk])

    nested = Chunk(
        chunk_id="SHOE-270-BK-42#00-00",
        doc="SHOE-270-BK-42",
        section="商品信息",
        text="x",
        source_path="nested/SHOE-270-BK-42.md",
        authority=product_kb.PRODUCT_AUTHORITY,
    )
    with pytest.raises(product_kb.ProductNamespaceViolation):
        product_kb.assert_product_namespace([nested])


# ============================================================
# 命名空间隔离（不与政策库混检）
# ============================================================
def test_product_docs_are_outside_policy_kb_scan_root():
    docs_dir = product_kb.product_docs_dir().resolve()
    kb_dir = (REPO_ROOT / settings.kb_dir).resolve()
    assert docs_dir != kb_dir
    assert not docs_dir.is_relative_to(kb_dir)

    rel = docs_dir.relative_to(REPO_ROOT).as_posix()
    assert governance.is_indexable(f"{rel}/SHOE-270-BK-42.md") is False
    assert governance.is_allowed_scope(f"{rel}/SHOE-270-BK-42.md") is False


def test_policy_metadata_validator_rejects_product_authority():
    """商品文档即使误入政策库扫描范围，也过不了治理校验（fail-closed）。"""
    product_meta = {
        "status": "active",
        "authority": product_kb.PRODUCT_AUTHORITY,
        "effective_date": product_kb.PRODUCT_DOC_EFFECTIVE_DATE,
    }
    with pytest.raises(governance.DocumentGovernanceError):
        governance.validate_metadata("product_docs/SHOE-270-BK-42.md", product_meta)
    # 正对照：合法平台文档通过（说明拒绝的是 product 命名空间，不是校验器坏了）
    governance.validate_metadata("退款政策.md", {
        "status": "active", "authority": "platform", "effective_date": "2024-01-01",
    })


def test_product_index_path_is_isolated_from_policy_index():
    assert product_kb.product_index_path().resolve() != (
        REPO_ROOT / settings.kb_index_path
    ).resolve()
    assert product_kb.product_docs_dir().resolve() != (REPO_ROOT / settings.kb_dir).resolve()
    # 商品索引文件不在政策库索引目录内
    assert not product_kb.product_index_path().resolve().is_relative_to(
        (REPO_ROOT / settings.kb_dir).resolve()
    )


def test_policy_retrieval_modules_have_no_product_dependency():
    """政策检索链路（knowledge/retriever_factory/rejection）不引用商品库。"""
    for rel in (
        "app/agent/tools/knowledge.py",
        "app/agent/rag/retriever_factory.py",
        "app/agent/rag/rejection.py",
        "app/agent/rag/query_rewrite.py",
    ):
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "product_kb" not in source, f"{rel} 不应依赖商品库"
        assert "product_docs" not in source, f"{rel} 不应引用商品文档目录"


def test_product_kb_module_does_not_import_policy_retrieval():
    """商品库模块不得 import 政策检索链路（源码级 AST 校验，注释提及不算）。"""
    import ast

    source = (REPO_ROOT / "app/agent/rag/product_kb.py").read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    forbidden_prefixes = (
        "app.agent.tools.knowledge",
        "app.agent.rag.retriever_factory",
        "app.agent.rag.rejection",
        "app.agent.rag.query_rewrite",
    )
    for module in imported:
        assert not module.startswith(forbidden_prefixes), f"商品库不应 import {module}"


# ============================================================
# 检索召回（词法路，确定性、零网络）
# ============================================================
@pytest.mark.parametrize(
    "query,expected_id",
    [
        ("Nike Air Max 270 运动鞋 多少钱", "SHOE-270-BK-42"),
        ("Apple AirPods Pro 2", "ELEC-APP-002"),
        ("小米14 Ultra 手机 价格", "PHONE-MI14U-BK"),
        ("Levi's 501 经典牛仔裤 有货吗", "CLOTH-LEVI-501-30"),
        ("戴森 V15 吸尘器", "HOME-DYSON-V15"),
        ("AirPods 保护壳 价格", "ACC-AP-CASE-01"),
    ],
)
def test_full_name_queries_rank_seed_product_first(query, expected_id):
    """黄金集 product_* 用例按「商品名 → 价格」断言，种子商品必须排第一。"""
    hits = product_kb.lexical_search(query, _catalog())
    assert hits, f"{query!r} 无召回"
    assert hits[0]["product_id"] == expected_id


@pytest.mark.parametrize(
    "query,category",
    [
        ("运动鞋", "运动鞋"), ("有运动鞋卖吗", "运动鞋"),
        ("耳机", "耳机"), ("有耳机卖吗", "耳机"),
        ("手机", "手机"), ("有手机卖吗", "手机"),
        ("牛仔裤", "牛仔裤"), ("有牛仔裤卖吗", "牛仔裤"),
    ],
)
def test_category_queries_return_that_category(query, category):
    """类目问法（含「有 X 卖吗」口语式）命中的是同品类商品，而非名称巧合的配件。"""
    hits = product_kb.lexical_search(query, _catalog(), top_k=8)
    assert hits, f"{query!r} 无召回"
    assert hits[0]["category"] == category
    assert sum(1 for p in hits if p["category"] == category) >= len(hits) - 1


def test_lexical_recall_on_expanded_products():
    catalog = _catalog()
    generated = {pid: p for pid, p in catalog.items() if pid not in mock_data.SEED_PRODUCTS}
    assert len(generated) >= 100
    for query, category in (("笔记本电脑", "笔记本电脑"), ("相机", "相机"), ("箱包", "箱包")):
        hits = product_kb.lexical_search(query, catalog)
        assert hits and hits[0]["category"] == category


def test_lexical_search_is_deterministic_and_bounded():
    catalog = _catalog()
    first = [p["product_id"] for p in product_kb.lexical_search("运动", catalog, top_k=5)]
    second = [p["product_id"] for p in product_kb.lexical_search("运动", catalog, top_k=5)]
    assert first == second
    assert len(first) <= 5


def test_unrelated_query_has_no_recall():
    catalog = _catalog()
    assert product_kb.lexical_search("zzzzqqqq", catalog) == []
    # 政策类关键词不得召回商品（政策库与商品库不混检）
    assert product_kb.lexical_search("七天无理由退货政策", catalog) == []


# ============================================================
# 向量索引路（独立文件 + 命名空间断言）
# ============================================================
def test_vector_index_is_isolated_and_used_when_embedder_provided(tmp_path: Path):
    from tests.unit.conftest import FakeEmbedder

    catalog = {"SHOE-270-BK-42": copy.deepcopy(mock_data.SEED_PRODUCTS["SHOE-270-BK-42"])}
    docs_dir = tmp_path / "product_docs"
    index_path = tmp_path / "product_kb_index.json"
    embedder = FakeEmbedder()

    chunks = product_kb.build_product_index(
        catalog, embedder, index_path=index_path, docs_dir=docs_dir,
    )
    assert chunks >= 1 and index_path.exists()
    assert index_path.resolve() != (REPO_ROOT / settings.kb_index_path).resolve()

    # 无关查询在词法路无命中；向量路仍返回索引内商品 → 证明走向量而非词法回落
    assert product_kb.lexical_search("zzzzqqqq", catalog) == []
    hits = product_kb.search_products(
        "zzzzqqqq", catalog, embedder=embedder, index_path=index_path,
    )
    assert [p["product_id"] for p in hits] == ["SHOE-270-BK-42"]


def test_vector_search_rejects_foreign_namespace_index(tmp_path: Path):
    """索引里混入政策 chunk → 命名空间断言拒绝，绝不当作商品返回。"""
    from tests.unit.conftest import FakeEmbedder

    index_path = tmp_path / "polluted_index.json"
    policy_chunk = Chunk(
        chunk_id="退款政策#00-00",
        doc="退款政策",
        section="退款条件",
        text="七天无理由退货政策说明。",
        source_path="退款政策.md",
        authority="platform",
    )
    backend = product_kb.create_backend("numpy", index_path=index_path)
    embedder = FakeEmbedder()
    backend.upsert([policy_chunk], embedder.encode([policy_chunk.text]), embedder.model)

    catalog = {"SHOE-270-BK-42": copy.deepcopy(mock_data.SEED_PRODUCTS["SHOE-270-BK-42"])}
    with pytest.raises(product_kb.ProductNamespaceViolation):
        product_kb._vector_search("退款", catalog, embedder, 3, index_path)
    # 公开入口 fail-open 到词法：政策内容绝不进商品结果
    hits = product_kb.search_products(
        "退款", catalog, embedder=embedder, index_path=index_path,
    )
    assert all(p["product_id"] in catalog for p in hits)


# ============================================================
# query_product 接入
# ============================================================
def test_query_product_exact_id_fast_path():
    out = query_product("SHOE-270-BK-42")
    assert out["success"] is True
    assert [p["product_id"] for p in out["products"]] == ["SHOE-270-BK-42"]
    assert out["products"][0]["price"] == 899.00


def test_query_product_returns_copies_not_catalog_references():
    out = query_product("SHOE-270-BK-42")
    out["products"][0]["price"] = 0.0
    assert mock_data.get_dataset()["products"]["SHOE-270-BK-42"]["price"] == 899.00


def test_query_product_keyword_recall_covers_expanded_catalog():
    out = query_product("笔记本电脑")
    ids = [p["product_id"] for p in out["products"]]
    assert ids and all(pid.startswith("LAPTOP-") for pid in ids)
    assert len(ids) <= 8  # 单次返回上限


def test_query_product_falls_back_to_mock_product_when_no_hit():
    out = query_product("zzzzqqqq")
    assert out["success"] is True
    assert len(out["products"]) == 1
    assert out["products"][0]["product_id"].startswith("MOCK-")


def test_query_product_policy_query_does_not_return_catalog_products():
    out = query_product("七天无理由退货政策")
    assert all(p["product_id"].startswith("MOCK-") for p in out["products"])


def test_query_product_uses_expanded_catalog_when_dataset_present():
    catalog = mock_data.get_dataset()["products"]
    if len(catalog) <= len(mock_data.SEED_PRODUCTS):
        pytest.skip("扩容数据集未落盘")
    out = query_product("相机")
    assert any(pid not in mock_data.SEED_PRODUCTS for pid in
               [p["product_id"] for p in out["products"]])


def test_golden_product_cases_still_satisfied_by_tool():
    """扩容后黄金集 product_* 用例的 expected_keywords 仍被工具结果满足（不降）。"""
    path = REPO_ROOT / "app" / "evaluation" / "cases_large.json"
    if not path.exists():
        pytest.skip("黄金集未落盘")
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    checked = 0
    for case in cases:
        if "query_product" not in (case.get("expected_tools") or []):
            continue
        for turn in case.get("turns", []):
            blob = json.dumps(query_product(turn), ensure_ascii=False)
            for keyword in case.get("expected_keywords", []):
                assert keyword in blob, f"{case['id']} {turn!r} 结果缺少 {keyword!r}"
            checked += 1
    assert checked >= 40


# ============================================================
# 技能联动
# ============================================================
def test_product_recommend_skill_documents_product_kb():
    text = SKILL_PATH.read_text(encoding="utf-8")
    assert "query_product" in text
    assert "商品知识库" in text
    assert "独立索引" in text  # 明确「政策库与商品库两套独立索引」


# ============================================================
# 落盘产物一致性（数据集存在时）
# ============================================================
def test_shipped_product_docs_match_dataset():
    docs_dir = product_kb.product_docs_dir()
    if not mock_data.GENERATED_DATA_PATH.exists():
        pytest.skip("扩容数据集未落盘")
    dataset = json.loads(mock_data.GENERATED_DATA_PATH.read_text(encoding="utf-8"))
    assert product_kb.iter_product_doc_ids() == sorted(dataset["products"])
    for product_id, product in dataset["products"].items():
        path = docs_dir / f"{product_id}.md"
        assert path.read_text(encoding="utf-8") == product_kb.render_product_doc(product)
