"""P2-1 扩容数据生成器：确定性幂等 / 种子逐字节保留 / 规模达标 / 加载契约。

覆盖验收：
- 同一 seed 两次生成逐字节一致（生成 + 落盘）；
- 重复运行幂等（同路径重写字节不变、条目数不增长）；
- u1–u5 既有种子条目在生成结果中逐条原样保留（追加不改写）；
- 规模：订单 ≥ 1000（20 用户、五状态齐全）、商品 ≥ 100（类目/价格带齐全）、
  物流按时间线生成；
- ``mock_data.get_dataset`` 的合并语义：生成条目只增、既有键永不覆盖、
  文件缺失回落内置种子。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agent.tools import mock_data
from app.scripts import seed_commerce_data as gen

SEEDED_STATUSES = {"pending", "shipped", "delivered", "refund_processing", "cancelled"}


# ============================================================
# 确定性 / 幂等
# ============================================================
def test_generation_is_byte_identical_across_runs():
    first = gen.dataset_json(gen.generate_dataset(gen.DEFAULT_SEED))
    second = gen.dataset_json(gen.generate_dataset(gen.DEFAULT_SEED))
    assert first == second


def test_generation_differs_across_seeds():
    base = gen.generate_dataset(gen.DEFAULT_SEED)
    other = gen.generate_dataset(gen.DEFAULT_SEED + 1)
    assert gen.dataset_json(base) != gen.dataset_json(other)


def test_writer_is_idempotent_on_same_path(tmp_path: Path):
    path = tmp_path / "commerce_seed.json"
    gen.write_dataset(gen.generate_dataset(), path)
    first_bytes = path.read_bytes()
    order_count = len(json.loads(first_bytes)["orders"])

    gen.write_dataset(gen.generate_dataset(), path)
    assert path.read_bytes() == first_bytes  # 重复运行逐字节不变
    assert len(json.loads(path.read_bytes())["orders"]) == order_count  # 不产生重复条目


def test_write_dataset_to_two_paths_is_byte_identical(tmp_path: Path):
    dataset = gen.generate_dataset()
    a = gen.write_dataset(dataset, tmp_path / "a.json")
    b = gen.write_dataset(dataset, tmp_path / "b.json")
    assert a.read_bytes() == b.read_bytes()


def test_cli_check_passes_then_detects_drift(tmp_path: Path):
    path = tmp_path / "commerce_seed.json"
    assert gen.main(["--out", str(path), "--no-docs"]) == 0
    assert gen.main(["--out", str(path), "--no-docs", "--check"]) == 0

    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert gen.main(["--out", str(path), "--no-docs", "--check"]) == 1


# ============================================================
# 种子逐字节保留（追加不改写）
# ============================================================
@pytest.mark.parametrize(
    "section,seeds",
    [
        ("orders", mock_data.SEED_ORDERS),
        ("products", mock_data.SEED_PRODUCTS),
        ("logistics", mock_data.SEED_LOGISTICS),
    ],
)
def test_seed_entries_survive_generation_byte_identical(section, seeds):
    """落盘 JSON 里的种子条目与 mock_data 内置种子逐条逐字节一致。"""
    loaded = json.loads(gen.dataset_json(gen.generate_dataset()))[section]
    for key, seed_entry in seeds.items():
        assert key in loaded, f"{section}/{key} 在扩容结果中丢失"
        # 同一序列化口径（含键序）下逐字节一致
        assert (
            json.dumps(loaded[key], ensure_ascii=False)
            == json.dumps(seed_entry, ensure_ascii=False)
        )
        assert loaded[key] == seed_entry


def test_seed_entries_keep_original_position_first():
    """种子条目排在新增条目之前（文件顺序 = 扩容前 + 追加）。"""
    dataset = gen.generate_dataset()
    for section, seeds in (
        ("orders", mock_data.SEED_ORDERS),
        ("products", mock_data.SEED_PRODUCTS),
        ("logistics", mock_data.SEED_LOGISTICS),
    ):
        keys = list(dataset[section])
        assert keys[: len(seeds)] == list(seeds)


def test_generated_orders_do_not_touch_seed_users():
    """新增订单只落在 u6–u20：u1–u5 保持 1 单，既有列表/评测口径不变。"""
    dataset = gen.generate_dataset()
    generated_ids = [k for k in dataset["orders"] if k not in mock_data.SEED_ORDERS]
    generated_users = {dataset["orders"][k]["user_id"] for k in generated_ids}
    assert generated_users == {user_id for user_id, _ in gen.EXTRA_USERS}
    for user_id in ("u1", "u2", "u3", "u4", "u5"):
        assert sum(
            1 for order in dataset["orders"].values() if order["user_id"] == user_id
        ) == 1


# ============================================================
# 规模与数据完整性
# ============================================================
def test_scale_targets_met():
    stats = gen.dataset_stats(gen.generate_dataset())
    assert stats["orders"] == len(mock_data.SEED_ORDERS) + len(gen.EXTRA_USERS) * gen.ORDERS_PER_USER
    assert stats["orders"] >= 1000
    assert stats["products"] >= 100
    assert stats["users"] == 20
    assert stats["logistics"] >= 500


def test_status_distribution_covers_all_five_states():
    dataset = gen.generate_dataset()
    statuses = {order["status"] for order in dataset["orders"].values()}
    assert statuses == SEEDED_STATUSES
    stats = gen.dataset_stats(dataset)["statuses"]
    assert all(count > 0 for count in stats.values())


def test_categories_and_price_bands_are_complete():
    products = gen.generate_dataset()["products"]
    categories = {p["category"] for p in products.values()}
    assert len(categories) >= 10
    prices = sorted(p["price"] for p in products.values())
    assert prices[0] < 200 and prices[-1] > 3000  # 低价带与高价带都有
    assert any(p["stock"] == 0 for p in products.values())  # 缺货样本
    assert all(p["specs"] for p in products.values())  # 规格齐全
    assert len({p["name"] for p in products.values()}) == len(products)  # 无重名


def test_identifiers_unique_and_never_collide_with_seeds():
    dataset = gen.generate_dataset()
    orders = dataset["orders"]
    products = dataset["products"]
    logistics = dataset["logistics"]
    assert set(orders) & set(mock_data.SEED_ORDERS) == set(mock_data.SEED_ORDERS)
    assert set(products) & set(mock_data.SEED_PRODUCTS) == set(mock_data.SEED_PRODUCTS)
    tracking = [o["tracking_number"] for o in orders.values() if o["tracking_number"]]
    assert len(tracking) == len(set(tracking))
    assert set(tracking) & set(mock_data.SEED_LOGISTICS) == set(mock_data.SEED_LOGISTICS)
    assert set(logistics) == set(tracking) | set(mock_data.SEED_LOGISTICS)


def test_order_fields_are_consistent_with_status():
    dataset = gen.generate_dataset()
    for order_id, order in dataset["orders"].items():
        if order_id in mock_data.SEED_ORDERS:
            continue
        status = order["status"]
        assert order["total"] == round(
            sum(item["price"] * item["quantity"] for item in order["items"]), 2
        )
        assert order["items"], f"{order_id} 无商品行"
        if status in ("shipped", "delivered", "refund_processing"):
            assert order["shipped_at"] and order["tracking_number"] and order["carrier"]
            assert order["estimated_delivery"]
        else:
            assert order["tracking_number"] is None and order["shipped_at"] is None
        if status == "delivered":
            assert order["delivered_at"] >= order["shipped_at"]
        if status == "refund_processing":
            assert order["delivered_at"] and order["refund_status"] == "审核中"
            assert order["refund_reason"] and order["refund_requested_at"]
        if status == "cancelled":
            assert order["cancelled_at"] and order["cancel_reason"]
        if status == "pending":
            assert "delivered_at" not in order


def test_logistics_timeline_is_monotonic_and_matches_order():
    dataset = gen.generate_dataset()
    orders_by_tracking = {
        order["tracking_number"]: order
        for order in dataset["orders"].values()
        if order["tracking_number"]
    }
    checked = 0
    for tracking, record in dataset["logistics"].items():
        if tracking in mock_data.SEED_LOGISTICS:
            continue
        events = record["events"]
        assert len(events) >= 5
        # "%Y-%m-%d %H:%M" 定宽格式：字符串序即时间序（无需解析成 datetime）
        times = [event["time"] for event in events]
        assert times == sorted(times) and len(set(times)) == len(times)
        order = orders_by_tracking[tracking]
        assert events[0]["time"] == order["shipped_at"][:16]  # 揽收 == 发货时间
        delivered = record["status"] == "delivered"
        assert (events[-1]["description"] == "已签收") is delivered
        assert (order["status"] in ("delivered", "refund_processing")) is delivered
        if delivered:
            assert events[-1]["time"] == order["delivered_at"][:16]
        checked += 1
    assert checked >= 500


# ============================================================
# mock_data 加载契约（追加不改写 / 缺失回落）
# ============================================================
def test_module_level_seeds_stay_at_pre_expansion_scale():
    """模块级 ORDERS/PRODUCTS/LOGISTICS 只含种子（既有单测口径不被扩容打破）。"""
    assert mock_data.ORDERS == mock_data.SEED_ORDERS
    assert mock_data.PRODUCTS == mock_data.SEED_PRODUCTS
    assert mock_data.LOGISTICS == mock_data.SEED_LOGISTICS
    assert len(mock_data.ORDERS) == 5 and len(mock_data.PRODUCTS) == 6


def test_get_dataset_falls_back_to_seeds_without_file(tmp_path: Path):
    dataset = mock_data.get_dataset(tmp_path / "missing.json")
    assert dataset["orders"] == mock_data.SEED_ORDERS
    assert dataset["products"] == mock_data.SEED_PRODUCTS
    assert dataset["logistics"] == mock_data.SEED_LOGISTICS


def test_load_generated_data_rejects_broken_files(tmp_path: Path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert mock_data.load_generated_data(broken) is None

    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"orders": {}}), encoding="utf-8")
    assert mock_data.load_generated_data(incomplete) is None
    assert mock_data.load_generated_data(tmp_path / "nope.json") is None


def test_get_dataset_merges_append_only(tmp_path: Path):
    """合成数据集：既有键被种子守住，新键追加进来。"""
    path = tmp_path / "commerce_seed.json"
    path.write_text(json.dumps({
        "orders": {
            "ORD-20240115-001": {"order_id": "HIJACK", "user_id": "attacker"},
            "ORD-GEN-TEST-0001": {"order_id": "ORD-GEN-TEST-0001", "user_id": "u6"},
        },
        "products": {"SKU-NEW-1": {"product_id": "SKU-NEW-1", "name": "新品"}},
        "logistics": {"XX0001": {"tracking_number": "XX0001", "events": []}},
    }, ensure_ascii=False), encoding="utf-8")

    dataset = mock_data.get_dataset(path)
    # 既有键：内置种子胜出（黄金集 ground truth 不可被覆盖）
    assert dataset["orders"]["ORD-20240115-001"] == mock_data.SEED_ORDERS["ORD-20240115-001"]
    # 新键：追加生效
    assert dataset["orders"]["ORD-GEN-TEST-0001"]["user_id"] == "u6"
    assert "SKU-NEW-1" in dataset["products"]
    assert "XX0001" in dataset["logistics"]
    assert len(dataset["orders"]) == len(mock_data.SEED_ORDERS) + 1


def test_shipped_dataset_matches_fresh_generation():
    """仓库内已落盘的数据集必须与重新生成逐字节一致（无漂移）。"""
    path = mock_data.GENERATED_DATA_PATH
    if not path.exists():
        pytest.skip("扩容数据集未落盘（先运行 python -m app.scripts.seed_commerce_data）")
    assert path.read_text(encoding="utf-8") == gen.dataset_json(gen.generate_dataset())


# ============================================================
# 网关接线（P2-1 补齐）：扩容数据集必须真的可被访问到
# ============================================================
def test_gateway_serves_expanded_dataset_by_default():
    """MockCommerceGateway 默认取扩容数据集——否则扩容只是空转。

    回归背景：生成器与商品库先行落地时，订单侧仍读模块级 5 条种子，
    「1000+ 订单」在业务面上不可达。
    """
    from app.agent.tools.mock_data import get_dataset
    from app.integrations.commerce.mock import (
        DEFAULT_ORDER_LIST_LIMIT,
        MockCommerceGateway,
    )

    dataset = get_dataset()
    assert len(dataset["orders"]) > 5, "扩容数据集未生效（仍是纯种子）"

    gw = MockCommerceGateway()
    # 无身份 → 全部（legacy 契约），但有界；total 表达总量
    data = gw.list_orders("").data
    assert data["total"] == len(dataset["orders"])
    assert len(data["orders"]) <= DEFAULT_ORDER_LIST_LIMIT

    # 种子用户 u1 的订单原样可查（黄金集 ground truth 依赖）
    assert gw.get_order("u1", "ORD-20240115-001").success is True
    mine = gw.list_orders("u1").data
    assert "ORD-20240115-001" in {o["order_id"] for o in mine["orders"]}


def test_gateway_order_list_is_bounded_and_recent_first():
    """订单列表有界且最近优先（扩容后不得无界返回）。"""
    from app.integrations.commerce.mock import MockCommerceGateway

    gw = MockCommerceGateway()
    data = gw.list_orders("", limit=10).data
    assert len(data["orders"]) == 10
    assert data["total"] > 10
    dates = [str(o.get("created_at", "")) for o in data["orders"]]
    assert dates == sorted(dates, reverse=True)


def test_gateway_explicit_orders_override_still_works():
    """显式传入 orders/logistics 仍优先（评测/测试的隔离口径不被破坏）。"""
    from app.agent.tools.mock_data import ORDERS
    from app.integrations.commerce.mock import MockCommerceGateway

    gw = MockCommerceGateway(orders=ORDERS, logistics={})
    assert gw.list_orders("").data["total"] == len(ORDERS)
    assert gw.list_orders("").data["total"] == 5
