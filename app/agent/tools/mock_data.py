"""电商 Mock 数据：订单、商品、物流（阶段一：订单增加 user_id 供按用户过滤）

P2-1 扩容（追加不改写）：
- 下面的 ``ORDERS`` / ``PRODUCTS`` / ``LOGISTICS`` 是**内置种子**，逐字节冻结——
  u1–u5 的既有条目被 ``app/evaluation/cases_large.json`` 黄金集 ground truth
  绑定，任何改写都会让评测口径失真；
- 扩容数据由 ``app/scripts/seed_commerce_data.py`` 确定性生成到
  ``app/agent/tools/data/commerce_seed.json``，经 :func:`get_dataset` 按
  「既有键优先、只增不改」合并后读取（文件缺失 → 回落内置种子，import 契约
  与行为不变）；
- 模块级 ``ORDERS`` / ``PRODUCTS`` / ``LOGISTICS`` 永远只含种子条目，
  既有调用方（含断言「全部订单数 == 5」的单测）不受扩容影响；
- 需要让 Mock 网关服务扩容订单时**显式接线**（不默认接管，否则会打破
  ``test_commerce_gateway`` / ``test_tool_context`` 的「全部订单数 == 5」口径）：
  ``MockCommerceGateway(orders=get_dataset()["orders"],
  logistics=get_dataset()["logistics"])``；商品侧已默认走扩容目录
  （``app/agent/tools/product.py`` → ``get_dataset()["products"]``）。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

ORDERS = {
    "ORD-20240115-001": {
        "order_id": "ORD-20240115-001",
        "user": "小明",
        "user_id": "u1",
        "status": "shipped",
        "items": [
            {"name": "Nike Air Max 270 运动鞋", "sku": "SHOE-270-BK-42", "quantity": 1, "price": 899.00}
        ],
        "total": 899.00,
        "created_at": "2024-01-15 10:30:00",
        "shipped_at": "2024-01-16 14:20:00",
        "tracking_number": "SF1234567890",
        "carrier": "顺丰速运",
        "estimated_delivery": "2024-01-19",
    },
    "ORD-20240120-002": {
        "order_id": "ORD-20240120-002",
        "user": "小红",
        "user_id": "u2",
        "status": "pending",
        "items": [
            {"name": "Apple AirPods Pro 2", "sku": "ELEC-APP-002", "quantity": 1, "price": 1799.00},
            {"name": "AirPods 保护壳（透明）", "sku": "ACC-AP-CASE-01", "quantity": 1, "price": 29.90},
        ],
        "total": 1828.90,
        "created_at": "2024-01-20 09:15:00",
        "shipped_at": None,
        "tracking_number": None,
        "carrier": None,
        "estimated_delivery": None,
    },
    "ORD-20240110-003": {
        "order_id": "ORD-20240110-003",
        "user": "大壮",
        "user_id": "u3",
        "status": "delivered",
        "items": [
            {"name": "小米14 Ultra 手机", "sku": "PHONE-MI14U-BK", "quantity": 1, "price": 5999.00}
        ],
        "total": 5999.00,
        "created_at": "2024-01-10 16:00:00",
        "shipped_at": "2024-01-11 08:00:00",
        "tracking_number": "JD9876543210",
        "carrier": "京东物流",
        "estimated_delivery": "2024-01-13",
        "delivered_at": "2024-01-13 11:30:00",
    },
    "ORD-20240118-004": {
        "order_id": "ORD-20240118-004",
        "user": "小丽",
        "user_id": "u4",
        "status": "refund_processing",
        "items": [
            {"name": "Levi's 501 经典牛仔裤", "sku": "CLOTH-LEVI-501-30", "quantity": 1, "price": 699.00}
        ],
        "total": 699.00,
        "created_at": "2024-01-18 12:00:00",
        "shipped_at": "2024-01-19 09:00:00",
        "tracking_number": "YT6655443322",
        "carrier": "圆通速递",
        "estimated_delivery": "2024-01-22",
        "delivered_at": "2024-01-21 15:00:00",
        "refund_reason": "尺码不合适",
        "refund_status": "审核中",
        "refund_requested_at": "2024-01-22 10:00:00",
    },
    "ORD-20240122-005": {
        "order_id": "ORD-20240122-005",
        "user": "阿杰",
        "user_id": "u5",
        "status": "pending",
        "items": [
            {"name": "戴森 V15 吸尘器", "sku": "HOME-DYSON-V15", "quantity": 1, "price": 4299.00},
            {"name": "戴森 V15 替换滤芯", "sku": "HOME-DYSON-FLTR", "quantity": 2, "price": 199.00},
        ],
        "total": 4697.00,
        "created_at": "2024-01-22 20:00:00",
        "shipped_at": None,
        "tracking_number": None,
        "carrier": None,
        "estimated_delivery": None,
    },
}

PRODUCTS = {
    "SHOE-270-BK-42": {
        "product_id": "SHOE-270-BK-42",
        "name": "Nike Air Max 270 运动鞋",
        "category": "运动鞋",
        "price": 899.00,
        "stock": 156,
        "description": "经典气垫缓震，透气网面鞋身，适合日常跑步和休闲穿搭",
        "specs": {"颜色": "黑色", "尺码": "42", "材质": "网面+合成革"},
    },
    "ELEC-APP-002": {
        "product_id": "ELEC-APP-002",
        "name": "Apple AirPods Pro 2",
        "category": "耳机",
        "price": 1799.00,
        "stock": 89,
        "description": "主动降噪，自适应透明模式，个性化空间音频，USB-C 充电",
        "specs": {"颜色": "白色", "连接方式": "蓝牙5.3", "续航": "6小时(ANC开启)"},
    },
    "PHONE-MI14U-BK": {
        "product_id": "PHONE-MI14U-BK",
        "name": "小米14 Ultra 手机",
        "category": "手机",
        "price": 5999.00,
        "stock": 42,
        "description": "骁龙8 Gen3，徕卡光学四摄，2K 护眼屏，5000mAh 大电池",
        "specs": {"颜色": "黑色", "存储": "16GB+512GB", "屏幕": "6.73英寸 2K AMOLED"},
    },
    "CLOTH-LEVI-501-30": {
        "product_id": "CLOTH-LEVI-501-30",
        "name": "Levi's 501 经典牛仔裤",
        "category": "牛仔裤",
        "price": 699.00,
        "stock": 0,
        "description": "经典直筒版型，原色丹宁面料，纽扣门襟",
        "specs": {"颜色": "原色", "尺码": "30", "材质": "100%棉"},
    },
    "HOME-DYSON-V15": {
        "product_id": "HOME-DYSON-V15",
        "name": "戴森 V15 Detect 吸尘器",
        "category": "家电",
        "price": 4299.00,
        "stock": 23,
        "description": "激光探测微尘，压电式声学传感器，LCD 屏幕实时显示吸入颗粒",
        "specs": {"颜色": "金色", "续航": "60分钟", "吸力": "230AW"},
    },
    "ACC-AP-CASE-01": {
        "product_id": "ACC-AP-CASE-01",
        "name": "AirPods 保护壳（透明）",
        "category": "配件",
        "price": 29.90,
        "stock": 500,
        "description": "TPU 透明软壳，防摔防刮，精准开孔",
        "specs": {"材质": "TPU", "适配": "AirPods Pro 2"},
    },
}

LOGISTICS = {
    "SF1234567890": {
        "tracking_number": "SF1234567890",
        "carrier": "顺丰速运",
        "status": "in_transit",
        "events": [
            {"time": "2024-01-16 14:20", "location": "深圳南山区", "description": "快件已揽收"},
            {"time": "2024-01-17 06:00", "location": "广州转运中心", "description": "已到达"},
            {"time": "2024-01-17 22:00", "location": "上海转运中心", "description": "已到达"},
            {"time": "2024-01-18 08:30", "location": "上海浦东区", "description": "正在派送中"},
        ],
    },
    "JD9876543210": {
        "tracking_number": "JD9876543210",
        "carrier": "京东物流",
        "status": "delivered",
        "events": [
            {"time": "2024-01-11 08:00", "location": "北京亦庄仓库", "description": "快件已出库"},
            {"time": "2024-01-12 10:00", "location": "北京海淀区", "description": "正在派送中"},
            {"time": "2024-01-13 11:30", "location": "北京海淀区", "description": "已签收"},
        ],
    },
    "YT6655443322": {
        "tracking_number": "YT6655443322",
        "carrier": "圆通速递",
        "status": "delivered",
        "events": [
            {"time": "2024-01-19 09:00", "location": "杭州余杭区", "description": "快件已揽收"},
            {"time": "2024-01-20 06:00", "location": "杭州转运中心", "description": "已发出"},
            {"time": "2024-01-20 18:00", "location": "上海转运中心", "description": "已到达"},
            {"time": "2024-01-21 09:00", "location": "上海普陀区", "description": "正在派送中"},
            {"time": "2024-01-21 15:00", "location": "上海普陀区", "description": "已签收"},
        ],
    },
}

# ============================================================
# P2-1：扩容数据集（追加不改写）
# ============================================================
DATA_DIR = Path(__file__).resolve().parent / "data"
GENERATED_DATA_PATH = DATA_DIR / "commerce_seed.json"

# 内置种子快照：生成器与测试的「扩容前原样」基准。扩容只增不改，任何生成
# 条目都不得覆盖这些键（merge_generated 用 setdefault 语义强制保证）。
SEED_ORDERS = copy.deepcopy(ORDERS)
SEED_PRODUCTS = copy.deepcopy(PRODUCTS)
SEED_LOGISTICS = copy.deepcopy(LOGISTICS)

_DATASET_SECTIONS = ("orders", "products", "logistics")

# path(str) → ((mtime_ns, size) | None, dataset)；文件变更即失效（生成器落盘后
# 同进程读取也能拿到新数据）。
_dataset_cache: dict[str, tuple[tuple[int, int] | None, dict]] = {}


def load_generated_data(path: str | Path | None = None) -> dict | None:
    """读取扩容数据集 JSON；缺失/损坏/结构非法 → None（绝不抛，回落内置种子）。

    只做结构校验（三个 section 必须都是 dict），不做内容校验——内容由生成器
    的 ``assert_seed_preserved`` 在生成侧保证。
    """
    target = Path(path) if path is not None else GENERATED_DATA_PATH
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    out: dict = {}
    for key in _DATASET_SECTIONS:
        section = data.get(key)
        if not isinstance(section, dict):
            return None
        out[key] = section
    return out


def merge_generated(base: dict, extra: dict) -> dict:
    """追加不改写合并：既有键一律保留（``setdefault`` 语义）。

    生成器输出自带种子条目，这里再兜一层——即使生成文件里出现同 ID 条目，
    模块内置种子也永远胜出（黄金集 ground truth 不可被覆盖）。
    """
    merged = dict(base)
    for key, value in extra.items():
        merged.setdefault(key, value)
    return merged


def get_dataset(path: str | Path | None = None) -> dict:
    """扩容数据集视图：``{orders, products, logistics}`` = 种子 + 生成条目。

    - 生成文件缺失 → 三个 section 即内置种子（行为与扩容前一致）；
    - 命中缓存按 (mtime_ns, size) 失效，避免每次工具调用重复解析大 JSON；
    - 返回的 dict 是独立副本（section 内条目对象与种子共享，调用方只读；
      需要修改请自行 deepcopy）。
    """
    target = Path(path) if path is not None else GENERATED_DATA_PATH
    try:
        stat = target.stat()
        stamp: tuple[int, int] | None = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        stamp = None

    key = str(target)
    cached = _dataset_cache.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    generated = load_generated_data(target)
    dataset = {
        "orders": merge_generated(ORDERS, generated["orders"]) if generated else dict(ORDERS),
        "products": merge_generated(PRODUCTS, generated["products"]) if generated else dict(PRODUCTS),
        "logistics": merge_generated(LOGISTICS, generated["logistics"]) if generated else dict(LOGISTICS),
    }
    _dataset_cache[key] = (stamp, dataset)
    return dataset


def reset_dataset_cache() -> None:
    """清空数据集缓存（测试/生成器落盘后强制重读用）。"""
    _dataset_cache.clear()
