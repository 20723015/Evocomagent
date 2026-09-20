"""P2-1 确定性 mock 商城数据生成器：订单 5 → 1000+、商品 6 → 100+、物流按时间线。

纪律（追加不改写）：
- ``app/agent/tools/mock_data.py`` 的 u1–u5 种子条目**逐字节保留**——黄金集
  ``app/evaluation/cases_large.json``（317 条）的 ground truth 绑定它们；
  生成器只新增条目，生成后还会做 ``assert_seed_preserved`` 自检（漂移即报错）；
- 新增订单全部落在 u6–u20（u1–u5 各保持 1 条既有订单），因此
  ``list_user_orders(u1)`` 等既有语义与评测口径不变。

确定性：
- 只用 ``random.Random(seed)`` 实例（按 section 派生独立流，禁用全局 random）；
- 所有遍历都基于 ``sorted(...)``，不依赖 dict/set 迭代顺序；
- 无时钟/环境依赖：同一 seed 两次生成逐字节一致，重复运行幂等（同路径重写
  字节不变，不产生重复条目）。

产出（默认路径）：
- ``app/agent/tools/data/commerce_seed.json``   扩容数据集（种子 + 新增）
- ``app/agent/tools/data/product_docs/*.md``    商品知识库文档（独立命名空间）

用法：
  python -m app.scripts.seed_commerce_data                 # 生成数据集 + 商品文档
  python -m app.scripts.seed_commerce_data --check         # 幂等校验（有漂移退出码 1）
  python -m app.scripts.seed_commerce_data --build-index   # 另建商品向量索引（需 embedding 配置）

读取侧：``app.agent.tools.mock_data.get_dataset()``（缺失 → 回落内置种子）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.tools.mock_data import (  # noqa: E402
    GENERATED_DATA_PATH,
    SEED_LOGISTICS,
    SEED_ORDERS,
    SEED_PRODUCTS,
)

DEFAULT_SEED = 20240918
GENERATOR_VERSION = 1

# 扩容规模：15 个新用户 × 67 单 = 1005 新增 + 5 种子 = 1010
EXTRA_USERS = (
    ("u6", "陈晨"), ("u7", "刘洋"), ("u8", "王芳"), ("u9", "李强"), ("u10", "张伟"),
    ("u11", "赵敏"), ("u12", "周涛"), ("u13", "吴静"), ("u14", "郑凯"), ("u15", "孙丽"),
    ("u16", "马超"), ("u17", "朱婷"), ("u18", "胡军"), ("u19", "林芳"), ("u20", "何静"),
)
ORDERS_PER_USER = 67

# 状态轮转：每个用户都覆盖五种状态（确定性覆盖，不靠随机碰运气）
STATUS_PATTERN = (
    "delivered", "delivered", "shipped", "delivered", "pending",
    "refund_processing", "delivered", "cancelled", "shipped", "pending",
)
SHIPPED_STATUSES = ("shipped", "delivered", "refund_processing")

CARRIERS = (
    ("顺丰速运", "SF"), ("京东物流", "JD"), ("圆通速递", "YT"),
    ("中通快递", "ZTO"), ("韵达速递", "YD"), ("邮政EMS", "EMS"),
)
CITIES = ("北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "苏州", "重庆", "长沙")
DISTRICTS = ("朝阳区", "海淀区", "浦东新区", "南山区", "余杭区", "武侯区", "洪山区", "雁塔区", "鼓楼区", "姑苏区", "渝北区", "岳麓区")
REFUND_REASONS = ("尺码不合适", "质量问题", "不喜欢/不想要", "发错货", "商品与描述不符")
CANCEL_REASONS = ("用户主动取消", "超时未支付", "库存不足自动取消")

# 运单号：种子为 SF/JD/YT + 10 位数字，这里从 2000000000 起编（不与既有冲突）
TRACKING_BASE = 2_000_000_000
ORDER_SEQ_BASE = 1000

# ============================================================
# 商品目录模板：12 类目 × 9 款 = 108 新增 + 6 种子 = 114
# ============================================================
_PRODUCT_SPECS: tuple[dict, ...] = (
    {
        "category": "运动鞋", "code": "SHOE", "base_price": 599.0,
        "models": (
            ("Nike", "Air Max 97"), ("Nike", "Pegasus 41"), ("李宁", "赤兔 7"),
            ("李宁", "超轻 21"), ("安踏", "马赫 5"), ("安踏", "氢跑 6"),
            ("阿迪达斯", "Ultraboost 22"), ("亚瑟士", "Gel-Kayano 30"), ("New Balance", "1080 v13"),
        ),
        "specs": {
            "颜色": ("黑色", "白色", "灰色", "蓝色", "红色"),
            "尺码": ("39", "40", "41", "42", "43", "44"),
            "材质": ("网面+合成革", "飞织", "皮革"),
        },
        "description": "轻量缓震中底，透气鞋面，适合日常通勤与慢跑",
    },
    {
        "category": "手机", "code": "PHONE", "base_price": 3999.0,
        "models": (
            ("小米", "14 Pro"), ("小米", "Redmi K70 Pro"), ("华为", "Mate 60 Pro"),
            ("华为", "nova 12"), ("OPPO", "Find X7"), ("vivo", "X100 Pro"),
            ("荣耀", "Magic6"), ("三星", "Galaxy S24"), ("一加", "12 Pro"),
        ),
        "specs": {
            "颜色": ("黑色", "白色", "青色", "紫色"),
            "存储": ("12GB+256GB", "16GB+512GB", "24GB+1TB"),
            "屏幕": ("6.36英寸 1.5K OLED", "6.73英寸 2K AMOLED", "6.78英寸 2K LTPO"),
        },
        "description": "旗舰芯片，高刷护眼屏，大底主摄，长续航快充",
    },
    {
        "category": "耳机", "code": "ELEC", "base_price": 899.0,
        "models": (
            ("Apple", "AirPods Max"), ("Apple", "AirPods 4"), ("索尼", "WF-1000XM5"),
            ("索尼", "WH-1000XM5"), ("华为", "FreeBuds Pro 3"), ("小米", "Buds 5 Pro"),
            ("Bose", "QuietComfort Ultra"), ("三星", "Galaxy Buds3 Pro"), ("漫步者", "NeoBuds Pro 2"),
        ),
        "specs": {
            "颜色": ("白色", "黑色", "银色"),
            "连接方式": ("蓝牙5.3", "蓝牙5.4", "2.4G+蓝牙双模"),
            "续航": ("6小时(ANC开启)", "8小时(ANC开启)", "30小时(含充电盒)"),
        },
        "description": "主动降噪，通透模式，低延迟连接，通话降噪",
    },
    {
        "category": "家电", "code": "HOME", "base_price": 1899.0,
        "models": (
            ("戴森", "V12 Detect Slim"), ("戴森", "Purifier Hot+Cool"), ("石头", "G20S 扫地机器人"),
            ("科沃斯", "X2 Pro"), ("美的", "空气循环扇"), ("格力", "云锦三代空调"),
            ("海尔", "洗烘一体机"), ("九阳", "破壁机 Y88"), ("苏泊尔", "电饭煲 Pro"),
        ),
        "specs": {
            "颜色": ("白色", "金色", "灰色"),
            "功率": ("1200W", "2200W", "3500W"),
            "能效等级": ("一级能效", "二级能效"),
        },
        "description": "智能控制，静音运行，易清洁维护，家用省心",
    },
    {
        "category": "牛仔裤", "code": "CLOTH", "base_price": 399.0,
        "models": (
            ("Levi's", "511 修身"), ("Levi's", "501 经典直筒"), ("Lee", "101 复古"),
            ("Wrangler", "13MWZ"), ("优衣库", "宽腿直筒"), ("太平鸟", "高腰阔腿"),
            ("杰克琼斯", "水洗直筒"), ("森马", "弹力小脚"), ("海澜之家", "商务直筒"),
        ),
        "specs": {
            "颜色": ("原色", "深蓝", "浅蓝", "黑色"),
            "尺码": ("28", "29", "30", "31", "32", "34"),
            "材质": ("100%棉", "棉+氨纶", "天丝混纺"),
        },
        "description": "经典版型，耐穿丹宁面料，水洗做旧工艺",
    },
    {
        "category": "配件", "code": "ACC", "base_price": 129.0,
        "models": (
            ("Anker", "65W 氮化镓充电器"), ("Anker", "磁吸充电宝"), ("绿联", "Type-C 扩展坞"),
            ("倍思", "车载支架"), ("闪迪", "1TB 移动固态"), ("罗技", "MX Master 3S"),
            ("小米", "无线充电板"), ("贝尔金", "三合一充电座"), ("摩米士", "防摔手机壳"),
        ),
        "specs": {
            "颜色": ("黑色", "白色", "深空灰"),
            "材质": ("TPU", "铝合金", "ABS+PC"),
            "适配": ("通用", "iPhone", "安卓"),
        },
        "description": "原厂级兼容，做工扎实，日常通勤好搭子",
    },
    {
        "category": "笔记本电脑", "code": "LAPTOP", "base_price": 5499.0,
        "models": (
            ("联想", "小新 Pro 16"), ("联想", "ThinkBook 14+"), ("华为", "MateBook X Pro"),
            ("小米", "Redmi Book Pro 15"), ("华硕", "灵耀 14"), ("戴尔", "XPS 13"),
            ("惠普", "星 Book Pro"), ("苹果", "MacBook Air M3"), ("机械革命", "极光 Pro"),
        ),
        "specs": {
            "颜色": ("银色", "深空灰", "星光色"),
            "处理器": ("i5-13500H", "i7-13700H", "R7-8845H", "Apple M3"),
            "内存/硬盘": ("16GB+512GB", "16GB+1TB", "32GB+1TB"),
        },
        "description": "高色域屏，长续航，轻薄机身，办公创作两相宜",
    },
    {
        "category": "平板电脑", "code": "PAD", "base_price": 2499.0,
        "models": (
            ("苹果", "iPad Air 11"), ("苹果", "iPad Pro 13"), ("华为", "MatePad Pro 13.2"),
            ("小米", "Pad 6S Pro"), ("荣耀", "MagicPad 13"), ("三星", "Galaxy Tab S9"),
            ("联想", "小新 Pad Pro"), ("vivo", "Pad 3 Pro"), ("OPPO", "Pad 3"),
        ),
        "specs": {
            "颜色": ("深空灰", "银色", "蓝色"),
            "存储": ("8GB+128GB", "12GB+256GB", "16GB+512GB"),
            "屏幕": ("11英寸 2.5K 120Hz", "13英寸 3K 144Hz"),
        },
        "description": "高刷全面屏，手写笔低延迟，影音娱乐与轻办公",
    },
    {
        "category": "智能手表", "code": "WATCH", "base_price": 1299.0,
        "models": (
            ("Apple", "Watch Series 9"), ("Apple", "Watch Ultra 2"), ("华为", "Watch GT4"),
            ("小米", "Watch S3"), ("佳明", "Forerunner 265"), ("颂拓", "Race S"),
            ("荣耀", "Watch 4 Pro"), ("OPPO", "Watch X"), ("三星", "Galaxy Watch6"),
        ),
        "specs": {
            "颜色": ("黑色", "银色", "钛金属"),
            "表盘": ("41mm", "45mm", "49mm"),
            "续航": ("2天", "7天", "14天"),
        },
        "description": "全天候心率血氧监测，多运动模式，独立通话",
    },
    {
        "category": "相机", "code": "CAMERA", "base_price": 4999.0,
        "models": (
            ("索尼", "A7C II"), ("索尼", "ZV-E10 II"), ("佳能", "EOS R8"),
            ("佳能", "EOS R50"), ("尼康", "Z fc"), ("富士", "X-T5"),
            ("松下", "Lumix S9"), ("奥林巴斯", "OM-5"), ("大疆", "Pocket 3"),
        ),
        "specs": {
            "颜色": ("黑色", "银色"),
            "传感器": ("全画幅", "APS-C", "1英寸"),
            "像素": ("2420万", "3300万", "4020万"),
        },
        "description": "快速对焦，五轴防抖，4K 视频录制，随身创作",
    },
    {
        "category": "箱包", "code": "BAG", "base_price": 499.0,
        "models": (
            ("新秀丽", "拉杆箱 20寸"), ("新秀丽", "商务双肩包"), ("美旅", "登机箱"),
            ("小米", "90分旅行箱"), ("瑞士军刀", "电脑背包"), ("外交官", "万向轮行李箱"),
            ("北极狐", "Kanken 双肩包"), ("JanSport", "校园背包"), ("爱华仕", "轻旅拉杆箱"),
        ),
        "specs": {
            "颜色": ("黑色", "深蓝", "灰色"),
            "尺寸": ("20寸", "24寸", "28寸"),
            "材质": ("PC+ABS", "尼龙", "帆布"),
        },
        "description": "静音万向轮，防刮箱体，TSA 海关锁，出差旅行适用",
    },
    {
        "category": "运动服饰", "code": "SPORT", "base_price": 259.0,
        "models": (
            ("Nike", "Dri-FIT 速干T恤"), ("Nike", "Tech Fleece 卫衣"), ("阿迪达斯", "AEROREADY 短袖"),
            ("阿迪达斯", "运动长裤"), ("李宁", "运动风衣"), ("安踏", "冰肤防晒衣"),
            ("lululemon", "Align 瑜伽裤"), ("安德玛", "训练背心"), ("迪卡侬", "速干短裤"),
        ),
        "specs": {
            "颜色": ("黑色", "白色", "藏青", "荧光绿"),
            "尺码": ("S", "M", "L", "XL", "XXL"),
            "材质": ("聚酯纤维", "棉+氨纶", "锦纶"),
        },
        "description": "吸湿速干，四面弹力，运动通勤两穿",
    },
)

PRICE_BANDS = (
    ("low", 0.35),   # 低价带
    ("mid", 1.0),    # 中价带
    ("high", 2.6),   # 高价带
)


# ============================================================
# 商品生成
# ============================================================
def _assert_no_name_shadowing(products: dict) -> None:
    """扩容商品名不得与种子商品名互相包含（也不得自重名）。

    黄金集 ``product_*`` 用例断言「商品名 → 价格」（如 Nike Air Max 270 → 899），
    若扩容条目在名称上遮蔽种子条目，LLM 可能引用到扩容条目的价格而让 ground
    truth 失真——生成期直接拒绝这类名字。
    """
    names = [str(p["name"]) for p in products.values()]
    if len(set(names)) != len(names):
        raise AssertionError("扩容商品名重复（会污染按名检索的排序）")
    seed_names = [str(p["name"]) for p in SEED_PRODUCTS.values()]
    for name in names:
        for seed_name in seed_names:
            if name in seed_name or seed_name in name:
                raise AssertionError(
                    f"扩容商品名遮蔽种子商品名: {name!r} vs {seed_name!r}"
                )


def generate_products(seed: int = DEFAULT_SEED) -> dict:
    """生成扩容商品目录（确定性；含类目/规格/价格带/缺货样本）。"""
    rng = Random(f"{seed}:products")
    products: dict[str, dict] = {}
    seq = 0
    for spec in _PRODUCT_SPECS:
        for idx, (brand, model) in enumerate(spec["models"]):
            seq += 1
            band_index = 0 if idx < 3 else (1 if idx < 6 else 2)
            _, factor = PRICE_BANDS[band_index]
            base = spec["base_price"] * factor
            price = round(base * rng.uniform(0.92, 1.08), 2)
            product_id = f"{spec['code']}-GEN-{seq:04d}"
            specs = {
                key: rng.choice(options)
                for key, options in spec["specs"].items()
            }
            # 每 11 款留一个缺货样本（推荐技能需演示「库存为 0 换替代品」）
            stock = 0 if seq % 11 == 5 else rng.randrange(8, 480)
            products[product_id] = {
                "product_id": product_id,
                "name": f"{brand} {model} {spec['category']}",
                "category": spec["category"],
                "price": price,
                "stock": stock,
                "description": spec["description"],
                "specs": specs,
            }
    _assert_no_name_shadowing(products)
    return products


# ============================================================
# 订单 + 物流生成
# ============================================================
def _fmt_minute(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M")


def _fmt_second(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def _build_timeline(rng: Random, shipped_at: datetime, origin: str, origin_district: str,
                    hub: str, dest: str, dest_district: str,
                    delivered: bool) -> tuple[list[dict], datetime]:
    """按时间线生成物流轨迹（与种子同款事件文案/时间粒度）。

    返回 (events, 最后一个事件时刻)——签收时刻直接复用，避免再解析字符串。
    """
    events = [{
        "time": _fmt_minute(shipped_at),
        "location": f"{origin}{origin_district}",
        "description": "快件已揽收",
    }]
    moment = shipped_at + timedelta(hours=rng.randrange(3, 9))
    events.append({
        "time": _fmt_minute(moment),
        "location": f"{origin}转运中心",
        "description": "已到达",
    })
    moment += timedelta(hours=rng.randrange(5, 15))
    events.append({
        "time": _fmt_minute(moment),
        "location": f"{hub}转运中心",
        "description": "已发出",
    })
    moment += timedelta(hours=rng.randrange(4, 12))
    events.append({
        "time": _fmt_minute(moment),
        "location": f"{dest}转运中心",
        "description": "已到达",
    })
    moment += timedelta(hours=rng.randrange(2, 7))
    events.append({
        "time": _fmt_minute(moment),
        "location": f"{dest}{dest_district}",
        "description": "正在派送中",
    })
    if delivered:
        moment += timedelta(hours=rng.randrange(1, 7))
        events.append({
            "time": _fmt_minute(moment),
            "location": f"{dest}{dest_district}",
            "description": "已签收",
        })
    return events, moment


def generate_orders_and_logistics(products: dict, seed: int = DEFAULT_SEED) -> tuple[dict, dict]:
    """生成新增订单与配套物流轨迹（确定性；u1–u5 不新增订单，保证既有口径不变）。"""
    rng = Random(f"{seed}:orders")
    product_ids = sorted(products)
    orders: dict[str, dict] = {}
    logistics: dict[str, dict] = {}
    seq = ORDER_SEQ_BASE
    for user_index, (user_id, user_name) in enumerate(EXTRA_USERS):
        for index in range(ORDERS_PER_USER):
            status = STATUS_PATTERN[index % len(STATUS_PATTERN)]
            # mock 时间戳与种子数据一致：本地 naive 时间字符串（无时区语义）
            created = (
                datetime(2024, 6, 1)  # noqa: DTZ001
                + timedelta(
                    days=rng.randrange(0, 540),
                    hours=rng.randrange(8, 23),
                    minutes=rng.choice((0, 5, 15, 20, 30, 40, 45, 50)),
                )
            )
            seq += 1
            order_id = f"ORD-{created:%Y%m%d}-{seq:04d}"

            item_count = rng.choice((1, 1, 1, 2, 2, 3))
            items = []
            for product_id in sorted(rng.sample(product_ids, item_count)):
                product = products[product_id]
                items.append({
                    "name": product["name"],
                    "sku": product["product_id"],
                    "quantity": rng.choice((1, 1, 1, 2)),
                    "price": product["price"],
                })
            total = round(sum(item["price"] * item["quantity"] for item in items), 2)

            order = {
                "order_id": order_id,
                "user": user_name,
                "user_id": user_id,
                "status": status,
                "items": items,
                "total": total,
                "created_at": _fmt_second(created),
                "shipped_at": None,
                "tracking_number": None,
                "carrier": None,
                "estimated_delivery": None,
            }

            if status in SHIPPED_STATUSES:
                shipped_at = created + timedelta(
                    hours=rng.randrange(4, 40), minutes=rng.choice((0, 10, 25, 40, 55)),
                )
                carrier_name, carrier_code = CARRIERS[rng.randrange(len(CARRIERS))]
                tracking_number = f"{carrier_code}{TRACKING_BASE + seq}"
                origin_idx = rng.randrange(len(CITIES))
                dest_idx = (origin_idx + 1 + rng.randrange(len(CITIES) - 1)) % len(CITIES)
                delivered = status in ("delivered", "refund_processing")
                events, last_moment = _build_timeline(
                    rng, shipped_at,
                    CITIES[origin_idx], DISTRICTS[origin_idx],
                    CITIES[(origin_idx + 3) % len(CITIES)],
                    CITIES[dest_idx], DISTRICTS[dest_idx],
                    delivered=delivered,
                )
                order["shipped_at"] = _fmt_second(shipped_at)
                order["tracking_number"] = tracking_number
                order["carrier"] = carrier_name
                order["estimated_delivery"] = (
                    shipped_at + timedelta(days=rng.randrange(2, 6))
                ).strftime("%Y-%m-%d")
                logistics[tracking_number] = {
                    "tracking_number": tracking_number,
                    "carrier": carrier_name,
                    "status": "delivered" if delivered else "in_transit",
                    "events": events,
                }
                if delivered:
                    order["delivered_at"] = _fmt_second(last_moment)
                    if status == "refund_processing":
                        order["refund_reason"] = REFUND_REASONS[rng.randrange(len(REFUND_REASONS))]
                        order["refund_status"] = "审核中"
                        order["refund_requested_at"] = _fmt_second(
                            last_moment + timedelta(hours=rng.randrange(2, 48))
                        )
            elif status == "cancelled":
                order["cancelled_at"] = _fmt_second(
                    created + timedelta(hours=rng.randrange(1, 20))
                )
                order["cancel_reason"] = CANCEL_REASONS[rng.randrange(len(CANCEL_REASONS))]

            orders[order_id] = order
    return orders, logistics


# ============================================================
# 数据集装配 / 落盘
# ============================================================
def assert_seed_preserved(dataset: dict) -> None:
    """生成结果必须逐条包含且不改写内置种子（漂移即失败，不给静默机会）。"""
    for section, seeds in (
        ("orders", SEED_ORDERS), ("products", SEED_PRODUCTS), ("logistics", SEED_LOGISTICS),
    ):
        generated = dataset.get(section) or {}
        for key, seed_value in seeds.items():
            if key not in generated:
                raise AssertionError(f"扩容数据集丢失种子条目: {section}/{key}")
            if generated[key] != seed_value:
                raise AssertionError(f"扩容数据集改写了种子条目: {section}/{key}")


def generate_dataset(seed: int = DEFAULT_SEED) -> dict:
    """种子 + 新增 → 完整数据集（确定性，逐字节可复现）。"""
    products = generate_products(seed)
    orders, logistics = generate_orders_and_logistics(products, seed)
    dataset = {
        "meta": {
            "generator": "app/scripts/seed_commerce_data.py",
            "version": GENERATOR_VERSION,
            "seed": seed,
            "orders_per_user": ORDERS_PER_USER,
        },
        # 种子在前：保证文件顺序与「扩容前」一致（追加不改写）
        "orders": {**SEED_ORDERS, **orders},
        "products": {**SEED_PRODUCTS, **products},
        "logistics": {**SEED_LOGISTICS, **logistics},
    }
    assert_seed_preserved(dataset)
    return dataset


def dataset_json(dataset: dict) -> str:
    """数据集 → 稳定 JSON 文本（LF、两空格缩进、不排序键、UTF-8 原样中文）。"""
    return json.dumps(dataset, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def write_dataset(dataset: dict, path: str | Path | None = None) -> Path:
    """原子写数据集（同输入两次写出逐字节一致）。"""
    target = Path(path) if path is not None else GENERATED_DATA_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(dataset_json(dataset), encoding="utf-8", newline="\n")
    tmp.replace(target)
    return target


def write_docs(dataset: dict, docs_dir: str | Path | None = None) -> int:
    """渲染商品知识库文档（独立命名空间；见 app/agent/rag/product_kb.py）。"""
    from app.agent.rag.product_kb import write_product_docs

    return write_product_docs(dataset["products"], docs_dir)


def dataset_stats(dataset: dict) -> dict:
    """规模统计（报告/门禁用）。"""
    orders = dataset["orders"]
    statuses: dict[str, int] = {}
    for order in orders.values():
        statuses[order["status"]] = statuses.get(order["status"], 0) + 1
    categories: dict[str, int] = {}
    for product in dataset["products"].values():
        categories[product["category"]] = categories.get(product["category"], 0) + 1
    return {
        "orders": len(orders),
        "products": len(dataset["products"]),
        "logistics": len(dataset["logistics"]),
        "users": len({o["user_id"] for o in orders.values()}),
        "statuses": dict(sorted(statuses.items())),
        "categories": dict(sorted(categories.items())),
    }


# ============================================================
# CLI
# ============================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="P2-1 确定性 mock 商城数据生成器（追加不改写）",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    parser.add_argument("--out", default=str(GENERATED_DATA_PATH), help="数据集输出路径")
    parser.add_argument("--docs-dir", default="", help="商品文档输出目录（默认 product_docs/）")
    parser.add_argument("--no-docs", action="store_true", help="只写数据集，不渲染商品文档")
    parser.add_argument("--check", action="store_true",
                        help="只做幂等校验（不写盘）；与现有产物不一致时退出码 1")
    parser.add_argument("--build-index", action="store_true",
                        help="另建商品向量索引（需 embedding 配置；失败不影响数据集）")
    args = parser.parse_args(argv)

    dataset = generate_dataset(args.seed)
    text = dataset_json(dataset)
    stats = dataset_stats(dataset)

    if args.check:
        target = Path(args.out)
        current = target.read_text(encoding="utf-8") if target.exists() else None
        if current != text:
            print(f"❌ 数据集与重新生成结果不一致（seed={args.seed}）: {target}")
            return 1
        if not args.no_docs:
            from app.agent.rag.product_kb import product_docs_dir as _docs_dir
            from app.agent.rag.product_kb import render_product_doc

            docs_dir = Path(args.docs_dir) if args.docs_dir else _docs_dir()
            expected = {
                f"{pid}.md": render_product_doc(product)
                for pid, product in dataset["products"].items()
            }
            for name, content in sorted(expected.items()):
                path = docs_dir / name
                if not path.exists() or path.read_text(encoding="utf-8") != content:
                    print(f"❌ 商品文档与重新生成结果不一致: {path}")
                    return 1
            print(f"✅ 幂等校验通过（seed={args.seed}，{len(expected)} 份商品文档）")
        else:
            print(f"✅ 幂等校验通过（seed={args.seed}）")
        return 0

    path = write_dataset(dataset, args.out)
    docs_written = 0
    if not args.no_docs:
        docs_written = write_docs(dataset, args.docs_dir or None)
    print(f"✅ 数据集已写入 {path}")
    print(f"   订单 {stats['orders']} / 商品 {stats['products']} / "
          f"物流 {stats['logistics']} / 用户 {stats['users']}")
    print(f"   状态分布 {stats['statuses']}")
    if not args.no_docs:
        print(f"   商品文档 {docs_written} 份（独立命名空间 product）")

    if args.build_index:
        try:
            from app.agent.rag.embedder import create_embedder
            from app.agent.rag.product_kb import build_product_index

            count = build_product_index(dataset["products"], create_embedder())
            print(f"✅ 商品向量索引已构建（{count} chunks，独立于政策库索引）")
        except Exception as e:  # noqa: BLE001 —— 索引失败不阻塞数据生成
            print(f"⚠️  商品向量索引未构建（{type(e).__name__}: {e}）；"
                  "商品检索将走词法回落")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
