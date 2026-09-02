"""工具结果结构化精简（digest.py + extraction/summarizer 接线）无网络单测。

样本 JSON 逐一取自真实工具返回（docs/工具结果结构化精简计划.md 已核实）：
query_order 嵌套在 order 下；query_product 字段在 products[] 内；物流字段名
carrier；list_user_orders 项含中文 status；apply_refund 两段式 status=pending_confirmation。
"""

from __future__ import annotations

import json

import pytest

from app.agent.tools import digest as d
from app.agent.tools.digest import (
    digest_tool_result,
    render_tool_result_line,
    tool_call_name_map,
)
from app.config.settings import settings


# ============================================================
# 真实形状样本
# ============================================================
ORDER_JSON = json.dumps({
    "success": True,
    "order": {
        "order_id": "ORD-20240115-001", "user": "小明", "user_id": "u1",
        "status": "shipped",
        "items": [{"name": "Nike Air Max 270 运动鞋", "sku": "SHOE-270-BK-42",
                   "quantity": 1, "price": 899.0}],
        "total": 899.0, "created_at": "2024-01-15 10:30:00",
        "shipped_at": "2024-01-16 14:20:00", "tracking_number": "SF1234567890",
        "carrier": "顺丰速运", "estimated_delivery": "2024-01-19",
    },
}, ensure_ascii=False)

PRODUCT_JSON = json.dumps({
    "success": True,
    "products": [{
        "product_id": "ELEC-APP-002", "name": "Apple AirPods Pro 2",
        "category": "耳机", "price": 1799.0, "stock": 89,
        "description": "主动降噪，自适应透明模式，个性化空间音频，USB-C 充电",
        "specs": {"颜色": "白色", "连接方式": "蓝牙5.3", "续航": "6小时"},
    }],
}, ensure_ascii=False)

LOGISTICS_JSON = json.dumps({
    "success": True,
    "logistics": {
        "tracking_number": "SF1234567890", "carrier": "顺丰速运",
        "status": "in_transit",
        "events": [
            {"time": "2024-01-16 14:20", "location": "深圳南山区", "description": "快件已揽收"},
            {"time": "2024-01-17 06:00", "location": "广州转运中心", "description": "已到达"},
            {"time": "2024-01-18 08:30", "location": "上海浦东区", "description": "正在派送中"},
        ],
    },
}, ensure_ascii=False)

USER_ORDERS_JSON = json.dumps({
    "success": True, "count": 12,
    "orders": [
        {"order_id": f"ORD-2024{i:05d}", "status": "已发货",
         "items_summary": f"商品批次 {i} 号（很长很长很长的商品摘要文本用于测试截断行为）",
         "total": 100.0 + i, "created_at": "2024-01-15 10:30:00"}
        for i in range(12)
    ],
}, ensure_ascii=False)

REFUND_PENDING_JSON = json.dumps({
    "success": True, "status": "pending_confirmation",
    "refund_id": "16cde6619b5046228cab61100a68af59",
    "confirmation_token": "ec9b33049da3468bae071ba9ccb90e7b",
    "expires_in_seconds": 300,
    "idempotency_key": "16cde6619b5046228cab61100a68af59",
}, ensure_ascii=False)

KNOWLEDGE_JSON = json.dumps({
    "success": True, "backend": "numpy", "query": "退货政策",
    "results": [
        {"doc": "退换货政策.md", "section": "七天无理由", "score": 0.9,
         "text": "支持七天无理由退货……", "source_path": "知识库/退换货政策.md"},
        {"doc": "退换货政策.md", "section": "运费", "score": 0.8, "text": "……"},
        {"doc": "配送说明.md", "section": "时效", "score": 0.7, "text": "……"},
        {"doc": "会员权益.md", "section": "积分", "score": 0.6, "text": "……"},
        {"doc": "常见问题FAQ.md", "section": "FAQ", "score": 0.5, "text": "……"},
        {"doc": "第六篇文档.md", "score": 0.4, "text": "……"},
    ],
}, ensure_ascii=False)


# ============================================================
# 1. 逐工具投影正确性（保留字段 / 丢弃字段）
# ============================================================
def test_project_order_keeps_real_fields():
    got = digest_tool_result("query_order", ORDER_JSON)
    assert "order_id=ORD-20240115-001" in got
    assert "status=shipped" in got
    assert "total=899.0" in got
    assert "tracking_number=SF1234567890" in got
    assert "商品=Nike Air Max 270 运动鞋" in got
    # 丢弃：用户身份、时间戳、承运人等无关字段
    assert "user_id" not in got
    assert "created_at" not in got
    # 嵌套 order 外字段不出现
    assert '"order"' not in got


def test_project_product_uses_products_list():
    got = digest_tool_result("query_product", PRODUCT_JSON)
    # ⚠ 字段在 products[] 内（真实结构），不是顶层标量
    assert "商品=Apple AirPods Pro 2" in got
    assert "price=1799.0" in got
    assert "stock=89" in got
    assert "product_id=ELEC-APP-002" in got
    assert "description" not in got and "specs" not in got

    many = json.dumps({"success": True, "products": [
        {"product_id": f"P{i}", "name": f"商品{i}", "price": i, "stock": i}
        for i in range(5)
    ]}, ensure_ascii=False)
    got_many = digest_tool_result("query_product", many)
    assert "…另有3项" in got_many  # 5 - 2 = 3


def test_project_logistics_uses_carrier_and_last_event():
    got = digest_tool_result("query_logistics", LOGISTICS_JSON)
    assert "carrier=顺丰速运" in got  # 真实字段名 carrier（不是 courier）
    assert "status=in_transit" in got
    assert "tracking_number=SF1234567890" in got
    assert "最新=2024-01-18 08:30" in got and "正在派送中" in got
    # 最早事件被丢弃（旧版前 200 字恰好是它）
    assert "快件已揽收" not in got


def test_project_refund_states():
    got = digest_tool_result("apply_refund", REFUND_PENDING_JSON)
    assert "success=True" in got
    assert "status=pending_confirmation" in got
    # 一次性凭证/幂等锚点不进转录层
    assert "confirmation_token" not in got and "refund_id" not in got

    ok = json.dumps({"success": True, "message": "退款申请已提交。请等待审核。"},
                    ensure_ascii=False)
    assert "message=" in digest_tool_result("apply_refund", ok)

    bad = json.dumps({"success": False, "error": "未找到订单 ORD-X，请核实订单号"},
                     ensure_ascii=False)
    got_bad = digest_tool_result("apply_refund", bad)
    assert "success=False" in got_bad and "error=" in got_bad


def test_project_skill_success_and_failure():
    ok = json.dumps({"success": True, "skill_name": "track-order",
                     "instructions": "# 完整指令\n…很长"},
                    ensure_ascii=False)
    assert digest_tool_result("load_skill", ok) == "已加载技能: track-order（指令略）"

    bad = json.dumps({"success": False, "error": "技能系统未启用"}, ensure_ascii=False)
    got = digest_tool_result("load_skill", bad)
    # 失败分支走通用压缩（sort_keys JSON）：error 保留
    assert '"success": false' in got and '"error": "技能系统未启用"' in got


def test_project_recall_user_memory_keeps_raw():
    raw = json.dumps({"success": True, "query": "", "long_term_facts": [
        {"content": "偏好顺丰", "category": "preference"}
    ]}, ensure_ascii=False)
    assert digest_tool_result("recall_user_memory", raw) == raw


# ============================================================
# 2. 原始动机场景：相关订单在第 8 个（旧版 200 截断必被切掉）
# ============================================================
def test_list_user_orders_motivation_case_8th_item():
    got = digest_tool_result("list_user_orders", USER_ORDERS_JSON)
    assert "ORD-202400008" in got       # 第 8 个订单（7 号索引）不再被截断切掉
    assert "已发货" in got
    assert "…另有2单" in got             # 12 - 10
    # 旧版行为对比：前 200 字符必然不含第 8 单
    old = USER_ORDERS_JSON[:200] + "..."
    assert "ORD-202400008" not in old

    # 每项 ≤60：预算紧张时压缩到 order_id+status（摘要在高预算档才出现）
    for part in got.split("|"):
        part = part.strip()
        if part.startswith("ORD-"):
            assert len(part) <= 61  # 60 字 + 省略号
    assert len(got) <= 200


# ============================================================
# 3. 预算不回归：全部输出 ≤200（含最坏样本）
# ============================================================
@pytest.mark.parametrize("tool_name,content", [
    ("query_order", ORDER_JSON),
    ("query_product", PRODUCT_JSON),
    ("query_logistics", LOGISTICS_JSON),
    ("list_user_orders", USER_ORDERS_JSON),
    ("search_knowledge", KNOWLEDGE_JSON),
    ("apply_refund", REFUND_PENDING_JSON),
    ("load_skill", json.dumps({"success": True, "skill_name": "track-order",
                               "instructions": "x" * 500}, ensure_ascii=False)),
    ("recall_user_memory", json.dumps({"success": True, "long_term_facts": [
        {"content": "y" * 100, "category": "other"} for _ in range(20)]},
        ensure_ascii=False)),
    ("future_tool", json.dumps({"a": 1, "b": "x" * 300, "nested": {"z": 1}})),
])
def test_budget_never_exceeds(tool_name, content):
    assert len(digest_tool_result(tool_name, content)) <= 200


# ============================================================
# 4. 三层降级
# ============================================================
def test_digest_layer2_generic_compress_unknown_tool():
    raw = json.dumps({"a": 1, "b": "x" * 100, "nested": {"z": 1}, "flag": True,
                      "nothing": None}, ensure_ascii=False)
    got = digest_tool_result("future_tool", raw)
    # 顶层标量保留（str 截 40），嵌套对象/None 丢弃
    assert got.startswith('{"a": 1,')
    assert "z" not in got and "nothing" not in got
    # 确定性键序（sort_keys）
    assert json.loads(got) == {"a": 1, "b": "x" * 39 + "…", "flag": True}


def test_digest_layer3_non_json_prefix():
    assert digest_tool_result("query_order", "这不是JSON") == "这不是JSON"
    long_text = "z" * 300
    got = digest_tool_result("query_order", long_text)
    assert got == long_text[:200] + "..."


def test_digest_layer3_projector_exception_falls_back(monkeypatch):
    def boom(payload):
        raise RuntimeError("投影崩了")

    # digest_tool_result 从投影表取函数引用，需替换表内条目才能触发异常路径
    monkeypatch.setitem(d._PROJECTORS, "query_order", boom)
    got = digest_tool_result("query_order", ORDER_JSON)
    assert got == ORDER_JSON[:200] + "..."  # 任何内部异常落到第 3 层


# ============================================================
# 5. 双接线一致（同一输入 → extraction 与 summarize 相同 tool 行）
# ============================================================
def _messages_with_tool_call():
    return [
        {"role": "user", "content": "我的订单到哪了？"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "query_order",
                          "arguments": '{"order_id": "ORD-20240115-001"}'}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": ORDER_JSON},
    ]


def test_dual_wiring_same_tool_line():
    from app.agent.memory.extraction import _build_transcript
    from app.agent.summarizer import summarize
    from tests.unit.conftest import FakeChatClient

    messages = _messages_with_tool_call()

    transcript = _build_transcript(messages)
    extract_line = [l for l in transcript.splitlines() if l.startswith("[工具结果]")][0]
    assert extract_line.startswith("[工具结果] query_order ")

    client = FakeChatClient().enqueue_chat("摘要")
    summarize(client, model="m", old_messages=messages, prev_summary=None)
    user_content = client.calls[-1][1]["messages"][1]["content"]
    summarize_line = [l for l in user_content.splitlines() if l.startswith("[工具结果]")][0]
    # 两处共用同一渲染实现：逐字节相同（防再漂移）
    assert summarize_line == extract_line


def test_call_name_map_resolves_tool_name():
    messages = _messages_with_tool_call()
    assert tool_call_name_map(messages) == {"call_1": "query_order"}
    # 无 assistant 调用行时 tool 行以 ? 自证（仍自足渲染，不抛异常）
    assert render_tool_result_line(
        tool_call_name_map([messages[-1]]).get("call_1", "?"), ORDER_JSON,
    ).startswith("[工具结果] ? ")


# ============================================================
# 6. 确定性
# ============================================================
def test_digest_deterministic():
    assert digest_tool_result("query_order", ORDER_JSON) == digest_tool_result(
        "query_order", ORDER_JSON,
    )
    assert digest_tool_result("list_user_orders", USER_ORDERS_JSON) == digest_tool_result(
        "list_user_orders", USER_ORDERS_JSON,
    )


# ============================================================
# 7. 开关：回退旧版（golden 逐字节一致，无工具名）
# ============================================================
def test_digest_disabled_matches_legacy_golden(reset_settings):
    settings.tool_digest_enabled = False
    content = ORDER_JSON

    line = render_tool_result_line("query_order", content, budget=200)
    legacy_display = content if len(content) <= 200 else content[:200] + "..."
    assert line == f"[工具结果] {legacy_display}"  # 与旧版完全一致
    assert "query_order" not in line                # 旧格式无工具名

    # digest 函数本身在开关关闭时仍按投影路径（render 层统一控制开关）
    assert "order_id=" in digest_tool_result("query_order", content)


# ============================================================
# 8. knowledge 特例：只看 doc 名、去重、≤5
# ============================================================
def test_knowledge_digest_docs_only():
    got = digest_tool_result("search_knowledge", KNOWLEDGE_JSON)
    assert got.startswith("命中5篇: ")       # 6 篇中 5 个唯一 doc（退换货政策重复）
    assert got.count("退换货政策.md") == 1   # 去重
    assert "支持七天无理由退货" not in got   # 只取 doc 名，不碰 text/section/score

    # 超过 5 个唯一 doc → 尾部标注；无命中/失败结果 → 不抛异常
    many = {"success": True, "results": [
        {"doc": f"文档{i}.md", "text": "x"} for i in range(7)
    ]}
    got_many = digest_tool_result("search_knowledge", json.dumps(many, ensure_ascii=False))
    assert got_many.startswith("命中7篇: ") and "…等7篇" in got_many

    empty = json.dumps({"success": False, "error": "知识库未初始化", "results": []},
                       ensure_ascii=False)
    assert digest_tool_result("search_knowledge", empty) == "命中0篇"


def test_knowledge_tainted_field_does_not_interfere():
    payload = {
        "success": True,
        "results": [
            {"doc": "退换货政策.md", "tainted": False, "text": "A" * 300},
            {"doc": "污染文档.md", "tainted": True, "text": "B" * 300},
        ],
    }
    got = digest_tool_result("search_knowledge", json.dumps(payload, ensure_ascii=False))
    # 转录层只取 doc 名——与检索来源合法性（citations 层）职责分离
    assert "退换货政策.md" in got and "污染文档.md" in got
    assert "A" * 300 not in got and "B" * 300 not in got
