"""评估数据集：测试用例的数据结构与加载（第9期）。

一条 EvalCase 描述「输入是什么」+「期望表现是什么」。期望分两类：
- 结果期望：末轮回复应识别的意图、应命中的关键词、是否该转人工。
- 过程期望：应调用哪些工具、理论最少调用次数、token 预算、（多 Agent）期望路由。

留空的期望项在评分时会被跳过（不计分），而不是判 0，避免无工具/无关键词的
用例（如问候、投诉）拖垮聚合均值。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalCase:
    """单条评估用例。turns 支持多轮，按顺序喂给同一个 agent 实例。"""

    id: str
    description: str
    turns: list[str]  # 多轮输入，单轮即 len==1

    # ---------- 结果期望 ----------
    expected_intent: str | None = None  # 末轮期望意图（IntentType.value），None=不校验
    expected_keywords: list[str] = field(default_factory=list)  # 末轮 reply 应命中的关键词
    expected_requires_human: bool | None = None  # 是否应转人工，None=不校验

    # ---------- 过程期望 ----------
    expected_tools: list[str] = field(default_factory=list)  # 整个会话应调用过的工具
    min_tool_calls: int | None = None  # 理论最少工具调用次数（算效率用），None=不校验
    max_tokens: int | None = None  # token 预算上限，None=不设上限
    expected_route: str | None = None  # 多 Agent 期望路由（presale/postsale/complaint），可选

    # ---------- 引用真实性（改造三）----------
    expected_citations: list[str] = field(default_factory=list)  # 应命中的引用（原文或规范化值）
    forbid_unretrieved_citations: bool = False  # 出现未命中引用 → 直接 0 分

    # ---------- 安全与身份（2.2）----------
    actor_user_id: str | None = None  # 构建 Agent 时使用的身份（默认 u1）；None=默认身份
    expected_tool_outcomes: list[dict] = field(default_factory=list)
    # 结构化工具结果期望：[{"tool": "query_order", "args": {"order_id": "..."},
    #   "outcome": {"code": "ORDER_ACCESS_DENIED"}}]——args 为子集匹配（都在实参中即可）；
    # outcome 匹配 code（若给出）与 success 布尔（若给出）。
    forbidden_reply_terms: list[str] = field(default_factory=list)
    # 回复不得出现的敏感词（他人订单金额/商品/物流信息）；出现任一 → 泄露
    critical: bool = False  # 硬门禁：安全维度未达成时整条直接失败，不参与平滑评分


def load_dataset(path: str | Path) -> list[EvalCase]:
    """从 JSON 文件加载用例列表，文件格式为 {"cases": [ {EvalCase 字段}, ... ]}。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalCase(**item) for item in data["cases"]]
