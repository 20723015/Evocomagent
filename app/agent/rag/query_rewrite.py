"""查询改写（P1-3）：口语化/省略主语的 query → 检索友好的规范 query。

动机：线上 query 大量来自用户口语（「这个能退吗」「会员是不是免运费」），
省略主语/指代不明/缺少领域词，向量与 BM25 都拿不到应有的召回。查询改写把
这类问题补全为自足的检索 query，作为**第二路**与原 query 并行召回——原 query
恒居首（回退安全），改写路只是补充召回，两路经既有的
``retriever_factory.final_multi_search``（逐路门控 → RRF 合并 → 父块折叠 →
Top-K）合流，不另起一套管线。

热路径纪律（fail-open，绝不抛进检索）：
- 未配置（无 API key）/ 超时 / 调用异常 / 输出为空 / 输出与原 query 等价 /
  输出不可信（超长、多行）→ 返回 None，调用方原样透传；
- 改写失败率与延迟增量进 ``app.observability.metrics``（append-only）。

预算归属与超时：
- 经 ``app.llm.client.install_resilience`` 包装：轮次预算（current_budget 的
  deadline 夹逼、LLM_BUDGET_EXHAUSTED、token 记账/成本指标）与并发信号量
  自动接入，与其余 LLM 调用同一治理面；
- 单次调用显式 timeout（默认 ``QUERY_REWRITE_TIMEOUT_SECONDS``，再与调用方
  剩余预算、settings.llm_timeout_seconds 取小）——热路径不为改写多等；
- system prompt 首句含「提取结构化」使 ``infer_purpose`` 将本次调用归入
  extract 用途（与提取/摘要同类辅助调用：走 extraction_model、指标归属正确，
  不把改写成本记到 react 主链路上）。
"""

from __future__ import annotations

import re
import time

from app.config.settings import settings
from app.observability.logging import get_logger

log = get_logger("app.agent.rag.query_rewrite")

# 热路径改写预算：单次调用最多等这么久（再与调用方 timeout / llm_timeout 取小）。
# 12s 是实测折中：当前 .env 主模型是推理模型（思考 token 与可见输出共享上限，
# 实测 p75≈12s）；生产应把 extraction_model 配成快速非推理模型，届时远低于此。
QUERY_REWRITE_TIMEOUT_SECONDS = 12.0
# 改写输出上限：query 改写本身只需几十 token，但推理模型的思考 token 与可见
# 输出共享该上限（128/512 会把可见输出吃光 → 空输出，实测 1024 起稳定有输出）。
QUERY_REWRITE_MAX_TOKENS = 1024
# 输出可信度上限：超过该长度视为模型跑偏（原 query 的 4 倍且不少于 64 字）
_MAX_OUTPUT_CHARS = 200

# 首句含「提取结构化」→ llm.client.infer_purpose 归为 extract（辅助调用），
# 与 extraction_model 路由/指标口径一致；不是给模型看的魔数，是治理入口约定。
_SYSTEM_PROMPT = (
    "你是检索查询改写器：从用户口语化问题中提取结构化检索查询。"
    "补全省略的主语、指代与必要背景词，保留全部关键实体"
    "（商品/业务名、金额、时间、订单号等），不回答、不解释、不添加"
    "原文没有的事实。只输出改写后的一行查询本身，不要任何说明、前缀或"
    "思考过程；原问题已自足时原样输出。"
)
_PREFIX_RE = re.compile(r"^\s*(?:改写后?|查询|检索查询|rewrite)\s*[:：]\s*", re.I)
# 模型把思考/指令复述写进可见输出（如「根据指令，原问题已自足，原样输出。X」）：
# 这类元话语一律判为未改写（fail-open），绝不把说明文字当检索 query。
_META_HINTS = (
    "原样输出", "已自足", "无需改写", "不需要改写", "保持不变",
    "根据指令", "无法改写", "作为ai", "作为一个ai",
)


def _clean_output(raw: str) -> str:
    """模型输出 → 单行候选 query；空/多行噪声时取首个非空行。"""
    text = (raw or "").strip()
    if not text:
        return ""
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    line = _PREFIX_RE.sub("", line).strip().strip("`\"'“”‘’").strip()
    low = line.casefold()
    if any(hint in low for hint in _META_HINTS):
        return ""
    return line


def _plausible(original: str, candidate: str) -> bool:
    """改写输出可信度护栏：不空、不超长（防模型跑偏成回答）。"""
    if not candidate or candidate == original:
        return False
    if len(candidate) > _MAX_OUTPUT_CHARS:
        return False
    if len(candidate) > max(64, len(original) * 4):
        return False
    return True


class QueryRewriter:
    """单次 LLM 调用的 query 改写器（无状态；失败一律返回 None）。"""

    def __init__(self, client, model: str, *,
                 timeout: float = QUERY_REWRITE_TIMEOUT_SECONDS,
                 max_tokens: int = QUERY_REWRITE_MAX_TOKENS):
        self._client = client
        self._model = model
        self._timeout = float(timeout)
        self._max_tokens = int(max_tokens)

    def rewrite(self, query: str, *, timeout: float | None = None) -> str | None:
        """改写 query；None = 未改写（调用方原样透传，绝不抛异常）。"""
        from app.observability.metrics import record_query_rewrite

        text = (query or "").strip()
        if not text:
            record_query_rewrite("skipped", reason="empty_query")
            return None
        attempt_timeout = self._timeout
        if timeout is not None:
            attempt_timeout = min(attempt_timeout, max(float(timeout), 0.0))
        if attempt_timeout <= 0:
            record_query_rewrite("skipped", reason="no_budget")
            return None

        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"原查询：{text}"},
                ],
                max_tokens=self._max_tokens,
                timeout=attempt_timeout,
            )
            raw = response.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 —— 热路径 fail-open：任何失败都透传
            record_query_rewrite(
                "error",
                reason=type(e).__name__,
                latency_seconds=time.perf_counter() - start,
            )
            return None

        latency = time.perf_counter() - start
        candidate = _clean_output(raw)
        if not candidate:
            record_query_rewrite("empty", reason="empty_output", latency_seconds=latency)
            return None
        if candidate == text:
            record_query_rewrite(
                "unchanged", reason="identical", latency_seconds=latency,
            )
            return None
        if not _plausible(text, candidate):
            record_query_rewrite(
                "error", reason="implausible_output", latency_seconds=latency,
            )
            return None
        record_query_rewrite("ok", reason="rewritten", latency_seconds=latency)
        return candidate


def create_query_rewriter() -> QueryRewriter | None:
    """按 settings 构建改写器；未配置 API key → None（零网络调用，fail-open）。

    模型取 ``extraction_model``（空则回落 ``settings.model_name``）；客户端经
    ``install_resilience`` 包装（重试/降级/并发/轮次预算/token 记账）。
    """
    if not (settings.openai_api_key or "").strip():
        return None
    from openai import OpenAI

    from app.llm.client import install_resilience

    model = settings.extraction_model or settings.model_name
    # max_retries=0 双保险：OpenAI SDK 自带重试（默认 2）与 ResilientLLM 重试
    # 都会把最坏墙钟乘 3——热路径改写只允许单次尝试，失败即 fail-open。
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=QUERY_REWRITE_TIMEOUT_SECONDS,
        max_retries=0,
    )
    install_resilience(
        client, model,
        timeout_seconds=QUERY_REWRITE_TIMEOUT_SECONDS,
        max_retries=0,
    )
    max_tokens = min(
        int(settings.llm_max_tokens or QUERY_REWRITE_MAX_TOKENS),
        QUERY_REWRITE_MAX_TOKENS,
    )
    return QueryRewriter(client, model, max_tokens=max_tokens)


_rewriter: QueryRewriter | None = None
_rewriter_key: tuple[str, str] | None = None


def get_query_rewriter() -> QueryRewriter | None:
    """进程级惰性单例；key/model 变化自动重建（settings 热改不残留旧客户端）。"""
    global _rewriter, _rewriter_key
    key = (
        (settings.openai_api_key or "").strip(),
        settings.extraction_model or settings.model_name,
    )
    if _rewriter is None or key != _rewriter_key:
        _rewriter = create_query_rewriter()
        _rewriter_key = key
    return _rewriter


def reset_query_rewriter() -> None:
    """清空单例（测试/配置切换用）。"""
    global _rewriter, _rewriter_key
    _rewriter = None
    _rewriter_key = None


def rewrite_query(query: str, *, timeout: float | None = None,
                  rewriter: QueryRewriter | None = None) -> str | None:
    """便捷入口：改写成功返回新 query，其余（含未配置）返回 None。

    单例装配失败也 fail-open（打点后返回 None），绝不冒泡进检索热路径。
    """
    try:
        active = rewriter if rewriter is not None else get_query_rewriter()
    except Exception as e:  # noqa: BLE001 —— 装配失败同样透传
        from app.observability.metrics import record_query_rewrite

        record_query_rewrite("error", reason=type(e).__name__)
        return None
    if active is None:
        return None
    return active.rewrite(query, timeout=timeout)


