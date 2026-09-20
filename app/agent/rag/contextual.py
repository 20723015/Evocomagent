"""构建期上下文增强（P1-1：Contextual Retrieval，Anthropic 2024.09）。

问题：小块检索时块本身缺乏「它在整篇文档里回答什么」的定位信息——本库语料
均值 175 字，机械前缀（【文档 · 标题路径】）是最简版上下文，块间区分度低。

方案：构建期对每块生成 1~2 句定位上下文 → 只进入 embedding/BM25 输入
（``Chunk.index_text``），块文本与证据文本保持原文精确切片（可校验、可引用，
生成内容绝不进证据）。LLM 失败/超时 → 该块回退机械前缀（index_text 留空）
并计入降级率，绝不阻断构建。

缓存正本：key = sha256(索引文本 + 模型 + prompt 版本)——未变化块零成本重建
（与增量索引共用 content-hash 设施）。改 prompt 必须 bump
``rag_contextual_prompt_version``（已进配置指纹），否则旧缓存会被静默复用。

成本量级（实测口径）：782 块 × ~600 tokens 输入 / 60 输出 ≈ 50 万 tokens，
gpt-4o-mini 级 < ¥1/次全量构建；命中缓存后为 0。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.observability.logging import get_logger

log = get_logger("app.agent.rag.contextual")

# 生成 prompt 的正本版本：改 system/user 模板必须 bump（进配置指纹）
PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "你是知识库索引助手。给定文档名、章节标题路径与一段正文，写出 1~2 句"
    "定位上下文，说明这段内容属于哪份规则的哪个主题、能回答什么问题，"
    "以便检索时区分它与其他片段。要求：只描述定位信息，不新增正文没有的"
    "事实、不写编号、不加引号、不用 Markdown，直接输出这段文字。"
)

USER_TEMPLATE = (
    "文档名：{doc}\n"
    "章节路径：{heading_path}\n"
    "正文：\n{body}\n\n"
    "请输出不超过 {max_chars} 字的定位上下文。"
)

# 生成结果清洗：去掉可能被模型包上的引号/前缀/Markdown 标记
_QUOTE_RE = re.compile(r'^[\s"\'“”『「]+|[\s"\'“”』」]+$')
_PREFIX_RE = re.compile(r"^(定位上下文|上下文|context)\s*[:：]\s*", re.IGNORECASE)


def context_key(index_text: str, model: str, prompt_version: str = PROMPT_VERSION) -> str:
    """缓存键：内容哈希 + 模型 + prompt 版本（任一变化即失效重算）。"""
    raw = f"{prompt_version}\x00{model}\x00{index_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_context_messages(
    *, doc: str, heading_path: str, body: str, max_chars: int,
) -> list[dict]:
    """定位上下文生成的消息体（prompt 模板正本，改此处必须 bump 版本）。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                doc=doc, heading_path=heading_path or doc,
                body=body[:2000], max_chars=max_chars,
            ),
        },
    ]


def clean_context(raw: str, max_chars: int) -> str:
    """清洗生成结果：去引号/前缀、折叠空白、按上限截断（超限在句读处收口）。"""
    text = " ".join(str(raw or "").split())
    text = _PREFIX_RE.sub("", text)
    text = _QUOTE_RE.sub("", text).strip()
    if len(text) <= max_chars:
        return text
    cut = max(
        (text.rfind(ch, 0, max_chars + 1) for ch in "。；，、,;."),
        default=-1,
    )
    return text[: cut + 1] if cut > 0 else text[:max_chars]


class ContextCache:
    """content-hash 上下文缓存（JSON 正本；缺失/损坏即视为空缓存）。"""

    def __init__(self, path):
        self._path = Path(path)
        self._data: dict[str, str] | None = None

    def _load(self) -> dict[str, str]:
        if self._data is None:
            try:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            self._data = (
                {str(k): str(v) for k, v in payload.items()}
                if isinstance(payload, dict) else {}
            )
        return self._data

    def get(self, key: str) -> str | None:
        value = self._load().get(key)
        return value or None

    def put(self, key: str, value: str) -> None:
        self._load()[key] = value

    def save(self) -> None:
        """落盘缓存（失败仅告警：缓存是加速设施，不是构建正确性的一部分）。"""
        if self._data is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=0, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as e:  # noqa: BLE001 —— 缓存写失败不影响索引正确性
            log.warning(f"上下文缓存落盘失败: {type(e).__name__}")


@dataclass
class EnrichReport:
    """构建期增强报告（生成/命中/降级计数，进构建日志与发布记录）。"""

    total: int = 0
    generated: int = 0
    cached: int = 0
    failed: int = 0  # 回退机械前缀（LLM 失败/空输出）的块数
    cache_hits: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "generated": self.generated,
            "cached": self.cached,
            "failed": self.failed,
            "cache_hits": self.cache_hits,
            "degrade_ratio": (
                round(self.failed / self.total, 4) if self.total else 0.0
            ),
        }


class ContextualEnricher:
    """给块生成定位上下文并写入 ``index_text``（只影响检索输入）。

    complete 为注入的 chat 调用（``complete(messages, max_tokens) -> str``）——
    生产用 :func:`create_openai_completer`，测试直接注入假函数（离线可测）。
    """

    def __init__(
        self,
        complete,
        *,
        model: str,
        prompt_version: str = PROMPT_VERSION,
        max_chars: int = 80,
        cache: ContextCache | None = None,
        max_context_tokens: int = 200,
    ):
        self._complete = complete
        self._model = model
        self._prompt_version = prompt_version
        self._max_chars = int(max_chars)
        self._cache = cache
        self._max_tokens = int(max_context_tokens)

    @property
    def model(self) -> str:
        return self._model

    def enrich(self, chunks) -> EnrichReport:
        """为每个块补齐 index_text 中的生成上下文；返回计数报告。

        - 已有 index_text 的块（如 evolved 沉淀的问题文本）直接复用，不调 LLM；
        - 缓存命中不计入 LLM 调用；
        - 单块失败 → 回退（该块 index_text 保持不变），累计 failed 后继续。
        """
        report = EnrichReport(total=len(chunks))
        changed = False
        pending: list[tuple] = []
        for chunk in chunks:
            if chunk.index_text:
                report.cached += 1
                continue
            key = context_key(chunk.text, self._model, self._prompt_version)
            cached = self._cache.get(key) if self._cache is not None else None
            if cached:
                chunk.index_text = f"{cached}\n{chunk.text}"
                report.cache_hits += 1
                report.cached += 1
                changed = True
                continue
            pending.append((chunk, key))

        for chunk, key in pending:
            try:
                context = self._generate(chunk)
            except Exception as e:  # noqa: BLE001 —— 单块失败不阻断构建
                report.failed += 1
                if len(report.errors) < 5:
                    report.errors.append(f"{chunk.chunk_id}: {type(e).__name__}")
                continue
            if not context:
                report.failed += 1
                continue
            chunk.index_text = f"{context}\n{chunk.text}"
            report.generated += 1
            changed = True
            if self._cache is not None:
                self._cache.put(key, context)

        if changed and self._cache is not None:
            self._cache.save()
        if report.failed:
            log.warning(
                f"上下文增强降级 {report.failed}/{report.total} 块"
                f"（回退机械前缀）: {'; '.join(report.errors)}"
            )
        return report

    def _generate(self, chunk) -> str:
        messages = build_context_messages(
            doc=chunk.doc, heading_path=chunk.heading_path,
            body=chunk.text, max_chars=self._max_chars,
        )
        return clean_context(
            self._complete(messages, self._max_tokens), self._max_chars
        )


def create_openai_completer(client, model: str, *, timeout: float = 60.0):
    """OpenAI 兼容客户端 → ``complete(messages, max_tokens) -> str``。

    经 ``install_resilience`` 包装（重试/降级链/并发/超时一致）；模型不可用或
    调用失败由 ``ContextualEnricher`` 按块降级，不在此吞异常。
    """

    def complete(messages: list[dict], max_tokens: int) -> str:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.0, max_tokens=max_tokens,
            timeout=timeout,
        )
        return resp.choices[0].message.content or ""

    return complete


def create_enricher_from_settings() -> ContextualEnricher:
    """按 settings 构建增强器（构建链路默认入口；OpenAI 客户端延迟创建）。

    rag_contextual_index 未开启时调用方不应走到这里；模型取
    rag_contextual_model，空则回退 settings.model_name。
    """
    from openai import OpenAI

    from app.config.settings import settings
    from app.llm.client import install_resilience

    model = settings.rag_contextual_model or settings.model_name
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    install_resilience(client, model)
    cache = (
        ContextCache(settings.rag_contextual_cache_path)
        if settings.rag_contextual_cache_path else None
    )
    return ContextualEnricher(
        create_openai_completer(client, model, timeout=settings.llm_timeout_seconds),
        model=model,
        prompt_version=settings.rag_contextual_prompt_version,
        max_chars=settings.rag_contextual_max_chars,
        cache=cache,
    )
