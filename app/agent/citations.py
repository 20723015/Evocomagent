"""引用来源真实性（Agent能力强化计划·改造三）。

命名口径：本能力证明**引用名称来自本轮召回集合**，不证明回答内容被该文档
支持（后者是演进管线接地 Judge 的领域）。

- 引用提取（宁漏勿误杀）：仅认两种——
  ① 引用语境中的《书名》：书名号前 6 字符内含「根据/依据/来源/参考/政策/文档」之一；
  ② 显式文件名（含 .md/.txt/.pdf/.docx/.doc/.html 后缀）。
  孤立书名号（商品名、书名）不触发。
- 来源归一：NFKC → basename → 去扩展名 → casefold；来源集合同时收
  doc 与 source_path 两路别名；tainted=true 的检索块不进合法来源集合
  （被污染的召回不能为引用背书；`kb_chunk_tainted` 是函数名不是字段名）。
- Verdict 三态：{cited, matched, missing} 全量保留——比较用规范化值，
  报告保留原文。
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

# 引用语境词：书名号前 6 字符内出现其一才视为引用
CITATION_CONTEXT_WORDS = ("根据", "依据", "来源", "参考", "政策", "文档")

_CITATION_FILE_EXT = (".md", ".txt", ".pdf", ".docx", ".doc", ".html")

# 显式文件名：路径字符（不空格）+ 扩展名；前后不得紧贴 ASCII 词字符
# （避免把「详见 退货政策.md」的汉语连词吞进文件名、或截到 x.md5 之类）
_FILENAME_RE = re.compile(
    r"(?<![A-Za-z0-9_\-./\\])"
    r"[A-Za-z0-9\u4e00-\u9fff_\-./\\]+"
    r"\.(?:md|txt|pdf|docx|doc|html)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_BOOK_RE = re.compile(r"《([^《》]{1,80})》")

_CONTEXT_WINDOW = 6

# 文件名的中文话语前缀（修复计划：先识别后缀，再剥离这些前缀——
# 「请参考退货政策.md」「详见 配送说明.txt」都归一为文件名本身）
_DISCOURSE_PREFIXES = ("请参考", "参考", "详见", "见", "来源", "根据", "依据")


def _strip_discourse_prefix(value: str) -> str:
    """剥离开头的话语前缀与空格（只作用于文件名引用，不动书名号语境）。"""
    out = value.strip()
    changed = True
    while changed:
        changed = False
        for prefix in _DISCOURSE_PREFIXES:
            if out.startswith(prefix):
                out = out[len(prefix):].lstrip()
                changed = True
    return out


def extract_citations(text: str) -> list[str]:
    """从回复文本提取引用（原文值，顺序保留、去重）。宁漏勿误杀。"""
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        key = value.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)

    for match in _BOOK_RE.finditer(text):
        start = match.start()
        context = text[max(0, start - _CONTEXT_WINDOW):start]
        if any(word in context for word in CITATION_CONTEXT_WORDS):
            _add(match.group(1))

    for match in _FILENAME_RE.finditer(text):
        _add(_strip_discourse_prefix(match.group(0)))

    return out


def normalize_source(name: str) -> str:
    """来源归一：NFKC → basename → 去扩展名 → casefold。"""
    value = unicodedata.normalize("NFKC", str(name)).strip()
    value = value.replace("\\", "/")
    value = Path(value).name  # basename（含扩展名）
    lowered = value.lower()
    for ext in _CITATION_FILE_EXT:
        if lowered.endswith(ext):
            value = value[: -len(ext)]
            break
    return value.casefold()


def verify_citations(reply: str, sources: set[str]) -> dict:
    """校验回复中的引用是否都来自本轮召回集合。

    sources 为规范化值集合（本轮回召合法来源）；返回：
    - cited：提取到的引用原文列表（可能为空）
    - matched：其中命中来源集合的原文
    - missing：未命中的原文（比较用规范化值）
    """
    cited = extract_citations(reply)
    normalized_sources = {normalize_source(s) for s in sources if s}
    matched: list[str] = []
    missing: list[str] = []
    for item in cited:
        (matched if normalize_source(item) in normalized_sources else missing).append(item)
    return {"cited": cited, "matched": matched, "missing": missing}


def apply_citation_policy(result, sources: set[str]) -> Optional[dict]:
    """Agent 内部分级处置（改造三；单/多 Agent 同约定）。

    - 无引用（纯闲聊/纯工具数据）→ 放行；
    - 零检索却有引用 → requires_human=true + confidence 压低 + 告警；
    - 有引用不匹配（部分 missing）→ confidence 压低 + 告警；
    - 全部命中 → 放行。
    返回 verdict（供 RunTrace/日志）；调用方须在 _record_turn/_save_session
    之前调用（confidence/requires_human 修改先于落库与演进采集）。
    """
    import logging

    from app.config.settings import settings

    if not settings.citation_check_enabled:
        return None
    verdict = verify_citations(result.reply, sources)
    if not verdict["cited"]:
        return verdict  # 无引用 → 放行
    log = logging.getLogger("app.agent.citations")
    if not sources:
        result.requires_human = True
        result.confidence = round(min(result.confidence, 0.5), 4)
        log.warning("citation.zero_source cited=%s", verdict["cited"])
        return verdict
    if verdict["missing"]:
        result.confidence = round(result.confidence * 0.6, 4)
        log.warning("citation.mismatch missing=%s", verdict["missing"])
    return verdict
