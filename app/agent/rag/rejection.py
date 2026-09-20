"""四信号联合拒绝（P1-4 生产模块）：把校准脚本的判定逻辑收敛为唯一实现。

拒答不再只看单一 top1 分数——联合四个信号（任一已配置信号不达标即拒绝）：
1. ``top1``    ：最终 Top-1 分数（绝对相关度；RRF 秩融合分/降级结果无语义，
                 判定整体跳过，见 ``decide_rejection``）；
2. ``gap``     ：top1 - top2（可分性；只有一条命中时无法计算 → 不判定）；
3. ``coverage``：query 与 top1 文本的汉字 bigram 词面覆盖率（防「高分但答非
                 所问」；与记忆相关性同一口径）；
4. ``rerank``  ：精排分（``RetrievedChunk.rerank_score``，与向量分不同尺度，
                 仅用于二次门控；未挂精排/未校准 → 不判定）。

与校准脚本同源：``app/scripts/calibrate_rejection.py`` 从本模块 import
``compute_signals`` / ``should_reject``，杜绝两份实现漂移。

与单阈值 ``rag_min_relevance_score`` 的组合语义（调用方 knowledge.py 落实）：
联合拒绝是「单阈值不可达」的**替代**——只要任一联合阈值已配置
（``RejectionParams.active``），最终门控就由联合判定独占，legacy 单阈值不再
作为最终口径传入（避免同一层正例被两道门双重砍）；仅当四个联合阈值全为
None 时才回落到 legacy 单阈值。逐路召回的阈值门控（final_multi_search 的
``min_score``）不属于最终口径，语义不变。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from app.agent.rag.retriever import SCORE_SOURCE_RERANK, SCORE_SOURCE_VECTOR

SIGNAL_TOP1 = "top1"
SIGNAL_GAP = "gap"
SIGNAL_COVERAGE = "coverage"
SIGNAL_RERANK = "rerank"
SIGNAL_NO_HITS = "no_hits"

# 分数有绝对语义的 score_source（与 retriever.RetrievalResult.scores_meaningful 同源）
_MEANINGFUL_SOURCES = (SCORE_SOURCE_VECTOR, SCORE_SOURCE_RERANK)


def bigram_set(text: str) -> set[str]:
    """汉字 bigram + ASCII 整词（与记忆相关性同一口径）。"""
    value = unicodedata.normalize("NFKC", text or "")
    tokens: set[str] = set()
    tokens.update(w.casefold() for w in re.findall(r"[A-Za-z0-9]+", value))
    run: list[str] = []
    for ch in value:
        if "\u4e00" <= ch <= "\u9fff":
            run.append(ch)
        else:
            if run:
                tokens.update(f"{a}{b}" for a, b in zip(run, run[1:]))
                run = []
    if run:
        tokens.update(f"{a}{b}" for a, b in zip(run, run[1:]))
    return tokens


def lexical_coverage(query: str, text: str) -> float:
    """query 与文本的词面覆盖率（|∩| / |query tokens|）。"""
    q_tokens = bigram_set(query)
    if not q_tokens:
        return 0.0
    t_tokens = bigram_set(text)
    return len(q_tokens & t_tokens) / len(q_tokens)


@dataclass(frozen=True)
class RejectionParams:
    """联合拒绝的四个阈值；None = 该信号未校准/不参与判定。

    ``min_rerank`` 在未挂精排（rerank=none）或精排分不可得时天然为 None：
    4 号信号休眠而不是拿一个未校准的尺度硬判。
    """

    min_top1: float | None = None
    min_gap: float | None = None
    min_coverage: float | None = None
    min_rerank: float | None = None

    @property
    def active(self) -> bool:
        """任一阈值已配置即启用联合门控（替代 legacy 单阈值）。"""
        return any(
            value is not None
            for value in (self.min_top1, self.min_gap, self.min_coverage,
                          self.min_rerank)
        )

    def to_dict(self) -> dict:
        return {
            "min_top1": self.min_top1,
            "min_gap": self.min_gap,
            "min_coverage": self.min_coverage,
            "min_rerank": self.min_rerank,
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> RejectionParams:
        payload = payload or {}
        return cls(
            min_top1=payload.get("min_top1"),
            min_gap=payload.get("min_gap"),
            min_coverage=payload.get("min_coverage"),
            min_rerank=payload.get("min_rerank"),
        )


def params_from_settings() -> RejectionParams:
    """冻结当前 settings 的联合拒绝参数（后续改动不复读 settings）。"""
    from app.config.settings import settings

    return RejectionParams(
        min_top1=settings.rag_rejection_min_top1,
        min_gap=settings.rag_rejection_min_gap,
        min_coverage=settings.rag_rejection_min_coverage,
        min_rerank=settings.rag_rejection_min_rerank,
    )


def rrf_mode() -> bool:
    """当前配置是否产出「无语义分数」（hybrid 且未挂精排）。

    单一来源：检索评测的 `--calibrate` 守卫、拒绝参数校准的守卫、
    `knowledge` 工具的适用性判定共用本函数，避免三处各自实现漂移。
    RRF 只依赖排名，分数量纲与余弦/精排分不可比——在其上校准阈值
    会得到一批「看起来合理、实则无意义」的阈值。
    """
    from app.config.settings import settings

    rerank = (settings.rag_rerank or "none").lower()
    return bool(settings.rag_hybrid) and rerank in ("", "none")


def _threshold(params, name: str):
    """阈值读取：RejectionParams 字段 / dict 键；None = 信号未配置。"""
    if isinstance(params, RejectionParams):
        return getattr(params, name)
    if isinstance(params, dict):
        return params.get(name)
    return getattr(params, name, None)


def compute_signals(hits: list, query: str) -> dict:
    """在最终口径 hits（父块折叠后）上取四信号；缺失信号为 None/0.0。"""
    top1 = hits[0].score if hits else None
    gap = (hits[0].score - hits[1].score) if len(hits) > 1 else None
    coverage = lexical_coverage(
        query, hits[0].chunk.text if hits else "",
    )
    rerank = None
    if hits:
        rerank = getattr(hits[0], "rerank_score", None)
        if rerank is None:
            rerank = getattr(hits[0].chunk, "rerank_score", None)
    return {
        "top1": top1,
        "gap": gap,
        "coverage": coverage,
        "rerank": rerank,
        "n_hits": len(hits),
    }


def rejection_reasons(signal: dict, params) -> list[str]:
    """命中的拒绝信号列表（空 = 通过）；top1 缺失 → 直接拒绝（fail-closed）。"""
    reasons: list[str] = []
    top1 = signal.get("top1")
    if top1 is None:
        return [SIGNAL_NO_HITS]
    min_top1 = _threshold(params, "min_top1")
    if min_top1 is not None and top1 < min_top1:
        reasons.append(SIGNAL_TOP1)
    gap = signal.get("gap")
    min_gap = _threshold(params, "min_gap")
    if min_gap is not None and gap is not None and gap < min_gap:
        reasons.append(SIGNAL_GAP)
    min_coverage = _threshold(params, "min_coverage")
    if min_coverage is not None and signal.get("coverage", 0.0) < min_coverage:
        reasons.append(SIGNAL_COVERAGE)
    rerank = signal.get("rerank")
    min_rerank = _threshold(params, "min_rerank")
    if min_rerank is not None and rerank is not None and rerank < min_rerank:
        reasons.append(SIGNAL_RERANK)
    return reasons


def should_reject(signal: dict, params) -> bool:
    """联合信号拒答判定（确定性；校准脚本与生产共用同一实现）。"""
    return bool(rejection_reasons(signal, params))


@dataclass(frozen=True)
class RejectionDecision:
    """一次联合拒绝判定的完整结果（供调用方 fail-closed 与观测）。"""

    rejected: bool = False
    applicable: bool = True
    reasons: tuple[str, ...] = ()
    signals: dict = field(default_factory=dict)
    skip_reason: str = ""

    @property
    def reason(self) -> str:
        """拒绝原因（多信号时逗号连接）；未拒绝为空串。"""
        return ",".join(self.reasons)


def decide_rejection(hits: list, query: str, params: RejectionParams,
                     *, score_source: str = SCORE_SOURCE_VECTOR,
                     degraded: bool = False) -> RejectionDecision:
    """最终口径的联合拒绝判定。

    - 参数未配置（全 None）→ applicable=False（调用方回落 legacy 单阈值）；
    - 降级 / RRF 秩融合分 → applicable=False（分数无语义，不得判定）；
    - 空 hits → rejected=True（no_hits，fail-closed）。
    """
    if not params.active:
        return RejectionDecision(applicable=False, skip_reason="not_configured")
    if degraded:
        return RejectionDecision(applicable=False, skip_reason="degraded")
    if score_source not in _MEANINGFUL_SOURCES:
        return RejectionDecision(
            applicable=False, skip_reason=f"score_source={score_source}",
        )
    signals = compute_signals(hits, query)
    reasons = rejection_reasons(signals, params)
    return RejectionDecision(
        rejected=bool(reasons),
        applicable=True,
        reasons=tuple(reasons),
        signals=signals,
    )
