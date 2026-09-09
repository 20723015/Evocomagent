"""EvidencePack（单 Agent 全量优化计划·阶段D）：检索证据统一容器。

一次（多子查询）检索的全部可用证据，供：
- 模型上下文（结果 JSON 的 results 数组，兼容原字段）；
- 声明级事实接地（fact_guard 的 evidence_texts）；
- 引用校验（sources 集合，tainted 片段不进合法来源）；
- 拒答校准（top1 分数、分数间隔、词面覆盖率、reranker 分数）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvidenceItem:
    """单条证据片段。"""

    doc: str
    section: str = ""
    heading_path: str = ""
    chunk_id: str = ""
    parent_id: str = ""
    query: str = ""              # 命中该片段的子查询（多路召回时）
    score: float = 0.0           # 合并后分数
    raw_scores: dict = field(default_factory=dict)  # 各路原始分（诊断）
    text: str = ""               # 进模型上下文的文本（父块优先）
    matched_text: str = ""       # 命中子块原文
    source_path: str = ""
    tainted: bool = False
    context_type: str = "self"

    def to_result(self) -> dict:
        """兼容 search_knowledge 原结果字段。"""
        return {
            "doc": self.doc,
            "section": self.section,
            "score": round(self.score, 4),
            "matched_text": self.matched_text,
            "text": self.text,
            "heading_path": self.heading_path,
            "parent_id": self.parent_id,
            "context_type": self.context_type,
            "source_path": self.source_path or self.doc,
            "provenance": "",
        }


@dataclass
class EvidencePack:
    """一轮检索的证据包。"""

    query: str                    # 原始查询
    subqueries: list[str] = field(default_factory=list)
    items: list[EvidenceItem] = field(default_factory=list)

    @property
    def tainted_hits(self) -> int:
        return sum(1 for item in self.items if item.tainted)

    def sources(self) -> list[str]:
        """合法来源（tainted 不背书引用）。"""
        out: list[str] = []
        seen: set[str] = set()
        for item in self.items:
            if item.tainted:
                continue
            for value in (item.doc, item.source_path):
                if value and value not in seen:
                    seen.add(value)
                    out.append(value)
        return out

    def evidence_texts(self) -> list[str]:
        """声明级接地用原文（未污染片段）。"""
        return [
            item.text or item.matched_text
            for item in self.items if not item.tainted
        ]

    def diagnostics(self) -> dict:
        """拒答校准诊断字段（top1/间隔/来源数）。"""
        scores = sorted((item.score for item in self.items if not item.tainted),
                        reverse=True)
        top1 = scores[0] if scores else None
        gap = (scores[0] - scores[1]) if len(scores) > 1 else None
        return {
            "top1_score": top1,
            "top1_gap": gap,
            "n_items": len(scores),
            "tainted": self.tainted_hits,
            "subqueries": self.subqueries,
        }

    def to_results(self) -> list[dict]:
        return [item.to_result() for item in self.items]


def rrf_merge(ranked_lists: list[list[EvidenceItem]], k: int = 60,
              top_n: int = 10) -> list[EvidenceItem]:
    """Reciprocal Rank Fusion 合并多路子查询召回。

    同一 parent_id / chunk 的命中只保留最高融合分的一条（父块去重）；
    raw_scores 保留各路原始分供诊断。
    """
    fused: dict[str, EvidenceItem] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            key = item.parent_id or f"{item.doc}:{item.section}:{item.chunk_id}"
            contribution = 1.0 / (k + rank + 1)
            existing = fused.get(key)
            if existing is None:
                item.score = contribution
                item.raw_scores.setdefault("rrf", round(contribution, 6))
                fused[key] = item
            else:
                existing.score += contribution
                existing.raw_scores["rrf"] = round(existing.score, 6)
    merged = list(fused.values())
    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[:top_n]
