"""judges.py：价值 Judge 与接地 Judge（第10期 QA 自动沉淀）。

- ValueJudge：判断候选是否值得沉淀，输出规范化后的问题/答案与质量分。
  结构化输出（beta.parse + pydantic）→ 文本 prompt + _parse_json 降级 →
  一切失败返回 failed=True（只能进 pending 人工审核）。
- GroundingJudge：对照人工知识（非 evolved 来源）检验答案断言，找出无证据断言。
  证据集 = 非 evolved/ 来源（根目录人工文档 + uploads/ 上传文档）；自进化内容一律剔除。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from openai import OpenAI
from pydantic import BaseModel, Field

from app.evolution.models import CandidateQA, SourceRef
from app.prompts.evolution import (
    GROUNDING_PROMPT,
    VALUE_JUDGE_SYSTEM_PROMPT,
    VALUE_JUDGE_TEXT_PROMPT,
)


class ValueJudgement(BaseModel):
    """价值 Judge 的结构化输出。"""

    worth_saving: bool
    quality_score: float = Field(ge=0.0, le=1.0)
    question: str
    answer: str
    reason: str = ""


class GroundingJudgement(BaseModel):
    """接地 Judge 的结构化输出。"""

    grounded: bool
    unsupported: list[str] = Field(default_factory=list)
    reason: str = ""


def _parse_json(raw: str) -> dict:
    """文本降级输出的 JSON 解析（剥离可能的 ``` 代码块）。"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


@dataclass
class ValueDecision:
    """价值 Judge 的判决结果。failed=True 表示 Judge 完全失败（只能进 pending）。"""

    worth_saving: bool
    quality_score: float
    question: str
    answer: str
    reason: str = ""
    failed: bool = False


def _render_input(qa: CandidateQA, sources: list[SourceRef]) -> str:
    if sources:
        source_text = "\n".join(
            f"- 【{s.doc}/{s.section or '概览'}】{s.text[:300]}" for s in sources[:8]
        )
    else:
        source_text = "（无检索来源）"
    return (
        f"用户问题：{qa.question}\n"
        f"客服回答：{qa.answer}\n\n"
        f"检索来源：\n{source_text}"
    )


class ValueJudge:
    """判断候选问答是否值得沉淀进知识库。"""

    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    def judge(self, qa: CandidateQA, sources: list[SourceRef]) -> ValueDecision:
        try:
            return self._judge_structured(qa, sources)
        except Exception:  # noqa: BLE001 —— 结构化失败降级文本
            pass
        try:
            return self._judge_text(qa, sources)
        except Exception:  # noqa: BLE001 —— 全部失败：只能进 pending
            return ValueDecision(
                worth_saving=True,
                quality_score=0.0,
                question=qa.question,
                answer=qa.answer,
                reason="judge_failed",
                failed=True,
            )

    def _judge_structured(self, qa: CandidateQA, sources: list[SourceRef]) -> ValueDecision:
        response = self._client.beta.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": VALUE_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": _render_input(qa, sources)},
            ],
            temperature=0.0,
            response_format=ValueJudgement,
        )
        d = response.choices[0].message.parsed
        return ValueDecision(
            worth_saving=bool(d.worth_saving),
            quality_score=float(d.quality_score or 0.0),
            question=d.question or qa.question,
            answer=d.answer or qa.answer,
            reason=d.reason or "",
        )

    def _judge_text(self, qa: CandidateQA, sources: list[SourceRef]) -> ValueDecision:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": VALUE_JUDGE_TEXT_PROMPT},
                {"role": "user", "content": _render_input(qa, sources)},
            ],
        )
        data = _parse_json(response.choices[0].message.content or "")
        return ValueDecision(
            worth_saving=bool(data["worth_saving"]),
            quality_score=float(data["quality_score"]),
            question=data.get("question") or qa.question,
            answer=data.get("answer") or qa.answer,
            reason=data.get("reason", ""),
        )


class GroundingJudge:
    """接地检验：答案的每个关键断言是否都能在人工知识中找到证据。"""

    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    @staticmethod
    def human_chunks(sources: list[SourceRef]) -> list[SourceRef]:
        """证据集 = 非 evolved/ 来源（根目录人工文档 + uploads/ 上传文档）；自进化一律剔除。"""
        return [s for s in sources if s.source_path and not s.source_path.startswith("evolved/")]

    def judge(self, answer: str, sources: list[SourceRef]) -> dict:
        """返回 {"grounded": bool, "unsupported": list[str], "reason": str}。"""
        human = self.human_chunks(sources)
        if not human:
            return {
                "grounded": False,
                "unsupported": [],
                "reason": "no_human_sources",
            }
        evidence = "\n".join(
            f"- 【{s.doc}/{s.section or '概览'}】{s.text[:300]}" for s in human[:8]
        )
        try:
            return self._judge_structured(answer, evidence)
        except Exception:  # noqa: BLE001 —— 结构化失败降级文本
            pass
        try:
            return self._judge_text(answer, evidence)
        except Exception:  # noqa: BLE001
            return {
                "grounded": False,
                "unsupported": [],
                "reason": "judge_failed",
            }

    def _judge_structured(self, answer: str, evidence: str) -> dict:
        response = self._client.beta.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": GROUNDING_PROMPT},
                {"role": "user", "content": f"客服回答：\n{answer}\n\n知识库证据：\n{evidence}"},
            ],
            temperature=0.0,
            response_format=GroundingJudgement,
        )
        d = response.choices[0].message.parsed
        return {
            "grounded": bool(d.grounded),
            "unsupported": list(d.unsupported or []),
            "reason": d.reason or "",
        }

    def _judge_text(self, answer: str, evidence: str) -> dict:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": GROUNDING_PROMPT},
                {"role": "user", "content": f"客服回答：\n{answer}\n\n知识库证据：\n{evidence}"},
            ],
        )
        data = _parse_json(response.choices[0].message.content or "")
        return {
            "grounded": bool(data["grounded"]),
            "unsupported": list(data.get("unsupported", [])),
            "reason": data.get("reason", ""),
        }