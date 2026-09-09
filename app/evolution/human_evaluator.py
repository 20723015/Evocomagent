"""human_evaluator.py：人工会话知识评审器（LLM 抽取 → 语义去重评分 → 分类落库）。

流程（human_evaluation_jobs 一个任务 = 一段会话或一条编辑后候选）：
    claim（SKIP LOCKED + lease token，心持续租）
    → LLM 抽取 0-N 条规范问答（证据绑定：evidence_message_ids 必须全部命中
      **人工坐席消息子集**——任一引用不在子集 → invalid_evidence 硬门禁拒绝；
      LLM/embedding/RAG 失败 → 抛异常 → retry_wait，不写完成）
    → 逐条校验（长度/PII/注入/证据）+ 构建来源/证据双快照（随候选落库）
    → 语义去重（SemanticDedupService 纯向量通道，双侧 top-1，余弦截断归一化）
    → 分类（双侧矩阵）：
        问题侧命中 managed 且 ≥0.90 → 更新判定（LLM judge；价值分 ≥0.70 进审核）
        答案侧单独命中 → 一律 duplicate 拒绝（答案侧永不触发替换）
        命中 authoritative（根目录/uploads/frontmatter 缺失）→ 审计告警，
        不自动替换，终结为 authoritative_conflict
        低于阈值 → 新知识：worth_saving 且 composite ≥ 0.70 → 人工审核；
        否则自动淘汰（low_value）
    → 短事务结算（complete_evaluation：候选行 + completed + 会话 evaluated）

composite_score = 0.6 × value_score + 0.4 × novelty_score；
novelty_score = 1 - max_similarity（max_similarity = 双侧最高归一化相似度）。
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.config.settings import settings
from app.evolution.authority import AUTHORITATIVE, authority_kind
from app.evolution.human_store import (
    CAND_PENDING_REVIEW,
    CAND_REJECTED,
    CLASS_DUPLICATE,
    CLASS_NEW,
    CLASS_UPDATE,
    JOB_TYPE_CANDIDATE,
    HumanLeaseLost,
)
from app.evolution.sanitizer import (
    has_injection,
    has_pii,
    normalize_answer,
    normalize_question,
)
from app.observability.logging import get_logger

log = get_logger("app.evolution.human_evaluator")

PROMPT_VERSION = "human-extract-v1"

# 评分阈值：新知识 sim<0.90 且 composite≥门槛（settings.human_composite_threshold，
# 初始冻结 0.70，可校准——2026-09 真实环境实测见 settings 注释）；更新价值分 ≥0.70
NOVELTY_THRESHOLD = 0.90
UPDATE_VALUE_THRESHOLD = 0.70
COMPOSITE_VALUE_WEIGHT = 0.6
COMPOSITE_NOVELTY_WEIGHT = 0.4


def _composite_threshold() -> float:
    """新知识复合分门槛（settings 可校准；缺省沿用冻结值 0.70）。"""
    return float(settings.human_composite_threshold)

_EVIDENCE_SNIPPET_CHARS = 2000


# ============================================================
# LLM 抽取（结构化 → 文本降级 → 全失败抛异常进入重试）
# ============================================================
class ExtractionItem(BaseModel):
    question: str
    answer: str
    value_score: float = Field(ge=0.0, le=1.0)
    worth_saving: bool
    reason: str = ""
    evidence_message_ids: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    items: list[ExtractionItem] = Field(default_factory=list)


class UpdateJudgement(BaseModel):
    is_update: bool
    reason: str = ""


def _parse_json_block(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def _render_transcript(messages: list[dict]) -> str:
    """脱敏会话 → 抽取输入（不含 system 噪声消息；带 message_id 供证据引用）。"""
    lines = []
    for m in messages:
        actor = m.get("actor_type", "system")
        if actor == "system":
            continue
        lines.append(
            f"[{m.get('message_id', '')}]({actor}) {str(m.get('content', ''))[:2000]}"
        )
    return "\n".join(lines)


class HumanConversationExtractor:
    """LLM 抽取器：完整会话 → 0-N 条候选（任何失败抛异常 → 任务重试）。"""

    def __init__(self, client, model: str):
        self._client = client
        self._model = model

    def extract(self, messages: list[dict]) -> list[ExtractionItem]:
        from app.prompts.human_knowledge import (
            EXTRACTION_SYSTEM_PROMPT,
            EXTRACTION_TEXT_PROMPT,
        )

        transcript = _render_transcript(messages)
        try:
            return self._extract_structured(EXTRACTION_SYSTEM_PROMPT, transcript)
        except Exception as exc:  # noqa: BLE001 - structured API fallback
            log.info("human_eval.structured_fallback err=%s", type(exc).__name__)
        # 文本降级（解析失败即抛 → 任务重试，绝不硬凑空结果）
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": EXTRACTION_TEXT_PROMPT.format(transcript=transcript),
                },
            ],
            temperature=0.0,
        )
        data = _parse_json_block(response.choices[0].message.content or "{}")
        return [ExtractionItem(**it) for it in data.get("items", [])]

    def _extract_structured(
        self, system_prompt: str, transcript: str
    ) -> list[ExtractionItem]:
        from app.prompts.human_knowledge import EXTRACTION_TEXT_PROMPT

        response = self._client.beta.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": EXTRACTION_TEXT_PROMPT.format(transcript=transcript),
                },
            ],
            temperature=0.0,
            response_format=ExtractionResult,
        )
        return list(response.choices[0].message.parsed.items)

    def is_substantive_update(
        self, question: str, existing: str, candidate: str
    ) -> bool:
        """重复命中后的更新判定（LLM；失败按无实质变化处理 → duplicate）。"""
        from app.prompts.human_knowledge import (
            UPDATE_JUDGE_SYSTEM_PROMPT,
            UPDATE_JUDGE_TEXT_PROMPT,
        )

        response = self._client.beta.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": UPDATE_JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": UPDATE_JUDGE_TEXT_PROMPT.format(
                        question=question,
                        existing=existing[:2000],
                        candidate=candidate[:2000],
                    ),
                },
            ],
            temperature=0.0,
            response_format=UpdateJudgement,
        )
        return bool(response.choices[0].message.parsed.is_update)


# ============================================================
# RAG 评分（经语义去重服务：纯向量通道，双侧 top-1）
# ============================================================
@dataclass
class RagScore:
    max_similarity: float
    novelty_score: float
    composite_score: float
    hit_path: str = ""
    hit_text: str = ""
    question_hit: object | None = None
    answer_hit: object | None = None
    dedup_snapshot: dict = field(default_factory=dict)


class RagScorer:
    """问题与答案分别经 SemanticDedupService 检索 top1，取两侧最高相似度。

    检索装配来自去重服务（GenerationStore.active → 纯向量通道），不再依赖
    线上 knowledge 单例与 hybrid/reranker；分数归一化 min(max(cos,0),1)。
    """

    def __init__(self, dedup_service):
        self._dedup = dedup_service

    def at_threshold(self, hit, *, side: str) -> bool:
        """单侧 top-1 是否达到该侧阈值（归一化余弦 + 容差）。"""
        return self._dedup.at_threshold(hit, side=side)

    def score(self, question: str, answer: str, value_score: float) -> RagScore:
        result = self._dedup.score(question, answer)  # 失败抛异常 → 任务重试
        sides = [s for s in (result.question, result.answer) if s is not None]
        best_sim = max((s.score for s in sides), default=0.0)
        best = max(sides, key=lambda s: s.score) if sides else None
        novelty = 1.0 - best_sim
        composite = (
            COMPOSITE_VALUE_WEIGHT * value_score + COMPOSITE_NOVELTY_WEIGHT * novelty
        )
        meta = self._dedup.snapshot_meta()
        return RagScore(
            max_similarity=best_sim,
            novelty_score=novelty,
            composite_score=composite,
            hit_path=best.path if best else "",
            hit_text=best.text if best else "",
            question_hit=result.question,
            answer_hit=result.answer,
            dedup_snapshot={
                "question": (
                    {"path": result.question.path, "score": result.question.score}
                    if result.question
                    else None
                ),
                "answer": (
                    {"path": result.answer.path, "score": result.answer.score}
                    if result.answer
                    else None
                ),
                "generation": meta.get("generation", ""),
                "embedding_version": meta.get("embedding_version", ""),
            },
        )


def build_evidence_snapshot(
    messages: list[dict], evidence_ids: list[str]
) -> list[dict]:
    """证据快照：每条被引用的人工坐席消息 + 前/后最近的 customer 消息。

    单条截 2000 字；prev/next_customer 向外扫描最近的 customer 消息
    （bot/system 消息不打断上下文关联）。
    """
    index_by_id = {
        str(m.get("message_id", "")): i
        for i, m in enumerate(messages)
        if m.get("message_id")
    }

    def _nearest_customer(start: int, step: int) -> dict | None:
        i = start + step
        while 0 <= i < len(messages):
            m = messages[i]
            if m.get("actor_type") == "customer":
                return {
                    "message_id": str(m.get("message_id", "")),
                    "content": str(m.get("content", ""))[:_EVIDENCE_SNIPPET_CHARS],
                }
            i += step
        return None

    snapshot: list[dict] = []
    for eid in evidence_ids:
        idx = index_by_id.get(eid)
        if idx is None:
            continue  # 非坐席引用已被硬门禁拒绝，理论不可达
        m = messages[idx]
        snapshot.append(
            {
                "message_id": eid,
                "content": str(m.get("content", ""))[:_EVIDENCE_SNIPPET_CHARS],
                "sent_at": str(m.get("sent_at", "") or ""),
                "prev_customer": _nearest_customer(idx, -1),
                "next_customer": _nearest_customer(idx, +1),
            }
        )
    return snapshot


# ============================================================
# 评审器
# ============================================================
class HumanKnowledgeEvaluator:
    """领取评审任务 → 抽取/评分 → 短事务结算（不占 KB 写锁）。"""

    def __init__(
        self,
        store,
        extractor: HumanConversationExtractor,
        scorer: RagScorer,
        *,
        worker_id: str,
        generation_store=None,
    ):
        self._store = store
        self._extractor = extractor
        self._scorer = scorer
        self._worker_id = worker_id
        self._generation_store = generation_store

    def process_once(self) -> bool:
        job = self._store.claim_evaluation_job(self._worker_id)
        if job is None:
            return False
        lost = threading.Event()
        stop_event = threading.Event()
        heartbeat = self._start_heartbeat(job, lost, stop_event)
        try:
            if job["job_type"] == JOB_TYPE_CANDIDATE:
                self._evaluate_candidate(job)
            else:
                self._evaluate_conversation(job)
        except HumanLeaseLost as e:
            log.warning("human_eval.lease_lost job=%s err=%s", job["id"], e)
        except Exception as e:  # noqa: BLE001 - retry policy owns provider errors
            log.warning("human_eval.failed job=%s err=%s", job["id"], type(e).__name__)
            try:
                self._store.fail_evaluation(job, e)
            except HumanLeaseLost:
                log.warning("human_eval.fail_write_lost job=%s", job["id"])
        finally:
            stop_event.set()
            if heartbeat is not None:
                heartbeat.join(timeout=2)
        return True

    def _start_heartbeat(
        self, job: dict, lost: threading.Event, stop_event: threading.Event
    ):
        interval = max(0.05, float(getattr(self._store, "_lease", 30)) / 3.0)

        def _loop():
            while not stop_event.wait(interval):
                try:
                    ok = self._store.heartbeat_evaluation(
                        job["id"],
                        job["lease_token"],
                    )
                except Exception:  # noqa: BLE001
                    ok = False
                if not ok:
                    lost.set()
                    return

        thread = threading.Thread(
            target=_loop,
            name=f"human-eval-hb-{job['id']}",
            daemon=True,
        )
        thread.start()
        return thread

    # ---------- 会话抽取 ----------
    def _evaluate_conversation(self, job: dict) -> None:
        conv = self._store.get_conversation(job["conversation_id"])
        messages = json.loads(conv["transcript_json"]) if conv else []
        items = self._extractor.extract(messages)
        # 证据集合：仅人工坐席消息（引用 customer/bot/system → 硬门禁拒绝）
        agent_ids = {
            str(m.get("message_id", ""))
            for m in messages
            if m.get("actor_type") == "human_agent" and m.get("message_id")
        }
        candidates: list[dict] = []
        for item in items:
            candidates.append(
                self._build_candidate(item, agent_ids, messages, conv)
            )
        self._store.complete_evaluation(
            job,
            candidates=candidates,
            eval_meta=self._eval_meta(),
        )

    def _build_candidate(
        self, item, agent_ids: set[str], messages: list[dict], conv: dict | None
    ) -> dict:
        """单条抽取结果 → 校验/评分/分类（不落库；结算在 complete_evaluation）。"""
        question = normalize_question(item.question)
        answer = normalize_answer(item.answer)
        cited = [str(e) for e in item.evidence_message_ids]
        base = {
            "question": question or str(item.question)[:200],
            "answer": answer or str(item.answer)[:1200],
            "evidence_message_ids": cited,
            "value_score": float(item.value_score),
            "worth_saving": bool(item.worth_saving),
            "value_reason": str(item.reason or "")[:500],
        }
        # 格式闸门：规范化失败 → 拒绝（格式不合法）
        if not normalize_question(item.question) or not normalize_answer(item.answer):
            return {**base, "status": CAND_REJECTED, "reject_reason": "invalid_format"}
        # 证据闸门（硬门禁）：引用必须全部命中人工坐席消息子集，不再静默过滤
        if not cited:
            return {**base, "status": CAND_REJECTED, "reject_reason": "no_evidence"}
        if any(e not in agent_ids for e in cited):
            return {**base, "status": CAND_REJECTED, "reject_reason": "invalid_evidence"}
        # 内容闸门：PII / 注入
        sample = question + "\n" + answer
        if has_pii(sample) or has_injection(sample):
            return {**base, "status": CAND_REJECTED, "reject_reason": "sensitive"}
        # 来源/证据双快照（证据链：批准与追溯依据）
        if conv is not None:
            ended_at = conv.get("ended_at")
            base["source_snapshot"] = {
                "source": conv.get("source", ""),
                "external_conversation_id": conv.get("external_conversation_id", ""),
                "source_version": int(conv.get("source_version") or 0),
                "agent_id": conv.get("agent_id", ""),
                "ended_at": (
                    ended_at.isoformat(sep=" ", timespec="seconds")
                    if getattr(ended_at, "isoformat", None)
                    else str(ended_at or "")
                ),
            }
            base["evidence_snapshot"] = build_evidence_snapshot(messages, cited)
        # 语义去重评分（检索失败 → 异常上抛 → 任务重试）
        rag = self._scorer.score(question, answer, base["value_score"])
        base.update(
            {
                "max_similarity": rag.max_similarity,
                "novelty_score": rag.novelty_score,
                "composite_score": rag.composite_score,
                "rag_hit_path": rag.hit_path,
                "dedup_snapshot": rag.dedup_snapshot,
            }
        )
        return {**base, **self._classify(base, rag)}

    def _classify(self, base: dict, rag) -> dict:
        from app.observability.metrics import (
            record_human_dedup_decision,
            record_human_knowledge_classification,
        )

        q, a = rag.question_hit, rag.answer_hit
        q_hit = self._scorer.at_threshold(q, side="question")
        a_hit = self._scorer.at_threshold(a, side="answer")

        def _authoritative(hit) -> bool:
            return hit is not None and authority_kind(hit.path) == AUTHORITATIVE

        # 命中权威文档（根目录/uploads/frontmatter 缺失）：审计告警，绝不自动替换
        if (q_hit and _authoritative(q)) or (a_hit and _authoritative(a)):
            record_human_knowledge_classification("authoritative_conflict")
            record_human_dedup_decision("duplicate")
            return {
                "classification": CLASS_DUPLICATE,
                "rag_hit_kind": "authoritative_conflict",
                "status": CAND_REJECTED,
                "reject_reason": "authoritative_conflict",
            }
        if q_hit:
            # 仅问题侧命中 managed 文档 → 走更新判定（LLM judge）
            is_update = self._extractor.is_substantive_update(
                base["question"],
                q.text,
                base["answer"],
            )
            if not is_update:
                record_human_knowledge_classification("duplicate")
                record_human_dedup_decision("duplicate")
                return {
                    "classification": CLASS_DUPLICATE,
                    "rag_hit_kind": "duplicate",
                    "status": CAND_REJECTED,
                    "reject_reason": "duplicate",
                }
            # 更新知识：价值分达标即进审核（不受低新颖度淘汰）
            if base["value_score"] >= UPDATE_VALUE_THRESHOLD:
                record_human_knowledge_classification("update")
                return {
                    "classification": CLASS_UPDATE,
                    "rag_hit_kind": "update",
                    "status": CAND_PENDING_REVIEW,
                }
            record_human_knowledge_classification("update_low_value")
            return {
                "classification": CLASS_UPDATE,
                "rag_hit_kind": "update",
                "status": CAND_REJECTED,
                "reject_reason": "low_value",
            }
        if a_hit:
            # 答案侧单独命中（问题侧低于阈值）：一律 duplicate 拒绝
            record_human_knowledge_classification("duplicate")
            record_human_dedup_decision("answer_side")
            return {
                "classification": CLASS_DUPLICATE,
                "rag_hit_kind": "answer_side_duplicate",
                "status": CAND_REJECTED,
                "reject_reason": "duplicate",
            }
        # 新知识：worth_saving 且综合分达标 → 人工审核
        if base["worth_saving"] and rag.composite_score >= _composite_threshold():
            record_human_knowledge_classification("new")
            record_human_dedup_decision("kept")
            return {
                "classification": CLASS_NEW,
                "rag_hit_kind": "new",
                "status": CAND_PENDING_REVIEW,
            }
        record_human_knowledge_classification("low_value")
        return {
            "classification": CLASS_NEW,
            "rag_hit_kind": "new",
            "status": CAND_REJECTED,
            "reject_reason": "low_value",
        }

    # ---------- 编辑后重评 ----------
    def _evaluate_candidate(self, job: dict) -> None:
        candidate = self._store.get_candidate(job["candidate_id"])
        if candidate is None or candidate["status"] != CAND_PENDING_REVIEW:
            self._store.complete_evaluation(job, candidates=[], eval_meta={})
            return

        # 编辑后的文本必须重新通过价值判断，不能复用编辑前的评分。
        reassessed = self._extractor.extract(
            [
                {
                    "message_id": "edited-q",
                    "actor_type": "customer",
                    "content": candidate["question"],
                },
                {
                    "message_id": "edited-a",
                    "actor_type": "human_agent",
                    "content": candidate["answer"],
                },
            ]
        )
        if not reassessed:
            value_score, worth_saving, value_reason = 0.0, False, "edited_low_value"
        else:
            value_score = float(reassessed[0].value_score)
            worth_saving = bool(reassessed[0].worth_saving)
            value_reason = str(reassessed[0].reason or "")[:500]
        rag = self._scorer.score(
            candidate["question"], candidate["answer"], value_score
        )
        base = {
            "question": candidate["question"],
            "answer": candidate["answer"],
            "evidence_message_ids": json.loads(
                candidate.get("evidence_message_ids") or "[]"
            ),
            "value_score": value_score,
            "worth_saving": worth_saving,
            "value_reason": value_reason,
            "max_similarity": rag.max_similarity,
            "novelty_score": rag.novelty_score,
            "composite_score": rag.composite_score,
            "rag_hit_path": rag.hit_path,
            "dedup_snapshot": rag.dedup_snapshot,
        }
        verdict = self._classify(base, rag)
        # 重评不改 question/answer（审核员编辑为准）；分类变化只影响状态：
        # 重复/低价值/权威冲突 → 终结；新/更新 → 留在人工审核
        self._store.settle_candidate_evaluation(
            job,
            {
                "expected_revision": int(candidate["revision"] or 0),
                "value_score": base["value_score"],
                "worth_saving": base["worth_saving"],
                "value_reason": base["value_reason"],
                "max_similarity": rag.max_similarity,
                "novelty_score": rag.novelty_score,
                "composite_score": rag.composite_score,
                "rag_hit_path": rag.hit_path,
                "rag_hit_kind": verdict.get("rag_hit_kind", ""),
                "classification": verdict.get("classification", ""),
                "status": verdict.get("status", CAND_PENDING_REVIEW),
                "reject_reason": verdict.get("reject_reason", ""),
                "dedup_snapshot": rag.dedup_snapshot,
            },
            self._eval_meta(),
        )

    # ---------- 审计元数据 ----------
    def _eval_meta(self) -> dict:
        generation = ""
        if self._generation_store is not None:
            try:
                from app.config.settings import settings as _s

                info = self._generation_store.active(_s.rag_backend.lower())
                generation = info.generation_id if info is not None else ""
            except Exception:  # noqa: BLE001 - audit metadata is best effort
                generation = ""
        return {
            "model": self._extractor._model,
            "prompt_version": PROMPT_VERSION,
            "embedding_version": getattr(self._scorer._dedup, "embedder_model", ""),
            "kb_generation": generation,
        }
