"""miner.py：未处理 turn 读取、legacy session 状态机、规则过滤、candidate ID（第10期）。

职责：
1. iter_turn_files / load_turn / mine_turns：读取 turns 目录中未被 ledger 处理的 turn。
2. scan_legacy_session：解析 v1 session 文件（无 session_id）为 TurnRecord(mode="legacy")，
   处理 正常轮次 / 工具链多步 / 双 assistant / 损坏 JSON / 孤立消息 五种形态。
3. build_candidate：规则过滤（置信度、转人工、无来源、长度、注入/PII）→ CandidateQA。
4. candidate ID：turn 类 sha256("turn:v1:"+turn_id)；legacy 的 turn_id 本身由
   session_key+msg_index 确定性生成，因此候选 ID 跨 LLM 输出稳定。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from app.config.settings import settings
from app.evolution.models import CandidateQA, SourceRef, TurnRecord
from app.evolution.recorder import parse_turn_slice
from app.evolution.sanitizer import (
    candidate_id_for_turn,
    has_injection,
    has_pii,
    normalize_answer,
    normalize_question,
    sanitize_text,
)


def legacy_turn_id(session_key: str, msg_index: int) -> str:
    """legacy 轮次的确定性 turn_id：同 session 同轮次重复挖掘得到同一 ID。"""
    return hashlib.sha256(
        f"legacy-turn:{session_key}:{msg_index}".encode("utf-8")
    ).hexdigest()[:32]


# ============================================================
# 未处理 turn 读取
# ============================================================
def iter_turn_files(turns_dir: Path) -> list[Path]:
    """返回 turns/YYYYMMDD/*.json 的排序文件清单。"""
    base = Path(turns_dir)
    if not base.exists():
        return []
    return sorted(p for p in base.rglob("*.json") if not p.name.endswith(".tmp"))


def load_turn(path: Path) -> Optional[TurnRecord]:
    """读取单个 turn 文件；损坏返回 None。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return TurnRecord.from_dict(data)
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


def mine_turns(turns_dir: Path, processed: set[str]) -> list[TurnRecord]:
    """读取所有未被 ledger 处理过的 turn 记录。"""
    out: list[TurnRecord] = []
    for path in iter_turn_files(turns_dir):
        rec = load_turn(path)
        if rec is not None and rec.turn_id not in processed:
            out.append(rec)
    return out


# ============================================================
# legacy session 状态机
# ============================================================
def _parse_final_json(content: str) -> tuple[str, str, float, bool, Optional[str]]:
    """解析最终 assistant 的结构化 JSON 内容。

    损坏 JSON → 原始文本降级（reply=原文，其余取默认），保证不丢候选。
    """
    text = (content or "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("reply"):
            return (
                str(data["reply"]),
                str(data.get("intent", "")),
                float(data.get("confidence", 0.0) or 0.0),
                bool(data.get("requires_human", False)),
                data.get("follow_up_question") or data.get("follow_up"),
            )
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return text, "", 0.0, False, None


def _build_legacy_turn(
    session_key: str,
    session_id: Optional[str],
    msg_index: int,
    session_ts: str,
    turn_msgs: list[dict],
) -> Optional[TurnRecord]:
    """从一轮消息构造 legacy TurnRecord；孤立消息（无最终回答）返回 None。

    双 assistant / 工具链形态：取最后一个无 tool_calls 的 assistant 消息作为回答，
    来源通过 parse_turn_slice 从整个轮次中提取。
    """
    question = ""
    for m in turn_msgs:
        if m.get("role") == "user":
            question = m.get("content", "") or ""
            break

    finals = [
        m for m in turn_msgs
        if m.get("role") == "assistant" and not m.get("tool_calls")
    ]
    if not finals:
        return None  # 孤立消息：无最终回答

    reply, intent, confidence, requires_human, follow_up = _parse_final_json(
        finals[-1].get("content") or ""
    )
    if not reply:
        return None

    return TurnRecord(
        turn_id=legacy_turn_id(session_key, msg_index),
        session_id=session_id or session_key,
        mode="legacy",
        ts=session_ts,
        question=sanitize_text(question),
        reply=sanitize_text(reply),
        intent=intent,
        confidence=confidence,
        requires_human=requires_human,
        follow_up=follow_up,
        sources=parse_turn_slice(turn_msgs, 0),
        status="captured",
    )


def scan_legacy_session(session_path: Path) -> list[TurnRecord]:
    """解析一个 v1 session 文件（无 session_id）为轮次清单。

    轮次以 user 消息为边界；损坏的 session 文件返回空列表。
    """
    try:
        data = json.loads(session_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, dict):
        return []

    messages = data.get("messages", [])
    session_key = session_path.stem  # 如 "session"（多用户可并发多份文件）
    session_id = data.get("session_id")
    session_ts = data.get("updated_at", "")

    turns: list[TurnRecord] = []
    turn_msgs: list[dict] = []
    msg_index = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            if turn_msgs:
                rec = _build_legacy_turn(
                    session_key, session_id, msg_index, session_ts, turn_msgs
                )
                if rec is not None:
                    turns.append(rec)
                msg_index += 1
            turn_msgs = [msg]
            continue
        if turn_msgs:
            turn_msgs.append(msg)

    if turn_msgs:
        rec = _build_legacy_turn(
            session_key, session_id, msg_index, session_ts, turn_msgs
        )
        if rec is not None:
            turns.append(rec)

    return turns


def scan_legacy_sessions(session_files: list[Path]) -> list[TurnRecord]:
    """批量解析多个 v1 session 文件并汇总。"""
    out: list[TurnRecord] = []
    for path in session_files:
        out.extend(scan_legacy_session(path))
    return out


# ============================================================
# 规则过滤 → CandidateQA
# ============================================================
def build_candidate(
    turn: TurnRecord,
    min_confidence: float | None = None,
) -> tuple[Optional[CandidateQA], Optional[str]]:
    """规则过滤并构造候选；返回 (candidate, None) 或 (None, skip_reason)。

    skip_reason ∈ {low_confidence, requires_human, no_sources, short, sensitive}，
    与 EvolutionReport.skipped 的键对应。
    """
    if min_confidence is None:
        min_confidence = settings.evolve_min_confidence
    if turn.confidence < min_confidence:
        return None, "low_confidence"
    if turn.requires_human:
        return None, "requires_human"
    if not turn.sources:
        return None, "no_sources"

    question = normalize_question(turn.question)
    answer = normalize_answer(turn.reply)
    if not question or not answer:
        return None, "short"

    sample = question + "\n" + answer
    if has_injection(sample) or has_pii(sample):
        return None, "sensitive"

    candidate = CandidateQA(
        candidate_id=candidate_id_for_turn(turn.turn_id),
        turn_id=turn.turn_id,
        question=question,
        answer=answer,
        raw_question=turn.question,
        intent=turn.intent,
        confidence=turn.confidence,
        sources=turn.sources,
        filter_state="pending",
    )
    return candidate, None