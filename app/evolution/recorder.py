"""TurnRecorder：轮次切片解析 + 原子落盘（第10期 QA 自动沉淀）。

职责：
1. parse_turn_slice：从本轮新增消息中提取 search_knowledge 工具命中
   （匹配 tool_calls 的 tool_call_id → json.loads 对应 tool 消息 content，
   仅保留 success=True 的 results）。
2. record：sanitizer 脱敏 → 原子写 turns/YYYYMMDD/<turn_id>.json
   （tmp + os.replace）。任何异常只打印警告，绝不阻断主流程。

挂载于 chat.py / orchestrator.py 的结构化 assistant 消息 append 之后、
_compress_history 之前；历史压缩不丢候选（turn 已落盘）。
"""

from __future__ import annotations
from app.observability.logging import get_logger
log = get_logger("app.evolution.recorder")


import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.evolution.models import SourceRef, TurnRecord
from app.evolution.sanitizer import sanitize_text

TOOL_NAME = "search_knowledge"


def parse_tool_calls(messages: list[dict], start_idx: int) -> list[dict]:
    """审计（3.6）：从本轮切片提取工具调用明细 [name, arguments, result]。

    arguments 过 sanitizer 脱敏（凭证/敏感参数永不落盘）；result 为
    "ok"/"error" 结果码。供 TurnRecord.tool_calls 审计追责。
    """
    slice_ = messages[start_idx:]
    outcomes: dict[str, str] = {}
    for msg in slice_:
        if msg.get("role") != "tool":
            continue
        content = msg.get("content") or ""
        code = "error"
        try:
            data = json.loads(content)
            code = "ok" if data.get("success", True) is not False else "error"
        except (json.JSONDecodeError, TypeError):
            code = "error"
        outcomes[msg.get("tool_call_id", "")] = code

    out: list[dict] = []
    for msg in slice_:
        for tc in msg.get("tool_calls", []) if msg.get("role") == "assistant" else []:
            try:
                fn = tc["function"]
                name = fn["name"]
            except (KeyError, TypeError):
                continue
            args = fn.get("arguments", "{}")
            try:
                args_obj = json.loads(args) if isinstance(args, str) else args
            except (json.JSONDecodeError, TypeError):
                args_obj = {}
            out.append({
                "name": name,
                "arguments": sanitize_text(json.dumps(args_obj, ensure_ascii=False)),
                "result": outcomes.get(tc.get("id", ""), "error"),
            })
    return out


def parse_turn_slice(messages: list[dict], start_idx: int) -> list[SourceRef]:
    """扫描 messages[start_idx:] 中的 search_knowledge 工具链，返回命中来源。

    两遍扫描：先收集所有 tool 消息结果（assistant 的 tool_calls 消息总在其
    tool 结果之前，线性单遍会漏配），再匹配 search_knowledge 调用。
    """
    slice_ = messages[start_idx:]

    tool_results: dict[str, dict] = {}
    for msg in slice_:
        if msg.get("role") != "tool":
            continue
        try:
            tool_results[msg["tool_call_id"]] = json.loads(msg.get("content") or "{}")
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    sources: list[SourceRef] = []
    for msg in slice_:
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        for tc in msg["tool_calls"]:
            try:
                name = tc["function"]["name"]
            except (KeyError, TypeError):
                continue
            if name != TOOL_NAME:
                continue
            payload = tool_results.get(tc["id"])
            if not payload or payload.get("success") is not True:
                continue
            for item in payload.get("results", []):
                sources.append(
                    SourceRef(
                        source_path=item.get("source_path", ""),
                        doc=item.get("doc", ""),
                        section=item.get("section", ""),
                        score=float(item.get("score", 0.0) or 0.0),
                        text=sanitize_text(item.get("text", "")),
                    )
                )
    return sources


class TurnRecorder:
    """轮次记录器：脱敏 + 原子落盘（阶段二 2.4：可注入 ObjectStore 归档）。"""

    def __init__(self, turns_dir, clock=None, archive=None):
        self._turns_dir = Path(turns_dir)
        self._clock = clock
        self._archive = archive  # ObjectStore | None；None 时本地落盘

    def _now(self) -> datetime:
        return self._clock.now() if self._clock else datetime.now()

    def record(
        self,
        session_id: str,
        mode: str,
        question: str,
        structured_reply,
        turn_slice: list[dict],
        user_id: str = "",
    ) -> Optional[str]:
        """落盘一条 turn 记录，返回 turn_id；失败返回 None（仅打印警告）。"""
        try:
            now = self._now()
            turn_id = uuid.uuid4().hex
            intent = structured_reply.intent
            rec = TurnRecord(
                turn_id=turn_id,
                session_id=session_id,
                mode=mode,
                user_id=user_id,
                ts=now.isoformat(timespec="seconds"),
                question=sanitize_text(question),
                reply=sanitize_text(structured_reply.reply),
                intent=intent.value if hasattr(intent, "value") else str(intent),
                confidence=float(structured_reply.confidence or 0.0),
                requires_human=bool(structured_reply.requires_human),
                follow_up=structured_reply.follow_up_question,
                sources=parse_turn_slice(turn_slice, 0),
                status="captured",
                tool_calls=parse_tool_calls(turn_slice, 0),
            )

            data = json.dumps(rec.to_dict(), ensure_ascii=False, indent=2)
            if self._archive is not None:
                key = f"turns/{now.strftime('%Y%m%d')}/{turn_id}.json"
                self._archive.put(key, data.encode("utf-8"))
                return turn_id

            out_dir = self._turns_dir / now.strftime("%Y%m%d")
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / f"{turn_id}.json"
            tmp = target.with_suffix(".tmp")
            tmp.write_text(data, encoding="utf-8")
            os.replace(tmp, target)
            return turn_id
        except Exception as e:  # noqa: BLE001 —— 记录失败不得影响主流程
            log.info(f"⚠️  turn 记录失败（不影响主流程）: {e}")
            return None