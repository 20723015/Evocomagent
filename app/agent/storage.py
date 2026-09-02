import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from app.observability.logging import get_logger
log = get_logger("app.agent.storage")


SESSION_VERSION = 2


def save_session(
    path: str,
    messages: list[dict],
    summary: Optional[str],
    short_term_memory: Optional[dict] = None,
    session_id: Optional[str] = None,
) -> None:
    """把对话状态原子写入 JSON 文件。

    messages 只包含原始 user/assistant 条目（不含 system / summary）。
    short_term_memory 为短期记忆的序列化数据（第7期）。
    session_id 为会话标识（第10期），缺省生成新的。v1 文件读取后
    首次保存会带上新生成的 session_id。
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": SESSION_VERSION,
        "session_id": session_id or uuid.uuid4().hex,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "messages": messages,
        "short_term_memory": short_term_memory,
    }

    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, file_path)


def load_session(path: str) -> Optional[dict]:
    """读取会话文件。不存在或损坏都返回 None（降级为新会话）。

    返回值增加 session_id 键（第10期）：v1 文件没有该字段时返回 None，
    由调用方生成新会话 ID。
    """
    file_path = Path(path)
    if not file_path.exists():
        return None

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.info(f"⚠️  会话文件损坏，已忽略（{e}）")
        return None

    if not isinstance(data, dict) or "messages" not in data:
        log.info("⚠️  会话文件格式不识别，已忽略")
        return None

    return {
        "session_id": data.get("session_id"),
        "summary": data.get("summary"),
        "messages": data.get("messages", []),
        "short_term_memory": data.get("short_term_memory"),
    }


def delete_session(path: str) -> None:
    """删除会话文件，不存在时静默。"""
    file_path = Path(path)
    if file_path.exists():
        file_path.unlink()
