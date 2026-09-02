"""静默会话兜底巩固（阶段二 2.3）。

K8s 里 close() 永远不可靠——如果 pod 在用户会话静默期被重启，或会话攒了
很多轮没到增量阈值就被杀掉，LTM 就丢了。后台 job 按
memory_consolidate_idle_minutes 扫描超时未动的会话，把消息巩固进 LTM。

约定：
- 会话「最后活动时间」取自 SessionState.updated_at（store 写回时刷新）；
- 只处理「有未巩固消息」（consolidated_len < len(messages)）的会话——
  水位随会话文档持久化（安全修复 P2），历史实现每轮扫描对同一静默会话
  全量重复巩固（interaction_summaries 无限膨胀）；
- 巩固后把水位 CAS 写回会话文档；写回冲突（期间用户来了新消息）只记
  日志放弃——水位由 Agent 下一轮自行推进；
- 单次处理失败只记日志，不中断扫描（job 重跑幂等：add_facts 有内容去重）。
"""

from __future__ import annotations

import logging
from datetime import datetime

from app.stores.base import SessionConflictError, SessionState

logger = logging.getLogger("app.stores.consolidator")


def _parse_ts(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def find_idle_sessions(
    store,
    *,
    idle_minutes: int,
    now: datetime | None = None,
) -> list[tuple[str, str]]:
    """返回 (user_id, session_id) 列表：最后活动超过 idle_minutes 的会话。

    store 需同时支持「枚举全部会话」——LocalFileSessionStore 无此能力，
    仅在 RedisSessionStore（SCAN session:*）下生效；其余返回 []。
    """
    scanner = getattr(store, "iter_all", None)
    if scanner is None:
        return []
    now = now or datetime.now()
    cutoff = now.timestamp() - idle_minutes * 60
    out: list[tuple[str, str]] = []
    for user_id, session_id in scanner():
        state = store.load(user_id, session_id)
        if state is None or not state.messages:
            continue
        if state.consolidated_len >= len(state.messages):
            continue  # 全量已巩固（含旧格式升级默认值），无需兜底
        ts = _parse_ts(state.updated_at)
        if ts is not None and ts.timestamp() < cutoff:
            out.append((user_id, session_id))
    return out


def run_idle_consolidation(
    session_store,
    ltm_store,
    client,
    model,
    *,
    idle_minutes: int,
    memory_dir: str,
    max_ltm_facts: int = 50,
) -> list[str]:
    """对全部静默会话执行一次兜底巩固，返回处理的 (user_id, session_id) 列表。"""
    handled: list[str] = []
    from app.agent.memory import MemoryManager

    for user_id, session_id in find_idle_sessions(
        session_store, idle_minutes=idle_minutes,
    ):
        state = session_store.load(user_id, session_id)
        if state is None or not state.messages:
            continue
        watermark = min(state.consolidated_len, len(state.messages))
        if watermark >= len(state.messages):
            continue
        try:
            mm = MemoryManager(
                client=client,
                model=model,
                user_id=user_id,
                memory_dir=memory_dir,
                memory_enabled=True,
                max_ltm_facts=max_ltm_facts,
                ltm_store=ltm_store,
            )
            mm.bind_session(session_id)
            mm.consolidate_to_long_term(
                state.messages[watermark:], state.summary,
            )
            # 水位 CAS 写回（防下一轮扫描重复巩固）；冲突 = 期间有新写入，
            # 放弃写回（Agent 下一轮自行推进水位）
            session_store.save(
                user_id, session_id,
                SessionState(
                    **{
                        **state.__dict__,
                        "consolidated_len": len(state.messages),
                    }
                ),
                new_messages=[],
            )
            handled.append(f"{user_id}/{session_id}")
        except SessionConflictError as e:
            logger.info("idle consolidate 水位写回冲突 %s/%s: %s",
                        user_id, session_id, e)
        except Exception as e:  # noqa: BLE001 —— 单会话失败不中断扫描
            logger.warning("idle consolidate 失败 %s/%s: %s", user_id, session_id, e)
    return handled
