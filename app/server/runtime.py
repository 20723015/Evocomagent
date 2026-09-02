"""同步/异步执行模型（阶段一 1.7）——先在阶段一就定死，避免 2.2/4.5 返工。

规则：
1. Agent 全链路（openai SDK、MCP 线程桥、工具执行）是同步代码；
2. FastAPI/SSE 是 async 生态，事件循环内**禁止**直接调用同步 LLM/工具
   （一处同步卡死整个 pod 的并发）；
3. 统一出口：run_agent_turn —— 把同步调用丢进 anyio 线程池，
   并用 pod 级 CapacityLimiter 限制并发 Agent 数（与阶段四 4.4 信号量衔接）。
"""

from __future__ import annotations

import logging
from typing import Any

from anyio import CapacityLimiter, to_thread

from app.config.settings import settings

logger = logging.getLogger("app.server.runtime")

_limiter: CapacityLimiter | None = None


def get_agent_limiter() -> CapacityLimiter:
    """pod 级 Agent 并发上限（可被测试替换）。"""
    global _limiter
    if _limiter is None:
        _limiter = CapacityLimiter(settings.server_agent_threads)
    return _limiter


def reset_agent_limiter() -> None:
    global _limiter
    _limiter = None


def run_agent_turn_sync(agent: Any, message: str) -> Any:
    """同步执行一轮对话（在 Agent 线程内调用）。"""
    return agent.chat(message)


def run_agent_close_sync(agent: Any) -> None:
    try:
        agent.close()
    except Exception:  # noqa: BLE001 —— 收尾失败不影响响应已返回
        logger.warning("agent.close() 收尾失败", exc_info=True)


async def run_agent_turn(agent: Any, message: str, on_start=None) -> Any:
    """在线程池执行一轮对话（事件循环内唯一入口）。

    on_start：在 Worker 线程内、agent.chat 之前调用（阶段三 3.7 用量归集
    需要 thread-local 用户标记——必须在工作线程内设置）。
    """

    def _run():
        if on_start is not None:
            on_start()
        return agent.chat(message)

    return await to_thread.run_sync(_run, limiter=get_agent_limiter())


async def run_agent_close(agent: Any) -> None:
    """线程池内执行收尾（LTM 巩固/连接清理），与主调用共用容量限制。"""
    await to_thread.run_sync(
        run_agent_close_sync, agent, limiter=get_agent_limiter()
    )


async def run_agent_reset(agent: Any) -> Any:
    return await to_thread.run_sync(agent.reset, limiter=get_agent_limiter())
