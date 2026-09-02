"""阶段一 1.7：同步 Agent 经线程池执行——事件循环内不直接调同步代码。"""

from __future__ import annotations

import threading
import time

import anyio

from app.server import runtime
from app.config.settings import settings


class _Agent:
    def __init__(self, delay=0.0, fail=False):
        self.delay = delay
        self.fail = fail
        self.thread_id = None
        self.closed = False

    def chat(self, message: str):
        self.thread_id = threading.get_ident()
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("agent failed")
        return f"echo:{message}"

    def reset(self):
        return "reset-ok"

    def close(self):
        self.closed = True


def test_run_agent_turn_returns_result():
    agent = _Agent()
    result = anyio.run(runtime.run_agent_turn, agent, "你好")
    assert result == "echo:你好"


def test_run_agent_turn_executes_in_threadpool():
    """执行发生在工作线程，而非事件循环所在线程。"""
    agent = _Agent()
    loop_thread = threading.get_ident()
    anyio.run(runtime.run_agent_turn, agent, "你好")
    assert agent.thread_id != loop_thread


def test_run_agent_turn_propagates_agent_error():
    agent = _Agent(fail=True)
    try:
        anyio.run(runtime.run_agent_turn, agent, "你好")
        assert False, "应当抛 RuntimeError"
    except RuntimeError as e:
        assert "agent failed" in str(e)


def test_run_agent_reset_returns_value():
    agent = _Agent()
    assert anyio.run(runtime.run_agent_reset, agent) == "reset-ok"


def test_limiter_caps_concurrent_agents(reset_settings):
    """server_agent_threads=1 时并发请求被串行化（pod 级并发上限生效）。"""
    settings.server_agent_threads = 1
    runtime.reset_agent_limiter()
    try:
        active = 0
        peak = 0
        lock = threading.Lock()

        class _SlowAgent(_Agent):
            def chat(self, message: str):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.15)
                with lock:
                    active -= 1
                return "ok"

        async def main():
            async with anyio.create_task_group() as tg:
                for _ in range(4):
                    tg.start_soon(runtime.run_agent_turn, _SlowAgent(), "x")

        anyio.run(main)
        assert peak == 1  # 容量 1 → 任何时刻只有一个 Agent 在跑
    finally:
        runtime.reset_agent_limiter()
