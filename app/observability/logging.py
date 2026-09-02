"""structlog 配置与统一取日志（阶段四 4.1）。

- 生产（server）：JSON 输出（stdout），request_id/session_id/user_id 等在
  调用处绑定上下文；
- CLI：彩色控制台（LOG_FORMAT=console 时生效），保留人可读体验。

用法：from app.observability.logging import get_logger
     log = get_logger("app.agent.chat"); log.info("agent.turn", user_id=...)
"""

from __future__ import annotations

import logging
import sys

import structlog

from app.config.settings import settings

_CONFIGURED = False


def configure_logging(json_output: bool | None = None) -> None:
    """幂等配置 structlog；json_output=None 时读 settings.log_format。"""
    global _CONFIGURED

    fmt = settings.log_format.lower() if json_output is None else (
        "json" if json_output else "console"
    )
    # 给标准 logging 一个兜底 handler，避免第三方库裸用 logging 时全丢
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    renderer = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if fmt == "json"
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str = "app") -> structlog.stdlib.BoundLogger:
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)


def bind_context(**kwargs) -> None:
    """绑定请求级上下文（request_id/user_id/session_id），请求结束时 clear_context。"""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()
