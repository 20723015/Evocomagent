"""OpenTelemetry 手工埋点（阶段四 4.2）。

- init_tracing：配置 OTLP 导出（OTEL_EXPORTER_OTLP_ENDPOINT 存在才真实导出）；
  未配置时用 NoOpTracerProvider——埋点代码零成本、零依赖运行。
- 结构：一轮对话 = trace（span: server.chat），ReAct 步 / LLM 调用 / 工具调用
  = span，携带 token 数与延迟（4.3 看板的 trace 来源）。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Optional

from app.config.settings import settings

_tracer = None


def init_tracing(service_name: str = "ecom-agent") -> None:
    """初始化 tracer；OTEL 端点未配置时保持 no-op（空跑）。"""
    global _tracer
    endpoint = settings.otel_exporter_endpoint
    from opentelemetry import trace

    if not endpoint:
        _tracer = trace.get_tracer(service_name)
        return
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    )
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)


def get_tracer():
    global _tracer
    if _tracer is None:
        init_tracing()
    return _tracer


@contextmanager
def span(name: str, **attributes):
    """开启一个 span（key: value 属性）；失败不影响业务（埋点不抛错）。"""
    from opentelemetry import trace

    start = time.time()
    with get_tracer().start_as_current_span(name) as current:
        for k, v in attributes.items():
            if v is not None:
                try:
                    current.set_attribute(k, v)
                except Exception:  # noqa: BLE001 —— 属性不合法忽略
                    pass
        try:
            yield current
        finally:
            current.set_attribute("duration_ms", (time.time() - start) * 1000)


def record_llm_call(model: str, purpose: str, latency_ms: float,
                    prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
    """LLM 调用落 span 属性（嵌入缺省 span：无活动时 no-op）。"""
    from opentelemetry import trace

    current = trace.get_current_span()
    if not current.is_recording():
        return
    current.set_attribute("llm.model", model)
    current.set_attribute("llm.purpose", purpose)
    current.set_attribute("llm.latency_ms", latency_ms)
    current.set_attribute("llm.prompt_tokens", prompt_tokens)
    current.set_attribute("llm.completion_tokens", completion_tokens)
