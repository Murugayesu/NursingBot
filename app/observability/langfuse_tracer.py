from __future__ import annotations

import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class NoOpTracer:
    """Stub tracer used when Langfuse is disabled."""

    @contextmanager
    def trace(self, name: str, **kwargs):
        yield self

    @contextmanager
    def span(self, name: str, **kwargs):
        yield self

    def update(self, **kwargs) -> None:
        pass

    def end(self, **kwargs) -> None:
        pass


class LangfuseTracer:
    """
    Thin wrapper around the Langfuse Python SDK.
    Instruments each RAG pipeline stage as a named span within a trace.

    Usage:
        async with tracer.trace("rag_request", query=query) as trace:
            async with tracer.span(trace, "dense_retrieval"):
                results = await retriever.retrieve(...)
    """

    def __init__(self) -> None:
        from app.config.settings import get_settings
        settings = get_settings()
        from langfuse import Langfuse
        self._lf = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )

    @contextmanager
    def new_trace(self, name: str, metadata: dict[str, Any] | None = None):
        trace = self._lf.trace(name=name, metadata=metadata or {})
        t0 = time.perf_counter()
        try:
            yield trace
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000
            trace.update(metadata={**(metadata or {}), "total_latency_ms": round(latency_ms, 2)})

    @contextmanager
    def span(self, trace, name: str, input_data: Any = None, metadata: dict | None = None):
        span = trace.span(name=name, input=input_data, metadata=metadata or {})
        t0 = time.perf_counter()
        try:
            yield span
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000
            span.end(metadata={**(metadata or {}), "latency_ms": round(latency_ms, 2)})

    def flush(self) -> None:
        self._lf.flush()


_tracer: LangfuseTracer | NoOpTracer | None = None


def get_tracer() -> LangfuseTracer | NoOpTracer:
    global _tracer
    if _tracer is not None:
        return _tracer

    from app.config.settings import get_settings
    settings = get_settings()

    if settings.langfuse_enabled and settings.langfuse_public_key:
        try:
            _tracer = LangfuseTracer()
            logger.info("langfuse tracing enabled")
        except Exception as e:
            logger.warning("langfuse init failed, using no-op tracer", error=str(e))
            _tracer = NoOpTracer()
    else:
        _tracer = NoOpTracer()

    return _tracer
