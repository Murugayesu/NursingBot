"""
Langfuse observability via the official LangChain callback handler.

Correct import for langfuse 4.x:
    from langfuse.langchain import CallbackHandler

Usage:
    handler = get_langfuse_handler(trace_name="rag_query", metadata={...})
    answer = await chain.ainvoke(inputs, config={"callbacks": [handler]})

If Langfuse is not configured or unavailable, a no-op list is returned
so callers don't need to branch.
"""
from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def get_langfuse_handler(
    trace_name: str = "rag",
    metadata: dict[str, Any] | None = None,
):
    """
    Return a configured Langfuse CallbackHandler, or None if disabled.

    Usage:
        callbacks = [h] if (h := get_langfuse_handler("rag_query")) else []
        await chain.ainvoke(inputs, config={"callbacks": callbacks})
    """
    from app.config.settings import get_settings
    settings = get_settings()

    if not (settings.langfuse_enabled and settings.langfuse_public_key):
        return None

    try:
        from langfuse.langchain import CallbackHandler
        handler = CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        handler.trace_name = trace_name
        if metadata:
            handler.metadata = metadata
        return handler
    except Exception as exc:
        logger.warning("langfuse handler init failed, tracing disabled", error=str(exc))
        return None
