from __future__ import annotations

from qdrant_client import AsyncQdrantClient, QdrantClient

from app.config.settings import get_settings

_async_client: AsyncQdrantClient | None = None
_sync_client: QdrantClient | None = None


def get_qdrant_client() -> AsyncQdrantClient:
    """Return the module-level singleton Qdrant async client."""
    global _async_client
    if _async_client is None:
        settings = get_settings()
        _async_client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_http_port,
            api_key=settings.qdrant_api_key or None,
            prefer_grpc=False,
        )
    return _async_client


def get_sync_qdrant_client() -> QdrantClient:
    """Return a synchronous Qdrant client (used by LangChain vector store)."""
    global _sync_client
    if _sync_client is None:
        settings = get_settings()
        _sync_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_http_port,
            api_key=settings.qdrant_api_key or None,
            prefer_grpc=False,
        )
    return _sync_client


async def close_qdrant_client() -> None:
    global _async_client, _sync_client
    if _async_client is not None:
        await _async_client.close()
        _async_client = None
    if _sync_client is not None:
        _sync_client.close()
        _sync_client = None


async def check_qdrant_health() -> bool:
    """Return True if Qdrant is reachable."""
    try:
        client = get_qdrant_client()
        await client.get_collections()
        return True
    except Exception:
        return False
