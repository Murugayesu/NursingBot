from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config.settings import get_settings

_client: AsyncQdrantClient | None = None


def get_qdrant_client() -> AsyncQdrantClient:
    """Return the module-level singleton Qdrant async client."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_http_port,
            api_key=settings.qdrant_api_key or None,
            prefer_grpc=False,
        )
    return _client


async def close_qdrant_client() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None


async def check_qdrant_health() -> bool:
    """Return True if Qdrant is reachable."""
    try:
        client = get_qdrant_client()
        await client.get_collections()
        return True
    except Exception:
        return False
