from __future__ import annotations

from functools import lru_cache

from app.config.settings import get_settings
from app.embeddings.base import EmbeddingProvider


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """
    Module-level singleton factory.
    Returns the configured embedding provider.
    """
    settings = get_settings()
    provider = settings.embedding_provider.lower()

    if provider == "bge_m3":
        from app.embeddings.providers.bge_m3_provider import BGEM3Provider
        return BGEM3Provider(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
        )
    elif provider == "openai":
        from app.embeddings.providers.openai_provider import OpenAIEmbeddingProvider
        return OpenAIEmbeddingProvider(model_name=settings.embedding_model)
    else:
        raise ValueError(
            f"Unknown embedding provider: '{provider}'. "
            f"Supported: bge_m3, openai"
        )
