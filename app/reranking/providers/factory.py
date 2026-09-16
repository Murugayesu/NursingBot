from __future__ import annotations

from functools import lru_cache

from app.config.settings import get_settings
from app.reranking.base import Reranker


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    settings = get_settings()
    provider = settings.reranker_provider.lower()

    if provider == "jina":
        from app.reranking.providers.jina_reranker import JinaReranker
        return JinaReranker(
            model_name=settings.reranker_model,
            device=settings.reranker_device,
        )
    elif provider == "cross_encoder":
        from app.reranking.providers.cross_encoder import CrossEncoderReranker
        return CrossEncoderReranker(device=settings.reranker_device)
    else:
        raise ValueError(f"Unknown reranker provider: '{provider}'. Supported: jina, cross_encoder")
