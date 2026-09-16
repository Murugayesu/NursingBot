from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

from app.config.settings import get_settings
from app.embeddings.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI-compatible embedding provider.
    Works with any API that speaks the OpenAI embeddings protocol
    (OpenAI, Azure, vLLM, etc.).
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        settings = get_settings()
        self._model_name = model_name
        self._client = AsyncOpenAI(
            api_key=api_key or settings.openai_api_key,
            base_url=base_url or settings.openai_base_url,
        )
        # Dimension depends on model; defaults for common models
        _DIMS = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self._dimension = _DIMS.get(model_name, 1536)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return "openai"

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self._model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]
