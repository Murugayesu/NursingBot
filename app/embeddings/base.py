from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """
    Abstract embedding provider.
    All implementations must be thread/async-safe for batch processing.
    """

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts."""
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query (may use a different instruction prefix)."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier (e.g. 'bge_m3', 'openai')."""
        ...
