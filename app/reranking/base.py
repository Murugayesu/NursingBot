from __future__ import annotations

from abc import ABC, abstractmethod

from app.retrieval.filters import RetrievedChunk


class Reranker(ABC):
    """Abstract reranker interface."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...
