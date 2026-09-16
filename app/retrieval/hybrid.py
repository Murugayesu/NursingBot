from __future__ import annotations

import uuid
from typing import Any

from app.config.rag_config import RAGConfig
from app.embeddings.base import EmbeddingProvider
from app.retrieval.dense import DenseRetriever
from app.retrieval.filters import RetrievedChunk
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.sparse import SparseRetriever
from app.storage.qdrant.client import get_qdrant_client


class HybridRetriever:
    """
    Orchestrates dense + sparse retrieval and fuses results with RRF.
    Enforces tenant_id and knowledge_base_id filters server-side.
    """

    def __init__(self) -> None:
        client = get_qdrant_client()
        self._dense = DenseRetriever(client)
        self._sparse = SparseRetriever(client)

    async def retrieve(
        self,
        query: str,
        query_vector: list[float],
        knowledge_base_id: uuid.UUID,
        tenant_id: str,
        rag_config: RAGConfig,
        extra_filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        # Always enforce tenant + KB scope server-side
        base_filters: dict[str, Any] = {
            "tenant_id": tenant_id,
            "knowledge_base_id": str(knowledge_base_id),
        }
        if extra_filters:
            base_filters.update(extra_filters)

        result_lists: list[list[RetrievedChunk]] = []

        if rag_config.retrieval.dense:
            dense_results = await self._dense.retrieve(
                query_vector=query_vector,
                knowledge_base_id=knowledge_base_id,
                top_k=rag_config.retrieval.dense_top_k,
                filters=base_filters,
            )
            result_lists.append(dense_results)

        if rag_config.retrieval.sparse:
            sparse_results = await self._sparse.retrieve(
                query=query,
                knowledge_base_id=knowledge_base_id,
                top_k=rag_config.retrieval.sparse_top_k,
                filters=base_filters,
            )
            result_lists.append(sparse_results)

        if not result_lists:
            return []

        if len(result_lists) == 1:
            return result_lists[0]

        return reciprocal_rank_fusion(result_lists)
