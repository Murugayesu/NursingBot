from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import NamedVector, SearchRequest

from app.retrieval.filters import RetrievedChunk, build_qdrant_filter
from app.storage.qdrant.collections import DENSE_VECTOR_NAME, collection_name


class DenseRetriever:
    """
    Semantic retrieval using dense embeddings against Qdrant.
    """

    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client

    async def retrieve(
        self,
        query_vector: list[float],
        knowledge_base_id: uuid.UUID,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        coll = collection_name(knowledge_base_id)

        qdrant_filter = None
        if filters:
            qdrant_filter = build_qdrant_filter(filters)

        results = await self._client.search(
            collection_name=coll,
            query_vector=NamedVector(name=DENSE_VECTOR_NAME, vector=query_vector),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        chunks: list[RetrievedChunk] = []
        for hit in results:
            payload = hit.payload or {}
            chunks.append(
                RetrievedChunk(
                    point_id=str(hit.id),
                    text=payload.get("text", ""),
                    score=hit.score,
                    document_id=payload.get("document_id", ""),
                    knowledge_base_id=payload.get("knowledge_base_id", ""),
                    chunk_index=payload.get("chunk_index", 0),
                    title=payload.get("title", ""),
                    source_url=payload.get("source_url"),
                    category=payload.get("category"),
                    document_type=payload.get("document_type"),
                    extra_payload={
                        k: v
                        for k, v in payload.items()
                        if k not in {
                            "text", "document_id", "knowledge_base_id",
                            "chunk_index", "title", "source_url", "category",
                            "document_type",
                        }
                    },
                )
            )
        return chunks
