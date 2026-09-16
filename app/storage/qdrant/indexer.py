from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    NamedSparseVector,
    NamedVector,
    PointStruct,
    SparseVector,
)

from app.storage.qdrant.collections import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    collection_name,
    ensure_collection,
)

# Sparse model (SPLADE) — loaded once at module level
_sparse_model: SparseTextEmbedding | None = None


def _get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    return _sparse_model


def _make_sparse_vector(text: str) -> SparseVector:
    model = _get_sparse_model()
    results = list(model.embed([text]))
    embedding = results[0]
    return SparseVector(
        indices=embedding.indices.tolist(),
        values=embedding.values.tolist(),
    )


@dataclass
class ChunkToIndex:
    point_id: str
    text: str
    dense_vector: list[float]
    payload: dict[str, Any]


class QdrantIndexer:
    """
    Indexes document chunks into Qdrant with both dense and sparse vectors.
    """

    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client

    async def index_chunks(
        self,
        knowledge_base_id: uuid.UUID,
        chunks: list[ChunkToIndex],
        dense_dim: int = 1024,
        batch_size: int = 64,
    ) -> int:
        """
        Upsert chunks into Qdrant. Returns number of points indexed.
        Creates collection if needed.
        """
        coll = await ensure_collection(self._client, knowledge_base_id, dense_dim)

        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            points: list[PointStruct] = []

            for chunk in batch:
                sparse_vec = _make_sparse_vector(chunk.text)
                points.append(
                    PointStruct(
                        id=chunk.point_id,
                        vector={
                            DENSE_VECTOR_NAME: chunk.dense_vector,
                            SPARSE_VECTOR_NAME: sparse_vec,
                        },
                        payload={**chunk.payload, "text": chunk.text},
                    )
                )

            await self._client.upsert(collection_name=coll, points=points)
            total += len(points)

        return total

    async def delete_chunks(
        self,
        knowledge_base_id: uuid.UUID,
        point_ids: list[str],
    ) -> None:
        """Delete specific points from the collection."""
        from qdrant_client.models import PointIdsList
        coll = collection_name(knowledge_base_id)
        await self._client.delete(
            collection_name=coll,
            points_selector=PointIdsList(points=point_ids),
        )
