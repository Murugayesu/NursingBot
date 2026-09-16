from __future__ import annotations

import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    VectorsConfig,
)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def collection_name(knowledge_base_id: uuid.UUID) -> str:
    """Deterministic Qdrant collection name for a knowledge base."""
    return f"kb_{knowledge_base_id.hex}"


async def ensure_collection(
    client: AsyncQdrantClient,
    knowledge_base_id: uuid.UUID,
    dense_dim: int = 1024,
) -> str:
    """
    Create a Qdrant collection for the knowledge base if it does not exist.
    Returns the collection name.
    """
    name = collection_name(knowledge_base_id)

    existing = {c.name for c in (await client.get_collections()).collections}
    if name in existing:
        return name

    await client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=dense_dim,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )
    return name


async def delete_collection(
    client: AsyncQdrantClient,
    knowledge_base_id: uuid.UUID,
) -> None:
    name = collection_name(knowledge_base_id)
    await client.delete_collection(name)
