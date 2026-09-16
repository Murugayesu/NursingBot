#!/usr/bin/env python3
"""
Rebuild the Qdrant vector index from PostgreSQL + stored text.

Usage:
    python scripts/rebuild_qdrant_index.py --kb-id <uuid>

This re-reads all INDEXED DocumentChunks for the knowledge base from
PostgreSQL, re-embeds their text, and upserts them into Qdrant.
Useful when Qdrant data is lost or corrupted.
"""
from __future__ import annotations

import argparse
import asyncio
import uuid

import structlog

logger = structlog.get_logger(__name__)


async def rebuild(kb_id: uuid.UUID) -> None:
    from sqlalchemy import select
    from app.embeddings.providers.factory import get_embedding_provider
    from app.models.chunk import DocumentChunk
    from app.storage.postgres.database import AsyncSessionFactory
    from app.storage.qdrant.client import get_qdrant_client
    from app.storage.qdrant.indexer import ChunkToIndex, QdrantIndexer
    from app.models.knowledge_base import KnowledgeBase

    async with AsyncSessionFactory() as session:
        kb = await session.get(KnowledgeBase, kb_id)
        if not kb:
            logger.error("knowledge base not found", kb_id=str(kb_id))
            return

        result = await session.execute(
            select(DocumentChunk).where(DocumentChunk.knowledge_base_id == kb_id)
        )
        chunks = result.scalars().all()
        logger.info("rebuilding index", kb_id=str(kb_id), chunk_count=len(chunks))

        embedder = get_embedding_provider()
        indexer = QdrantIndexer(get_qdrant_client())

        texts = [c.text for c in chunks]
        BATCH = 32
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            vecs = await embedder.embed_documents(texts[i : i + BATCH])
            all_vectors.extend(vecs)

        to_index = [
            ChunkToIndex(
                point_id=c.qdrant_point_id or str(c.id),
                text=c.text,
                dense_vector=vec,
                payload=c.chunk_metadata,
            )
            for c, vec in zip(chunks, all_vectors)
        ]

        indexed = await indexer.index_chunks(
            knowledge_base_id=kb_id,
            chunks=to_index,
            dense_dim=embedder.dimension,
        )
        logger.info("rebuild complete", indexed=indexed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild Qdrant index from PostgreSQL")
    parser.add_argument("--kb-id", required=True, help="Knowledge base UUID")
    args = parser.parse_args()
    asyncio.run(rebuild(uuid.UUID(args.kb_id)))
