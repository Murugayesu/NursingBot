#!/usr/bin/env python3
"""
Rebuild the Qdrant vector index from PostgreSQL + stored text.

Usage:
    python scripts/rebuild_qdrant_index.py --kb-id <uuid>

This re-reads all DocumentChunks for the knowledge base from PostgreSQL,
re-embeds their text using the current embedding model, and upserts them
back into Qdrant. Run this when Qdrant data is lost or corrupted.

Prerequisites:
    - .env loaded (POSTGRES_*, QDRANT_*, EMBEDDING_MODEL)
    - Qdrant running and accessible
    - BGE-M3 model available locally
"""
from __future__ import annotations

import argparse
import asyncio
import uuid

import structlog

logger = structlog.get_logger(__name__)


async def rebuild(kb_id: uuid.UUID, batch_size: int = 32) -> None:
    from sqlalchemy import select

    from app.config.rag_config import RAGConfig
    from app.config.settings import get_settings
    from app.embeddings.lc_embeddings import get_embeddings
    from app.models.chunk import DocumentChunk
    from app.models.knowledge_base import KnowledgeBase
    from app.storage.postgres.database import AsyncSessionFactory
    from app.storage.qdrant.collections import ensure_collection
    from app.storage.qdrant.client import get_qdrant_client
    from app.storage.qdrant.vector_store import get_vector_store
    from langchain_core.documents import Document

    settings = get_settings()

    async with AsyncSessionFactory() as session:
        kb = await session.get(KnowledgeBase, kb_id)
        if not kb:
            logger.error("knowledge base not found", kb_id=str(kb_id))
            return

        rag_config = RAGConfig.from_dict(kb.rag_config)
        logger.info("rebuilding qdrant index", kb_id=str(kb_id), kb_name=kb.name)

        result = await session.execute(
            select(DocumentChunk).where(DocumentChunk.knowledge_base_id == kb_id)
        )
        chunks = result.scalars().all()
        logger.info("chunks to re-index", count=len(chunks))

        if not chunks:
            logger.info("no chunks found, nothing to do")
            return

        # Load embeddings model (cached singleton)
        embeddings = get_embeddings(
            model_name=rag_config.embedding.model,
            device=rag_config.embedding.device,
        )

        # Ensure Qdrant collection exists with correct dimensions
        await ensure_collection(
            client=get_qdrant_client(),
            knowledge_base_id=kb_id,
            dense_dim=1024,
        )

        vs = get_vector_store(kb_id, embeddings)

        # Re-index in batches using LangChain's aadd_documents
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            lc_docs = [
                Document(
                    page_content=c.text,
                    metadata={
                        **(c.chunk_metadata or {}),
                        "document_id": str(c.document_id),
                        "knowledge_base_id": str(c.knowledge_base_id),
                        "chunk_index": c.chunk_index,
                    },
                )
                for c in batch
            ]
            point_ids = [c.qdrant_point_id or str(c.id) for c in batch]
            await vs.aadd_documents(lc_docs, ids=point_ids)
            total += len(batch)
            logger.info("progress", indexed=total, total=len(chunks))

        logger.info("rebuild complete", kb_id=str(kb_id), indexed=total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild Qdrant index from PostgreSQL")
    parser.add_argument("--kb-id", required=True, help="Knowledge base UUID")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    args = parser.parse_args()
    asyncio.run(rebuild(uuid.UUID(args.kb_id), batch_size=args.batch_size))
