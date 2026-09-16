from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config.rag_config import RAGConfig, ChunkingStrategy
from app.embeddings import bge_m3
from app.ingestion.chunkers.recursive_chunker import RecursiveChunker
from app.ingestion.chunkers.markdown_chunker import MarkdownChunker
from app.ingestion.cleaners.text_cleaner import TextCleaner
from app.ingestion.loaders.factory import get_loader
from app.ingestion.parsers.text_parser import TextParser
from app.models.document import Document, DocumentStatus
from app.models.document_version import DocumentVersion
from app.models.chunk import DocumentChunk
from app.storage.qdrant.client import get_qdrant_client
from app.storage.qdrant.indexer import ChunkToIndex, QdrantIndexer

logger = structlog.get_logger(__name__)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def _set_status(
    session: AsyncSession,
    document: Document,
    status: DocumentStatus,
    error: str | None = None,
    failed_stage: str | None = None,
) -> None:
    document.status = status
    if error:
        document.error_message = error
        document.failed_stage = failed_stage
    await session.commit()


class IngestionPipeline:
    """
    Orchestrates the full ingestion pipeline for a single document:
    Load → Parse → Clean → Chunk → Embed → Index → Persist

    Checkpoints document status in PostgreSQL at each stage.
    A single bad document raises an exception (caught by the job runner).
    """

    def __init__(self) -> None:
        self._parser = TextParser()
        self._cleaner = TextCleaner()

    async def run(
        self,
        session: AsyncSession,
        document: Document,
        file_path: Path,
        rag_config: RAGConfig,
        knowledge_base_id: uuid.UUID,
        tenant_id: str,
    ) -> DocumentVersion:
        log = logger.bind(document_id=str(document.id), filename=document.filename)

        try:
            # ── Stage 1: Load ────────────────────────────────────────────────
            await _set_status(session, document, DocumentStatus.EXTRACTING)
            log.info("loading document")
            loader = get_loader(file_path, document.mime_type)
            raw = await loader.load(file_path)

            # ── Stage 2: Parse ───────────────────────────────────────────────
            parsed = self._parser.parse(raw)

            # ── Stage 3: Clean ───────────────────────────────────────────────
            await _set_status(session, document, DocumentStatus.CLEANING)
            log.info("cleaning document")
            cleaned = self._cleaner.clean(parsed)

            # ── Deduplication via content hash ───────────────────────────────
            content_hash = _content_hash(cleaned.text)

            # Check if this exact version already exists
            existing_version = await session.scalar(
                select(DocumentVersion).where(
                    DocumentVersion.document_id == document.id,
                    DocumentVersion.content_hash == content_hash,
                )
            )
            if existing_version is not None:
                log.info("document unchanged, skipping re-indexing", hash=content_hash)
                document.status = DocumentStatus.INDEXED
                await session.commit()
                return existing_version

            # New version number
            version_number = document.current_version + 1 if document.content_hash else 1
            document.content_hash = content_hash
            document.current_version = version_number
            await _set_status(session, document, DocumentStatus.CLEANED)

            # Persist version
            doc_version = DocumentVersion(
                document_id=document.id,
                version_number=version_number,
                content_hash=content_hash,
                raw_text=parsed.text,
                cleaned_text=cleaned.text,
                token_count=len(cleaned.text) // 4,
            )
            session.add(doc_version)
            await session.flush()  # get doc_version.id

            # ── Stage 4: Chunk ───────────────────────────────────────────────
            await _set_status(session, document, DocumentStatus.CHUNKING)
            log.info("chunking document")
            # Use MarkdownChunker for .md files, RecursiveChunker for everything else
            if document.mime_type in ("text/markdown", "text/x-markdown"):
                chunker = MarkdownChunker(
                    chunk_size=rag_config.chunking.chunk_size,
                    chunk_overlap=rag_config.chunking.chunk_overlap,
                )
            else:
                chunker = RecursiveChunker(
                    chunk_size=rag_config.chunking.chunk_size,
                    chunk_overlap=rag_config.chunking.chunk_overlap,
                )
            chunk_results = chunker.chunk(
                cleaned.text,
                metadata={
                    "document_id": str(document.id),
                    "knowledge_base_id": str(knowledge_base_id),
                    "tenant_id": tenant_id,
                    "title": cleaned.title or document.filename,
                    "source_url": document.source_url,
                    "category": document.category,
                    "document_type": document.document_type,
                },
            )
            await _set_status(session, document, DocumentStatus.CHUNKED)

            # ── Stage 5: Embed ───────────────────────────────────────────────
            await _set_status(session, document, DocumentStatus.EMBEDDING)
            log.info("embedding chunks", chunk_count=len(chunk_results))
            texts = [c.text for c in chunk_results]
            dense_vectors = await bge_m3.embed_documents(
                texts,
                model_name=rag_config.embedding.model,
                device=rag_config.embedding.device,
            )

            doc_version.embedding_provider = "bge_m3"
            doc_version.embedding_model = rag_config.embedding.model
            doc_version.embedding_dimension = bge_m3.DIMENSION
            await _set_status(session, document, DocumentStatus.EMBEDDED)

            # ── Stage 6: Index into Qdrant ───────────────────────────────────
            await _set_status(session, document, DocumentStatus.INDEXING)
            log.info("indexing into qdrant")
            qdrant_client = get_qdrant_client()
            indexer = QdrantIndexer(qdrant_client)

            db_chunks: list[DocumentChunk] = []
            chunks_to_index: list[ChunkToIndex] = []

            for chunk_result, dense_vec in zip(chunk_results, dense_vectors):
                point_id = str(uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"{document.id}:{version_number}:{chunk_result.chunk_index}",
                ))
                payload = {
                    **chunk_result.metadata,
                    "document_version_id": str(doc_version.id),
                    "chunk_index": chunk_result.chunk_index,
                }
                db_chunk = DocumentChunk(
                    document_id=document.id,
                    document_version_id=doc_version.id,
                    knowledge_base_id=knowledge_base_id,
                    chunk_index=chunk_result.chunk_index,
                    text=chunk_result.text,
                    token_count=chunk_result.token_count,
                    qdrant_point_id=point_id,
                    chunk_metadata=payload,
                )
                db_chunks.append(db_chunk)
                chunks_to_index.append(
                    ChunkToIndex(
                        point_id=point_id,
                        text=chunk_result.text,
                        dense_vector=dense_vec,
                        payload=payload,
                    )
                )

            indexed = await indexer.index_chunks(
                knowledge_base_id=knowledge_base_id,
                chunks=chunks_to_index,
                dense_dim=embedder.dimension,
            )

            # ── Persist chunks to PostgreSQL ─────────────────────────────────
            session.add_all(db_chunks)
            doc_version.chunk_count = len(db_chunks)
            document.status = DocumentStatus.INDEXED
            await session.commit()

            log.info("ingestion complete", chunks=indexed)
            return doc_version

        except Exception as exc:
            log.error("ingestion failed", error=str(exc))
            document.retry_count += 1
            await _set_status(
                session,
                document,
                DocumentStatus.FAILED,
                error=str(exc),
                failed_stage=document.status.value,
            )
            raise
