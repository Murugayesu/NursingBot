"""
Ingestion pipeline — LangChain-native implementation.

Flow per document:
    Load (LangChain loader)
    → split_documents (LangChain text splitter)
    → aadd_documents (LangChain QdrantVectorStore — embeds + indexes)
    → persist DocumentChunk records in PostgreSQL

PostgreSQL document status machine is preserved unchanged.
Content-hash deduplication prevents re-indexing unchanged documents.
"""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import structlog
from langchain_core.documents import Document
from qdrant_client import models
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.rag_config import RAGConfig
from app.embeddings.lc_embeddings import get_embeddings
from app.ingestion.loaders.lc_loaders import load_document
from app.models.chunk import DocumentChunk
from app.models.document import Document as DBDocument, DocumentStatus
from app.models.document_version import DocumentVersion
from app.storage.qdrant.collections import collection_name, ensure_collection
from app.storage.qdrant.client import get_qdrant_client
from app.storage.qdrant.vector_store import get_vector_store

logger = structlog.get_logger(__name__)

_MARKDOWN_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]


def _content_hash(docs: list[Document]) -> str:
    combined = "".join(d.page_content for d in docs)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _build_splitter(mime_type: str, cfg) -> RecursiveCharacterTextSplitter | MarkdownHeaderTextSplitter:
    if mime_type in ("text/markdown", "text/x-markdown"):
        # Two-stage: header split → recursive split handled in _split_docs
        return None  # signals markdown path
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )


def _split_docs(
    docs: list[Document],
    mime_type: str,
    cfg,
    base_metadata: dict,
) -> list[Document]:
    """Split documents and attach base_metadata to every chunk."""
    if mime_type in ("text/markdown", "text/x-markdown"):
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_MARKDOWN_HEADERS, strip_headers=False
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
        )
        header_docs = []
        for doc in docs:
            header_docs.extend(header_splitter.split_text(doc.page_content))
        chunks = text_splitter.split_documents(header_docs)
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        chunks = splitter.split_documents(docs)

    # Inject base metadata + chunk index into every chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata = {**base_metadata, **chunk.metadata, "chunk_index": i}

    return chunks


async def _set_status(
    session: AsyncSession,
    document: DBDocument,
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
    Orchestrates the full ingestion pipeline for a single document
    using LangChain primitives throughout.

    Stages:
      1. Load   — LangChain community loader (PyMuPDF, TextLoader, etc.)
      2. Split  — LangChain RecursiveCharacterTextSplitter / MarkdownHeaderTextSplitter
      3. Embed + Index — LangChain QdrantVectorStore.aadd_documents()
      4. Persist — SQLAlchemy DocumentChunk records

    PostgreSQL status checkpoints are maintained at each stage.
    Hash-based deduplication skips re-indexing unchanged documents.
    """

    async def run(
        self,
        session: AsyncSession,
        document: DBDocument,
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
            lc_docs = load_document(file_path, document.mime_type)

            # ── Stage 2: Hash for deduplication ─────────────────────────────
            content_hash = _content_hash(lc_docs)
            existing = await session.scalar(
                select(DocumentVersion).where(
                    DocumentVersion.document_id == document.id,
                    DocumentVersion.content_hash == content_hash,
                )
            )
            if existing is not None:
                log.info("document unchanged, skipping", hash=content_hash)
                document.status = DocumentStatus.INDEXED
                await session.commit()
                return existing

            version_number = document.current_version + 1 if document.content_hash else 1
            document.content_hash = content_hash
            document.current_version = version_number
            raw_text = "\n\n".join(d.page_content for d in lc_docs)
            await _set_status(session, document, DocumentStatus.CLEANING)

            doc_version = DocumentVersion(
                document_id=document.id,
                version_number=version_number,
                content_hash=content_hash,
                raw_text=raw_text,
                cleaned_text=raw_text,
                token_count=len(raw_text) // 4,
                embedding_provider="huggingface",
                embedding_model=rag_config.embedding.model,
                embedding_dimension=1024,
            )
            session.add(doc_version)
            await session.flush()

            # ── Stage 3: Split ───────────────────────────────────────────────
            await _set_status(session, document, DocumentStatus.CHUNKING)
            log.info("splitting document")
            base_metadata = {
                "document_id": str(document.id),
                "document_version_id": str(doc_version.id),
                "knowledge_base_id": str(knowledge_base_id),
                "tenant_id": tenant_id,
                "title": document.title or document.filename,
                "source_url": document.source_url,
                "category": document.category,
                "document_type": document.document_type,
            }
            chunks: list[Document] = _split_docs(
                lc_docs, document.mime_type, rag_config.chunking, base_metadata
            )
            await _set_status(session, document, DocumentStatus.CHUNKED)
            log.info("split complete", chunks=len(chunks))

            # ── Stage 4: Embed + Index (one call via LangChain) ───────────────
            await _set_status(session, document, DocumentStatus.EMBEDDING)
            log.info("embedding and indexing chunks")

            embeddings = get_embeddings(
                model_name=rag_config.embedding.model,
                device=rag_config.embedding.device,
            )
            # Ensure collection exists (with sparse vector support)
            await ensure_collection(
                client=get_qdrant_client(),
                knowledge_base_id=knowledge_base_id,
                dense_dim=1024,
            )
            vs = get_vector_store(knowledge_base_id, embeddings)

            # Generate deterministic point IDs
            point_ids = [
                str(uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"{document.id}:{version_number}:{c.metadata['chunk_index']}",
                ))
                for c in chunks
            ]
            # aadd_documents embeds and upserts in one batched call (dense only)
            await vs.aadd_documents(chunks, ids=point_ids)

            # ── Sparse vector upsert (SPLADE via fastembed) ───────────────────
            # LangChain's Qdrant wrapper only writes to the dense named vector.
            # We compute sparse vectors separately and upsert them onto the same
            # point IDs so hybrid search finds populated sparse fields.
            log.info("computing sparse vectors")
            from fastembed import SparseTextEmbedding
            from app.storage.qdrant.collections import SPARSE_VECTOR_NAME

            sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
            texts = [c.page_content for c in chunks]
            sparse_embeddings = list(sparse_model.embed(texts))

            qdrant_client = get_qdrant_client()
            coll_name = collection_name(knowledge_base_id)
            sparse_points = [
                models.PointVectors(
                    id=pid,
                    vectors={
                        SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=emb.indices.tolist(),
                            values=emb.values.tolist(),
                        )
                    },
                )
                for pid, emb in zip(point_ids, sparse_embeddings)
            ]
            await qdrant_client.update_vectors(
                collection_name=coll_name,
                points=sparse_points,
            )
            log.info("sparse vectors upserted", count=len(sparse_points))

            await _set_status(session, document, DocumentStatus.EMBEDDED)

            # ── Stage 5: Persist chunk records ───────────────────────────────
            await _set_status(session, document, DocumentStatus.INDEXING)
            db_chunks = [
                DocumentChunk(
                    document_id=document.id,
                    document_version_id=doc_version.id,
                    knowledge_base_id=knowledge_base_id,
                    chunk_index=c.metadata["chunk_index"],
                    text=c.page_content,
                    token_count=len(c.page_content) // 4,
                    qdrant_point_id=pid,
                    chunk_metadata=c.metadata,
                )
                for c, pid in zip(chunks, point_ids)
            ]
            session.add_all(db_chunks)
            doc_version.chunk_count = len(db_chunks)
            document.status = DocumentStatus.INDEXED
            await session.commit()

            log.info("ingestion complete", chunks=len(db_chunks))
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
