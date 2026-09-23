from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import List

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.db import get_db
from app.config.rag_config import RAGConfig
from app.ingestion.pipeline import IngestionPipeline
from app.models.document import Document, DocumentStatus
from app.models.ingestion_job import IngestionJob, IngestionJobStatus
from app.models.knowledge_base import KnowledgeBase
from app.schemas.document import DocumentResponse, DocumentUploadMetadata, IngestionJobResponse
from app.schemas.knowledge_base import KnowledgeBaseCreate, KnowledgeBaseResponse

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])

# ── Upload safety ─────────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB per file

ALLOWED_MIME_TYPES: set[str] = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "text/html",
    "text/htm",
    "text/csv",
    "application/csv",
}

_MIME_TO_EXT: dict[str, str] = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/x-markdown": ".md",
    "text/html": ".html",
    "text/htm": ".html",
    "text/csv": ".csv",
    "application/csv": ".csv",
}


def _safe_file_path(doc_id: uuid.UUID, mime_type: str) -> Path:
    """
    Return a server-controlled path derived only from doc UUID and MIME type.
    Never uses the client-supplied filename — prevents path traversal.
    """
    ext = _MIME_TO_EXT.get(mime_type, ".bin")
    doc_dir = Path(tempfile.gettempdir()) / "rag_uploads" / str(doc_id)
    doc_dir.mkdir(parents=True, exist_ok=True)
    return doc_dir / f"upload{ext}"


# ── Knowledge Base CRUD ───────────────────────────────────────────────────────

@router.post("", response_model=KnowledgeBaseResponse, status_code=201)
async def create_knowledge_base(
    body: KnowledgeBaseCreate,
    db: AsyncSession = Depends(get_db),
):
    rag_cfg = RAGConfig.from_dict(body.rag_config) if body.rag_config else RAGConfig.default()
    kb = KnowledgeBase(
        name=body.name,
        description=body.description,
        tenant_id=body.tenant_id,
        rag_config=rag_cfg.to_dict(),
    )
    db.add(kb)
    await db.commit()
    await db.refresh(kb)
    return kb


@router.get("", response_model=list[KnowledgeBaseResponse])
async def list_knowledge_bases(
    tenant_id: str = "default",
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.tenant_id == tenant_id)
    )
    return result.scalars().all()


@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(kb_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


# ── Document Upload ───────────────────────────────────────────────────────────

@router.post("/{kb_id}/documents", response_model=list[DocumentResponse], status_code=201)
async def upload_documents(
    kb_id: uuid.UUID,
    files: List[UploadFile] = File(...),
    title: str | None = Form(None),
    source_url: str | None = Form(None),
    category: str | None = Form(None),
    document_type: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    documents: list[Document] = []
    for upload in files:
        # ── MIME type allowlist (#16) ─────────────────────────────────────────
        mime_type = upload.content_type or "application/octet-stream"
        if mime_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type '{mime_type}' for file "
                    f"'{upload.filename}'. Allowed types: {sorted(ALLOWED_MIME_TYPES)}"
                ),
            )

        # ── Size cap (#16) ────────────────────────────────────────────────────
        content = await upload.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File '{upload.filename}' exceeds the maximum allowed size "
                    f"of {MAX_UPLOAD_BYTES // (1024*1024)} MB."
                ),
            )

        # Store original name for display only; never used for the on-disk path
        original_filename = upload.filename or "unknown"
        doc = Document(
            knowledge_base_id=kb_id,
            filename=original_filename,
            original_filename=original_filename,
            mime_type=mime_type,
            file_size=len(content),
            status=DocumentStatus.UPLOADED,
            title=title,
            source_url=source_url,
            category=category,
            document_type=document_type,
        )
        db.add(doc)
        await db.flush()

        # Safe path: UUID + MIME-derived extension — no client input (#2)
        file_path = _safe_file_path(doc.id, mime_type)
        file_path.write_bytes(content)

        documents.append(doc)

    await db.commit()
    return documents



@router.get("/{kb_id}/documents", response_model=list[DocumentResponse])
async def list_documents(kb_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Document).where(Document.knowledge_base_id == kb_id)
    )
    return result.scalars().all()


# ── Ingestion Trigger ─────────────────────────────────────────────────────────

async def _run_ingestion_job(job_id: uuid.UUID, kb_id: uuid.UUID) -> None:
    """Background task: processes all UPLOADED documents in the job."""
    from app.storage.postgres.database import AsyncSessionFactory

    async with AsyncSessionFactory() as session:
        job = await session.get(IngestionJob, job_id)
        if not job:
            return

        job.status = IngestionJobStatus.RUNNING
        await session.commit()

        kb = await session.get(KnowledgeBase, kb_id)
        rag_config = RAGConfig.from_dict(kb.rag_config)
        pipeline = IngestionPipeline()

        doc_ids = [uuid.UUID(did) for did in job.document_ids]
        for doc_id in doc_ids:
            doc = await session.get(Document, doc_id)
            if not doc or doc.status == DocumentStatus.INDEXED:
                job.documents_skipped += 1
                continue

            file_path = Path(tempfile.gettempdir()) / "rag_uploads" / str(doc.id) / doc.filename
            try:
                version = await pipeline.run(
                    session=session,
                    document=doc,
                    file_path=file_path,
                    rag_config=rag_config,
                    knowledge_base_id=kb_id,
                    tenant_id=kb.tenant_id,
                )
                job.documents_processed += 1
                job.chunks_created += version.chunk_count
            except Exception as exc:
                logger.error("document ingestion failed", doc_id=str(doc_id), error=str(exc))
                job.documents_failed += 1

            await session.commit()

        # Final status
        if job.documents_failed == 0:
            job.status = IngestionJobStatus.COMPLETED
        elif job.documents_processed > 0:
            job.status = IngestionJobStatus.PARTIALLY_FAILED
        else:
            job.status = IngestionJobStatus.FAILED
        await session.commit()


@router.post("/{kb_id}/ingest", response_model=IngestionJobResponse, status_code=202)
async def trigger_ingestion(
    kb_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    # Collect all UPLOADED documents
    result = await db.execute(
        select(Document).where(
            Document.knowledge_base_id == kb_id,
            Document.status == DocumentStatus.UPLOADED,
        )
    )
    docs = result.scalars().all()
    if not docs:
        raise HTTPException(status_code=422, detail="No documents in UPLOADED state to ingest")

    job = IngestionJob(
        knowledge_base_id=kb_id,
        status=IngestionJobStatus.PENDING,
        documents_total=len(docs),
        document_ids=[str(d.id) for d in docs],
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    background_tasks.add_task(_run_ingestion_job, job.id, kb_id)
    return job
