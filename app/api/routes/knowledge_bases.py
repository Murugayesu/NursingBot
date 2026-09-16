from __future__ import annotations

import io
import shutil
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
        filename = upload.filename or "unknown"
        content = await upload.read()
        doc = Document(
            knowledge_base_id=kb_id,
            filename=filename,
            original_filename=filename,
            mime_type=upload.content_type or "application/octet-stream",
            file_size=len(content),
            status=DocumentStatus.UPLOADED,
            title=title,
            source_url=source_url,
            category=category,
            document_type=document_type,
        )
        db.add(doc)
        await db.flush()

        # Save file to a temp directory named by doc ID (persists until ingestion)
        doc_dir = Path(tempfile.gettempdir()) / "rag_uploads" / str(doc.id)
        doc_dir.mkdir(parents=True, exist_ok=True)
        file_path = doc_dir / filename
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
