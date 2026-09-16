from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from app.models.document import DocumentStatus
from app.models.ingestion_job import IngestionJobStatus


class DocumentUploadMetadata(BaseModel):
    """Optional metadata supplied at upload time."""
    title: str | None = None
    source_url: str | None = None
    category: str | None = None
    document_type: str | None = None


class DocumentResponse(BaseModel):
    id: uuid.UUID
    knowledge_base_id: uuid.UUID
    filename: str
    original_filename: str
    mime_type: str
    file_size: int
    status: DocumentStatus
    content_hash: str | None
    current_version: int
    error_message: str | None
    retry_count: int
    title: str | None
    source_url: str | None
    category: str | None
    document_type: str | None

    model_config = {"from_attributes": True}


class IngestionJobResponse(BaseModel):
    id: uuid.UUID
    knowledge_base_id: uuid.UUID
    status: IngestionJobStatus
    documents_total: int
    documents_processed: int
    documents_skipped: int
    documents_failed: int
    chunks_created: int
    error_message: str | None

    model_config = {"from_attributes": True}
