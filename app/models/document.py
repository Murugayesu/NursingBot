from __future__ import annotations

import uuid
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ENUM as PgEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.document_version import DocumentVersion
    from app.models.knowledge_base import KnowledgeBase


class DocumentStatus(str, Enum):
    UPLOADED = "UPLOADED"
    EXTRACTING = "EXTRACTING"
    EXTRACTED = "EXTRACTED"
    CLEANING = "CLEANING"
    CLEANED = "CLEANED"
    CHUNKING = "CHUNKING"
    CHUNKED = "CHUNKED"
    EMBEDDING = "EMBEDDING"
    EMBEDDED = "EMBEDDED"
    INDEXING = "INDEXING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"


class Document(Base):
    __tablename__ = "documents"

    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Latest content hash (SHA-256 hex); drives deduplication
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    status: Mapped[DocumentStatus] = mapped_column(
        PgEnum(DocumentStatus, name="documentstatus", create_type=True),
        nullable=False,
        default=DocumentStatus.UPLOADED,
        index=True,
    )

    # Error tracking
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    failed_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Current active version
    current_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # Optional user-supplied metadata
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    category: Mapped[str | None] = mapped_column(String(255), nullable=True)
    document_type: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Relationships
    knowledge_base: Mapped["KnowledgeBase"] = relationship(
        "KnowledgeBase", back_populates="documents"
    )
    versions: Mapped[list["DocumentVersion"]] = relationship(
        "DocumentVersion", back_populates="document", cascade="all, delete-orphan"
    )
