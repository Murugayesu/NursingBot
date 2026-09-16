from __future__ import annotations

import uuid
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ENUM as PgEnum
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.knowledge_base import KnowledgeBase


class IngestionJobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIALLY_FAILED = "PARTIALLY_FAILED"
    FAILED = "FAILED"


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    status: Mapped[IngestionJobStatus] = mapped_column(
        PgEnum(IngestionJobStatus, name="ingestionjobstatus", create_type=True),
        nullable=False,
        default=IngestionJobStatus.PENDING,
        index=True,
    )

    documents_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    documents_processed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    documents_skipped: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    documents_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunks_created: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # IDs of documents included in this job
    document_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    knowledge_base: Mapped["KnowledgeBase"] = relationship(
        "KnowledgeBase", back_populates="ingestion_jobs"
    )
