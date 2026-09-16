from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.document import Document
    from app.models.ingestion_job import IngestionJob


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # RAGConfig stored as JSON (validated at app layer by Pydantic)
    rag_config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # Relationships
    documents: Mapped[list["Document"]] = relationship(
        "Document", back_populates="knowledge_base", cascade="all, delete-orphan"
    )
    ingestion_jobs: Mapped[list["IngestionJob"]] = relationship(
        "IngestionJob", back_populates="knowledge_base", cascade="all, delete-orphan"
    )
