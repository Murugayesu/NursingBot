from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.chunk import DocumentChunk
    from app.models.document import Document


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # Processing output
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    cleaned_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Embedding metadata for this version
    embedding_provider: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    embedding_dimension: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="versions")
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="document_version", cascade="all, delete-orphan"
    )
