from __future__ import annotations

import uuid
from enum import Enum

from sqlalchemy import ForeignKey, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import ENUM as PgEnum
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class EvaluationRunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class EvaluationDataset(Base):
    __tablename__ = "evaluation_datasets"

    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    question_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class EvaluationQuestion(Base):
    __tablename__ = "evaluation_questions"

    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    ground_truth: Mapped[str] = mapped_column(Text, nullable=False)
    expected_context: Mapped[str | None] = mapped_column(Text, nullable=True)
    extra_metadata: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_datasets.id", ondelete="SET NULL"),
        nullable=True,
    )

    # RAGConfig snapshot used for this run (for reproducibility)
    rag_config_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    status: Mapped[EvaluationRunStatus] = mapped_column(
        PgEnum(EvaluationRunStatus, name="evaluationrunstatus", create_type=True),
        nullable=False,
        default=EvaluationRunStatus.PENDING,
    )

    # RAGAS metrics
    context_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_relevance: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_correctness: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Retrieval metrics
    recall_at_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    mrr: Mapped[float | None] = mapped_column(Float, nullable=True)
    ndcg: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Full per-question results
    results: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
