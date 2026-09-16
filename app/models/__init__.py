"""
Central import point — ensures all ORM models are registered with the
Base metadata before Alembic autogenerate runs.
"""
from app.models.base import Base  # noqa: F401
from app.models.chunk import DocumentChunk  # noqa: F401
from app.models.document import Document, DocumentStatus  # noqa: F401
from app.models.document_version import DocumentVersion  # noqa: F401
from app.models.evaluation import (  # noqa: F401
    EvaluationDataset,
    EvaluationQuestion,
    EvaluationRun,
)
from app.models.ingestion_job import IngestionJob, IngestionJobStatus  # noqa: F401
from app.models.knowledge_base import KnowledgeBase  # noqa: F401

__all__ = [
    "Base",
    "KnowledgeBase",
    "Document",
    "DocumentStatus",
    "DocumentVersion",
    "DocumentChunk",
    "IngestionJob",
    "IngestionJobStatus",
    "EvaluationDataset",
    "EvaluationQuestion",
    "EvaluationRun",
]
