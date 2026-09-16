"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-09-15

"""
from typing import Sequence, Union
import uuid

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON, UUID
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Enums ────────────────────────────────────────────────────────────────
    op.execute("""
        CREATE TYPE documentstatus AS ENUM (
            'UPLOADED','EXTRACTING','EXTRACTED','CLEANING','CLEANED',
            'CHUNKING','CHUNKED','EMBEDDING','EMBEDDED','INDEXING','INDEXED','FAILED'
        )
    """)
    op.execute("""
        CREATE TYPE ingestionjobstatus AS ENUM (
            'PENDING','RUNNING','COMPLETED','PARTIALLY_FAILED','FAILED'
        )
    """)
    op.execute("""
        CREATE TYPE evaluationrunstatus AS ENUM (
            'PENDING','RUNNING','COMPLETED','FAILED'
        )
    """)

    # ── knowledge_bases ──────────────────────────────────────────────────────
    op.create_table(
        "knowledge_bases",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("tenant_id", sa.String(255), nullable=False, index=True),
        sa.Column("rag_config", JSON, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── documents ────────────────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column(
            "knowledge_base_id",
            UUID(as_uuid=True),
            sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("original_filename", sa.String(512), nullable=False),
        sa.Column("mime_type", sa.String(128), nullable=False),
        sa.Column("file_size", sa.Integer, nullable=False, server_default="0"),
        sa.Column("content_hash", sa.String(64), nullable=True, index=True),
        sa.Column(
            "status",
            sa.Enum(
                "UPLOADED","EXTRACTING","EXTRACTED","CLEANING","CLEANED",
                "CHUNKING","CHUNKED","EMBEDDING","EMBEDDED","INDEXING","INDEXED","FAILED",
                name="documentstatus",
                create_type=False,
            ),
            nullable=False,
            server_default="UPLOADED",
            index=True,
        ),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("failed_stage", sa.String(64), nullable=True),
        sa.Column("retry_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("current_version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("title", sa.String(512), nullable=True),
        sa.Column("source_url", sa.String(2048), nullable=True),
        sa.Column("category", sa.String(255), nullable=True),
        sa.Column("document_type", sa.String(128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── document_versions ────────────────────────────────────────────────────
    op.create_table(
        "document_versions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column(
            "document_id",
            UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("version_number", sa.Integer, nullable=False, server_default="1"),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("raw_text", sa.Text, nullable=True),
        sa.Column("cleaned_text", sa.Text, nullable=True),
        sa.Column("token_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("chunk_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("embedding_provider", sa.String(128), nullable=True),
        sa.Column("embedding_model", sa.String(255), nullable=True),
        sa.Column("embedding_dimension", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── document_chunks ──────────────────────────────────────────────────────
    op.create_table(
        "document_chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column(
            "document_id",
            UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "document_version_id",
            UUID(as_uuid=True),
            sa.ForeignKey("document_versions.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "knowledge_base_id",
            UUID(as_uuid=True),
            sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("qdrant_point_id", sa.String(36), nullable=True, index=True),
        sa.Column("chunk_metadata", JSON, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── ingestion_jobs ───────────────────────────────────────────────────────
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column(
            "knowledge_base_id",
            UUID(as_uuid=True),
            sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING","RUNNING","COMPLETED","PARTIALLY_FAILED","FAILED",
                name="ingestionjobstatus",
                create_type=False,
            ),
            nullable=False,
            server_default="PENDING",
            index=True,
        ),
        sa.Column("documents_total", sa.Integer, nullable=False, server_default="0"),
        sa.Column("documents_processed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("documents_skipped", sa.Integer, nullable=False, server_default="0"),
        sa.Column("documents_failed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("chunks_created", sa.Integer, nullable=False, server_default="0"),
        sa.Column("document_ids", JSON, nullable=False, server_default="[]"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── evaluation_datasets ──────────────────────────────────────────────────
    op.create_table(
        "evaluation_datasets",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column(
            "knowledge_base_id",
            UUID(as_uuid=True),
            sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("question_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── evaluation_questions ─────────────────────────────────────────────────
    op.create_table(
        "evaluation_questions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column(
            "dataset_id",
            UUID(as_uuid=True),
            sa.ForeignKey("evaluation_datasets.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("ground_truth", sa.Text, nullable=False),
        sa.Column("expected_context", sa.Text, nullable=True),
        sa.Column("extra_metadata", JSON, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── evaluation_runs ──────────────────────────────────────────────────────
    op.create_table(
        "evaluation_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column(
            "knowledge_base_id",
            UUID(as_uuid=True),
            sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "dataset_id",
            UUID(as_uuid=True),
            sa.ForeignKey("evaluation_datasets.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("rag_config_snapshot", JSON, nullable=False, server_default="{}"),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING","RUNNING","COMPLETED","FAILED",
                name="evaluationrunstatus",
                create_type=False,
            ),
            nullable=False,
            server_default="PENDING",
        ),
        sa.Column("context_precision", sa.Float, nullable=True),
        sa.Column("context_recall", sa.Float, nullable=True),
        sa.Column("faithfulness", sa.Float, nullable=True),
        sa.Column("answer_relevance", sa.Float, nullable=True),
        sa.Column("answer_correctness", sa.Float, nullable=True),
        sa.Column("recall_at_k", sa.Float, nullable=True),
        sa.Column("mrr", sa.Float, nullable=True),
        sa.Column("ndcg", sa.Float, nullable=True),
        sa.Column("results", JSON, nullable=False, server_default="{}"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("evaluation_runs")
    op.drop_table("evaluation_questions")
    op.drop_table("evaluation_datasets")
    op.drop_table("ingestion_jobs")
    op.drop_table("document_chunks")
    op.drop_table("document_versions")
    op.drop_table("documents")
    op.drop_table("knowledge_bases")
    op.execute("DROP TYPE IF EXISTS evaluationrunstatus")
    op.execute("DROP TYPE IF EXISTS ingestionjobstatus")
    op.execute("DROP TYPE IF EXISTS documentstatus")
