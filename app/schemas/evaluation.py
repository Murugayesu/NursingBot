from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from app.models.evaluation import EvaluationRunStatus


class EvaluationDatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    questions: list[dict[str, Any]] = Field(
        ...,
        description="List of {question, ground_truth, expected_context} dicts",
    )


class EvaluationRunCreate(BaseModel):
    dataset_id: uuid.UUID
    rag_config_override: dict[str, Any] | None = None


class EvaluationRunResponse(BaseModel):
    id: uuid.UUID
    knowledge_base_id: uuid.UUID
    dataset_id: uuid.UUID | None
    status: EvaluationRunStatus
    context_precision: float | None
    context_recall: float | None
    faithfulness: float | None
    answer_relevance: float | None
    answer_correctness: float | None
    recall_at_k: float | None
    mrr: float | None
    ndcg: float | None
    error_message: str | None

    model_config = {"from_attributes": True}
