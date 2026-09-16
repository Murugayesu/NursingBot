from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from app.config.rag_config import RAGConfig


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    tenant_id: str = Field(default="default", min_length=1, max_length=255)
    rag_config: dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    rag_config: dict[str, Any] | None = None


class KnowledgeBaseResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    tenant_id: str
    rag_config: dict[str, Any]

    model_config = {"from_attributes": True}
