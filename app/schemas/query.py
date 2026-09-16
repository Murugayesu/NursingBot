from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=5, ge=1, le=50)
    include_sources: bool = True


class SourceResponse(BaseModel):
    index: int
    document_id: str
    title: str
    source_url: str | None
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceResponse]
    prompt_version: str
    model: str
    input_tokens: int
    output_tokens: int
