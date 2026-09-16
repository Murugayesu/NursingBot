from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChunkingStrategy(str, Enum):
    recursive = "recursive"
    markdown = "markdown"
    sentence = "sentence"


class FusionStrategy(str, Enum):
    rrf = "rrf"


class ChunkingConfig(BaseModel):
    strategy: ChunkingStrategy = ChunkingStrategy.recursive
    chunk_size: int = 800
    chunk_overlap: int = 100


class EmbeddingConfig(BaseModel):
    provider: str = "bge_m3"
    model: str = "BAAI/bge-m3"
    device: str = "cpu"
    batch_size: int = 32


class RetrievalConfig(BaseModel):
    dense: bool = True
    sparse: bool = True
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion: FusionStrategy = FusionStrategy.rrf


class RerankingConfig(BaseModel):
    enabled: bool = True
    provider: str = "jina"
    model: str = "jinaai/jina-reranker-v2-base-multilingual"
    device: str = "cpu"
    top_k: int = 5


class GenerationConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 2048
    max_context_tokens: int = 4096


class PromptConfig(BaseModel):
    version: str = "qa_v1"


class RAGConfig(BaseModel):
    """
    Per-knowledge-base configuration.
    Stored as JSON in PostgreSQL; controls every stage of the pipeline.
    """

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RAGConfig":
        return cls.model_validate(data)

    @classmethod
    def default(cls) -> "RAGConfig":
        return cls()
