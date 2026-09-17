from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_env: Literal["development", "production"] = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── PostgreSQL ───────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "rag"
    postgres_password: str = "ragpassword"
    postgres_db: str = "ragdb"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        """Used by Alembic migrations (sync driver)."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_http_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_api_key: str | None = None

    # ── LLM (OpenAI-compatible) ──────────────────────────────────────────────
    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048

    # ── Embeddings (BGE-M3 via HuggingFaceEmbeddings) ────────────────────────
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"

    # ── Reranker (Jina local via transformers) ────────────────────────────────
    reranker_model: str = "jinaai/jina-reranker-v2-base-multilingual"
    reranker_device: str = "cpu"

    # ── Langfuse ─────────────────────────────────────────────────────────────
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ── Retrieval defaults ───────────────────────────────────────────────────
    default_dense_top_k: int = 20
    default_sparse_top_k: int = 20
    default_reranker_top_k: int = 5
    default_chunk_size: int = 800
    default_chunk_overlap: int = 100

    @field_validator("log_level")
    @classmethod
    def upper_log_level(cls, v: str) -> str:
        return v.upper()


@lru_cache
def get_settings() -> Settings:
    return Settings()
