from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

from app.embeddings.base import EmbeddingProvider


class BGEM3Provider(EmbeddingProvider):
    """
    Local embedding provider using BAAI/bge-m3 via FlagEmbedding.

    BGE-M3 is a multi-lingual, multi-granularity model that produces:
    - Dense vectors (1024-dim) for semantic search
    - Sparse vectors (colbert/lexical) — used separately for Qdrant sparse

    This provider handles the dense embeddings.
    Sparse vectors are generated at index time via fastembed.
    """

    _instance: "BGEM3Provider | None" = None
    _model: Any = None

    def __new__(cls, model_name: str = "BAAI/bge-m3", device: str = "cpu") -> "BGEM3Provider":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu") -> None:
        if self._model is not None:
            return  # already initialised (singleton)
        from FlagEmbedding import BGEM3FlagModel
        self._model_name = model_name
        self._device = device
        self._model = BGEM3FlagModel(model_name, use_fp16=device != "cpu")

    @property
    def dimension(self) -> int:
        return 1024

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return "bge_m3"

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                texts,
                batch_size=12,
                max_length=8192,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            ),
        )
        return result["dense_vecs"].tolist()

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]
