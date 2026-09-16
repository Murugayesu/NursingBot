"""
BGE-M3 dense embeddings via FlagEmbedding.

Single concrete implementation — no base class, no factory.
Singleton pattern so the model loads once per process.
"""
from __future__ import annotations

import asyncio
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_model: Any = None


def _load_model(model_name: str, device: str) -> Any:
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel
        logger.info("loading bge-m3 model", model=model_name, device=device)
        _model = BGEM3FlagModel(model_name, use_fp16=device != "cpu")
    return _model


def _embed_sync(texts: list[str], model_name: str, device: str) -> list[list[float]]:
    model = _load_model(model_name, device)
    result = model.encode(
        texts,
        batch_size=12,
        max_length=8192,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return result["dense_vecs"].tolist()


async def embed_documents(texts: list[str], model_name: str, device: str) -> list[list[float]]:
    """Embed a batch of document texts using BGE-M3."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_sync, texts, model_name, device)


async def embed_query(text: str, model_name: str, device: str) -> list[float]:
    """Embed a single query string."""
    vectors = await embed_documents([text], model_name, device)
    return vectors[0]


DIMENSION = 1024  # BGE-M3 dense vector dimension
