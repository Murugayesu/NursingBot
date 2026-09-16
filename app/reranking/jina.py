"""
Jina Reranker v2 — local model, no API key required.
Model: jinaai/jina-reranker-v2-base-multilingual
Loaded via transformers with trust_remote_code=True.

Single concrete implementation — no base class, no factory.
"""
from __future__ import annotations

import asyncio
from typing import Any

import structlog

from app.retrieval.filters import RetrievedChunk

logger = structlog.get_logger(__name__)

_model: Any = None


def _load_model(model_name: str, device: str) -> Any:
    global _model
    if _model is None:
        from transformers import AutoModelForSequenceClassification
        logger.info("loading jina reranker", model=model_name, device=device)
        _model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        _model.to(device)
        _model.eval()
    return _model


def _rerank_sync(
    query: str,
    texts: list[str],
    top_k: int,
    model_name: str,
    device: str,
) -> list[tuple[int, float]]:
    import torch
    model = _load_model(model_name, device)
    pairs = [[query, text] for text in texts]
    with torch.no_grad():
        scores = model.compute_score(pairs, max_length=1024)
    if isinstance(scores, (int, float)):
        scores = [scores]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


async def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int,
    model_name: str,
    device: str,
) -> list[RetrievedChunk]:
    """Rerank chunks by cross-encoder relevance score."""
    if not chunks:
        return []
    texts = [c.text for c in chunks]
    loop = asyncio.get_event_loop()
    ranked = await loop.run_in_executor(
        None, _rerank_sync, query, texts, top_k, model_name, device
    )
    result: list[RetrievedChunk] = []
    for orig_idx, score in ranked:
        c = chunks[orig_idx]
        result.append(RetrievedChunk(
            point_id=c.point_id,
            text=c.text,
            score=float(score),
            document_id=c.document_id,
            knowledge_base_id=c.knowledge_base_id,
            chunk_index=c.chunk_index,
            title=c.title,
            source_url=c.source_url,
            category=c.category,
            document_type=c.document_type,
            extra_payload=c.extra_payload,
        ))
    return result
