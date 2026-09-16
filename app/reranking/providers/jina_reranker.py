from __future__ import annotations

import asyncio
from typing import Any

import structlog

from app.reranking.base import Reranker
from app.retrieval.filters import RetrievedChunk

logger = structlog.get_logger(__name__)

_model_cache: dict[str, Any] = {}


class JinaReranker(Reranker):
    """
    Local Jina Reranker v2 (jinaai/jina-reranker-v2-base-multilingual).
    Loaded via transformers with trust_remote_code=True.
    Runs in a thread executor to stay async-safe.
    """

    provider_name = "jina"

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model_name in _model_cache:
            return _model_cache[self._model_name]

        from transformers import AutoModelForSequenceClassification
        import torch

        logger.info("loading jina reranker", model=self._model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        model.to(self._device)
        model.eval()
        _model_cache[self._model_name] = model
        return model

    def _rerank_sync(
        self, query: str, texts: list[str], top_k: int
    ) -> list[tuple[int, float]]:
        """Synchronous rerank — called via run_in_executor."""
        import torch

        model = self._load_model()
        pairs = [[query, text] for text in texts]
        with torch.no_grad():
            scores = model.compute_score(pairs, max_length=1024)

        # scores is a list of floats
        if isinstance(scores, (int, float)):
            scores = [scores]

        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return indexed[:top_k]

    async def rerank(
        self,
        query: str,
        documents: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if not documents:
            return []

        texts = [doc.text for doc in documents]
        loop = asyncio.get_event_loop()
        ranked = await loop.run_in_executor(
            None, self._rerank_sync, query, texts, top_k
        )

        result: list[RetrievedChunk] = []
        for orig_idx, score in ranked:
            chunk = documents[orig_idx]
            result.append(
                RetrievedChunk(
                    point_id=chunk.point_id,
                    text=chunk.text,
                    score=float(score),
                    document_id=chunk.document_id,
                    knowledge_base_id=chunk.knowledge_base_id,
                    chunk_index=chunk.chunk_index,
                    title=chunk.title,
                    source_url=chunk.source_url,
                    category=chunk.category,
                    document_type=chunk.document_type,
                    extra_payload=chunk.extra_payload,
                )
            )
        return result
