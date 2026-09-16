from __future__ import annotations

import asyncio
from typing import Any

from app.reranking.base import Reranker
from app.retrieval.filters import RetrievedChunk

_model_cache: dict[str, Any] = {}


class CrossEncoderReranker(Reranker):
    """
    Local cross-encoder reranker via sentence-transformers.
    Good fallback that requires no extra dependencies.
    """

    provider_name = "cross_encoder"

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model_name in _model_cache:
            return _model_cache[self._model_name]
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(self._model_name, device=self._device)
        _model_cache[self._model_name] = model
        return model

    def _rerank_sync(self, query: str, texts: list[str], top_k: int) -> list[tuple[int, float]]:
        model = self._load_model()
        pairs = [(query, text) for text in texts]
        scores = model.predict(pairs)
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
