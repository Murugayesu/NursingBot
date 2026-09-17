"""
Local Jina Reranker v2 implemented as a LangChain BaseDocumentCompressor.

Uses jinaai/jina-reranker-v2-base-multilingual loaded via transformers
(no API key — runs fully offline).

Plugs directly into any LangChain retrieval chain via compress_documents().
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional, Sequence

import structlog
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import Field

logger = structlog.get_logger(__name__)

_model_cache: dict[str, Any] = {}


def _load_model(model_name: str, device: str) -> Any:
    key = f"{model_name}:{device}"
    if key not in _model_cache:
        from transformers import AutoModelForSequenceClassification
        logger.info("loading jina reranker", model=model_name, device=device)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()
        _model_cache[key] = model
    return _model_cache[key]


def _score_sync(
    query: str,
    texts: list[str],
    model_name: str,
    device: str,
) -> list[float]:
    import torch
    model = _load_model(model_name, device)
    pairs = [[query, t] for t in texts]
    with torch.no_grad():
        scores = model.compute_score(pairs, max_length=1024)
    if isinstance(scores, (int, float)):
        scores = [float(scores)]
    return [float(s) for s in scores]


class LocalJinaReranker(BaseDocumentCompressor):
    """
    LangChain BaseDocumentCompressor wrapping the local Jina Reranker v2.

    Compatible with any LangChain retrieval pipeline.
    top_n documents are returned, sorted by reranker score descending.
    """

    model_name: str = Field(default="jinaai/jina-reranker-v2-base-multilingual")
    device: str = Field(default="cpu")
    top_n: int = Field(default=5)

    model_config = {"arbitrary_types_allowed": True}

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> list[Document]:
        if not documents:
            return []
        texts = [d.page_content for d in documents]
        scores = _score_sync(query, texts, self.model_name, self.device)
        ranked = sorted(
            zip(documents, scores), key=lambda x: x[1], reverse=True
        )
        result = []
        for doc, score in ranked[: self.top_n]:
            # Inject reranker score into metadata for downstream use
            enriched = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "reranker_score": score},
            )
            result.append(enriched)
        return result

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> list[Document]:
        """Async wrapper — runs sync reranking in a thread executor."""
        if not documents:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.compress_documents, documents, query, callbacks
        )
