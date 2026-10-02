"""
RAG retriever built on LangChain primitives.

Provides a factory that returns a configured retriever combining:
- Dense vector search via LangChain's Qdrant.as_retriever()
- Sparse SPLADE search via a custom BaseRetriever wrapping fastembed
- RRF fusion of dense + sparse results
- Optional Jina reranking via LocalJinaReranker.acompress_documents()

The returned retriever is a plain LangChain BaseRetriever and can be
used directly in LCEL chains or invoked standalone.
"""
from __future__ import annotations

import uuid
from typing import Any

import structlog
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from app.retrieval.filters import RetrievedChunk, build_qdrant_filter

logger = structlog.get_logger(__name__)


# ── Sparse model cache (#10) ──────────────────────────────────────────────────
_sparse_model_cache: dict[str, Any] = {}


def _get_sparse_model(model_name: str = "prithivida/Splade_PP_en_v1") -> Any:
    """Return a cached SparseTextEmbedding instance (load-once pattern)."""
    if model_name not in _sparse_model_cache:
        from fastembed import SparseTextEmbedding
        logger.info("loading sparse embedding model", model=model_name)
        _sparse_model_cache[model_name] = SparseTextEmbedding(model_name=model_name)
    return _sparse_model_cache[model_name]


# ── Sparse retriever (SPLADE via fastembed) ───────────────────────────────────

class SparseQdrantRetriever(BaseRetriever):
    """
    LangChain BaseRetriever backed by Qdrant sparse (SPLADE) vectors.
    Uses fastembed to generate the sparse query vector.
    """

    knowledge_base_id: uuid.UUID
    top_k: int = 20
    filters: dict[str, Any] = Field(default_factory=dict)
    model_name: str = "prithivida/Splade_PP_en_v1"

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        raise NotImplementedError("Use async ainvoke instead.")

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        from fastembed import SparseTextEmbedding
        from qdrant_client import models

        from app.storage.qdrant.client import get_qdrant_client
        from app.storage.qdrant.collections import (
            SPARSE_VECTOR_NAME,
            collection_name,
        )

        # #10: cache sparse model — same pattern as dense embedder and reranker
        model = _get_sparse_model(self.model_name)
        embedding = list(model.embed([query]))[0]
        sparse_vec = models.SparseVector(
            indices=embedding.indices.tolist(),
            values=embedding.values.tolist(),
        )

        qdrant_filter = build_qdrant_filter(self.filters) if self.filters else None
        client = get_qdrant_client()
        if hasattr(client, "query_points"):
            response = await client.query_points(
                collection_name=collection_name(self.knowledge_base_id),
                query=sparse_vec,
                using=SPARSE_VECTOR_NAME,
                limit=self.top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            hits = response.points
        else:
            hits = await client.search(
                collection_name=collection_name(self.knowledge_base_id),
                query_vector=models.NamedSparseVector(name=SPARSE_VECTOR_NAME, vector=sparse_vec),
                limit=self.top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
        docs: list[Document] = []
        for hit in hits:
            payload = hit.payload or {}
            text = payload.pop("page_content", payload.pop("text", ""))
            docs.append(Document(page_content=text, metadata={**payload, "score": hit.score}))
        return docs


# ── RRF fusion ────────────────────────────────────────────────────────────────

def _rrf_fuse(
    result_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """Reciprocal Rank Fusion across multiple Document lists."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            key = doc.page_content[:200]  # use content as dedup key
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in docs:
                docs[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docs[key] for key, _ in ranked]


# ── Hybrid retriever ──────────────────────────────────────────────────────────

class HybridRetriever(BaseRetriever):
    """
    LangChain BaseRetriever combining dense + sparse search with RRF fusion.

    dense_retriever:  LangChain Qdrant.as_retriever() instance
    sparse_retriever: SparseQdrantRetriever instance
    """

    dense_retriever: Any  # VectorStoreRetriever from langchain_community
    sparse_retriever: SparseQdrantRetriever
    use_sparse: bool = True

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        raise NotImplementedError("Use async ainvoke instead.")

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        import asyncio

        tasks = [self.dense_retriever.ainvoke(query)]
        if self.use_sparse:
            tasks.append(self.sparse_retriever.ainvoke(query))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid = [r for r in results if isinstance(r, list)]

        if len(valid) == 1:
            return valid[0]
        return _rrf_fuse(valid)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_retriever(
    knowledge_base_id: uuid.UUID,
    tenant_id: str,
    embeddings,
    dense_top_k: int = 20,
    sparse_top_k: int = 20,
    use_sparse: bool = True,
    extra_filters: dict[str, Any] | None = None,
) -> HybridRetriever:
    """
    Build and return a HybridRetriever for the given knowledge base.
    Qdrant filter always enforces tenant_id + knowledge_base_id scope.
    """
    from app.storage.qdrant.vector_store import get_vector_store
    from app.retrieval.filters import build_qdrant_filter

    base_filters: dict[str, Any] = {
        "tenant_id": tenant_id,
        "knowledge_base_id": str(knowledge_base_id),
    }
    if extra_filters:
        base_filters.update(extra_filters)

    qdrant_filter = build_qdrant_filter(base_filters)

    vs = get_vector_store(knowledge_base_id, embeddings)
    dense_retriever = vs.as_retriever(
        search_kwargs={"k": dense_top_k, "filter": qdrant_filter}
    )

    sparse_retriever = SparseQdrantRetriever(
        knowledge_base_id=knowledge_base_id,
        top_k=sparse_top_k,
        filters=base_filters,
    )

    return HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        use_sparse=use_sparse,
    )
