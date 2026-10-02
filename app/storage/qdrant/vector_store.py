"""
LangChain Qdrant vector store wrapper.

Provides a robust, native LangChain VectorStore backed directly by QdrantClient
(supporting both sync and async operations) without fragile dependencies
on external or deprecated LangChain provider wrappers.
"""
from __future__ import annotations

import asyncio
from typing import Any, Iterable, List, Optional
import uuid

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import AsyncQdrantClient, QdrantClient, models

from app.storage.qdrant.client import get_qdrant_client, get_sync_qdrant_client
from app.storage.qdrant.collections import DENSE_VECTOR_NAME, collection_name

CONTENT_KEY = "page_content"
METADATA_KEY = "metadata"


class NativeQdrantVectorStore(VectorStore):
    """
    Self-contained LangChain VectorStore for Qdrant.
    
    Implements aadd_documents and asimilarity_search using AsyncQdrantClient
    so it works asynchronously with FastAPI without blocking the event loop.
    """

    def __init__(
        self,
        collection_name: str,
        embeddings: Embeddings,
        client: Optional[QdrantClient] = None,
        async_client: Optional[AsyncQdrantClient] = None,
        vector_name: str = DENSE_VECTOR_NAME,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
    ):
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.client = client or get_sync_qdrant_client()
        self.async_client = async_client or get_qdrant_client()
        self.vector_name = vector_name
        self.content_payload_key = content_payload_key
        self.metadata_payload_key = metadata_payload_key

    async def aadd_documents(
        self,
        documents: list[Document],
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Asynchronously embed and upsert documents into Qdrant."""
        if not documents:
            return []

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        texts = [doc.page_content for doc in documents]
        if hasattr(self.embeddings, "aembed_documents"):
            dense_vectors = await self.embeddings.aembed_documents(texts)
        else:
            dense_vectors = await asyncio.to_thread(self.embeddings.embed_documents, texts)

        points = []
        for doc_id, doc, vec in zip(ids, documents, dense_vectors):
            payload = dict(doc.metadata or {})
            payload[self.content_payload_key] = doc.page_content
            payload[self.metadata_payload_key] = doc.metadata or {}
            points.append(
                models.PointStruct(
                    id=doc_id,
                    vector={self.vector_name: vec},
                    payload=payload,
                )
            )

        await self.async_client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        docs = [
            Document(
                page_content=t,
                metadata=metadatas[i] if metadatas and i < len(metadatas) else {},
            )
            for i, t in enumerate(texts)
        ]
        return asyncio.run(self.aadd_documents(docs, ids=ids, **kwargs))

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Any] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Asynchronously search for similar documents in Qdrant."""
        if hasattr(self.embeddings, "aembed_query"):
            query_vector = await self.embeddings.aembed_query(query)
        else:
            query_vector = await asyncio.to_thread(self.embeddings.embed_query, query)

        hits = await self.async_client.search(
            collection_name=self.collection_name,
            query_vector=models.NamedVector(
                name=self.vector_name,
                vector=query_vector,
            ),
            limit=k,
            query_filter=filter,
            with_payload=True,
        )

        documents: list[Document] = []
        for hit in hits:
            payload = hit.payload or {}
            content = payload.get(self.content_payload_key, payload.get("text", ""))
            meta = dict(payload.get(self.metadata_payload_key, {}))
            if not meta and payload:
                meta = {
                    k: v
                    for k, v in payload.items()
                    if k not in (self.content_payload_key, self.metadata_payload_key)
                }
            meta["score"] = hit.score
            documents.append(Document(page_content=content, metadata=meta))

        return documents

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Any] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Synchronous similarity search."""
        return asyncio.run(self.asimilarity_search(query, k=k, filter=filter, **kwargs))

    @classmethod
    def from_texts(cls, texts: list[str], embedding: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use get_vector_store() factory instead.")


def get_vector_store(
    knowledge_base_id: uuid.UUID,
    embeddings: HuggingFaceEmbeddings,
) -> NativeQdrantVectorStore:
    """Return a self-contained Qdrant vector store for the given knowledge base."""
    coll = collection_name(knowledge_base_id)
    return NativeQdrantVectorStore(
        collection_name=coll,
        embeddings=embeddings,
        client=get_sync_qdrant_client(),
        async_client=get_qdrant_client(),
        vector_name=DENSE_VECTOR_NAME,
        content_payload_key=CONTENT_KEY,
        metadata_payload_key=METADATA_KEY,
    )
