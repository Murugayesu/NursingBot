"""
LangChain Qdrant vector store wrapper.

Provides a factory that returns a langchain_community.vectorstores.Qdrant
instance wired to both sync and async clients. Replaces the old QdrantIndexer.

Usage in ingestion (write):
    vs = get_vector_store(kb_id, embeddings, cfg)
    await vs.aadd_documents(lc_docs, ids=[...])

Usage in retrieval (read):
    vs = get_vector_store(kb_id, embeddings, cfg)
    retriever = vs.as_retriever(search_kwargs={"k": top_k, "filter": qdrant_filter})
    docs = await retriever.ainvoke(query)
"""
from __future__ import annotations

import uuid

from langchain_community.vectorstores import Qdrant as LCQdrant
from langchain_huggingface import HuggingFaceEmbeddings

from app.storage.qdrant.client import get_qdrant_client, get_sync_qdrant_client
from app.storage.qdrant.collections import collection_name

# LangChain stores document text under this payload key by default.
# We use the LangChain default so loaders, retrievers and the pipeline
# all speak the same language.
CONTENT_KEY = "page_content"
METADATA_KEY = "metadata"


def get_vector_store(
    knowledge_base_id: uuid.UUID,
    embeddings: HuggingFaceEmbeddings,
) -> LCQdrant:
    """
    Return a LangChain Qdrant vector store for the given knowledge base.

    Both sync and async Qdrant clients are passed so that:
    - aadd_documents / asimilarity_search use the async client (FastAPI)
    - add_documents / similarity_search use the sync client (scripts)
    """
    coll = collection_name(knowledge_base_id)
    return LCQdrant(
        client=get_sync_qdrant_client(),
        async_client=get_qdrant_client(),
        collection_name=coll,
        embeddings=embeddings,
        content_payload_key=CONTENT_KEY,
        metadata_payload_key=METADATA_KEY,
    )
