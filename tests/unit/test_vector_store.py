from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.storage.qdrant.vector_store import NativeQdrantVectorStore


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1024 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 1024

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1024 for _ in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return [0.1] * 1024


def test_vector_store_init_and_embeddings_property():
    emb = FakeEmbeddings()
    vs = NativeQdrantVectorStore(
        collection_name="test_col",
        embeddings=emb,
    )
    assert vs.embeddings is emb
    assert vs._embeddings is emb
    assert vs.collection_name == "test_col"


@pytest.mark.asyncio
async def test_vector_store_aadd_documents():
    emb = FakeEmbeddings()
    mock_async_client = AsyncMock()
    mock_async_client.upsert = AsyncMock(return_value=None)

    vs = NativeQdrantVectorStore(
        collection_name="test_col",
        embeddings=emb,
        async_client=mock_async_client,
    )

    docs = [
        Document(page_content="hello world", metadata={"chunk_index": 0}),
        Document(page_content="foo bar", metadata={"chunk_index": 1}),
    ]

    ids = await vs.aadd_documents(docs)
    assert len(ids) == 2
    mock_async_client.upsert.assert_awaited_once()


@pytest.mark.asyncio
async def test_vector_store_asimilarity_search_query_points():
    emb = FakeEmbeddings()
    mock_async_client = AsyncMock()

    mock_point = MagicMock()
    mock_point.score = 0.95
    mock_point.payload = {
        "page_content": "retrieved content",
        "metadata": {"source": "test.pdf", "chunk_index": 0},
    }

    mock_response = MagicMock()
    mock_response.points = [mock_point]
    mock_async_client.query_points = AsyncMock(return_value=mock_response)

    vs = NativeQdrantVectorStore(
        collection_name="test_col",
        embeddings=emb,
        async_client=mock_async_client,
    )

    results = await vs.asimilarity_search("query test", k=3)
    assert len(results) == 1
    assert results[0].page_content == "retrieved content"
    assert results[0].metadata["source"] == "test.pdf"
    assert results[0].metadata["score"] == 0.95
    mock_async_client.query_points.assert_awaited_once()
