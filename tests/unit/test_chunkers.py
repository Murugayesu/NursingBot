from __future__ import annotations

import pytest

from app.ingestion.chunkers.recursive_chunker import RecursiveChunker
from app.ingestion.chunkers.markdown_chunker import MarkdownChunker


LONG_TEXT = " ".join(["word"] * 500)  # 500 words ≈ 2500 chars

MARKDOWN_TEXT = """
# Introduction

This is the introduction section with some content about the topic.

## Background

Here is the background section explaining context.

### Details

More detailed information goes here.

## Conclusion

Final thoughts and conclusions.
"""


def test_recursive_chunker_splits():
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk(LONG_TEXT)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 250  # allow a bit of overlap slack


def test_recursive_chunker_indices():
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk(LONG_TEXT)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_recursive_chunker_token_count():
    chunker = RecursiveChunker(chunk_size=400, chunk_overlap=50)
    chunks = chunker.chunk(LONG_TEXT)
    for chunk in chunks:
        assert chunk.token_count > 0


def test_markdown_chunker_splits_on_headers():
    chunker = MarkdownChunker(chunk_size=300, chunk_overlap=30)
    chunks = chunker.chunk(MARKDOWN_TEXT)
    assert len(chunks) >= 2


def test_recursive_metadata_propagation():
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
    meta = {"document_id": "doc_123", "category": "test"}
    chunks = chunker.chunk(LONG_TEXT, metadata=meta)
    for chunk in chunks:
        assert chunk.metadata["document_id"] == "doc_123"
        assert chunk.metadata["category"] == "test"
