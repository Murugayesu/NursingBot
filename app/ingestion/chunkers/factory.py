from __future__ import annotations

from app.config.rag_config import ChunkingStrategy
from app.ingestion.chunkers.base import BaseChunker
from app.ingestion.chunkers.markdown_chunker import MarkdownChunker
from app.ingestion.chunkers.recursive_chunker import RecursiveChunker
from app.ingestion.chunkers.sentence_chunker import SentenceChunker


def get_chunker(
    strategy: ChunkingStrategy | str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> BaseChunker:
    s = ChunkingStrategy(strategy) if isinstance(strategy, str) else strategy
    if s == ChunkingStrategy.recursive:
        return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif s == ChunkingStrategy.markdown:
        return MarkdownChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif s == ChunkingStrategy.sentence:
        return SentenceChunker(chunk_size=chunk_size // 4, chunk_overlap=chunk_overlap // 4)
    raise ValueError(f"Unknown chunking strategy: {strategy}")
