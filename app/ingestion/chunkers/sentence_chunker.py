from __future__ import annotations

from langchain_text_splitters import SentenceTransformersTokenTextSplitter

from app.ingestion.chunkers.base import BaseChunker, ChunkResult


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class SentenceChunker(BaseChunker):
    """
    Splits on sentence boundaries with token-aware sizing.
    Uses SentenceTransformersTokenTextSplitter for accurate tokenisation.
    Falls back to a paragraph split if the model isn't available.
    """

    strategy_name = "sentence"

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 20) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        try:
            self._splitter = SentenceTransformersTokenTextSplitter(
                chunk_overlap=chunk_overlap,
                tokens_per_chunk=chunk_size,
            )
            self._use_lc = True
        except Exception:
            self._use_lc = False

    def chunk(self, text: str, metadata: dict | None = None) -> list[ChunkResult]:
        meta = metadata or {}

        if self._use_lc:
            texts = self._splitter.split_text(text)
        else:
            # Fallback: split on double newlines
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            texts = paragraphs

        return [
            ChunkResult(
                text=t,
                chunk_index=i,
                token_count=_count_tokens(t),
                metadata=dict(meta),
            )
            for i, t in enumerate(texts)
        ]
