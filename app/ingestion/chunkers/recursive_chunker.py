from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.chunkers.base import BaseChunker, ChunkResult


def _count_tokens(text: str) -> int:
    """Approximate token count (1 token ≈ 4 chars for multilingual text)."""
    return max(1, len(text) // 4)


class RecursiveChunker(BaseChunker):
    """
    Recursively splits on paragraph → sentence → word boundaries.
    Uses LangChain's RecursiveCharacterTextSplitter under the hood.
    """

    strategy_name = "recursive"

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

    def chunk(self, text: str, metadata: dict | None = None) -> list[ChunkResult]:
        texts = self._splitter.split_text(text)
        meta = metadata or {}
        return [
            ChunkResult(
                text=t,
                chunk_index=i,
                token_count=_count_tokens(t),
                metadata=dict(meta),
            )
            for i, t in enumerate(texts)
        ]
