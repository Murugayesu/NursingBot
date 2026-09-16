from __future__ import annotations

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.ingestion.chunkers.recursive_chunker import ChunkResult


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


class MarkdownChunker:
    """Splits Markdown by headers first, then recursively splits oversized sections."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100) -> None:
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_HEADERS,
            strip_headers=False,
        )
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, text: str, metadata: dict | None = None) -> list[ChunkResult]:
        header_docs = self._header_splitter.split_text(text)
        final_docs = self._text_splitter.split_documents(header_docs)

        meta = metadata or {}
        results: list[ChunkResult] = []
        for i, doc in enumerate(final_docs):
            chunk_meta = {**meta, **doc.metadata}
            results.append(
                ChunkResult(
                    text=doc.page_content,
                    chunk_index=i,
                    token_count=_count_tokens(doc.page_content),
                    metadata=chunk_meta,
                )
            )
        return results
