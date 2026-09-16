from __future__ import annotations

from app.retrieval.filters import RetrievedChunk


class ContextBuilder:
    """
    Assembles retrieved chunks into a prompt-ready context string.

    Steps:
    1. Deduplicate by point_id
    2. Remove chunks below min_score threshold
    3. Apply token budget (approximate: 1 token ≈ 4 chars)
    4. Order by score descending
    5. Format each chunk with source citation
    """

    def __init__(
        self,
        max_context_tokens: int = 4096,
        min_score: float = 0.0,
    ) -> None:
        self._max_chars = max_context_tokens * 4  # rough char budget
        self._min_score = min_score

    def build(self, chunks: list[RetrievedChunk]) -> tuple[str, list[dict]]:
        """
        Returns:
            context_str: Formatted context block for the LLM
            sources: List of source dicts for the API response
        """
        # 1. Deduplicate
        seen: set[str] = set()
        unique: list[RetrievedChunk] = []
        for chunk in chunks:
            if chunk.point_id not in seen:
                seen.add(chunk.point_id)
                unique.append(chunk)

        # 2. Filter low scores
        filtered = [c for c in unique if c.score >= self._min_score]

        # 3. Sort by score
        filtered.sort(key=lambda c: c.score, reverse=True)

        # 4. Apply token budget
        selected: list[RetrievedChunk] = []
        used_chars = 0
        for chunk in filtered:
            chunk_chars = len(chunk.text)
            if used_chars + chunk_chars > self._max_chars:
                break
            selected.append(chunk)
            used_chars += chunk_chars

        # 5. Format
        parts: list[str] = []
        sources: list[dict] = []
        for i, chunk in enumerate(selected, start=1):
            header = f"[{i}] {chunk.title or 'Source'}"
            if chunk.source_url:
                header += f" ({chunk.source_url})"
            parts.append(f"{header}\n{chunk.text}")
            sources.append(
                {
                    "index": i,
                    "point_id": chunk.point_id,
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "source_url": chunk.source_url,
                    "chunk_index": chunk.chunk_index,
                    "score": chunk.score,
                }
            )

        context_str = "\n\n---\n\n".join(parts)
        return context_str, sources
