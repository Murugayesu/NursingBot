from __future__ import annotations

from app.retrieval.filters import RetrievedChunk


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Reciprocal Rank Fusion across multiple retrieval result lists.

    RRF score = sum(1 / (k + rank_i)) for each list the chunk appears in.
    Higher is better.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list, start=1):
            pid = chunk.point_id
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
            if pid not in chunks:
                chunks[pid] = chunk

    # Sort by RRF score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    fused: list[RetrievedChunk] = []
    for pid, score in ranked:
        chunk = chunks[pid]
        fused.append(
            RetrievedChunk(
                point_id=chunk.point_id,
                text=chunk.text,
                score=score,  # RRF score replaces raw similarity
                document_id=chunk.document_id,
                knowledge_base_id=chunk.knowledge_base_id,
                chunk_index=chunk.chunk_index,
                title=chunk.title,
                source_url=chunk.source_url,
                category=chunk.category,
                document_type=chunk.document_type,
                extra_payload=chunk.extra_payload,
            )
        )
    return fused
