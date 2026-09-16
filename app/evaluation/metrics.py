from __future__ import annotations

import math
from typing import Any


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Fraction of relevant docs retrieved in the top-k results."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for rid in relevant_ids if rid in top_k)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """MRR: 1/rank of first relevant result."""
    relevant_set = set(relevant_ids)
    for rank, pid in enumerate(retrieved_ids, start=1):
        if pid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """NDCG@K with binary relevance."""
    relevant_set = set(relevant_ids)

    def dcg(ids: list[str]) -> float:
        return sum(
            (1.0 if pid in relevant_set else 0.0) / math.log2(rank + 1)
            for rank, pid in enumerate(ids[:k], start=1)
        )

    ideal = sorted(retrieved_ids[:k], key=lambda pid: pid in relevant_set, reverse=True)
    idcg = dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg(retrieved_ids[:k]) / idcg


def aggregate_retrieval_metrics(
    per_question_results: list[dict[str, Any]],
    k: int = 10,
) -> dict[str, float]:
    """Average retrieval metrics across all evaluation questions."""
    if not per_question_results:
        return {}

    recall_scores = [r.get("recall_at_k", 0.0) for r in per_question_results]
    mrr_scores = [r.get("mrr", 0.0) for r in per_question_results]
    ndcg_scores = [r.get("ndcg_at_k", 0.0) for r in per_question_results]

    return {
        f"recall_at_{k}": sum(recall_scores) / len(recall_scores),
        "mrr": sum(mrr_scores) / len(mrr_scores),
        f"ndcg_at_{k}": sum(ndcg_scores) / len(ndcg_scores),
    }
