from __future__ import annotations

import pytest

from app.evaluation.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k


def test_recall_at_k_perfect():
    assert recall_at_k(["a", "b", "c"], ["a", "b"], k=2) == 1.0


def test_recall_at_k_partial():
    assert recall_at_k(["a", "b", "c"], ["a", "d"], k=3) == 0.5


def test_recall_at_k_zero():
    assert recall_at_k(["a", "b"], ["c", "d"], k=2) == 0.0


def test_recall_at_k_empty_relevant():
    assert recall_at_k(["a", "b"], [], k=2) == 0.0


def test_mrr_first_hit():
    assert mean_reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0


def test_mrr_second_hit():
    assert mean_reciprocal_rank(["x", "a", "b"], ["a"]) == pytest.approx(0.5)


def test_mrr_no_hit():
    assert mean_reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_ndcg_perfect():
    score = ndcg_at_k(["a", "b", "c"], ["a", "b"], k=2)
    assert score == pytest.approx(1.0)


def test_ndcg_zero():
    score = ndcg_at_k(["x", "y"], ["a", "b"], k=2)
    assert score == 0.0
