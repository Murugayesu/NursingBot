from __future__ import annotations

import pytest

from app.retrieval.filters import build_qdrant_filter
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.filters import RetrievedChunk


def _chunk(pid: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        point_id=pid,
        text="sample text",
        score=score,
        document_id="doc1",
        knowledge_base_id="kb1",
        chunk_index=0,
    )


# ── Filter tests ──────────────────────────────────────────────────────────────

def test_equality_filter():
    f = build_qdrant_filter({"category": "kubernetes"})
    assert f is not None
    assert f.must is not None


def test_list_filter():
    f = build_qdrant_filter({"category": ["k8s", "docker"]})
    assert f.must[0].match.any == ["k8s", "docker"]


def test_range_filter():
    f = build_qdrant_filter({"score": {"gte": 0.5, "lte": 1.0}})
    r = f.must[0].range
    assert r.gte == 0.5
    assert r.lte == 1.0


def test_or_filter():
    f = build_qdrant_filter({"$or": [{"cat": "a"}, {"cat": "b"}]})
    assert f.should is not None
    assert len(f.should) == 2


# ── RRF fusion tests ──────────────────────────────────────────────────────────

def test_rrf_deduplicates():
    list1 = [_chunk("a", 0.9), _chunk("b", 0.8)]
    list2 = [_chunk("a", 0.7), _chunk("c", 0.6)]
    fused = reciprocal_rank_fusion([list1, list2])
    ids = [c.point_id for c in fused]
    assert len(ids) == len(set(ids))  # no duplicates


def test_rrf_boosts_appearing_in_both():
    list1 = [_chunk("a"), _chunk("b")]
    list2 = [_chunk("c"), _chunk("a")]
    fused = reciprocal_rank_fusion([list1, list2])
    # "a" appears in both lists, should have higher RRF score than "b" and "c"
    scores = {c.point_id: c.score for c in fused}
    assert scores["a"] > scores["b"]
    assert scores["a"] > scores["c"]


def test_rrf_single_list_passthrough():
    list1 = [_chunk("x"), _chunk("y")]
    fused = reciprocal_rank_fusion([list1])
    assert [c.point_id for c in fused] == ["x", "y"]
