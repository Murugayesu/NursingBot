from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.retrieval.filters import build_qdrant_filter
from app.retrieval.retriever import _rrf_fuse


def _doc(page_content: str, score: float = 1.0) -> Document:
    return Document(page_content=page_content, metadata={"score": score})


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
    list1 = [_doc("alpha", 0.9), _doc("beta", 0.8)]
    list2 = [_doc("alpha", 0.7), _doc("gamma", 0.6)]
    fused = _rrf_fuse([list1, list2])
    contents = [d.page_content for d in fused]
    assert len(contents) == len(set(contents))  # no duplicates


def test_rrf_boosts_appearing_in_both():
    list1 = [_doc("alpha"), _doc("beta")]
    list2 = [_doc("gamma"), _doc("alpha")]
    fused = _rrf_fuse([list1, list2])
    # "alpha" appears in both — should rank first
    assert fused[0].page_content == "alpha"


def test_rrf_single_list_passthrough():
    list1 = [_doc("x"), _doc("y")]
    fused = _rrf_fuse([list1])
    assert [d.page_content for d in fused] == ["x", "y"]
