from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval with its score and payload."""
    point_id: str
    text: str
    score: float
    document_id: str
    knowledge_base_id: str
    chunk_index: int
    title: str = ""
    source_url: str | None = None
    category: str | None = None
    document_type: str | None = None
    extra_payload: dict = field(default_factory=dict)


def build_qdrant_filter(filters: dict[str, Any]):
    """
    Convert a filter dict into a Qdrant Filter object.

    Supported operators:
        equality:   {"category": "kubernetes"}
        list (in):  {"category": ["k8s", "docker"]}
        range:      {"published_at": {"gte": "2024-01-01", "lte": "2024-12-31"}}
        AND:        default (all top-level keys are ANDed)
        OR:         {"$or": [{"category": "k8s"}, {"category": "docker"}]}

    Always enforces tenant_id and knowledge_base_id if present.
    """
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        MatchAny,
        MatchValue,
        Range,
    )

    def _condition(key: str, value: Any) -> FieldCondition:
        if isinstance(value, list):
            return FieldCondition(key=key, match=MatchAny(any=value))
        elif isinstance(value, dict):
            # Range filter
            return FieldCondition(
                key=key,
                range=Range(
                    gte=value.get("gte"),
                    lte=value.get("lte"),
                    gt=value.get("gt"),
                    lt=value.get("lt"),
                ),
            )
        else:
            return FieldCondition(key=key, match=MatchValue(value=value))

    must: list = []
    should: list = []

    if "$or" in filters:
        for clause in filters["$or"]:
            for k, v in clause.items():
                should.append(_condition(k, v))
        remaining = {k: v for k, v in filters.items() if k != "$or"}
    else:
        remaining = filters

    for key, value in remaining.items():
        must.append(_condition(key, value))

    return Filter(must=must or None, should=should or None)
