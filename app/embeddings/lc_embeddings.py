"""
LangChain HuggingFace embeddings singleton for BGE-M3.

Uses langchain_huggingface.HuggingFaceEmbeddings which wraps
sentence-transformers under the hood.

Usage:
    from app.embeddings.lc_embeddings import get_embeddings
    embeddings = get_embeddings(model_name="BAAI/bge-m3", device="cpu")
    vectors = embeddings.embed_documents(["text1", "text2"])
    query_vec = embeddings.embed_query("question")
"""
from __future__ import annotations

import structlog
from langchain_huggingface import HuggingFaceEmbeddings

logger = structlog.get_logger(__name__)

_embeddings: HuggingFaceEmbeddings | None = None
_current_model: str = ""

DIMENSION = 1024  # BGE-M3 dense output dimension


def get_embeddings(
    model_name: str = "BAAI/bge-m3",
    device: str = "cpu",
) -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFaceEmbeddings instance.
    Reloads only if model_name changes (rare at runtime).
    """
    global _embeddings, _current_model
    if _embeddings is None or _current_model != model_name:
        logger.info("loading embedding model", model=model_name, device=device)
        _embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        _current_model = model_name
    return _embeddings
