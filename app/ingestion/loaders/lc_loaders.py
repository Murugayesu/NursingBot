"""
LangChain document loader dispatcher.

Replaces 7 custom loader files with a single function that picks
the right langchain_community loader by MIME type and returns
a list of LangChain Document objects.
"""
from __future__ import annotations

from pathlib import Path

import structlog
from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

logger = structlog.get_logger(__name__)

_MIME_TO_LOADER = {
    "application/pdf": PyMuPDFLoader,
    "text/plain": TextLoader,
    "text/markdown": TextLoader,       # UnstructuredMarkdownLoader needs heavy deps
    "text/x-markdown": TextLoader,
    "text/html": BSHTMLLoader,
    "text/htm": BSHTMLLoader,
    "text/csv": CSVLoader,
    "application/csv": CSVLoader,
}


def load_document(file_path: Path, mime_type: str) -> list[Document]:
    """
    Load a file using the appropriate LangChain loader.
    Returns a list of LangChain Document objects (one per page for PDFs,
    one per row for CSVs, one for everything else).
    Falls back to TextLoader for unknown MIME types.
    """
    loader_cls = _MIME_TO_LOADER.get(mime_type, TextLoader)
    logger.debug("loading document", loader=loader_cls.__name__, file=str(file_path))

    kwargs: dict = {}
    if loader_cls is TextLoader:
        kwargs["encoding"] = "utf-8"
        kwargs["autodetect_encoding"] = True

    loader = loader_cls(str(file_path), **kwargs)
    docs = loader.load()
    logger.debug("loaded", pages=len(docs))
    return docs
