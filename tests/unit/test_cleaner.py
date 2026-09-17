"""
Tests for LangChain loader dispatcher and basic text behavior.

TextCleaner was removed — LangChain document loaders handle
text extraction and normalization. These tests verify that
the loader dispatcher returns proper Document objects and that
text is reasonably clean.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document


def test_load_txt_returns_documents(tmp_path):
    """TextLoader should return at least one Document."""
    f = tmp_path / "sample.txt"
    f.write_text("Hello world\nThis is a test.", encoding="utf-8")

    from app.ingestion.loaders.lc_loaders import load_document
    docs = load_document(f, "text/plain")
    assert len(docs) >= 1
    assert isinstance(docs[0], Document)
    assert "Hello world" in docs[0].page_content


def test_load_unknown_mime_falls_back_to_text(tmp_path):
    """Unknown MIME type should fall back to TextLoader."""
    f = tmp_path / "data.xyz"
    f.write_text("Fallback content here.", encoding="utf-8")

    from app.ingestion.loaders.lc_loaders import load_document
    docs = load_document(f, "application/unknown-xyz")
    assert len(docs) >= 1
    assert "Fallback content" in docs[0].page_content


def test_load_markdown_returns_documents(tmp_path):
    """Markdown loader should return Documents from .md content."""
    f = tmp_path / "readme.md"
    f.write_text("# Title\n\nSome markdown content.\n\n## Section\n\nMore content.", encoding="utf-8")

    from app.ingestion.loaders.lc_loaders import load_document
    docs = load_document(f, "text/markdown")
    assert len(docs) >= 1
    # All content should be somewhere across docs
    all_text = " ".join(d.page_content for d in docs)
    assert "Title" in all_text or "Section" in all_text


def test_docs_have_no_null_bytes(tmp_path):
    """Loaded text should not contain null bytes."""
    f = tmp_path / "clean.txt"
    f.write_text("Clean text without control characters.", encoding="utf-8")

    from app.ingestion.loaders.lc_loaders import load_document
    docs = load_document(f, "text/plain")
    for doc in docs:
        assert "\x00" not in doc.page_content
