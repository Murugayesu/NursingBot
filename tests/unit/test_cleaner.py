from __future__ import annotations

import pytest

from app.ingestion.cleaners.text_cleaner import TextCleaner
from app.ingestion.parsers.text_parser import ParsedDocument


def _make_doc(text: str) -> ParsedDocument:
    return ParsedDocument(text=text, filename="test.txt", mime_type="text/plain")


@pytest.fixture
def cleaner():
    return TextCleaner()


def test_whitespace_normalisation(cleaner):
    doc = _make_doc("hello   world\t\t!")
    result = cleaner.clean(doc)
    assert "  " not in result.text
    assert "\t" not in result.text


def test_multi_newline_collapse(cleaner):
    doc = _make_doc("para1\n\n\n\n\npara2")
    result = cleaner.clean(doc)
    assert "\n\n\n" not in result.text


def test_control_char_removal(cleaner):
    doc = _make_doc("hello\x00world\x07!")
    result = cleaner.clean(doc)
    assert "\x00" not in result.text
    assert "\x07" not in result.text


def test_strip(cleaner):
    doc = _make_doc("\n\n  hello world  \n\n")
    result = cleaner.clean(doc)
    assert result.text == "hello world"


def test_empty_text(cleaner):
    doc = _make_doc("")
    result = cleaner.clean(doc)
    assert result.text == ""
