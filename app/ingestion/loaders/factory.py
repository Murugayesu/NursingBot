from __future__ import annotations

from pathlib import Path

from app.ingestion.loaders.base import BaseLoader, RawDocument
from app.ingestion.loaders.csv_loader import CSVLoader
from app.ingestion.loaders.html_loader import HTMLLoader
from app.ingestion.loaders.markdown_loader import MarkdownLoader
from app.ingestion.loaders.pdf_loader import PDFLoader
from app.ingestion.loaders.txt_loader import TxtLoader

_MIME_MAP: dict[str, BaseLoader] = {}
_EXT_MAP: dict[str, BaseLoader] = {}


def _register(loader: BaseLoader, extensions: list[str]) -> None:
    for mime in loader.supported_mime_types:
        _MIME_MAP[mime] = loader
    for ext in extensions:
        _EXT_MAP[ext.lower()] = loader


_register(PDFLoader(), [".pdf"])
_register(TxtLoader(), [".txt"])
_register(MarkdownLoader(), [".md", ".markdown"])
_register(HTMLLoader(), [".html", ".htm", ".xhtml"])
_register(CSVLoader(), [".csv"])


def get_loader(path: Path | str, mime_type: str | None = None) -> BaseLoader:
    """
    Return the appropriate loader for the given file path or MIME type.
    Raises ValueError if no loader is found.
    """
    if mime_type and mime_type in _MIME_MAP:
        return _MIME_MAP[mime_type]

    ext = Path(path).suffix.lower()
    if ext in _EXT_MAP:
        return _EXT_MAP[ext]

    raise ValueError(
        f"No loader found for path='{path}', mime_type='{mime_type}'. "
        f"Supported extensions: {list(_EXT_MAP.keys())}"
    )
