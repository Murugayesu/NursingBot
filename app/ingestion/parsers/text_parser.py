from __future__ import annotations

from dataclasses import dataclass, field

from app.ingestion.loaders.base import RawDocument


@dataclass
class ParsedDocument:
    """
    Normalised document representation produced by the text parser.
    Always contains plain text regardless of source format.
    """
    text: str
    filename: str
    mime_type: str
    source_url: str | None = None
    title: str | None = None
    extra_metadata: dict = field(default_factory=dict)


class TextParser:
    """
    Converts a RawDocument into a ParsedDocument.
    Currently performs format-agnostic text normalisation.
    Format-specific parsing is delegated to loaders.
    """

    def parse(self, raw: RawDocument) -> ParsedDocument:
        text = raw.content if isinstance(raw.content, str) else raw.content.decode("utf-8", errors="replace")

        title = raw.extra_metadata.get("title") or raw.filename

        return ParsedDocument(
            text=text,
            filename=raw.filename,
            mime_type=raw.mime_type,
            source_url=raw.source_url,
            title=title,
            extra_metadata=raw.extra_metadata,
        )
