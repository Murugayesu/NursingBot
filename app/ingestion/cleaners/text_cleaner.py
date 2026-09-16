from __future__ import annotations

import re
import unicodedata

from app.ingestion.parsers.text_parser import ParsedDocument


class TextCleaner:
    """
    Normalises parsed text for downstream chunking and embedding.

    Steps:
    1. Unicode normalisation (NFC)
    2. Replace Windows/Mac line endings with \\n
    3. Strip null bytes and control characters (except \\n and \\t)
    4. Collapse excessive whitespace on each line
    5. Collapse 3+ consecutive blank lines into 2
    6. Strip leading/trailing whitespace
    """

    _CONTROL_CHARS = re.compile(r"[^\S\n\t]|\x00")
    _MULTI_SPACE = re.compile(r"[ \t]{2,}")
    _MULTI_NEWLINE = re.compile(r"\n{3,}")

    def clean(self, doc: ParsedDocument) -> ParsedDocument:
        text = doc.text

        # 1. Unicode NFC
        text = unicodedata.normalize("NFC", text)

        # 2. Normalise line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Remove null bytes and other control chars (keep \n and \t)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # 4. Collapse horizontal whitespace on each line
        lines = [self._MULTI_SPACE.sub(" ", line).strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # 5. Collapse 3+ blank lines → 2
        text = self._MULTI_NEWLINE.sub("\n\n", text)

        # 6. Strip
        text = text.strip()

        return ParsedDocument(
            text=text,
            filename=doc.filename,
            mime_type=doc.mime_type,
            source_url=doc.source_url,
            title=doc.title,
            extra_metadata=doc.extra_metadata,
        )
