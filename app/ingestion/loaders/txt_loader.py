from __future__ import annotations

from pathlib import Path

import chardet

from app.ingestion.loaders.base import BaseLoader, RawDocument


class TxtLoader(BaseLoader):
    supported_mime_types = ["text/plain"]

    async def load(self, source: Path | str, **kwargs) -> RawDocument:
        path = Path(source)
        raw_bytes = path.read_bytes()
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding") or "utf-8"
        text = raw_bytes.decode(encoding, errors="replace")
        return RawDocument(
            content=text,
            filename=path.name,
            mime_type="text/plain",
        )
