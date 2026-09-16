from __future__ import annotations

from pathlib import Path

import chardet
from bs4 import BeautifulSoup

from app.ingestion.loaders.base import BaseLoader, RawDocument


class HTMLLoader(BaseLoader):
    supported_mime_types = ["text/html", "application/xhtml+xml"]

    async def load(self, source: Path | str, **kwargs) -> RawDocument:
        path = Path(source)
        raw_bytes = path.read_bytes()
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding") or "utf-8"
        html = raw_bytes.decode(encoding, errors="replace")

        soup = BeautifulSoup(html, "lxml")

        # Remove scripts, styles, and navigation noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        title = soup.title.string if soup.title else ""
        text = soup.get_text(separator="\n", strip=True)

        return RawDocument(
            content=text,
            filename=path.name,
            mime_type="text/html",
            extra_metadata={"title": title},
        )
