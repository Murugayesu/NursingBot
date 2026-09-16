from __future__ import annotations

import csv
import io
from pathlib import Path

import chardet

from app.ingestion.loaders.base import BaseLoader, RawDocument


class CSVLoader(BaseLoader):
    supported_mime_types = ["text/csv", "application/csv"]

    async def load(self, source: Path | str, **kwargs) -> RawDocument:
        path = Path(source)
        raw_bytes = path.read_bytes()
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding") or "utf-8"
        text_data = raw_bytes.decode(encoding, errors="replace")

        reader = csv.DictReader(io.StringIO(text_data))
        rows: list[str] = []
        for row in reader:
            # Convert each row to "key: value" sentence pairs
            row_text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
            rows.append(row_text)

        text = "\n".join(rows)
        return RawDocument(
            content=text,
            filename=path.name,
            mime_type="text/csv",
            extra_metadata={"row_count": len(rows)},
        )
