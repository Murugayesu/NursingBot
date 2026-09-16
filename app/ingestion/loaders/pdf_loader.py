from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from app.ingestion.loaders.base import BaseLoader, RawDocument


class PDFLoader(BaseLoader):
    supported_mime_types = ["application/pdf"]

    async def load(self, source: Path | str, **kwargs) -> RawDocument:
        path = Path(source)
        doc = fitz.open(str(path))
        pages: list[str] = []
        for page in doc:
            pages.append(page.get_text("text"))
        doc.close()
        text = "\n\n".join(pages)
        return RawDocument(
            content=text,
            filename=path.name,
            mime_type="application/pdf",
            extra_metadata={"page_count": len(pages)},
        )
