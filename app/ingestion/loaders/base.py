from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RawDocument:
    """Raw output from a loader before any parsing or cleaning."""
    content: bytes | str
    filename: str
    mime_type: str
    source_url: str | None = None
    extra_metadata: dict = field(default_factory=dict)


class BaseLoader(ABC):
    """Abstract loader — reads a file or URL and returns a RawDocument."""

    @abstractmethod
    async def load(self, source: Path | str, **kwargs) -> RawDocument:
        """Load content from the given source."""
        ...

    @property
    @abstractmethod
    def supported_mime_types(self) -> list[str]:
        """MIME types this loader handles."""
        ...
