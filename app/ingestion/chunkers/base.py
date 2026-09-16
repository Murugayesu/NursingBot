from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ChunkResult:
    """A single text chunk produced by a chunker."""
    text: str
    chunk_index: int
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


class BaseChunker(ABC):
    """Abstract chunker — splits cleaned text into ChunkResult list."""

    @abstractmethod
    def chunk(self, text: str, metadata: dict | None = None) -> list[ChunkResult]:
        ...

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        ...
