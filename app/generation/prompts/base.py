from __future__ import annotations

from abc import ABC, abstractmethod


class BasePromptTemplate(ABC):
    version: str = "base"

    @abstractmethod
    def system_prompt(self) -> str:
        ...

    @abstractmethod
    def user_prompt(self, question: str, context: str) -> str:
        ...
