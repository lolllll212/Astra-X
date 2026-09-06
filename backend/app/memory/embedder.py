from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.memory.models.embedding import EmbeddingResult


class Embedder(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        ...

    @abstractmethod
    async def embed_batch(self, texts: Sequence[str]) -> EmbeddingResult:
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
