from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.memory.models.memory import Memory
from app.memory.models.retrieval import RetrievalResult


class RetrievalStrategy(ABC):
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        memories: Sequence[Memory],
        *,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        ...
