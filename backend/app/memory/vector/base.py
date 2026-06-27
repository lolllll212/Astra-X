from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field


class VectorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Unique vector record identifier")
    vector: list[float] = Field(min_length=1, description="Embedding vector")
    metadata: dict[str, object] = Field(default_factory=dict)


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Matched record identifier")
    score: float = Field(ge=0.0, description="Similarity score (higher = more similar)")
    metadata: dict[str, object] = Field(default_factory=dict)


class VectorStore(ABC):
    @abstractmethod
    async def insert(self, record: VectorRecord) -> None:
        ...

    @abstractmethod
    async def insert_batch(self, records: Sequence[VectorRecord]) -> None:
        ...

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 10,
        filter_: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        ...

    @abstractmethod
    async def delete(self, record_id: str) -> None:
        ...

    @abstractmethod
    async def delete_batch(self, record_ids: Sequence[str]) -> None:
        ...

    @abstractmethod
    async def get(self, record_id: str) -> VectorRecord | None:
        ...

    @abstractmethod
    async def count(self) -> int:
        ...

    @abstractmethod
    async def clear(self) -> None:
        ...
