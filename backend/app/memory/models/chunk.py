from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Unique chunk identifier")
    document_id: str = Field(description="Source document identifier")
    content: str = Field(min_length=1, description="Chunk text content")
    chunk_index: int = Field(ge=0, description="Order within the source document")
    embedding_id: str | None = Field(default=None)
    metadata: dict[str, object] = Field(default_factory=dict)


class ChunkResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunks: Sequence[Chunk] = Field(description="Produced chunks")
    strategy: str = Field(default="fixed_size", description="Chunking strategy used")
