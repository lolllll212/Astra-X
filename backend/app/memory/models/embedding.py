from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field


class Embedding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Unique embedding identifier")
    vector: list[float] = Field(min_length=1, description="Embedding vector")
    model: str = Field(default="", description="Model used to generate this embedding")
    dimensions: int = Field(ge=1, description="Vector dimensionality")


class EmbeddingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    embeddings: Sequence[list[float]] = Field(description="Batch of embedding vectors")
    model: str = Field(default="", description="Model used")
    dimensions: int = Field(ge=1, description="Vector dimensionality")
