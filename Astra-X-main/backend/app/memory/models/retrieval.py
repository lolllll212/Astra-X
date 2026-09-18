from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import MemoryScope
from app.memory.models.memory import Memory, MemoryType


class MemoryQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Natural language query")
    conversation_id: str | None = Field(default=None)
    memory_types: Sequence[MemoryType] | None = Field(default=None)
    scopes: Sequence[MemoryScope] | None = Field(default=None)
    user_id: str | None = Field(default=None)
    limit: int = Field(default=10, ge=1, le=100)
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0)
    include_working: bool = Field(default=False)
    include_episodic: bool = Field(default=True)
    include_semantic: bool = Field(default=True)
    include_procedural: bool = Field(default=False)


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory: Memory = Field(description="The retrieved memory")
    score: float = Field(ge=0.0, le=1.0, description="Combined relevance score")
    similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    recency_bonus: float = Field(default=0.0, ge=0.0, le=1.0)
    importance_bonus: float = Field(default=0.0, ge=0.0, le=1.0)
    rank: int = Field(default=0, ge=0)
