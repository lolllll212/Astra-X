from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import MemoryScope


class MemoryType(StrEnum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class Memory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Unique memory identifier")
    content: str = Field(min_length=1, description="Memory content text")
    memory_type: MemoryType = Field(default=MemoryType.EPISODIC)
    scope: MemoryScope = Field(default=MemoryScope.CONVERSATION)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    conversation_id: str | None = Field(default=None)
    user_id: str | None = Field(default=None)
    embedding_id: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_accessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    access_count: int = Field(default=0, ge=0)
    ttl_days: int | None = Field(default=None)
