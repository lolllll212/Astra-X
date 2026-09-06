from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeTriple(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Unique triple identifier")
    subject: str = Field(min_length=1, description="Entity doing the action")
    predicate: str = Field(min_length=1, description="Relationship or action")
    object_: str = Field(alias="object", min_length=1, description="Entity receiving the action")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = Field(default="", description="Origin of this triple")
    conversation_id: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
