"""Usage ORM model."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base
from app.database.types import JSON, AutoUUID


class UsageModel(Base):
    __tablename__ = "usage"

    id: Mapped[str] = mapped_column(AutoUUID, primary_key=True)
    message_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("message.id", ondelete="CASCADE"), index=True,
    )
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens_details: Mapped[dict[str, int] | None] = mapped_column(
        JSON, nullable=True,
    )
    completion_tokens_details: Mapped[dict[str, int] | None] = mapped_column(
        JSON, nullable=True,
    )
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC),
    )
