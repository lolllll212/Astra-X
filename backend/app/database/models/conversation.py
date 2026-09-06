"""Conversation ORM model."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base
from app.database.types import JSON, AutoUUID


class ConversationModel(Base):
    __tablename__ = "conversation"

    id: Mapped[str] = mapped_column(AutoUUID, primary_key=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active")
    participants: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, default=dict, nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    message_count: Mapped[int] = mapped_column(Integer, default=0)
