"""Message ORM model."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base
from app.database.types import JSON, AutoUUID


class MessageModel(Base):
    __tablename__ = "message"

    id: Mapped[str] = mapped_column(AutoUUID, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversation.id", ondelete="CASCADE"), index=True,
    )
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC),
    )
    parent_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("message.id", ondelete="SET NULL"), nullable=True,
    )
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, default=dict, nullable=True,
    )

    __table_args__ = (
        Index("ix_message_conversation_created", "conversation_id", "created_at"),
    )
