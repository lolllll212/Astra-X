"""Attachment ORM model."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base
from app.database.types import AutoUUID


class AttachmentModel(Base):
    __tablename__ = "attachment"

    id: Mapped[str] = mapped_column(AutoUUID, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversation.id", ondelete="CASCADE"), index=True,
    )
    file_name: Mapped[str] = mapped_column(String(255))
    mime_type: Mapped[str | None] = mapped_column(String(127), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    attachment_type: Mapped[str] = mapped_column(String(20))
    storage_path: Mapped[str] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC),
    )
