from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base
from app.database.types import JSON, EncryptedString


class ProviderModel(Base):
    __tablename__ = "provider"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    provider_type: Mapped[str] = mapped_column(String(30))
    display_name: Mapped[str] = mapped_column(String(100))
    base_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    api_key: Mapped[str | None] = mapped_column(
        EncryptedString(512), nullable=True,
    )
    protocol: Mapped[str | None] = mapped_column(String(30), default="chat_completions", nullable=True)
    api_version_url: Mapped[str | None] = mapped_column(String(256), nullable=True)
    capabilities: Mapped[list[str] | None] = mapped_column(JSON, default=list, nullable=True)
    supports_responses_api: Mapped[bool | None] = mapped_column(Boolean, default=False, nullable=True)
    supports_vision: Mapped[bool | None] = mapped_column(Boolean, default=False, nullable=True)
    max_tool_calls_per_request: Mapped[int | None] = mapped_column(Integer, default=10, nullable=True)
    models: Mapped[list[str] | None] = mapped_column(JSON, default=list, nullable=True)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
