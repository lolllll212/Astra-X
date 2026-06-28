from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, String
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
    capabilities: Mapped[list[str] | None] = mapped_column(JSON, default=list, nullable=True)
    models: Mapped[list[str] | None] = mapped_column(JSON, default=list, nullable=True)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
