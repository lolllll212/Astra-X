from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class ExecutionPatternModel(Base):
    __tablename__ = "execution_pattern"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    goal_pattern: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    capability: Mapped[str | None] = mapped_column(String(128), nullable=True)
    strategy_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    plan_template: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[str] = mapped_column(Text, default="", nullable=False)
    success_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    total_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    avg_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    avg_importance: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    avg_execution_cost_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    last_reflection_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(True), nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(True), default=func.now(), nullable=True
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(True), default=func.now(), onupdate=func.now(), nullable=True
    )
