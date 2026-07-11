"""Goal hierarchy ORM models — Mission → Objective → Goal."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base
from app.database.types import AutoUUID, JSON


class MissionModel(Base):
    __tablename__ = "mission"

    id: Mapped[str] = mapped_column(AutoUUID, primary_key=True)
    title: Mapped[str] = mapped_column(String(255), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="active")
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))

    objectives = relationship("ObjectiveModel", back_populates="mission", cascade="all, delete-orphan")


class ObjectiveModel(Base):
    __tablename__ = "objective"

    id: Mapped[str] = mapped_column(AutoUUID, primary_key=True)
    mission_id: Mapped[str] = mapped_column(String(36), ForeignKey("mission.id"), index=True)
    title: Mapped[str] = mapped_column(String(255), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="pending")
    order: Mapped[int] = mapped_column(Integer, default=0)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    mission = relationship("MissionModel", back_populates="objectives")
    goals = relationship("GoalModel", back_populates="objective", cascade="all, delete-orphan")


class GoalModel(Base):
    __tablename__ = "goal"

    id: Mapped[str] = mapped_column(AutoUUID, primary_key=True)
    objective_id: Mapped[str] = mapped_column(String(36), ForeignKey("objective.id"), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="pending")
    plan_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    result_summary: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    objective = relationship("ObjectiveModel", back_populates="goals")


class ActionModel(Base):
    __tablename__ = "action"

    id: Mapped[str] = mapped_column(AutoUUID, primary_key=True)
    goal_id: Mapped[str] = mapped_column(String(36), ForeignKey("goal.id"), index=True)
    task_id: Mapped[str] = mapped_column(String(36), default="")
    tool_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    description: Mapped[str] = mapped_column(Text, default="")
    input_summary: Mapped[str] = mapped_column(Text, default="")
    output_summary: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="completed")
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    reflection_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
