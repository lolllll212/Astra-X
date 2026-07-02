"""Plugin ORM models — core table and sub-tables for plugin metadata.

The plugin aggregate is split across five tables:

* ``plugin``        — core plugin record (metadata, status, install info)
* ``plugin_config``  — key/value configuration with per-key encryption support
* ``plugin_dependency`` — plugin→plugin dependency declarations
* ``plugin_permission`` — declared permissions with operator approval
* ``plugin_limit``   — resource limits (CPU, memory, network, timeout)

All sub-tables cascade-delete with their parent plugin.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base
from app.database.types import AutoUUID, JSON


class PluginModel(Base):
    __tablename__ = "plugin"

    id: Mapped[str] = mapped_column(AutoUUID, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    display_name: Mapped[str] = mapped_column(String(255), default="")
    version: Mapped[str] = mapped_column(String(30), default="0.1.0")
    sdk_version: Mapped[str] = mapped_column(String(50), default=">=0.1.0")
    description: Mapped[str] = mapped_column(Text, default="")
    author: Mapped[str] = mapped_column(String(255), default="")
    homepage: Mapped[str | None] = mapped_column(String(512), nullable=True)
    license_: Mapped[str | None] = mapped_column(
        "license", String(50), nullable=True,
    )
    trust_tier: Mapped[str] = mapped_column(String(20), default="community")
    enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    install_type: Mapped[str] = mapped_column(String(20), default="entry_point")
    entry_point: Mapped[str | None] = mapped_column(String(255), nullable=True)
    manifest_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    install_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="installed")
    status_message: Mapped[str] = mapped_column(Text, default="")
    capabilities: Mapped[list[str]] = mapped_column(JSON, default=list)
    minimum_core_version: Mapped[str | None] = mapped_column(String(30), nullable=True)
    maximum_core_version: Mapped[str | None] = mapped_column(String(30), nullable=True)
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    # -- relationships (cascade delete with parent) --------------------------
    configs: Mapped[list[PluginConfigModel]] = relationship(
        back_populates="plugin", cascade="all, delete-orphan",
    )
    dependencies: Mapped[list[PluginDependencyModel]] = relationship(
        back_populates="plugin", cascade="all, delete-orphan",
    )
    permissions: Mapped[list[PluginPermissionModel]] = relationship(
        back_populates="plugin", cascade="all, delete-orphan",
    )
    limits: Mapped[list[PluginLimitModel]] = relationship(
        back_populates="plugin", cascade="all, delete-orphan",
    )


class PluginConfigModel(Base):
    __tablename__ = "plugin_config"

    id: Mapped[str] = mapped_column(AutoUUID, primary_key=True)
    plugin_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("plugin.id", ondelete="CASCADE"), index=True,
    )
    key: Mapped[str] = mapped_column(String(255))
    value: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    is_secret: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    plugin: Mapped[PluginModel] = relationship(back_populates="configs")

    __table_args__ = (
        UniqueConstraint("plugin_id", "key", name="uq_plugin_config_key"),
    )


class PluginDependencyModel(Base):
    __tablename__ = "plugin_dependency"

    id: Mapped[str] = mapped_column(AutoUUID, primary_key=True)
    plugin_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("plugin.id", ondelete="CASCADE"), index=True,
    )
    dependency_name: Mapped[str] = mapped_column(String(100))
    version_spec: Mapped[str] = mapped_column(String(50), default="")
    is_optional: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC),
    )

    plugin: Mapped[PluginModel] = relationship(back_populates="dependencies")

    __table_args__ = (
        UniqueConstraint(
            "plugin_id", "dependency_name", name="uq_plugin_dependency_name",
        ),
    )


class PluginPermissionModel(Base):
    __tablename__ = "plugin_permission"

    id: Mapped[str] = mapped_column(AutoUUID, primary_key=True)
    plugin_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("plugin.id", ondelete="CASCADE"), index=True,
    )
    permission: Mapped[str] = mapped_column(String(50))
    granted: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC),
    )

    plugin: Mapped[PluginModel] = relationship(back_populates="permissions")

    __table_args__ = (
        UniqueConstraint(
            "plugin_id", "permission", name="uq_plugin_permission",
        ),
    )


class PluginLimitModel(Base):
    __tablename__ = "plugin_limit"

    id: Mapped[str] = mapped_column(AutoUUID, primary_key=True)
    plugin_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("plugin.id", ondelete="CASCADE"), index=True,
    )
    resource_type: Mapped[str] = mapped_column(String(30))
    soft_limit: Mapped[str | None] = mapped_column(String(50), nullable=True)
    hard_limit: Mapped[str | None] = mapped_column(String(50), nullable=True)
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    plugin: Mapped[PluginModel] = relationship(back_populates="limits")

    __table_args__ = (
        UniqueConstraint(
            "plugin_id", "resource_type", name="uq_plugin_limit_resource",
        ),
    )
