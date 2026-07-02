"""Plugin API schemas.

Request and response models for the plugin management endpoints,
following the pattern established in ``provider.py``.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PluginInstallRequest(BaseModel):
    """Request body for installing a new plugin."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=100,
        description="Unique plugin identifier, e.g. 'weather-tools'.",
        examples=["weather-tools", "code-analyzer"],
    )
    display_name: str = Field(
        default="",
        max_length=255,
        description="Human-readable label shown in UI.",
    )
    version: str = Field(
        default="0.1.0",
        max_length=30,
        description="Semver version string.",
    )
    sdk_version: str = Field(
        default=">=0.1.0",
        max_length=50,
        description="PEP 440 specifier declaring SDK ABI compatibility.",
    )
    description: str = Field(
        default="",
        description="Free-text summary of plugin purpose and features.",
    )
    author: str = Field(
        default="",
        max_length=255,
        description="Plugin author for attribution.",
    )
    homepage: str | None = Field(
        default=None,
        max_length=512,
        description="URL to documentation, issue tracker, or source repository.",
    )
    entry_point: str | None = Field(
        default=None,
        max_length=255,
        description="Python entry point, e.g. 'my_pkg:MyPlugin'.",
    )
    manifest_path: str | None = Field(
        default=None,
        max_length=512,
        description="Filesystem path to pyproject.toml or metadata file.",
    )
    install_path: str | None = Field(
        default=None,
        max_length=512,
        description="Absolute filesystem path to the plugin root.",
    )
    enabled: bool = Field(
        default=False,
        description="Whether the plugin is active at runtime.",
    )
    trust_tier: str = Field(
        default="community",
        description="Operator-assigned trust tier.",
    )


class PluginUpdateRequest(BaseModel):
    """Request body for updating an existing plugin.

    All fields are optional — only provided fields are changed.
    """

    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(
        default=None,
        max_length=255,
        description="New human-readable label.",
    )
    version: str | None = Field(
        default=None,
        max_length=30,
        description="New semver version string.",
    )
    description: str | None = Field(
        default=None,
        description="New free-text summary.",
    )
    enabled: bool | None = Field(
        default=None,
        description="Whether the plugin is active at runtime.",
    )
    status: str | None = Field(
        default=None,
        description="New lifecycle status.",
    )
    status_message: str | None = Field(
        default=None,
        description="New status message or error description.",
    )
    trust_tier: str | None = Field(
        default=None,
        description="New operator-assigned trust tier.",
    )


class PluginResponse(BaseModel):
    """Plugin representation returned by the API."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="UUID primary key.")
    name: str = Field(description="Unique plugin identifier.")
    display_name: str = Field(default="", description="Human-readable label.")
    version: str = Field(description="Semver version string.")
    sdk_version: str = Field(description="SDK ABI compatibility specifier.")
    description: str = Field(default="", description="Free-text summary.")
    author: str = Field(default="", description="Plugin author.")
    homepage: str | None = Field(default=None, description="Documentation URL.")
    license: str | None = Field(default=None, alias="license_", description="SPDX license identifier.")
    trust_tier: str = Field(description="Operator-assigned trust tier.")
    enabled: bool = Field(description="Whether the plugin is active.")
    install_type: str = Field(description="How the plugin was installed.")
    entry_point: str | None = Field(default=None, description="Python entry point.")
    manifest_path: str | None = Field(default=None, description="Path to plugin metadata.")
    install_path: str | None = Field(default=None, description="Filesystem path to plugin root.")
    status: str = Field(description="High-level lifecycle state.")
    status_message: str = Field(default="", description="Status description or error message.")
    checksum: str | None = Field(default=None, description="SHA-256 of plugin manifest.")
    created_at: datetime | None = Field(default=None, description="Registration timestamp.")
    updated_at: datetime | None = Field(default=None, description="Last modification timestamp.")


class PluginListResponse(BaseModel):
    """Wrapper for a list of plugins."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plugins: list[PluginResponse] = Field(description="Installed plugins.")
