"""Plugin domain models.

Every plugin entity is a frozen, immutable Pydantic model following the
``ProviderSpec`` pattern established in ``app/domain/provider.py``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "PluginSpec",
    "PluginConfigSpec",
    "PluginDependencySpec",
    "PluginPermissionSpec",
    "PluginLimitSpec",
    "TrustTier",
    "InstallType",
    "PluginStatus",
]


class TrustTier(str):
    """Operator-assigned trust level for a plugin.

    Determines sandboxing aggressiveness at runtime.
    """

    OFFICIAL = "official"
    VERIFIED = "verified"
    COMMUNITY = "community"
    SANDBOXED = "sandboxed"


class InstallType(str):
    """How the plugin was installed."""

    ENTRY_POINT = "entry_point"
    DIRECTORY = "directory"
    PATH = "path"


class PluginStatus(str):
    """Lifecycle state stored in the database."""

    INSTALLED = "installed"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"


class PluginSpec(BaseModel):
    """Immutable specification of an installed plugin.

    Stores persistent plugin metadata. Runtime objects (tool instances,
    provider instances, hook registrations) belong to PluginManager and
    are never stored here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(
        description="UUID primary key, auto-generated.",
    )
    name: str = Field(
        min_length=1,
        description="Unique plugin identifier, e.g. 'weather-tools'.",
    )
    display_name: str = Field(
        default="",
        description="Human-readable label for UI and API responses.",
    )
    version: str = Field(
        default="0.1.0",
        description="Semver version string, e.g. '1.2.0'.",
    )
    sdk_version: str = Field(
        default=">=0.1.0",
        description="PEP 440 specifier declaring SDK ABI compatibility.",
    )
    description: str = Field(
        default="",
        description="Free-text summary of plugin purpose and features.",
    )
    author: str = Field(
        default="",
        description="Plugin author for attribution.",
    )
    homepage: str | None = Field(
        default=None,
        description="URL to documentation, issue tracker, or source repository.",
    )
    license_: str | None = Field(
        default=None,
        description="SPDX identifier or free-text license string.",
    )
    trust_tier: str = Field(
        default="community",
        description="Operator-assigned trust tier.",
    )
    enabled: bool = Field(
        default=False,
        description="Whether the plugin is active at runtime.",
    )
    install_type: str = Field(
        default="entry_point",
        description="How the plugin was installed.",
    )
    entry_point: str | None = Field(
        default=None,
        description="Python entry point, e.g. 'my_pkg:MyPlugin'.",
    )
    manifest_path: str | None = Field(
        default=None,
        description="Filesystem path to pyproject.toml or metadata file.",
    )
    install_path: str | None = Field(
        default=None,
        description="Absolute filesystem path to the plugin root.",
    )
    status: str = Field(
        default="installed",
        description="High-level lifecycle state.",
    )
    status_message: str = Field(
        default="",
        description="Last error message or human-readable status description.",
    )
    checksum: str | None = Field(
        default=None,
        description="SHA-256 of the plugin manifest for integrity verification.",
    )
    created_at: datetime | None = Field(
        default=None,
        description="When the plugin was registered.",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="When the plugin was last modified.",
    )


class PluginConfigSpec(BaseModel):
    """A single configuration key/value pair for a plugin.

    Stored in the ``plugin_config`` table. Supports per-key encryption
    via the ``is_secret`` flag.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(
        description="UUID primary key.",
    )
    plugin_id: str = Field(
        description="Foreign key to the owning PluginSpec.",
    )
    key: str = Field(
        min_length=1,
        description="Configuration key, e.g. 'api_key' or 'model'.",
    )
    value: Any | None = Field(
        default=None,
        description="Configuration value (any JSON-serialisable type).",
    )
    is_secret: bool = Field(
        default=False,
        description="If True, the value is encrypted at rest.",
    )
    created_at: datetime | None = Field(
        default=None,
    )
    updated_at: datetime | None = Field(
        default=None,
    )


class PluginDependencySpec(BaseModel):
    """A dependency declaration from one plugin to another.

    The ``dependency_name`` is a logical reference — it matches the
    ``name`` field of another ``PluginSpec``. It is NOT a foreign key
    because the target plugin may not be installed yet.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(
        description="UUID primary key.",
    )
    plugin_id: str = Field(
        description="Foreign key to the owning PluginSpec.",
    )
    dependency_name: str = Field(
        min_length=1,
        description="Logical name of the required plugin.",
    )
    version_spec: str = Field(
        default="",
        description="PEP 440 specifier, e.g. '>=1.0,<2'.",
    )
    is_optional: bool = Field(
        default=False,
        description="If True, a missing dependency does not block loading.",
    )
    created_at: datetime | None = Field(
        default=None,
    )


class PluginPermissionSpec(BaseModel):
    """A permission declared by a plugin with operator approval status.

    The plugin's manifest declares required permissions. The operator
    approves or denies them via the ``granted`` field.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(
        description="UUID primary key.",
    )
    plugin_id: str = Field(
        description="Foreign key to the owning PluginSpec.",
    )
    permission: str = Field(
        min_length=1,
        description="Permission identifier, e.g. 'tool.register'.",
    )
    granted: bool = Field(
        default=True,
        description="Operator approval. Denied permissions are not enforced.",
    )
    created_at: datetime | None = Field(
        default=None,
    )


class PluginLimitSpec(BaseModel):
    """A resource limit applied to a plugin at runtime.

    Limits are enforced by PluginContext wrappers and the subprocess
    sandbox. Soft limits produce warnings; hard limits cause errors.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(
        description="UUID primary key.",
    )
    plugin_id: str = Field(
        description="Foreign key to the owning PluginSpec.",
    )
    resource_type: str = Field(
        min_length=1,
        description="Resource type: 'cpu', 'memory', 'filesystem', 'network', 'timeout'.",
    )
    soft_limit: str | None = Field(
        default=None,
        description="Soft limit value, e.g. '256MB', '50' (requests/min).",
    )
    hard_limit: str | None = Field(
        default=None,
        description="Hard limit value, e.g. '512MB', '100' (requests/min).",
    )
    duration_seconds: int | None = Field(
        default=None,
        description="Time window in seconds for rate-based limits.",
    )
    created_at: datetime | None = Field(
        default=None,
    )
    updated_at: datetime | None = Field(
        default=None,
    )
