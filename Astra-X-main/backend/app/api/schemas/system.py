"""Request and response schemas for system information endpoints."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VersionInfo(BaseModel):
    """Application version information.

    Attributes:
        app_name: Human-readable application name.
        app_version: Semantic version string.
        python_version: Python runtime version.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    app_name: str = Field(description="Application name.")
    app_version: str = Field(description="Semantic version.")
    python_version: str = Field(description="Python runtime version.")


class ConfigInfo(BaseModel):
    """Sanitised application configuration (no secrets).

    Attributes:
        environment: Runtime environment.
        debug: Whether debug mode is enabled.
        log_level: Logging severity level.
        log_format: Structured log format.
        database_url: Database connection URL (truncated for safety).
        default_llm_provider: Default LLM provider identifier.
        default_llm_model: Default LLM model name.
        api_v1_prefix: API version 1 URL prefix.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    environment: str = Field(description="Runtime environment.")
    debug: bool = Field(description="Debug mode enabled.")
    log_level: str = Field(description="Logging level.")
    log_format: str = Field(description="Log format.")
    database_url: str = Field(description="Database URL (scheme only shown).")
    default_llm_provider: str = Field(description="Default LLM provider.")
    default_llm_model: str = Field(description="Default LLM model.")
    api_v1_prefix: str = Field(description="API v1 URL prefix.")


class MetricsResponse(BaseModel):
    """Application metrics snapshot.

    Attributes:
        uptime_seconds: Application uptime in seconds.
        active_conversations: Active conversation count.
        total_messages: Total message count across all conversations.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    uptime_seconds: float = Field(ge=0, description="Application uptime in seconds.")
    active_conversations: int = Field(ge=0, description="Active conversation count.")
    total_messages: int = Field(ge=0, description="Total message count.")
