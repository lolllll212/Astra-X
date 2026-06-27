"""Request and response schemas for the provider management API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import ProviderType


class ProviderRegisterRequest(BaseModel):
    """Request body for registering a new provider.

    Attributes:
        provider_id: Unique provider identifier (e.g. ``"ollama"``).
        provider_type: The provider backend type.
        display_name: Human-readable name.
    """

    model_config = ConfigDict(extra="forbid")

    provider_id: str = Field(min_length=1, description="Unique provider identifier.")
    provider_type: ProviderType = Field(description="Provider backend type.")
    display_name: str = Field(min_length=1, description="Human-readable name.")


class ProviderResponse(BaseModel):
    """Provider representation returned by the API.

    Attributes:
        id: Provider identifier.
        provider_type: Backend type.
        display_name: Human-readable name.
        supported_capabilities: Capabilities this provider supports.
        created_at: When the provider was registered.
        updated_at: When the provider was last modified.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Provider identifier.")
    provider_type: ProviderType = Field(description="Backend type.")
    display_name: str = Field(description="Human-readable name.")
    supported_capabilities: list[str] = Field(
        default_factory=list,
        description="Supported capabilities.",
    )
    created_at: datetime | None = Field(default=None, description="Registration timestamp.")
    updated_at: datetime | None = Field(default=None, description="Last modification timestamp.")


class ProviderListResponse(BaseModel):
    """Wrapper for a list of providers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    providers: list[ProviderResponse] = Field(description="Registered providers.")


class ProviderCheckRequest(BaseModel):
    """Request body for checking a specific provider's health."""

    model_config = ConfigDict(extra="forbid")

    provider_id: str = Field(min_length=1, description="Provider identifier to check.")


class ProviderCheckResponse(BaseModel):
    """Health check result for a single provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(description="Provider identifier.")
    healthy: bool = Field(description="Whether the provider is reachable.")


class ModelInfoResponse(BaseModel):
    """A model available from a provider.

    Attributes:
        id: Model identifier.
        name: Human-readable model name.
        capabilities: Model capabilities.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Model identifier.")
    name: str = Field(description="Human-readable model name.")
    capabilities: list[str] = Field(
        default_factory=list,
        description="Model capabilities.",
    )


class ModelListResponse(BaseModel):
    """Wrapper for a list of available models."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    models: list[ModelInfoResponse] = Field(description="Available models.")
