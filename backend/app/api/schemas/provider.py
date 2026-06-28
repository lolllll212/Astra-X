from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import ProviderType


def _validate_not_placeholder(value: str) -> str:
    stripped = value.strip()
    if stripped.lower() == "string":
        raise ValueError(
            f"'{value}' is a placeholder value and is not allowed. "
            "Please provide a meaningful identifier.",
        )
    return stripped


class ProviderRegisterRequest(BaseModel):
    """Request body for registering a new provider.

    Examples:
        >>> ProviderRegisterRequest(
        ...     provider_id="ollama-local",
        ...     provider_type="ollama",
        ...     display_name="Local Ollama",
        ...     base_url="http://localhost:11434",
        ...     models=["llama3.1", "mistral"],
        ... )

    Attributes:
        provider_id: Unique provider identifier (e.g. ``"ollama-local"``).
            Must not be ``"string"`` or empty.
        provider_type: The provider backend type (e.g. ``"ollama"``,
            ``"lm_studio"``, ``"openai_compatible"``).
        display_name: Human-readable name shown in the UI.
        base_url: Base URL for the provider API. Falls back to the
            type-specific default when omitted.
        api_key: API key for providers that require authentication.
            Stored encrypted; never returned in API responses.
        models: Optional list of model identifiers (e.g. ``["llama3.1"]``).
            If omitted, the provider's auto-detected models are used.
    """

    model_config = ConfigDict(extra="forbid")

    provider_id: str = Field(
        min_length=1,
        max_length=100,
        description="Unique provider identifier.",
        examples=["ollama-local", "openai-prod", "lm-studio-1"],
    )
    provider_type: ProviderType = Field(
        description="Provider backend type.",
        examples=["ollama", "lm_studio", "openai_compatible"],
    )
    display_name: str = Field(
        min_length=1,
        max_length=100,
        description="Human-readable name.",
        examples=["Local Ollama", "OpenAI Production", "LM Studio Instance"],
    )
    base_url: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
        description="Provider API base URL. Must be an absolute http(s) URL.",
        examples=["http://localhost:11434", "http://localhost:1234/v1"],
    )
    api_key: str | None = Field(
        default=None,
        min_length=1,
        description="API key for authentication. Stored encrypted.",
        examples=["sk-..."],
    )
    models: list[str] | None = Field(
        default=None,
        description="List of model identifiers this provider serves.",
        examples=[["llama3.1", "mistral"], ["gpt-4o", "gpt-4o-mini"]],
    )


class ProviderUpdateRequest(BaseModel):
    """Request body for updating an existing provider.

    All fields are optional — only provided fields are updated.

    Examples:
        >>> ProviderUpdateRequest(
        ...     display_name="Updated Name",
        ...     base_url="http://localhost:11434",
        ...     is_enabled=True,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Human-readable name.",
    )
    base_url: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
        description="Provider API base URL. Must be an absolute http(s) URL.",
    )
    api_key: str | None = Field(
        default=None,
        min_length=1,
        description="API key for authentication. Stored encrypted.",
    )
    is_enabled: bool | None = Field(
        default=None,
        description="Whether this provider is active and can serve requests.",
    )
    models: list[str] | None = Field(
        default=None,
        description="List of model identifiers this provider serves.",
    )


class ProviderResponse(BaseModel):
    """Provider representation returned by the API.

    The ``api_key`` field is never included in responses.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Provider identifier.", examples=["ollama-local"])
    provider_type: ProviderType = Field(description="Backend type.")
    display_name: str = Field(description="Human-readable name.")
    base_url: str | None = Field(default=None, description="Provider API base URL.")
    is_enabled: bool = Field(default=True, description="Whether the provider is active.")
    supported_capabilities: list[str] = Field(
        default_factory=list,
        description="Supported capabilities.",
    )
    models: list[str] = Field(
        default_factory=list,
        description="Models this provider serves.",
    )
    healthy: bool | None = Field(
        default=None,
        description="Liveness check result, if available.",
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
    """A model available from a provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Model identifier.", examples=["llama3.1", "gpt-4o"])
    name: str = Field(description="Human-readable model name.")
    capabilities: list[str] = Field(
        default_factory=list,
        description="Model capabilities.",
    )


class ModelListResponse(BaseModel):
    """Wrapper for a list of available models."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    models: list[ModelInfoResponse] = Field(description="Available models.")
