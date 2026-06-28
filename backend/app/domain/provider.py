from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from app.domain.enums import ModelCapability, ProviderType

__all__ = [
    "ModelID",
    "ModelSpec",
    "ProviderAuth",
    "ProviderID",
    "ProviderSpec",
]


class ProviderID(str):
    """A validated, lowercase provider identifier."""

    def __new__(cls, value: str) -> ProviderID:
        normalized = value.strip().lower()
        if not normalized:
            msg = "ProviderID must not be empty"
            raise ValueError(msg)
        return super().__new__(cls, normalized)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: type, handler: object) -> object:
        from pydantic_core.core_schema import no_info_plain_validator_function

        return no_info_plain_validator_function(cls)


class ModelID(str):
    """A validated, lowercase model identifier."""

    def __new__(cls, value: str) -> ModelID:
        normalized = value.strip().lower()
        if not normalized:
            msg = "ModelID must not be empty"
            raise ValueError(msg)
        return super().__new__(cls, normalized)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: type, handler: object) -> object:
        from pydantic_core.core_schema import no_info_plain_validator_function

        return no_info_plain_validator_function(cls)


class ProviderAuth(BaseModel):
    """Authentication credentials for a single provider backend."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_key: SecretStr | None = Field(
        default=None,
        description="API key or token for the provider.",
    )
    base_url: str | None = Field(
        default=None,
        min_length=1,
        description="Provider-specific base URL when non-default.",
    )
    extra_headers: dict[str, str] | None = Field(
        default=None,
        description="Additional HTTP headers sent with every request to this provider.",
    )


class ProviderSpec(BaseModel):
    """Immutable specification of an LLM provider backend.

    Describes *what* a provider is and *what* it can do, not *how* to
    interact with it (that is the responsibility of the provider adapter).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: ProviderID = Field(
        description="Unique, stable identifier for this provider.",
    )
    provider_type: ProviderType = Field(
        description="Enum variant identifying the concrete adapter to use.",
    )
    display_name: str = Field(
        min_length=1,
        description="Human-readable name shown in UI and logs.",
    )
    is_enabled: bool = Field(
        default=True,
        description="Whether this provider is active and can be used for routing.",
    )
    base_url: str | None = Field(
        default=None,
        description="Base URL for the provider API. Falls back to the type's default.",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="Encrypted API key for the provider. Never exposed in responses.",
    )
    supported_capabilities: frozenset[ModelCapability] = Field(
        default_factory=lambda: frozenset({ModelCapability.CHAT}),
        description="Set of capabilities every model on this provider supports.",
    )
    models: list[str] = Field(
        default_factory=list,
        description="List of model identifiers this provider can serve.",
    )
    created_at: datetime | None = Field(
        default=None,
        description="When the provider was registered.",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="When the provider was last modified.",
    )

    @field_validator("supported_capabilities", mode="before")
    @classmethod
    def _coerce_to_frozenset(
        cls,
        value: Any,
    ) -> frozenset[ModelCapability]:
        if isinstance(value, frozenset):
            return value
        if isinstance(value, (set, list, tuple)):
            return frozenset(value)
        msg = f"Expected a set or list of capabilities, got {type(value)}"
        raise TypeError(msg)


class ModelSpec(BaseModel):
    """Immutable specification of a language model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: ModelID = Field(
        description="Unique model identifier, normalised to lowercase.",
    )
    provider_id: ProviderID = Field(
        description="Provider that serves this model.",
    )
    display_name: str = Field(
        min_length=1,
        description="Human-readable model name for UI display.",
    )
    capabilities: frozenset[ModelCapability] = Field(
        default_factory=lambda: frozenset({ModelCapability.CHAT}),
        description="Capabilities this specific model supports.",
    )
    context_length: int = Field(
        default=4096,
        ge=1,
        description="Maximum number of tokens the model accepts as input.",
    )
    max_output_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of tokens the model can generate, if bounded.",
    )
    cost_per_1k_input_tokens: float | None = Field(
        default=None,
        ge=0,
        description="Cost in USD per 1,000 input tokens, if known.",
    )
    cost_per_1k_output_tokens: float | None = Field(
        default=None,
        ge=0,
        description="Cost in USD per 1,000 output tokens, if known.",
    )

    @field_validator("capabilities", mode="before")
    @classmethod
    def _coerce_to_frozenset(
        cls,
        value: Any,
    ) -> frozenset[ModelCapability]:
        if isinstance(value, frozenset):
            return value
        if isinstance(value, (set, list, tuple)):
            return frozenset(value)
        msg = f"Expected a set or list of capabilities, got {type(value)}"
        raise TypeError(msg)
