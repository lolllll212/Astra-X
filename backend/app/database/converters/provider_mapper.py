from __future__ import annotations

from pydantic import SecretStr

from app.database.models.provider import ProviderModel
from app.domain.enums import ModelCapability, OpenAIProtocol, ProviderType
from app.domain.provider import ProviderID, ProviderSpec

__all__ = [
    "provider_from_model",
    "provider_to_model",
]


def provider_to_model(domain: ProviderSpec) -> ProviderModel:
    capabilities: list[str] = sorted(c.value for c in domain.supported_capabilities)
    api_key_value: str | None = (
        domain.api_key.get_secret_value() if domain.api_key is not None else None
    )
    return ProviderModel(
        id=str(domain.id),
        provider_type=domain.provider_type.value,
        display_name=domain.display_name,
        base_url=domain.base_url,
        api_key=api_key_value,
        protocol=domain.protocol.value,
        api_version_url=domain.api_version_url,
        capabilities=capabilities,
        supports_responses_api=domain.supports_responses_api,
        supports_vision=domain.supports_vision,
        max_tool_calls_per_request=domain.max_tool_calls_per_request,
        models=domain.models,
        is_enabled=domain.is_enabled,
    )


def provider_from_model(model: ProviderModel) -> ProviderSpec:
    capabilities: frozenset[ModelCapability] = frozenset(
        ModelCapability(c) for c in (model.capabilities or []) if c in ModelCapability._value2member_map_
    )

    provider_type: ProviderType
    try:
        provider_type = ProviderType(model.provider_type)
    except ValueError:
        provider_type = ProviderType.OLLAMA

    protocol: OpenAIProtocol
    try:
        protocol = OpenAIProtocol(model.protocol) if model.protocol else OpenAIProtocol.CHAT_COMPLETIONS
    except ValueError:
        protocol = OpenAIProtocol.CHAT_COMPLETIONS

    return ProviderSpec(
        id=ProviderID(model.id),
        provider_type=provider_type,
        display_name=model.display_name,
        is_enabled=model.is_enabled,
        base_url=model.base_url,
        api_key=SecretStr(model.api_key) if model.api_key else None,
        protocol=protocol,
        api_version_url=model.api_version_url,
        supported_capabilities=capabilities,
        supports_responses_api=model.supports_responses_api or False,
        supports_vision=model.supports_vision or False,
        max_tool_calls_per_request=model.max_tool_calls_per_request or 10,
        models=model.models or [],
        created_at=model.created_at,
        updated_at=model.updated_at,
    )
