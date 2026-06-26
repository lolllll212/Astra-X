"""Provider domain-to-ORM mapper."""

from __future__ import annotations

from app.database.models.provider import ProviderModel
from app.domain.enums import ModelCapability, ProviderType
from app.domain.provider import ProviderID, ProviderSpec

__all__ = [
    "provider_from_model",
    "provider_to_model",
]


def provider_to_model(domain: ProviderSpec) -> ProviderModel:
    capabilities: list[str] = sorted(c.value for c in domain.supported_capabilities)
    return ProviderModel(
        id=str(domain.id),
        provider_type=domain.provider_type.value,
        display_name=domain.display_name,
        capabilities=capabilities,
    )


def provider_from_model(model: ProviderModel) -> ProviderSpec:
    capabilities: frozenset[ModelCapability] = frozenset(
        ModelCapability(c) for c in (model.capabilities or []) if c in ModelCapability._value2member_map_
    )
    return ProviderSpec(
        id=ProviderID(model.id),
        provider_type=ProviderType(model.provider_type),
        display_name=model.display_name,
        supported_capabilities=capabilities,
    )
