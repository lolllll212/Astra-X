"""Provider adapter factory.

Creates a concrete :class:`LLMProvider` adapter from a
:class:`ProviderSpec`, mapping the ``provider_type`` enum to the
correct adapter class and wiring configuration from the spec.
"""

from __future__ import annotations

from app.config.settings import Settings
from app.domain.enums import ProviderType
from app.domain.provider import ProviderSpec
from app.llm.base import LLMProvider
from app.llm.providers.lmstudio import LMStudioProvider
from app.llm.providers.ollama import OllamaProvider
from app.llm.providers.openai_compatible import OpenAICompatibleProvider

__all__ = [
    "create_provider_adapter",
    "provider_type_to_default_base_url",
]


def provider_type_to_default_base_url(provider_type: ProviderType, settings: Settings) -> str | None:
    """Return the default base URL for a given provider type based on settings.

    Args:
        provider_type: The provider backend type.
        settings: Application settings.

    Returns:
        The default base URL for this provider type, or ``None`` if
        the type has no configured default.
    """
    mapping: dict[ProviderType, str | None] = {
        ProviderType.OLLAMA: settings.ollama_base_url,
        ProviderType.LM_STUDIO: settings.lm_studio_base_url,
        ProviderType.OPENAI_COMPATIBLE: settings.openai_compatible_base_url,
    }
    return mapping.get(provider_type)


def create_provider_adapter(
    spec: ProviderSpec,
    settings: Settings,
    timeout_seconds: float | None = None,
) -> LLMProvider:
    """Create a concrete :class:`LLMProvider` adapter from a provider spec.

    Args:
        spec: The provider specification from the database.
        settings: Application settings (used for defaults).
        timeout_seconds: Optional request timeout override.

    Returns:
        An initialised provider adapter.

    Raises:
        ValueError: If the provider type is not recognised.
    """
    timeout = timeout_seconds if timeout_seconds is not None else settings.llm_request_timeout_seconds
    base_url: str | None = spec.base_url or provider_type_to_default_base_url(
        spec.provider_type, settings,
    )
    api_key_str: str | None = (
        spec.api_key.get_secret_value() if spec.api_key is not None else None
    )

    provider_id = str(spec.id)

    if spec.provider_type == ProviderType.OLLAMA:
        return OllamaProvider(
            base_url=base_url or "http://localhost:11434",
            model=spec.models[0] if spec.models else settings.default_llm_model,
            timeout_seconds=timeout,
            provider_id=provider_id,
        )

    if spec.provider_type == ProviderType.LM_STUDIO:
        return LMStudioProvider(
            base_url=base_url or "http://localhost:1234/v1",
            model=spec.models[0] if spec.models else "local-model",
            timeout_seconds=timeout,
            provider_id=provider_id,
        )

    if spec.provider_type == ProviderType.OPENAI_COMPATIBLE:
        return OpenAICompatibleProvider(
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key_str,
            model=spec.models[0] if spec.models else "gpt-4o",
            timeout_seconds=timeout,
            provider_id=provider_id,
        )

    raise ValueError(
        f"Unknown provider type: {spec.provider_type!r}. "
        f"Supported types: {[t.value for t in ProviderType]}",
    )
