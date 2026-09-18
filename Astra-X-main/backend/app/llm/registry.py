"""Provider registry.

The :class:`ProviderRegistry` stores all available provider adapters and
provides lookup by provider ID. Providers register themselves once during
application startup so that the router can discover them dynamically.
"""

from __future__ import annotations

from app.llm.base import LLMProvider
from app.llm.exceptions import RouterNoProviderError

__all__ = [
    "ProviderRegistry",
]


class ProviderRegistry:
    """Thread-safe registry of LLM provider adapters.

    Usage::

        registry = ProviderRegistry()
        registry.register(ollama_provider)
        registry.register(lmstudio_provider)

        provider = registry.get("ollama")  # raises if missing
        all_providers = registry.list()
    """

    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}

    def register(self, provider: LLMProvider) -> None:
        """Register a provider adapter.

        If a provider with the same ``provider_id`` is already registered
        it is overwritten (last registration wins).

        Args:
            provider: An initialised provider adapter instance.
        """
        self._providers[provider.provider_id] = provider

    def get(self, provider_id: str) -> LLMProvider:
        """Retrieve a registered provider by its identifier.

        Args:
            provider_id: The provider identifier.

        Returns:
            The registered provider adapter.

        Raises:
            RouterNoProviderError: If no provider is registered under
                the given identifier.
        """
        provider = self._providers.get(provider_id)
        if provider is None:
            msg = f"No provider registered for '{provider_id}'"
            raise RouterNoProviderError(msg)
        return provider

    def list(self) -> list[LLMProvider]:
        """Return all registered provider adapters.

        Returns:
            A list of registered provider instances.
        """
        return list(self._providers.values())

    def remove(self, provider_id: str) -> None:
        """Remove a registered provider.

        Args:
            provider_id: The provider identifier to remove.
        """
        self._providers.pop(provider_id, None)

    def is_registered(self, provider_id: str) -> bool:
        """Check whether a provider is registered.

        Args:
            provider_id: The provider identifier.

        Returns:
            ``True`` if the provider is registered.
        """
        return provider_id in self._providers
