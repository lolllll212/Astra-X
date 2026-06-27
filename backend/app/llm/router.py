"""LLM request router.

The :class:`LLMRouter` is the single entry point for all LLM requests.
It selects a provider adapter based on the requested model and provider,
delegates the generation, and handles fallback logic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.domain.stream import StreamEvent
from app.llm.base import LLMProvider
from app.llm.exceptions import RouterNoProviderError
from app.llm.models import CompletionRequest, CompletionResponse
from app.llm.registry import ProviderRegistry

__all__ = [
    "LLMRouter",
]

_DEFAULT_PROVIDER_ID = "ollama"


class LLMRouter:
    """Routes completion requests to the appropriate provider.

    Usage::

        router = LLMRouter(registry, default_provider_id="ollama")
        response = await router.generate(request)
    """

    def __init__(
        self,
        registry: ProviderRegistry,
        default_provider_id: str = _DEFAULT_PROVIDER_ID,
    ) -> None:
        self._registry = registry
        self._default_provider_id = default_provider_id

    def _resolve_provider(self, request: CompletionRequest) -> LLMProvider:
        """Select the provider for a given request.

        Resolution order:
        1. ``request.provider`` (explicit provider ID from caller).
        2. ``self._default_provider_id``.

        Args:
            request: The incoming completion request.

        Returns:
            A registered provider adapter.

        Raises:
            RouterNoProviderError: If no suitable provider is found.
        """
        provider_id = request.provider or self._default_provider_id
        try:
            return self._registry.get(provider_id)
        except RouterNoProviderError:
            available = [p.provider_id for p in self._registry.list()]
            msg = (
                f"No provider found for '{provider_id}'. "
                f"Available: {available}"
            )
            raise RouterNoProviderError(msg) from None

    async def generate(
        self,
        request: CompletionRequest,
    ) -> CompletionResponse:
        """Generate a non-streaming completion.

        Args:
            request: The completion request.

        Returns:
            The provider's response.

        Raises:
            RouterNoProviderError: If no suitable provider is registered.
        """
        provider = self._resolve_provider(request)
        return await provider.generate(request)

    async def generate_stream(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[StreamEvent]:
        """Generate a streaming completion.

        Args:
            request: The completion request (``request.stream`` should
                be ``True``).

        Yields:
            Stream events from the provider.

        Raises:
            RouterNoProviderError: If no suitable provider is registered.
        """
        provider = self._resolve_provider(request)
        async for event in provider.generate_stream(request):
            yield event

    async def check_health(self, provider_id: str | None = None) -> dict[str, bool]:
        """Check health of one or all providers.

        Args:
            provider_id: If set, checks only this provider. Otherwise
                checks every registered provider.

        Returns:
            A mapping of provider ID to health status.
        """
        providers: list[LLMProvider]
        if provider_id is not None:
            providers = [self._registry.get(provider_id)]
        else:
            providers = self._registry.list()

        result: dict[str, bool] = {}
        for provider in providers:
            try:
                result[provider.provider_id] = await provider.check_health()
            except Exception:
                result[provider.provider_id] = False
        return result
