"""Abstract LLM provider interface.

Every provider adapter (Ollama, LM Studio, OpenAI-compatible, etc.)
inherits from :class:`LLMProvider` and implements its abstract methods.
This keeps provider-specific code isolated behind a uniform contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.domain.enums import ModelCapability, OpenAIProtocol
from app.domain.stream import StreamEvent
from app.llm.models import CompletionRequest, CompletionResponse


class LLMProvider(ABC):
    """Abstract interface for an LLM provider.

    Subclasses must set :attr:`provider_id` and :attr:`model_id` as
    instance attributes and implement :meth:`generate` (and optionally
    :meth:`generate_stream`).
    """

    provider_id: str
    model_id: str

    @property
    def protocol(self) -> OpenAIProtocol:
        """Which protocol this provider speaks.

        Subclasses may override to return the correct protocol for their
        backend.  Defaults to :attr:`OpenAIProtocol.CHAT_COMPLETIONS`.
        """
        return OpenAIProtocol.CHAT_COMPLETIONS

    @abstractmethod
    async def generate(
        self,
        request: CompletionRequest,
    ) -> CompletionResponse:
        """Send a non-streaming completion request.

        Args:
            request: The completion parameters and messages.

        Returns:
            A complete response with the assistant's message and usage.

        Raises:
            ProviderConnectionError: If the provider is unreachable.
            ProviderAuthenticationError: If authentication fails.
            GenerationError: If the provider returns an error response.
        """
        ...

    def generate_stream(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[StreamEvent]:
        """Send a streaming completion request.

        The default implementation raises :class:`NotImplementedError`.
        Providers that support streaming override this method with an
        async generator.

        Args:
            request: The completion parameters and messages (with
                ``stream=True``).

        Yields:
            :class:`StreamEvent` instances as they arrive from the provider.

        Raises:
            NotImplementedError: If the provider does not support streaming.
            ProviderConnectionError: If the provider is unreachable.
        """
        msg = f"{self.__class__.__name__} does not support streaming"
        raise NotImplementedError(msg)

    async def check_health(self) -> bool:
        """Check whether the provider backend is reachable.

        Returns:
            ``True`` if the provider responds to a lightweight health
            probe, ``False`` otherwise.
        """
        return True

    async def list_models(self) -> list[str]:
        """Return the list of models this provider can serve.

        Returns:
            A list of model identifier strings.
        """
        return [self.model_id]

    def supports_capability(self, capability: ModelCapability) -> bool:
        """Check whether this provider supports a given capability.

        Args:
            capability: The capability to check.

        Returns:
            ``True`` if the capability is supported.
        """
        return capability in (ModelCapability.CHAT, ModelCapability.STREAMING)
