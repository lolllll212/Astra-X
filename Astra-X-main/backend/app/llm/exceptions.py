"""LLM-layer exceptions.

All exceptions raised by provider adapters, the registry, or the router
inherit from :class:`LLMError`. API-level handlers should catch
:class:`LLMError` and translate it to a structured HTTP response.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base exception for all LLM-layer errors."""


class ProviderTimeoutError(LLMError):
    """Raised when a provider request exceeds the configured timeout."""


class ProviderConnectionError(LLMError):
    """Raised when a provider is unreachable or connection is refused."""


class ProviderAuthenticationError(LLMError):
    """Raised when authentication with the provider fails."""


class ModelNotSupportedError(LLMError):
    """Raised when the requested model is not available on the selected provider."""


class GenerationError(LLMError):
    """Raised when a generation call returns an error response."""


class RouterNoProviderError(LLMError):
    """Raised when the router cannot find a suitable provider for the request."""
