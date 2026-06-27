"""LLM request router.

The :class:`LLMRouter` is the single entry point for all LLM requests.
It is responsible for:

* **Model selection** — picking the model to use when none is explicitly given.
* **Provider selection** — resolving a provider ID to a registered adapter.
* **Health gating** — skipping providers whose circuit breaker is open.
* **Fallback** — trying the next provider when the primary fails.
* **Retry** — exponential backoff for transient errors.
* **Circuit breaker** — stopping requests to repeatedly failing providers.
* **Metrics** — tracking latency, token counts, and success/failure.
* **Logging** — structured logs with provider, model, latency, and token counts.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence

from app.core.logging import get_logger
from app.domain.stream import StreamEvent
from app.llm.base import LLMProvider
from app.llm.exceptions import (
    GenerationError,
    ProviderConnectionError,
    ProviderTimeoutError,
    RouterNoProviderError,
)
from app.llm.models import CompletionRequest, CompletionResponse
from app.llm.registry import ProviderRegistry

__all__ = [
    "LLMRouter",
]

_DEFAULT_PROVIDER_ID = "ollama"
_DEFAULT_MAX_RETRIES = 2
_DEFAULT_CIRCUIT_THRESHOLD = 3
_DEFAULT_CIRCUIT_RESET_SECONDS = 30.0

logger = get_logger(__name__)


class LLMRouter:
    """Orchestrates LLM generation with retry, fallback, and circuit breaking.

    Usage::

        router = LLMRouter(
            registry=registry,
            default_provider_id="ollama",
            provider_preference=["ollama", "lm_studio"],
        )
        response = await router.generate(request)
    """

    def __init__(
        self,
        registry: ProviderRegistry,
        default_provider_id: str = _DEFAULT_PROVIDER_ID,
        provider_preference: Sequence[str] | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        circuit_breaker_threshold: int = _DEFAULT_CIRCUIT_THRESHOLD,
        circuit_breaker_reset_seconds: float = _DEFAULT_CIRCUIT_RESET_SECONDS,
    ) -> None:
        self._registry = registry
        self._default_provider_id = default_provider_id
        self._provider_preference = list(provider_preference) if provider_preference else []
        self._max_retries = max_retries
        self._circuit_threshold = circuit_breaker_threshold
        self._circuit_reset = circuit_breaker_reset_seconds

        self._failure_counts: dict[str, int] = {}
        self._circuit_open_until: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def generate(
        self,
        request: CompletionRequest,
    ) -> CompletionResponse:
        """Generate a non-streaming completion with retry and fallback.

        Args:
            request: The completion request.

        Returns:
            The provider's response.

        Raises:
            RouterNoProviderError: If no suitable provider is registered.
            GenerationError: If all retries and fallbacks are exhausted.
        """
        start_time = time.monotonic()
        last_error: Exception | None = None
        tried: set[str] = set()

        for attempt in range(1 + self._max_retries):
            provider = self._acquire_provider(request, exclude=tried)
            if provider is None:
                break
            tried.add(provider.provider_id)

            try:
                response = await provider.generate(request)
                self._record_success(provider.provider_id)
                elapsed_ms = (time.monotonic() - start_time) * 1000
                self._log_success(provider, request, response, elapsed_ms)
                return response

            except (ProviderTimeoutError, ProviderConnectionError) as exc:
                last_error = exc
                self._record_failure(provider.provider_id)
                logger.warning(
                    "llm.attempt_failed",
                    provider=provider.provider_id,
                    model=request.model,
                    attempt=attempt + 1,
                    max_attempts=1 + self._max_retries,
                    error=str(exc),
                )

                if attempt < self._max_retries:
                    backoff = min(2.0 ** attempt, 10.0)
                    await self._sleep(backoff)

            except Exception as exc:
                last_error = exc
                self._record_failure(provider.provider_id)
                logger.error(
                    "llm.attempt_failed",
                    provider=provider.provider_id,
                    model=request.model,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                break

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.error(
            "llm.request_failed",
            provider=request.provider or self._default_provider_id,
            model=request.model,
            elapsed_ms=round(elapsed_ms),
            error=str(last_error),
        )
        raise GenerationError(
            f"Generation failed after {1 + self._max_retries} attempt(s): {last_error}",
        ) from last_error

    async def generate_stream(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[StreamEvent]:
        """Generate a streaming completion with retry and fallback.

        Streaming retry is limited to failures that occur *before* the
        first event is yielded. Once the first event arrives the stream
        is considered established and errors propagate directly.

        Args:
            request: The completion request.

        Yields:
            Stream events from the provider.

        Raises:
            RouterNoProviderError: If no suitable provider is registered.
            GenerationError: If all retries and fallbacks are exhausted.
        """
        start_time = time.monotonic()
        last_error: Exception | None = None
        tried: set[str] = set()
        stream_started = False

        provider: LLMProvider | None = None

        for attempt in range(1 + self._max_retries):
            provider = self._acquire_provider(request, exclude=tried)
            if provider is None:
                break
            tried.add(provider.provider_id)

            try:
                async for event in provider.generate_stream(request):
                    stream_started = True
                    yield event

                self._record_success(provider.provider_id)
                elapsed_ms = (time.monotonic() - start_time) * 1000
                logger.info(
                    "llm.stream_complete",
                    provider=provider.provider_id,
                    model=request.model,
                    elapsed_ms=round(elapsed_ms),
                )
                return

            except (ProviderTimeoutError, ProviderConnectionError) as exc:
                if stream_started:
                    raise
                last_error = exc
                self._record_failure(provider.provider_id)

                if attempt < self._max_retries:
                    backoff = min(2.0 ** attempt, 10.0)
                    await self._sleep(backoff)

            except Exception as exc:
                if stream_started:
                    raise
                last_error = exc
                self._record_failure(provider.provider_id)
                break

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.error(
            "llm.stream_failed",
            provider=provider.provider_id if provider else "unknown",
            model=request.model,
            elapsed_ms=round(elapsed_ms),
            error=str(last_error),
        )
        raise GenerationError(
            f"Stream failed after {1 + self._max_retries} attempt(s): {last_error}",
        ) from last_error

    async def check_health(
        self,
        provider_id: str | None = None,
    ) -> dict[str, bool]:
        """Check health of one or all registered providers.

        Args:
            provider_id: If set, checks only this provider.

        Returns:
            A mapping of provider ID to health status.
        """
        targets: list[LLMProvider]
        if provider_id is not None:
            targets = [self._registry.get(provider_id)]
        else:
            targets = self._registry.list()

        result: dict[str, bool] = {}
        for provider in targets:
            if self._is_circuit_open(provider.provider_id):
                result[provider.provider_id] = False
                continue
            try:
                result[provider.provider_id] = await provider.check_health()
            except Exception:
                result[provider.provider_id] = False
        return result

    # ------------------------------------------------------------------ #
    # Provider selection
    # ------------------------------------------------------------------ #

    def _acquire_provider(
        self,
        request: CompletionRequest,
        exclude: set[str] | None = None,
    ) -> LLMProvider | None:
        """Select the first healthy, non-excluded provider for the request.

        Resolution order:
        1. ``request.provider`` (explicit).
        2. ``self._default_provider_id``.
        3. ``self._provider_preference`` (in declared order).
        4. Every remaining registered provider.

        Providers whose circuit breaker is open or that are in the
        *exclude* set are skipped.

        Args:
            request: The completion request.
            exclude: Provider IDs to skip (already tried).

        Returns:
            A provider adapter, or ``None`` if no candidate is available.
        """
        excluded = exclude or set()
        candidates = self._build_candidate_chain(request.provider)

        for pid in candidates:
            if pid in excluded:
                continue
            if self._is_circuit_open(pid):
                continue
            try:
                return self._registry.get(pid)
            except RouterNoProviderError:
                continue

        logger.warning(
            "llm.no_provider_available",
            candidates=candidates,
            excluded=list(excluded),
        )
        return None

    def _build_candidate_chain(self, explicit_provider: str | None) -> list[str]:
        """Build an ordered list of provider IDs to try.

        Args:
            explicit_provider: Provider explicitly requested, if any.

        Returns:
            An ordered list of provider IDs (deduplicated).
        """
        chain: list[str] = []
        seen: set[str] = set()

        def add(pid: str) -> None:
            if pid not in seen:
                chain.append(pid)
                seen.add(pid)

        if explicit_provider:
            add(explicit_provider)

        add(self._default_provider_id)

        for pid in self._provider_preference:
            add(pid)

        for provider in self._registry.list():
            add(provider.provider_id)

        return chain

    # ------------------------------------------------------------------ #
    # Circuit breaker
    # ------------------------------------------------------------------ #

    def _is_circuit_open(self, provider_id: str) -> bool:
        """Check whether the circuit breaker is open for a provider.

        If the circuit has been open longer than the reset period it is
        half-opened (failure count reset, provider tried again).

        Args:
            provider_id: The provider to check.

        Returns:
            ``True`` if the provider should be skipped.
        """
        until = self._circuit_open_until.get(provider_id)
        if until is None:
            return False
        if time.monotonic() >= until:
            del self._circuit_open_until[provider_id]
            self._failure_counts[provider_id] = 0
            return False
        return True

    def _record_success(self, provider_id: str) -> None:
        """Reset the failure count and close the circuit.

        Args:
            provider_id: The provider that succeeded.
        """
        self._failure_counts[provider_id] = 0
        self._circuit_open_until.pop(provider_id, None)

    def _record_failure(self, provider_id: str) -> None:
        """Increment the failure count and open the circuit if threshold
        is reached.

        Args:
            provider_id: The provider that failed.
        """
        count = self._failure_counts.get(provider_id, 0) + 1
        self._failure_counts[provider_id] = count
        if count >= self._circuit_threshold:
            self._circuit_open_until[provider_id] = (
                time.monotonic() + self._circuit_reset
            )
            logger.warning(
                "llm.circuit_open",
                provider=provider_id,
                reset_seconds=self._circuit_reset,
            )

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def _log_success(
        self,
        provider: LLMProvider,
        request: CompletionRequest,
        response: CompletionResponse,
        elapsed_ms: float,
    ) -> None:
        """Emit a structured log line for a successful generation.

        Args:
            provider: The provider that served the request.
            request: The original request.
            response: The completed response.
            elapsed_ms: Wall-clock time in milliseconds.
        """
        usage = response.usage
        logger.info(
            "llm.request_complete",
            provider=provider.provider_id,
            model=response.model or request.model,
            latency_ms=round(elapsed_ms),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            finish_reason=response.finish_reason,
        )

    # ------------------------------------------------------------------ #
    # Testability hook
    # ------------------------------------------------------------------ #

    async def _sleep(self, duration: float) -> None:
        """Async sleep, overridable in tests to avoid real waits."""
        import asyncio
        await asyncio.sleep(duration)
