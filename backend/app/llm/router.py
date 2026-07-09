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

from app.config.settings import Settings
from app.core.logging import get_logger
from app.domain.provider import ProviderSpec
from app.observability import provider_request_duration, provider_request_total
from app.domain.stream import StreamEvent
from app.llm.base import LLMProvider
from app.llm.exceptions import (
    GenerationError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderTimeoutError,
    RouterNoProviderError,
)
from app.llm.factory import create_provider_adapter
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

    Provider adapters are registered via :meth:`register_adapter` and
    :meth:`remove_adapter` so that the provider service can sync database
    state into the in-memory registry at startup and on changes.

    Usage::

        router = LLMRouter(
            settings=settings,
            default_provider_id="ollama",
            provider_preference=["ollama", "lm_studio"],
        )
        response = await router.generate(request)
    """

    def __init__(
        self,
        settings: Settings,
        default_provider_id: str = _DEFAULT_PROVIDER_ID,
        provider_preference: Sequence[str] | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        circuit_breaker_threshold: int = _DEFAULT_CIRCUIT_THRESHOLD,
        circuit_breaker_reset_seconds: float = _DEFAULT_CIRCUIT_RESET_SECONDS,
    ) -> None:
        self._settings = settings
        self._registry = ProviderRegistry()
        self._default_provider_id = default_provider_id
        self._provider_preference = list(provider_preference) if provider_preference else []
        self._max_retries = max_retries
        self._circuit_threshold = circuit_breaker_threshold
        self._circuit_reset = circuit_breaker_reset_seconds

        self._failure_counts: dict[str, int] = {}
        self._circuit_open_until: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Provider adapter lifecycle — called by ProviderService
    # ------------------------------------------------------------------ #

    def register_adapter(self, spec: ProviderSpec) -> LLMProvider:
        """Create and register a provider adapter from a spec.

        The adapter is created via :func:`create_provider_adapter` and
        stored in the in-memory registry. Subsequent requests will find
        it immediately.

        Args:
            spec: The provider specification (from the database).

        Returns:
            The created adapter instance.
        """
        adapter = create_provider_adapter(spec, self._settings)
        self._registry.register(adapter)
        logger.info("llm.adapter_registered", provider_id=str(spec.id))
        return adapter

    def remove_adapter(self, provider_id: str) -> None:
        """Remove a provider adapter from the in-memory registry.

        Args:
            provider_id: The provider identifier to remove.
        """
        self._registry.remove(provider_id)
        self._failure_counts.pop(provider_id, None)
        self._circuit_open_until.pop(provider_id, None)
        logger.info("llm.adapter_removed", provider_id=provider_id)

    def refresh_adapter(self, spec: ProviderSpec) -> LLMProvider:
        """Replace an existing adapter with a new one from an updated spec.

        This is equivalent to calling :meth:`remove_adapter` followed by
        :meth:`register_adapter`.

        Args:
            spec: The updated provider specification.

        Returns:
            The new adapter instance.
        """
        self.remove_adapter(str(spec.id))
        return self.register_adapter(spec)

    async def load_providers_from_db(
        self,
        specs: list[ProviderSpec],
    ) -> None:
        """Load provider specs from the database into the in-memory registry.

        Only enabled providers are registered. Disabled providers are
        removed from the registry if they were previously registered.

        Args:
            specs: Provider specs from the database.
        """
        enabled_ids: set[str] = set()
        for spec in specs:
            if spec.is_enabled and str(spec.id) != "string":
                enabled_ids.add(str(spec.id))
                self.register_adapter(spec)
                logger.info(
                    "llm.provider_loaded",
                    provider_id=str(spec.id),
                    provider_type=spec.provider_type.value,
                )

        # Remove any previously-registered providers that are no longer
        # enabled or no longer exist in the DB.
        for existing in self._registry.list():
            if existing.provider_id not in enabled_ids:
                self.remove_adapter(existing.provider_id)
                logger.info(
                    "llm.provider_unloaded",
                    provider_id=existing.provider_id,
                )

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
                candidates = self._build_candidate_chain(request.provider)
                missed = []
                for pid in candidates:
                    if pid in tried:
                        missed.append(f"{pid} (already tried)")
                    elif self._is_circuit_open(pid):
                        missed.append(f"{pid} (circuit open)")
                    elif not self._registry.is_registered(pid):
                        missed.append(f"{pid} (not registered)")
                    else:
                        missed.append(f"{pid} (unknown)")
                msg = (
                    f"No provider available for request. "
                    f"Provider requested: {request.provider!r}, "
                    f"Default: {self._default_provider_id!r}, "
                    f"Candidates: {candidates}, "
                    f"Tried: {list(tried)}, "
                    f"Registered: {[p.provider_id for p in self._registry.list()]}"
                )
                raise RouterNoProviderError(msg)
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
                provider_request_total.labels(
                    provider=provider.provider_id,
                    model=request.model,
                    result="error",
                ).inc()
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

            except (GenerationError, ProviderAuthenticationError) as exc:
                last_error = exc
                self._record_failure(provider.provider_id)
                provider_request_total.labels(
                    provider=provider.provider_id,
                    model=request.model,
                    result="error",
                ).inc()
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
            error=str(last_error) if last_error else "No provider available",
        )
        raise GenerationError(
            f"Generation failed after {1 + self._max_retries} attempt(s). "
            f"Last error: {last_error}" if last_error else
            f"Generation failed after {1 + self._max_retries} attempt(s). "
            f"No provider was available.",
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
            error=str(last_error) if last_error else "No provider available",
        )
        raise GenerationError(
            f"Stream failed after {1 + self._max_retries} attempt(s): "
            f"{last_error}" if last_error else
            f"Stream failed after {1 + self._max_retries} attempt(s). "
            f"No provider was available.",
        ) from last_error

    def list_providers(self) -> list[LLMProvider]:
        """Return all registered provider adapters.

        Returns:
            A list of provider instances from the registry.
        """
        return self._registry.list()

    def is_provider_available(self, provider_id: str) -> bool:
        """Check whether a provider is registered and its circuit isn't open.

        Args:
            provider_id: The provider identifier to check.

        Returns:
            ``True`` if the provider is registered and healthy.
        """
        return (
            self._registry.is_registered(provider_id)
            and not self._is_circuit_open(provider_id)
        )

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
            try:
                targets = [self._registry.get(provider_id)]
            except RouterNoProviderError:
                return {provider_id: False}
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
        model_id = response.model or request.model
        provider_id = provider.provider_id
        elapsed_sec = elapsed_ms / 1000.0

        provider_request_duration.labels(
            provider=provider_id,
            model=model_id,
        ).observe(elapsed_sec)

        provider_request_total.labels(
            provider=provider_id,
            model=model_id,
            result="success",
        ).inc()

        usage = response.usage
        logger.info(
            "llm.request_complete",
            provider=provider_id,
            model=model_id,
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
