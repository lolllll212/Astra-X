"""Provider performance metrics for data-driven model selection.

The :class:`ProviderMetricsTracker` collects latency, token usage,
reflection scores, and success rates per ``(provider_id, model_id)``
pair. The :class:`ModelSelector` can then query historical performance
to make data-driven rather than purely heuristic selections.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# How many recent observations to keep per (provider, model) pair.
MAX_OBSERVATIONS_PER_KEY = 200


@dataclass
class ProviderObservation:
    """A single observation of provider/model performance."""

    timestamp: float
    latency_ms: float
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    success: bool
    reflection_confidence: float | None = None


@dataclass
class ProviderStats:
    """Aggregated statistics for a (provider_id, model_id) pair."""

    provider_id: str
    model_id: str
    total_calls: int = 0
    successful_calls: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_total_tokens: float = 0.0
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0
    avg_reflection_confidence: float = 0.0
    success_rate: float = 1.0
    last_observed: float = 0.0


class ProviderMetricsTracker:
    """Collects and aggregates provider performance metrics.

    Usage::

        tracker = ProviderMetricsTracker()
        tracker.record(
            provider_id="ollama",
            model_id="llama3.1",
            latency_ms=1500.0,
            total_tokens=512,
            success=True,
        )
        stats = tracker.get_stats("ollama", "llama3.1")
        best = tracker.best_for_profile(...)
    """

    def __init__(self) -> None:
        # provider_id -> model_id -> list[ProviderObservation]
        self._observations: dict[str, dict[str, list[ProviderObservation]]] = (
            defaultdict(lambda: defaultdict(list))
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        provider_id: str,
        model_id: str,
        *,
        latency_ms: float = 0.0,
        total_tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        success: bool = True,
        reflection_confidence: float | None = None,
    ) -> None:
        """Record a single observation."""
        obs = ProviderObservation(
            timestamp=time.time(),
            latency_ms=latency_ms,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            success=success,
            reflection_confidence=reflection_confidence,
        )
        bucket = self._observations[provider_id][model_id]
        bucket.append(obs)
        if len(bucket) > MAX_OBSERVATIONS_PER_KEY:
            bucket.pop(0)

        logger.debug(
            "metrics.recorded",
            provider=provider_id,
            model=model_id,
            latency_ms=round(latency_ms, 1),
            success=success,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_stats(
        self,
        provider_id: str,
        model_id: str,
    ) -> ProviderStats | None:
        """Return aggregated stats for a (provider, model) pair.

        Returns ``None`` if no observations exist.
        """
        bucket = self._observations.get(provider_id, {}).get(model_id)
        if not bucket:
            return None

        n = len(bucket)
        latencies = [o.latency_ms for o in bucket]
        latencies_sorted = sorted(latencies)
        successes = sum(1 for o in bucket if o.success)
        confidences = [
            o.reflection_confidence
            for o in bucket
            if o.reflection_confidence is not None
        ]

        return ProviderStats(
            provider_id=provider_id,
            model_id=model_id,
            total_calls=n,
            successful_calls=successes,
            avg_latency_ms=sum(latencies) / n,
            p50_latency_ms=latencies_sorted[n // 2] if latencies_sorted else 0.0,
            p95_latency_ms=latencies_sorted[int(n * 0.95) - 1] if n > 1 else (latencies_sorted[-1] if latencies_sorted else 0.0),
            avg_total_tokens=sum(o.total_tokens for o in bucket) / n,
            avg_prompt_tokens=sum(o.prompt_tokens for o in bucket) / n,
            avg_completion_tokens=sum(o.completion_tokens for o in bucket) / n,
            avg_reflection_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
            success_rate=successes / n if n > 0 else 1.0,
            last_observed=bucket[-1].timestamp,
        )

    def best_for_profile(
        self,
        *,
        min_success_rate: float = 0.7,
        prefer_speed: bool = False,
        prefer_reasoning: bool = False,
    ) -> tuple[str, str] | None:
        """Return the best (provider_id, model_id) based on historical data.

        Args:
            min_success_rate: Minimum acceptable success rate.
            prefer_speed: If True, rank by lowest latency.
            prefer_reasoning: If True, rank by highest reflection confidence.

        Returns:
            The best ``(provider_id, model_id)`` or ``None`` if no data.
        """
        candidates: list[tuple[float, str, str]] = []

        for provider_id, models in self._observations.items():
            for model_id, bucket in models.items():
                stats = self.get_stats(provider_id, model_id)
                if stats is None or stats.success_rate < min_success_rate:
                    continue

                # Composite score: success_rate * (speed or confidence bonus).
                score = stats.success_rate
                if prefer_speed:
                    # Invert latency so faster = higher score.
                    avg_latency = max(stats.avg_latency_ms, 1.0)
                    score *= 1.0 + (1.0 / math.log2(avg_latency + 1.0))
                if prefer_reasoning:
                    score *= 1.0 + stats.avg_reflection_confidence * 0.5

                candidates.append((score, provider_id, model_id))

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1], candidates[0][2]

    def list_all_stats(self) -> list[ProviderStats]:
        """Return aggregated stats for all observed (provider, model) pairs."""
        result: list[ProviderStats] = []
        for provider_id, models in self._observations.items():
            for model_id in models:
                stats = self.get_stats(provider_id, model_id)
                if stats is not None:
                    result.append(stats)
        return result

    def clear(self) -> None:
        """Reset all observations."""
        self._observations.clear()
