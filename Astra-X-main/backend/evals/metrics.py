from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from evals.models import EvalResult


class MetricsTracker:
    """Collects and aggregates eval metrics across cases.

    Tracks pass/fail, latency distribution, token usage,
    per-tag breakdowns, and arbitrary custom metrics
    recorded by each scenario.
    """

    def __init__(self) -> None:
        self._results: list[EvalResult] = []
        self._by_tag: dict[str, list[EvalResult]] = defaultdict(list)
        self._custom: dict[str, list[float]] = defaultdict(list)

    # ── Recording ────────────────────────────────────────────────────

    def record(self, result: EvalResult) -> None:
        self._results.append(result)
        for tag in result.tags:
            self._by_tag[tag].append(result)

    def record_custom(self, name: str, value: float) -> None:
        """Record a per-case custom metric value.

        Use this in :meth:`BaseEval.run_case` to surface
        scenario-specific diagnostics such as tool accuracy,
        hallucination rate, or replan count.

        Args:
            name: Metric name (e.g. ``"tool_selection_accuracy"``).
            value: A numeric value for this case.
        """
        self._custom[name].append(value)

    # ── Pass / fail ──────────────────────────────────────────────────

    @property
    def total(self) -> int:
        return len(self._results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self._results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self._results if not r.passed)

    @property
    def success_rate(self) -> float:
        if not self._results:
            return 0.0
        return self.passed / self.total

    # ── Latency ──────────────────────────────────────────────────────

    @property
    def total_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self._results)

    @property
    def avg_latency_ms(self) -> float:
        if not self._results:
            return 0.0
        return self.total_latency_ms / self.total

    @property
    def p50_latency_ms(self) -> float:
        vals = self._sorted_latencies()
        if not vals:
            return 0.0
        return vals[len(vals) // 2]

    @property
    def p95_latency_ms(self) -> float:
        vals = self._sorted_latencies()
        if not vals:
            return 0.0
        idx = math.ceil(0.95 * len(vals)) - 1
        return vals[max(0, idx)]

    @property
    def p99_latency_ms(self) -> float:
        vals = self._sorted_latencies()
        if not vals:
            return 0.0
        idx = math.ceil(0.99 * len(vals)) - 1
        return vals[max(0, idx)]

    def _sorted_latencies(self) -> list[float]:
        return sorted(r.latency_ms for r in self._results)

    # ── Tokens ───────────────────────────────────────────────────────

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens_used for r in self._results if r.tokens_used)

    # ── Per-tag breakdown ────────────────────────────────────────────

    def tag_success_rate(self, tag: str) -> float:
        tagged = self._by_tag.get(tag, [])
        if not tagged:
            return 0.0
        return sum(1 for r in tagged if r.passed) / len(tagged)

    # ── Custom metrics ───────────────────────────────────────────────

    @property
    def custom_metric_names(self) -> list[str]:
        return list(self._custom)

    def custom_summary(self, name: str) -> dict[str, float]:
        """Aggregate summary for a single custom metric.

        Returns ``min``, ``max``, ``avg``, ``sum``, and ``count``.
        Returns all zeros if the metric name is unknown.
        """
        values = self._custom.get(name, [])
        if not values:
            return {"min": 0.0, "max": 0.0, "avg": 0.0, "sum": 0.0, "count": 0}

        return {
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "avg": round(sum(values) / len(values), 4),
            "sum": round(sum(values), 4),
            "count": len(values),
        }

    def all_custom_summaries(self) -> dict[str, dict[str, float]]:
        return {name: self.custom_summary(name) for name in self._custom}

    # ── Combined summary ─────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "success_rate": round(self.success_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p50_latency_ms": round(self.p50_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "p99_latency_ms": round(self.p99_latency_ms, 1),
            "total_tokens": self.total_tokens,
        }

        custom = self.all_custom_summaries()
        if custom:
            # Flatten the most useful aggregate for each custom metric
            flat: dict[str, float] = {}
            for name, agg in custom.items():
                flat[f"custom_{name}_avg"] = agg["avg"]
                flat[f"custom_{name}_min"] = agg["min"]
                flat[f"custom_{name}_max"] = agg["max"]
                flat[f"custom_{name}_sum"] = agg["sum"]
            result["custom_metrics"] = custom
            result.update(flat)

        return result

    def errors(self) -> list[tuple[str, str]]:
        return [(r.case_id, r.error) for r in self._results if r.error]
