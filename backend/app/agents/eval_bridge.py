"""Closed-loop learning bridge between evaluation results and the
pattern/ranking system.

The :class:`EvalFeedbackBridge` reads evaluation report JSON files,
classifies each suite into a domain, computes pass/fail rates, and feeds
them back into:

1. Pattern success scores (boosting/decaying patterns that match eval cases).
2. Ranking weight adaptation (adjusting per-domain weights when eval
   outcomes consistently favour certain factors).

Usage::

    bridge = EvalFeedbackBridge(learning_manager)
    summary = await bridge.ingest_report("path/to/report.json")
    # summary contains per-domain deltas
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.agents.learning_store import GoalDomain, classify_goal
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.agents.learning_manager import LearningManager

logger = get_logger(__name__)

# Map eval suite names to goal domains for direct classification.
SUITE_DOMAIN_MAP: dict[str, GoalDomain] = {
    "chat": GoalDomain.GENERAL,
    "tool_selection": GoalDomain.CODING,
    "planning": GoalDomain.REASONING,
    "reflection": GoalDomain.REASONING,
    "memory_retrieval": GoalDomain.GENERAL,
    "rag": GoalDomain.RESEARCH,
    "streaming": GoalDomain.GENERAL,
    "error_recovery": GoalDomain.CODING,
    "multi_turn": GoalDomain.GENERAL,
    "long_context": GoalDomain.GENERAL,
    "parallel_tool": GoalDomain.CODING,
    "provider_failover": GoalDomain.GENERAL,
    "streaming_cancel": GoalDomain.GENERAL,
    "memory_consolidation": GoalDomain.GENERAL,
    "mixed_input": GoalDomain.VISION,
}


@dataclass
class DomainEvalSummary:
    """Aggregated eval feedback for a single domain."""

    domain: GoalDomain
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    avg_latency_ms: float = 0.0
    weighted_success_rate: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_cases == 0:
            return 0.0
        return self.passed / max(self.total_cases, 1)


@dataclass
class EvalFeedbackSummary:
    """Overall result of ingesting an eval report."""

    report_path: str
    suites_processed: int = 0
    domains: dict[str, DomainEvalSummary] = field(default_factory=dict)
    weight_adjustments: dict[str, dict[str, float]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class EvalFeedbackBridge:
    """Bridges eval results into the learning system.

    Args:
        learning_manager: The active learning manager (can be ``None``
            in which case ``ingest_report`` is a no-op).
    """

    def __init__(
        self,
        learning_manager: LearningManager | None = None,
    ) -> None:
        self._learning_manager = learning_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest_report(self, report_path: str) -> EvalFeedbackSummary:
        """Read an eval report JSON file and feed results into learning.

        Returns:
            A summary of what was ingested, including per-domain metrics
            and any weight adjustments applied.
        """
        summary = EvalFeedbackSummary(report_path=report_path)

        if self._learning_manager is None:
            summary.errors.append("No learning manager available; skipping feedback.")
            return summary

        try:
            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            summary.errors.append(f"Failed to read report: {exc}")
            return summary

        suites = report.get("suites", [])
        if not suites:
            summary.errors.append("Report contains no suites.")
            return summary

        domain_accumulators: dict[str, DomainEvalSummary] = {}

        for suite in suites:
            suite_name = suite.get("name", "unknown")
            suite_summary = suite.get("summary", {})
            total = suite_summary.get("total", 0)
            passed = suite_summary.get("passed", 0)
            failed = suite_summary.get("failed", 0)
            avg_lat = suite_summary.get("avg_latency_ms", 0.0)

            domain = SUITE_DOMAIN_MAP.get(suite_name)
            if domain is None:
                domain = classify_goal(suite_name)

            if domain.value not in domain_accumulators:
                domain_accumulators[domain.value] = DomainEvalSummary(domain=domain)

            acc = domain_accumulators[domain.value]
            acc.total_cases += total
            acc.passed += passed
            acc.failed += failed
            acc.avg_latency_ms = (
                acc.avg_latency_ms * (acc.total_cases - total) / max(acc.total_cases, 1)
                + avg_lat * total / max(acc.total_cases, 1)
            )
            summary.suites_processed += 1

        # Weighted success rate: blend domain-wide eval success with
        # existing pattern success so we don't overreact to a single run.
        for domain_key, acc in domain_accumulators.items():
            eval_sr = acc.success_rate
            pattern_sr = await self._get_pattern_success_rate(domain_key)
            alpha = 0.3  # how much weight to give eval vs historical pattern data
            acc.weighted_success_rate = (
                alpha * eval_sr + (1.0 - alpha) * pattern_sr
            )
            logger.info(
                "eval_bridge.domain",
                domain=domain_key,
                cases=acc.total_cases,
                pass_rate=round(eval_sr, 2),
                pattern_sr=round(pattern_sr, 2),
                blended=round(acc.weighted_success_rate, 2),
            )

        summary.domains = domain_accumulators

        # Adapt per-domain ranking weights based on eval feedback.
        adjustments = await self._adapt_weights(domain_accumulators)
        summary.weight_adjustments = adjustments

        logger.info(
            "eval_bridge.complete",
            suites=summary.suites_processed,
            domains=list(domain_accumulators.keys()),
            errors=len(summary.errors),
        )
        return summary

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _get_pattern_success_rate(self, domain: str) -> float:
        """Compute the average success rate of all cached patterns for a domain."""
        if self._learning_manager is None:
            return 0.5
        patterns = self._learning_manager.get_patterns_for_domain(domain)
        if not patterns:
            return 0.5
        rates = [p.success_count / max(p.total_count, 1) for p in patterns]
        return sum(rates) / len(rates)

    async def _adapt_weights(
        self,
        domains: dict[str, DomainEvalSummary],
    ) -> dict[str, dict[str, float]]:
        """Adjust per-domain ranking weights based on eval outcomes.

        The heuristic:
        - If a domain has high eval success AND high latency, increase the
          ``success_rate`` weight slightly and decrease ``cost``.
        - If a domain has low eval success but high confidence in its
          patterns, increase ``confidence`` weight.
        - If eval success is high and patterns are recent, increase
          ``recency`` weight.
        """
        if self._learning_manager is None:
            return {}

        adjustments: dict[str, dict[str, float]] = {}

        for domain_key, acc in domains.items():
            if acc.total_cases < 5:
                continue  # not enough data to adjust

            current_weights = dict(
                self._learning_manager.get_weights_for_domain(domain_key)
            )
            delta: dict[str, float] = {}

            sr = acc.weighted_success_rate
            lat_norm = min(acc.avg_latency_ms / 10000.0, 1.0)

            # High success + high latency → value success over cost.
            if sr > 0.8 and lat_norm > 0.3:
                delta["success_rate"] = round(current_weights.get("success_rate", 0.25) * 0.05, 3)
                delta["cost"] = round(-current_weights.get("cost", 0.10) * 0.03, 3)

            # Low success → value confidence more to improve reflection quality.
            if sr < 0.6:
                delta["confidence"] = round(current_weights.get("confidence", 0.15) * 0.05, 3)
                delta["similarity"] = round(-current_weights.get("similarity", 0.30) * 0.02, 3)

            # High success + high volume → trust recency.
            if sr > 0.75 and acc.total_cases > 20:
                delta["recency"] = round(current_weights.get("recency", 0.20) * 0.03, 3)

            if delta:
                adjustments[domain_key] = delta
                self._learning_manager.adjust_weights(domain_key, delta)

        return adjustments
