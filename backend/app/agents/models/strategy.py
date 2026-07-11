from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Strategy:
    """Structured guidance produced by the Strategy Engine.

    Combines data from Patterns, Experience Graph, World Model, and
    Provider Metrics Tracker into a single actionable recommendation
    that the Planner can consume directly.
    """

    goal: str
    goal_domain: str

    # Recommended approach
    strategy_summary: str = ""
    recommended_provider: str | None = None
    recommended_model: str | None = None
    recommended_tool_sequence: list[str] = field(default_factory=list)

    # Expected outcomes
    expected_success_rate: float = 0.0
    expected_latency_ms: float = 0.0
    confidence: float = 0.0

    # Fallback
    fallback_provider: str | None = None
    fallback_model: str | None = None

    # Warnings — anti-patterns / pitfalls to avoid
    warnings: list[str] = field(default_factory=list)

    # Metadata
    source_pattern_count: int = 0
    source: str = "strategy_engine"


def format_strategy_for_prompt(strategy: Strategy) -> str:
    """Format a ``Strategy`` into structured text for the planner prompt."""
    parts: list[str] = []

    parts.append(f"Domain: {strategy.goal_domain}")

    if strategy.strategy_summary:
        parts.append(f"Recommended approach: {strategy.strategy_summary}")

    if strategy.recommended_provider:
        parts.append(f"Recommended provider: {strategy.recommended_provider}")
    if strategy.recommended_model:
        parts.append(f"Recommended model: {strategy.recommended_model}")

    if strategy.recommended_tool_sequence:
        seq = " → ".join(strategy.recommended_tool_sequence)
        parts.append(f"Recommended tool sequence: {seq}")

    if strategy.expected_success_rate > 0:
        pct = strategy.expected_success_rate * 100
        parts.append(f"Expected success rate: {pct:.0f}%")

    if strategy.expected_latency_ms > 0:
        parts.append(f"Expected latency: {strategy.expected_latency_ms:.0f}ms")

    if strategy.confidence > 0:
        parts.append(f"Confidence: {strategy.confidence:.2f}")

    if strategy.fallback_provider or strategy.fallback_model:
        fb = []
        if strategy.fallback_provider:
            fb.append(f"provider={strategy.fallback_provider}")
        if strategy.fallback_model:
            fb.append(f"model={strategy.fallback_model}")
        parts.append(f"Fallback: {', '.join(fb)}")

    if strategy.warnings:
        parts.append("Warnings:")
        for w in strategy.warnings:
            parts.append(f"  - {w}")

    if strategy.source_pattern_count:
        parts.append(
            f"Source: {strategy.source_pattern_count} pattern(s) "
            f"via {strategy.source}"
        )

    return "\n".join(parts)
