"""Execution policies — user-selectable modes that control which agent
capabilities are enabled.

Each policy sets boolean flags for: planning, reflection, learning,
and simulation. This lets users choose a mode that fits their latency
budget and quality requirements.

Usage::

    from app.agents.models.policy import ExecutionPolicy, ExecutionMode

    policy = ExecutionMode.FAST.policy()
    # Results in: planning=False, reflection=False, learning=False, simulation=False
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


@dataclass(frozen=True)
class ExecutionPolicy:
    """Controls which agent capabilities are enabled.

    Attributes:
        planning: Decompose the goal into a structured task plan.
        reflection: Evaluate each task's output for quality.
        learning: Extract patterns and record anti-patterns from executions.
        simulation: Estimate plan cost/success before execution.
        label: Human-readable name for logging and display.
        max_iterations: Override for the agent loop iteration limit
            (``None`` = use system default).
        extra: Additional mode-specific configuration (e.g. temperature,
            provider preference hints).
    """

    planning: bool = True
    reflection: bool = True
    learning: bool = True
    simulation: bool = True
    label: str = "balanced"
    max_iterations: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ExecutionMode(StrEnum):
    """Predefined execution modes.

    Each mode represents a different trade-off between speed,
    quality, and capability depth.
    """

    FAST = "fast"
    BALANCED = "balanced"
    AUTONOMOUS = "autonomous"
    RESEARCH = "research"
    CODING = "coding"

    def policy(self) -> ExecutionPolicy:
        """Return the :class:`ExecutionPolicy` for this mode."""
        return _POLICY_REGISTRY[self]


# ---------------------------------------------------------------------------
# Policy definitions
# ---------------------------------------------------------------------------

_FAST = ExecutionPolicy(
    planning=False,
    reflection=False,
    learning=False,
    simulation=False,
    label="fast",
    max_iterations=1,
    extra={"prefers_speed": True},
)

_BALANCED = ExecutionPolicy(
    planning=True,
    reflection=True,
    learning=True,
    simulation=True,
    label="balanced",
)

_AUTONOMOUS = ExecutionPolicy(
    planning=True,
    reflection=True,
    learning=True,
    simulation=True,
    label="autonomous",
    max_iterations=10,
    extra={"prefers_reasoning": True},
)

_RESEARCH = ExecutionPolicy(
    planning=True,
    reflection=True,
    learning=True,
    simulation=True,
    label="research",
    max_iterations=5,
    extra={"reasoning_depth": "deep", "prefers_large_context": True},
)

_CODING = ExecutionPolicy(
    planning=True,
    reflection=True,
    learning=True,
    simulation=True,
    label="coding",
    max_iterations=5,
    extra={"requires_coding": True},
)

_POLICY_REGISTRY: dict[ExecutionMode, ExecutionPolicy] = {
    ExecutionMode.FAST: _FAST,
    ExecutionMode.BALANCED: _BALANCED,
    ExecutionMode.AUTONOMOUS: _AUTONOMOUS,
    ExecutionMode.RESEARCH: _RESEARCH,
    ExecutionMode.CODING: _CODING,
}
