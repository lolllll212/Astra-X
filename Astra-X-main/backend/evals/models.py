from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalCase:
    """A single evaluation case."""

    id: str
    description: str
    input: Any
    expected: Any
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Artifact:
    """Captured trace of an evaluation case for reproducibility.

    Stores the full input/output trace including prompt, planner
    output, reflection output, tool calls, final response, timing
    breakdown, and execution logs.
    """

    prompt: str
    planner_output: str | None = None
    reflection_output: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    final_response: str | None = None
    timing: dict[str, float] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """The outcome of running a single eval case."""

    case_id: str
    passed: bool
    latency_ms: float = 0.0
    tokens_used: int = 0
    output: Any = None
    expected: Any = None
    error: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    artifact: Artifact | None = None
