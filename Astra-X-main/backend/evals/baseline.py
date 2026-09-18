"""Baseline snapshots -- persist "known-good" results and diff against runs.

Usage:
    # After a clean run, save the baseline:
    python -m evals.runner --baseline-save evals_baseline.json

    # On a later run, compare:
    python -m evals.runner --baseline-compare evals_baseline.json

The comparison shows per-metric deltas with direction arrows so you
can see at a glance whether things improved or regressed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.metrics import MetricsTracker

# ── Data models ─────────────────────────────────────────────────────


@dataclass
class SuiteSnapshot:
    """Immutable snapshot of a single suite's aggregate results."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    total_tokens: int = 0
    custom: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class BaselineSnapshot:
    """Full baseline from a previous evaluation run."""

    generated_at: str = ""
    suites: dict[str, SuiteSnapshot] = field(default_factory=dict)


# ── Save / load ─────────────────────────────────────────────────────


def _serialize(snapshot: BaselineSnapshot) -> dict[str, Any]:
    return {
        "generated_at": snapshot.generated_at,
        "suites": {
            name: {
                "total": s.total,
                "passed": s.passed,
                "failed": s.failed,
                "success_rate": s.success_rate,
                "avg_latency_ms": s.avg_latency_ms,
                "total_tokens": s.total_tokens,
                "custom": s.custom,
            }
            for name, s in snapshot.suites.items()
        },
    }


def _deserialize(data: dict[str, Any]) -> BaselineSnapshot:
    suites = {}
    for name, s in data.get("suites", {}).items():
        suites[name] = SuiteSnapshot(
            total=s.get("total", 0),
            passed=s.get("passed", 0),
            failed=s.get("failed", 0),
            success_rate=s.get("success_rate", 0.0),
            avg_latency_ms=s.get("avg_latency_ms", 0.0),
            total_tokens=s.get("total_tokens", 0),
            custom=s.get("custom", {}),
        )
    return BaselineSnapshot(
        generated_at=data.get("generated_at", ""),
        suites=suites,
    )


def _build_snapshot(metrics: MetricsTracker) -> SuiteSnapshot:
    s = metrics.summary()
    return SuiteSnapshot(
        total=s["total"],
        passed=s["passed"],
        failed=s["failed"],
        success_rate=s["success_rate"],
        avg_latency_ms=s["avg_latency_ms"],
        total_tokens=s["total_tokens"],
        custom=metrics.all_custom_summaries(),
    )


def save_baseline(
    path: str | Path,
    suite_metrics: dict[str, MetricsTracker],
) -> str:
    """Build a baseline from the given suite metrics and write to *path*.

    Returns the generated-at timestamp.
    """
    generated_at = datetime.now(UTC).isoformat()
    snapshot = BaselineSnapshot(generated_at=generated_at)
    for name, metrics in suite_metrics.items():
        snapshot.suites[name] = _build_snapshot(metrics)

    Path(path).write_text(
        json.dumps(_serialize(snapshot), indent=2, default=str),
        encoding="utf-8",
    )
    return generated_at


def load_baseline(path: str | Path) -> BaselineSnapshot:
    """Load a baseline from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return _deserialize(data)


# ── Comparison ──────────────────────────────────────────────────────


@dataclass
class MetricDelta:
    """Difference for a single metric between baseline and current."""

    name: str
    baseline: float
    current: float
    delta: float
    pct_change: float | None
    direction: str  # "up", "down", "unchanged"
    fmt: str  # "pct", "int", "ms", "sum"


def _direction(delta: float, *, higher_is_better: bool = True) -> str:
    if abs(delta) < 1e-9:
        return "unchanged"
    if higher_is_better:
        return "up" if delta > 0 else "down"
    return "down" if delta > 0 else "up"


def _compute_deltas_for(
    label: str,
    base_val: float,
    cur_val: float,
    fmt: str,
    higher_is_better: bool = True,
) -> MetricDelta:
    delta = cur_val - base_val
    pct = None
    if abs(base_val) > 1e-9:
        pct = (delta / base_val) * 100
    return MetricDelta(
        name=label,
        baseline=base_val,
        current=cur_val,
        delta=delta,
        pct_change=pct,
        direction=_direction(delta, higher_is_better=higher_is_better),
        fmt=fmt,
    )


def compare_suite(
    label: str,
    current: SuiteSnapshot,
    baseline: SuiteSnapshot | None,
) -> list[MetricDelta]:
    """Compare current vs baseline for a single suite.

    Returns a list of :class:`MetricDelta` for every tracked metric.
    If *baseline* is ``None``, every delta is zero (first run).
    """
    deltas: list[MetricDelta] = []

    if baseline is None:
        baseline = SuiteSnapshot()

    # Core metrics
    deltas.append(_compute_deltas_for(
        "Pass rate",
        baseline.success_rate,
        current.success_rate,
        fmt="pct",
        higher_is_better=True,
    ))
    deltas.append(_compute_deltas_for(
        "Avg latency",
        baseline.avg_latency_ms,
        current.avg_latency_ms,
        fmt="ms",
        higher_is_better=False,
    ))
    deltas.append(_compute_deltas_for(
        "Tokens used",
        float(baseline.total_tokens),
        float(current.total_tokens),
        fmt="int",
        higher_is_better=False,
    ))

    # Custom per-suite metrics
    all_metric_names = set(current.custom) | set(baseline.custom)
    for name in sorted(all_metric_names):
        base_agg = baseline.custom.get(name, {})
        cur_agg = current.custom.get(name, {})
        base_val = base_agg.get("avg", 0.0)
        cur_val = cur_agg.get("avg", 0.0)
        higher_is_better = True
        if name.endswith("_count") or name.startswith("decision_"):
            fmt = "sum"
            base_val = base_agg.get("sum", 0.0)
            cur_val = cur_agg.get("sum", 0.0)
            if name == "decision_abort":
                higher_is_better = False
        elif name in ("response_length", "plan_task_count"):
            fmt = "int"
        elif name in ("hallucination_rate",):
            fmt = "pct"
            higher_is_better = False
        else:
            fmt = "pct"

        deltas.append(_compute_deltas_for(
            name,
            base_val,
            cur_val,
            fmt=fmt,
            higher_is_better=higher_is_better,
        ))

    return deltas


# ── Display ─────────────────────────────────────────────────────────


_ARROWS = {"up": "(+)", "down": "(-)", "unchanged": "(=)"}


def _fmt_val(value: float, fmt: str) -> str:
    if fmt == "pct":
        return f"{value:.1%}"
    if fmt == "ms":
        return f"{value:.0f} ms"
    if fmt == "int":
        return f"{value:.0f}"
    if fmt == "sum":
        return f"{value:.0f}"
    return f"{value:.3f}"


def _fmt_delta(d: MetricDelta) -> str:
    arrow = _ARROWS.get(d.direction, " ")
    value_str = _fmt_val(abs(d.delta), d.fmt)

    # For percentages, show both absolute and relative
    if d.fmt == "pct" and d.pct_change is not None and abs(d.pct_change) > 0.1:
        return f"{value_str}  ({d.pct_change:+.1f}%)  {arrow}"
    if d.pct_change is not None and abs(d.pct_change) > 0.1:
        return f"{value_str}  ({d.pct_change:+.1f}%)  {arrow}"
    if abs(d.delta) > 1e-9:
        return f"{value_str}  {arrow}"
    return "--  unchanged"


def _suite_compare_header(baseline_ts: str) -> str:
    return f"  vs baseline ({baseline_ts})"


def print_baseline_suite(header: str, deltas: list[MetricDelta]) -> None:
    """Print per-suite comparison block for custom diagnostic metrics."""
    print(f"      {header}")

    if not deltas:
        print("        No significant changes from baseline.")
        return

    for d in deltas:
        delta_str = _fmt_delta(d)
        label = d.name.replace("_", " ").title()
        print(f"        {label:30s}  {_fmt_val(d.current, d.fmt):>12s}  {delta_str}")


def print_global_comparison(
    global_deltas: list[MetricDelta],
    baseline_ts: str,
) -> None:
    """Print global summary comparison block."""
    print(f"\n  vs baseline ({baseline_ts})")
    print(f"  {'-' * 58}")
    for d in global_deltas:
        delta_str = _fmt_delta(d)
        label = d.name.replace("_", " ").title()
        print(f"  {label:30s}  {_fmt_val(d.current, d.fmt):>12s}  {delta_str}")
