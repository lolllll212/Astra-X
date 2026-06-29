from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from evals.baseline import (
    BaselineSnapshot,
    MetricDelta,
    SuiteSnapshot,
    _suite_compare_header,
    compare_suite,
    print_baseline_suite,
    print_global_comparison,
)
from evals.metrics import MetricsTracker
from evals.models import EvalResult


@dataclass
class ReportEntry:
    name: str
    description: str
    summary: dict[str, Any]
    custom_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    results: list[EvalResult] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)


_CUSTOM_LABELS: dict[str, str] = {
    "qa_accuracy": "QA accuracy",
    "answer_coverage": "Answer coverage",
    "response_length": "Response length",
    "tool_selection_accuracy": "Tool selection accuracy",
    "execution_success_rate": "Tool execution success",
    "plan_task_count": "Plan task count",
    "plan_accuracy": "Planning accuracy",
    "dependency_rate": "Dependency rate",
    "reflection_accuracy": "Reflection accuracy",
    "decision_accept": "Accept count",
    "decision_retry": "Retry count",
    "decision_replan": "Replan count",
    "decision_abort": "Abort count",
    "retrieval_precision": "Memory retrieval precision",
    "groundedness": "Groundedness",
    "hallucination_rate": "Hallucination rate",
    "event_completeness": "Event completeness",
    "recovery_success": "Recovery success",
    "recovery_error_present": "Error reported",
}

_CUSTOM_FORMATS: dict[str, str] = {
    "response_length": "int",
    "plan_task_count": "int",
    "decision_accept": "sum",
    "decision_retry": "sum",
    "decision_replan": "sum",
    "decision_abort": "sum",
}


def _fmt_custom(name: str, agg: dict[str, float]) -> str:
    """Format a custom metric value for display."""
    fmt = _CUSTOM_FORMATS.get(name, "avg")
    label = _CUSTOM_LABELS.get(name, name.replace("_", " ").title())
    if fmt == "int":
        return f"{label}: {agg['avg']:.0f}"
    if fmt == "sum":
        return f"{label}: {agg['sum']:.0f}"
    if name.endswith("_rate") or name.endswith("_accuracy") or name in (
        "groundedness", "retrieval_precision", "hallucination_rate",
        "event_completeness", "answer_coverage", "recovery_success",
    ):
        return f"{label}: {agg['avg']:.1%}"
    return f"{label}: {agg['avg']:.3f}"


@dataclass
class Report:
    """Aggregated report for one or more eval suites."""

    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    suites: list[ReportEntry] = field(default_factory=list)
    global_summary: dict[str, Any] = field(default_factory=dict)

    # Baseline comparison (optional)
    baseline: BaselineSnapshot | None = None
    _suite_deltas: dict[str, list[MetricDelta]] = field(default_factory=dict)
    _global_deltas: list[MetricDelta] = field(default_factory=list)

    def attach_baseline(self, baseline: BaselineSnapshot) -> None:
        """Attach a baseline snapshot and compute per-suite deltas."""
        self.baseline = baseline
        seen_global: set[str] = set()
        global_deltas: list[MetricDelta] = []

        for entry in self.suites:
            base = baseline.suites.get(entry.name)
            current = SuiteSnapshot(
                total=entry.summary.get("total", 0),
                passed=entry.summary.get("passed", 0),
                failed=entry.summary.get("failed", 0),
                success_rate=entry.summary.get("success_rate", 0.0),
                avg_latency_ms=entry.summary.get("avg_latency_ms", 0.0),
                total_tokens=entry.summary.get("total_tokens", 0),
                custom=entry.custom_metrics,
            )
            suite_deltas = compare_suite(entry.name, current, base)
            # Per-suite: only custom diagnostic metrics (not core Pass rate etc.)
            self._suite_deltas[entry.name] = [
                d for d in suite_deltas if d.name not in ("Pass rate", "Avg latency", "Tokens used")
            ]
            # Global: first occurrence of each core metric
            for d in suite_deltas:
                if d.name in ("Pass rate", "Avg latency", "Tokens used") and d.name not in seen_global:
                    global_deltas.append(d)
                    seen_global.add(d.name)

        self._global_deltas = global_deltas

    def add_suite(
        self,
        name: str,
        description: str,
        metrics: MetricsTracker,
        results: list[EvalResult],
    ) -> None:
        self.suites.append(
            ReportEntry(
                name=name,
                description=description,
                summary=metrics.summary(),
                custom_metrics=metrics.all_custom_summaries(),
                results=results,
                errors=metrics.errors(),
            )
        )
        self._recompute_global()

    def _recompute_global(self) -> None:
        total = sum(e.summary["total"] for e in self.suites)
        passed = sum(e.summary["passed"] for e in self.suites)
        all_latencies: list[float] = []
        all_tokens = 0
        for entry in self.suites:
            for r in entry.results:
                all_latencies.append(r.latency_ms)
                all_tokens += r.tokens_used

        self.global_summary = {
            "suites": len(self.suites),
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": round(passed / total, 4) if total else 0.0,
            "avg_latency_ms": round(sum(all_latencies) / len(all_latencies), 1) if all_latencies else 0.0,
            "total_tokens": all_tokens,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "global_summary": self.global_summary,
            "suites": [
                {
                    "name": e.name,
                    "description": e.description,
                    "summary": e.summary,
                    "custom_metrics": e.custom_metrics,
                    "errors": e.errors,
                }
                for e in self.suites
            ],
        }

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def print_text(self) -> None:
        sep = "=" * 64
        print(sep)
        print(f"  EVALUATION REPORT  ({self.generated_at})")
        print(sep)

        g = self.global_summary
        print(
            f"  Suites: {g['suites']}  |  "
            f"Passed: {g['passed']}/{g['total']}  "
            f"({g['success_rate']:.1%})  |  "
            f"Avg latency: {g['avg_latency_ms']:.0f} ms  |  "
            f"Tokens: {g['total_tokens']}"
        )

        # ── Global baseline comparison ────────────────────────────
        if self.baseline is not None and self._global_deltas:
            print_global_comparison(self._global_deltas, self.baseline.generated_at)

        print(sep)

        for entry in self.suites:
            s = entry.summary
            status = "PASS" if s["failed"] == 0 else "FAIL"
            print(f"\n  [{status}] {entry.name}")
            print(f"      {entry.description}")
            print(
                f"      {s['passed']}/{s['total']} passed  "
                f"|  avg {s['avg_latency_ms']:.0f} ms  "
                f"|  p95 {s['p95_latency_ms']:.0f} ms  "
                f"|  tokens {s['total_tokens']}"
            )

            # ── Per-suite baseline comparison ────────────────────
            if self.baseline is not None:
                suite_deltas = self._suite_deltas.get(entry.name, [])
                if suite_deltas:
                    header = _suite_compare_header(self.baseline.generated_at)
                    print_baseline_suite(header, suite_deltas)

            # Custom metrics
            if entry.custom_metrics:
                lines = []
                for name, agg in sorted(entry.custom_metrics.items()):
                    lines.append(_fmt_custom(name, agg))
                print("      Diagnostics:  " + "  |  ".join(lines))

            if entry.errors:
                for case_id, err in entry.errors:
                    print(f"      ! {case_id}: {err[:120]}")
