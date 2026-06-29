#!/usr/bin/env python3
"""Evaluation harness CLI.

Run all or a subset of eval suites and produce a report.

Examples:
    python -m evals.runner
    python -m evals.runner --suite chat
    python -m evals.runner --suite chat,tool_selection --json report.json
    python -m evals.runner --artifacts ./artifacts
    python -m evals.runner --baseline-save baseline.json          # save gold
    python -m evals.runner --baseline-compare baseline.json       # diff against gold
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from evals.artifacts import print_artifact_summary, save_artifacts
from evals.base import BaseEval
from evals.baseline import load_baseline, save_baseline
from evals.metrics import MetricsTracker
from evals.models import EvalResult
from evals.report import Report


def _discover_suites() -> dict[str, type[BaseEval]]:
    from evals.scenarios.chat import ChatEval
    from evals.scenarios.error_recovery import ErrorRecoveryEval
    from evals.scenarios.long_context import LongContextEval
    from evals.scenarios.memory_consolidation import MemoryConsolidationEval
    from evals.scenarios.memory_retrieval import MemoryRetrievalEval
    from evals.scenarios.mixed_input import MixedInputEval
    from evals.scenarios.multi_turn import MultiTurnEval
    from evals.scenarios.parallel_tool import ParallelToolEval
    from evals.scenarios.planning import PlanningEval
    from evals.scenarios.provider_failover import ProviderFailoverEval
    from evals.scenarios.rag import RAGEval
    from evals.scenarios.reflection import ReflectionEval
    from evals.scenarios.streaming import StreamingEval
    from evals.scenarios.streaming_cancel import StreamingCancelEval
    from evals.scenarios.tool_selection import ToolSelectionEval

    return {
        "chat": ChatEval,
        "tool_selection": ToolSelectionEval,
        "planning": PlanningEval,
        "reflection": ReflectionEval,
        "memory_retrieval": MemoryRetrievalEval,
        "rag": RAGEval,
        "streaming": StreamingEval,
        "error_recovery": ErrorRecoveryEval,
        "multi_turn": MultiTurnEval,
        "long_context": LongContextEval,
        "parallel_tool": ParallelToolEval,
        "provider_failover": ProviderFailoverEval,
        "streaming_cancel": StreamingCancelEval,
        "memory_consolidation": MemoryConsolidationEval,
        "mixed_input": MixedInputEval,
    }


async def _run_suite(cls: type[BaseEval]) -> tuple[str, str, list[EvalResult], MetricsTracker]:
    suite: BaseEval = cls()
    print(f"  Running {suite.name} ... ", end="", flush=True)
    results = await suite.run_all()
    print(f"{suite.metrics.passed}/{suite.metrics.total} passed")
    return suite.name, suite.description, results, suite.metrics


async def main() -> None:
    parser = argparse.ArgumentParser(description="Astra X evaluation harness")
    parser.add_argument("--suite", help="Comma-separated list of suites to run (default: all)")
    parser.add_argument("--json", help="Export report to JSON file")
    parser.add_argument(
        "--artifacts",
        help="Directory to save per-case artifact traces (e.g. ./artifacts)",
        default=None,
    )
    parser.add_argument(
        "--baseline-save",
        help="Path to save a baseline snapshot after running (e.g. baseline.json)",
        default=None,
    )
    parser.add_argument(
        "--baseline-compare",
        help="Path to a previous baseline JSON to diff against",
        default=None,
    )
    args = parser.parse_args()

    suites = _discover_suites()

    if args.suite:
        names = [s.strip() for s in args.suite.split(",")]
        selected = {}
        for name in names:
            if name not in suites:
                print(f"Unknown suite: {name}. Available: {', '.join(suites)}")
                sys.exit(1)
            selected[name] = suites[name]
    else:
        selected = suites

    # Load baseline up front (before running so we fail fast on bad path)
    baseline = None
    if args.baseline_compare:
        try:
            baseline = load_baseline(args.baseline_compare)
        except FileNotFoundError:
            print(f"Baseline file not found: {args.baseline_compare}")
            sys.exit(1)

    report = Report()

    print(f"\n{'=' * 64}")
    print("  Astra X - Evaluation Harness")
    print(f"  Running {len(selected)} suite(s): {', '.join(selected)}")
    print(f"{'=' * 64}\n")

    suite_metrics: dict[str, MetricsTracker] = {}

    for _name, cls in selected.items():
        suite_name, desc, results, metrics = await _run_suite(cls)
        report.add_suite(suite_name, desc, metrics, results)
        suite_metrics[suite_name] = metrics

        if args.artifacts:
            save_artifacts(args.artifacts, suite_name, results, clean=False)
            print_artifact_summary(results)
            print(f"      Artifacts saved to {args.artifacts}/{suite_name}/")

    # Attach baseline comparison if requested
    if baseline is not None:
        report.attach_baseline(baseline)

    print()
    report.print_text()

    # Save baseline after displaying results
    if args.baseline_save:
        ts = save_baseline(args.baseline_save, suite_metrics)
        print(f"\n  Baseline saved to {args.baseline_save}  ({ts})")

    if args.json:
        report.to_json(args.json)
        print(f"\n  Report saved to {args.json}")

    if args.artifacts:
        print(f"\n  Artifacts root: {args.artifacts}/")

    g = report.global_summary
    if g["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
