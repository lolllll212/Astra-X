"""Artifact persistence -- saves per-case traces to disk for debugging."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.models import Artifact, EvalResult


def _artifact_to_dict(artifact: Artifact) -> dict[str, Any]:
    return {
        "prompt": artifact.prompt,
        "planner_output": artifact.planner_output,
        "reflection_output": artifact.reflection_output,
        "tool_calls": artifact.tool_calls,
        "final_response": artifact.final_response,
        "timing": artifact.timing,
        "logs": artifact.logs,
    }


def _result_to_dict(
    result: EvalResult,
    suite_name: str,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "case_id": result.case_id,
        "suite": suite_name,
        "passed": result.passed,
        "latency_ms": result.latency_ms,
        "tokens_used": result.tokens_used,
        "output": result.output,
        "expected": result.expected,
        "error": result.error,
        "tags": result.tags,
    }
    if result.artifact is not None:
        d["artifact"] = _artifact_to_dict(result.artifact)
    return d


def save_artifacts(
    output_dir: str | Path,
    suite_name: str,
    results: list[EvalResult],
    *,
    clean: bool = False,
) -> int:
    """Save all artifacts from a suite run to *output_dir*.

    Returns the number of artifact files written.
    """
    dest = Path(output_dir)
    if clean and dest.exists():
        shutil.rmtree(dest)

    suite_dir = dest / suite_name.replace(" ", "_")
    suite_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for result in results:
        if result.artifact is None:
            continue

        data = _result_to_dict(result, suite_name)
        case_file = suite_dir / f"{result.case_id}.json"
        case_file.write_text(
            json.dumps(data, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        count += 1

    # Write index
    index: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "suite": suite_name,
        "cases": [
            {
                "case_id": r.case_id,
                "passed": r.passed,
                "latency_ms": r.latency_ms,
                "has_artifact": r.artifact is not None,
            }
            for r in results
        ],
    }
    index_path = suite_dir / "_index.json"
    index_path.write_text(json.dumps(index, indent=2, default=str), encoding="utf-8")

    return count


def print_artifact_summary(results: list[EvalResult]) -> None:
    """Print a one-line summary of artifact availability."""
    with_artifact = sum(1 for r in results if r.artifact is not None)
    print(f"      Artifacts: {with_artifact}/{len(results)} cases captured")
