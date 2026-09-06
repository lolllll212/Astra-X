"""Parallel tool execution evaluation -- dependency resolution and concurrency."""

from __future__ import annotations

from evals.base import BaseEval
from evals.data import PARALLEL_TOOL_CASES
from evals.models import Artifact, EvalCase, EvalResult


class ParallelToolEval(BaseEval):
    name = "parallel_tool"
    description = "Parallel tool execution -- dependency resolution, concurrency, DAG scheduling"

    def __init__(self) -> None:
        super().__init__()
        self.cases = [
            EvalCase(**c)  # type: ignore[arg-type]
            for c in PARALLEL_TOOL_CASES
        ]

    @staticmethod
    def _simulate_scheduler(tasks: list[dict]) -> dict:
        if not tasks:
            return {"sequential_steps": 0, "max_parallel": 0, "total_tasks": 0, "executed": []}

        dependencies = {i: set(t.get("depends_on", [])) for i, t in enumerate(tasks)}
        remaining = set(range(len(tasks)))
        executed: list[int] = []
        steps = 0
        max_parallel = 0

        while remaining:
            ready = {i for i in remaining if dependencies[i].issubset(set(executed))}
            if not ready:
                break
            steps += 1
            max_parallel = max(max_parallel, len(ready))
            executed.extend(sorted(ready))
            remaining -= ready

        return {
            "sequential_steps": steps,
            "max_parallel": max_parallel,
            "total_tasks": len(tasks),
            "executed": executed,
        }

    async def run_case(self, case: EvalCase) -> EvalResult:
        data: dict = case.input
        expected: dict = case.expected
        tasks: list[dict] = data.get("tasks", [])

        result = self._simulate_scheduler(tasks)
        passed = True
        reasons: list[str] = []

        if result["max_parallel"] < expected.get("min_parallel", 0):
            passed = False
            reasons.append(f"max_parallel={result['max_parallel']} < min {expected['min_parallel']}")

        if expected.get("max_sequential_steps", 999) is not None and result["sequential_steps"] > expected["max_sequential_steps"]:
                passed = False
                reasons.append(f"sequential_steps={result['sequential_steps']} > max {expected['max_sequential_steps']}")

        if len(result["executed"]) < result["total_tasks"]:
            passed = False
            reasons.append(f"only {len(result['executed'])}/{result['total_tasks']} tasks executed")

        total = max(result["total_tasks"], 1)
        execution_rate = len(result["executed"]) / total
        self.metrics.record_custom("execution_success_rate", execution_rate)
        self.metrics.record_custom("max_parallel", float(result["max_parallel"]))

        return EvalResult(
            case_id=case.id,
            passed=passed,
            output=result,
            metadata={
                "tasks": result["total_tasks"],
                "parallel": result["max_parallel"],
                "steps": result["sequential_steps"],
            },
            error="; ".join(reasons) if reasons else None,
            artifact=Artifact(
                prompt=f"Parallel execution: {result['total_tasks']} tasks",
                tool_calls=tasks,
                final_response=str(result),
                logs=[f"tasks={result['total_tasks']}",
                      f"max_parallel={result['max_parallel']}",
                      f"sequential_steps={result['sequential_steps']}"],
            ),
        )
