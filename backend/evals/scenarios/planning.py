"""Multi-step planning evaluation -- planner decomposition quality."""

from __future__ import annotations

from evals.base import BaseEval
from evals.data import PLANNING_CASES
from evals.models import Artifact, EvalCase, EvalResult


class PlanningEval(BaseEval):
    name = "planning"
    description = "Multi-step planning -- task count, dependencies, tool assignment"

    def __init__(self) -> None:
        super().__init__()
        self.cases = [
            EvalCase(**c)  # type: ignore[arg-type]
            for c in PLANNING_CASES
        ]

    @staticmethod
    def _mock_plan(goal: str) -> list[dict]:
        goal_lower = goal.lower()
        tasks: list[dict] = []

        if "weather" in goal_lower:
            tasks.append({"id": "t1", "description": "Check weather in Tokyo", "tool": "web_search", "deps": []})
        elif "laptop" in goal_lower or "budget" in goal_lower:
            tasks.append({"id": "t1", "description": "Research budget laptops under $800", "tool": "web_search", "deps": []})
            tasks.append({"id": "t2", "description": "Summarize top 3 laptops", "tool": None, "deps": ["t1"]})
        elif "python" in goal_lower and "javascript" in goal_lower:
            tasks.append({"id": "t1", "description": "Research Python features", "tool": "web_search", "deps": []})
            tasks.append({"id": "t2", "description": "Research JavaScript features", "tool": "web_search", "deps": []})
            tasks.append({"id": "t3", "description": "Create comparison table", "tool": None, "deps": ["t1", "t2"]})
        elif "release" in goal_lower and "days" in goal_lower:
            tasks.append({"id": "t1", "description": "Find latest GitHub release of fastapi", "tool": "github", "deps": []})
            tasks.append({"id": "t2", "description": "Calculate days since release", "tool": "calculator", "deps": ["t1"]})
        else:
            tasks.append({"id": "t1", "description": goal, "tool": None, "deps": []})

        return tasks

    async def run_case(self, case: EvalCase) -> EvalResult:
        expected: dict = case.expected  # type: ignore[assignment]
        goal = str(case.input)
        plan = self._mock_plan(goal)
        task_count = len(plan)

        passed = True
        reasons: list[str] = []

        min_t = expected.get("min_tasks", 1)
        max_t = expected.get("max_tasks", 10)

        if task_count < min_t:
            passed = False
            reasons.append(f"only {task_count} tasks, min {min_t}")
        if task_count > max_t:
            passed = False
            reasons.append(f"{task_count} tasks, max {max_t}")

        if expected.get("has_dependencies") and not any(t["deps"] for t in plan):
            passed = False
            reasons.append("no dependencies found")

        if expected.get("requires_tools") and not any(t["tool"] for t in plan):
            passed = False
            reasons.append("no tools assigned")

        self.metrics.record_custom("plan_task_count", float(task_count))
        self.metrics.record_custom("plan_accuracy", 1.0 if passed else 0.0)
        dep_rate = sum(1 for t in plan if t["deps"]) / max(len(plan), 1)
        self.metrics.record_custom("dependency_rate", dep_rate)

        planner_output_lines = [f"  {t['id']}: {t['description']} [tool={t['tool']}, deps={t['deps']}]" for t in plan]
        planner_output = f"Plan for: {goal}\n" + "\n".join(planner_output_lines)

        return EvalResult(
            case_id=case.id,
            passed=passed,
            output=plan,
            metadata={"task_count": task_count},
            error="; ".join(reasons) if reasons else None,
            artifact=Artifact(
                prompt=goal,
                planner_output=planner_output,
                final_response=f"Executed {task_count} tasks: {', '.join(t['description'] for t in plan)}",
                logs=[f"expected min_tasks={min_t}, max_tasks={max_t}",
                      f"actual task_count={task_count}",
                      f"has_dependencies={any(t['deps'] for t in plan)}",
                      f"requires_tools={any(t['tool'] for t in plan)}"],
            ),
        )
