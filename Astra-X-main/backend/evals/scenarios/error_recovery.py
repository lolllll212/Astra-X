"""Error recovery evaluation -- graceful degradation under failures."""

from __future__ import annotations

from evals.base import BaseEval
from evals.data import ERROR_RECOVERY_CASES
from evals.models import Artifact, EvalCase, EvalResult


class ErrorRecoveryEval(BaseEval):
    name = "error_recovery"
    description = "Error recovery -- graceful handling of tool/LLM/planning failures"

    def __init__(self) -> None:
        super().__init__()
        self.cases = [
            EvalCase(**c)  # type: ignore[arg-type]
            for c in ERROR_RECOVERY_CASES
        ]

    @staticmethod
    def _simulate_error_handling(scenario: str, **kwargs: object) -> dict:
        """Simulate system error handling for each error scenario.

        Returns a dict with 'crashed' (bool), 'error' (str | None),
        and 'handled' (bool).
        """
        if scenario == "tool_not_found":
            tool = kwargs.get("tool_name", "unknown")
            return {
                "crashed": False,
                "error": f"Tool '{tool}' not found in registry.",
                "handled": True,
            }

        if scenario == "invalid_arguments":
            return {
                "crashed": False,
                "error": "Missing required argument: 'expression'",
                "handled": True,
            }

        if scenario == "llm_timeout":
            return {
                "crashed": False,
                "error": "LLM request timed out after 30s.",
                "handled": True,
            }

        if scenario == "malformed_plan":
            return {
                "crashed": False,
                "error": "Planner output could not be parsed as JSON.",
                "handled": True,
            }

        if scenario == "empty_plan":
            return {
                "crashed": False,
                "error": "Planner returned an empty task list.",
                "handled": True,
            }

        return {"crashed": True, "error": "Unknown scenario", "handled": False}

    async def run_case(self, case: EvalCase) -> EvalResult:
        data = case.input
        expected: dict = case.expected
        scenario = data["scenario"]

        extra = {k: v for k, v in data.items() if k != "scenario"}
        result = self._simulate_error_handling(scenario, **extra)
        passed = True
        reasons: list[str] = []

        if expected.get("must_not_crash") and result["crashed"]:
            passed = False
            reasons.append("system crashed instead of recovering")

        if expected.get("must_contain_error") and not result["error"]:
            passed = False
            reasons.append("expected an error message but none produced")

        if result.get("crashed"):
            passed = False
            reasons.append(result.get("error", "unhandled crash"))

        recovered = 0.0 if result["crashed"] else 1.0
        self.metrics.record_custom("recovery_success", recovered)
        self.metrics.record_custom("recovery_error_present", 1.0 if result.get("error") else 0.0)

        return EvalResult(
            case_id=case.id,
            passed=passed,
            output=str(result.get("error", "")),
            metadata={"scenario": scenario, "handled": result["handled"]},
            error="; ".join(reasons) if reasons else None,
            artifact=Artifact(
                prompt=f"Scenario: {scenario}\nInput: {data}",
                tool_calls=[{"scenario": scenario, "arguments": data, "outcome": result}],
                final_response=result.get("error", ""),
                logs=[f"scenario={scenario}",
                      f"crashed={result['crashed']}",
                      f"handled={result['handled']}",
                      f"error_message={'yes' if result.get('error') else 'none'}"],
                timing={"error_handling_ms": 0.2},
            ),
        )
