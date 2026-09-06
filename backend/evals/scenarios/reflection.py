"""Reflection decisions evaluation -- quality assessment accuracy."""

from __future__ import annotations

from evals.base import BaseEval
from evals.data import REFLECTION_CASES
from evals.models import Artifact, EvalCase, EvalResult


class ReflectionEval(BaseEval):
    name = "reflection"
    description = "Reflection decisions -- accept/retry/abort accuracy"

    def __init__(self) -> None:
        super().__init__()
        self.cases = [
            EvalCase(
                id=c["id"],
                description=c["description"],
                input=c["input"],
                expected=c["expected_decision"],
                tags=c["tags"],
            )
            for c in REFLECTION_CASES
        ]

    @staticmethod
    def _mock_reflect(data: dict) -> str:
        status = data.get("status", "completed")
        output = data.get("output") or ""
        error = data.get("error")

        if status == "failed" or error:
            return "retry"

        if not output.strip():
            return "retry"

        cleaned = output.strip()
        words = cleaned.split()
        if len(words) <= 2:
            return "retry"

        real_word_ratio = sum(1 for w in words if len(w) > 2 and w.isalpha()) / max(len(words), 1)
        if real_word_ratio < 0.4:
            return "retry"

        return "accept"

    async def run_case(self, case: EvalCase) -> EvalResult:
        data = case.input
        expected_decision = case.expected
        decision = self._mock_reflect(data)
        passed = decision == expected_decision

        self.metrics.record_custom("reflection_accuracy", 1.0 if passed else 0.0)
        self.metrics.record_custom("decision_accept", 1.0 if decision == "accept" else 0.0)
        self.metrics.record_custom("decision_retry", 1.0 if decision == "retry" else 0.0)
        self.metrics.record_custom("decision_replan", 1.0 if decision == "replan" else 0.0)
        self.metrics.record_custom("decision_abort", 1.0 if decision == "abort" else 0.0)

        return EvalResult(
            case_id=case.id,
            passed=passed,
            output=decision,
            error=None if passed else f"expected={expected_decision!r}, got={decision!r}",
            artifact=Artifact(
                prompt=f"Reflect on: status={data.get('status')}, output={str(data.get('output',''))[:100]}",
                reflection_output=f"Decision: {decision}\nExpected: {expected_decision}\nPassed: {passed}",
                logs=[f"status={data.get('status')}",
                      f"output_length={len(str(data.get('output','')))}",
                      f"mock_heuristic_result={decision}",
                      f"expected_decision={expected_decision}"],
                timing={"reflect_ms": 0.5},
            ),
        )
