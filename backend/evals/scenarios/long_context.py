"""Long-context conversation evaluation -- context window management."""

from __future__ import annotations

from evals.base import BaseEval
from evals.data import LONG_CONTEXT_CASES
from evals.models import Artifact, EvalCase, EvalResult


class LongContextEval(BaseEval):
    name = "long_context"
    description = "Long-context conversation -- truncation, window management, token handling"

    def __init__(self) -> None:
        super().__init__()
        self.cases = [
            EvalCase(**c)  # type: ignore[arg-type]
            for c in LONG_CONTEXT_CASES
        ]

    @staticmethod
    def _simulate_context_handler(token_count: int) -> dict:
        MAX_TOKENS = 4096
        if token_count > MAX_TOKENS:
            truncated_pct = round(MAX_TOKENS / token_count * 100, 1)
            actual_tokens = MAX_TOKENS
            truncated = True
        else:
            truncated_pct = 100.0
            actual_tokens = token_count
            truncated = False
        return {
            "truncated": truncated,
            "original_tokens": token_count,
            "actual_tokens": actual_tokens,
            "context_utilization_pct": truncated_pct,
        }

    async def run_case(self, case: EvalCase) -> EvalResult:
        data: dict = case.input
        expected: dict = case.expected
        token_estimate = data.get("token_count_estimate", 100)

        result = self._simulate_context_handler(token_estimate)
        passed = True
        reasons: list[str] = []

        if expected.get("should_truncate") and not result["truncated"]:
            passed = False
            reasons.append("expected truncation but context was not truncated")

        if not expected.get("should_truncate") and result["truncated"]:
            passed = False
            reasons.append("did not expect truncation but context was truncated")

        max_pct = expected.get("max_context_pct")
        if max_pct is not None and result["context_utilization_pct"] > max_pct:
            passed = False
            reasons.append(f"context utilization {result['context_utilization_pct']}% exceeds max {max_pct}%")

        self.metrics.record_custom("completion_rate", 1.0 if passed else 0.0)
        self.metrics.record_custom("context_window_usage", float(result["context_utilization_pct"]))

        return EvalResult(
            case_id=case.id,
            passed=passed,
            output=result,
            metadata={
                "original_tokens": result["original_tokens"],
                "actual_tokens": result["actual_tokens"],
                "truncated": result["truncated"],
            },
            error="; ".join(reasons) if reasons else None,
            artifact=Artifact(
                prompt=f"Long context ({token_estimate} tokens)",
                final_response=str(result),
                logs=[f"original_tokens={result['original_tokens']}",
                      f"actual_tokens={result['actual_tokens']}",
                      f"truncated={result['truncated']}",
                      f"context_usage={result['context_utilization_pct']}%"],
            ),
        )
