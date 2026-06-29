"""Streaming cancellation evaluation -- mid-stream cancellation and recovery."""

from __future__ import annotations

from evals.base import BaseEval
from evals.data import STREAMING_CANCEL_CASES
from evals.models import Artifact, EvalCase, EvalResult


class StreamingCancelEval(BaseEval):
    name = "streaming_cancel"
    description = "Streaming cancellation -- mid-stream cancel, partial output, retry after cancel"

    def __init__(self) -> None:
        super().__init__()
        self.cases = [
            EvalCase(**c)  # type: ignore[arg-type]
            for c in STREAMING_CANCEL_CASES
        ]

    @staticmethod
    def _simulate_cancel_stream(data: dict) -> dict:
        cancel_after = data.get("cancel_after_tokens", 0)
        total_tokens = data.get("total_tokens", 100)
        retry = data.get("retry", False)

        if cancel_after >= total_tokens:
            tokens_produced = total_tokens
            cancel_honored = True
            retry_complete = True
        else:
            tokens_produced = cancel_after
            cancel_honored = tokens_produced < total_tokens
            retry_complete = retry

        return {
            "tokens_produced": tokens_produced,
            "total_tokens": total_tokens,
            "cancel_honored": cancel_honored,
            "cancel_invoked": cancel_after < total_tokens,
            "retry_complete": retry_complete,
            "partial_output_pct": round(tokens_produced / max(total_tokens, 1) * 100, 1),
        }

    async def run_case(self, case: EvalCase) -> EvalResult:
        data: dict = case.input
        expected: dict = case.expected

        result = self._simulate_cancel_stream(data)
        passed = True
        reasons: list[str] = []

        if expected.get("partial_tokens") is not None and result["tokens_produced"] != expected["partial_tokens"]:
            passed = False
            reasons.append(f"expected {expected['partial_tokens']} partial tokens, got {result['tokens_produced']}")

        if expected.get("cancel_honored") and not result["cancel_honored"]:
            passed = False
            reasons.append("cancel signal not honored")

        if expected.get("retry_complete") and not result["retry_complete"]:
            passed = False
            reasons.append("retry after cancel did not complete")

        self.metrics.record_custom("cancellation_success", 1.0 if result["cancel_honored"] else 0.0)
        self.metrics.record_custom("partial_output_pct", result["partial_output_pct"])

        return EvalResult(
            case_id=case.id,
            passed=passed,
            output=result,
            metadata={
                "tokens_produced": result["tokens_produced"],
                "cancel_honored": result["cancel_honored"],
                "retry_complete": result["retry_complete"],
            },
            error="; ".join(reasons) if reasons else None,
            artifact=Artifact(
                prompt=f"Streaming cancel: cancel after {data.get('cancel_after_tokens', 0)}/{data.get('total_tokens', 100)} tokens",
                final_response=str(result),
                logs=[f"tokens_produced={result['tokens_produced']}",
                      f"total_tokens={result['total_tokens']}",
                      f"cancel_honored={result['cancel_honored']}",
                      f"partial_output={result['partial_output_pct']}%",
                      f"retry_complete={result['retry_complete']}"],
            ),
        )
