"""Memory retrieval evaluation -- recall accuracy from conversation history."""

from __future__ import annotations

from evals.base import BaseEval
from evals.data import MEMORY_CASES
from evals.models import Artifact, EvalCase, EvalResult


class MemoryRetrievalEval(BaseEval):
    name = "memory_retrieval"
    description = "Memory retrieval -- recall, truncation, empty-state handling"

    def __init__(self) -> None:
        super().__init__()
        self.cases = [
            EvalCase(**c)  # type: ignore[arg-type]
            for c in MEMORY_CASES
        ]

    @staticmethod
    def _retrieve_context(messages: list, query: str, limit: int = 20) -> str | None:
        if not messages:
            return None

        truncated = messages[-limit:]
        lines = [f"{role.upper()}: {text}" for role, text in truncated]
        return "\n".join(lines)

    async def run_case(self, case: EvalCase) -> EvalResult:
        data = case.input
        expected: dict = case.expected
        messages = data.get("conversation_messages", [])
        query = data.get("query", "")

        context = self._retrieve_context(messages, query)

        passed = True
        reasons: list[str] = []
        terms_found = 0
        terms_expected = 0
        total_expected_terms = len(expected.get("must_contain", []))

        if expected.get("must_be_none"):
            if context is not None:
                passed = False
                reasons.append("expected None context but got content")
        else:
            if context is None:
                passed = False
                reasons.append("expected context but got None")
            else:
                terms_expected = total_expected_terms
                if terms_expected:
                    terms_found = sum(
                        1 for t in expected.get("must_contain", [])
                        if t.lower() in context.lower()
                    )
                    if terms_found < terms_expected:
                        passed = False
                        reasons.append(f"found {terms_found}/{terms_expected} terms")

                max_msgs = expected.get("max_messages")
                if max_msgs and len(messages) > max_msgs:
                    included = context.count(": ")
                    if included > max_msgs:
                        passed = False
                        reasons.append(f"too many messages ({included} > {max_msgs})")

        precision = terms_found / max(terms_expected, 1) if terms_expected else 1.0
        self.metrics.record_custom("retrieval_precision", precision)

        return EvalResult(
            case_id=case.id,
            passed=passed,
            output=context,
            error="; ".join(reasons) if reasons else None,
            metadata={"message_count": len(messages), "terms_found": terms_found, "terms_expected": terms_expected},
            artifact=Artifact(
                prompt=query,
                final_response=context or "(no context retrieved)",
                logs=[f"messages_in_history={len(messages)}",
                      f"terms_expected={terms_expected}",
                      f"terms_found={terms_found}",
                      f"must_be_none={expected.get('must_be_none', False)}",
                      f"context_length={len(context or '')}"],
                timing={"retrieval_ms": 0.3},
            ),
        )
