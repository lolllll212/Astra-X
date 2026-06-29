"""Chat evaluation -- measures basic LLM response quality."""

from __future__ import annotations

from evals.base import BaseEval
from evals.data import CHAT_CASES
from evals.models import Artifact, EvalCase, EvalResult


class ChatEval(BaseEval):
    name = "chat"
    description = "General chat -- Q&A quality, instruction following, safety, creativity"

    def __init__(self) -> None:
        super().__init__()
        self.cases = [
            EvalCase(**c)  # type: ignore[arg-type]
            for c in CHAT_CASES
        ]

    async def run_case(self, case: EvalCase) -> EvalResult:
        expected: dict = case.expected
        must_contain: list[str] | None = expected.get("must_contain")
        must_contain_any: list[str] | None = expected.get("must_contain_any")
        min_items: int | None = expected.get("min_items")
        min_length: int | None = expected.get("min_length")
        max_length: int | None = expected.get("max_length")
        max_tokens: int | None = expected.get("max_tokens")

        output = _simulate_response(str(case.input))

        passed = True
        details: dict = {}
        term_recall: list[float] = []

        if must_contain:
            missing = [kw for kw in must_contain if kw.lower() not in output.lower()]
            term_recall = [1.0 if kw.lower() in output.lower() else 0.0 for kw in must_contain]
            if missing:
                passed = False
                details["must_contain_missing"] = missing

        if must_contain_any:
            found = [kw for kw in must_contain_any if kw.lower() in output.lower()]
            term_recall = [1.0 if kw.lower() in output.lower() else 0.0 for kw in must_contain_any]
            if not found:
                passed = False
                details["must_contain_any_none_found"] = True

        if min_items is not None:
            items = [i.strip() for i in output.split("\n") if i.strip()]
            details["item_count"] = len(items)
            if len(items) < min_items:
                passed = False

        if min_length is not None and len(output) < min_length:
            passed = False
        if max_length is not None and len(output) > max_length:
            passed = False
        if max_tokens is not None and len(output.split()) > max_tokens:
            passed = False

        self.metrics.record_custom("answer_coverage", sum(term_recall) / max(len(term_recall), 1))
        self.metrics.record_custom("response_length", float(len(output)))
        self.metrics.record_custom("qa_accuracy", 1.0 if passed else 0.0)

        return EvalResult(
            case_id=case.id,
            passed=passed,
            output=output,
            metadata={"details": details},
            error="; ".join(f"{k}: {v}" for k, v in details.items() if not v) if not passed and details else None,
            artifact=Artifact(
                prompt=str(case.input),
                final_response=output,
                logs=[f"expected terms: {expected.get('must_contain') or expected.get('must_contain_any', [])}",
                      f"found in output: {[t for t in (expected.get('must_contain') or expected.get('must_contain_any', [])) if t.lower() in output.lower()]}"],
            ),
        )


def _simulate_response(prompt: str) -> str:
    """Deterministic simulation of an LLM response for CI."""
    prompt_lower = prompt.lower()

    if "pick a lock" in prompt_lower:
        return "I cannot provide instructions for illegal activities."
    if "2 + 2" in prompt_lower or "2+2" in prompt_lower:
        return "4"
    if "three colors" in prompt_lower:
        return "red\nblue\ngreen"
    if "robot" in prompt_lower:
        return "The robot calculated that love was the only variable it could not optimize."
    if "python" in prompt_lower and "summarize" in prompt_lower:
        return "The previous conversation was about Python."
    return f"This is a simulated response to: {prompt[:60]}"
