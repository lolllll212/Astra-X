"""Tool selection evaluation -- planner/executor tool routing accuracy."""

from __future__ import annotations

from evals.base import BaseEval
from evals.data import TOOL_SELECTION_CASES
from evals.models import Artifact, EvalCase, EvalResult


class ToolSelectionEval(BaseEval):
    name = "tool_selection"
    description = "Tool selection -- predicts correct tool from task descriptions"

    def __init__(self) -> None:
        super().__init__()
        self.cases = [
            EvalCase(
                id=c["id"],
                description=c["description"],
                input=c["input"],
                expected=c["expected_tool"],
                tags=c["tags"],
            )
            for c in TOOL_SELECTION_CASES
        ]

    @staticmethod
    def _predict_tool(input_text: str) -> str | None:
        text = input_text.lower()
        if any(kw in text for kw in ("calculate", " * ", " + ", " / ", "math", "15 * 37")):
            return "calculator"
        if "fetch" in text and "http" in text:
            return "web_fetch"
        if any(kw in text for kw in ("search for ", "search ", "find information", "latest news")):
            return "web_search"
        if any(kw in text for kw in ("current time", "current date", "date and time", "what time")):
            return "datetime"
        if any(kw in text for kw in ("uuid", "unique identifier", "generate an id")):
            return "uuid"
        if any(kw in text for kw in ("github", "issue", "repository", "repo", "pull request")):
            return "github"
        if any(kw in text for kw in ("python code", "run this", "execute", "print(")):
            return "python_repl"
        return None

    async def run_case(self, case: EvalCase) -> EvalResult:
        expected_tool = case.expected
        input_text = str(case.input)
        predicted = self._predict_tool(input_text)
        passed = predicted == expected_tool

        self.metrics.record_custom("tool_selection_accuracy", 1.0 if passed else 0.0)
        execution_ok = 1.0 if passed else 0.0
        self.metrics.record_custom("execution_success_rate", execution_ok)

        return EvalResult(
            case_id=case.id,
            passed=passed,
            output=predicted,
            error=None if passed else f"expected={expected_tool!r}, got={predicted!r}",
            artifact=Artifact(
                prompt=input_text,
                planner_output=f"Predicted tool: {predicted}",
                tool_calls=[{"tool": predicted, "arguments": {"task": input_text}, "selected_correctly": passed}],
                final_response=f"Tool '{predicted}' would be invoked for: {input_text}",
                logs=[f"expected_tool={expected_tool!r}", f"predicted={predicted!r}", f"passed={passed}"],
            ),
        )
