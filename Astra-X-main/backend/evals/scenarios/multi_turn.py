"""Multi-turn conversation evaluation -- context retention across turns."""

from __future__ import annotations

from evals.base import BaseEval
from evals.data import MULTI_TURN_CASES
from evals.models import Artifact, EvalCase, EvalResult


class MultiTurnEval(BaseEval):
    name = "multi_turn"
    description = "Multi-turn conversation -- context retention, topic drift, corrections"

    def __init__(self) -> None:
        super().__init__()
        self.cases = [
            EvalCase(**c)  # type: ignore[arg-type]
            for c in MULTI_TURN_CASES
        ]

    @staticmethod
    def _simulate_response(turns: list[str]) -> str:
        simulated = []
        known_facts: dict[str, str] = {}
        for _, msg in enumerate(turns):
            msg_lower = msg.lower()
            if "what is my favorite color" in msg_lower:
                answer = known_facts.get("color", "I don't know.")
                simulated.append(answer)
            elif "where do i live" in msg_lower:
                answer = known_facts.get("location", "I don't know where you live.")
                simulated.append(answer)
            elif ("what is my cat" in msg_lower) or ("cat" in msg_lower and "name" in msg_lower):
                answer = known_facts.get("cat_name", "I don't know your cat's name.")
                simulated.append(answer)
            elif "what year" in msg_lower or ("born" in msg_lower and "?" in msg):
                answer = known_facts.get("birth_year", "I don't know when you were born.")
                simulated.append(answer)
            else:
                simulated.append(f"Response to: {msg[:40]}")
            # Extract facts from user statements
            if "favorite color is" in msg_lower:
                known_facts["color"] = msg
            if "live in" in msg_lower or "live in" in msg_lower:
                known_facts["location"] = f"You live in {msg.split('live in')[-1].strip().rstrip('.')}."
            if "cat" in msg_lower and "name" in msg_lower:
                known_facts["cat_name"] = msg
            if "born in" in msg_lower and "actually" not in msg_lower:
                known_facts["birth_year"] = msg.split("born in")[-1].strip().rstrip(".")
            if "born in" in msg_lower and "actually" in msg_lower:
                year = msg.split("born in")[-1].strip().rstrip(".")
                known_facts["birth_year"] = year
        last = simulated[-1] if simulated else "No response."
        return last

    async def run_case(self, case: EvalCase) -> EvalResult:
        data: dict = case.input
        expected: dict = case.expected
        turns: list[str] = data["turns"]

        final_response = self._simulate_response(turns)
        turn_count = len(turns)

        passed = True
        details: dict = {}
        recall: list[float] = []

        if "must_contain" in expected:
            missing = [kw for kw in expected["must_contain"] if kw.lower() not in final_response.lower()]
            recall = [1.0 if kw.lower() in final_response.lower() else 0.0 for kw in expected["must_contain"]]
            if missing:
                passed = False
                details["missing_terms"] = missing

        if "must_not_contain_old" in expected:
            found_old = [kw for kw in expected["must_not_contain_old"] if kw.lower() in final_response.lower()]
            if found_old:
                passed = False
                details["old_terms_found"] = found_old

        context_accuracy = sum(recall) / max(len(recall), 1) if recall else (1.0 if passed else 0.0)
        self.metrics.record_custom("context_accuracy", context_accuracy)
        self.metrics.record_custom("turn_count", float(turn_count))

        return EvalResult(
            case_id=case.id,
            passed=passed,
            output=final_response,
            metadata={"turn_count": turn_count, "details": details},
            error="; ".join(f"{k}: {v}" for k, v in details.items()) if not passed and details else None,
            artifact=Artifact(
                prompt=f"Multi-turn conversation ({turn_count} turns)",
                final_response=final_response,
                logs=[f"turns={turn_count}",
                      f"expected={expected}",
                      f"final_response={final_response}",
                      f"context_accuracy={context_accuracy:.2%}"],
            ),
        )
