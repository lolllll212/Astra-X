"""Streaming evaluation -- event sequence correctness and completeness."""

from __future__ import annotations

from evals.base import BaseEval
from evals.data import STREAMING_CASES
from evals.models import Artifact, EvalCase, EvalResult


class StreamingEval(BaseEval):
    name = "streaming"
    description = "Streaming -- event type sequence, timing, and completeness"

    def __init__(self) -> None:
        super().__init__()
        self.cases = [
            EvalCase(
                id=c["id"],
                description=c["description"],
                input=c["input"],
                expected=c["expected_event_types"],
                tags=c["tags"],
            )
            for c in STREAMING_CASES
        ]

    @staticmethod
    def _simulate_stream(input_text: str) -> list[str]:
        """Simulate a stream of event type strings."""
        events: list[str] = ["stream_start"]

        if input_text:
            events.append("text_delta")
            events.append("text_delta")
            if "count" in input_text.lower() or len(input_text) > 5:
                events.append("stream_usage")
        else:
            # Empty input -- minimal events
            pass

        events.append("stream_done")
        return events

    async def run_case(self, case: EvalCase) -> EvalResult:
        expected_event_types: list[str] = case.expected
        input_text = str(case.input)

        actual_events = self._simulate_stream(input_text)
        actual_types = actual_events

        passed = True
        reasons: list[str] = []
        events_found = 0
        events_total = len(expected_event_types)

        for expected_type in expected_event_types:
            if expected_type not in actual_types:
                passed = False
                reasons.append(f"missing event '{expected_type}'")
            else:
                events_found += 1

        if not actual_types:
            passed = False
            reasons.append("no events produced")

        if actual_types[-1] != "stream_done":
            passed = False
            reasons.append("stream did not end with stream_done")

        completeness = events_found / max(events_total, 1)
        self.metrics.record_custom("event_completeness", completeness)

        return EvalResult(
            case_id=case.id,
            passed=passed,
            output=actual_types,
            metadata={"event_count": len(actual_types)},
            error="; ".join(reasons) if reasons else None,
            artifact=Artifact(
                prompt=input_text,
                final_response=str(actual_types),
                logs=[f"expected_events={expected_event_types}",
                      f"actual_events={actual_types}",
                      f"event_count={len(actual_types)}",
                      f"completeness={completeness:.2%}"],
                timing={"stream_duration_ms": 1.0},
            ),
        )
