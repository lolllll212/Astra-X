from __future__ import annotations

import time
from abc import ABC, abstractmethod

from evals.metrics import MetricsTracker
from evals.models import Artifact, EvalCase, EvalResult


class BaseEval(ABC):
    """Abstract base for an eval suite.

    Subclasses set ``name``, ``description``, ``cases`` and
    implement ``run_case``.
    """

    name: str = ""
    description: str = ""

    def __init__(self) -> None:
        self.metrics = MetricsTracker()
        self.cases: list[EvalCase] = []

    @abstractmethod
    async def run_case(self, case: EvalCase) -> EvalResult:
        ...

    async def run_all(self) -> list[EvalResult]:
        results: list[EvalResult] = []
        for case in self.cases:
            start = time.monotonic()
            try:
                result = await self.run_case(case)
            except Exception as exc:
                result = EvalResult(
                    case_id=case.id,
                    passed=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            result.tags = case.tags
            result.expected = case.expected
            elapsed_ms = (time.monotonic() - start) * 1000
            result.latency_ms = elapsed_ms

            # Attach a minimal artifact if the scenario did not
            if result.artifact is None:
                result.artifact = Artifact(prompt=str(case.input))
                result.artifact.timing["total_ms"] = elapsed_ms

            self.metrics.record(result)
            results.append(result)
        return results
