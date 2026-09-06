"""Mixed text/image input evaluation -- multimodal handling (stub)."""

from __future__ import annotations

from evals.base import BaseEval
from evals.data import MIXED_INPUT_CASES
from evals.models import Artifact, EvalCase, EvalResult


class MixedInputEval(BaseEval):
    name = "mixed_input"
    description = "Mixed text/image input -- multimodal detection, attachment handling, fallback (stub)"

    def __init__(self) -> None:
        super().__init__()
        self.cases = [
            EvalCase(**c)  # type: ignore[arg-type]
            for c in MIXED_INPUT_CASES
        ]

    @staticmethod
    def _simulate_multimodal_handler(text: str, attachments: list[dict]) -> dict:
        has_image = any(a.get("type") == "image" for a in attachments)
        return {
            "has_image": has_image,
            "image_supported": False,
            "fallback_triggered": has_image,
            "response": f"Processed text input ({len(text)} chars)" if not has_image else "Image received but processing not yet supported.",
        }

    async def run_case(self, case: EvalCase) -> EvalResult:
        data: dict = case.input
        expected: dict = case.expected

        result = self._simulate_multimodal_handler(data.get("text", ""), data.get("attachments", []))
        passed = True
        reasons: list[str] = []

        if expected.get("image_supported") is not None and result["image_supported"] != expected["image_supported"]:
            passed = False
            reasons.append(f"expected image_supported={expected['image_supported']}, got {result['image_supported']}")

        if expected.get("fallback_triggered") is not None and result["fallback_triggered"] != expected["fallback_triggered"]:
            passed = False
            reasons.append(f"expected fallback_triggered={expected['fallback_triggered']}, got {result['fallback_triggered']}")

        self.metrics.record_custom("fallback_triggered", 1.0 if result["fallback_triggered"] else 0.0)
        self.metrics.record_custom("image_present", 1.0 if result["has_image"] else 0.0)

        return EvalResult(
            case_id=case.id,
            passed=passed,
            output=result["response"],
            metadata={
                "has_image": result["has_image"],
                "image_supported": result["image_supported"],
                "fallback_triggered": result["fallback_triggered"],
            },
            error="; ".join(reasons) if reasons else None,
            artifact=Artifact(
                prompt=f"Mixed input: text={len(data.get('text', ''))} chars, attachments={len(data.get('attachments', []))}",
                final_response=result["response"],
                logs=[f"has_image={result['has_image']}",
                      f"image_supported={result['image_supported']}",
                      f"fallback_triggered={result['fallback_triggered']}"],
            ),
        )
