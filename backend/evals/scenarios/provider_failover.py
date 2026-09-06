"""Provider failover evaluation -- fallback, retry, and health-check behavior."""

from __future__ import annotations

from evals.base import BaseEval
from evals.data import PROVIDER_FAILOVER_CASES
from evals.models import Artifact, EvalCase, EvalResult


class ProviderFailoverEval(BaseEval):
    name = "provider_failover"
    description = "Provider failover -- fallback routing, retry logic, health-check gating"

    def __init__(self) -> None:
        super().__init__()
        self.cases = [
            EvalCase(**c)  # type: ignore[arg-type]
            for c in PROVIDER_FAILOVER_CASES
        ]

    @staticmethod
    def _simulate_failover(data: dict) -> dict:
        scenario_type = _detect_scenario(data)

        if scenario_type == "primary_dies":
            if data.get("fallback", {}).get("status") == "success":
                return {
                    "failover": True,
                    "success": True,
                    "provider_used": data["providers"][1],
                    "latency_ms": data["fallback"]["latency_ms"],
                    "retries": 1,
                }
            return {
                "failover": True,
                "success": False,
                "provider_used": None,
                "error": "all providers failed",
                "retries": len(data["providers"]),
            }

        if scenario_type == "recovery":
            attempts = data.get("primary_attempts", [])
            for i, attempt in enumerate(attempts):
                if attempt["status"] == "success":
                    return {
                        "failover": False,
                        "success": True,
                        "provider_used": data["providers"][0] if isinstance(data.get("providers"), list) else None,
                        "latency_ms": attempt.get("latency_ms", 0),
                        "retries": i,
                    }
            return {"failover": False, "success": False, "error": "all retries failed", "retries": len(attempts)}

        if scenario_type == "health_check":
            health = data.get("health_results", {})
            primary = data.get("primary", "")
            if health.get(primary, True):
                return {"failover": False, "healthy_provider": primary, "success": True}
            for provider in data.get("providers", []):
                if provider != primary and health.get(provider, False):
                    return {"failover": True, "healthy_provider": provider, "success": True}
            return {"failover": True, "healthy_provider": None, "success": False}

        return {"failover": False, "success": False, "error": "unknown scenario"}

    async def run_case(self, case: EvalCase) -> EvalResult:
        data: dict = case.input
        expected: dict = case.expected

        result = self._simulate_failover(data)
        passed = True
        reasons: list[str] = []

        if expected.get("should_failover") and not result["failover"]:
            passed = False
            reasons.append("expected failover but none occurred")

        if not expected.get("should_failover") and result["failover"]:
            passed = False
            reasons.append("failover occurred when not expected")

        if "should_succeed" in expected:
            if expected["should_succeed"] and not result["success"]:
                passed = False
                reasons.append("expected overall success but failed")
            if not expected["should_succeed"] and result["success"]:
                passed = False
                reasons.append("expected overall failure but succeeded")

        if expected.get("retry_count") is not None and result.get("retries", 0) != expected["retry_count"]:
                passed = False
                reasons.append(f"expected {expected['retry_count']} retries, got {result.get('retries', 0)}")

        self.metrics.record_custom("failover_success", 1.0 if result["success"] else 0.0)
        self.metrics.record_custom("failover_occurred", 1.0 if result["failover"] else 0.0)

        return EvalResult(
            case_id=case.id,
            passed=passed,
            output=result,
            metadata={
                "failover": result["failover"],
                "success": result["success"],
                "retries": result.get("retries", 0),
            },
            error="; ".join(reasons) if reasons else None,
            artifact=Artifact(
                prompt="Provider failover scenario",
                tool_calls=[{"data": data, "result": result}],
                final_response=str(result),
                logs=[f"failover={result['failover']}",
                      f"success={result['success']}",
                      f"retries={result.get('retries', 0)}",
                      f"provider_used={result.get('provider_used')}"],
            ),
        )


def _detect_scenario(data: dict) -> str:
    if "health_results" in data:
        return "health_check"
    if "primary_attempts" in data:
        return "recovery"
    return "primary_dies"
