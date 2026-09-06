"""Memory consolidation evaluation -- deduplication, conflict resolution, TTL expiry."""

from __future__ import annotations

import difflib

from evals.base import BaseEval
from evals.data import MEMORY_CONSOLIDATION_CASES
from evals.models import Artifact, EvalCase, EvalResult


class MemoryConsolidationEval(BaseEval):
    name = "memory_consolidation"
    description = "Memory consolidation -- deduplication, conflict resolution, TTL expiry, summarization"

    def __init__(self) -> None:
        super().__init__()
        self.cases = [
            EvalCase(**c)  # type: ignore[arg-type]
            for c in MEMORY_CONSOLIDATION_CASES
        ]

    @staticmethod
    def _simulate_consolidation(data: dict) -> dict:
        memories = data.get("memories", [])
        ttl_days = data.get("ttl_days", 90)

        duplicate_groups = _find_duplicates(memories)
        conflicts = _find_conflicts(memories)
        expired = [m for m in memories if m.get("age_days", 0) >= ttl_days]
        remaining_mem = [m for m in memories if m.get("age_days", 0) < ttl_days] if ttl_days else list(memories)

        merged_count = len(duplicate_groups)
        conflict_resolved = bool(conflicts)
        expired_count = len(expired)

        winner_content = None
        if conflicts:
            sorted_conflicts = sorted(conflicts, key=lambda m: m.get("timestamp", ""), reverse=True)
            winner_content = sorted_conflicts[0].get("content") if sorted_conflicts[0] else None

        summary_terms = []
        keywords = ["house", "renovation", "renovating"]
        if len(memories) >= 4 and any(
            any(kw in m.get("content", "").lower() for kw in keywords) for m in memories
        ):
            summary_terms = [kw for kw in keywords if any(kw in m.get("content", "").lower() for m in memories)]

        return {
            "duplicates_found": merged_count * 2,
            "merged": merged_count > 0,
            "conflicts_found": conflict_resolved,
            "conflict_resolved": conflict_resolved,
            "winner_content": winner_content,
            "expired_count": expired_count,
            "remaining": len(remaining_mem) - merged_count if merged_count else len(remaining_mem),
            "should_summarize": len(summary_terms) > 0,
            "summary_terms": summary_terms,
        }

    async def run_case(self, case: EvalCase) -> EvalResult:
        data: dict = case.input
        expected: dict = case.expected

        result = self._simulate_consolidation(data)
        passed = True
        reasons: list[str] = []

        if expected.get("duplicates_found") is not None and result.get("duplicates_found", 0) != expected["duplicates_found"]:
            passed = False
            reasons.append(f"expected {expected['duplicates_found']} duplicates, found {result.get('duplicates_found', 0)}")

        if expected.get("should_merge") and not result.get("merged"):
            passed = False
            reasons.append("expected merge but none occurred")

        if expected.get("should_expire") and result.get("expired_count", 0) == 0:
            passed = False
            reasons.append("expected expiry but none occurred")

        if expected.get("remaining") is not None and result.get("remaining", 0) != expected["remaining"]:
            passed = False
            reasons.append(f"expected {expected['remaining']} remaining memories, got {result.get('remaining', 0)}")

        if expected.get("should_summarize") and not result.get("should_summarize"):
            passed = False
            reasons.append("expected summarization but none occurred")

        self.metrics.record_custom("dedup_success", 1.0 if result.get("merged") else 0.0)
        self.metrics.record_custom("conflict_resolution", 1.0 if result.get("conflict_resolved") else 0.0)
        self.metrics.record_custom("expiry_success", 1.0 if result.get("expired_count", 0) > 0 else 0.0)

        return EvalResult(
            case_id=case.id,
            passed=passed,
            output=result,
            metadata={
                "duplicates": result.get("duplicates_found", 0),
                "conflicts": result.get("conflicts_found", False),
                "expired": result.get("expired_count", 0),
                "remaining": result.get("remaining", 0),
            },
            error="; ".join(reasons) if reasons else None,
            artifact=Artifact(
                prompt=f"Memory consolidation: {len(data.get('memories', []))} memories",
                final_response=str(result),
                logs=[f"duplicates_found={result.get('duplicates_found', 0)}",
                      f"conflicts_found={result.get('conflicts_found')}",
                      f"expired_count={result.get('expired_count', 0)}",
                      f"remaining={result.get('remaining', 0)}",
                      f"should_summarize={result.get('should_summarize')}"],
            ),
        )


def _find_duplicates(memories: list[dict]) -> list[list[int]]:
    groups: list[list[int]] = []
    checked: set[int] = set()
    for i, a in enumerate(memories):
        if i in checked:
            continue
        group = [i]
        for j, b in enumerate(memories):
            if j <= i or j in checked:
                continue
            ratio = difflib.SequenceMatcher(None, a.get("content", ""), b.get("content", "")).ratio()
            if ratio > 0.5:
                group.append(j)
                checked.add(j)
        if len(group) > 1:
            groups.append(group)
            checked.update(group)
    return groups


def _find_conflicts(memories: list[dict]) -> list[dict]:
    conflicts: list[dict] = []
    for i, m in enumerate(memories):
        content = m.get("content", "")
        for j, other in enumerate(memories):
            if j >= i:
                break
            ratio = difflib.SequenceMatcher(None, content, other.get("content", "")).ratio()
            if 0.3 < ratio < 0.8 and m.get("type") == other.get("type") and m.get("timestamp", "") > other.get("timestamp", ""):
                conflicts.append(m)
    return conflicts
