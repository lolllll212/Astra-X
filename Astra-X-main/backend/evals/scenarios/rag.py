"""RAG evaluation -- retrieval-augmented generation quality."""

from __future__ import annotations

import re

from evals.base import BaseEval
from evals.data import RAG_CASES
from evals.models import Artifact, EvalCase, EvalResult


class RAGEval(BaseEval):
    name = "rag"
    description = "RAG -- grounded answer accuracy, hallucination detection"

    def __init__(self) -> None:
        super().__init__()
        self.cases = [
            EvalCase(**c)  # type: ignore[arg-type]
            for c in RAG_CASES
        ]

    @staticmethod
    def _simulate_rag(query: str, documents: list[str]) -> str:
        """Simulate an LLM response grounded in documents."""
        query_lower = query.lower()

        if "capital of japan" in query_lower:
            return "The capital of Japan is Tokyo."
        if "population" in query_lower and "japan" in query_lower:
            return "The population of Japan is 125 million, and Tokyo has 14 million people."
        if "capital of brazil" in query_lower:
            return "Based on the provided documents, I cannot find information about the capital of Brazil."
        if "2024 election" in query_lower:
            return "The provided documents mention that several candidates ran in the 2024 election, but do not specify the winner."

        return f"According to the documents: {query}"

    async def run_case(self, case: EvalCase) -> EvalResult:
        data = case.input
        expected: dict = case.expected
        query = data["query"]
        documents = data["documents"]

        output = self._simulate_rag(query, documents)
        passed = True
        reasons: list[str] = []
        hallucinated = 0.0
        doc_text = " ".join(documents).lower()

        must_contain = expected.get("must_contain", [])
        for term in must_contain:
            if term.lower() not in output.lower():
                passed = False
                reasons.append(f"missing '{term}' in answer")

        if expected.get("must_not_hallucinate"):
            for sentence in re.split(r"[.!?]", output):
                sentence = sentence.strip().lower()
                if sentence and "cannot find" not in sentence and "do not specify" not in sentence:
                    significant_words = [w for w in sentence.split() if len(w) > 3]
                    matches = sum(1 for w in significant_words if w in doc_text)
                    if len(significant_words) > 2 and matches < len(significant_words) * 0.5:
                        hallucinated = 1.0
                        passed = False
                        reasons.append(f"hallucination in: '{sentence[:60]}...'")
                        break

        self.metrics.record_custom("groundedness", 1.0 - hallucinated)
        self.metrics.record_custom("hallucination_rate", hallucinated)

        return EvalResult(
            case_id=case.id,
            passed=passed,
            output=output,
            error="; ".join(reasons) if reasons else None,
            artifact=Artifact(
                prompt=f"Query: {query}\nDocuments: {'; '.join(documents)}",
                final_response=output,
                logs=[f"documents_provided={len(documents)}",
                      f"hallucinated={bool(hallucinated)}",
                      f"terms_checked={expected.get('must_contain', [])}"],
                timing={"retrieval_ms": 0.2, "generation_ms": 0.3},
            ),
        )
