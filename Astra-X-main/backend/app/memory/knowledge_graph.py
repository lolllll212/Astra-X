"""Entity-relationship knowledge graph.

Extracts and stores (subject, predicate, object) triples from
conversations to enable richer retrieval than plain vector search.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import uuid4

from app.core.logging import get_logger
from app.domain.enums import MessageRole
from app.domain.message import Message, TextBlock
from app.llm.models import CompletionRequest, CompletionResponse, GenerationParams
from app.llm.router import LLMRouter
from app.memory.models.knowledge import KnowledgeTriple

logger = get_logger(__name__)

_KG_EXTRACTION_PROMPT = (
    "Extract entity-relationship triples from the following text. "
    "Output a JSON array of objects with 'subject', 'predicate', and 'object' fields. "
    "Use simple present tense for predicates. "
    "Example: [{\"subject\": \"Alice\", \"predicate\": \"plays\", \"object\": \"badminton\"}]\n\n"
    "Output ONLY the JSON array, no explanation."
)


class KnowledgeGraph:
    """Manages entity-relationship triples extracted from conversations."""

    def __init__(
        self,
        llm_router: LLMRouter | None = None,
        model: str | None = None,
    ) -> None:
        self._llm_router = llm_router
        self._model = model
        self._triples: dict[str, KnowledgeTriple] = {}

    def add_triple(self, triple: KnowledgeTriple) -> None:
        self._triples[triple.id] = triple

    def add_triples(self, triples: Sequence[KnowledgeTriple]) -> None:
        for t in triples:
            self._triples[t.id] = t

    def get_triple(self, triple_id: str) -> KnowledgeTriple | None:
        return self._triples.get(triple_id)

    def query(self, *, subject: str | None = None, predicate: str | None = None, object_: str | None = None) -> list[KnowledgeTriple]:
        results = list(self._triples.values())
        if subject:
            results = [t for t in results if subject.lower() in t.subject.lower()]
        if predicate:
            results = [t for t in results if predicate.lower() in t.predicate.lower()]
        if object_:
            results = [t for t in results if object_.lower() in t.object_.lower()]
        return results

    def query_entity(self, entity: str) -> dict[str, list[dict[str, str]]]:
        related = self.query(subject=entity) + self.query(object_=entity)
        as_subject = []
        as_object = []
        seen: set[str] = set()
        for t in related:
            if t.id in seen:
                continue
            seen.add(t.id)
            if entity.lower() in t.subject.lower():
                as_subject.append({"predicate": t.predicate, "object": t.object_})
            if entity.lower() in t.object_.lower():
                as_object.append({"subject": t.subject, "predicate": t.predicate})
        return {"as_subject": as_subject, "as_object": as_object}

    async def extract_from_text(self, text: str, conversation_id: str) -> list[KnowledgeTriple]:
        if self._llm_router is None:
            return []

        messages = [
            Message(
                id=str(uuid4()),
                conversation_id=conversation_id,
                role=MessageRole.SYSTEM,
                content=[TextBlock(text=_KG_EXTRACTION_PROMPT)],
                created_at=datetime.now(UTC),
            ),
            Message(
                id=str(uuid4()),
                conversation_id=conversation_id,
                role=MessageRole.USER,
                content=[TextBlock(text=text)],
                created_at=datetime.now(UTC),
            ),
        ]

        request = CompletionRequest(
            messages=messages,
            model=self._model or "llama3.1",
            params=GenerationParams(temperature=0.1, max_tokens=1024),
        )

        response: CompletionResponse = await self._llm_router.generate(request)
        output = ""
        for block in response.message.content:
            if isinstance(block, TextBlock):
                output += block.text

        triples = self._parse_triples(output, conversation_id)
        self.add_triples(triples)
        return triples

    def _parse_triples(self, text: str, conversation_id: str) -> list[KnowledgeTriple]:
        json_str = text.strip()
        if json_str.startswith("```"):
            lines = json_str.split("\n", 1)
            json_str = lines[-1] if len(lines) > 1 else ""
            json_str = json_str.rsplit("```", 1)[0]

        try:
            items: list[dict[str, str]] = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("kg.parse_failed", raw=text[:200])
            return []

        triples: list[KnowledgeTriple] = []
        for item in items:
            sub = item.get("subject", "").strip()
            pred = item.get("predicate", "").strip()
            obj = item.get("object", "").strip()
            if not sub or not pred or not obj:
                continue
            triples.append(
                KnowledgeTriple(
                    id=str(uuid4()),
                    subject=sub,
                    predicate=pred,
                    **{"object": obj},
                    source="conversation_extraction",
                    conversation_id=conversation_id,
                )
            )
        return triples

    async def clear(self) -> None:
        self._triples.clear()

    @property
    def count(self) -> int:
        return len(self._triples)
