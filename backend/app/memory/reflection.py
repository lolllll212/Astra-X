"""Long-term reflection over conversation history.

Instead of storing raw interactions, reflection extracts structured
insights: preferences, facts, decisions, and behavioral patterns.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.core.logging import get_logger
from app.domain.enums import MemoryScope, MessageRole
from app.domain.message import Message, TextBlock
from app.llm.models import CompletionRequest, CompletionResponse, GenerationParams
from app.llm.router import LLMRouter
from app.memory.models.memory import Memory, MemoryType

logger = get_logger(__name__)

_REFLECTION_SYSTEM_PROMPT = (
    "You are a precise memory extractor. Analyze the following conversation "
    "and extract structured memories as a JSON array. Each memory should have:\n"
    "- content: a concise factual statement\n"
    "- type: 'episodic' (specific event), 'semantic' (general fact), "
    "or 'procedural' (how-to knowledge)\n"
    "- importance: a float between 0.0 (trivial) and 1.0 (critical)\n\n"
    "Only extract information worth remembering. Ignore greetings, small talk, "
    "and filler. Output ONLY the JSON array, no preamble."
)


class Reflection:
    """Extracts structured memories from conversation history using an LLM."""

    def __init__(
        self,
        llm_router: LLMRouter,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        self._llm_router = llm_router
        self._model = model
        self._provider = provider

    async def reflect(self, conversation_id: str, history_text: str) -> list[Memory]:
        """Extract memories from a conversation transcript.

        Args:
            conversation_id: Source conversation.
            history_text: Formatted conversation history.

        Returns:
            A list of extracted Memory objects.
        """
        messages = [
            Message(
                id=str(uuid4()),
                conversation_id=conversation_id,
                role=MessageRole.SYSTEM,
                content=[TextBlock(text=_REFLECTION_SYSTEM_PROMPT)],
                created_at=datetime.now(UTC),
            ),
            Message(
                id=str(uuid4()),
                conversation_id=conversation_id,
                role=MessageRole.USER,
                content=[TextBlock(text=history_text)],
                created_at=datetime.now(UTC),
            ),
        ]

        request = CompletionRequest(
            messages=messages,
            model=self._model or "llama3.1",
            provider=self._provider,
            params=GenerationParams(temperature=0.2, max_tokens=1024),
        )

        response: CompletionResponse = await self._llm_router.generate(request)

        text = ""
        for block in response.message.content:
            if isinstance(block, TextBlock):
                text += block.text

        memories = self._parse_memories(text, conversation_id)
        logger.info(
            "reflection.extracted",
            conversation_id=conversation_id,
            count=len(memories),
        )
        return memories

    def _parse_memories(self, text: str, conversation_id: str) -> list[Memory]:
        json_str = text.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("\n", 1)[-1]
            json_str = json_str.rsplit("```", 1)[0]

        try:
            items: list[dict[str, Any]] = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("reflection.parse_failed", raw=text[:200])
            return []

        memories: list[Memory] = []
        for item in items:
            try:
                content = item.get("content", "")
                if not content:
                    continue
                type_str = item.get("type", "episodic")
                importance = float(item.get("importance", 0.5))
                memory_type = MemoryType(type_str) if type_str in tuple(t.value for t in MemoryType) else MemoryType.EPISODIC

                memories.append(
                    Memory(
                        id=str(uuid4()),
                        content=content,
                        memory_type=memory_type,
                        scope=MemoryScope.CONVERSATION,
                        importance=min(max(importance, 0.0), 1.0),
                        conversation_id=conversation_id,
                        metadata={"source": "reflection"},
                    )
                )
            except (ValueError, TypeError) as exc:
                logger.warning("reflection.item_skip", error=str(exc), item=item)

        return memories
