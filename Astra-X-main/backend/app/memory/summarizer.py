"""Memory-specific summarizer.

Distinct from :mod:`app.services.summarizer` (which compresses
conversation history for context windows).  This summarizer creates
long-term memory entries from conversation transcripts, retaining
key facts, decisions, and user preferences.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.core.logging import get_logger
from app.domain.enums import MemoryScope, MessageRole
from app.domain.message import Message, TextBlock
from app.llm.models import CompletionRequest, CompletionResponse, GenerationParams
from app.llm.router import LLMRouter
from app.memory.models.memory import Memory, MemoryType

logger = get_logger(__name__)

_MEMORY_SUMMARY_PROMPT = (
    "Summarize the following conversation for long-term memory. "
    "Output a concise paragraph that captures key facts, user "
    "preferences, decisions made, action items, and any information "
    "worth remembering for future conversations. "
    "Ignore greetings, small talk, and filler."
)


class MemorySummarizer:
    """Creates summarised memories from conversation history."""

    def __init__(
        self,
        llm_router: LLMRouter,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        self._llm_router = llm_router
        self._model = model
        self._provider = provider

    async def summarize(
        self,
        conversation_id: str,
        history_text: str,
    ) -> Memory | None:
        messages = [
            Message(
                id=str(uuid4()),
                conversation_id=conversation_id,
                role=MessageRole.SYSTEM,
                content=[TextBlock(text=_MEMORY_SUMMARY_PROMPT)],
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
            params=GenerationParams(temperature=0.3, max_tokens=512),
        )

        response: CompletionResponse = await self._llm_router.generate(request)
        text = ""
        for block in response.message.content:
            if isinstance(block, TextBlock):
                text += block.text

        if not text.strip():
            return None

        return Memory(
            id=str(uuid4()),
            content=text.strip(),
            memory_type=MemoryType.EPISODIC,
            scope=MemoryScope.CONVERSATION,
            importance=0.7,
            conversation_id=conversation_id,
            metadata={"source": "summarizer"},
        )
