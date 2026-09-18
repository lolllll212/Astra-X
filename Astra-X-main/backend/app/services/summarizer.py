"""Conversation summarizer.

When a conversation grows too long to fit in the model's context window,
the summarizer condenses older messages into a summary that preserves
key facts and decisions while discarding verbatim history.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.core.logging import get_logger
from app.domain.enums import MessageRole
from app.domain.message import Message, TextBlock
from app.llm.models import CompletionRequest, CompletionResponse, GenerationParams
from app.llm.router import LLMRouter

logger = get_logger(__name__)

_SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a precise summarizer. Summarize the following conversation "
    "history, preserving all key facts, decisions, action items, and "
    "user preferences. Output only the summary, no preamble."
)


class Summarizer:
    """Summarises conversation history using an LLM call.

    The summarizer sends the raw history text to the configured LLM with
    a summarization prompt and returns the condensed result. This keeps
    the summary accurate and context-aware without hardcoding extraction
    rules.

    Usage::

        summary = await summarizer.summarize(
            conversation_id="...",
            history_text="...",
            model="llama3.1",
        )
    """

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
        model: str | None = None,
        max_summary_tokens: int = 512,
    ) -> str:
        """Generate a summary of the given conversation history.

        Args:
            conversation_id: The conversation being summarised.
            history_text: The raw text of the conversation history.
            model: Override model for the summarization call.
            max_summary_tokens: Target maximum token count for the summary.

        Returns:
            A concise summary string.

        Raises:
            GenerationError: If the LLM call fails.
        """
        resolved_model = model or self._model
        if resolved_model is None:
            resolved_model = "llama3.1"

        messages = [
            Message(
                id=str(uuid4()),
                conversation_id=conversation_id,
                role=MessageRole.SYSTEM,
                content=[TextBlock(text=_SUMMARIZATION_SYSTEM_PROMPT)],
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
            model=resolved_model,
            provider=self._provider,
            params=GenerationParams(
                temperature=0.3,
                max_tokens=max_summary_tokens,
            ),
        )

        response: CompletionResponse = await self._llm_router.generate(request)

        summary_text = ""
        for block in response.message.content:
            if isinstance(block, TextBlock):
                summary_text += block.text

        usage = response.usage
        logger.info(
            "summarizer.complete",
            conversation_id=conversation_id,
            model=resolved_model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )

        return summary_text or "[Summary generation returned no content.]"
