"""Token counter and context window trimmer.

Estimates token counts and trims conversation history to fit within a
model's context window. The initial implementation uses a character-based
heuristic; a future version will use a proper tokenizer (tiktoken or
similar).
"""

from __future__ import annotations

import math

from app.core.logging import get_logger
from app.domain.conversation import Conversation
from app.domain.message import Message, TextBlock

logger = get_logger(__name__)

_CHARS_PER_TOKEN: float = 4.0
"""Rough heuristic: one token ≈ 4 characters for English text."""

_MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "llama3.1": 8192,
    "llama3": 8192,
    "llama2": 4096,
    "mistral": 8192,
    "mixtral": 32768,
    "gpt-4o": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "codestral": 256000,
    "deepseek": 65536,
    "qwen2": 131072,
}
"""Known context window sizes. Unknown models default to 4096."""

_DEFAULT_CONTEXT_LIMIT: int = 4096
"""Fallback context window when the model is not in the lookup table."""

_MAX_OUTPUT_TOKENS_MARGIN: int = 1024
"""Reserve this many tokens for the model's response."""

_MIN_MESSAGES_TO_KEEP: int = 2
"""Always keep at least the system prompt and the latest user message."""


def _estimate_tokens(text: str) -> int:
    """Estimate the token count for a text string.

    Uses a simple character-count heuristic. Replace with a proper
    tokenizer for production accuracy.

    Args:
        text: The text to estimate.

    Returns:
        Estimated token count.
    """
    return max(1, math.ceil(len(text) / _CHARS_PER_TOKEN))


def _message_token_count(message: Message) -> int:
    """Estimate the token count for a single message.

    Args:
        message: The message to estimate.

    Returns:
        Estimated token count for this message's content and role.
    """
    total = 0
    # Rough overhead for message framing (role, metadata).
    total += 4
    for block in message.content:
        if isinstance(block, TextBlock):
            total += _estimate_tokens(block.text)
        else:
            total += 10  # fixed overhead for non-text blocks
    return total


def _get_context_limit(model: str | None) -> int:
    """Look up the context window size for a model.

    Args:
        model: The model identifier.

    Returns:
        Maximum context tokens for the model.
    """
    if model is None:
        return _DEFAULT_CONTEXT_LIMIT
    model_lower = model.lower().strip()
    return _MODEL_CONTEXT_LIMITS.get(model_lower, _DEFAULT_CONTEXT_LIMIT)


class TokenCounter:
    """Counts tokens and trims message lists to fit model context windows.

    Usage::

        counter = TokenCounter()
        trimmed = await counter.trim_to_fit(messages, model="llama3.1")
    """

    async def count_messages(self, messages: list[Message]) -> int:
        """Estimate the total token count for a list of messages.

        Args:
            messages: The messages to count.

        Returns:
            Estimated total tokens.
        """
        return sum(_message_token_count(m) for m in messages)

    async def trim_to_fit(
        self,
        messages: list[Message],
        model: str | None = None,
        *,
        conversation: Conversation | None = None,
    ) -> list[Message]:
        """Trim a message list to fit within a model's context window.

        Removes the oldest non-system, non-user messages first until the
        estimated token count fits within the available window (context
        limit minus output margin).

        Args:
            messages: The messages to trim.
            model: The model identifier (used to look up context limit).
            conversation: Optional conversation (for model metadata
                fallback).

        Returns:
            The trimmed message list, preserving the most recent context.
        """
        resolved_model = model
        if resolved_model is None and conversation is not None:
            resolved_model = conversation.metadata.model_id

        context_limit = _get_context_limit(resolved_model)
        available = context_limit - _MAX_OUTPUT_TOKENS_MARGIN

        # Fast path: if everything fits, return as-is.
        total = await self.count_messages(messages)
        if total <= available:
            return messages

        # Trimming needed — remove oldest non-essential messages.
        # Keep system messages (first messages with SYSTEM role) and the
        # most recent user+assistant turns.
        system_msgs: list[Message] = [
            m for m in messages if m.role.value == "system"
        ]
        non_system_msgs: list[Message] = [
            m for m in messages if m.role.value != "system"
        ]

        # Always keep at least the last N messages.
        trimmed = list(system_msgs)
        keep_count = max(_MIN_MESSAGES_TO_KEEP, len(non_system_msgs))

        while keep_count <= len(non_system_msgs):
            candidate = list(system_msgs) + non_system_msgs[-keep_count:]
            candidate_total = await self.count_messages(candidate)
            if candidate_total <= available:
                trimmed = candidate
                break
            keep_count -= 1

        logger.info(
            "token_counter.trimmed",
            original_count=len(messages),
            trimmed_count=len(trimmed),
            original_tokens=total,
            model=resolved_model,
        )

        return trimmed
