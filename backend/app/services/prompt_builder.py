"""Prompt builder.

Transforms domain-level context messages into the final message list
sent to the LLM. This is the last transformation before token counting
and routing — it handles formatting, instruction injection, and
provider-specific adaptations.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.conversation import Conversation
from app.domain.message import Message

logger = get_logger(__name__)


class PromptBuilder:
    """Builds the final LLM prompt from domain context messages.

    The builder is responsible for:

    * Merging system messages.
    * Injecting formatting instructions or guardrails.
    * Applying provider-specific message format adjustments.
    * Ensuring messages are in the correct order (system → history → user).

    This class never touches the database or HTTP — it is a pure
    transformation layer.
    """

    async def build(
        self,
        *,
        conversation: Conversation,
        context_messages: list[Message],
    ) -> list[Message]:
        """Transform context messages into the final prompt.

        Args:
            conversation: The conversation (used for metadata like
                model preferences that may affect formatting).
            context_messages: The enriched context from
                :class:`ContextBuilder`.

        Returns:
            The final list of messages ready for token counting and
            the LLM router.
        """
        result = list(context_messages)

        logger.debug(
            "prompt.built",
            total_messages=len(result),
            conversation_id=conversation.id,
        )

        return result
