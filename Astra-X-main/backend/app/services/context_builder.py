"""Context builder.

Assembles the final list of messages sent to the LLM by combining
conversation metadata, system prompt, memory context, conversation
history, attachments, and the current user message.

This is the single place where all context sources are merged before
prompt assembly.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.conversation import Conversation
from app.domain.enums import MessageRole
from app.domain.message import Message, TextBlock

logger = get_logger(__name__)


class ContextBuilder:
    """Builds the message context for an LLM request.

    The builder applies the following transformations in order:

    1. Prepends a system message from the conversation's ``system_prompt``
       (if set).
    2. Injects memory context as a system message (if available).
    3. Appends the full conversation history.
    4. Appends the current user message.

    Future iterations will also inject retrieved documents, attachment
    summaries, and tool-result context.
    """

    def __init__(self) -> None:
        self._next_id: int = 0

    async def build(
        self,
        *,
        conversation: Conversation,
        history: list[Message],
        user_message: Message,
        memory_context: str | None = None,
    ) -> list[Message]:
        """Build the context message list.

        Args:
            conversation: The current conversation (used for its metadata
                and system prompt).
            history: The conversation message history (may include the
                user message already, depending on when it was persisted).
            user_message: The current user message.
            memory_context: Optional context string from the memory service.

        Returns:
            An ordered list of messages ready for prompt building and
            token counting.
        """
        result: list[Message] = []

        # 1. System prompt from conversation metadata.
        system_prompt = conversation.metadata.system_prompt
        if system_prompt:
            result.append(self._make_system_message(system_prompt))

        # 2. Memory context as a system message (before history so the
        #    model sees it as background context).
        if memory_context:
            result.append(
                self._make_system_message(
                    f"Relevant conversation context:\n{memory_context}",
                ),
            )

        # 3. Conversation history.
        result.extend(history)

        # 4. Current user message (may already be in history depending
        #    on the service layer's save order; deduplicate by ID).
        if not any(msg.id == user_message.id for msg in result):
            result.append(user_message)

        logger.debug(
            "context.built",
            system_prompt=bool(system_prompt),
            memory_context=bool(memory_context),
            history_count=len(history),
            total_messages=len(result),
        )

        return result

    def _make_system_message(self, text: str) -> Message:
        """Create a system message with the given text.

        Args:
            text: The system instruction text.

        Returns:
            A system-role message.
        """
        self._next_id += 1
        import uuid
        return Message(
            id=str(uuid.uuid4()),
            conversation_id="",
            role=MessageRole.SYSTEM,
            content=[TextBlock(text=text)],
        )
