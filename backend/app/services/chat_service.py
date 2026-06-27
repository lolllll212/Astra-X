"""Chat orchestration service.

:class:`ChatService` is the central orchestrator for every chat
interaction. It receives a user message, coordinates all sub-services
(context building, memory, prompt assembly, token counting, LLM
routing), persists the result, and returns the assistant's response.

This is the only service the API layer talks to for chat — everything
AI-related flows through here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from app.core.logging import get_logger
from app.database.repositories.conversation_repository import ConversationRepository
from app.database.repositories.message_repository import MessageRepository
from app.database.repositories.usage_repository import UsageRepository
from app.domain.conversation import Conversation
from app.domain.enums import MessageRole
from app.domain.message import ContentBlock, Message
from app.domain.usage import Usage
from app.llm.models import CompletionRequest, CompletionResponse, GenerationParams
from app.llm.router import LLMRouter

if TYPE_CHECKING:
    from app.services.context_builder import ContextBuilder
    from app.services.conversation_service import ConversationService
    from app.services.memory_service import MemoryService
    from app.services.prompt_builder import PromptBuilder
    from app.services.token_counter import TokenCounter

logger = get_logger(__name__)


@dataclass(frozen=True)
class ChatResult:
    """The result of processing a single user message.

    Attributes:
        assistant_message: The assistant's response message.
        conversation: The updated conversation.
        usage: Token usage for the generation, if available.
    """

    assistant_message: Message
    conversation: Conversation
    usage: Usage | None = None


class ChatService:
    """Orchestrates the full chat lifecycle.

    Usage::

        result = await chat_service.process_message(
            conversation_id="...",
            user_content=[TextBlock(text="Hello")],
        )
        print(result.assistant_message.content)
    """

    def __init__(
        self,
        conversation_repo: ConversationRepository,
        message_repo: MessageRepository,
        usage_repo: UsageRepository,
        llm_router: LLMRouter,
        conversation_service: ConversationService | None = None,
        memory_service: MemoryService | None = None,
        context_builder: ContextBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._conversation_repo = conversation_repo
        self._message_repo = message_repo
        self._usage_repo = usage_repo
        self._llm_router = llm_router

        # Optional sub-services — wired once they are implemented.
        self._conversation_service = conversation_service
        self._memory_service = memory_service
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._token_counter = token_counter

    async def process_message(
        self,
        conversation_id: str,
        user_content: list[ContentBlock],
        *,
        model: str | None = None,
        provider: str | None = None,
        params: GenerationParams | None = None,
    ) -> ChatResult:
        """Process a user message and return the assistant's response.

        The full pipeline:

        1. Load and validate the conversation.
        2. Persist the user message.
        3. Load conversation history.
        4. Load memory context (if memory service wired).
        5. Build context (if context builder wired).
        6. Build prompt messages (if prompt builder wired).
        7. Count tokens / trim (if token counter wired).
        8. Route to LLM.
        9. Persist the assistant response.
        10. Persist usage.
        11. Return result.

        Args:
            conversation_id: The conversation to continue.
            user_content: The user's message content blocks.
            model: Optional model override.
            provider: Optional provider override.
            params: Optional generation parameter overrides.

        Returns:
            The assistant's response, updated conversation, and usage.

        Raises:
            ResourceNotFoundError: If the conversation does not exist.
        """
        # Step 1 — Load and validate conversation.
        conversation = await self._conversation_repo.get(conversation_id)
        if conversation is None:
            from app.core.exceptions import ResourceNotFoundError
            raise ResourceNotFoundError(
                message=f"Conversation '{conversation_id}' not found.",
            )

        # Step 2 — Persist the user message.
        user_message = Message(
            id=str(uuid4()),
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=user_content,
            created_at=datetime.now(UTC),
        )
        await self._message_repo.add(user_message)

        # Step 3 — Load conversation history.
        history = await self._message_repo.list_by_conversation(conversation_id)

        # Step 4 — Load memory context.
        memory_context: str | None = None
        if self._memory_service is not None:
            memory_context = await self._memory_service.get_relevant_context(
                conversation_id=conversation_id,
                user_content=user_content,
            )

        # Step 5 — Build enriched context.
        context_messages: list[Message] = history
        if self._context_builder is not None:
            context_messages = await self._context_builder.build(
                conversation=conversation,
                history=history,
                user_message=user_message,
                memory_context=memory_context,
            )

        # Step 6 — Build final prompt for the LLM.
        prompt_messages: list[Message] = context_messages
        if self._prompt_builder is not None:
            prompt_messages = await self._prompt_builder.build(
                conversation=conversation,
                context_messages=context_messages,
            )

        # Step 7 — Token counting and context window trimming.
        if self._token_counter is not None:
            prompt_messages = await self._token_counter.trim_to_fit(
                messages=prompt_messages,
                model=model or conversation.metadata.model_id,
            )

        # Step 8 — Route to LLM.
        llm_request = CompletionRequest(
            messages=prompt_messages,
            model=model or conversation.metadata.model_id or "",
            provider=provider or conversation.metadata.provider_id,
            params=params or GenerationParams(),
        )

        response: CompletionResponse = await self._llm_router.generate(llm_request)

        # Step 9 — Persist the assistant response.
        assistant_message = response.message
        await self._message_repo.add(assistant_message)

        # Step 10 — Persist usage.
        if response.usage is not None:
            await self._usage_repo.add_for_message(assistant_message.id, response.usage)

        # Update conversation denormalised message count.
        conversation.message_count += 1
        conversation.updated_at = datetime.now(UTC)
        await self._conversation_repo.update(conversation)

        logger.info(
            "chat.message_processed",
            conversation_id=conversation_id,
            message_id=assistant_message.id,
            model=response.model or model,
            provider=provider,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
        )

        return ChatResult(
            assistant_message=assistant_message,
            conversation=conversation,
            usage=response.usage,
        )

    async def process_message_stream(
        self,
        conversation_id: str,
        user_content: list[ContentBlock],
        *,
        model: str | None = None,
        provider: str | None = None,
        params: GenerationParams | None = None,
    ) -> ChatResult:
        """Process a user message and collect the streaming response.

        Behaves like :meth:`process_message` but uses the streaming
        code path internally. The events are accumulated via
        :class:`app.llm.streaming.StreamCollector` and the final
        result is returned once the stream completes.

        For true per-event streaming (SSE) the API layer should call
        the LLM router directly; this method exists for callers that
        want a streaming-compatible result without managing events.

        Args:
            Same as :meth:`process_message`.

        Returns:
            Same as :meth:`process_message`.
        """
        from app.llm.streaming import collect_stream

        # Steps 1-7: identical to non-streaming path.
        conversation = await self._conversation_repo.get(conversation_id)
        if conversation is None:
            from app.core.exceptions import ResourceNotFoundError
            raise ResourceNotFoundError(
                message=f"Conversation '{conversation_id}' not found.",
            )

        user_message = Message(
            id=str(uuid4()),
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=user_content,
            created_at=datetime.now(UTC),
        )
        await self._message_repo.add(user_message)

        history = await self._message_repo.list_by_conversation(conversation_id)

        memory_context: str | None = None
        if self._memory_service is not None:
            memory_context = await self._memory_service.get_relevant_context(
                conversation_id=conversation_id,
                user_content=user_content,
            )

        context_messages: list[Message] = history
        if self._context_builder is not None:
            context_messages = await self._context_builder.build(
                conversation=conversation,
                history=history,
                user_message=user_message,
                memory_context=memory_context,
            )

        prompt_messages: list[Message] = context_messages
        if self._prompt_builder is not None:
            prompt_messages = await self._prompt_builder.build(
                conversation=conversation,
                context_messages=context_messages,
            )

        if self._token_counter is not None:
            prompt_messages = await self._token_counter.trim_to_fit(
                messages=prompt_messages,
                model=model or conversation.metadata.model_id,
            )

        # Stream and collect.
        llm_request = CompletionRequest(
            messages=prompt_messages,
            model=model or conversation.metadata.model_id or "",
            provider=provider or conversation.metadata.provider_id,
            params=params or GenerationParams(),
            stream=True,
        )

        collector = await collect_stream(
            self._llm_router.generate_stream(llm_request),
            conversation_id=conversation_id,
        )

        assistant_message = collector.build_message()
        await self._message_repo.add(assistant_message)

        usage = collector.usage
        if usage is not None:
            await self._usage_repo.add_for_message(assistant_message.id, usage)

        conversation.message_count += 1
        conversation.updated_at = datetime.now(UTC)
        await self._conversation_repo.update(conversation)

        return ChatResult(
            assistant_message=assistant_message,
            conversation=conversation,
            usage=usage,
        )
