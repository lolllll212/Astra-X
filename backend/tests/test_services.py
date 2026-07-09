"""Unit tests for the service layer.

Covers ContextBuilder, ConversationService, and ChatService.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.conversation import Conversation, ConversationMetadata
from app.domain.enums import ConversationStatus, MessageRole
from app.domain.message import Message, TextBlock
from app.domain.usage import Usage
from app.llm.models import CompletionResponse
from app.llm.router import LLMRouter
from app.services.chat_service import ChatResult, ChatService
from app.services.context_builder import ContextBuilder
from app.services.conversation_service import ConversationService
from app.services.usage_service import UsageService

# =========================================================================
# ContextBuilder
# =========================================================================


class TestContextBuilder:
    @pytest.mark.asyncio
    async def test_build_without_system_prompt_or_memory(self) -> None:
        builder = ContextBuilder()
        conv = Conversation(
            id="c1",
            metadata=ConversationMetadata(),
        )
        history = [
            Message(id="h1", conversation_id="c1", role=MessageRole.ASSISTANT, content=[TextBlock(text="hi")]),
        ]
        user_msg = Message(id="u1", conversation_id="c1", role=MessageRole.USER, content=[TextBlock(text="hello")])

        result = await builder.build(
            conversation=conv,
            history=history,
            user_message=user_msg,
        )
        assert len(result) == 2
        assert result[0].id == "h1"
        assert result[1].id == "u1"

    @pytest.mark.asyncio
    async def test_build_with_system_prompt(self) -> None:
        builder = ContextBuilder()
        conv = Conversation(
            id="c1",
            metadata=ConversationMetadata(system_prompt="You are a helpful assistant."),
        )
        history = [
            Message(id="h1", conversation_id="c1", role=MessageRole.USER, content=[TextBlock(text="hi")]),
        ]
        user_msg = Message(id="u1", conversation_id="c1", role=MessageRole.USER, content=[TextBlock(text="hello")])

        result = await builder.build(
            conversation=conv,
            history=history,
            user_message=user_msg,
        )
        assert len(result) == 3
        assert result[0].role is MessageRole.SYSTEM
        assert "helpful assistant" in str(result[0].content)

    @pytest.mark.asyncio
    async def test_build_with_memory_context(self) -> None:
        builder = ContextBuilder()
        conv = Conversation(id="c1", metadata=ConversationMetadata())
        result = await builder.build(
            conversation=conv,
            history=[],
            user_message=Message(id="u1", conversation_id="c1", role=MessageRole.USER, content=[TextBlock(text="q")]),
            memory_context="User likes Python.",
        )
        assert len(result) == 2
        assert result[0].role is MessageRole.SYSTEM
        assert "Python" in str(result[0].content)

    @pytest.mark.asyncio
    async def test_build_deduplicates_user_message(self) -> None:
        builder = ContextBuilder()
        conv = Conversation(id="c1", metadata=ConversationMetadata())
        user_msg = Message(id="u1", conversation_id="c1", role=MessageRole.USER, content=[TextBlock(text="hello")])
        history = [user_msg]

        result = await builder.build(
            conversation=conv,
            history=history,
            user_message=user_msg,
        )
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_build_system_prompt_before_memory(self) -> None:
        builder = ContextBuilder()
        conv = Conversation(
            id="c1",
            metadata=ConversationMetadata(system_prompt="System instruction."),
        )
        result = await builder.build(
            conversation=conv,
            history=[],
            user_message=Message(id="u1", conversation_id="c1", role=MessageRole.USER, content=[TextBlock(text="q")]),
            memory_context="Memory context.",
        )
        assert len(result) == 3
        assert result[0].role is MessageRole.SYSTEM
        assert "System" in str(result[0].content)
        assert result[1].role is MessageRole.SYSTEM
        assert "Memory" in str(result[1].content)


# =========================================================================
# ConversationService
# =========================================================================


class TestConversationService:
    @pytest.fixture
    def repo(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture
    def service(self, repo: AsyncMock) -> ConversationService:
        return ConversationService(repository=repo)

    @pytest.mark.asyncio
    async def test_create(self, service: ConversationService, repo: AsyncMock) -> None:
        repo.add = AsyncMock(side_effect=lambda conv: conv)

        conv = await service.create(title="New Chat", model="llama3.1", provider="ollama")

        assert conv.title == "New Chat"
        assert conv.metadata.model == "llama3.1"
        assert conv.metadata.provider == "ollama"
        assert conv.status is ConversationStatus.ACTIVE
        repo.add.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_found(self, service: ConversationService, repo: AsyncMock) -> None:
        expected = Conversation(id="c1")
        repo.get = AsyncMock(return_value=expected)

        conv = await service.get("c1")
        assert conv.id == "c1"
        repo.get.assert_awaited_once_with("c1")

    @pytest.mark.asyncio
    async def test_get_not_found(self, service: ConversationService, repo: AsyncMock) -> None:
        repo.get = AsyncMock(return_value=None)

        from app.core.exceptions import ResourceNotFoundError

        with pytest.raises(ResourceNotFoundError, match="not found"):
            await service.get("missing")

    @pytest.mark.asyncio
    async def test_rename(self, service: ConversationService, repo: AsyncMock) -> None:
        original = Conversation(id="c1", title="Old")
        repo.get = AsyncMock(return_value=original)
        repo.update = AsyncMock(side_effect=lambda conv: conv)

        updated = await service.rename("c1", "New Title")
        assert updated.title == "New Title"
        repo.update.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rename_missing_conversation(self, service: ConversationService, repo: AsyncMock) -> None:
        repo.get = AsyncMock(return_value=None)

        from app.core.exceptions import ResourceNotFoundError

        with pytest.raises(ResourceNotFoundError):
            await service.rename("missing", "Title")

    @pytest.mark.asyncio
    async def test_archive(self, service: ConversationService, repo: AsyncMock) -> None:
        original = Conversation(id="c1", status=ConversationStatus.ACTIVE)
        repo.get = AsyncMock(return_value=original)
        repo.update = AsyncMock(side_effect=lambda conv: conv)

        archived = await service.archive("c1")
        assert archived.status is ConversationStatus.ARCHIVED

    @pytest.mark.asyncio
    async def test_archive_missing_conversation(self, service: ConversationService, repo: AsyncMock) -> None:
        repo.get = AsyncMock(return_value=None)

        from app.core.exceptions import ResourceNotFoundError

        with pytest.raises(ResourceNotFoundError):
            await service.archive("missing")

    @pytest.mark.asyncio
    async def test_delete(self, service: ConversationService, repo: AsyncMock) -> None:
        repo.delete = AsyncMock(return_value=True)

        await service.delete("c1")
        repo.delete.assert_awaited_once_with("c1")

    @pytest.mark.asyncio
    async def test_delete_not_found(self, service: ConversationService, repo: AsyncMock) -> None:
        repo.delete = AsyncMock(return_value=False)

        from app.core.exceptions import ResourceNotFoundError

        with pytest.raises(ResourceNotFoundError, match="not found"):
            await service.delete("missing")

    @pytest.mark.asyncio
    async def test_list_active(self, service: ConversationService, repo: AsyncMock) -> None:
        expected = [Conversation(id="c1"), Conversation(id="c2")]
        repo.list_all = AsyncMock(return_value=expected)

        result = await service.list_active()
        assert len(result) == 2
        repo.list_all.assert_awaited_once_with(status="active")

    @pytest.mark.asyncio
    async def test_update_metadata(self, service: ConversationService, repo: AsyncMock) -> None:
        original = Conversation(id="c1", metadata=ConversationMetadata(system_prompt="Old"))
        repo.get = AsyncMock(return_value=original)
        repo.update = AsyncMock(side_effect=lambda conv: conv)

        updated = await service.update_metadata("c1", system_prompt="New Prompt")
        assert updated.metadata.system_prompt == "New Prompt"


# =========================================================================
# UsageService
# =========================================================================


class TestUsageService:
    @pytest.fixture
    def repo(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture
    def service(self, repo: AsyncMock) -> UsageService:
        return UsageService(repository=repo)

    @pytest.mark.asyncio
    async def test_get_for_message(self, service: UsageService, repo: AsyncMock) -> None:
        expected = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        repo.get_by_message = AsyncMock(return_value=expected)

        result = await service.get_for_message("m1")
        assert result is not None
        assert result.total_tokens == 15

    @pytest.mark.asyncio
    async def test_get_for_message_not_found(self, service: UsageService, repo: AsyncMock) -> None:
        repo.get_by_message = AsyncMock(return_value=None)

        result = await service.get_for_message("missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_record(self, service: UsageService, repo: AsyncMock) -> None:
        usage = Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8)
        repo.add_for_message = AsyncMock(return_value=usage)

        result = await service.record("m1", usage)
        assert result.total_tokens == 8
        repo.add_for_message.assert_awaited_once_with("m1", usage)


# =========================================================================
# ChatService
# =========================================================================


class TestChatService:
    @pytest.fixture
    def conversation_repo(self) -> AsyncMock:
        repo = AsyncMock()
        repo.get = AsyncMock(return_value=Conversation(id="c1"))
        repo.update = AsyncMock(side_effect=lambda conv: conv)
        return repo

    @pytest.fixture
    def message_repo(self) -> AsyncMock:
        repo = AsyncMock()
        repo.add = AsyncMock(side_effect=lambda msg: msg)
        repo.list_by_conversation = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def usage_repo(self) -> AsyncMock:
        repo = AsyncMock()
        repo.add_for_message = AsyncMock(
            return_value=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        return repo

    @pytest.fixture
    def llm_router(self) -> MagicMock:
        router = MagicMock(spec=LLMRouter)
        router.generate = AsyncMock(
            return_value=CompletionResponse(
                message=Message(
                    id="r1", conversation_id="c1", role=MessageRole.ASSISTANT,
                    content=[TextBlock(text="Hello world")],
                ),
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        )
        router.generate_stream = AsyncMock()  # not used in process_message
        return router

    @pytest.fixture
    def service(
        self,
        conversation_repo: AsyncMock,
        message_repo: AsyncMock,
        usage_repo: AsyncMock,
        llm_router: MagicMock,
    ) -> ChatService:
        return ChatService(
            conversation_repo=conversation_repo,
            message_repo=message_repo,
            usage_repo=usage_repo,
            llm_router=llm_router,
        )

    # -- process_message ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_process_message_basic(self, service: ChatService) -> None:
        result = await service.process_message(
            conversation_id="c1",
            user_content=[TextBlock(text="Hello")],
        )

        assert isinstance(result, ChatResult)
        assert result.assistant_message.role is MessageRole.ASSISTANT
        assert result.conversation.id == "c1"
        assert result.usage is not None
        assert result.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_process_message_missing_conversation(
        self,
        conversation_repo: AsyncMock,
        service: ChatService,
    ) -> None:
        conversation_repo.get = AsyncMock(return_value=None)

        from app.core.exceptions import ResourceNotFoundError

        with pytest.raises(ResourceNotFoundError, match="not found"):
            await service.process_message(
                conversation_id="missing",
                user_content=[TextBlock(text="Hi")],
            )

    @pytest.mark.asyncio
    async def test_process_message_persists_user_message(
        self,
        service: ChatService,
        message_repo: AsyncMock,
    ) -> None:
        await service.process_message(
            conversation_id="c1",
            user_content=[TextBlock(text="Hello")],
        )
        assert message_repo.add.await_count >= 1

    @pytest.mark.asyncio
    async def test_process_message_persists_assistant_message(
        self,
        service: ChatService,
        message_repo: AsyncMock,
    ) -> None:
        await service.process_message(
            conversation_id="c1",
            user_content=[TextBlock(text="Hello")],
        )
        # User message + assistant message = 2 calls to add
        assert message_repo.add.await_count == 2

    @pytest.mark.asyncio
    async def test_process_message_persists_usage(
        self,
        service: ChatService,
        usage_repo: AsyncMock,
    ) -> None:
        await service.process_message(
            conversation_id="c1",
            user_content=[TextBlock(text="Hello")],
        )
        usage_repo.add_for_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_process_message_with_context_builder(
        self,
        conversation_repo: AsyncMock,
        message_repo: AsyncMock,
        usage_repo: AsyncMock,
        llm_router: MagicMock,
    ) -> None:
        context_builder = AsyncMock()
        context_builder.build = AsyncMock(
            side_effect=lambda **kw: kw.get("history", []),
        )

        service = ChatService(
            conversation_repo=conversation_repo,
            message_repo=message_repo,
            usage_repo=usage_repo,
            llm_router=llm_router,
            context_builder=context_builder,
        )
        await service.process_message(
            conversation_id="c1",
            user_content=[TextBlock(text="Hello")],
        )
        context_builder.build.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_process_message_with_memory_service(
        self,
        conversation_repo: AsyncMock,
        message_repo: AsyncMock,
        usage_repo: AsyncMock,
        llm_router: MagicMock,
    ) -> None:
        memory_service = AsyncMock()
        memory_service.get_relevant_context = AsyncMock(return_value="prior context")
        context_builder = AsyncMock()
        context_builder.build = AsyncMock(
            side_effect=lambda **kw: kw.get("history", []),
        )

        service = ChatService(
            conversation_repo=conversation_repo,
            message_repo=message_repo,
            usage_repo=usage_repo,
            llm_router=llm_router,
            memory_service=memory_service,
            context_builder=context_builder,
        )
        await service.process_message(
            conversation_id="c1",
            user_content=[TextBlock(text="Hello")],
        )
        memory_service.get_relevant_context.assert_awaited_once_with(
            conversation_id="c1",
            user_content=[TextBlock(text="Hello")],
        )
        context_builder.build.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_process_message_updates_conversation_count(
        self,
        service: ChatService,
        conversation_repo: AsyncMock,
    ) -> None:
        conv = Conversation(id="c1")
        conversation_repo.get = AsyncMock(return_value=conv)
        conversation_repo.update = AsyncMock(side_effect=lambda conv: conv)

        await service.process_message(
            conversation_id="c1",
            user_content=[TextBlock(text="Hi")],
        )
        assert conv.message_count == 1

    # -- process_message_stream --------------------------------------------

    @pytest.mark.asyncio
    async def test_process_message_stream_basic(
        self,
        conversation_repo: AsyncMock,
        message_repo: AsyncMock,
        usage_repo: AsyncMock,
        llm_router: MagicMock,
    ) -> None:
        async def _gen(*args: object, **kw: object) -> object:
            from app.domain.stream import TextDeltaEvent

            yield TextDeltaEvent(delta="Hello")
            yield TextDeltaEvent(delta=" world")

        llm_router.generate_stream = _gen

        service = ChatService(
            conversation_repo=conversation_repo,
            message_repo=message_repo,
            usage_repo=usage_repo,
            llm_router=llm_router,
        )

        result = await service.process_message_stream(
            conversation_id="c1",
            user_content=[TextBlock(text="Hi")],
        )
        assert isinstance(result, ChatResult)
        assert result.assistant_message.role is MessageRole.ASSISTANT

    @pytest.mark.asyncio
    async def test_process_message_stream_missing_conversation(
        self,
        conversation_repo: AsyncMock,
        service: ChatService,
    ) -> None:
        conversation_repo.get = AsyncMock(return_value=None)

        from app.core.exceptions import ResourceNotFoundError

        with pytest.raises(ResourceNotFoundError):
            await service.process_message_stream(
                conversation_id="missing",
                user_content=[TextBlock(text="Hi")],
            )

    # -- stream_message ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_stream_message_yields_events(
        self,
        conversation_repo: AsyncMock,
        message_repo: AsyncMock,
        usage_repo: AsyncMock,
        llm_router: MagicMock,
    ) -> None:
        from app.domain.stream import StreamEvent, StreamStartEvent, TextDeltaEvent

        async def _gen_stream(*args: object, **kw: object) -> object:
            yield TextDeltaEvent(delta="Hello")

        llm_router.generate_stream = _gen_stream
        llm_router.generate = AsyncMock(
            return_value=CompletionResponse(
                message=Message(
                    id="r1", conversation_id="c1", role=MessageRole.ASSISTANT,
                    content=[TextBlock(text="no")],
                ),
            ),
        )

        service = ChatService(
            conversation_repo=conversation_repo,
            message_repo=message_repo,
            usage_repo=usage_repo,
            llm_router=llm_router,
        )

        events: list[StreamEvent] = []
        async for event in service.stream_message(
            conversation_id="c1",
            user_content=[TextBlock(text="Hi")],
        ):
            events.append(event)

        assert len(events) > 0
        assert isinstance(events[0], StreamStartEvent)

    @pytest.mark.asyncio
    async def test_stream_message_missing_conversation(
        self,
        conversation_repo: AsyncMock,
        service: ChatService,
    ) -> None:
        conversation_repo.get = AsyncMock(return_value=None)

        from app.core.exceptions import ResourceNotFoundError

        with pytest.raises(ResourceNotFoundError):
            async for _ in service.stream_message(
                conversation_id="missing",
                user_content=[TextBlock(text="Hi")],
            ):
                pass
