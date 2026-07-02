# ruff: noqa: B008 — FastAPI Depends() in default args is intentional

"""FastAPI dependency injection helpers.

Provides ``Depends()`` callables that FastAPI route handlers use to
obtain services, repositories, and configuration. Everything is resolved
through ``request.app.state`` — the composition root set up during the
application lifespan.

Routes MUST NOT instantiate services or repositories directly; every
dependency flows through this module.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import Settings
from app.database.repositories.attachment_repository import AttachmentRepository
from app.database.repositories.conversation_repository import ConversationRepository
from app.database.repositories.message_repository import MessageRepository
from app.database.repositories.plugin_repository import PluginRepository
from app.database.repositories.provider_repository import ProviderRepository
from app.database.repositories.usage_repository import UsageRepository
from app.database.session import create_session_factory, session_context
from app.llm.router import LLMRouter
from app.services.attachment_service import AttachmentService
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService
from app.services.memory_service import MemoryService
from app.services.plugin_service import PluginService
from app.services.provider_service import ProviderService
from app.services.usage_service import UsageService


async def get_settings(request: Request) -> Settings:
    """Return the application settings singleton from app state.

    Set during startup by :func:`app.core.lifecycle.lifespan`.
    """
    settings: Settings = request.app.state.settings
    return settings


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield a request-scoped database session.

    The session is committed on success, rolled back on exception, and
    closed when the caller exits the context.

    Yields:
        An ``AsyncSession`` bound to the current engine.
    """
    engine = request.app.state.db_engine
    factory = create_session_factory(engine)
    async with session_context(factory) as session:
        yield session


async def get_conversation_repository(
    session: AsyncSession = Depends(get_db_session),
) -> ConversationRepository:
    """Provide a conversation repository for the current request."""
    return ConversationRepository(session)


async def get_message_repository(
    session: AsyncSession = Depends(get_db_session),
) -> MessageRepository:
    """Provide a message repository for the current request."""
    return MessageRepository(session)


async def get_usage_repository(
    session: AsyncSession = Depends(get_db_session),
) -> UsageRepository:
    """Provide a usage repository for the current request."""
    return UsageRepository(session)


async def get_attachment_repository(
    session: AsyncSession = Depends(get_db_session),
) -> AttachmentRepository:
    """Provide an attachment repository for the current request."""
    return AttachmentRepository(session)


async def get_attachment_service(
    repository: AttachmentRepository = Depends(get_attachment_repository),
) -> AttachmentService:
    """Provide an attachment management service."""
    return AttachmentService(repository)


async def get_llm_router(request: Request) -> LLMRouter:
    """Return the LLM router singleton.

    The router is created during startup and stored on
    ``app.state.llm_router``. If not yet set (e.g. during a gradual
    rollout), a default router is constructed lazily.

    Returns:
        The application's LLM router instance.
    """
    router: object = getattr(request.app.state, "llm_router", None)
    if isinstance(router, LLMRouter):
        return router

    settings: Settings = request.app.state.settings
    new_router = LLMRouter(
        settings=settings,
        default_provider_id=settings.default_llm_provider,
    )
    request.app.state.llm_router = new_router
    return new_router


async def get_conversation_service(
    repository: ConversationRepository = Depends(get_conversation_repository),
) -> ConversationService:
    """Provide a conversation management service."""
    return ConversationService(repository)


async def get_core_memory_manager(request: Request) -> object | None:
    """Return the core memory manager from app state, if initialised."""
    return getattr(request.app.state, "memory_manager", None)


async def get_memory_service(
    message_repository: MessageRepository = Depends(get_message_repository),
    core_memory_manager: object | None = Depends(get_core_memory_manager),
) -> MemoryService:
    """Provide a memory service bound to the current session.

    If the core memory manager has been initialised at startup, it is
    injected into the service for semantic retrieval and storage.
    """
    return MemoryService(
        message_repository,
        core_memory_manager=core_memory_manager,  # type: ignore[arg-type]
    )


async def get_provider_repository(
    session: AsyncSession = Depends(get_db_session),
) -> ProviderRepository:
    """Provide a provider repository for the current request."""
    return ProviderRepository(session)


async def get_provider_service(
    repository: ProviderRepository = Depends(get_provider_repository),
    llm_router: LLMRouter = Depends(get_llm_router),
    settings: Settings = Depends(get_settings),
) -> ProviderService:
    """Provide a provider management service.

    The service is wired with the database repository and the LLM router so
    that database changes are automatically reflected in the in-memory
    provider adapter registry.
    """
    return ProviderService(
        repository=repository,
        llm_router=llm_router,
        settings=settings,
    )


async def get_usage_service(
    repository: UsageRepository = Depends(get_usage_repository),
) -> UsageService:
    """Provide a usage tracking service."""
    return UsageService(repository)


async def get_plugin_repository(
    session: AsyncSession = Depends(get_db_session),
) -> PluginRepository:
    """Provide a plugin repository for the current request."""
    return PluginRepository(session)


async def get_plugin_service(
    repository: PluginRepository = Depends(get_plugin_repository),
) -> PluginService:
    """Provide a plugin management service."""
    return PluginService(repository=repository)


async def get_chat_service(
    conversation_repo: ConversationRepository = Depends(get_conversation_repository),
    message_repo: MessageRepository = Depends(get_message_repository),
    usage_repo: UsageRepository = Depends(get_usage_repository),
    llm_router: LLMRouter = Depends(get_llm_router),
    memory_service: MemoryService = Depends(get_memory_service),
) -> ChatService:
    """Provide a chat orchestration service.

    The service is constructed per-request with request-scoped
    repositories so that a failure in one request does not affect
    others.

    Returns:
        A fully-wired ``ChatService`` instance.
    """
    return ChatService(
        conversation_repo=conversation_repo,
        message_repo=message_repo,
        usage_repo=usage_repo,
        llm_router=llm_router,
        memory_service=memory_service,
    )
