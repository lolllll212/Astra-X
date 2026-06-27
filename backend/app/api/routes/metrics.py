# ruff: noqa: B008 — FastAPI Depends() in default args is intentional

"""Application metrics endpoint.

Provides a snapshot of key application metrics for monitoring and
observability.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_conversation_service, get_message_repository
from app.api.errors import get_uptime
from app.api.schemas.system import MetricsResponse
from app.database.repositories.message_repository import MessageRepository
from app.services.conversation_service import ConversationService

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get(
    "",
    response_model=MetricsResponse,
    summary="Application metrics snapshot",
)
async def get_metrics(
    conversation_service: ConversationService = Depends(get_conversation_service),
    message_repository: MessageRepository = Depends(get_message_repository),
) -> MetricsResponse:
    """Return a snapshot of key application metrics.

    Includes uptime, active conversation count, and total messages
    across all conversations.
    """
    active_conversations = len(await conversation_service.list_active())
    total_messages = await message_repository.count()

    return MetricsResponse(
        uptime_seconds=get_uptime(),
        active_conversations=active_conversations,
        total_messages=total_messages,
    )
