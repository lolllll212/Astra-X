# ruff: noqa: B008 — FastAPI Depends() in default args is intentional

"""Application metrics endpoint.

Returns Prometheus exposition format by default, or a JSON snapshot
when the ``Accept`` header contains ``application/json``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

from app.api.dependencies import get_conversation_service, get_message_repository
from app.api.errors import get_uptime
from app.api.schemas.system import MetricsResponse
from app.database.repositories.message_repository import MessageRepository
from app.services.conversation_service import ConversationService

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get(
    "",
    summary="Application metrics (auto-detects format)",
)
async def get_metrics(
    request: Request,
    conversation_service: ConversationService = Depends(get_conversation_service),
    message_repository: MessageRepository = Depends(get_message_repository),
) -> Response:
    """Return metrics in the requested format.

    Returns a JSON snapshot when ``Accept: application/json`` is
    present; otherwise returns Prometheus text exposition format.
    """
    accept = request.headers.get("accept", "")

    if "application/json" in accept:
        active_conversations = len(await conversation_service.list_active())
        total_messages = await message_repository.count()

        return JSONResponse(
            MetricsResponse(
                uptime_seconds=get_uptime(),
                active_conversations=active_conversations,
                total_messages=total_messages,
            ).model_dump(),
        )

    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@router.get(
    "/json",
    response_model=MetricsResponse,
    summary="Application metrics JSON snapshot",
)
async def get_metrics_json(
    conversation_service: ConversationService = Depends(get_conversation_service),
    message_repository: MessageRepository = Depends(get_message_repository),
) -> MetricsResponse:
    """Return a JSON snapshot of key application metrics.

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
