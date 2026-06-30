# ruff: noqa: B008 — FastAPI Depends() in default args is intentional

"""Application metrics endpoint.

Provides a JSON snapshot endpoint and a Prometheus scrape endpoint
at ``GET /metrics``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import PlainTextResponse, Response

from app.api.dependencies import get_conversation_service, get_message_repository
from app.api.errors import get_uptime
from app.api.schemas.system import MetricsResponse
from app.database.repositories.message_repository import MessageRepository
from app.services.conversation_service import ConversationService

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get(
    "",
    response_class=PlainTextResponse,
    summary="Prometheus metrics scrape endpoint",
    include_in_schema=False,
)
async def get_prometheus_metrics() -> Response:
    """Return metrics in Prometheus text exposition format.

    This is the endpoint that Prometheus scrapes.  The JSON snapshot
    endpoint is available at ``GET /api/v1/metrics/json``.
    """
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

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
