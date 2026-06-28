# ruff: noqa: B008 — FastAPI Depends() in default args is intentional

"""Conversation CRUD endpoints.

Provides standard resource lifecycle operations for conversations —
create, read, update (rename/archive), delete, and list active.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_conversation_service
from app.api.schemas.common import MessageResponse
from app.api.schemas.conversation import (
    ConversationCreate,
    ConversationListResponse,
    ConversationResponse,
    ConversationUpdate,
)
from app.domain.conversation import Conversation
from app.services.conversation_service import ConversationService

router = APIRouter(prefix="/conversations", tags=["conversations"])


def _conversation_to_response(conv: Conversation) -> ConversationResponse:
    """Convert a domain ``Conversation`` to its API response schema.

    Args:
        conv: The domain conversation.

    Returns:
        A serialisable response schema.
    """
    return ConversationResponse(
        id=conv.id,
        title=conv.title,
        status=conv.status,
        participants=[p.model_dump() for p in conv.participants],
        metadata=conv.metadata.model_dump(),
        message_count=conv.message_count,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


@router.get(
    "",
    response_model=ConversationListResponse,
    summary="List active conversations",
)
async def list_conversations(
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationListResponse:
    """Return all active conversations, ordered by most recently updated."""
    conversations = await conversation_service.list_active()
    items = [_conversation_to_response(c) for c in conversations]
    return ConversationListResponse(
        items=items,
        total=len(items),
        page=1,
        page_size=max(len(items), 1) if items else 1,
        pages=1,
    )


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation",
)
async def create_conversation(
    body: ConversationCreate,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationResponse:
    """Create a new empty conversation with optional defaults."""
    conv = await conversation_service.create(
        title=body.title,
        system_prompt=body.system_prompt,
        model=body.model,
        provider=body.provider,
    )
    return _conversation_to_response(conv)


@router.get(
    "/{conversation_id}",
    response_model=ConversationResponse,
    summary="Get a conversation by ID",
)
async def get_conversation(
    conversation_id: str,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationResponse:
    """Retrieve a single conversation with full metadata."""
    conv = await conversation_service.get(conversation_id)
    return _conversation_to_response(conv)


@router.patch(
    "/{conversation_id}",
    response_model=ConversationResponse,
    summary="Update a conversation",
)
async def update_conversation(
    conversation_id: str,
    body: ConversationUpdate,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationResponse:
    """Update one or more fields on a conversation.

    Accepted fields: ``title``, ``status``, ``system_prompt``,
    ``model``, ``provider``.
    """
    if body.title is not None:
        await conversation_service.rename(conversation_id, body.title)

    if body.status is not None and body.status.value == "archived":
        await conversation_service.archive(conversation_id)

    metadata_fields: dict[str, str | None] = {}
    if body.system_prompt is not None:
        metadata_fields["system_prompt"] = body.system_prompt
    if body.model is not None:
        metadata_fields["model"] = body.model
    if body.provider is not None:
        metadata_fields["provider"] = body.provider

    if metadata_fields:
        await conversation_service.update_metadata(conversation_id, **metadata_fields)

    conv = await conversation_service.get(conversation_id)
    return _conversation_to_response(conv)


@router.delete(
    "/{conversation_id}",
    response_model=MessageResponse,
    summary="Delete a conversation",
)
async def delete_conversation(
    conversation_id: str,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> MessageResponse:
    """Permanently delete a conversation and its messages."""
    await conversation_service.delete(conversation_id)
    return MessageResponse(message="Conversation deleted.")
