# ruff: noqa: B008 — FastAPI Depends() in default args is intentional

"""Attachment management endpoints.

Provides CRUD operations for file attachments linked to conversations.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_attachment_service
from app.api.schemas.attachment import (
    AttachmentCreate,
    AttachmentListResponse,
    AttachmentResponse,
)
from app.api.schemas.common import MessageResponse
from app.domain.attachment import Attachment
from app.services.attachment_service import AttachmentService

router = APIRouter(prefix="/attachments", tags=["attachments"])


def _attachment_to_response(attachment: Attachment) -> AttachmentResponse:
    """Convert a domain ``Attachment`` to its API response schema.

    Args:
        attachment: The domain attachment.

    Returns:
        A serialisable response schema.
    """
    return AttachmentResponse(
        id=attachment.id,
        conversation_id=attachment.conversation_id,
        file_name=attachment.file_name,
        mime_type=attachment.mime_type,
        size_bytes=attachment.size_bytes,
        attachment_type=attachment.attachment_type,
        created_at=attachment.created_at,
    )


@router.get(
    "",
    response_model=AttachmentListResponse,
    summary="List attachments for a conversation",
)
async def list_attachments(
    conversation_id: str,
    attachment_service: AttachmentService = Depends(get_attachment_service),
) -> AttachmentListResponse:
    """Return all attachments belonging to the given conversation."""
    attachments = await attachment_service.list_by_conversation(conversation_id)
    items = [_attachment_to_response(a) for a in attachments]
    return AttachmentListResponse(attachments=items, total=len(items))


@router.post(
    "",
    response_model=AttachmentResponse,
    status_code=201,
    summary="Create an attachment record",
)
async def create_attachment(
    body: AttachmentCreate,
    attachment_service: AttachmentService = Depends(get_attachment_service),
) -> AttachmentResponse:
    """Create a new attachment record for a conversation."""
    attachment = await attachment_service.create(
        conversation_id=body.conversation_id,
        file_name=body.file_name,
        storage_path="",  # Set by upload handler when file upload is implemented.
        attachment_type=body.attachment_type,
        mime_type=body.mime_type,
        size_bytes=body.size_bytes,
    )
    return _attachment_to_response(attachment)


@router.get(
    "/{attachment_id}",
    response_model=AttachmentResponse,
    summary="Get an attachment by ID",
)
async def get_attachment(
    attachment_id: str,
    attachment_service: AttachmentService = Depends(get_attachment_service),
) -> AttachmentResponse:
    """Retrieve a single attachment with full metadata."""
    attachment = await attachment_service.get(attachment_id)
    return _attachment_to_response(attachment)


@router.delete(
    "/{attachment_id}",
    response_model=MessageResponse,
    summary="Delete an attachment",
)
async def delete_attachment(
    attachment_id: str,
    attachment_service: AttachmentService = Depends(get_attachment_service),
) -> MessageResponse:
    """Permanently delete an attachment record."""
    await attachment_service.delete(attachment_id)
    return MessageResponse(message="Attachment deleted.")
