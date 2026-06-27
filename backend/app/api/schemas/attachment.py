"""Request and response schemas for the attachment API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import AttachmentType


class AttachmentCreate(BaseModel):
    """Request body for creating a new attachment record.

    Attributes:
        conversation_id: The conversation this attachment belongs to.
        file_name: Original uploaded file name.
        attachment_type: Category of attachment.
        mime_type: Detected MIME type.
        size_bytes: File size in bytes.
    """

    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(min_length=1, description="Conversation identifier.")
    file_name: str = Field(min_length=1, description="Original file name.")
    attachment_type: AttachmentType = Field(description="Attachment category.")
    mime_type: str | None = Field(default=None, description="MIME type.")
    size_bytes: int = Field(default=0, ge=0, description="File size in bytes.")


class AttachmentResponse(BaseModel):
    """Attachment representation returned by the API.

    Attributes:
        id: Unique attachment identifier.
        conversation_id: The conversation this belongs to.
        file_name: Original uploaded file name.
        mime_type: Detected MIME type.
        size_bytes: File size in bytes.
        attachment_type: Category of attachment.
        created_at: When the attachment was created.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Attachment identifier.")
    conversation_id: str = Field(description="Conversation identifier.")
    file_name: str = Field(description="Original file name.")
    mime_type: str | None = Field(default=None, description="MIME type.")
    size_bytes: int = Field(ge=0, description="File size in bytes.")
    attachment_type: AttachmentType = Field(description="Attachment category.")
    created_at: datetime = Field(description="Creation timestamp.")


class AttachmentListResponse(BaseModel):
    """Wrapper for a list of attachments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attachments: list[AttachmentResponse] = Field(description="Attachment list.")
    total: int = Field(ge=0, description="Total number of attachments.")
