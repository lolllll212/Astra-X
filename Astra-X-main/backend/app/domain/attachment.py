"""File attachment domain models.

Defines the metadata model for file attachments uploaded to the platform.
Attachments are referenced by messages and may be processed by tools or
included as content blocks in provider requests.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from app.domain.enums import AttachmentType

__all__ = [
    "Attachment",
]


_MIME_TYPE_MAP: dict[AttachmentType, set[str]] = {
    AttachmentType.IMAGE: {"image/png", "image/jpeg", "image/gif", "image/webp"},
    AttachmentType.DOCUMENT: {
        "text/plain",
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    },
    AttachmentType.AUDIO: {"audio/mpeg", "audio/wav", "audio/ogg"},
    AttachmentType.VIDEO: {"video/mp4", "video/webm"},
    AttachmentType.FILE: set(),
}


class Attachment(BaseModel):
    """Metadata for a file attachment.

    The binary content is stored by a storage backend (local filesystem,
    S3-compatible object store, etc.) and referenced by path. This model
    carries only the metadata required to reference, display, and validate
    the attachment.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(
        description="Unique attachment identifier (UUID).",
    )
    conversation_id: str = Field(
        description="Conversation this attachment belongs to.",
    )
    file_name: str = Field(
        min_length=1,
        description="Original file name as uploaded by the user.",
    )
    mime_type: str | None = Field(
        default=None,
        description="MIME type of the file, if detected.",
    )
    size_bytes: int = Field(
        default=0,
        ge=0,
        description="File size in bytes.",
    )
    attachment_type: AttachmentType = Field(
        description="Categorisation of the attachment (image, document, etc.).",
    )
    storage_path: str = Field(
        min_length=1,
        description="Path or key within the storage backend where the file is stored.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the attachment was created.",
    )

    @field_validator("mime_type")
    @classmethod
    def _validate_mime_type(
        cls,
        value: str | None,
        info: ValidationInfo,
    ) -> str | None:
        if value is None:
            return None
        attachment_type: AttachmentType | None = info.data.get("attachment_type")
        if attachment_type is None:
            return value
        allowed = _MIME_TYPE_MAP.get(attachment_type)
        if allowed and value not in allowed:
            allowed_str = ", ".join(sorted(allowed))
            msg = f"MIME type '{value}' is not valid for attachment type '{attachment_type.value}'. Allowed: {allowed_str}"
            raise ValueError(msg)
        return value
