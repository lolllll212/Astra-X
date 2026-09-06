"""Attachment domain-to-ORM mapper."""

from __future__ import annotations

from app.database.models.attachment import AttachmentModel
from app.domain.attachment import Attachment
from app.domain.enums import AttachmentType

__all__ = [
    "attachment_from_model",
    "attachment_to_model",
]


def attachment_to_model(domain: Attachment) -> AttachmentModel:
    return AttachmentModel(
        id=domain.id,
        conversation_id=domain.conversation_id,
        file_name=domain.file_name,
        mime_type=domain.mime_type,
        size_bytes=domain.size_bytes,
        attachment_type=domain.attachment_type.value,
        storage_path=domain.storage_path,
        created_at=domain.created_at,
    )


def attachment_from_model(model: AttachmentModel) -> Attachment:
    return Attachment(
        id=model.id,
        conversation_id=model.conversation_id,
        file_name=model.file_name,
        mime_type=model.mime_type,
        size_bytes=model.size_bytes,
        attachment_type=AttachmentType(model.attachment_type),
        storage_path=model.storage_path,
        created_at=model.created_at,
    )
