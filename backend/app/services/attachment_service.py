"""Attachment management service.

Handles CRUD operations and storage backends for file attachments.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.core.exceptions import ResourceNotFoundError
from app.core.logging import get_logger
from app.database.repositories.attachment_repository import AttachmentRepository
from app.domain.attachment import Attachment
from app.domain.enums import AttachmentType

logger = get_logger(__name__)


class AttachmentService:
    """Manages file attachments."""

    def __init__(self, repository: AttachmentRepository) -> None:
        self._repo = repository

    async def create(
        self,
        conversation_id: str,
        file_name: str,
        storage_path: str,
        attachment_type: AttachmentType,
        mime_type: str | None = None,
        size_bytes: int = 0,
    ) -> Attachment:
        """Create a new attachment record.

        Args:
            conversation_id: The conversation this belongs to.
            file_name: Original uploaded file name.
            storage_path: Path within the storage backend.
            attachment_type: Category of attachment.
            mime_type: Detected MIME type.
            size_bytes: File size in bytes.

        Returns:
            The created attachment.
        """
        attachment = Attachment(
            id=str(uuid4()),
            conversation_id=conversation_id,
            file_name=file_name,
            mime_type=mime_type,
            size_bytes=size_bytes,
            attachment_type=attachment_type,
            storage_path=storage_path,
            created_at=datetime.now(UTC),
        )
        result = await self._repo.add(attachment)
        logger.info(
            "attachment.created",
            attachment_id=result.id,
            conversation_id=conversation_id,
        )
        return result

    async def get(self, attachment_id: str) -> Attachment:
        """Retrieve an attachment by ID.

        Args:
            attachment_id: The attachment identifier.
        """
        attachment = await self._repo.get(attachment_id)
        if attachment is None:
            raise ResourceNotFoundError(
                message=f"Attachment '{attachment_id}' not found.",
            )
        return attachment

    async def list_by_conversation(self, conversation_id: str) -> list[Attachment]:
        """List all attachments for a conversation.

        Args:
            conversation_id: The conversation identifier.
        """
        return await self._repo.list(conversation_id=conversation_id)

    async def delete(self, attachment_id: str) -> None:
        """Delete an attachment.

        Args:
            attachment_id: The attachment identifier.
        """
        deleted = await self._repo.delete(attachment_id)
        if not deleted:
            raise ResourceNotFoundError(
                message=f"Attachment '{attachment_id}' not found.",
            )
        logger.info("attachment.deleted", attachment_id=attachment_id)
