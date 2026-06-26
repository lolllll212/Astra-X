"""Attachment repository."""

from __future__ import annotations

from app.database.converters.attachment_mapper import (
    attachment_from_model,
    attachment_to_model,
)
from app.database.models.attachment import AttachmentModel
from app.database.repositories.base import BaseRepository
from app.domain.attachment import Attachment


class AttachmentRepository(BaseRepository[Attachment, AttachmentModel]):
    """Repository for :class:`Attachment` entities."""

    @property
    def _model_cls(self) -> type[AttachmentModel]:
        return AttachmentModel

    def _to_domain(self, model: AttachmentModel) -> Attachment:
        return attachment_from_model(model)

    def _to_model(self, domain: Attachment) -> AttachmentModel:
        return attachment_to_model(domain)
