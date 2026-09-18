"""Message domain-to-ORM mapper."""

from __future__ import annotations

from typing import Any

from app.database.models.message import MessageModel
from app.domain.enums import ContentBlockType, MessageRole
from app.domain.message import (
    ContentBlock,
    ImageBlock,
    Message,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)

__all__ = [
    "message_from_model",
    "message_to_model",
]

_BLOCK_TYPE_MAP: dict[ContentBlockType, type[ContentBlock]] = {
    ContentBlockType.TEXT: TextBlock,
    ContentBlockType.IMAGE: ImageBlock,
    ContentBlockType.TOOL_CALL: ToolCallBlock,
    ContentBlockType.TOOL_RESULT: ToolResultBlock,
}


def _block_to_dict(block: ContentBlock) -> dict[str, Any]:
    return block.model_dump(mode="json", by_alias=True)


def _dict_to_block(data: dict[str, Any]) -> ContentBlock:
    block_type_str = data.get("type", "text")
    try:
        block_type = ContentBlockType(block_type_str)
    except ValueError:
        block_type = ContentBlockType.TEXT
    block_cls = _BLOCK_TYPE_MAP.get(block_type, TextBlock)
    return block_cls.model_validate(data)


def message_to_model(domain: Message) -> MessageModel:
    content_data: list[dict[str, Any]] = [_block_to_dict(b) for b in domain.content]
    metadata_data: dict[str, Any] | None = domain.metadata or None

    return MessageModel(
        id=domain.id,
        conversation_id=domain.conversation_id,
        role=domain.role.value,
        content=content_data,
        created_at=domain.created_at,
        parent_id=domain.parent_id,
        metadata_=metadata_data,
    )


def message_from_model(model: MessageModel) -> Message:
    content: list[ContentBlock] = [_dict_to_block(b) for b in (model.content or [])]
    metadata: dict[str, Any] = model.metadata_ or {}

    return Message(
        id=model.id,
        conversation_id=model.conversation_id,
        role=MessageRole(model.role),
        content=content,
        created_at=model.created_at,
        parent_id=model.parent_id,
        metadata=metadata,
    )
