"""Domain model aggregator.

Re-exports every public symbol from the domain sub-modules so that
downstream code may import from a single location::

    from app.domain.models import Message, Conversation, Usage

This is purely a convenience; individual modules may also be imported
directly without circular-reference risk since nothing in ``domain``
imports from this aggregator.
"""

from __future__ import annotations

from app.domain.attachment import Attachment
from app.domain.conversation import Conversation, ConversationMetadata, ConversationParticipant
from app.domain.enums import (
    AgentStatus,
    AttachmentType,
    ContentBlockType,
    ConversationStatus,
    MemoryScope,
    MessageRole,
    ModelCapability,
    ProviderType,
    ReasoningEffort,
    StreamEventType,
    ToolType,
)
from app.domain.message import (
    ContentBlock,
    ImageBlock,
    Message,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from app.domain.provider import ModelID, ModelSpec, ProviderAuth, ProviderID, ProviderSpec
from app.domain.stream import (
    StreamDoneEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamMetadataEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from app.domain.tool import ToolCall, ToolParameter, ToolResult, ToolSpec
from app.domain.usage import Usage

__all__ = [
    "AgentStatus",
    "Attachment",
    "AttachmentType",
    "ContentBlock",
    "ContentBlockType",
    "Conversation",
    "ConversationMetadata",
    "ConversationParticipant",
    "ConversationStatus",
    "ImageBlock",
    "MemoryScope",
    "Message",
    "MessageRole",
    "ModelCapability",
    "ModelID",
    "ModelSpec",
    "ProviderAuth",
    "ProviderID",
    "ProviderSpec",
    "ProviderType",
    "ReasoningEffort",
    "StreamDoneEvent",
    "StreamErrorEvent",
    "StreamEvent",
    "StreamEventType",
    "StreamMetadataEvent",
    "TextBlock",
    "TextDeltaEvent",
    "ToolCall",
    "ToolCallBlock",
    "ToolCallDeltaEvent",
    "ToolCallEndEvent",
    "ToolCallStartEvent",
    "ToolParameter",
    "ToolResult",
    "ToolResultBlock",
    "ToolSpec",
    "ToolType",
    "Usage",
]
