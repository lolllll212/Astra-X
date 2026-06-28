"""Domain enumerations for Astra X.

Every enumeration used by domain models, services, repositories, and API
schemas is defined here. Centralising enums in a single module prevents
circular imports and keeps a single source of truth for domain constants.
"""

from __future__ import annotations

from enum import StrEnum


class ProviderType(StrEnum):
    """Supported LLM provider backends.

    Each member corresponds to a concrete provider adapter in
    ``app.providers``.
    """

    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    OPENAI_COMPATIBLE = "openai_compatible"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"


class ModelCapability(StrEnum):
    """Capabilities a model may advertise.

    Used at runtime to select a suitable model for a given request and
    to validate that a provider can satisfy a requested feature.
    """

    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    VISION = "vision"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"
    JSON_MODE = "json_mode"


class MessageRole(StrEnum):
    """Role of a message participant in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentBlockType(StrEnum):
    """Kind of content within a single message part.

    A single message may contain multiple content blocks of different
    types (e.g. text + image).
    """

    TEXT = "text"
    IMAGE = "image"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class ConversationStatus(StrEnum):
    """Lifecycle states for a conversation."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class ToolType(StrEnum):
    """Categories of executable tools."""

    FUNCTION = "function"
    CODE_INTERPRETER = "code_interpreter"
    RETRIEVAL = "retrieval"
    WEB_SEARCH = "web_search"


class StreamEventType(StrEnum):
    """Server-sent event types emitted during streaming responses."""

    START = "start"
    THINKING = "thinking"
    TEXT_DELTA = "text_delta"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"
    TOOL_PROGRESS = "tool_progress"
    TOOL_RESULT = "tool_result"
    CITATION = "citation"
    USAGE = "usage"
    ERROR = "error"
    DONE = "done"
    METADATA = "metadata"


class AttachmentType(StrEnum):
    """Types of file attachments supported by the platform."""

    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"


class MemoryScope(StrEnum):
    """Scope at which a memory entry is stored and retrieved."""

    CONVERSATION = "conversation"
    USER = "user"
    SESSION = "session"
    GLOBAL = "global"


class AgentStatus(StrEnum):
    """Lifecycle states for an autonomous agent."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"


class ReasoningEffort(StrEnum):
    """Level of reasoning effort an agent should apply to a task."""

    NONE = "none"
    SHALLOW = "shallow"
    MEDIUM = "medium"
    DEEP = "deep"
