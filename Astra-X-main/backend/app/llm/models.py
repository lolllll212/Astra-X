"""LLM request/response models.

Defines the input and output contracts for provider adapters. Every
adapter receives a :class:`CompletionRequest` and returns a
:class:`CompletionResponse` (or yields :class:`StreamEvent` for
streaming).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.message import Message
from app.domain.usage import Usage


class FinishReason:
    """Canonical finish reasons for a generation."""

    STOP: str = "stop"
    LENGTH: str = "length"
    TOOL_CALLS: str = "tool_calls"
    CONTENT_FILTER: str = "content_filter"
    ERROR: str = "error"
    NULL: str = "null"


@dataclass(frozen=True)
class GenerationParams:
    """Parameters controlling text generation behaviour.

    Every field has a sensible default so callers only need to override
    values they care about.
    """

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int | None = None
    stop: list[str] = field(default_factory=list)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None


@dataclass(frozen=True)
class CompletionRequest:
    """A request to generate a completion from an LLM.

    Attributes:
        messages: The conversation messages so far.
        model: The model identifier to use.
        provider: The provider identifier to use.
        params: Generation parameters.
        stream: Whether to stream the response.
    """

    messages: list[Message]
    model: str
    provider: str | None = None
    params: GenerationParams = field(default_factory=GenerationParams)
    stream: bool = False
    tools: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class CompletionResponse:
    """The result of a non-streaming LLM completion.

    Attributes:
        message: The assistant's response message.
        usage: Token usage for the generation.
        finish_reason: Why generation stopped.
        model: The model that served the response.
    """

    message: Message
    usage: Usage | None = None
    finish_reason: str = FinishReason.STOP
    model: str = ""
