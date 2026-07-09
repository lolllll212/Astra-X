"""Protocol adapter abstract base class.

Each protocol (Chat Completions, Responses API, …) implements this
interface so that provider adapters can delegate request building and
response parsing without knowing the wire format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.domain.enums import OpenAIProtocol
from app.llm.models import CompletionRequest, CompletionResponse
from app.domain.stream import StreamEvent


class ProtocolAdapter(ABC):
    """Stateless adapter that knows how to speak one API protocol.

    Subclasses implement the four core operations:

    * :meth:`build_request` — domain → wire format
    * :meth:`parse_response` — wire format → domain (non-streaming)
    * :meth:`parse_stream_chunk` — wire chunk → domain events
    * :meth:`parse_stream_chunk_done` — wire chunk end-of-stream detection
    """

    @property
    @abstractmethod
    def protocol(self) -> OpenAIProtocol:
        """Which protocol this adapter implements."""
        ...

    @property
    @abstractmethod
    def endpoint(self) -> str:
        """URL path (e.g. ``/chat/completions`` or ``/responses``)."""
        ...

    @abstractmethod
    def build_request(
        self,
        request: CompletionRequest,
        model: str,
    ) -> dict[str, Any]:
        """Convert a domain :class:`CompletionRequest` into the provider's wire format.

        Args:
            request: The domain-level completion request.
            model: The model identifier to use.

        Returns:
            A JSON-serialisable dictionary ready to POST.
        """
        ...

    @abstractmethod
    def parse_response(
        self,
        raw: dict[str, Any],
        model: str,
    ) -> CompletionResponse:
        """Convert a non-streaming provider response into a domain response.

        Args:
            raw: The JSON-decoded provider response body.
            model: The model that served the response.

        Returns:
            A domain :class:`CompletionResponse`.
        """
        ...

    @abstractmethod
    def parse_stream_chunk(
        self,
        chunk: dict[str, Any],
        tool_call_state: dict[int, dict[str, str]] | None = None,
    ) -> tuple[list[StreamEvent], dict[int, dict[str, str]]]:
        """Convert a single stream chunk into domain stream events.

        This method is purely functional — it takes the current tool call
        tracking state and returns updated state alongside the events.

        Args:
            chunk: A JSON-decoded chunk from the SSE stream.
            tool_call_state: Current tool call tracking state keyed by
                tool-call index. Pass ``None`` (or omitt) to start fresh.

        Returns:
            A ``(events, tool_call_state)`` tuple.
        """
        ...

    @abstractmethod
    def parse_stream_chunk_done(
        self,
        chunk: dict[str, Any],
    ) -> bool:
        """Return ``True`` when a chunk signals the end of the stream.

        Args:
            chunk: A JSON-decoded chunk from the SSE stream.

        Returns:
            ``True`` if the stream should stop, ``False`` otherwise.
        """
        ...
