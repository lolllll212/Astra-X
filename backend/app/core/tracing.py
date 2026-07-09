"""Lightweight hierarchical tracing for observability.

Provides a zero-dependency :class:`Tracer` that records nested spans
with sub-millisecond precision and renders them as a human-readable
timeline tree.

Usage::

    from app.core.tracing import get_tracer

    with get_tracer().span("request", category="agent"):
        with get_tracer().span("Memory Retrieval"):
            ...
        with get_tracer().span("Planner", category="agent"):
            ...

    print(get_tracer().render_tree())
    # Request (1.2s)
    #  ├── Memory Retrieval (35 ms)
    #  ├── Planner (41 ms)
    #  └── ...

Spans are request-scoped via ``contextvars`` — set the tracer once at
request entry and all downstream code automatically shares it.

The tracer is thread-safe (each span uses ``time.monotonic_ns``) and
adds no dependencies beyond the standard library.

W3C trace context propagation
-----------------------------
The :func:`parse_traceparent` and :func:`format_traceparent` helpers
enable distributed trace propagation via the ``traceparent`` header
(https://www.w3.org/TR/trace-context/).  The local ``trace_id`` and
``span_id`` are stored on the :class:`Span` so they can be correlated
with external systems.
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

# -- W3C trace context helpers ----------------------------------------------

_TRACE_VERSION = "00"


def generate_trace_id() -> str:
    """Generate a random 32-hex-character W3C trace id."""
    return format(random.getrandbits(128), "032x")


def generate_span_id() -> str:
    """Generate a random 16-hex-character W3C span id."""
    return format(random.getrandbits(64), "016x")


def parse_traceparent(header: str | None) -> tuple[str, str] | None:
    """Parse a W3C ``traceparent`` header into ``(trace_id, parent_span_id)``.

    Returns ``None`` if the header is missing, malformed, or has an
    unsupported version.

    Example::

        tp = parse_traceparent("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")
        # -> ("0af7651916cd43dd8448eb211c80319c", "b7ad6b7169203331")
    """
    if not header:
        return None
    parts = header.strip().split("-")
    if len(parts) != 4:
        return None
    version, trace_id, parent_span_id, trace_flags = parts
    if version != _TRACE_VERSION:
        return None
    if len(trace_id) != 32 or len(parent_span_id) != 16:
        return None
    try:
        int(trace_id, 16)
        int(parent_span_id, 16)
        int(trace_flags, 16)
    except ValueError:
        return None
    return trace_id, parent_span_id


def format_traceparent(trace_id: str, span_id: str, flags: str = "01") -> str:
    """Format a W3C ``traceparent`` header value.

    Args:
        trace_id: 32-char hex trace id.
        span_id: 16-char hex span id.
        flags: 2-char hex trace flags (default ``"01"`` = sampled).

    Returns:
        A header value like ``"00-<trace_id>-<span_id>-01"``.
    """
    return f"{_TRACE_VERSION}-{trace_id}-{span_id}-{flags}"


# -- request-scoped tracer --------------------------------------------------

_TRACER: ContextVar[Tracer | None] = ContextVar("_tracer", default=None)


def get_tracer() -> Tracer:
    """Return the request-scoped tracer.

    Returns a no-op tracer when no tracer has been set, so callers can
    safely create spans without checking for ``None``.
    """
    tracer = _TRACER.get()
    if tracer is None:
        return _NO_OP  # type: ignore[return-value]
    return tracer


def set_tracer(tracer: Tracer) -> None:
    """Bind a tracer for the current request / async task.

    Every call to :func:`get_tracer` inside the same task will return
    this tracer until the task completes.
    """
    _TRACER.set(tracer)


def reset_tracer() -> None:
    """Remove the tracer from the current context."""
    _TRACER.set(None)


# -- no-op tracer (avoids None-checks) -------------------------------------


class NoOpTracer:
    """Tracer that discards all spans. Used as the default when no real
    tracer has been set for the current request."""

    @contextmanager
    def span(
        self,
        name: str = "",
        *,
        category: str = "",
        **metadata: Any,
    ) -> Iterator[None]:
        yield

    def start(
        self,
        name: str = "",
        *,
        category: str = "",
        **metadata: Any,
    ) -> _NoOpSpan:
        return _NoOpSpan()

    def end(
        self,
        span: Any = None,
    ) -> None:
        pass

    def render_tree(self) -> str:
        return ""

    def serialize(self) -> dict[str, Any]:
        return {}


class _NoOpSpan:
    def close(self) -> None:
        pass


_NO_OP = NoOpTracer()


# -- real tracer ------------------------------------------------------------


@dataclass
class Span:
    """A single timed operation in the trace tree."""

    id: str
    name: str
    category: str
    parent_id: str | None
    started_at_ns: int
    completed_at_ns: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list[Span] = field(default_factory=list)
    trace_id: str = ""
    """W3C trace id for distributed trace correlation."""
    span_id: str = ""
    """W3C span id for distributed trace correlation."""

    @property
    def duration_ms(self) -> float:
        """Return the span duration in milliseconds."""
        if self.completed_at_ns is None:
            return 0.0
        return (self.completed_at_ns - self.started_at_ns) / 1_000_000

    @property
    def is_complete(self) -> bool:
        return self.completed_at_ns is not None

    def close(self) -> None:
        """Mark the span as complete."""
        if self.completed_at_ns is None:
            self.completed_at_ns = time.monotonic_ns()


def _format_duration(ms: float) -> str:
    """Format a duration in milliseconds to a human-readable string."""
    if ms < 1_000:
        return f"{round(ms)} ms"
    if ms < 60_000:
        return f"{ms / 1_000:.1f} s"
    return f"{ms / 60_000:.1f} min"


class Tracer:
    """Collects and renders hierarchical timing spans.

    Usage::

        tracer = Tracer()
        with tracer.span("request", category="agent"):
            with tracer.span("search", category="tool"):
                time.sleep(0.1)
        print(tracer.render_tree())
    """

    def __init__(self, trace_id: str | None = None) -> None:
        self._trace_id: str = trace_id or generate_trace_id()
        self._root: Span | None = None
        self._stack: list[Span] = []
        set_tracer(self)

    # -- context manager ----------------------------------------------------

    @contextmanager
    def span(
        self,
        name: str,
        *,
        category: str = "",
        **metadata: Any,
    ) -> Iterator[Span]:
        """Create a timed span as a context manager.

        Args:
            name: Human-readable label (e.g. ``"Planner"``, ``"Tool: GitHub"``).
            category: Optional grouping category (e.g. ``"tool"``, ``"agent"``).
            **metadata: Arbitrary key-value pairs attached to the span.

        Yields:
            The :class:`Span` being timed.
        """
        span = self._create_span(name, category=category, metadata=metadata)
        try:
            yield span
        finally:
            span.close()
            self._pop(span)

    # -- programmatic API ---------------------------------------------------

    def start(self, name: str, *, category: str = "", **metadata: Any) -> Span:
        """Start a new span and push it onto the stack.

        Must be paired with a corresponding :meth:`end` call::

            span = tracer.start("Planner")
            try:
                ...
            finally:
                tracer.end(span)

        Args:
            name: Human-readable label.
            category: Optional category.
            **metadata: Additional metadata.

        Returns:
            The newly created span.
        """
        return self._create_span(name, category=category, metadata=metadata)

    def end(self, span: Span) -> None:
        """Complete a span previously started via :meth:`start`.

        Args:
            span: The span to complete. Must match the top of the stack.
        """
        span.close()
        self._pop(span)

    # -- tree building ------------------------------------------------------

    def _create_span(
        self,
        name: str,
        *,
        category: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Span:
        span_id = generate_span_id()
        span = Span(
            id=str(uuid4()),
            name=name,
            category=category,
            parent_id=self._stack[-1].id if self._stack else None,
            started_at_ns=time.monotonic_ns(),
            metadata=metadata or {},
            trace_id=self._trace_id,
            span_id=span_id,
        )
        if self._stack:
            self._stack[-1].children.append(span)
        else:
            self._root = span
        self._stack.append(span)
        return span

    def _pop(self, span: Span) -> None:
        if self._stack and self._stack[-1].id == span.id:
            self._stack.pop()

    # -- rendering ----------------------------------------------------------

    def render_tree(self) -> str:
        """Render the trace tree as a human-readable timeline string.

        Returns:
            A multi-line string like::

                Request (1.2s)
                 ├── Memory Retrieval (35 ms)
                 ├── Planner (41 ms)
                 └── Tool: GitHub (190 ms)

            Returns an empty string if no spans have been recorded.
        """
        if self._root is None:
            return ""
        lines: list[str] = []
        self._render_node(self._root, "", True, lines)
        return "\n".join(lines)

    @staticmethod
    def _render_node(node: Span, prefix: str, is_last: bool, lines: list[str]) -> None:
        connector = "└── " if is_last else "├── "
        duration = _format_duration(node.duration_ms)
        label = f"{prefix}{connector}{node.name} ({duration})"
        lines.append(label)

        children = node.children
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(children):
            Tracer._render_node(child, child_prefix, i == len(children) - 1, lines)

    def serialize(self) -> dict[str, Any]:
        """Serialize the trace tree to a JSON-compatible dict.

        Returns:
            A nested dict representation of all spans.
        """
        if self._root is None:
            return {}
        return self._serialize_node(self._root)

    @staticmethod
    def _serialize_node(node: Span) -> dict[str, Any]:
        return {
            "name": node.name,
            "category": node.category,
            "duration_ms": round(node.duration_ms, 1),
            "metadata": node.metadata,
            "children": [Tracer._serialize_node(c) for c in node.children],
        }
