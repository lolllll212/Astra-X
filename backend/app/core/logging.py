"""Structured logging configuration for Astra X.

This module is the single place where the application's logging behavior
is configured. It unifies two log sources under one processor pipeline:

* **structlog-native logs** — produced by application code via
  :func:`get_logger`.
* **stdlib-originated logs** — produced by third-party libraries (Uvicorn,
  SQLAlchemy, etc.) that log through the standard library ``logging``
  module and know nothing about structlog.

Both are routed through the same processor chain so that, regardless of
origin, every log line in development is colorized and human-readable,
and every log line in production is a single JSON object suitable for a
log aggregator.

Field conventions used throughout the application:

* ``request_id`` and ``conversation_id`` are **ambient context**: true for
  every log line emitted while a particular request or conversation is
  being processed. Bind them once via :func:`bind_request_context` (or
  the :func:`request_context` context manager) and every subsequent log
  call automatically includes them — no need to pass them explicitly.
* ``provider``, ``model``, ``latency_ms``, and token counts are
  **point-in-time facts about a single event**, not the whole request.
  They are passed as ordinary keyword arguments at the call site, e.g.::

      logger.info(
          "llm_generation_complete",
          provider=provider_name,
          model=model_name,
          latency_ms=latency_ms,
          prompt_tokens=usage.prompt_tokens,
          completion_tokens=usage.completion_tokens,
      )

  Binding these to ambient context instead would leak stale values into
  unrelated log lines emitted later in the same request.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final, cast

import orjson
import structlog
from structlog.processors import CallsiteParameter, CallsiteParameterAdder
from structlog.typing import EventDict, Processor

from app.config.settings import LogFormat, LogLevel, Settings, get_settings

__all__ = [
    "configure_logging",
    "get_logger",
    "bind_request_context",
    "bind_conversation_id",
    "clear_request_context",
    "request_context",
]

_REQUEST_ID_KEY: Final[str] = "request_id"
_CONVERSATION_ID_KEY: Final[str] = "conversation_id"

_LOG_LEVEL_TO_STDLIB: Final[dict[LogLevel, int]] = {
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARNING: logging.WARNING,
    LogLevel.ERROR: logging.ERROR,
    LogLevel.CRITICAL: logging.CRITICAL,
}

_configured: bool = False
"""Idempotency guard for :func:`configure_logging`.

Logging configuration is inherently process-global — there is exactly one
root logger. This flag prevents accidentally installing duplicate handlers
if ``configure_logging`` is called more than once, while ``force=True``
still allows deliberate reconfiguration (e.g. between test cases).
"""


def _orjson_serializer(event_dict: EventDict, **_: object) -> str:
    """Serialize a structlog event dictionary to a JSON string via orjson.

    Used as the ``serializer`` for :class:`structlog.processors.JSONRenderer`
    in production. orjson natively serializes ``UUID`` and ``datetime``
    values — both used pervasively in this application — without requiring
    a custom ``default=`` fallback.

    Args:
        event_dict: The fully-processed structlog event dictionary.
        **_: Additional keyword arguments ``JSONRenderer`` may pass through
            (unused by orjson, accepted for interface compatibility).

    Returns:
        A single-line JSON string with no trailing newline.
    """
    return orjson.dumps(event_dict).decode("utf-8")


def configure_logging(settings: Settings | None = None, *, force: bool = False) -> None:
    """Configure structlog and the stdlib root logger for the whole process.

    Must be called once, early in application startup, before any logger
    is used. Safe to call multiple times: subsequent calls are a no-op
    unless ``force=True`` is passed.

    The renderer and minimum severity are entirely config-driven:

    * :attr:`LogFormat.CONSOLE` renders colorized, human-readable lines
      and additionally attaches callsite info (file, function, line)
      when ``settings.is_development`` — useful locally, unnecessary
      overhead anywhere else.
    * :attr:`LogFormat.JSON` renders one JSON object per line and
      flattens exception tracebacks into a string field, suitable for
      ingestion by a log aggregator.

    Args:
        settings: The application settings to read logging configuration
            from. Defaults to the process-wide cached settings via
            :func:`app.config.settings.get_settings` when omitted —
            accepting it explicitly here keeps this function testable
            without requiring environment-variable patching.
        force: When ``True``, reconfigure even if already configured.
            Intended for use in test suites that exercise multiple
            logging configurations within one process.
    """
    global _configured
    if _configured and not force:
        return

    resolved_settings = settings if settings is not None else get_settings()
    stdlib_level = _LOG_LEVEL_TO_STDLIB[resolved_settings.log_level]

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if resolved_settings.is_development:
        shared_processors.append(
            CallsiteParameterAdder(
                {
                    CallsiteParameter.FILENAME,
                    CallsiteParameter.FUNC_NAME,
                    CallsiteParameter.LINENO,
                }
            )
        )

    renderer: Processor
    if resolved_settings.log_format is LogFormat.JSON:
        shared_processors.append(structlog.processors.format_exc_info)
        renderer = structlog.processors.JSONRenderer(serializer=_orjson_serializer)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler: logging.Handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(stdlib_level)

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger for use throughout the application.

    Must be called only after :func:`configure_logging` has run (typically
    once, at application startup); calling it earlier still returns a
    usable logger, but it will not yet reflect the configured
    renderer/level.

    Args:
        name: Logical name for the logger, conventionally the calling
            module's ``__name__``. Appears in output as the ``logger``
            field. Defaults to structlog's own default when omitted.

    Returns:
        A bound logger supporting structured keyword arguments on every
        call, e.g. ``get_logger(__name__).info("event_name", key=value)``.
    """
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))


def bind_request_context(request_id: str, conversation_id: str | None = None) -> None:
    """Bind request-scoped identifiers onto every subsequent log call.

    Intended to be called once per request, typically from HTTP
    middleware, immediately after a request ID is generated. Every log
    line emitted afterward — by application code or by libraries logging
    through the stdlib — automatically includes these fields until
    :func:`clear_request_context` is called.

    Args:
        request_id: Unique identifier for the current request.
        conversation_id: Unique identifier for the current conversation,
            if one is already known when the request context is bound
            (e.g. supplied in the request path or body). Use
            :func:`bind_conversation_id` if it only becomes known partway
            through handling the request.
    """
    context: dict[str, str] = {_REQUEST_ID_KEY: request_id}
    if conversation_id is not None:
        context[_CONVERSATION_ID_KEY] = conversation_id
    structlog.contextvars.bind_contextvars(**context)


def bind_conversation_id(conversation_id: str) -> None:
    """Attach a conversation ID to the current request's logging context.

    Use this when the conversation ID is not yet known at the time
    :func:`bind_request_context` was called — for example, when a new
    conversation is created partway through handling a request.

    Args:
        conversation_id: Unique identifier for the current conversation.
    """
    structlog.contextvars.bind_contextvars(**{_CONVERSATION_ID_KEY: conversation_id})


def clear_request_context() -> None:
    """Remove all request-scoped fields from the logging context.

    Must be called at the end of every request to prevent one request's
    ``request_id``/``conversation_id`` from leaking into log lines
    emitted while handling a later, unrelated request on the same worker.
    Prefer the :func:`request_context` context manager, which calls this
    automatically.
    """
    structlog.contextvars.clear_contextvars()


@contextmanager
def request_context(request_id: str, conversation_id: str | None = None) -> Iterator[None]:
    """Bind request-scoped logging context for the duration of a block.

    Guarantees that bound context is cleared on exit — including when the
    wrapped code raises — so it is the preferred way to scope
    ``request_id``/``conversation_id`` around a single request, typically
    from within HTTP middleware:

        with request_context(request_id=request_id):
            response = await call_next(request)

    Args:
        request_id: Unique identifier for the current request.
        conversation_id: Unique identifier for the current conversation,
            if already known.

    Yields:
        None. The block executes with the request context bound.
    """
    bind_request_context(request_id, conversation_id)
    try:
        yield
    finally:
        clear_request_context()