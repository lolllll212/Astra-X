"""Optional OpenTelemetry integration.

This module is only imported when ``ASTRA_OTEL_EXPORTER_OTLP_ENDPOINT`` is
set and the ``opentelemetry-api`` / ``opentelemetry-sdk`` packages are
installed (``pip install astra-x-backend[observability]``).

Usage::

    from app.observability.opentelemetry import setup_otel, shutdown_otel

    await setup_otel(settings)
    # ... run app ...
    await shutdown_otel()
"""

from __future__ import annotations

import logging
from typing import Any

from app.config.settings import Settings

logger = logging.getLogger(__name__)

_OTLP_AVAILABLE: bool = False
_TracerProvider: type | None = None
_BatchSpanProcessor: type | None = None
_OTLPSpanExporter: type | None = None
_Resource: type | None = None
_ResourceAttributes: Any = None

try:
    from opentelemetry import trace as _otel_trace  # noqa: F811
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.semconv.resource import ResourceAttributes

    _TracerProvider = TracerProvider
    _BatchSpanProcessor = BatchSpanProcessor
    _OTLPSpanExporter = OTLPSpanExporter
    _Resource = Resource
    _ResourceAttributes = ResourceAttributes
    _OTLP_AVAILABLE = True
except ImportError:
    pass


def is_otel_available() -> bool:
    """Check whether the optional OTel packages are installed."""
    return _OTLP_AVAILABLE


def _create_resource(settings: Settings) -> Any:
    """Build an OpenTelemetry Resource from application settings."""
    attrs: dict[str, Any] = {
        _ResourceAttributes.SERVICE_NAME: settings.otel_service_name,
        _ResourceAttributes.SERVICE_VERSION: settings.app_version,
        _ResourceAttributes.DEPLOYMENT_ENVIRONMENT: settings.environment.value,
    }
    return _Resource.create(attrs)


def setup_otel(settings: Settings) -> bool:
    """Configure the OpenTelemetry SDK and register the OTLP exporter.

    Returns ``True`` if OTel was successfully configured, ``False`` if
    the optional packages are not installed or no endpoint was configured.

    This should be called during application startup (in the lifespan
    hook), *before* any trace-carrying requests are processed.
    """
    if not _OTLP_AVAILABLE:
        logger.info("OpenTelemetry packages not installed — OTLP export disabled.")
        return False

    endpoint = settings.otel_exporter_otlp_endpoint
    if not endpoint:
        logger.info("No ASTRA_OTEL_EXPORTER_OTLP_ENDPOINT configured — OTLP export disabled.")
        return False

    from opentelemetry import trace

    resource = _create_resource(settings)
    provider = _TracerProvider(resource=resource)
    exporter = _OTLPSpanExporter(endpoint=endpoint)
    processor = _BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    logger.info(
        "OpenTelemetry initialized",
        endpoint=endpoint,
        service_name=settings.otel_service_name,
    )
    return True


def shutdown_otel() -> None:
    """Flush and shut down the OpenTelemetry SDK.

    Must be called during application shutdown to ensure pending spans
    are delivered before the process exits.
    """
    if not _OTLP_AVAILABLE:
        return

    from opentelemetry import trace

    provider = trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()
        logger.info("OpenTelemetry shut down.")
