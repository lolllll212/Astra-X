"""Observability — Prometheus metrics, optional OpenTelemetry, Grafana dashboards.

Usage::

    from app.observability import metrics_registry, request_latency, app_info

    # Increment a counter
    request_total.labels(method="GET", path="/api/v1/health", status="200").inc()

    # Observe a duration
    request_latency.labels(method="POST", path="/chat").observe(0.125)

The Prometheus client is bundled in the core dependency; OpenTelemetry
extras are optional (install via ``pip install astra-x-backend[observability]``).
"""

from __future__ import annotations

from app.observability.metrics import (
    app_info,
    active_conversations,
    auth_failures_total,
    in_flight_requests,
    llm_token_usage_total,
    metrics_registry,
    rate_limit_hits_total,
    request_latency,
    request_total,
    websocket_connections,
)

__all__ = [
    "metrics_registry",
    "app_info",
    "request_total",
    "request_latency",
    "in_flight_requests",
    "active_conversations",
    "websocket_connections",
    "llm_token_usage_total",
    "rate_limit_hits_total",
    "auth_failures_total",
]
