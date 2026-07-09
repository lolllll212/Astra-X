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
    active_conversations,
    app_info,
    auth_failures_total,
    in_flight_requests,
    llm_token_usage_total,
    memory_retrieval_duration,
    memory_store_duration,
    metrics_registry,
    planning_duration,
    plugin_load_duration,
    provider_request_duration,
    provider_request_total,
    rate_limit_hits_total,
    reflection_outcomes_total,
    request_latency,
    request_total,
    tool_execution_duration,
    websocket_connections,
)

__all__ = [
    "active_conversations",
    "app_info",
    "auth_failures_total",
    "in_flight_requests",
    "llm_token_usage_total",
    "memory_retrieval_duration",
    "memory_store_duration",
    "metrics_registry",
    "planning_duration",
    "plugin_load_duration",
    "provider_request_duration",
    "provider_request_total",
    "rate_limit_hits_total",
    "reflection_outcomes_total",
    "request_latency",
    "request_total",
    "tool_execution_duration",
    "websocket_connections",
]
