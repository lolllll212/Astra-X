"""Prometheus metric definitions for the Astra X backend.

All metrics are registered on a shared ``REGISTRY`` and use the
``astra_x`` prefix (configurable via ``ASTRA_METRICS_PREFIX``).
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info, REGISTRY

metrics_registry = REGISTRY

# -- App metadata -----------------------------------------------------------

app_info = Info(
    name="astra_x_app_info",
    documentation="Application metadata (version, environment).",
    registry=metrics_registry,
)

# -- HTTP metrics -----------------------------------------------------------

request_total = Counter(
    name="astra_x_http_requests_total",
    documentation="Total HTTP requests, partitioned by method, path, and status.",
    labelnames=["method", "path", "status"],
    registry=metrics_registry,
)

request_latency = Histogram(
    name="astra_x_http_request_duration_seconds",
    documentation="HTTP request latency in seconds.",
    labelnames=["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=metrics_registry,
)

in_flight_requests = Gauge(
    name="astra_x_http_requests_in_flight",
    documentation="Concurrent HTTP requests currently being processed.",
    labelnames=["method"],
    registry=metrics_registry,
)

# -- Business metrics -------------------------------------------------------

active_conversations = Gauge(
    name="astra_x_active_conversations",
    documentation="Number of currently active conversations.",
    registry=metrics_registry,
)

websocket_connections = Gauge(
    name="astra_x_websocket_connections",
    documentation="Number of currently open WebSocket connections.",
    registry=metrics_registry,
)

# -- LLM metrics ------------------------------------------------------------

llm_token_usage_total = Counter(
    name="astra_x_llm_tokens_total",
    documentation="Total LLM tokens consumed, partitioned by provider and model.",
    labelnames=["provider", "model", "type"],
    registry=metrics_registry,
)

# -- Security metrics -------------------------------------------------------

rate_limit_hits_total = Counter(
    name="astra_x_rate_limit_hits_total",
    documentation="Total rate-limit rejections, partitioned by client key prefix.",
    labelnames=["key_prefix"],
    registry=metrics_registry,
)

auth_failures_total = Counter(
    name="astra_x_auth_failures_total",
    documentation="Total authentication failures, partitioned by reason.",
    labelnames=["reason"],
    registry=metrics_registry,
)
