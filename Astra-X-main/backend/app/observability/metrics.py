"""Prometheus metric definitions for the Astra X backend.

All metrics are registered on a shared ``REGISTRY`` and use the
``astra_x`` prefix (configurable via ``ASTRA_METRICS_PREFIX``).
"""

from __future__ import annotations

from prometheus_client import REGISTRY, Counter, Gauge, Histogram, Info

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

# -- Agent/Planner metrics --------------------------------------------------

planning_duration = Histogram(
    name="astra_x_planning_duration_seconds",
    documentation="Time spent decomposing goals into plans.",
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    registry=metrics_registry,
)

# -- Tool execution metrics -------------------------------------------------

tool_execution_duration = Histogram(
    name="astra_x_tool_execution_duration_seconds",
    documentation="Time spent executing tools, partitioned by tool name and status.",
    labelnames=["tool_name", "status"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=metrics_registry,
)

# -- Memory metrics ---------------------------------------------------------

memory_retrieval_duration = Histogram(
    name="astra_x_memory_retrieval_duration_seconds",
    documentation="Time spent retrieving memories from the store.",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=metrics_registry,
)

memory_store_duration = Histogram(
    name="astra_x_memory_store_duration_seconds",
    documentation="Time spent storing results into memory.",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=metrics_registry,
)

# -- Provider latency metrics -----------------------------------------------

provider_request_duration = Histogram(
    name="astra_x_provider_request_duration_seconds",
    documentation="LLM provider request latency in seconds, partitioned by provider and model.",
    labelnames=["provider", "model"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=metrics_registry,
)

provider_request_total = Counter(
    name="astra_x_provider_requests_total",
    documentation="Total LLM provider requests, partitioned by provider, model, and result.",
    labelnames=["provider", "model", "result"],
    registry=metrics_registry,
)

# -- Plugin metrics ---------------------------------------------------------

plugin_load_duration = Histogram(
    name="astra_x_plugin_load_duration_seconds",
    documentation="Time spent loading plugins, partitioned by plugin name and status.",
    labelnames=["plugin_name", "status"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=metrics_registry,
)

# -- Reflection metrics -----------------------------------------------------

reflection_outcomes_total = Counter(
    name="astra_x_reflection_outcomes_total",
    documentation="Total reflection outcomes, partitioned by decision.",
    labelnames=["decision"],
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
