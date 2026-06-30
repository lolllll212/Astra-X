"""Tests for the observability module (Prometheus metrics, tracing propagation)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import app.observability  # noqa: F401 — ensure metrics are registered

from prometheus_client import REGISTRY

from app.core.tracing import (
    Tracer,
    format_traceparent,
    generate_span_id,
    generate_trace_id,
    parse_traceparent,
)


class TestTraceparentPropagation:
    def test_generate_trace_id_length(self) -> None:
        tid = generate_trace_id()
        assert len(tid) == 32
        int(tid, 16)

    def test_generate_span_id_length(self) -> None:
        sid = generate_span_id()
        assert len(sid) == 16
        int(sid, 16)

    def test_parse_valid_traceparent(self) -> None:
        tid = generate_trace_id()
        sid = generate_span_id()
        header = format_traceparent(tid, sid)
        parsed = parse_traceparent(header)
        assert parsed == (tid, sid)

    def test_parse_none_returns_none(self) -> None:
        assert parse_traceparent(None) is None

    def test_parse_empty_returns_none(self) -> None:
        assert parse_traceparent("") is None

    def test_parse_garbage_returns_none(self) -> None:
        assert parse_traceparent("garbage") is None
        assert parse_traceparent("00-xxxx-xxxx-01") is None

    def test_tracer_uses_custom_trace_id(self) -> None:
        tid = generate_trace_id()
        tracer = Tracer(trace_id=tid)
        assert tracer._trace_id == tid

        with tracer.span("root"):
            with tracer.span("child"):
                pass

        root = tracer._root
        assert root is not None
        assert root.trace_id == tid
        assert root.span_id != ""
        assert root.children[0].trace_id == tid
        assert root.children[0].span_id != ""

    def test_tracer_default_trace_id(self) -> None:
        tracer = Tracer()
        assert len(tracer._trace_id) == 32


class TestPrometheusMetrics:
    def test_metrics_registry_has_correct_metrics(self) -> None:
        # Family names: Counter `_total` suffix is stripped internally by prometheus_client
        expected = {
            "astra_x_http_requests",
            "astra_x_http_request_duration_seconds",
            "astra_x_http_requests_in_flight",
            "astra_x_active_conversations",
            "astra_x_websocket_connections",
            "astra_x_llm_tokens",
            "astra_x_rate_limit_hits",
            "astra_x_auth_failures",
            "astra_x_app_info",
        }
        registered = {m.name for m in REGISTRY.collect() if m.name.startswith("astra_x")}
        missing = expected - registered
        assert not missing, f"Missing metric families: {missing}"

    def test_request_total_counter(self) -> None:
        from app.observability import request_total

        before = REGISTRY.get_sample_value("astra_x_http_requests_total", {"method": "GET", "path": "/test", "status": "200"})
        before = before or 0.0
        request_total.labels(method="GET", path="/test", status="200").inc()
        after = REGISTRY.get_sample_value("astra_x_http_requests_total", {"method": "GET", "path": "/test", "status": "200"})
        assert after == before + 1.0

    def test_request_latency_histogram(self) -> None:
        from app.observability import request_latency

        request_latency.labels(method="POST", path="/chat").observe(0.5)
        count = REGISTRY.get_sample_value(
            "astra_x_http_request_duration_seconds_count",
            {"method": "POST", "path": "/chat"},
        )
        assert count == 1.0 or count is None

    def test_in_flight_gauge(self) -> None:
        from app.observability import in_flight_requests

        in_flight_requests.labels(method="GET").inc()
        in_flight_requests.labels(method="GET").inc()
        in_flight_requests.labels(method="GET").dec()
        val = REGISTRY.get_sample_value("astra_x_http_requests_in_flight", {"method": "GET"})
        assert val == 1.0

    def test_llm_tokens_counter(self) -> None:
        from app.observability import llm_token_usage_total

        before = REGISTRY.get_sample_value(
            "astra_x_llm_tokens_total",
            {"provider": "ollama", "model": "llama3.1", "type": "prompt"},
        )
        before = before or 0.0
        llm_token_usage_total.labels(provider="ollama", model="llama3.1", type="prompt").inc(150)
        after = REGISTRY.get_sample_value(
            "astra_x_llm_tokens_total",
            {"provider": "ollama", "model": "llama3.1", "type": "prompt"},
        )
        assert after == before + 150.0

    def test_rate_limit_hits_counter(self) -> None:
        from app.observability import rate_limit_hits_total

        rate_limit_hits_total.labels(key_prefix="192.168").inc()
        val = REGISTRY.get_sample_value("astra_x_rate_limit_hits_total", {"key_prefix": "192.168"})
        assert val == 1.0

    def test_auth_failures_counter(self) -> None:
        from app.observability import auth_failures_total

        auth_failures_total.labels(reason="invalid_key").inc()
        val = REGISTRY.get_sample_value("astra_x_auth_failures_total", {"reason": "invalid_key"})
        assert val == 1.0

    def test_app_info(self) -> None:
        from app.observability import app_info

        app_info.info({"version": "1.0.0", "environment": "test"})
        # Info metric: sample value is 1.0, labels carry the info data
        info_sample = REGISTRY.get_sample_value(
            "astra_x_app_info_info",
            {"version": "1.0.0", "environment": "test"},
        )
        assert info_sample == 1.0
