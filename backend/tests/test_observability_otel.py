"""Tests for the optional OpenTelemetry integration.

These tests run without the OTel packages installed (they are
optional extras).  The module should gracefully degrade.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestOpenTelemetryFallback:
    def test_is_otel_available_returns_false_when_not_installed(self) -> None:
        from app.observability.opentelemetry import is_otel_available

        assert not is_otel_available()

    def test_setup_otel_returns_false_when_not_installed(self) -> None:
        from app.observability.opentelemetry import setup_otel

        settings = MagicMock()
        settings.otel_exporter_otlp_endpoint = "http://localhost:4318"
        result = setup_otel(settings)
        assert not result

    def test_setup_otel_returns_false_when_no_endpoint(self) -> None:
        from app.observability.opentelemetry import setup_otel

        settings = MagicMock()
        settings.otel_exporter_otlp_endpoint = None
        result = setup_otel(settings)
        assert not result

    def test_shutdown_otel_noops_when_not_installed(self) -> None:
        from app.observability.opentelemetry import shutdown_otel

        shutdown_otel()
