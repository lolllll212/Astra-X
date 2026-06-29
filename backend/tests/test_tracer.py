"""Unit tests for the hierarchical tracer."""

from __future__ import annotations

import time

import pytest

from app.core.tracing import (
    NoOpTracer,
    Span,
    Tracer,
    get_tracer,
    reset_tracer,
    set_tracer,
)


class TestTracer:
    def test_span_context_manager_creates_and_closes(self) -> None:
        tracer = Tracer()
        with tracer.span("test", category="test") as span:
            assert isinstance(span, Span)
            assert span.name == "test"
            assert span.category == "test"
            assert span.is_complete is False
        assert span.is_complete is True

    def test_span_duration_ms(self) -> None:
        tracer = Tracer()
        with tracer.span("slow"):
            time.sleep(0.01)
        span = tracer._root
        assert span is not None
        assert span.duration_ms >= 5.0

    def test_incomplete_span_duration_returns_zero(self) -> None:
        span = Span(
            id="s1",
            name="test",
            category="",
            parent_id=None,
            started_at_ns=time.monotonic_ns(),
        )
        assert span.duration_ms == 0.0

    def test_close_idempotent(self) -> None:
        tracer = Tracer()
        with tracer.span("test") as span:
            pass
        completed = span.completed_at_ns
        span.close()
        assert span.completed_at_ns == completed

    def test_nested_spans_form_tree(self) -> None:
        tracer = Tracer()
        with tracer.span("root"):
            with tracer.span("child1"):
                pass
            with tracer.span("child2"):
                with tracer.span("grandchild"):
                    pass
        assert tracer._root is not None
        assert len(tracer._root.children) == 2
        assert tracer._root.children[0].name == "child1"
        assert tracer._root.children[1].name == "child2"
        assert len(tracer._root.children[1].children) == 1
        assert tracer._root.children[1].children[0].name == "grandchild"

    def test_start_end_pair(self) -> None:
        tracer = Tracer()
        span = tracer.start("manual")
        assert span.is_complete is False
        tracer.end(span)
        assert span.is_complete is True

    def test_render_tree_empty(self) -> None:
        tracer = Tracer()
        assert tracer.render_tree() == ""

    def test_render_tree_single_span(self) -> None:
        tracer = Tracer()
        with tracer.span("root"):
            pass
        tree = tracer.render_tree()
        assert "root" in tree

    def test_render_tree_nested(self) -> None:
        tracer = Tracer()
        with tracer.span("root"):
            with tracer.span("child"):
                pass
        tree = tracer.render_tree()
        assert "root" in tree
        assert "child" in tree

    def test_serialize_empty(self) -> None:
        tracer = Tracer()
        assert tracer.serialize() == {}

    def test_serialize_nested(self) -> None:
        tracer = Tracer()
        with tracer.span("root", category="test", foo="bar"):
            with tracer.span("child", category="sub"):
                pass
        data = tracer.serialize()
        assert data["name"] == "root"
        assert data["category"] == "test"
        assert data["metadata"] == {"foo": "bar"}
        assert len(data["children"]) == 1
        assert data["children"][0]["name"] == "child"

    def test_metadata_on_span(self) -> None:
        tracer = Tracer()
        with tracer.span("test", key="value", num=42):
            pass
        assert tracer._root is not None
        assert tracer._root.metadata == {"key": "value", "num": 42}

    def test_span_without_category(self) -> None:
        tracer = Tracer()
        with tracer.span("plain"):
            pass
        assert tracer._root is not None
        assert tracer._root.category == ""

    def test_contextvar_scoping(self) -> None:
        reset_tracer()
        t1 = get_tracer()
        t2 = Tracer()
        set_tracer(t2)
        assert get_tracer() is not t1
        assert get_tracer() is t2
        reset_tracer()
        assert isinstance(get_tracer(), NoOpTracer)

    def test_contextvar_reset_clears_tracer(self) -> None:
        tracer = Tracer()
        assert get_tracer() is tracer
        reset_tracer()
        assert isinstance(get_tracer(), NoOpTracer)


class TestNoOpTracer:
    def test_span_does_nothing(self) -> None:
        tracer = NoOpTracer()
        with tracer.span("anything"):
            pass

    def test_render_empty(self) -> None:
        tracer = NoOpTracer()
        assert tracer.render_tree() == ""

    def test_serialize_empty(self) -> None:
        tracer = NoOpTracer()
        assert tracer.serialize() == {}

    def test_start_end_does_nothing(self) -> None:
        tracer = NoOpTracer()
        span = tracer.start("test")
        tracer.end(span)


class TestSpan:
    def test_is_complete_true_after_close(self) -> None:
        span = Span(
            id="s1",
            name="test",
            category="",
            parent_id=None,
            started_at_ns=time.monotonic_ns(),
        )
        assert span.is_complete is False
        span.close()
        assert span.is_complete is True

    def test_close_sets_completed_at_ns(self) -> None:
        span = Span(
            id="s1",
            name="test",
            category="",
            parent_id=None,
            started_at_ns=time.monotonic_ns(),
        )
        span.close()
        assert span.completed_at_ns is not None
        assert span.completed_at_ns >= span.started_at_ns


class TestFormatDuration:
    def test_ms(self) -> None:
        from app.core.tracing import _format_duration
        assert _format_duration(500) == "500 ms"
        assert _format_duration(0) == "0 ms"

    def test_seconds(self) -> None:
        from app.core.tracing import _format_duration
        assert _format_duration(1500) == "1.5 s"
        assert _format_duration(59000) == "59.0 s"

    def test_minutes(self) -> None:
        from app.core.tracing import _format_duration
        assert _format_duration(120000) == "2.0 min"
