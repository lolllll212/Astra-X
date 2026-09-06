"""Unit tests for the capability registry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.tools.base import Tool
from app.tools.capabilities import CapabilityRegistry
from app.tools.errors import CapabilityNotFoundError
from app.tools.registry import ToolRegistry


class _FakeTool(Tool):
    """Minimal tool stub for testing."""

    def __init__(self, name: str, capabilities: list[str] | None = None) -> None:
        self._name = name
        self._caps = capabilities if capabilities is not None else [name]

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Fake {self._name}"

    @property
    def schema(self) -> MagicMock:
        return MagicMock()

    @property
    def capabilities(self) -> list[str]:
        return self._caps

    async def _execute(self, context: object, **kwargs: object) -> object:
        return MagicMock()


def _registry_with(*tools: _FakeTool) -> tuple[ToolRegistry, CapabilityRegistry]:
    tr = ToolRegistry()
    for t in tools:
        tr.register(t)
    cr = CapabilityRegistry(tr)
    return tr, cr


class TestCapabilityRegistry:
    def test_resolve_returns_tool_for_capability(self) -> None:
        _, cr = _registry_with(_FakeTool("web_search", capabilities=["search_web"]))
        tool = cr.resolve("search_web")
        assert tool.name == "web_search"

    def test_resolve_name_returns_tool_name(self) -> None:
        _, cr = _registry_with(_FakeTool("web_search", capabilities=["search_web"]))
        assert cr.resolve_name("search_web") == "web_search"

    def test_resolve_raises_for_unknown_capability(self) -> None:
        _, cr = _registry_with()
        with pytest.raises(CapabilityNotFoundError, match="search_web"):
            cr.resolve("search_web")

    def test_resolve_name_raises_for_unknown_capability(self) -> None:
        _, cr = _registry_with()
        with pytest.raises(CapabilityNotFoundError, match="search_web"):
            cr.resolve_name("search_web")

    def test_has_capability_returns_true_when_present(self) -> None:
        _, cr = _registry_with(_FakeTool("calc", capabilities=["calculate"]))
        assert cr.has_capability("calculate") is True

    def test_has_capability_returns_false_when_absent(self) -> None:
        _, cr = _registry_with()
        assert cr.has_capability("missing") is False

    def test_has_capability_returns_false_for_empty_list(self) -> None:
        _, cr = _registry_with(_FakeTool("tool", capabilities=[]))
        assert cr.has_capability("tool") is False

    def test_list_capabilities_returns_all_registered(self) -> None:
        _, cr = _registry_with(
            _FakeTool("web_search", capabilities=["search_web"]),
            _FakeTool("calc", capabilities=["calculate"]),
        )
        caps = cr.list_capabilities()
        assert sorted(caps) == ["calculate", "search_web"]

    def test_list_capabilities_empty_when_no_tools(self) -> None:
        _, cr = _registry_with()
        assert cr.list_capabilities() == []

    def test_get_tools_for_capability_returns_matching_tools(self) -> None:
        w1 = _FakeTool("web1", capabilities=["search_web"])
        w2 = _FakeTool("web2", capabilities=["search_web"])
        tr, cr = _registry_with(w1, w2)
        tools = cr.get_tools_for_capability("search_web")
        assert len(tools) == 2

    def test_get_tools_for_capability_returns_empty_for_unknown(self) -> None:
        _, cr = _registry_with()
        assert cr.get_tools_for_capability("missing") == []

    def test_rebuild_after_registering_new_tool(self) -> None:
        tr = ToolRegistry()
        cr = CapabilityRegistry(tr)
        assert cr.has_capability("new_cap") is False
        tr.register(_FakeTool("new_tool", capabilities=["new_cap"]))
        cr.rebuild()
        assert cr.has_capability("new_cap") is True

    def test_rebuild_reflects_unregistered_tools(self) -> None:
        tr = ToolRegistry()
        tool = _FakeTool("old_tool", capabilities=["old_cap"])
        tr.register(tool)
        cr = CapabilityRegistry(tr)
        assert cr.has_capability("old_cap") is True
        tr.unregister("old_tool")
        cr.rebuild()
        assert cr.has_capability("old_cap") is False

    def test_multiple_tools_same_capability_resolve_to_first(self) -> None:
        tr = ToolRegistry()
        tr.register(_FakeTool("web_search", capabilities=["search_web"]))
        tr.register(_FakeTool("web_scrape", capabilities=["search_web"]))
        cr = CapabilityRegistry(tr)
        tool = cr.resolve("search_web")
        assert tool.name == "web_search"

    def test_tool_defaults_name_as_capability(self) -> None:
        _, cr = _registry_with(_FakeTool("my_tool"))
        assert cr.has_capability("my_tool") is True
        assert cr.resolve_name("my_tool") == "my_tool"
