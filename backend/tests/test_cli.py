from __future__ import annotations

import os
import sys
import tempfile
from io import StringIO

import pytest

from app.cli import _build_parser, main
from app.config.settings import get_settings

# =========================================================================
# Parser unit tests
# =========================================================================


class TestParser:
    def test_plugin_list(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["plugin", "list"])
        assert args.command == "plugin"
        assert args.plugin_command == "list"

    def test_plugin_show(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["plugin", "show", "weather"])
        assert args.command == "plugin"
        assert args.plugin_command == "show"
        assert args.name == "weather"

    def test_plugin_install(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["plugin", "install", "weather"])
        assert args.command == "plugin"
        assert args.plugin_command == "install"
        assert args.name == "weather"
        assert args.path is None
        assert args.enable is False

    def test_plugin_install_with_path_and_enable(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["plugin", "install", "my-plugin", "--path", "/some/path", "--enable"])
        assert args.name == "my-plugin"
        assert args.path == "/some/path"
        assert args.enable is True

    def test_plugin_uninstall(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["plugin", "uninstall", "weather"])
        assert args.command == "plugin"
        assert args.plugin_command == "uninstall"
        assert args.name == "weather"

    def test_plugin_enable(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["plugin", "enable", "weather"])
        assert args.command == "plugin"
        assert args.plugin_command == "enable"
        assert args.name == "weather"

    def test_plugin_disable(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["plugin", "disable", "weather"])
        assert args.command == "plugin"
        assert args.plugin_command == "disable"
        assert args.name == "weather"

    def test_plugin_search(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["plugin", "search", "weather"])
        assert args.command == "plugin"
        assert args.plugin_command == "search"
        assert args.query == "weather"

    def test_plugin_update_all(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["plugin", "update"])
        assert args.command == "plugin"
        assert args.plugin_command == "update"
        assert args.name is None

    def test_plugin_update_specific(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["plugin", "update", "weather"])
        assert args.command == "plugin"
        assert args.plugin_command == "update"
        assert args.name == "weather"

    def test_no_command_prints_help(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


# =========================================================================
# Integration tests (each test is self-contained with a unique temp DB)
# =========================================================================


class TestCLIIntegration:
    @pytest.fixture
    def _fresh_db(self) -> None:
        """Set up a temporary file-based DB and restore env after."""
        fd, path = tempfile.mkstemp(suffix=".db", prefix="astra_cli_test_")
        os.close(fd)
        old_url = os.environ.get("ASTRA_DATABASE_URL")
        os.environ["ASTRA_DATABASE_URL"] = f"sqlite+aiosqlite:///{path}"
        get_settings.cache_clear()
        yield
        get_settings.cache_clear()
        if old_url is not None:
            os.environ["ASTRA_DATABASE_URL"] = old_url
        else:
            os.environ.pop("ASTRA_DATABASE_URL", None)
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(path + suffix)
            except OSError:
                pass

    def _capture_main(self, argv: list[str]) -> str:
        old_stdout = sys.stdout
        captured = StringIO()
        sys.stdout = captured
        try:
            main(argv)
        finally:
            sys.stdout = old_stdout
        return captured.getvalue()

    def test_plugin_list_empty(self, _fresh_db: None) -> None:
        output = self._capture_main(["plugin", "list"])
        assert "No plugins installed." in output

    def test_plugin_install_and_show(self, _fresh_db: None) -> None:
        self._capture_main(["plugin", "install", "weather"])
        output = self._capture_main(["plugin", "show", "weather"])
        assert "name: weather" in output
        assert "entry_point: app.plugins.weather:WeatherPlugin" in output
        assert "version: 0.1.0" in output

    def test_plugin_install_with_enable(self, _fresh_db: None) -> None:
        output = self._capture_main(["plugin", "install", "example_provider", "--enable"])
        assert "Installed plugin" in output
        assert "example_provider" in output

    def test_plugin_list(self, _fresh_db: None) -> None:
        self._capture_main(["plugin", "install", "weather"])
        self._capture_main(["plugin", "install", "example_provider"])
        output = self._capture_main(["plugin", "list"])
        assert "weather" in output
        assert "example_provider" in output

    def test_plugin_search_by_name(self, _fresh_db: None) -> None:
        self._capture_main(["plugin", "install", "weather"])
        output = self._capture_main(["plugin", "search", "weather"])
        assert "weather" in output

    def test_plugin_search_by_description(self, _fresh_db: None) -> None:
        self._capture_main(["plugin", "install", "weather"])
        output = self._capture_main(["plugin", "search", "mock"])
        assert "weather" in output

    def test_plugin_search_no_match(self, _fresh_db: None) -> None:
        self._capture_main(["plugin", "install", "weather"])
        output = self._capture_main(["plugin", "search", "nonexistent"])
        assert "No plugins matching" in output

    def test_plugin_disable(self, _fresh_db: None) -> None:
        self._capture_main(["plugin", "install", "weather", "--enable"])
        output = self._capture_main(["plugin", "disable", "weather"])
        assert "is now disabled" in output

    def test_plugin_enable(self, _fresh_db: None) -> None:
        self._capture_main(["plugin", "install", "weather"])
        output = self._capture_main(["plugin", "enable", "weather"])
        assert "is now enabled" in output

    def test_plugin_update_specific(self, _fresh_db: None) -> None:
        self._capture_main(["plugin", "install", "weather"])
        output = self._capture_main(["plugin", "update", "weather"])
        assert "Updating plugin 'weather'" in output
        assert "Updated" in output

    def test_plugin_update_all(self, _fresh_db: None) -> None:
        self._capture_main(["plugin", "install", "weather"])
        output = self._capture_main(["plugin", "update"])
        assert "Updating plugin" in output

    def test_plugin_uninstall(self, _fresh_db: None) -> None:
        self._capture_main(["plugin", "install", "weather"])
        output = self._capture_main(["plugin", "uninstall", "weather"])
        assert "Uninstalled plugin 'weather'" in output

    def test_plugin_show_not_found(self, _fresh_db: None) -> None:
        output = self._capture_main(["plugin", "show", "nonexistent"])
        assert "not found" in output

    def test_plugin_uninstall_not_found(self, _fresh_db: None) -> None:
        output = self._capture_main(["plugin", "uninstall", "nonexistent"])
        assert "not found" in output

    def test_plugin_install_unknown(self, _fresh_db: None) -> None:
        output = self._capture_main(["plugin", "install", "unknown-plugin"])
        assert "Unknown plugin" in output

    def test_plugin_install_duplicate(self, _fresh_db: None) -> None:
        self._capture_main(["plugin", "install", "weather"])
        output = self._capture_main(["plugin", "install", "weather"])
        assert "Failed to install" in output or "already exists" in output
