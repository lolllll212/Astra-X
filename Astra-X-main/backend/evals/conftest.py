"""Pytest integration for evaluation suites.

Allows running evals as pytest tests:
    pytest evals/ --eva-suite chat
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--eva-suite",
        action="store",
        default=None,
        help="Comma-separated eval suites to run (default: all)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    suite_opt = config.getoption("--eva-suite")
    if not suite_opt:
        return

    allowed = {s.strip() for s in suite_opt.split(",")}

    # Filter to only the selected suites
    selected: list[pytest.Item] = []
    for item in items:
        markers = {m.name for m in item.own_markers if m.name != "parametrize"}
        if markers & allowed:
            selected.append(item)

    items[:] = selected
