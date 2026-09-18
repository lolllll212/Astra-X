from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from sqlalchemy.ext.asyncio import create_async_engine

from app.config.settings import get_settings
from app.database.base import Base
from app.database.repositories.plugin_repository import PluginRepository
from app.database.session import create_session_factory, session_context
from app.services.plugin_service import PluginService

_BUILTIN_PLUGINS: dict[str, dict[str, Any]] = {
    "weather": {
        "display_name": "Weather Tools",
        "description": "Mock weather data for a given location.",
        "author": "Astra X Engineering Team",
        "version": "0.1.0",
        "entry_point": "app.plugins.weather:WeatherPlugin",
        "capabilities": ["get_weather"],
        "trust_tier": "official",
    },
    "example_provider": {
        "display_name": "Example Provider",
        "description": "Example LLM provider registration plugin.",
        "author": "Astra X Engineering Team",
        "version": "0.1.0",
        "entry_point": "app.plugins.example_provider:ExampleProviderPlugin",
        "capabilities": [],
        "trust_tier": "official",
    },
}


def _db_url() -> str:
    return get_settings().database_url


async def _ensure_tables(engine: Any) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def _run_plugin_list() -> None:
    engine = create_async_engine(_db_url())
    try:
        await _ensure_tables(engine)
        factory = create_session_factory(engine)
        async with session_context(factory) as session:
            repo = PluginRepository(session)
            service = PluginService(repo)
            plugins = await service.list_installed()
    finally:
        await engine.dispose()

    if not plugins:
        print("No plugins installed.")
        return

    header = f"{'Name':<24} {'Version':<12} {'Enabled':<9} {'Status':<12} {'Trust':<12} {'Capabilities'}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for p in plugins:
        caps = ", ".join(p.capabilities) if p.capabilities else "-"
        print(
            f"{p.name:<24} {p.version:<12} {p.enabled!s:<9} {p.status:<12} {p.trust_tier:<12} {caps}"
        )


async def _run_plugin_show(name: str) -> None:
    engine = create_async_engine(_db_url())
    try:
        await _ensure_tables(engine)
        factory = create_session_factory(engine)
        async with session_context(factory) as session:
            repo = PluginRepository(session)
            service = PluginService(repo)
            try:
                plugin = await service.get_by_name(name)
            except Exception:
                print(f"Plugin '{name}' not found.")
                return
    finally:
        await engine.dispose()

    for field, value in sorted(plugin.model_dump(exclude_none=True).items()):
        print(f"{field}: {value}")


async def _run_plugin_install(
    name: str,
    path: str | None,
    enable: bool,
) -> None:
    engine = create_async_engine(_db_url())
    try:
        await _ensure_tables(engine)
        factory = create_session_factory(engine)
        async with session_context(factory) as session:
            repo = PluginRepository(session)
            service = PluginService(repo)

            if name in _BUILTIN_PLUGINS:
                meta = _BUILTIN_PLUGINS[name]
                spec = await service.install(
                    name=name,
                    display_name=meta["display_name"],
                    version=meta["version"],
                    description=meta["description"],
                    author=meta["author"],
                    entry_point=meta["entry_point"],
                    capabilities=meta["capabilities"],
                    trust_tier=meta["trust_tier"],
                    enabled=enable,
                )
                print(f"Installed plugin '{spec.name}' (v{spec.version})")
                return

            if path is not None:
                spec = await service.install(
                    name=name,
                    display_name=name,
                    version="0.1.0",
                    install_path=path,
                    enabled=enable,
                )
                print(f"Installed plugin from path '{path}' as '{spec.name}'")
                return

            print(f"Unknown plugin '{name}'. Use --path to install from a directory.")
    except Exception as exc:
        print(f"Failed to install plugin: {exc}")
    finally:
        await engine.dispose()


async def _run_plugin_uninstall(name: str) -> None:
    engine = create_async_engine(_db_url())
    try:
        await _ensure_tables(engine)
        factory = create_session_factory(engine)
        async with session_context(factory) as session:
            repo = PluginRepository(session)
            service = PluginService(repo)
            try:
                plugin = await service.get_by_name(name)
                await service.uninstall(plugin.id)
                print(f"Uninstalled plugin '{name}'.")
            except Exception:
                print(f"Plugin '{name}' not found.")
    finally:
        await engine.dispose()


async def _run_plugin_enable(name: str) -> None:
    await _run_plugin_set_enabled(name, True)


async def _run_plugin_disable(name: str) -> None:
    await _run_plugin_set_enabled(name, False)


async def _run_plugin_set_enabled(name: str, enabled: bool) -> None:
    engine = create_async_engine(_db_url())
    try:
        await _ensure_tables(engine)
        factory = create_session_factory(engine)
        async with session_context(factory) as session:
            repo = PluginRepository(session)
            service = PluginService(repo)
            try:
                plugin = await service.get_by_name(name)
                if enabled:
                    await service.activate(plugin.id)
                else:
                    await service.deactivate(plugin.id)
                status = "enabled" if enabled else "disabled"
                print(f"Plugin '{name}' is now {status}.")
            except Exception:
                print(f"Plugin '{name}' not found.")
    finally:
        await engine.dispose()


async def _run_plugin_search(query: str) -> None:
    engine = create_async_engine(_db_url())
    try:
        await _ensure_tables(engine)
        factory = create_session_factory(engine)
        async with session_context(factory) as session:
            repo = PluginRepository(session)
            service = PluginService(repo)
            plugins = await service.list_installed()
    finally:
        await engine.dispose()

    q = query.lower()
    matches = [
        p
        for p in plugins
        if q in p.name.lower()
        or q in p.display_name.lower()
        or q in p.description.lower()
        or any(q in c.lower() for c in p.capabilities)
    ]

    if not matches:
        print(f"No plugins matching '{query}'.")
        return

    header = f"{'Name':<24} {'Version':<12} {'Trust Tier':<14} {'Capabilities'}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for p in matches:
        caps = ", ".join(p.capabilities) if p.capabilities else "-"
        print(f"{p.name:<24} {p.version:<12} {p.trust_tier:<14} {caps}")


async def _run_plugin_update(name: str | None) -> None:
    engine = create_async_engine(_db_url())
    try:
        await _ensure_tables(engine)
        factory = create_session_factory(engine)
        async with session_context(factory) as session:
            repo = PluginRepository(session)
            plugins = await repo.list_all()

            if name is not None:
                targets = [p for p in plugins if p.name == name]
                if not targets:
                    print(f"Plugin '{name}' not found.")
                    return
            else:
                targets = plugins

            if not targets:
                print("No plugins to update.")
                return

            for spec in targets:
                print(f"Updating plugin '{spec.name}'...")
                existing = await repo.find_by_name(spec.name)
                if existing is None:
                    print(f"  Plugin '{spec.name}' no longer exists.")
                    continue
                if spec.name in _BUILTIN_PLUGINS:
                    meta = _BUILTIN_PLUGINS[spec.name]
                    updated = existing.model_copy(
                        update={
                            "version": meta["version"],
                            "description": meta["description"],
                            "entry_point": meta["entry_point"],
                        }
                    )
                    await repo.update(updated)
                    print(f"  Updated to v{meta['version']}.")
                else:
                    print(f"  No update source for '{spec.name}'.")
    finally:
        await engine.dispose()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astra",
        description="Astra X — AI assistant backend management CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- plugin subcommand group --
    plugin_parser = sub.add_parser("plugin", help="Manage plugins")
    plugin_sub = plugin_parser.add_subparsers(dest="plugin_command", required=True)

    # plugin list
    plugin_sub.add_parser("list", help="List installed plugins")

    # plugin show
    show_p = plugin_sub.add_parser("show", help="Show plugin details")
    show_p.add_argument("name", help="Plugin name")

    # plugin search
    search_p = plugin_sub.add_parser("search", help="Search installed plugins")
    search_p.add_argument("query", help="Search term")

    # plugin install
    install_p = plugin_sub.add_parser("install", help="Install a plugin")
    install_p.add_argument("name", help="Plugin name or path")
    install_p.add_argument("--path", help="Install from local directory path")
    install_p.add_argument("--enable", action="store_true", help="Enable after install")

    # plugin uninstall
    uninstall_p = plugin_sub.add_parser("uninstall", help="Uninstall a plugin")
    uninstall_p.add_argument("name", help="Plugin name")

    # plugin enable / disable
    enable_p = plugin_sub.add_parser("enable", help="Enable a plugin")
    enable_p.add_argument("name", help="Plugin name")

    disable_p = plugin_sub.add_parser("disable", help="Disable a plugin")
    disable_p.add_argument("name", help="Plugin name")

    # plugin update
    update_p = plugin_sub.add_parser("update", help="Update plugin metadata")
    update_p.add_argument("name", nargs="?", default=None, help="Plugin name (all if omitted)")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "plugin":
        cmd = args.plugin_command

        if cmd == "list":
            asyncio.run(_run_plugin_list())
        elif cmd == "show":
            asyncio.run(_run_plugin_show(args.name))
        elif cmd == "install":
            asyncio.run(_run_plugin_install(args.name, args.path, args.enable))
        elif cmd == "uninstall":
            asyncio.run(_run_plugin_uninstall(args.name))
        elif cmd == "enable":
            asyncio.run(_run_plugin_enable(args.name))
        elif cmd == "disable":
            asyncio.run(_run_plugin_disable(args.name))
        elif cmd == "search":
            asyncio.run(_run_plugin_search(args.query))
        elif cmd == "update":
            asyncio.run(_run_plugin_update(args.name))
        else:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
