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
        description="Astra X — local AI assistant CLI.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {get_settings().app_version}",
    )
    sub = parser.add_subparsers(dest="command")

    # -- interactive REPL (default when no subcommand is given) --
    sub.add_parser("repl", help="Start an interactive chat session (default)")

    # -- one-shot chat --
    chat_p = sub.add_parser("chat", help="Send a single message and print the reply")
    chat_p.add_argument("text", help="Your message")
    chat_p.add_argument("--model", help="Model id override (LM Studio model)")
    chat_p.add_argument("-v", "--verbose", action="store_true", help="Show raw stream events")

    # -- one-shot agent run --
    run_p = sub.add_parser("run", help="Run an agent task (plan → execute → reflect → answer)")
    run_p.add_argument("text", help="The task description")
    run_p.add_argument("--model", help="Model id override (LM Studio model)")
    run_p.add_argument("-v", "--verbose", action="store_true", help="Show raw stream events")

    # -- tools group --
    tools_p = sub.add_parser("tools", help="Inspect and invoke tools")
    tools_sub = tools_p.add_subparsers(dest="tools_command", required=True)
    tools_sub.add_parser("list", help="List registered tools")
    call_p = tools_sub.add_parser("call", help="Invoke a tool directly")
    call_p.add_argument("tool", help="Tool name")
    call_p.add_argument("args", nargs="*", help="key=value arguments (values parsed as JSON/types)")

    # -- models --
    sub.add_parser("models", help="List models served by LM Studio")

    # -- doctor --
    doctor_p = sub.add_parser("doctor", help="Run tool + provider connectivity checks")
    doctor_p.add_argument("-v", "--verbose", action="store_true", help="Show extra detail")

    # -- serve --
    serve_p = sub.add_parser("serve", help="Start the FastAPI backend server")
    serve_p.add_argument("--host", default=None, help="Bind host (default from settings)")
    serve_p.add_argument("--port", type=int, default=None, help="Bind port (default from settings)")
    serve_p.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # -- session group --
    session_p = sub.add_parser("session", help="Manage sessions (conversations)")
    session_sub = session_p.add_subparsers(dest="session_command", required=True)
    sess_list_p = session_sub.add_parser("list", help="List sessions")
    sess_list_p.add_argument("-n", "--max-count", type=int, default=None, help="Limit to N most recent")
    sess_list_p.add_argument("--format", dest="fmt", choices=["table", "json"], default="table")
    sess_del_p = session_sub.add_parser("delete", help="Delete a session")
    sess_del_p.add_argument("session_id", help="Session ID")

    # -- stats --
    stats_p = sub.add_parser("stats", help="Show token usage statistics")
    stats_p.add_argument("--days", type=int, default=None, help="Only the last N days")

    # -- export / import --
    export_p = sub.add_parser("export", help="Export session data as JSON")
    export_p.add_argument("session_id", nargs="?", default=None, help="Session ID (prompts for selection if omitted)")
    import_p = sub.add_parser("import", help="Import session data from a JSON file")
    import_p.add_argument("path", help="Path to the JSON file")

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

    # -- config subcommand group --
    config_parser = sub.add_parser("config", help="Manage configuration")
    config_sub = config_parser.add_subparsers(dest="config_command", required=True)
    config_sub.add_parser("list", help="List all configuration values")
    config_get_p = config_sub.add_parser("get", help="Get a config value")
    config_get_p.add_argument("key", help="Config key")
    config_set_p = config_sub.add_parser("set", help="Set a config value")
    config_set_p.add_argument("key", help="Config key")
    config_set_p.add_argument("value", help="Config value")
    config_sub.add_parser("reset", help="Reset .env to defaults")

    # -- tui --
    sub.add_parser("tui", help="Launch the terminal UI (opencode-style)")

    # -- init --
    init_p = sub.add_parser("init", help="Initialize a new Astra X project")
    init_p.add_argument("path", nargs="?", default=".", help="Project directory")

    # -- completion --
    comp_p = sub.add_parser("completion", help="Generate or install shell completion")
    comp_p.add_argument("shell", nargs="?", choices=["bash", "zsh", "fish", "powershell"], help="Shell type")
    comp_p.add_argument("--install", action="store_true", help="Install completion to shell config")

    # -- provider subcommand group --
    provider_parser = sub.add_parser("provider", help="Manage LLM providers")
    provider_sub = provider_parser.add_subparsers(dest="provider_command", required=True)
    provider_sub.add_parser("list", help="List available providers")
    provider_use_p = provider_sub.add_parser("use", help="Switch default provider")
    provider_use_p.add_argument("name", help="Provider name (lm_studio, nvidia_nim, openrouter, ollama, openai_compatible)")
    provider_models_p = provider_sub.add_parser("models", help="List models for a provider")
    provider_models_p.add_argument("provider", nargs="?", help="Provider name (default: current)")

    return parser


def _configure_utf8_output() -> None:
    """Reconfigure stdout/stderr to UTF-8 so odd console code pages don't crash."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main(argv: list[str] | None = None) -> None:
    _configure_utf8_output()
    parser = _build_parser()
    args = parser.parse_args(argv)

    # No subcommand → interactive REPL.
    if args.command is None or args.command == "repl":
        from app.cli_repl_enhanced import run_repl

        sys.exit(run_repl())

    if args.command == "chat":
        import asyncio
        from app.cli_commands import cmd_chat
        from app.cli_runtime import build_runtime

        runtime = build_runtime()
        try:
            asyncio.run(cmd_chat(runtime, args.text, model=args.model, verbose=args.verbose))
        finally:
            asyncio.run(runtime.close())
        return

    if args.command == "run":
        import asyncio
        from app.cli_commands import cmd_run
        from app.cli_runtime import build_runtime

        runtime = build_runtime()
        try:
            asyncio.run(cmd_run(runtime, args.text, model=args.model, verbose=args.verbose))
        finally:
            asyncio.run(runtime.close())
        return

    if args.command == "tools":
        from app.cli_commands import cmd_tools_call, cmd_tools_list, parse_keyval
        from app.cli_runtime import build_runtime

        runtime = build_runtime()
        try:
            if args.tools_command == "list":
                sys.exit(asyncio.run(cmd_tools_list(runtime)))
            if args.tools_command == "call":
                args_dict = parse_keyval(args.args)
                sys.exit(asyncio.run(cmd_tools_call(runtime, args.tool, args_dict)))
        finally:
            asyncio.run(runtime.close())
        return

    if args.command == "models":
        from app.cli_commands import cmd_models
        from app.cli_runtime import build_runtime

        runtime = build_runtime()
        try:
            sys.exit(asyncio.run(cmd_models(runtime)))
        finally:
            asyncio.run(runtime.close())
        return

    if args.command == "doctor":
        from app.cli_doctor import run_doctor
        from app.cli_runtime import build_runtime

        runtime = build_runtime()
        try:
            asyncio.run(run_doctor(runtime))
        finally:
            asyncio.run(runtime.close())
        return

    if args.command == "serve":
        from app.config.settings import get_settings as _get_settings

        settings = _get_settings()
        host = args.host or settings.host
        port = args.port or settings.port
        from app.cli_commands import cmd_serve

        sys.exit(asyncio.run(cmd_serve(host=host, port=port, reload=args.reload)))

    if args.command == "session":
        from app.cli_session import cmd_session_delete, cmd_session_list

        if args.session_command == "list":
            sys.exit(asyncio.run(cmd_session_list(args.max_count, args.fmt == "json")))
        if args.session_command == "delete":
            sys.exit(asyncio.run(cmd_session_delete(args.session_id)))
        return

    if args.command == "stats":
        from app.cli_session import cmd_stats

        sys.exit(asyncio.run(cmd_stats(args.days)))

    if args.command == "export":
        from app.cli_session import cmd_session_export

        sys.exit(asyncio.run(cmd_session_export(args.session_id)))

    if args.command == "import":
        from app.cli_session import cmd_session_import

        sys.exit(asyncio.run(cmd_session_import(args.path)))

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

    if args.command == "config":
        from app.cli_config import cmd_config_get, cmd_config_list, cmd_config_reset, cmd_config_set

        cmd = args.config_command
        if cmd == "list":
            cmd_config_list()
        elif cmd == "get":
            cmd_config_get(args.key)
        elif cmd == "set":
            cmd_config_set(args.key, args.value)
        elif cmd == "reset":
            cmd_config_reset()
        return

    if args.command == "tui":
        from app.tui.main import run_tui
        sys.exit(run_tui())

    if args.command == "init":
        from app.cli_init import cmd_init
        sys.exit(cmd_init(args.path))

    if args.command == "provider":
        import asyncio
        from app.cli_runtime import build_runtime

        async def _run_provider_cmd():
            runtime = build_runtime()
            try:
                cmd = args.provider_command
                if cmd == "list":
                    providers = runtime.router.list_providers()
                    print("Available providers:")
                    for p in providers:
                        current = " (current)" if str(p.provider_id) == runtime.provider_id else ""
                        print(f"  {p.provider_id}{current}")
                elif cmd == "use":
                    provider_name = args.name
                    if runtime.router.get_provider(provider_name):
                        # Update the runtime's default provider
                        runtime.provider_id = provider_name
                        print(f"Switched to provider: {provider_name}")
                    else:
                        print(f"Provider '{provider_name}' not available")
                        sys.exit(1)
                elif cmd == "models":
                    provider_name = args.provider or runtime.provider_id
                    models = await runtime.list_provider_models(provider_name)
                    if models:
                        print(f"Models for {provider_name}:")
                        for m in sorted(models):
                            print(f"  {m}")
                    else:
                        print(f"No models found for provider '{provider_name}'")
                else:
                    parser.print_help()
                    sys.exit(1)
            finally:
                await runtime.close()

        asyncio.run(_run_provider_cmd())
        return
