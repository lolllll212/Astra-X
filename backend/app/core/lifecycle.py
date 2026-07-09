"""Application lifespan orchestration for Astra X.

This module owns the FastAPI ``lifespan`` async context manager — the single
entry point called when the ASGI server starts and stops. Every
initialisation and teardown step is explicitly sequenced, logged, and
guarded so that a failure at any point produces a clear, structured log
message rather than a silent partial start.

Modules that need to execute custom logic at startup or shutdown (e.g.
memory stores, agent/tool registries, schedulers, metrics collectors)
may use :func:`register_startup_hook` and :func:`register_shutdown_hook`
at module level. The registered callables receive the fully-initialised
container (``object``) and are awaited in registration order.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.core.container import Container

from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config.settings import Settings, get_settings
from app.core.exceptions import ConfigurationError
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public hook registry — allows modules to register lifecycle callbacks
# without coupling to this module or to each other.
# ---------------------------------------------------------------------------

StartupHook = Callable[..., Coroutine[Any, Any, None]]
"""Type alias for a startup hook: an async callable that receives the
fully-initialised dependency injection container."""

ShutdownHook = Callable[..., Coroutine[Any, Any, None]]
"""Type alias for a shutdown hook: an async callable that receives the
fully-initialised dependency injection container."""

_startup_hooks: list[StartupHook] = []
_shutdown_hooks: list[ShutdownHook] = []


def register_startup_hook(hook: StartupHook) -> None:
    """Register an async callable to be executed during application startup.

    Hooks are executed in registration order after the database engine has
    been verified and the dependency injection container is ready.

    Args:
        hook: An async callable accepting a :class:`Container` instance.
    """
    _startup_hooks.append(hook)


def register_shutdown_hook(hook: ShutdownHook) -> None:
    """Register an async callable to be executed during application shutdown.

    Hooks are executed in registration order before the container is torn
    down and the database engine is disposed.

    Args:
        hook: An async callable accepting a :class:`Container` instance.
    """
    _shutdown_hooks.append(hook)


# ---------------------------------------------------------------------------
# Internal helpers — one function per startup / shutdown step
# ---------------------------------------------------------------------------


def _build_settings() -> Settings:
    """Load and validate application configuration.

    Returns:
        The validated, immutable settings singleton.

    Raises:
        ConfigurationError: If settings validation fails.
    """
    logger.info("startup.load_settings")
    try:
        settings = get_settings()
    except Exception as exc:
        logger.critical("startup.settings_failed", error=str(exc))
        raise ConfigurationError(
            message="Failed to load application settings.",
            cause=exc,
        ) from exc

    logger.info(
        "startup.settings_loaded",
        environment=settings.environment.value,
        app_version=settings.app_version,
    )
    return settings


def _configure_logging(settings: Settings) -> None:
    """Wire up structlog and the stdlib root logger.

    Args:
        settings: Application settings providing logging configuration.
    """
    logger.info("startup.configure_logging")
    try:
        configure_logging(settings)
    except Exception as exc:
        logger.critical("startup.logging_failed", error=str(exc))
        raise ConfigurationError(
            message="Failed to configure logging.",
            cause=exc,
        ) from exc

    logger.info(
        "startup.logging_configured",
        log_format=settings.log_format.value,
        log_level=settings.log_level.value,
    )


async def _build_container(settings: Settings) -> Container | None:
    """Construct and wire the dependency injection container.

    Args:
        settings: Application settings used to configure container resources.

    Returns:
        A fully-wired :class:`Container` instance.

    Raises:
        ConfigurationError: If container initialisation fails.
    """
    logger.info("startup.build_container")
    from app.core.container import Container

    container = Container()
    container.from_pydantic(settings)
    container.wire(modules=[])
    logger.info("startup.container_built")
    return container


async def _build_database_engine(settings: Settings) -> AsyncEngine:
    """Create the SQLAlchemy async database engine.

    Args:
        settings: Application settings providing the database URL and pool
            configuration.

    Returns:
        A configured, but not yet verified, :class:`AsyncEngine` instance.

    Raises:
        ConfigurationError: If engine creation fails.
    """
    logger.info(
        "startup.build_engine",
        database_url=settings.database_url,
        pool_size=settings.database_pool_size,
        echo=settings.database_echo,
    )
    try:
        engine: AsyncEngine = create_async_engine(
            url=settings.database_url,
            echo=settings.database_echo,
            pool_size=settings.database_pool_size,
            max_overflow=10,
            pool_pre_ping=True,
        )
    except Exception as exc:
        logger.critical("startup.engine_failed", error=str(exc))
        raise ConfigurationError(
            message="Failed to create database engine.",
            cause=exc,
        ) from exc

    logger.info("startup.engine_created")
    return engine


async def _verify_database_connectivity(engine: AsyncEngine) -> None:
    """Verify the database is reachable by executing a lightweight query.

    Args:
        engine: The database engine to check.

    Raises:
        ConfigurationError: If the database connection cannot be established.
    """
    logger.info("startup.verify_database")
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("startup.database_verified")
    except Exception as exc:
        logger.critical("startup.database_unreachable", error=str(exc))
        raise ConfigurationError(
            message="Database is unreachable.",
            cause=exc,
        ) from exc


async def _run_startup_hooks(container: Container | None) -> None:
    """Execute all registered startup hooks in order.

    Each hook receives the fully-initialised container so it can resolve
    its own dependencies.

    Args:
        container: The dependency injection container.

    Raises:
        ConfigurationError: If any hook raises an exception.
    """
    for hook in _startup_hooks:
        hook_name = getattr(hook, "__name__", str(hook))
        logger.info("startup.run_hook", hook=hook_name)
        try:
            await hook(container)
        except Exception as exc:
            logger.critical("startup.hook_failed", hook=hook_name, error=str(exc))
            raise ConfigurationError(
                message=f"Startup hook '{hook_name}' failed.",
                cause=exc,
            ) from exc


async def _run_shutdown_hooks(container: Container | None) -> None:
    """Execute all registered shutdown hooks in order.

    Failures are logged but do not prevent subsequent hooks from running
    so that teardown is as complete as possible.

    Args:
        container: The dependency injection container.
    """
    for hook in _shutdown_hooks:
        hook_name = getattr(hook, "__name__", str(hook))
        logger.info("shutdown.run_hook", hook=hook_name)
        try:
            await hook(container)
        except Exception as exc:
            logger.error("shutdown.hook_failed", hook=hook_name, error=str(exc))


async def _dispose_database_engine(engine: AsyncEngine | None) -> None:
    """Safely dispose the database engine, releasing all connections.

    Args:
        engine: The database engine to dispose, or ``None`` if it was
            never initialised.
    """
    if engine is None:
        return
    logger.info("shutdown.dispose_engine")
    try:
        await engine.dispose()
        logger.info("shutdown.engine_disposed")
    except Exception as exc:
        logger.warning("shutdown.engine_dispose_failed", error=str(exc))


async def _shutdown_container(container: Container | None) -> None:
    """Shut down the dependency injection container.

    Args:
        container: The container to shut down, or ``None`` if it was
            never initialised.
    """
    if container is None:
        return
    logger.info("shutdown.shutdown_container")
    try:
        container.shutdown_resources()
        logger.info("shutdown.container_shut_down")
    except Exception as exc:
        logger.warning("shutdown.container_shutdown_failed", error=str(exc))


# ---------------------------------------------------------------------------
# FastAPI lifespan — wired into the application factory via main.py
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Orchestrate application startup and shutdown.

    This async context manager is passed as the ``lifespan`` parameter to
    the FastAPI constructor. It is the single place where the application's
    lifecycle is sequenced:

    **Startup**
    1. Load and validate :class:`app.config.settings.Settings`.
    2. Configure structlog and stdlib logging.
    3. Build and wire the dependency injection :class:`Container`.
    4. Create the SQLAlchemy :class:`AsyncEngine`.
    5. Verify database connectivity with ``SELECT 1``.
    6. Execute any :func:`register_startup_hook` callbacks.

    **Shutdown**
    1. Execute any :func:`register_shutdown_hook` callbacks.
    2. Shut down the dependency injection container.
    3. Dispose the database engine.

    Initialised components are stored on ``app.state`` for access by
    middleware and route handlers:

    * ``app.state.settings``
    * ``app.state.container``
    * ``app.state.db_engine``

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to the ASGI server, allowing request handling.
    """
    # ------------------------------------------------------------------ #
    # Startup
    # ------------------------------------------------------------------ #
    start_time: float = time.monotonic()

    settings = _build_settings()
    _configure_logging(settings)

    logger.info("startup.begin", environment=settings.environment.value)

    container = await _build_container(settings)
    assert container is not None
    engine = await _build_database_engine(settings)

    import importlib
    importlib.import_module("app.database.models")

    from app.database.base import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await _verify_database_connectivity(engine)
    await _run_startup_hooks(container)

    app.state.settings = settings
    app.state.container = container
    app.state.db_engine = engine

    # Initialise the LLM router and load enabled providers from the DB.
    from app.database.repositories.provider_repository import ProviderRepository
    from app.database.session import create_session_factory, session_context
    from app.llm.router import LLMRouter

    llm_router = LLMRouter(
        settings=settings,
        default_provider_id=settings.default_llm_provider,
    )

    try:
        session_factory = create_session_factory(engine)
        async with session_context(session_factory) as db_session:
            provider_repo = ProviderRepository(db_session)
            specs = await provider_repo.list_enabled()
            if specs:
                await llm_router.load_providers_from_db(specs)
                logger.info(
                    "startup.providers_loaded",
                    count=len(specs),
                )
            else:
                logger.info("startup.no_providers_found")
    except Exception as exc:
        logger.warning("startup.load_providers_failed", error=str(exc))

    app.state.llm_router = llm_router

    # Create the adaptive model selector.
    from app.llm.model_selector import ModelSelector

    model_selector = ModelSelector.from_router(
        router=llm_router,
        default_model=settings.default_llm_model,
        default_provider=settings.default_llm_provider,
    )
    app.state.model_selector = model_selector

    # Initialise the tool registry and capability registry.
    from app.tools.builtin import (
        CalculatorTool,
        DateTimeTool,
        JsonTool,
        TextTool,
        UuidTool,
    )
    from app.tools.capabilities import CapabilityRegistry
    from app.tools.registry import ToolRegistry

    tool_registry = ToolRegistry()
    for builtin_tool_cls in (CalculatorTool, DateTimeTool, JsonTool, TextTool, UuidTool):
        try:
            tool_registry.register(builtin_tool_cls())
        except Exception as exc:
            logger.warning(
                "startup.builtin_tool_register_failed",
                tool=builtin_tool_cls.__name__,
                error=str(exc),
            )
    capability_registry = CapabilityRegistry(tool_registry)
    app.state.tool_registry = tool_registry
    app.state.capability_registry = capability_registry
    logger.info(
        "startup.tool_registry_initialised",
        count=tool_registry.count,
    )

    # Set up ProviderService.
    from app.services.provider_service import ProviderService

    provider_service = ProviderService(
        repository=ProviderRepository.__new__(ProviderRepository),
        llm_router=llm_router,
        settings=settings,
    )

    # Load enabled plugins from the database.
    from app.database.repositories.plugin_repository import PluginRepository
    from app.plugin.manager import PluginManager

    try:
        session_factory = create_session_factory(engine)
        async with session_context(session_factory) as db_session:
            plugin_repo = PluginRepository(db_session)
            plugin_manager = PluginManager(
                plugin_repository=plugin_repo,
                capability_registry=capability_registry,
                app_version=settings.app_version,
                tool_registry=tool_registry,
                provider_service=provider_service,
            )
            loaded_count = await plugin_manager.load_enabled()
            if loaded_count:
                logger.info(
                    "startup.plugins_loaded",
                    count=loaded_count,
                )
            else:
                logger.info("startup.no_plugins_found")
    except Exception as exc:
        logger.warning("startup.load_plugins_failed", error=str(exc))
        plugin_manager = PluginManager(
            plugin_repository=PluginRepository.__new__(PluginRepository),
            app_version=settings.app_version,
        )

    app.state.plugin_manager = plugin_manager

    # Initialise observability.
    _init_observability(settings, app)

    # Initialise the memory pipeline.
    from app.memory import SQLiteVectorStore
    from app.memory.manager import MemoryManager
    from app.memory.providers import OllamaEmbedder

    vector_store = SQLiteVectorStore()
    embedder = OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
    )
    memory_manager = MemoryManager(
        vector_store=vector_store,
        embedder=embedder,
    )
    container.embedder = embedder
    container.memory_manager = memory_manager
    app.state.memory_manager = memory_manager
    logger.info(
        "startup.memory_initialised",
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
    )

    elapsed: float = time.monotonic() - start_time
    logger.info("startup.complete", elapsed_ms=round(elapsed * 1000))

    yield

    # ------------------------------------------------------------------ #
    # Shutdown
    # ------------------------------------------------------------------ #
    shutdown_start: float = time.monotonic()
    logger.info("shutdown.begin")

    # Unload all plugins.
    pm = getattr(app.state, "plugin_manager", None)
    if pm is not None:
        try:
            await pm.unload_all()
        except Exception as exc:
            logger.warning("shutdown.unload_plugins_failed", error=str(exc))

    await _run_shutdown_hooks(container)
    _shutdown_observability()
    await _shutdown_container(container)
    await _dispose_database_engine(engine)

    elapsed_shutdown: float = time.monotonic() - shutdown_start
    logger.info("shutdown.complete", elapsed_ms=round(elapsed_shutdown * 1000))


def _init_observability(settings: Settings, app: FastAPI) -> None:
    """Initialise Prometheus metadata and optional OpenTelemetry.

    This is called during application startup, after the LLM router has
    been initialised but before the memory pipeline.
    """
    try:
        from app.observability import app_info, metrics_registry

        app_info.info({
            "version": settings.app_version,
            "environment": settings.environment.value,
        })

        if settings.metrics_enabled:
            logger.info(
                "observability.prometheus_enabled",
                prefix=settings.metrics_prefix,
            )

        from app.observability.opentelemetry import is_otel_available, setup_otel

        if is_otel_available() and settings.otel_exporter_otlp_endpoint:
            otel_ok = setup_otel(settings)
            if otel_ok:
                app.state._otel_initialized = True
                logger.info("observability.opentelemetry_enabled")
    except ImportError:
        logger.info("observability.prometheus_client_not_installed")
    except Exception as exc:
        logger.warning("observability.init_failed", error=str(exc))


def _shutdown_observability() -> None:
    """Flush and shut down OpenTelemetry (if initialised)."""
    try:
        from app.observability.opentelemetry import shutdown_otel

        shutdown_otel()
    except Exception as exc:
        logger.warning("observability.shutdown_failed", error=str(exc))
