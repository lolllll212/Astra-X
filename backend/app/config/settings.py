

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache

from pydantic import Field, SecretStr, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "Environment",
    "LogFormat",
    "LogLevel",
    "Settings",
    "get_settings",
]

_DEFAULT_DEV_SECRET_KEY = "dev-secret-key-change-this-in-production-please"
"""Sentinel placeholder. Production startup is rejected if this exact
value is still in use, preventing accidental deployment with a dev secret."""


class Environment(StrEnum):
    """The runtime environment the application is executing in.

    Used to gate environment-specific defaults and safety checks (e.g.
    refusing to start in production with debug mode enabled).
    """

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    """Supported structlog/stdlib logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(StrEnum):
    """Output format for structured logs.

    ``CONSOLE`` is human-readable and colorized, intended for local
    development. ``JSON`` emits machine-parsable structured logs, intended
    for production log aggregation.
    """

    CONSOLE = "console"
    JSON = "json"


def _split_csv(value: object) -> object:
    """Allow list-typed settings to be supplied as a comma-separated string.

    Environment variables are always strings, so a field declared as
    ``list[str]`` needs an explicit bridge from ``"a, b, c"`` to
    ``["a", "b", "c"]``. If ``value`` is already a list (e.g. supplied via
    a JSON array in the environment, which pydantic-settings also
    supports), it is returned unchanged.

    Args:
        value: The raw value provided for the field, before type coercion.

    Returns:
        A list of stripped, non-empty strings, or the original value if it
        was not a string (letting pydantic's normal coercion/validation
        handle it, including raising a clear error for genuinely invalid
        input).
    """
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


def _ensure_http_scheme(value: str | None, field_name: str) -> str | None:
    """Validate that an optional URL string uses an explicit http(s) scheme.

    Args:
        value: The candidate URL, or ``None`` if the setting is unset.
        field_name: The name of the field being validated, used only to
            produce a clear error message.

    Returns:
        The original value, unchanged, if it is ``None`` or valid.

    Raises:
        ValueError: If ``value`` is a non-empty string that does not start
            with ``http://`` or ``https://``.
    """
    if value is None:
        return None
    if not value.startswith(("http://", "https://")):
        raise ValueError(
            f"{field_name} must be an absolute URL starting with 'http://' "
            f"or 'https://', got: {value!r}"
        )
    return value


class Settings(BaseSettings):
    """Validated, environment-aware configuration for the Astra X backend.

    All fields are populated from environment variables prefixed with
    ``ASTRA_`` (case-insensitive), optionally loaded from a ``.env`` file
    in the working directory. Use :func:`get_settings` to obtain the
    process-wide singleton rather than instantiating this class directly.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ASTRA_",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
        validate_default=True,
    )

    # --- Application identity -------------------------------------------
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="The runtime environment, gating environment-specific validation.",
    )
    debug: bool = Field(
        default=False,
        description="Enables verbose diagnostics. Must be False in production.",
    )
    app_name: str = Field(
        default="Astra_X",
        min_length=1,
        description="Human-readable application name, used in logs and API metadata.",
    )
    app_version: str = Field(
        default="0.1.0",
        min_length=1,
        description="Semantic version of the running application.",
    )
    api_v1_prefix: str = Field(
        default="/api/v1",
        pattern=r"^/.+",
        description="URL prefix mounted for the version 1 HTTP API.",
    )

    # --- Server -----------------------------------------------------------
    host: str = Field(
        default="127.0.0.1",
        min_length=1,
        description="Interface the ASGI server binds to.",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="TCP port the ASGI server listens on.",
    )

    # --- Logging ----------------------------------------------------------
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Minimum severity of log records that are emitted.",
    )
    log_format: LogFormat = Field(
        default=LogFormat.CONSOLE,
        description="Structured log rendering format.",
    )

    # --- Security -----------------------------------------------------------
    secret_key: SecretStr = Field(
        default=SecretStr(_DEFAULT_DEV_SECRET_KEY),
        description=(
            "Cryptographic secret used for signing (e.g. sessions, tokens). "
            "Must be a 64-character hex string in production. "
            "Generate with: python -c \"import secrets; print(secrets.token_hex(32))\""
        ),
    )
    allowed_hosts: list[str] = Field(
        default_factory=lambda: ["*"],
        min_length=1,
        description=(
            "Hostnames this server will accept requests for. '*' (any host) "
            "is only permitted outside of production."
        ),
    )
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173"],
        min_length=1,
        description=(
            "Origins permitted to make cross-origin requests to the API. "
            "Wildcards are only permitted outside of production."
        ),
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="Whether CORS responses may include credentials (cookies, auth headers).",
    )
    rate_limit_requests_per_minute: int = Field(
        default=60,
        ge=1,
        description="Default request budget per client, per minute, for rate-limiting middleware.",
    )

    # --- Database -----------------------------------------------------------
    database_url: str = Field(
        default="sqlite+aiosqlite:///./astra_x.db",
        description=(
            "SQLAlchemy 2.x async database URL. Must use an async driver "
            "scheme (e.g. 'sqlite+aiosqlite' or 'postgresql+asyncpg')."
        ),
    )
    database_echo: bool = Field(
        default=False,
        description="Whether SQLAlchemy logs every emitted SQL statement. Development only.",
    )
    database_pool_size: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Connection pool size. Not used by SQLite; relevant once on PostgreSQL.",
    )

    # --- LLM providers -----------------------------------------------------------
    default_llm_provider: str = Field(
        default="ollama",
        min_length=1,
        description=(
            "Identifier of the provider adapter to use by default. This is an "
            "open string, not an enum, so new providers can be registered "
            "without modifying this settings module."
        ),
    )
    default_llm_model: str = Field(
        default="llama3.1",
        min_length=1,
        description="Model name passed to the default provider when none is specified per-request.",
    )
    llm_request_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        description="Maximum time to wait on a single LLM provider request before timing out.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of a local Ollama server.",
    )
    lm_studio_base_url: str = Field(
        default="http://localhost:1234/v1",
        description="Base URL of a local LM Studio OpenAI-compatible server.",
    )
    openai_compatible_base_url: str | None = Field(
        default=None,
        description="Base URL of a generic OpenAI-compatible API, if configured.",
    )
    openai_compatible_api_key: SecretStr | None = Field(
        default=None,
        description="API key for the generic OpenAI-compatible provider, if configured.",
    )

    # --- Embeddings -----------------------------------------------------------
    embedding_provider: str = Field(
        default="ollama",
        min_length=1,
        description="Provider for generating text embeddings ('ollama', 'openai', 'sentence_transformers').",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        min_length=1,
        description="Model name used for generating embeddings.",
    )
    embedding_dimensions: int = Field(
        default=768,
        ge=1,
        description="Dimensionality of embedding vectors.",
    )

    # --- Security hardening (Phase 1) ------------------------------------------
    sandbox_enabled: bool = Field(
        default=False,
        description="Enable the hardened Python execution sandbox.",
    )
    sandbox_timeout_seconds: int = Field(
        default=10,
        ge=1,
        le=120,
        description="Default timeout for sandboxed Python execution.",
    )
    sandbox_max_memory_mb: int = Field(
        default=256,
        ge=16,
        description="Maximum memory for sandboxed Python execution (MiB).",
    )
    sandbox_blocked_modules: list[str] = Field(
        default_factory=lambda: [
            "os", "subprocess", "shutil", "signal", "ctypes", "socket",
            "http", "urllib", "requests", "httpx", "pathlib", "tempfile",
        ],
        description="Modules blocked in the Python sandbox.",
    )
    sandbox_network_access: bool = Field(
        default=False,
        description="Whether sandboxed Python code can make network requests.",
    )
    sandbox_filesystem_access: bool = Field(
        default=False,
        description="Whether sandboxed Python code can read/write files.",
    )
    prompt_injection_detection_enabled: bool = Field(
        default=True,
        description="Enable prompt injection detection on user input.",
    )
    prompt_injection_block_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence threshold above which injection blocks the request (0-1).",
    )

    # --- Observability ------------------------------------------
    metrics_enabled: bool = Field(
        default=True,
        description="Enable Prometheus metrics endpoint and collection.",
    )
    metrics_prefix: str = Field(
        default="astra_x",
        min_length=1,
        description="Prefix applied to all Prometheus metric names.",
    )
    otel_service_name: str = Field(
        default="astra-x-backend",
        description="Service name reported to OpenTelemetry.",
    )
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None,
        description="OTLP exporter endpoint (e.g. http://otel-collector:4318). None disables OTel export.",
    )
    otel_traces_sampler_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of traces to sample when OTel is enabled (0.0-1.0).",
    )
    trace_propagation_enabled: bool = Field(
        default=True,
        description="Enable W3C traceparent header propagation.",
    )

    # --- Field validators -----------------------------------------------------------

    @field_validator("allowed_hosts", "cors_origins", mode="before")
    @classmethod
    def _coerce_csv_lists(cls, value: object) -> object:
        """Accept comma-separated strings for list-typed fields.

        Allows ``ASTRA_CORS_ORIGINS=http://a.com,http://b.com`` in a `.env`
        file rather than requiring JSON-array syntax.
        """
        return _split_csv(value)

    @field_validator("default_llm_provider", "default_llm_model")
    @classmethod
    def _normalize_identifier(cls, value: str) -> str:
        """Normalize provider/model identifiers to a consistent lowercase form."""
        return value.strip().lower()

    @field_validator("database_url")
    @classmethod
    def _validate_async_database_url(cls, value: str) -> str:
        """Ensure the database URL uses an async-capable SQLAlchemy driver scheme."""
        allowed_schemes = ("sqlite+aiosqlite://", "postgresql+asyncpg://")
        if not value.startswith(allowed_schemes):
            raise ValueError(
                "database_url must use an async driver scheme, one of "
                f"{allowed_schemes}, got: {value!r}"
            )
        return value

    @field_validator("ollama_base_url", "lm_studio_base_url")
    @classmethod
    def _validate_required_url(cls, value: str, info: ValidationInfo) -> str:
        """Validate required base-URL fields use an explicit http(s) scheme."""
        result = _ensure_http_scheme(value, info.field_name or "url")
        assert result is not None  # required field: value is never None here
        return result

    @field_validator("openai_compatible_base_url")
    @classmethod
    def _validate_optional_url(cls, value: str | None, info: ValidationInfo) -> str | None:
        """Validate the optional OpenAI-compatible base URL, if one is set."""
        return _ensure_http_scheme(value, info.field_name or "url")

    # --- _FILE env var support (Docker Secrets / Vault agent) ---------------------------

    @model_validator(mode="before")
    @classmethod
    def _resolve_secret_files(cls, values: dict[str, object]) -> dict[str, object]:
        """Override sensitive fields from ``<FIELD>_FILE`` env vars.

        If ``ASTRA_SECRET_KEY_FILE`` is set (Docker secrets / Vault agent
        pattern), the file is read and its content replaces the inline
        ``secret_key`` value. This keeps credentials out of environment
        blocks and compose files.
        """
        for env_suffix, field_name in [("SECRET_KEY_FILE", "secret_key")]:
            file_path = os.environ.get(f"ASTRA_{env_suffix}")
            if not file_path:
                continue
            try:
                with open(file_path, encoding="utf-8") as f:
                    value = f.read().strip()
            except (FileNotFoundError, PermissionError, OSError) as exc:
                raise ValueError(
                    f"Cannot read secret file {file_path!r} "
                    f"(set via ASTRA_{env_suffix}): {exc}"
                ) from exc
            if not value:
                raise ValueError(
                    f"Secret file {file_path!r} "
                    f"(set via ASTRA_{env_suffix}) is empty"
                )
            values[field_name] = value
        return values

    # --- Cross-field, production-safety validation -----------------------------------------

    @model_validator(mode="after")
    def _enforce_production_safety(self) -> Settings:
        """Reject unsafe configuration combinations when running in production.

        These checks intentionally depend on more than one field, which is
        why they live here rather than in a single-field validator.

        Raises:
            ValueError: If any production-safety invariant is violated.
        """
        if self.environment is not Environment.PRODUCTION:
            return self

        violations: list[str] = []

        if self.debug:
            violations.append("debug must be False in production")

        if self.secret_key.get_secret_value() == _DEFAULT_DEV_SECRET_KEY:
            violations.append("secret_key must be overridden from its development default")

        secret_value = self.secret_key.get_secret_value()
        if len(secret_value) < 64:
            violations.append(
                "secret_key must be a 64-character hex string in production "
                "(generate with: python -c \"import secrets; print(secrets.token_hex(32))\")"
            )
        elif not all(c in "0123456789abcdef" for c in secret_value):
            violations.append("secret_key must be a valid 64-character hex string in production")

        if "*" in self.allowed_hosts:
            violations.append("allowed_hosts must not contain '*' in production")

        if "*" in self.cors_origins:
            violations.append("cors_origins must not contain '*' in production")

        if self.database_echo:
            violations.append("database_echo must be False in production")

        if violations:
            joined = "; ".join(violations)
            raise ValueError(f"Unsafe production configuration: {joined}")

        return self

    # --- Convenience accessors -----------------------------------------------------------

    @property
    def is_development(self) -> bool:
        """Whether the application is running in the development environment."""
        return self.environment is Environment.DEVELOPMENT

    @property
    def is_testing(self) -> bool:
        """Whether the application is running in the testing environment."""
        return self.environment is Environment.TESTING

    @property
    def is_production(self) -> bool:
        """Whether the application is running in the production environment."""
        return self.environment is Environment.PRODUCTION


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton.

    The first call constructs and validates a :class:`Settings` instance
    from the environment and ``.env`` file; every subsequent call returns
    the same cached, immutable instance, avoiding repeated environment
    parsing and guaranteeing a single consistent configuration view across
    the application.

    In tests that need to exercise different configurations, call
    ``get_settings.cache_clear()`` before constructing a new instance
    (typically with environment variables monkeypatched beforehand).

    Returns:
        The validated, immutable application settings.
    """
    return Settings()

