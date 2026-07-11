from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Annotated, Any, ClassVar

from pydantic import (
    BeforeValidator,
    Field,
    SecretStr,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from pydantic_settings.sources import DotEnvSettingsSource


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


_DEFAULT_DEV_SECRET_KEY = "dev-secret-key-change-this-in-production-please"
"""Sentinel placeholder. Production startup is rejected if this exact
value is still in use, preventing accidental deployment with a dev secret."""


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


class CsvDotEnvSettingsSource(DotEnvSettingsSource):
    """Custom .env file settings source that treats list fields as CSV, not JSON.

    The default DotEnvSettingsSource tries to parse values that look like
    lists/dicts as JSON. This breaks comma-separated values in .env files.
    This subclass overrides that behavior for known CSV list fields.
    """

    CSV_LIST_FIELDS: ClassVar[set[str]] = {"allowed_hosts", "cors_origins"}

    def decode_complex_value(
        self,
        field_name: str,
        field: Any,
        value: str,
    ) -> Any:
        # For CSV list fields, return the raw string to be parsed by BeforeValidator
        if field_name in self.CSV_LIST_FIELDS:
            return value
        # For other fields, use default JSON parsing
        return super().decode_complex_value(field_name, field, value)


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

    # --- Custom settings source to handle CSV lists in .env files ---
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            CsvDotEnvSettingsSource(settings_cls),
            file_secret_settings,
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
        description="Human-readable application name.",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Semantic version string.",
    )
    api_v1_prefix: str = Field(
        default="/api/v1",
        description="URL prefix for the v1 HTTP API.",
    )

    # --- Server -----------------------------------------------------------
    host: str = Field(
        default="127.0.0.1",
        description="Interface to bind the HTTP server to.",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="TCP port to listen on.",
    )

    # --- Logging ----------------------------------------------------------
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Minimum log severity: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
    )
    log_format: LogFormat = Field(
        default=LogFormat.CONSOLE,
        description="Structured log rendering format.",
    )
    trace_propagation: bool = Field(
        default=True,
        description="Enable W3C traceparent header propagation.",
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
    allowed_hosts: Annotated[
        list[str],
        BeforeValidator(_split_csv),
        Field(
            default_factory=lambda: ["*"],
            min_length=1,
            description=(
                "Hostnames this server will accept requests for. '*' (any host) "
                "is only permitted outside of production."
            ),
        ),
    ]
    cors_origins: Annotated[
        list[str],
        BeforeValidator(_split_csv),
        Field(
            default_factory=lambda: ["http://localhost:5173"],
            min_length=1,
            description=(
                "Origins permitted to make cross-origin requests to the API. "
                "Wildcards are only permitted outside of production."
            ),
        ),
    ]
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
        description="Base URL for local Ollama server.",
    )
    lm_studio_base_url: str = Field(
        default="http://localhost:1234/v1",
        description="Base URL for local LM Studio server.",
    )
    openai_compatible_base_url: str | None = Field(
        default=None,
        description="Base URL for generic OpenAI-compatible API (optional).",
    )
    openai_compatible_api_key: str | None = Field(
        default=None,
        description="API key for OpenAI-compatible provider (optional).",
    )

    # --- Embeddings ---
    embedding_provider: str = Field(
        default="ollama",
        min_length=1,
        description="Embedding provider: ollama, openai, sentence_transformers",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        min_length=1,
        description="Embedding model name",
    )
    embedding_dimensions: int = Field(
        default=768,
        ge=1,
        description="Embedding vector dimensionality",
    )

    # --- Feature flags ---
    enable_background_tasks: bool = Field(
        default=True,
        description="Enable background task processing.",
    )
    enable_reflection: bool = Field(
        default=True,
        description="Enable post-execution reflection step.",
    )
    max_iterations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum plan→execute→reflect cycles before forced completion.",
    )

    # --- Field validators -----------------------------------------------------------

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

    @field_validator("openai_compatible_base_url", "ollama_base_url", "lm_studio_base_url", mode="before")
    @classmethod
    def _validate_provider_url(cls, value: object, info: ValidationInfo) -> object:
        """Validate that provider URLs have explicit http(s) scheme."""
        return _ensure_http_scheme(value, info.field_name)

    @field_validator("embedding_model", mode="before")
    @classmethod
    def _validate_embedding_model(cls, value: str) -> str:
        """Normalize embedding model identifier."""
        return value.strip().lower()

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
        if self.environment is Environment.PRODUCTION and (
            len(secret_value) != 64
            or not all(c in "0123456789abcdef" for c in secret_value)
        ):
            violations.append(
                    "secret_key must be a 64-character hex string in production "
                    "(generate with: python -c \"import secrets; print(secrets.token_hex(32))\")"
                )

        if "*" in self.allowed_hosts:
            violations.append("allowed_hosts must not contain '*' in production")

        if "*" in self.cors_origins:
            violations.append("cors_origins must not contain '*' in production")

        if violations:
            raise ValueError("Production safety violations: " + "; ".join(violations))

        return self


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


_DEFAULT_DEV_SECRET_KEY = "dev-secret-key-change-this-in-production-please"
"""Sentinel placeholder. Production startup is rejected if this exact
value is still in use, preventing accidental deployment with a dev secret."""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide, memoised Settings singleton.

    The cache ensures environment variables and ``.env`` are only parsed
    once, which is critical for performance in long-running processes.
    """
    return Settings()


__all__ = [
    "Environment",
    "LogFormat",
    "LogLevel",
    "Settings",
    "get_settings",
]
