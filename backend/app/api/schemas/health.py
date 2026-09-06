"""Health-check response schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Response body for the root health-check endpoint.

    Attributes:
        status: Overall application health (``"healthy"`` or ``"unhealthy"``).
        database: Whether the database connection is responsive.
        providers: Provider availability. ``None`` when no provider is
            configured (not unhealthy — just absent).
        uptime: Application uptime in seconds.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str = Field(description="Overall health status.")
    database: bool = Field(description="Database connectivity.")
    providers: bool | None = Field(
        default=None,
        description="Provider availability. ``None`` if no providers configured.",
    )
    uptime: float = Field(ge=0, description="Uptime in seconds.")
