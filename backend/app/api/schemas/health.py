"""Health-check response schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Response body for the root health-check endpoint.

    Attributes:
        status: Overall application health (``"healthy"`` or ``"unhealthy"``).
        database: Whether the database connection is responsive.
        providers: Whether at least one LLM provider is reachable.
        uptime: Application uptime in seconds.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str = Field(description="Overall health status.")
    database: bool = Field(description="Database connectivity.")
    providers: bool = Field(description="Provider availability.")
    uptime: float = Field(ge=0, description="Uptime in seconds.")
