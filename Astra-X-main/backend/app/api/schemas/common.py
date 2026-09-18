"""Shared API schemas for pagination, metadata, and common types."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PaginationParams(BaseModel):
    """Query parameters for paginated list endpoints.

    Attributes:
        page: The 1-indexed page number.
        page_size: Number of items per page (capped server-side).
        sort: Field to sort by.
        order: Sort direction.
    """

    model_config = ConfigDict(extra="forbid")

    page: int = Field(default=1, ge=1, description="Page number (1-indexed).")
    page_size: int = Field(default=50, ge=1, le=200, description="Items per page.")
    sort: str = Field(default="created_at", description="Sort field.")
    order: str = Field(default="desc", pattern=r"^(asc|desc)$", description="Sort direction.")


class PaginatedResponse(BaseModel):
    """Wrapper for paginated list responses.

    Attributes:
        items: The page of results.
        total: Total number of items across all pages.
        page: The current page number.
        page_size: The number of items per page.
        pages: Total number of pages.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    items: list[object] = Field(description="Page of results.")
    total: int = Field(ge=0, description="Total items across all pages.")
    page: int = Field(ge=1, description="Current page number.")
    page_size: int = Field(ge=1, description="Items per page.")
    pages: int = Field(ge=0, description="Total number of pages.")


class MessageResponse(BaseModel):
    """A single API response message wrapper.

    Used for endpoints that return a simple success message rather than
    a resource.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    message: str = Field(description="Human-readable response message.")
