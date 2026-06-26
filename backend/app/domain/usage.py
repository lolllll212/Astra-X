"""Token usage domain models.

Defines the value objects that capture token consumption and cost
tracking for LLM invocations. These models are attached to messages and
stream completion events for observability and billing.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Usage",
]


class Usage(BaseModel):
    """Immutable token usage summary for a single LLM invocation.

    All counts are cumulative over the entire generation, including any
    internal retries or speculative decoding the provider may perform.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: int = Field(
        default=0,
        ge=0,
        description="Number of tokens in the prompt (input).",
    )
    completion_tokens: int = Field(
        default=0,
        ge=0,
        description="Number of tokens generated (output).",
    )
    total_tokens: int = Field(
        default=0,
        ge=0,
        description="Total tokens consumed (prompt + completion).",
    )
    prompt_tokens_details: dict[str, int] | None = Field(
        default=None,
        description="Provider-specific breakdown of prompt tokens (e.g. cached, audio).",
    )
    completion_tokens_details: dict[str, int] | None = Field(
        default=None,
        description="Provider-specific breakdown of completion tokens (e.g. reasoning).",
    )
    cost_usd: float | None = Field(
        default=None,
        ge=0,
        description="Estimated monetary cost in USD, if rate information is available.",
    )
