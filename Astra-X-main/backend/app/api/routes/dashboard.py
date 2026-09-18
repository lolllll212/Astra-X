# ruff: noqa: B008 — FastAPI Depends() in default args is intentional

"""Runtime dashboard endpoint.

Provides a unified view of learning system stats, provider performance,
and experience graph metrics for observability.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.agents.learning_store import DOMAIN_WEIGHTS
from app.api.dependencies import get_settings
from app.api.schemas.dashboard import (
    DashboardResponse,
    DomainWeightInfo,
    LearningStatsResponse,
    ProviderStatsEntry,
)
from app.config.settings import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get(
    "",
    response_model=DashboardResponse,
    summary="Runtime dashboard with learning and provider stats",
)
async def get_dashboard(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> DashboardResponse:
    """Return a snapshot of the runtime learning and provider systems."""
    # Learning stats.
    store = getattr(request.app.state, "learning_store", None)
    total_patterns = len(store._cache) if store is not None else 0  # type: ignore[union-attr]

    per_domain_weights: dict[str, DomainWeightInfo] = {}
    for domain, w in DOMAIN_WEIGHTS.items():
        per_domain_weights[domain.value if hasattr(domain, "value") else domain] = (
            DomainWeightInfo(**w)
        )

    anti_pattern_count = 0
    lm = getattr(request.app.state, "_learning_manager_global", None)
    if lm is not None:
        anti_pattern_count = len(lm._anti_patterns)  # type: ignore[union-attr]

    learning = LearningStatsResponse(
        total_patterns=total_patterns,
        per_domain_weights=per_domain_weights,
        anti_pattern_count=anti_pattern_count,
    )

    # Provider stats.
    mt = getattr(request.app.state, "metrics_tracker", None)
    provider_entries: list[ProviderStatsEntry] = []
    if mt is not None:
        for stats in mt.list_all_stats():  # type: ignore[union-attr]
            provider_entries.append(
                ProviderStatsEntry(
                    provider_id=stats.provider_id,
                    model_id=stats.model_id,
                    total_calls=stats.total_calls,
                    success_rate=stats.success_rate,
                    avg_latency_ms=round(stats.avg_latency_ms, 1),
                )
            )

    return DashboardResponse(
        learning=learning,
        providers=provider_entries,
    )
