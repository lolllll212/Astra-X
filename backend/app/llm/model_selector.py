"""Adaptive model selection based on task capability profiles.

The :class:`ModelSelector` matches a :class:`CapabilityProfile` (what a task
needs) to the best available provider/model (what the system has) by scoring
each registered provider against the profile and returning the highest-score
match.

Usage::

    selector = ModelSelector.from_router(router)
    model, provider = await selector.select(
        CapabilityProfile(requires_coding=True, reasoning="deep"),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.llm.provider_metrics import ProviderMetricsTracker
    from app.llm.router import LLMRouter
    from app.memory.experience_graph import ExperienceGraph

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Model profiles — describe what each model is good at
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PROFILES: dict[str, dict[str, object]] = {
    "deepseek-coder": {"strengths": ["coding"], "context_length": 32768, "reasoning": True},
    "deepseek-r1": {"strengths": ["reasoning", "coding"], "context_length": 32768},
    "qwen2.5": {"strengths": ["reasoning"], "context_length": 32768},
    "qwen2.5-coder": {"strengths": ["coding", "reasoning"], "context_length": 32768},
    "llama3.1": {"strengths": ["balanced", "fast"], "context_length": 8192},
    "llama3.2": {"strengths": ["balanced", "fast"], "context_length": 8192},
    "mistral-nemo": {"strengths": ["large_context", "reasoning"], "context_length": 128000},
    "mistral-small": {"strengths": ["balanced", "fast"], "context_length": 32768},
    "codellama": {"strengths": ["coding"], "context_length": 16384},
    "phi3": {"strengths": ["fast", "small"], "context_length": 4096},
    "gemma2": {"strengths": ["balanced", "fast"], "context_length": 8192},
    "llava": {"strengths": ["vision"], "context_length": 4096, "vision": True},
}


@dataclass(frozen=True)
class CapabilityProfile:
    """Describes what a task needs from the model.

    Attributes:
        requires_coding: Task involves code generation or analysis.
        reasoning: Depth of reasoning needed (``"none"``, ``"low"``,
            ``"medium"``, ``"deep"``).
        prefers_speed: Task benefits from fast token generation (e.g.
            simple text transformation).
        prefers_large_context: Task needs a large context window.
        requires_vision: Task involves image understanding.
        domain: Optional task domain (``"coding"``, ``"research"``, …)
            for adaptive per-domain provider selection.
    """

    requires_coding: bool = False
    reasoning: str = "none"
    prefers_speed: bool = False
    prefers_large_context: bool = False
    requires_vision: bool = False
    domain: str = ""


@dataclass
class ModelProfile:
    """Describes a model's capabilities.

    Attributes:
        model_id: The model identifier.
        strengths: List of strength labels (``"coding"``, ``"reasoning"``,
            ``"fast"``, ``"balanced"``, ``"large_context"``, ``"vision"``).
        context_length: Maximum context window in tokens.
    """

    model_id: str
    strengths: list[str] = field(default_factory=list)
    context_length: int = 4096


# ---------------------------------------------------------------------------
# ModelSelector
# ---------------------------------------------------------------------------


class ModelSelector:
    """Selects the best model/provider for a given capability profile.

    Usage::

        selector = ModelSelector(
            router=router,
            model_profiles={
                "deepseek-coder": {"strengths": ["coding"], "context_length": 32768},
            },
        )
        model_id, provider_id = await selector.select(
            CapabilityProfile(requires_coding=True),
        )
    """

    def __init__(
        self,
        router: LLMRouter,
        model_profiles: dict[str, dict[str, object]] | None = None,
        default_model: str = "llama3.1",
        default_provider: str | None = None,
        metrics_tracker: ProviderMetricsTracker | None = None,
        experience_graph: ExperienceGraph | None = None,
        performance_weight: float = 0.20,
        domain_weight: float = 0.30,
    ) -> None:
        self._router = router
        self._model_profiles: dict[str, ModelProfile] = self._build_profiles(
            model_profiles or DEFAULT_MODEL_PROFILES,
        )
        self._default_model = default_model
        self._default_provider = default_provider
        self._metrics_tracker = metrics_tracker
        self._experience_graph = experience_graph
        # How much to weigh historical performance vs capability profile.
        self._performance_weight = performance_weight
        # How much to weigh domain-specific experience data.
        self._domain_weight = domain_weight

    @classmethod
    def from_router(
        cls,
        router: LLMRouter,
        default_model: str = "llama3.1",
        default_provider: str | None = None,
        metrics_tracker: ProviderMetricsTracker | None = None,
        experience_graph: ExperienceGraph | None = None,
    ) -> ModelSelector:
        """Create a selector linked to a router with default model profiles.

        Args:
            router: The LLM router to query for registered providers.
            default_model: Fallback model when no provider matches.
            default_provider: Fallback provider when no match is found.
            metrics_tracker: Optional performance tracker for data-driven scoring.
            experience_graph: Optional experience graph for domain-aware selection.

        Returns:
            A new ModelSelector instance.
        """
        return cls(
            router=router,
            model_profiles=None,
            default_model=default_model,
            default_provider=default_provider,
            metrics_tracker=metrics_tracker,
            experience_graph=experience_graph,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def select(
        self,
        profile: CapabilityProfile,
    ) -> tuple[str, str | None]:
        """Select the best (model_id, provider_id) for a capability profile.

        Providers with open circuit breakers are excluded. If multiple
        providers serve the same model, the first healthy one wins.

        When a ``domain`` is set on the profile and an ``experience_graph``
        is available, domain-specific provider performance data is blended
        into the scoring so the router learns which providers excel at
        which domains.

        Args:
            profile: What the task needs from the model.

        Returns:
            A ``(model_id, provider_id)`` tuple. The provider may be
            ``None`` if no registered provider matches, in which case
            the default model is returned.
        """
        candidates = self._score_providers(profile)
        if candidates:
            model_id, provider_id = candidates[0]
            logger.debug(
                "model_selector.selected",
                model=model_id,
                provider=provider_id,
                profile=str(profile),
            )
            return model_id, provider_id

        logger.info(
            "model_selector.fallback",
            model=self._default_model,
            provider=self._default_provider,
        )
        return self._default_model, self._default_provider

    async def select_for_planning(
        self,
        domain: str = "",
    ) -> tuple[str, str | None]:
        """Select a model optimised for planning (reasoning-heavy).

        Args:
            domain: Optional domain for adaptive per-domain selection.

        Returns:
            A ``(model_id, provider_id)`` tuple.
        """
        return await self.select(
            CapabilityProfile(reasoning="deep", domain=domain),
        )

    async def select_for_reflection(
        self,
        domain: str = "",
    ) -> tuple[str, str | None]:
        """Select a model optimised for reflection (fast, balanced).

        Args:
            domain: Optional domain for adaptive per-domain selection.

        Returns:
            A ``(model_id, provider_id)`` tuple.
        """
        return await self.select(
            CapabilityProfile(prefers_speed=True, domain=domain),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _score_providers(
        self,
        profile: CapabilityProfile,
    ) -> list[tuple[str, str | None]]:
        """Score all registered providers against a profile, sorted best first.

        Args:
            profile: The capability profile to match against.

        Returns:
            A list of ``(model_id, provider_id)`` tuples sorted by
            descending score.
        """
        # Pre-load domain-specific experience stats once.
        domain_stats = self._load_domain_stats(profile.domain)

        scored: list[tuple[float, str, str | None]] = []

        for provider in self._router.list_providers():
            mid = provider.model_id
            pid = provider.provider_id

            if self._router.is_provider_available(pid):
                mp = self._model_profiles.get(mid)
                score = self._score(
                    profile, mp, mid, provider_id=pid,
                    domain_stats=domain_stats,
                )
                scored.append((score, mid, pid))

        # Sort by score descending, then by model name for stability.
        scored.sort(key=lambda x: (-x[0], x[1]))

        return [(mid, pid) for _, mid, pid in scored]

    def _score(
        self,
        profile: CapabilityProfile,
        mp: ModelProfile | None,
        model_id: str,
        provider_id: str | None = None,
        domain_stats: list[dict[str, Any]] | None = None,
    ) -> float:
        """Score a single model/profile combination.

        Blends three signals:

        1. **Capability profile** — what the task needs vs model strengths.
        2. **Historical metrics** — ``ProviderMetricsTracker`` success rate,
           latency, and reflection confidence.
        3. **Domain experience** — ``ExperienceGraph`` per-domain, per-provider
           outcomes, weighted by reflection and cost efficiency.

        Args:
            profile: What the task needs.
            mp: The model's profile (``None`` if unknown).
            model_id: The model identifier (used for fallback scoring).
            provider_id: The provider serving this model (for performance lookup).
            domain_stats: Pre-loaded domain stats from experience graph.

        Returns:
            A score where higher is better. Returns ``-float("inf")``
            for unknown models that don't satisfy basic constraints.
        """
        score = 0.0

        if mp is None:
            return -10.0

        strengths = set(mp.strengths)

        if profile.requires_coding:
            score += 3.0 if "coding" in strengths else -1.0

        if profile.reasoning in ("medium", "deep"):
            score += 3.0 if "reasoning" in strengths else -1.0
        elif profile.reasoning == "low":
            score += 1.0 if "reasoning" in strengths else 0.5

        if profile.prefers_speed:
            score += 2.0 if "fast" in strengths else -1.0

        if profile.prefers_large_context:
            if "large_context" in strengths:
                score += 2.0
            else:
                score += min(mp.context_length / 128000.0, 1.0) * 1.5

        if profile.requires_vision:
            score += 3.0 if "vision" in strengths else -10.0

        if "balanced" in strengths:
            score += 0.5

        # --- Signal 2: Historical metrics (ProviderMetricsTracker) ---
        perf_score = 0.0
        if self._metrics_tracker is not None and provider_id is not None:
            stats = self._metrics_tracker.get_stats(provider_id, model_id)
            if stats is not None and stats.total_calls >= 3:
                perf_score += stats.success_rate * 2.0
                if profile.prefers_speed:
                    latency_bonus = max(0.0, 1.0 - stats.avg_latency_ms / 10000.0)
                    perf_score += latency_bonus * 1.5
                if profile.reasoning in ("medium", "deep"):
                    perf_score += stats.avg_reflection_confidence * 1.0

        # --- Signal 3: Domain-specific experience (ExperienceGraph) ---
        domain_score = 0.0
        if domain_stats and provider_id is not None:
            matches = [
                d for d in domain_stats
                if d["provider_id"] == provider_id
                and d["model_id"] == model_id
            ]
            if matches:
                m = matches[0]
                n = m["total_runs"]
                # Success rate bonus: up to +3.0 for high domain success.
                domain_score += m["success_rate"] * 3.0
                # Reflection confidence bonus: up to +1.5.
                domain_score += m["avg_reflection_confidence"] * 1.5
                # Cost efficiency: lower latency gets up to +1.0.
                latency_bonus = max(0.0, 1.0 - m["avg_latency_ms"] / 30000.0)
                domain_score += latency_bonus * 1.0
                # Volume bonus: more observations → more trust.
                volume_bonus = min(n / 10.0, 1.0) * 0.5
                domain_score += volume_bonus

        # --- Blend ---
        w_perf = self._performance_weight
        w_domain = self._domain_weight if profile.domain else 0.0
        w_cap = 1.0 - w_perf - w_domain

        score = score * w_cap + perf_score * w_perf + domain_score * w_domain

        return score

    def _load_domain_stats(
        self,
        domain: str,
    ) -> list[dict[str, Any]]:
        """Pre-load domain-specific provider stats from experience graph."""
        if not domain or self._experience_graph is None:
            return []
        return self._experience_graph.provider_domain_stats(
            domain=domain,
            min_observations=2,
        )

    @staticmethod
    def _build_profiles(
        raw: dict[str, dict[str, object]],
    ) -> dict[str, ModelProfile]:
        """Convert raw config dicts into ModelProfile instances.

        Args:
            raw: Mapping of model_id → attributes.

        Returns:
            A mapping of model_id → ModelProfile.
        """
        profiles: dict[str, ModelProfile] = {}
        for model_id, attrs in raw.items():
            strengths = attrs.get("strengths", [])
            if isinstance(strengths, list):
                strengths = [s.lower().strip() for s in strengths if isinstance(s, str)]
            profiles[model_id] = ModelProfile(
                model_id=model_id,
                strengths=strengths,
                context_length=int(attrs.get("context_length", 4096)),
            )
        return profiles
