"""Reciprocal Rank Fusion (RRF) for combining multi-strategy retrieval results.

RRF is a rank-based fusion method that combines multiple ranked lists
by giving each document a score of 1 / (k + rank) per list it appears in.
The constant *k* (default 60) dampens the contribution of high ranks so
that presence across multiple lists is more valuable than a high rank in
a single list.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.memory.models.memory import Memory


def fuse_reciprocal_rank(
    paths: list[Sequence[Memory]],
    k: int = 60,
) -> dict[str, float]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        paths: One or more ranked lists of Memory objects. Lower index
               means higher rank (position 0 is best).
        k: The RRF constant, controlling how much weight is given to
           presence across lists vs. exact rank position.

    Returns:
        A dict mapping each unique memory ID to its fused RRF score.
        Scores are raw sums of 1/(k + rank) and are **not** normalised.
    """
    scores: dict[str, float] = {}
    for path in paths:
        for rank, mem in enumerate(path):
            scores[mem.id] = (
                scores.get(mem.id, 0.0) + 1.0 / (k + rank + 1)
            )
    return scores


def normalise_rrf(
    scores: dict[str, float],
) -> dict[str, float]:
    """Normalise RRF scores to the [0, 1] range (divide by max)."""
    if not scores:
        return {}
    max_score = max(scores.values())
    if max_score <= 0.0:
        return {k: 0.0 for k in scores}
    return {k: v / max_score for k, v in scores.items()}
