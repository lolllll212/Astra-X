"""Memory decay and cleanup.

Human memory forgets.  Astra should too.  This module manages the
lifecycle of memories: decay over time, cleanup of low-importance
entries, and TTL-based expiry.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from app.core.logging import get_logger
from app.memory.models.memory import Memory

logger = get_logger(__name__)


class Forgetting:
    """Manages memory decay and cleanup policies.

    Decay formula::
        retention = importance * 2^(-age / halflife) * access_frequency_factor
    """

    def __init__(
        self,
        halflife_days: float = 30.0,
        decay_threshold: float = 0.05,
        max_memories_per_type: int = 1000,
    ) -> None:
        self._halflife = halflife_days
        self._threshold = decay_threshold
        self._max_per_type = max_memories_per_type

    def compute_retention(self, memory: Memory) -> float:
        age = (datetime.now(UTC) - memory.created_at).total_seconds() / 86400.0
        decay = math.exp(-math.log(2) * age / self._halflife)
        freq_factor = math.log2(memory.access_count + 2) / math.log2(self._max_per_type + 2)
        return memory.importance * decay * freq_factor

    def should_forget(self, memory: Memory) -> bool:
        if memory.ttl_days is not None:
            age = (datetime.now(UTC) - memory.created_at).total_seconds() / 86400.0
            if age > memory.ttl_days:
                return True
        return self.compute_retention(memory) < self._threshold

    def select_candidates(self, memories: Sequence[Memory], target_count: int) -> list[Memory]:
        scored = [(m, self.compute_retention(m)) for m in memories]
        scored.sort(key=lambda x: x[1])
        overflow = len(memories) - target_count
        if overflow <= 0:
            return []
        return [m for m, _ in scored[:overflow]]

    async def apply_decay(
        self,
        memories: Sequence[Memory],
        memory_store: Any,
    ) -> tuple[int, int]:
        """Evaluate all memories and remove those below the decay threshold.

        Args:
            memories: All current memories.
            memory_store: Object with async ``delete(memory_id)`` method.

        Returns:
            (removed_count, kept_count).
        """
        removed = 0
        kept = 0

        for memory in memories:
            if self.should_forget(memory):
                await memory_store.delete(memory.id)
                removed += 1
            else:
                kept += 1

        if removed:
            logger.info(
                "forgetting.decay_applied",
                removed=removed,
                kept=kept,
            )

        return removed, kept

    async def enforce_capacity(
        self,
        memories: Sequence[Memory],
        memory_store: Any,
    ) -> int:
        """Remove the lowest-retention memories when type limits are exceeded.

        Args:
            memories: All current memories.
            memory_store: Object with async ``delete(memory_id)`` method.

        Returns:
            Number of memories removed.
        """
        by_type: dict[str, list[Memory]] = {}
        for m in memories:
            by_type.setdefault(m.memory_type.value, []).append(m)

        removed = 0
        for _type_name, typed_mems in by_type.items():
            if len(typed_mems) > self._max_per_type:
                candidates = self.select_candidates(typed_mems, self._max_per_type)
                for c in candidates:
                    await memory_store.delete(c.id)
                    removed += 1

        if removed:
            logger.info("forgetting.capacity_enforced", removed=removed)

        return removed
