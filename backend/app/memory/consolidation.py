"""Merge duplicate or overlapping memories.

Consolidation identifies memories with similar content and merges them
into a single record with updated importance and metadata.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import uuid4

from typing import Any

from app.core.logging import get_logger
from app.memory.models.memory import Memory

logger = get_logger(__name__)


class Consolidation:
    """Finds and merges duplicate or overlapping memories.

    Uses simple text similarity (ratio-based) to identify candidates.
    Replace with embedding similarity for better accuracy.
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self._threshold = similarity_threshold

    async def find_duplicates(
        self,
        memories: Sequence[Memory],
    ) -> list[tuple[Memory, Memory, float]]:
        """Return pairs of memories whose content exceeds the similarity threshold."""
        pairs: list[tuple[Memory, Memory, float]] = []
        mems = list(memories)

        for i in range(len(mems)):
            for j in range(i + 1, len(mems)):
                a = mems[i].content.lower()
                b = mems[j].content.lower()
                ratio = difflib.SequenceMatcher(None, a, b).ratio()
                if ratio >= self._threshold:
                    pairs.append((mems[i], mems[j], ratio))

        return pairs

    async def merge(self, primary: Memory, secondary: Memory) -> Memory:
        """Merge two memories into one, preserving the richer information."""

        merged_meta = {**secondary.metadata, **primary.metadata}
        merged_meta["merged_from"] = [secondary.id]
        merged_meta["merge_timestamp"] = datetime.now(UTC).isoformat()

        return Memory(
            id=str(uuid4()),
            content=primary.content if len(primary.content) >= len(secondary.content) else secondary.content,
            memory_type=primary.memory_type,
            scope=primary.scope,
            importance=max(primary.importance, secondary.importance),
            conversation_id=primary.conversation_id or secondary.conversation_id,
            user_id=primary.user_id or secondary.user_id,
            metadata=merged_meta,
            created_at=min(primary.created_at, secondary.created_at),
            last_accessed_at=max(primary.last_accessed_at, secondary.last_accessed_at),
            access_count=primary.access_count + secondary.access_count,
        )

    async def consolidate(
        self,
        memories: Sequence[Memory],
        memory_store: Any,
    ) -> int:
        """Run consolidation over all memories.

        Args:
            memories: All current memories.
            memory_store: Object with async ``save(memory)`` and
                ``delete(memory_id)`` methods.

        Returns:
            Number of memories removed (merged away).
        """
        duplicates = await self.find_duplicates(memories)
        removed: set[str] = set()
        merged_count = 0

        for primary, secondary, _ratio in duplicates:
            if primary.id in removed or secondary.id in removed:
                continue
            merged = await self.merge(primary, secondary)
            await memory_store.save(merged)
            await memory_store.delete(primary.id)
            await memory_store.delete(secondary.id)
            removed.add(primary.id)
            removed.add(secondary.id)
            merged_count += 1

        if merged_count:
            logger.info(
                "consolidation.complete",
                merged_pairs=merged_count,
                removed_count=len(removed),
            )

        return len(removed)
