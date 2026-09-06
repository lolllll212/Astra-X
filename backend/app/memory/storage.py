from __future__ import annotations

from datetime import UTC, datetime

from app.domain.enums import MemoryScope
from app.memory.models.memory import Memory, MemoryType


class MemoryStorage:
    """In-memory storage for Memory records.

    Provides basic CRUD with filtering.  The backing store is a dict for
    development and moderate workloads.  Replace with a database-backed
    implementation for production (see :mod:`app.database.repositories`).
    """

    def __init__(self) -> None:
        self._memories: dict[str, Memory] = {}

    async def save(self, memory: Memory) -> None:
        existing = self._memories.get(memory.id)
        if existing:
            memory = Memory(
                **{
                    **memory.model_dump(),
                    "last_accessed_at": existing.last_accessed_at,
                    "access_count": existing.access_count,
                }
            )
        self._memories[memory.id] = memory

    async def get(self, memory_id: str) -> Memory | None:
        return self._memories.get(memory_id)

    async def delete(self, memory_id: str) -> None:
        self._memories.pop(memory_id, None)

    async def list(
        self,
        *,
        memory_type: MemoryType | None = None,
        scope: MemoryScope | None = None,
        conversation_id: str | None = None,
        user_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        results = list(self._memories.values())

        if memory_type is not None:
            results = [m for m in results if m.memory_type == memory_type]
        if scope is not None:
            results = [m for m in results if m.scope == scope]
        if conversation_id is not None:
            results = [m for m in results if m.conversation_id == conversation_id]
        if user_id is not None:
            results = [m for m in results if m.user_id == user_id]

        results.sort(key=lambda m: m.created_at, reverse=True)
        return results[offset : offset + limit]

    async def count(
        self,
        *,
        memory_type: MemoryType | None = None,
        scope: MemoryScope | None = None,
    ) -> int:
        memories = await self.list(memory_type=memory_type, scope=scope, limit=10_000)
        return len(memories)

    async def update_access(self, memory_id: str) -> None:
        memory = self._memories.get(memory_id)
        if memory is None:
            return
        self._memories[memory_id] = Memory(
            **{
                **memory.model_dump(),
                "last_accessed_at": datetime.now(UTC),
                "access_count": memory.access_count + 1,
            }
        )

    async def clear(self) -> None:
        self._memories.clear()
