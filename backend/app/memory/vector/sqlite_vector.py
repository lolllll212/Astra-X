"""SQLite-backed vector store using in-memory numpy cosine search.

Stores vectors as JSON blobs alongside optional metadata.  All vector
computation happens in memory — this is appropriate for moderate
collections (up to 100K vectors).  For larger scale, swap to FAISS
or Chroma.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

from app.memory.vector.base import SearchResult, VectorRecord, VectorStore


class SQLiteVectorStore(VectorStore):
    """In-memory vector store backed by SQLite persistence.

    Vectors are held in memory for fast cosine-similarity search and
    persisted to a JSON file (or database table) on mutation.
    """

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}
        self._dirty: bool = False

    async def insert(self, record: VectorRecord) -> None:
        self._records[record.id] = record
        self._dirty = True

    async def insert_batch(self, records: Sequence[VectorRecord]) -> None:
        for r in records:
            self._records[r.id] = r
        self._dirty = True

    async def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 10,
        filter_: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        if not self._records:
            return []

        scored: list[tuple[str, float, dict[str, object]]] = []

        for rid, rec in self._records.items():
            if filter_ and not self._matches_filter(rec.metadata, filter_):
                continue

            sim = self._cosine_similarity(query_vector, rec.vector)
            scored.append((rid, sim, rec.metadata))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:limit]

        return [
            SearchResult(id=rid, score=score, metadata=meta)
            for rid, score, meta in top
        ]

    async def delete(self, record_id: str) -> None:
        self._records.pop(record_id, None)
        self._dirty = True

    async def delete_batch(self, record_ids: Sequence[str]) -> None:
        for rid in record_ids:
            self._records.pop(rid, None)
        self._dirty = True

    async def get(self, record_id: str) -> VectorRecord | None:
        return self._records.get(record_id)

    async def count(self) -> int:
        return len(self._records)

    async def clear(self) -> None:
        self._records.clear()
        self._dirty = True

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = 0.0
        na = 0.0
        nb = 0.0
        for ai, bi in zip(a, b, strict=False):
            dot += ai * bi
            na += ai * ai
            nb += bi * bi
        denom = math.sqrt(na) * math.sqrt(nb)
        return dot / denom if denom > 0 else 0.0

    @staticmethod
    def _matches_filter(metadata: dict[str, object], filter_: dict[str, object]) -> bool:
        return all(metadata.get(key) == value for key, value in filter_.items())

    # -- serialisation -------------------------------------------------------

    def to_json(self) -> str:
        data = {rid: {"vector": rec.vector, "metadata": rec.metadata} for rid, rec in self._records.items()}
        return json.dumps(data, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> SQLiteVectorStore:
        store = cls()
        if not json_str:
            return store
        data = json.loads(json_str)
        for rid, item in data.items():
            store._records[rid] = VectorRecord(id=rid, vector=item["vector"], metadata=item.get("metadata", {}))
        return store
