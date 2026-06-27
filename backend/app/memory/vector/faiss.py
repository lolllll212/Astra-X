"""FAISS-backed vector store.

Requires the ``faiss`` package::

    pip install faiss-cpu  # or faiss-gpu
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np  # type: ignore[import-not-found]

from app.memory.vector.base import SearchResult, VectorRecord, VectorStore


class FaissVectorStore(VectorStore):
    """Vector store backed by a FAISS index (in-memory, optional persistence)."""

    def __init__(self, dimensions: int = 768) -> None:
        self._dimensions = dimensions
        self._index: Any = None
        self._id_map: dict[int, str] = {}
        self._next_id: int = 0
        self._metadata: dict[str, dict[str, object]] = {}

    async def insert(self, record: VectorRecord) -> None:
        vectors = np.array([record.vector], dtype=np.float32)
        if self._index is None:
            try:
                import faiss  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError("faiss is required. Install with: pip install faiss-cpu") from exc
            self._index = faiss.IndexFlatIP(self._dimensions)

        faiss_id = self._next_id
        self._next_id += 1
        self._id_map[faiss_id] = record.id
        self._metadata[record.id] = dict(record.metadata)
        self._index.add(vectors)

    async def insert_batch(self, records: Sequence[VectorRecord]) -> None:
        for record in records:
            await self.insert(record)

    async def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 10,
        filter_: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        if self._index is None or self._index.ntotal == 0:
            return []

        query = np.array([query_vector], dtype=np.float32)
        scores, indices = self._index.search(query, limit)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            record_id = self._id_map.get(int(idx))
            if record_id is None:
                continue
            meta = self._metadata.get(record_id, {})
            if filter_ and not self._matches_filter(meta, filter_):
                continue
            results.append(SearchResult(id=record_id, score=float(score), metadata=meta))

        return results

    async def delete(self, record_id: str) -> None:
        self._metadata.pop(record_id, None)
        for faiss_id, rid in list(self._id_map.items()):
            if rid == record_id:
                del self._id_map[faiss_id]
                break

    async def delete_batch(self, record_ids: Sequence[str]) -> None:
        for rid in record_ids:
            await self.delete(rid)

    async def get(self, record_id: str) -> VectorRecord | None:
        meta = self._metadata.get(record_id)
        if meta is None:
            return None
        return VectorRecord(id=record_id, vector=[], metadata=meta)

    async def count(self) -> int:
        return len(self._metadata)

    async def clear(self) -> None:
        self._index = None
        self._id_map.clear()
        self._metadata.clear()
        self._next_id = 0

    @staticmethod
    def _matches_filter(metadata: dict[str, object], filter_: dict[str, object]) -> bool:
        return all(metadata.get(key) == value for key, value in filter_.items())
