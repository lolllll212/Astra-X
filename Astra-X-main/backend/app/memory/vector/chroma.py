"""ChromaDB-backed vector store.

Requires the ``chromadb`` package::

    pip install chromadb
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.memory.vector.base import SearchResult, VectorRecord, VectorStore


class ChromaVectorStore(VectorStore):
    """Vector store backed by ChromaDB (local or remote)."""

    def __init__(self, collection_name: str = "astra_memories", persist_directory: str = "./chroma_db") -> None:
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._collection: Any = None
        self._client: Any = None

    async def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import chromadb  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("chromadb is required. Install with: pip install chromadb") from exc

        self._client = chromadb.PersistentClient(path=self._persist_directory)
        self._collection = self._client.get_or_create_collection(self._collection_name)

    async def insert(self, record: VectorRecord) -> None:
        await self._ensure_client()
        self._collection.add(
            ids=[record.id],
            embeddings=[record.vector],
            metadatas=[record.metadata],
        )

    async def insert_batch(self, records: Sequence[VectorRecord]) -> None:
        await self._ensure_client()
        self._collection.add(
            ids=[r.id for r in records],
            embeddings=[r.vector for r in records],
            metadatas=[r.metadata for r in records],
        )

    async def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 10,
        filter_: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        await self._ensure_client()
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where=filter_,
        )
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        return [
            SearchResult(id=rid, score=1.0 - dist, metadata=meta or {})
            for rid, dist, meta in zip(ids, distances, metadatas, strict=False)
        ]

    async def delete(self, record_id: str) -> None:
        await self._ensure_client()
        self._collection.delete(ids=[record_id])

    async def delete_batch(self, record_ids: Sequence[str]) -> None:
        await self._ensure_client()
        self._collection.delete(ids=list(record_ids))

    async def get(self, record_id: str) -> VectorRecord | None:
        await self._ensure_client()
        results = self._collection.get(ids=[record_id])
        if not results["ids"]:
            return None
        return VectorRecord(
            id=results["ids"][0],
            vector=results["embeddings"][0] if results.get("embeddings") else [],
            metadata=results["metadatas"][0] if results.get("metadatas") else {},
        )

    async def count(self) -> int:
        await self._ensure_client()
        return int(self._collection.count())

    async def clear(self) -> None:
        await self._ensure_client()
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(self._collection_name)
