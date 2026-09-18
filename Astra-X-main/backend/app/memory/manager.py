"""Memory Manager — the public entry point for the memory subsystem.

Responsibilities:
- Store memories (with optional reflection and knowledge extraction)
- Retrieve memories via semantic + hybrid scoring
- Forget (decay / TTL / capacity enforcement)
- Consolidate duplicate memories
- Summarize conversations
- Reflect over conversation history to extract structured insights
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from app.core.logging import get_logger
from app.domain.enums import MemoryScope
from app.memory.chunker import Chunker
from app.memory.consolidation import Consolidation
from app.memory.embedder import Embedder
from app.memory.forgetting import Forgetting
from app.memory.knowledge_graph import KnowledgeGraph
from app.memory.models.memory import Memory, MemoryType
from app.memory.models.retrieval import MemoryQuery, RetrievalResult
from app.memory.reflection import Reflection
from app.memory.retriever import Retriever
from app.memory.scorer import Scorer
from app.memory.storage import MemoryStorage
from app.memory.summarizer import MemorySummarizer
from app.memory.vector.base import VectorStore

logger = get_logger(__name__)


class MemoryManager:
    """High-level orchestrator for the memory subsystem.

    Usage::

        manager = MemoryManager(vector_store, embedder, llm_router)
        await manager.store("User likes dark mode")
        results = await manager.retrieve("What does the user prefer?")
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        storage: MemoryStorage | None = None,
        retriever: Retriever | None = None,
        scorer: Scorer | None = None,
        chunker: Chunker | None = None,
        summarizer: MemorySummarizer | None = None,
        reflection: Reflection | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        consolidation: Consolidation | None = None,
        forgetting: Forgetting | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._storage = storage or MemoryStorage()
        self._scorer = scorer or Scorer()
        self._chunker = chunker or Chunker()
        self._summarizer = summarizer
        self._reflection = reflection
        self._knowledge_graph = knowledge_graph
        self._consolidation = consolidation or Consolidation(embedder=embedder)
        self._forgetting = forgetting

        self._retriever = retriever or Retriever(
            vector_store=vector_store,
            embedder=embedder,
            storage=self._storage,
            scorer=self._scorer,
            knowledge_graph=knowledge_graph,
        )

    # -- write -----------------------------------------------------------------

    async def store(
        self,
        content: str,
        *,
        memory_type: MemoryType = MemoryType.EPISODIC,
        scope: MemoryScope = MemoryScope.CONVERSATION,
        conversation_id: str | None = None,
        user_id: str | None = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        memory = Memory(
            id=str(uuid4()),
            content=content,
            memory_type=memory_type,
            scope=scope,
            importance=importance,
            conversation_id=conversation_id,
            user_id=user_id,
            metadata=metadata or {},
        )

        await self._storage.save(memory)

        vector = await self._embedder.embed(content)
        from app.memory.vector.base import VectorRecord
        await self._vector_store.insert(
            VectorRecord(
                id=memory.id,
                vector=vector,
                metadata={
                    "memory_type": memory_type.value,
                    "scope": scope.value,
                    "conversation_id": conversation_id or "",
                    "importance": importance,
                },
            )
        )

        logger.debug(
            "manager.stored",
            memory_id=memory.id,
            memory_type=memory_type.value,
            conversation_id=conversation_id,
        )

        return memory

    async def store_conversation(
        self,
        conversation_id: str,
        history_text: str,
        *,
        reflect: bool = True,
        summarize: bool = True,
        extract_knowledge: bool = True,
    ) -> list[Memory]:
        """Process a full conversation transcript through the memory pipeline.

        Pipeline::

            Conversation
               │
               ├── Reflection → structured memories
               ├── Summary → episodic memory
               └── Knowledge Graph → entity triples
        """
        stored: list[Memory] = []

        if self._reflection and reflect:
            reflected = await self._reflection.reflect(conversation_id, history_text)
            for mem in reflected:
                await self._store_memory_with_vector(mem)
                stored.append(mem)

        if self._summarizer and summarize:
            summary = await self._summarizer.summarize(conversation_id, history_text)
            if summary is not None:
                await self._store_memory_with_vector(summary)
                stored.append(summary)

        if self._knowledge_graph and extract_knowledge:
            triples = await self._knowledge_graph.extract_from_text(history_text, conversation_id)
            logger.debug(
                "manager.knowledge_extracted",
                conversation_id=conversation_id,
                triple_count=len(triples),
            )

        logger.info(
            "manager.conversation_stored",
            conversation_id=conversation_id,
            memories_created=len(stored),
        )

        return stored

    async def _store_memory_with_vector(self, memory: Memory) -> None:
        await self._storage.save(memory)

        vector = await self._embedder.embed(memory.content)
        from app.memory.vector.base import VectorRecord
        await self._vector_store.insert(
            VectorRecord(
                id=memory.id,
                vector=vector,
                metadata={
                    "memory_type": memory.memory_type.value,
                    "scope": memory.scope.value,
                    "conversation_id": memory.conversation_id or "",
                    "importance": memory.importance,
                },
            )
        )

    # -- read ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        *,
        conversation_id: str | None = None,
        memory_types: Sequence[MemoryType] | None = None,
        scopes: Sequence[MemoryScope] | None = None,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        query_obj = MemoryQuery(
            query=query,
            conversation_id=conversation_id,
            memory_types=memory_types,
            scopes=scopes,
            limit=limit,
        )
        return await self._retriever.retrieve(query_obj)

    async def get_memory(self, memory_id: str) -> Memory | None:
        return await self._storage.get(memory_id)

    async def list_memories(
        self,
        *,
        memory_type: MemoryType | None = None,
        scope: MemoryScope | None = None,
        conversation_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        return await self._storage.list(
            memory_type=memory_type,
            scope=scope,
            conversation_id=conversation_id,
            limit=limit,
            offset=offset,
        )

    # -- delete ----------------------------------------------------------------

    async def forget(self, memory_id: str) -> None:
        await self._storage.delete(memory_id)
        await self._vector_store.delete(memory_id)

    async def forget_conversation(self, conversation_id: str) -> int:
        memories = await self._storage.list(conversation_id=conversation_id)
        for m in memories:
            await self._storage.delete(m.id)
            await self._vector_store.delete(m.id)
        return len(memories)

    # -- maintenance -----------------------------------------------------------

    async def run_maintenance(self) -> dict[str, int]:
        """Run forgetting, consolidation, and capacity enforcement.

        Returns a summary dict of actions taken.
        """
        summary: dict[str, int] = {}

        all_memories = await self._storage.list(limit=10_000)

        if self._forgetting:
            removed, kept = await self._forgetting.apply_decay(all_memories, self._storage)
            summary["decay_removed"] = removed
            summary["decay_kept"] = kept

            capacity_removed = await self._forgetting.enforce_capacity(all_memories, self._storage)
            summary["capacity_removed"] = capacity_removed

        if self._consolidation:
            merged = await self._consolidation.consolidate(all_memories, self._storage)
            summary["consolidation_removed"] = merged

        logger.info("manager.maintenance_complete", **summary)
        return summary

    @property
    def storage(self) -> MemoryStorage:
        return self._storage

    @property
    def knowledge_graph(self) -> KnowledgeGraph | None:
        return self._knowledge_graph
