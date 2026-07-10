"""World Model — typed entities and relationships for structured reasoning.

Builds on the raw :class:`KnowledgeGraph` triples by adding entity typing
so that the system can reason about Users, Projects, Files, Tools,
Providers, Plugins, Goals, and Tasks as first-class concepts with
attributes.

Example entities::

    User(id="usr_1", name="Alice")
    Project(id="proj_1", name="Astra X", language="python")
    File(id="file_1", path="/app/main.py", project="proj_1")

Example relations::

    (User) — owns → (Project)
    (Project) — contains → (File)
    (Tool) — provided_by → (Provider)
    (Goal) — decomposed_into → (Task)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Entity types
# ---------------------------------------------------------------------------


class EntityType(StrEnum):
    USER = "user"
    PROJECT = "project"
    FILE = "file"
    TOOL = "tool"
    PROVIDER = "provider"
    PLUGIN = "plugin"
    GOAL = "goal"
    TASK = "task"
    PATTERN = "pattern"
    ANTIPATTERN = "antipattern"


class RelationType(StrEnum):
    OWNS = "owns"
    CONTAINS = "contains"
    DEPENDS_ON = "depends_on"
    USES = "uses"
    PROVIDED_BY = "provided_by"
    DECOMPOSED_INTO = "decomposed_into"
    PRODUCED = "produced"
    FOLLOWED = "followed"
    AVOIDED = "avoided"
    RAN_ON = "ran_on"
    RESULTED_IN = "resulted_in"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class WorldEntity:
    """A typed entity in the world model.

    Attributes:
        id: Unique identifier.
        entity_type: The kind of entity.
        name: Human-readable label.
        attributes: Arbitrary key-value metadata (language, path, version, …).
        created_at: When this entity was first observed.
    """

    def __init__(
        self,
        entity_type: EntityType,
        name: str,
        attributes: dict[str, Any] | None = None,
        entity_id: str | None = None,
    ) -> None:
        self.id = entity_id or str(uuid4())
        self.entity_type = entity_type
        self.name = name
        self.attributes = attributes or {}
        self.created_at = datetime.now(timezone.utc)

    def to_triple_subject(self) -> str:
        return f"{self.entity_type.value}:{self.id}"

    def __repr__(self) -> str:
        return f"WorldEntity({self.entity_type}:{self.name})"


class WorldRelation:
    """A typed relationship between two entities.

    Attributes:
        subject_id: Entity ID of the subject.
        relation_type: The kind of relationship.
        object_id: Entity ID of the object.
        confidence: 0.0–1.0.
        metadata: Optional extra info (e.g., timestamp of the relation).
    """

    def __init__(
        self,
        subject_id: str,
        relation_type: RelationType,
        object_id: str,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.id = str(uuid4())
        self.subject_id = subject_id
        self.relation_type = relation_type
        self.object_id = object_id
        self.confidence = confidence
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"WorldRelation({self.subject_id} —{self.relation_type}→ {self.object_id})"


# ---------------------------------------------------------------------------
# WorldModel
# ---------------------------------------------------------------------------


class WorldModel:
    """Structured world model with typed entities and relations.

    Provides query methods that the Planner and Coordinator can use to
    reason about the environment (e.g., "Which files does this project
    contain?", "Which tools are available via which providers?").
    """

    def __init__(self) -> None:
        self._entities: dict[str, WorldEntity] = {}
        self._relations: list[WorldRelation] = []

    # -- Entity CRUD ----------------------------------------------------------

    def add_entity(
        self,
        entity_type: EntityType,
        name: str,
        attributes: dict[str, Any] | None = None,
        entity_id: str | None = None,
    ) -> WorldEntity:
        entity = WorldEntity(
            entity_type=entity_type,
            name=name,
            attributes=attributes or {},
            entity_id=entity_id,
        )
        self._entities[entity.id] = entity
        return entity

    def get_entity(self, entity_id: str) -> WorldEntity | None:
        return self._entities.get(entity_id)

    def find_entities(
        self,
        entity_type: EntityType | None = None,
        name: str | None = None,
    ) -> list[WorldEntity]:
        results = list(self._entities.values())
        if entity_type:
            results = [e for e in results if e.entity_type == entity_type]
        if name:
            results = [e for e in results if name.lower() in e.name.lower()]
        return results

    def remove_entity(self, entity_id: str) -> None:
        self._entities.pop(entity_id, None)
        self._relations = [
            r
            for r in self._relations
            if r.subject_id != entity_id and r.object_id != entity_id
        ]

    # -- Relation CRUD --------------------------------------------------------

    def add_relation(
        self,
        subject_id: str,
        relation_type: RelationType,
        object_id: str,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> WorldRelation:
        relation = WorldRelation(
            subject_id=subject_id,
            relation_type=relation_type,
            object_id=object_id,
            confidence=confidence,
            metadata=metadata,
        )
        self._relations.append(relation)
        return relation

    def query_relations(
        self,
        subject_id: str | None = None,
        relation_type: RelationType | None = None,
        object_id: str | None = None,
    ) -> list[WorldRelation]:
        results = list(self._relations)
        if subject_id:
            results = [r for r in results if r.subject_id == subject_id]
        if relation_type:
            results = [r for r in results if r.relation_type == relation_type]
        if object_id:
            results = [r for r in results if r.object_id == object_id]
        return results

    def get_neighbors(
        self,
        entity_id: str,
        max_depth: int = 1,
    ) -> list[WorldEntity]:
        """BFS traversal from *entity_id* up to *max_depth* hops."""
        visited: set[str] = {entity_id}
        frontier: list[str] = [entity_id]
        neighbors: list[WorldEntity] = []

        for _ in range(max_depth):
            next_frontier: list[str] = []
            for eid in frontier:
                for r in self._relations:
                    other_id = None
                    if r.subject_id == eid and r.object_id not in visited:
                        other_id = r.object_id
                    elif r.object_id == eid and r.subject_id not in visited:
                        other_id = r.subject_id
                    if other_id is not None:
                        visited.add(other_id)
                        ent = self._entities.get(other_id)
                        if ent:
                            neighbors.append(ent)
                            next_frontier.append(other_id)
            frontier = next_frontier

        return neighbors

    # -- Convenience queries --------------------------------------------------

    def get_project_context(self, project_name: str) -> dict[str, Any]:
        """Return a structured context dict for a project, suitable for
        injecting into the planner prompt."""
        projects = self.find_entities(EntityType.PROJECT, project_name)
        if not projects:
            return {}

        proj = projects[0]
        files = [
            e.name
            for e in self.get_neighbors(proj.id)
            if e.entity_type == EntityType.FILE
        ]
        tools = [
            e.name
            for e in self.get_neighbors(proj.id)
            if e.entity_type == EntityType.TOOL
        ]
        providers = [
            e.name for e in self.find_entities(EntityType.PROVIDER)
        ]

        return {
            "project": proj.name,
            "language": proj.attributes.get("language", "unknown"),
            "files": files[:20],
            "tools": tools[:10],
            "available_providers": providers,
        }

    # -- Serialisation helpers ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": {
                eid: {
                    "id": e.id,
                    "type": e.entity_type.value,
                    "name": e.name,
                    "attributes": e.attributes,
                }
                for eid, e in self._entities.items()
            },
            "relations": [
                {
                    "id": r.id,
                    "subject_id": r.subject_id,
                    "relation": r.relation_type.value,
                    "object_id": r.object_id,
                    "confidence": r.confidence,
                }
                for r in self._relations
            ],
        }

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @property
    def relation_count(self) -> int:
        return len(self._relations)
