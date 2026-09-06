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
from app.memory.state_machine import StateMachineDefinition, StateTransition

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Entity types
# ---------------------------------------------------------------------------


class EntityType(StrEnum):
    USER = "user"
    PROJECT = "project"
    REPOSITORY = "repository"
    FILE = "file"
    TOOL = "tool"
    PROVIDER = "provider"
    PLUGIN = "plugin"
    GOAL = "goal"
    TASK = "task"
    PATTERN = "pattern"
    ANTIPATTERN = "antipattern"
    MISSION = "mission"


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
    TRANSITIONED_TO = "transitioned_to"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class EntityState:
    """A named state that an entity has transitioned through.

    Attributes:
        state_name: The state label (e.g. ``"Build Failed"``,
            ``"Tests Passing"``, ``"Ready For Release"``).
        timestamp: When this state was entered.
        metadata: Optional key-value details about this state transition
            (e.g. ``{"exit_code": 1, "duration_ms": 4500}``).
    """

    def __init__(
        self,
        state_name: str,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.state_name = state_name
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"EntityState({self.state_name} @ {self.timestamp.isoformat()})"


class WorldEntity:
    """A typed entity in the world model.

    Attributes:
        id: Unique identifier.
        entity_type: The kind of entity.
        name: Human-readable label.
        attributes: Arbitrary key-value metadata (language, path, version, …).
        current_state: The entity's most recent state (``None`` if never set).
        state_history: Ordered list of all past states (newest last).
        created_at: When this entity was first observed.
    """

    def __init__(
        self,
        entity_type: EntityType,
        name: str,
        attributes: dict[str, Any] | None = None,
        entity_id: str | None = None,
        current_state: str | None = None,
    ) -> None:
        self.id = entity_id or str(uuid4())
        self.entity_type = entity_type
        self.name = name
        self.attributes = attributes or {}
        self.state_history: list[EntityState] = []
        self.transitions: list[StateTransition] = []
        self.created_at = datetime.now(timezone.utc)

        if current_state:
            self.set_state(current_state)

    def set_state(
        self,
        state_name: str,
        metadata: dict[str, Any] | None = None,
        trigger: str = "manual",
    ) -> EntityState:
        """Transition this entity to a new state.

        Appends a :class:`EntityState` to the history, updates
        ``current_state``, and records a full :class:`StateTransition`
        with *trigger* and *from_state*.

        Args:
            state_name: The new state label.
            metadata: Optional details about this transition.
            trigger: What caused this transition (e.g. ``"task:run_tests"``).

        Returns:
            The newly created EntityState.
        """
        state = EntityState(
            state_name=state_name,
            metadata=metadata,
        )
        from_state = self.current_state
        self.state_history.append(state)
        self.attributes["current_state"] = state_name

        self.transitions.append(StateTransition(
            from_state=from_state,
            to_state=state_name,
            timestamp=state.timestamp,
            trigger=trigger,
            metadata=metadata or {},
        ))
        return state

    def get_transition_history(
        self,
        limit: int | None = None,
    ) -> list[StateTransition]:
        """Return the ordered transition history, newest last.

        Args:
            limit: Optional max number of recent transitions to return.
        """
        if limit is not None:
            return self.transitions[-limit:]
        return list(self.transitions)

    @property
    def current_state(self) -> str | None:
        """The most recent state, or ``None`` if never set."""
        if self.state_history:
            return self.state_history[-1].state_name
        return None

    def get_state_history(
        self,
        limit: int | None = None,
    ) -> list[EntityState]:
        """Return the ordered state history, newest last.

        Args:
            limit: Optional max number of recent states to return.

        Returns:
            A list of EntityState instances.
        """
        if limit is not None:
            return self.state_history[-limit:]
        return list(self.state_history)

    def get_state_at(
        self,
        timestamp: datetime,
    ) -> EntityState | None:
        """Return the state that was active at *timestamp*.

        Args:
            timestamp: The point in time to query.

        Returns:
            The state active at that time, or ``None`` if the entity
            had no state at that point.
        """
        active: EntityState | None = None
        for s in self.state_history:
            if s.timestamp <= timestamp:
                active = s
            else:
                break
        return active

    def to_triple_subject(self) -> str:
        return f"{self.entity_type.value}:{self.id}"

    def __repr__(self) -> str:
        state = f" [{self.current_state}]" if self.current_state else ""
        return f"WorldEntity({self.entity_type}:{self.name}{state})"


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
        self._state_machines: dict[EntityType, StateMachineDefinition] = {}

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

    # -- State queries --------------------------------------------------------

    def find_entities_by_state(
        self,
        state_name: str,
        entity_type: EntityType | None = None,
    ) -> list[WorldEntity]:
        """Find all entities currently in *state_name*.

        Args:
            state_name: The state to match (e.g. ``"Tests Passing"``).
            entity_type: Optional filter to a specific entity type.

        Returns:
            A list of matching entities.
        """
        results: list[WorldEntity] = []
        for e in self._entities.values():
            if e.current_state == state_name:
                if entity_type is None or e.entity_type == entity_type:
                    results.append(e)
        return results

    def get_entity_state_history(
        self,
        entity_id: str,
        limit: int | None = None,
    ) -> list[EntityState]:
        """Return the state history for an entity.

        Args:
            entity_id: The entity to query.
            limit: Optional max number of recent states.

        Returns:
            Ordered list of EntityState (newest last), or empty list
            if the entity doesn't exist or has no history.
        """
        entity = self._entities.get(entity_id)
        if entity is None:
            return []
        return entity.get_state_history(limit=limit)

    def get_state_timeline(
        self,
        entity_id: str,
    ) -> list[dict[str, Any]]:
        """Return a human-readable timeline of state transitions.

        When full :class:`StateTransition` records are available (with
        trigger info), they are used. Otherwise falls back to inferring
        the timeline from the ordered state history.

        Each entry contains::

            {"from": str | None, "to": str, "at": str, "trigger": str,
             "metadata": dict}

        Args:
            entity_id: The entity to query.

        Returns:
            A list of transition dicts in chronological order.
        """
        entity = self._entities.get(entity_id)
        if entity is None:
            return []

        # Prefer full transition records when available.
        if entity.transitions:
            return [
                {
                    "from": t.from_state,
                    "to": t.to_state,
                    "at": t.timestamp.isoformat(),
                    "trigger": t.trigger,
                    "metadata": t.metadata,
                }
                for t in entity.transitions
            ]

        # Legacy fallback: infer from state_history.
        if not entity.state_history:
            return []
        timeline: list[dict[str, Any]] = []
        prev: str | None = None
        for st in entity.state_history:
            timeline.append({
                "from": prev,
                "to": st.state_name,
                "at": st.timestamp.isoformat(),
                "metadata": st.metadata,
            })
            prev = st.state_name
        return timeline

    # -- State machine registration -------------------------------------------

    def register_state_machine(
        self, definition: StateMachineDefinition,
    ) -> None:
        """Register a state machine for an entity type.

        Once registered, :meth:`transition_entity` and
        :meth:`get_valid_next_states` can enforce transition rules.
        """
        self._state_machines[definition.entity_type] = definition

    def get_state_machine(
        self, entity_type: EntityType,
    ) -> StateMachineDefinition | None:
        """Return the registered state machine for *entity_type*, or ``None``."""
        return self._state_machines.get(entity_type)

    def get_valid_next_states(self, entity_id: str) -> set[str]:
        """Return the valid next states for an entity, based on its
        registered state machine. Returns an empty set when no state
        machine is registered.
        """
        entity = self._entities.get(entity_id)
        if entity is None:
            return set()
        sm = self._state_machines.get(entity.entity_type)
        if sm is None:
            return set()
        return sm.get_valid_next_states(entity.current_state)

    def transition_entity(
        self,
        entity_id: str,
        to_state: str,
        trigger: str = "manual",
        metadata: dict[str, Any] | None = None,
    ) -> EntityState | None:
        """Validate and apply a state transition.

        If a :class:`StateMachineDefinition` is registered for the
        entity's type, the transition is validated first. Invalid
        transitions are rejected with a warning.

        Returns:
            The new :class:`EntityState`, or ``None`` if the entity
            doesn't exist or the transition is invalid.
        """
        entity = self._entities.get(entity_id)
        if entity is None:
            return None

        sm = self._state_machines.get(entity.entity_type)
        if sm is not None and not sm.validate(entity.current_state, to_state):
            logger.warning(
                "world_model.invalid_transition",
                entity_id=entity_id,
                from_state=entity.current_state,
                to_state=to_state,
            )
            return None

        return entity.set_state(to_state, trigger=trigger, metadata=metadata)

    # -- State reasoning -----------------------------------------------------

    def get_state_narrative(
        self, entity_id: str, max_events: int = 10,
    ) -> str:
        """Produce a human-readable narrative of state changes.

        Walks the entity's transition history and builds a chronological
        summary such as::

            1. (initial) → Tests Failed (triggered by task:run_tests)
            2. Tests Failed → Fix Applied (triggered by task:fix_bug)
            3. Fix Applied → Tests Passing (triggered by task:run_tests)
            Next possible states: Ready to Deploy, In Development

        Args:
            entity_id: The entity to describe.
            max_events: Maximum number of transitions to include.

        Returns:
            A formatted narrative string, or an empty string when the
            entity doesn't exist or has no transitions.
        """
        entity = self._entities.get(entity_id)
        if not entity or not entity.transitions:
            return ""

        transitions = entity.transitions[-max_events:]
        parts: list[str] = []
        for i, t in enumerate(transitions):
            trigger_hint = (
                f" (triggered by {t.trigger})"
                if t.trigger != "manual" else ""
            )
            from_s = t.from_state or "(initial)"
            parts.append(f"{i + 1}. {from_s} → {t.to_state}{trigger_hint}")

        valid = self.get_valid_next_states(entity_id)
        if valid:
            parts.append(
                f"Next possible states: {', '.join(sorted(valid))}"
            )

        return "\n".join(parts)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format a duration in seconds to a human-friendly string."""
        if seconds < 0:
            return "0s"
        if seconds < 60:
            return f"{seconds:.0f}s"
        minutes = seconds / 60
        if minutes < 60:
            return f"{minutes:.0f}m"
        hours = seconds / 3600
        if hours < 24:
            return f"{hours:.1f}h"
        days = hours / 24
        return f"{days:.1f}d"

    def get_state_dwell_times(
        self, entity_id: str,
    ) -> list[dict[str, Any]]:
        """Calculate how long the entity stayed in each state.

        Compares consecutive transition timestamps. The last entry
        measures time from the most recent transition to *now*.

        Args:
            entity_id: The entity to analyse.

        Returns:
            A list of dicts, one per state visited, with keys::

                state, duration_seconds, duration_human,
                trigger, entered_at, (still_active)
        """
        entity = self._entities.get(entity_id)
        if not entity or not entity.transitions:
            return []

        dwells: list[dict[str, Any]] = []
        ts = entity.transitions

        for i in range(len(ts) - 1):
            current = ts[i]
            next_t = ts[i + 1]
            duration_s = (next_t.timestamp - current.timestamp).total_seconds()
            dwells.append({
                "state": current.to_state,
                "duration_seconds": duration_s,
                "duration_human": self._format_duration(duration_s),
                "trigger": current.trigger,
                "entered_at": current.timestamp.isoformat(),
            })

        # Current (last) state — dwell up to now.
        last = ts[-1]
        duration_s = (datetime.now(timezone.utc) - last.timestamp).total_seconds()
        dwells.append({
            "state": last.to_state,
            "duration_seconds": duration_s,
            "duration_human": self._format_duration(duration_s),
            "trigger": last.trigger,
            "entered_at": last.timestamp.isoformat(),
            "still_active": True,
        })
        return dwells

    def suggest_state_path(
        self, entity_id: str,
    ) -> list[dict[str, str]]:
        """Suggest next states with reasoning, based on the entity's
        current state and the registered state machine.

        Heuristics map common software-development states to likely
        follow-up states. If no heuristic matches, every valid next
        state is listed with a generic reason.

        Args:
            entity_id: The entity to advise.

        Returns:
            A list of suggestion dicts, each with ``to`` (state name)
            and ``reason`` (why this is a good next step).
        """
        entity = self._entities.get(entity_id)
        if not entity:
            return []

        valid = self.get_valid_next_states(entity_id)
        state = entity.current_state
        if not state or not valid:
            return []

        # Heuristic map: current state → suggested transitions.
        suggestions_map: dict[str, list[dict[str, str]]] = {
            "Tests Failed": [
                {"to": "In Development",
                 "reason": "Return to development to investigate and fix the failing tests."},
                {"to": "Dependencies Updated",
                 "reason": "Test failures can stem from stale or incompatible dependencies."},
            ],
            "Tests Passing": [
                {"to": "Ready for Merge",
                 "reason": "All tests pass — the change is ready for peer review and merge."},
                {"to": "Ready to Deploy",
                 "reason": "Tests pass and the change can proceed to deployment."},
            ],
            "In Development": [
                {"to": "Code Review",
                 "reason": "Development is complete; submit for peer review."},
                {"to": "Testing",
                 "reason": "Validate the changes before moving forward."},
            ],
            "Code Review": [
                {"to": "Testing",
                 "reason": "Code has been reviewed and approved; run tests to verify."},
                {"to": "In Development",
                 "reason": "Changes were requested during review; return to development."},
            ],
            "Dependencies Updated": [
                {"to": "Testing",
                 "reason": "Dependencies have been updated; run the test suite."},
            ],
            "Ready for Merge": [
                {"to": "Merged",
                 "reason": "The branch is ready to merge into the main line."},
            ],
            "Merged": [
                {"to": "Deployed",
                 "reason": "The merged changes are ready for deployment."},
            ],
            "Ready to Deploy": [
                {"to": "Deployed",
                 "reason": "All checks pass and the build is ready for production."},
            ],
            "Deployed": [
                {"to": "Healthy",
                 "reason": "Confirm the deployment is running correctly."},
                {"to": "Rolled Back",
                 "reason": "Something went wrong — roll back to the previous stable version."},
            ],
            "Healthy": [
                {"to": "Monitoring",
                 "reason": "Set up or continue monitoring the healthy deployment."},
            ],
            "Active": [
                {"to": "In Development",
                 "reason": "Project is active and ready for the next development cycle."},
            ],
            "Draft": [
                {"to": "Active",
                 "reason": "Move from planning to active development."},
            ],
        }

        matched = suggestions_map.get(state, [])
        suggestions = [s for s in matched if s["to"] in valid]

        if not suggestions:
            suggestions = [
                {"to": s, "reason": "Valid next state per the defined workflow."}
                for s in sorted(valid)
            ]

        return suggestions

    def get_state_insights(
        self, entity_id: str,
    ) -> dict[str, Any]:
        """Aggregate narrative, dwell times, and suggestions into one dict.

        This is a convenience wrapper that calls the three methods above
        and returns a structured block suitable for prompt injection.

        Args:
            entity_id: The entity to analyse.

        Returns:
            A dict with keys ``narrative``, ``dwell_times``, and
            ``suggestions``.  Returns an empty dict when the entity is
            unknown.
        """
        entity = self._entities.get(entity_id)
        if not entity:
            return {}

        return {
            "narrative": self.get_state_narrative(entity_id),
            "dwell_times": self.get_state_dwell_times(entity_id),
            "suggestions": self.suggest_state_path(entity_id),
        }

    # -- Convenience queries --------------------------------------------------

    def get_project_context(self, project_name: str) -> dict[str, Any]:
        """Return a structured context dict for a project, suitable for
        injecting into the planner prompt.

        Includes the project's current state and recent state transitions
        so the planner can reason about project evolution.
        """
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

        ctx: dict[str, Any] = {
            "project": proj.name,
            "language": proj.attributes.get("language", "unknown"),
            "files": files[:20],
            "tools": tools[:10],
            "available_providers": providers,
        }

        state = proj.current_state
        if state:
            ctx["current_state"] = state
            valid = self.get_valid_next_states(proj.id)
            if valid:
                ctx["valid_next_states"] = sorted(valid)
            timeline = proj.get_state_history(limit=5)
            if len(timeline) > 1:
                ctx["state_timeline"] = [
                    {"state": s.state_name, "at": s.timestamp.isoformat()}
                    for s in timeline
                ]

            # State reasoning — narrative, dwell times, suggested path.
            narrative = self.get_state_narrative(proj.id)
            if narrative:
                ctx["state_narrative"] = narrative
            dwell = self.get_state_dwell_times(proj.id)
            if dwell:
                ctx["state_dwell_times"] = dwell
            suggestions = self.suggest_state_path(proj.id)
            if suggestions:
                ctx["state_suggestions"] = suggestions

        return ctx

    # -- Serialisation helpers ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": {
                eid: {
                    "id": e.id,
                    "type": e.entity_type.value,
                    "name": e.name,
                    "attributes": e.attributes,
                    "current_state": e.current_state,
                    "state_history": [
                        {
                            "state": s.state_name,
                            "at": s.timestamp.isoformat(),
                            "metadata": s.metadata,
                        }
                        for s in e.state_history
                    ],
                    "transitions": [
                        {
                            "from": t.from_state,
                            "to": t.to_state,
                            "at": t.timestamp.isoformat(),
                            "trigger": t.trigger,
                            "metadata": t.metadata,
                        }
                        for t in e.transitions
                    ],
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
