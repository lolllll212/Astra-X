from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.memory.world_model import EntityType


@dataclass
class StateTransition:
    """An explicit record of one entity state transition.

    Unlike :class:`EntityState` (which only records the destination),
    this captures the full before/after/trigger picture.
    """

    from_state: str | None
    to_state: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    trigger: str = "manual"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateMachineDefinition:
    """Defines valid states and transitions for an entity type.

    Usage::

        sm = StateMachineDefinition(EntityType.PROJECT, initial_state="Draft")
        sm.add_transition("Draft", "Active")
        sm.add_transition("Active", "Maintenance")

        world_model.register_state_machine(sm)
    """

    entity_type: EntityType
    states: set[str] = field(default_factory=set)
    transitions: dict[str, set[str]] = field(default_factory=dict)
    initial_state: str | None = None

    def add_transition(self, from_state: str, to_state: str) -> None:
        self.states.add(from_state)
        self.states.add(to_state)
        self.transitions.setdefault(from_state, set()).add(to_state)

    def validate(self, from_state: str | None, to_state: str) -> bool:
        if from_state is None:
            return self.initial_state is None or to_state == self.initial_state
        return to_state in self.transitions.get(from_state, set())

    def get_valid_next_states(self, current_state: str | None) -> set[str]:
        if current_state is None:
            return {self.initial_state} if self.initial_state else set()
        return self.transitions.get(current_state, set())


# ---------------------------------------------------------------------------
# Default state machines
# ---------------------------------------------------------------------------


def build_default_state_machines() -> dict[EntityType, StateMachineDefinition]:
    """Return default state machines for common entity types.

    Callers can extend or replace these before registering on the
    :class:`WorldModel`.
    """
    from app.memory.world_model import EntityType

    return {
        EntityType.PROJECT: _build_dev_workflow(EntityType.PROJECT),
        EntityType.REPOSITORY: _build_dev_workflow(EntityType.REPOSITORY),
        EntityType.MISSION: _build_mission_workflow(EntityType.MISSION),
    }


def _build_mission_workflow(entity_type: EntityType | None = None) -> StateMachineDefinition:
    """Mission lifecycle state machine supporting long-running projects.

    Allows missions to be paused and resumed:
        Mission → Paused → Resume tomorrow → Continue
    """
    from app.agents.models.goal import MissionStatus
    from app.memory.world_model import EntityType

    sm = StateMachineDefinition(
        entity_type=entity_type or EntityType.MISSION,
        initial_state=MissionStatus.ACTIVE,
    )

    # Active → Paused (break due to external reasons)
    sm.add_transition(MissionStatus.ACTIVE, MissionStatus.PAUSED)

    # Paused → Active (resume after break)
    sm.add_transition(MissionStatus.PAUSED, MissionStatus.ACTIVE)

    # Completed or abandoned end states
    sm.add_transition(MissionStatus.ACTIVE, MissionStatus.COMPLETED)
    sm.add_transition(MissionStatus.ACTIVE, MissionStatus.ABANDONED)
    sm.add_transition(MissionStatus.PAUSED, MissionStatus.COMPLETED)
    sm.add_transition(MissionStatus.PAUSED, MissionStatus.ABANDONED)

    return sm


def _build_dev_workflow(entity_type: EntityType | None = None) -> StateMachineDefinition:
    """Comprehensive development-workflow state machine.

    Covers the full lifecycle of a code project/repository from draft
    through development, testing, review, deployment, and archival.
    Matches the user's example flow:

        Repository → Dirty → Tests Failed → Dependencies Updated → Ready
    """
    from app.memory.world_model import EntityType

    sm = StateMachineDefinition(
        entity_type=entity_type or EntityType.PROJECT,
        initial_state="Draft",
    )

    # --- Draft → Active ---
    sm.add_transition("Draft", "Active")
    sm.add_transition("Draft", "Archived")

    # --- Active → development loop ---
    sm.add_transition("Active", "In Development")

    # === Development loop ===
    # Dirty → testing
    sm.add_transition("In Development", "Code Review")
    sm.add_transition("In Development", "Testing")

    # Code review → rework or forward
    sm.add_transition("Code Review", "In Development")
    sm.add_transition("Code Review", "Testing")

    # Testing → pass / fail
    sm.add_transition("Testing", "Tests Passing")
    sm.add_transition("Testing", "Tests Failed")

    # Tests Failed → fix deps or fix code
    sm.add_transition("Tests Failed", "Dependencies Updated")
    sm.add_transition("Tests Failed", "In Development")

    # Dependencies Updated → re-test
    sm.add_transition("Dependencies Updated", "Testing")

    # Tests Passing → ready to ship
    sm.add_transition("Tests Passing", "Ready for Merge")
    sm.add_transition("Tests Passing", "Deployed")

    # Merge → deploy
    sm.add_transition("Ready for Merge", "Merged")
    sm.add_transition("Merged", "Deployed")

    # === Post-deployment ===
    sm.add_transition("Deployed", "Maintenance")
    sm.add_transition("Deployed", "Archived")
    sm.add_transition("Maintenance", "In Development")
    sm.add_transition("Maintenance", "Archived")

    # Direct archive from most states
    sm.add_transition("In Development", "Archived")
    sm.add_transition("Code Review", "Archived")
    sm.add_transition("Testing", "Archived")
    sm.add_transition("Tests Failed", "Archived")
    sm.add_transition("Ready for Merge", "Archived")

    return sm


def format_transition_rules_for_prompt(
    definitions: dict[EntityType, StateMachineDefinition],
) -> str:
    """Format registered state machines as text for the planner prompt."""
    if not definitions:
        return ""
    lines: list[str] = ["State machine definitions:"]
    for entity_type, sm in definitions.items():
        label = entity_type.value.capitalize()
        rules = []
        for from_state, to_states in sm.transitions.items():
            for to in sorted(to_states):
                rules.append(f"  {from_state} -> {to}")
        initial = f", initial={sm.initial_state}" if sm.initial_state else ""
        lines.append(f"- {label}{initial}:")
        lines.extend(rules)
    return "\n".join(lines)
