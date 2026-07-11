"""Agent coordinator.

The :class:`Coordinator` is the high-level orchestrator for the agent
framework. It is the single entry point that the API layer calls instead
of ``ChatService`` directly.

The coordinator:

1. Creates an :class:`AgentState` for the request.
2. Loads memory context via the :class:`MemoryManager`.
3. Calls the :class:`Planner` to decompose the goal into tasks.
4. For each task, calls the :class:`Executor` to run it.
5. After each task, calls :class:`Reflection` to evaluate quality.
6. If reflection indicates more work is needed, plans and executes
   follow-up tasks.
7. Repeats the plan → execute → reflect loop up to ``max_iterations``.
8. Returns the final assembled response.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from app.agents.base import AgentConfig
from app.agents.executor import Executor
from app.agents.goal_manager import GoalManager
from app.agents.learning_manager import LearningManager
from app.agents.memory_manager import MemoryManager
from app.agents.models.execution import ReflectionDecision, ReflectionResult
from app.agents.models.goal import Priority
from app.agents.models.policy import ExecutionMode, ExecutionPolicy
from app.agents.models.plan import Plan
from app.agents.models.strategy import Strategy
from app.agents.models.task import Task, TaskStatus
from app.memory.state_machine import build_default_state_machines, StateMachineDefinition
from app.agents.plan_simulator import PlanSimulator
from app.agents.planner import Planner
from app.agents.reflection import Reflection
from app.agents.state import AgentState
from app.agents.strategy_engine import StrategyEngine
from app.agents.task_graph import ExecutionGraph, TaskGraph
from app.core.logging import get_logger
from app.core.tracing import get_tracer
from app.domain.enums import MessageRole
from app.domain.message import Message, TextBlock
from app.llm.models import CompletionRequest, CompletionResponse, GenerationParams
from app.llm.provider_metrics import ProviderMetricsTracker
from app.llm.router import LLMRouter
from app.observability import (
    memory_retrieval_duration,
    memory_store_duration,
    planning_duration,
    reflection_outcomes_total,
    tool_execution_duration,
)

if TYPE_CHECKING:
    from app.llm.model_selector import ModelSelector
    from app.tools.capabilities import CapabilityRegistry

logger = get_logger(__name__)


@dataclass
class CoordinatorResult:
    """The final output of a coordinator run.

    Attributes:
        final_answer: The assembled response text.
        plan: The plan that was executed (for auditing).
        state: The final agent state.
        iterations: Number of plan→execute→reflect cycles.
        final_decision: The last reflection decision before stopping.
        timeline: Human-readable trace tree, if tracing was active.
    """

    final_answer: str
    plan: Plan | None
    state: AgentState
    iterations: int
    final_decision: ReflectionDecision = ReflectionDecision.ACCEPT
    timeline: str = ""
    goal_id: str | None = None


class Coordinator:
    """Orchestrates the full agent lifecycle.

    Usage::

        coordinator = Coordinator(
            planner=planner,
            executor=executor,
            reflection=reflection,
            memory_manager=memory_manager,
            llm_router=router,
            config=AgentConfig(model="llama3.1"),
        )
        result = await coordinator.run(
            conversation_id="...",
            goal="Research RTX 5070 laptops",
        )
        print(result.final_answer)
    """

    def __init__(
        self,
        planner: Planner,
        executor: Executor,
        reflection: Reflection,
        memory_manager: MemoryManager,
        llm_router: LLMRouter,
        capability_registry: CapabilityRegistry | None = None,
        model_selector: ModelSelector | None = None,
        learning_manager: LearningManager | None = None,
        plan_simulator: PlanSimulator | None = None,
        metrics_tracker: ProviderMetricsTracker | None = None,
        goal_manager: GoalManager | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._reflection = reflection
        self._memory_manager = memory_manager
        self._llm_router = llm_router
        self._capability_registry = capability_registry
        self._model_selector = model_selector
        self._learning_manager = learning_manager
        self._plan_simulator = plan_simulator
        self._metrics_tracker = metrics_tracker
        self._goal_manager = goal_manager
        self._config = config or AgentConfig()
        self._strategy_engine = StrategyEngine(
            learning_manager=learning_manager,
            experience_graph=learning_manager._experience_graph if learning_manager else None,
            world_model=learning_manager._world_model if learning_manager else None,
            metrics_tracker=metrics_tracker,
        )

        # Register default state machines on the world model.
        wm = self._get_world_model()
        if wm is not None:
            for sm in build_default_state_machines().values():
                wm.register_state_machine(sm)

    async def run(
        self,
        conversation_id: str,
        goal: str,
        policy: ExecutionPolicy | str | None = None,
        objective_id: str | None = None,
        goal_description: str | None = None,
        goal_priority: Priority = Priority.MEDIUM,
        goal_urgency: datetime | None = None,
        goal_dependencies: list[str] | None = None,
    ) -> CoordinatorResult:
        """Execute the full agent lifecycle for a user goal.

        Args:
            conversation_id: The active conversation.
            goal: The user's request.
            policy: Execution policy — a predefined mode (``"fast"``,
                ``"balanced"``, ``"autonomous"``, ``"research"``,
                ``"coding"``) or an :class:`ExecutionPolicy` instance.
                ``None`` defaults to ``balanced``.
            objective_id: If set, the goal is tracked under this objective
                in the goal hierarchy (persistent across sessions).
            goal_description: Human-readable label for the goal node
                (defaults to *goal* when not set).
            goal_priority: Priority level for this goal.
            goal_urgency: Deadline for this goal.
            goal_dependencies: Goal IDs that must complete first.

        Returns:
            The final answer, plan, and execution state.
        """
        # Build strategy early so it can influence policy selection.
        strategy: Strategy = await self._strategy_engine.build_strategy(goal)

        if policy is None:
            mode = strategy.recommended_mode
            if mode is not None:
                resolved_policy = ExecutionMode(mode).policy()
            else:
                resolved_policy = ExecutionMode.BALANCED.policy()
        elif isinstance(policy, str):
            resolved_policy = ExecutionMode(policy).policy()
        else:
            resolved_policy = policy

        goal_node_id: str | None = None
        if self._goal_manager is not None and objective_id is not None:
            desc = goal_description or goal[:200]
            goal_node = await self._goal_manager.start_or_resume_goal(
                objective_id, desc,
                priority=goal_priority,
                urgency=goal_urgency,
                metadata={"dependencies": goal_dependencies} if goal_dependencies else None,
            )
            goal_node_id = goal_node.id

        state = AgentState(
            conversation_id=conversation_id,
            goal=goal,
            max_iterations=(
                resolved_policy.max_iterations or AgentState.__init__.__defaults__[2]
            ),
        )

        tracer = get_tracer()
        with tracer.span("Request", category="agent", conversation_id=conversation_id):
            # Load memory context — planner always reasons with memory.
            _t0 = time.monotonic()
            with tracer.span("Memory Retrieval", category="memory"):
                state.memory_context = await self._memory_manager.get_context(
                    conversation_id=conversation_id,
                    goal=goal,
                ) or ""
            memory_retrieval_duration.observe(time.monotonic() - _t0)

            # Log the strategy (built above) but don't rebuild.
            logger.info(
                "coordinator.strategy",
                domain=strategy.goal_domain,
                recommended_mode=strategy.recommended_mode,
                pattern_count=strategy.source_pattern_count,
            )

            # Main agent loop.
            while not state.is_exhausted:
                logger.info(
                    "coordinator.iteration_start",
                    iteration=state.iteration + 1,
                    max_iterations=state.max_iterations,
                    goal=goal,
                )

                # Step 1 — Plan (or skip with a default single-task plan).
                _t0 = time.monotonic()
                if resolved_policy.planning:
                    with tracer.span("Planner", category="agent"):
                        plan = await self._planner.plan(
                            goal=goal,
                            memory_context=state.memory_context,
                            strategy=strategy,
                        )
                else:
                    plan = Plan(
                        goal=goal,
                        tasks=[Task(
                            id=str(uuid4()),
                            description=goal,
                            capability=None,
                            dependencies=[],
                        )],
                    )

                # Step 1a — Generate alternative plans and pick the best via simulation.
                if resolved_policy.simulation and self._plan_simulator is not None:
                    plans = [plan]
                    for alt_temp in (0.5, 0.9):
                        try:
                            alt = await self._planner.plan(
                                goal=goal,
                                memory_context=state.memory_context,
                                strategy=strategy,
                                temperature=alt_temp,
                            )
                            plans.append(alt)
                        except Exception:
                            continue

                    best_plan = plan
                    best_score = -1.0
                    for candidate in plans:
                        sim = await self._plan_simulator.simulate(
                            goal, candidate, strategy=strategy,
                        )
                        score = sim.overall_success_probability * 0.7 + (
                            1.0 - min(sim.total_estimated_cost_ms / 60000.0, 1.0)
                        ) * 0.3
                        if score > best_score:
                            best_score = score
                            best_plan = candidate

                    if best_plan is not plan:
                        logger.info(
                            "coordinator.plan_selected",
                            score=round(best_score, 2),
                        )
                    plan = best_plan

                planning_duration.observe(time.monotonic() - _t0)
                state.plan = plan
                graph = self._build_graph(plan)

                # Step 1b — Simulate the plan for logging (heuristic cost/success estimate).
                if resolved_policy.simulation and self._plan_simulator is not None:
                    with tracer.span("Plan Simulation", category="agent"):
                        sim = await self._plan_simulator.simulate(goal, plan)
                    logger.info(
                        "coordinator.simulation",
                        total_cost_s=round(sim.total_estimated_cost_ms / 1000, 1),
                        success_rate=round(sim.overall_success_probability, 2),
                        confidence=round(sim.confidence, 2),
                        notes=sim.notes,
                    )

                    # Feed simulation estimates back into the goal for prioritisation.
                    if goal_node_id is not None and self._goal_manager is not None:
                        await self._goal_manager.update_goal_estimates(
                            goal_node_id,
                            estimated_cost_ms=sim.total_estimated_cost_ms,
                            success_probability=sim.overall_success_probability,
                        )

                # Step 2 — Execute tasks with retry + checkpoint support.
                while not graph.is_complete():
                    ready_tasks = graph.get_ready()
                    if not ready_tasks:
                        logger.warning("coordinator.no_ready_tasks")
                        break

                    for task in ready_tasks:
                        graph.update_status(task.id, TaskStatus.RUNNING)

                        _t0 = time.monotonic()
                        with tracer.span(f"Execute {task.id}", category="execution", task_id=task.id):
                            result = await self._executor.execute(task)
                            state.completed_results[result.task_id] = result
                        tool_execution_duration.labels(
                            tool_name=result.tool_name or "llm",
                            status=result.status.value,
                        ).observe(time.monotonic() - _t0)

                        if result.status is TaskStatus.COMPLETED:
                            graph.update_status(task.id, TaskStatus.COMPLETED)
                            self._apply_intended_transition(task)
                        else:
                            retry_status = graph.record_failure(
                                task.id,
                                error=result.error or "execution failed",
                            )
                            if retry_status is TaskStatus.PENDING:
                                logger.info(
                                    "coordinator.retry",
                                    task_id=task.id,
                                    retry_count=graph._nodes.get(
                                        task.id, type("", (), {"retry_count": 0})()
                                    ).retry_count,  # type: ignore
                                )
                                continue  # skip reflection; will re-run

                        # Step 3 — Reflect (before storing so assessment is available).
                        if resolved_policy.reflection:
                            with tracer.span("Reflection", category="agent"):
                                assessment = await self._reflection.reflect(result)
                            state.reflections[task.id] = assessment
                        else:
                            assessment = ReflectionResult(
                                decision=ReflectionDecision.ACCEPT,
                                confidence=0.5,
                                reason="Reflection disabled by policy.",
                            )
                        reflection_outcomes_total.labels(
                            decision=assessment.decision.value,
                        ).inc()

                        # Record provider metrics for data-driven model selection.
                        if self._metrics_tracker is not None:
                            latency_ms = (
                                (result.completed_at - result.started_at).total_seconds() * 1000.0
                                if result.completed_at and result.started_at
                                else 0.0
                            )
                            self._metrics_tracker.record(
                                provider_id="",
                                model_id="",
                                latency_ms=latency_ms,
                                success=result.status is TaskStatus.COMPLETED,
                                reflection_confidence=assessment.confidence,
                            )

                        # Store in memory with feedback loop + assessment.
                        _t0 = time.monotonic()
                        with tracer.span("Memory Store", category="memory"):
                            await self._memory_manager.store_result(
                                conversation_id=conversation_id,
                                result=result,
                                assessment=assessment,
                            )
                        memory_store_duration.observe(time.monotonic() - _t0)

                        # Log atomic action in the goal hierarchy.
                        if goal_node_id is not None and self._goal_manager is not None:
                            latency_ms = (
                                (result.completed_at - result.started_at).total_seconds() * 1000.0
                                if result.completed_at and result.started_at
                                else 0.0
                            )
                            await self._goal_manager.record_action(
                                goal_id=goal_node_id,
                                task_id=task.id,
                                tool_name=result.tool_name or task.capability,
                                description=task.description[:200],
                                input_summary=task.description[:200],
                                output_summary=(result.output or "")[:500],
                                status=result.status.value,
                                latency_ms=latency_ms,
                                reflection_confidence=assessment.confidence,
                            )

                        handled = self._handle_reflection_decision(
                            decision=assessment.decision,
                            assessment=assessment,
                            graph=graph,
                            task=task,
                        )

                        # Record anti-pattern on failure.
                        if (
                            resolved_policy.learning
                            and self._learning_manager is not None
                            and result.status is TaskStatus.FAILED
                        ):
                            warning = (
                                f"Avoid {task.capability or 'this approach'} "
                                f"for '{task.description[:80]}' — "
                                f"{assessment.reason or result.error or 'failed'}"
                            )
                            self._learning_manager.record_failure(
                                goal=goal,
                                warning=warning,
                                failure_reason=result.error or assessment.reason or "unknown",
                            )

                        if handled == "abort":
                            state.record_error(assessment.reason)
                            break

                state.iteration += 1

                # Check whether the result is satisfactory.
                if resolved_policy.reflection:
                    with tracer.span("Plan-level Reflection", category="agent"):
                        final_assessment = await self._reflect_on_plan(state)
                    if final_assessment.decision is ReflectionDecision.ACCEPT:
                        break
                else:
                    # Without reflection, accept after first successful iteration.
                    if state.completed_results:
                        break

        # Extract learning pattern after successful execution.
        if resolved_policy.learning and self._learning_manager is not None and state.completed_results:
            with tracer.span("Pattern Extraction", category="learning"):
                results_list = list(state.completed_results.values())
                reflections_list = list(state.reflections.values())
                if state.plan is not None:
                    await self._learning_manager.extract_pattern(
                        goal=goal,
                        plan=state.plan,
                        results=results_list,
                        reflections=reflections_list,
                    )

        # Assemble final answer.
        final_answer = self._assemble_answer(state)

        # Update goal node in the hierarchy.
        if goal_node_id is not None and self._goal_manager is not None:
            if state.has_errors:
                await self._goal_manager.fail_goal(
                    goal_node_id,
                    error="; ".join(state.errors) if state.errors else "Unknown error",
                )
            else:
                await self._goal_manager.complete_goal(
                    goal_node_id,
                    result_summary=final_answer[:500],
                )
        logger.info(
            "coordinator.complete",
            iterations=state.iteration,
            tasks_completed=state.completed_task_count,
            tasks_failed=state.failed_task_count,
        )

        return CoordinatorResult(
            final_answer=final_answer,
            plan=state.plan,
            state=state,
            iterations=state.iteration,
            timeline=tracer.render_tree(),
            final_decision=(
                ReflectionDecision.ABORT
                if state.has_errors
                else ReflectionDecision.ACCEPT
            ),
            goal_id=goal_node_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_world_model(self):
        """Access the world model via the learning manager, if available."""
        if self._learning_manager is not None:
            return self._learning_manager._world_model
        return None

    def _apply_intended_transition(self, task: Task) -> None:
        """Apply the task's ``intended_transition`` metadata, if present."""
        transition_meta = task.metadata.get("intended_transition")
        if not transition_meta or not isinstance(transition_meta, dict):
            return
        entity_id = transition_meta.get("entity_id")
        to_state = transition_meta.get("to_state")
        if not entity_id or not to_state:
            return
        wm = self._get_world_model()
        if wm is None:
            return
        wm.transition_entity(
            entity_id,
            to_state,
            trigger=f"task:{task.id}",
            metadata={"task_description": task.description[:200]},
        )

    @staticmethod
    def _build_graph(plan: Plan) -> ExecutionGraph:
        """Build an :class:`ExecutionGraph` from a plan.

        Args:
            plan: The plan to convert into a graph.

        Returns:
            An execution graph with all tasks, dependencies, and retry config.
        """
        graph = ExecutionGraph()
        for task in plan.tasks:
            graph.add_task(task, depends_on=task.dependencies, max_retries=2)
        return graph

    @staticmethod
    def _handle_reflection_decision(
        decision: ReflectionDecision,
        assessment: ReflectionResult,
        graph: ExecutionGraph,
        task: Task,
    ) -> str:
        """Handle a reflection decision by mutating the execution graph.

        Args:
            decision: The reflection decision.
            assessment: The full reflection result.
            graph: The execution graph to mutate.
            task: The task that was reflected on.

        Returns:
            A signal string: ``"continue"``, ``"abort"``.
        """
        match decision:
            case ReflectionDecision.ACCEPT:
                return "continue"

            case ReflectionDecision.RETRY:
                follow_up = Task(
                    id=str(uuid4()),
                    description=task.description,
                    dependencies=[task.id],
                    capability=task.capability,
                    tool_name=task.tool_name,
                )
                graph.add_task(follow_up, depends_on=[task.id])
                return "continue"

            case ReflectionDecision.REPLAN:
                for desc in assessment.next_tasks:
                    follow_up = Task(
                        id=str(uuid4()),
                        description=desc,
                        dependencies=[task.id],
                    )
                    graph.add_task(follow_up, depends_on=[task.id])
                return "continue"

            case ReflectionDecision.ASK_USER:
                return "continue"

            case ReflectionDecision.ABORT:
                return "abort"

        return "continue"

    async def _reflect_on_plan(self, state: AgentState) -> ReflectionResult:
        """Evaluate overall progress using the LLM.

        Args:
            state: The current agent state.

        Returns:
            A reflection result.
        """
        summary = self._build_summary(state)
        if not summary:
            return ReflectionResult(
                decision=ReflectionDecision.ACCEPT,
                reason="No tasks were executed.",
                confidence=1.0,
            )

        messages = [
            Message(
                id=str(uuid4()),
                conversation_id="",
                role=MessageRole.USER,
                content=[TextBlock(text=summary)],
            ),
        ]

        if self._model_selector is not None:
            model, provider = await self._model_selector.select_for_reflection()
        else:
            model = self._config.model
            provider = self._config.provider

        request = CompletionRequest(
            messages=messages,
            model=model,
            provider=provider,
            params=GenerationParams(
                temperature=0.3,
                max_tokens=512,
            ),
        )

        try:
            response: CompletionResponse = await self._llm_router.generate(request)
            text = ""
            for block in response.message.content:
                if isinstance(block, TextBlock):
                    text += block.text

            decision = (
                ReflectionDecision.ACCEPT
                if "no" in text.lower()[:100]
                else ReflectionDecision.RETRY
            )
            return ReflectionResult(
                decision=decision,
                reason=text[:500],
                confidence=0.7,
            )
        except Exception:
            return ReflectionResult(
                decision=ReflectionDecision.ACCEPT,
                reason="Reflection LLM call failed; accepting current result.",
                confidence=0.5,
            )

    def _build_summary(self, state: AgentState) -> str:
        """Build a summary of execution for reflection.

        Args:
            state: The current agent state.

        Returns:
            A summary string, or empty string if no tasks executed.
        """
        if not state.completed_results:
            return ""

        lines: list[str] = [
            "Summary of completed tasks:",
        ]
        for task_id, result in state.completed_results.items():
            status = result.status.value.upper()
            output_preview = (result.output or "")[:200]
            lines.append(f"  {task_id} [{status}]: {output_preview}")

        lines.append("\nIs the goal fully achieved? Answer 'accept' or 'retry' with a brief reason.")
        return "\n".join(lines)

    def _assemble_answer(self, state: AgentState) -> str:
        """Assemble the final answer from completed task results.

        Args:
            state: The final agent state.

        Returns:
            The assembled answer text.
        """
        parts: list[str] = []

        for _task_id, result in state.completed_results.items():
            if result.output:
                parts.append(result.output)

        if not parts:
            if state.has_errors:
                return (
                    "I encountered errors while processing your request:\n"
                    + "\n".join(f"- {e}" for e in state.errors)
                )
            return "I was unable to produce a result. Please try rephrasing your request."

        return "\n\n".join(parts)
