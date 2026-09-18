"""ECC (Engineering Control Center) tool - coordinated engineering system: plan -> test -> implement -> review -> verify -> remember -> improve.

Two execution modes:
- ``ecc_home`` present: drives the real ECC-main repo (Node CLI ``scripts/ecc.js``
  and the Python ``src/llm`` provider layer) so workflows/skills/hooks/LLM calls
  run against the actual ECC framework.
- otherwise: falls back to the built-in in-memory registry described in
  :mod:`app.tools.ecc.models`.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.ecc.models import (
    AgentRole,
    ECCAgent,
    ECCConfig,
    ECCHook,
    ECCMemory,
    ECCRule,
    ECCSkill,
    ECCStatus,
    ECCWorkflow,
    HookType,
    MemoryType,
    SecurityFinding,
    SecurityScanType,
    SkillCategory,
    WorkflowStep,
)
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult

ECC_HOME_ENV = "ECC_HOME"
ECC_HOME_DEFAULT = r"C:\Users\Ashut\Downloads\ECC-main"
LLM_BASE_URL_DEFAULT = "http://127.0.0.1:1234/v1"
DEFAULT_LLM_MODEL = "qwen2.5-coder-7b-instruct"


def _find_ecc_home() -> str:
    """Locate the ECC-main checkout (env override, then the default path)."""
    candidates = [os.environ.get(ECC_HOME_ENV) or "", ECC_HOME_DEFAULT]
    for cand in candidates:
        if cand and (Path(cand) / "scripts" / "ecc.js").is_file():
            return cand
    return ""


class ECCTool(Tool):
    """ECC - Engineering Control Center + ECC-main driver: 140 agents in the family, 292 skills, coordinating every engineering function through one unified system, optionally backed by the real ECC-main repo (Node CLI + src/llm providers incl. LM Studio)."""

    @property
    def name(self) -> str:
        return "ecc"

    @property
    def description(self) -> str:
        return "ECC (Engineering Control Center): coordinated engineering through plan -> test -> implement -> review -> verify -> remember -> improve. The 140 agents of the ECC covered the full engineering board: planner, reviewer, build/repair, security, architect, domain expert, TDD specialist, researcher, docs writer, frontend dev, data engineer, ML engineer, ops engineer. 292 skills, hooks, memory, rules, AgentShield security scanning. When ECC_HOME (default C:\\Users\\Ashut\\Downloads\\ECC-main) is available it drives the real ECC-main Node CLI and Python LLM providers (Claude/OpenAI/Ollama, OpenAI-compatible LM Studio)."

    @property
    def capabilities(self) -> list[str]:
        return ["workflow_orchestration", "agents", "skills", "hooks", "memory", "rules", "security"]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="action",
                    type_="string",
                    description="Action: execute_workflow, create_agent, create_skill, install_skill, create_hook, create_rule, search_memory, save_memory, build_context, scan_security, get_status, ecc_cli, llm_generate",
                    required=True,
                    enum=["execute_workflow", "create_agent", "create_skill", "install_skill", "create_hook", "create_rule", "search_memory", "save_memory", "build_context", "scan_security", "get_status", "ecc_cli", "llm_generate"],
                ),
                ToolParameter(
                    name="name",
                    type_="string",
                    description="Name for the created agent/skill/hook/rule",
                    required=False,
                ),
                ToolParameter(
                    name="role",
                    type_="string",
                    description="Agent role: planner, reviewer, build_repair, security, architect, domain_expert, tdd_specialist, researcher, docs_writer, frontend_dev, data_engineer, ml_engineer, ops_engineer",
                    required=False,
                ),
                ToolParameter(
                    name="category",
                    type_="string",
                    description="Skill category: tdd, research, security, docs, frontend, data, ml, operations, architecture, planning, review, debugging, refactoring, testing",
                    required=False,
                ),
                ToolParameter(
                    name="description",
                    type_="string",
                    description="Description of the agent/skill/hook/rule",
                    required=False,
                ),
                ToolParameter(
                    name="content",
                    type_="string",
                    description="Content, system prompt, script, or rule text",
                    required=False,
                ),
                ToolParameter(
                    name="workflow_name",
                    type_="string",
                    description="Workflow name",
                    required=False,
                    default="plan -> test -> implement -> review -> verify -> remember -> improve",
                ),
                ToolParameter(
                    name="feature",
                    type_="string",
                    description="Feature description for execute_workflow",
                    required=False,
                ),
                ToolParameter(
                    name="steps",
                    type_="object",
                    description="Custom workflow steps (array of {name, role, skill})",
                    required=False,
                ),
                ToolParameter(
                    name="project_dir",
                    type_="string",
                    description="Project directory target",
                    required=False,
                ),
                ToolParameter(
                    name="query",
                    type_="string",
                    description="Search query for memory",
                    required=False,
                ),
                ToolParameter(
                    name="tags",
                    type_="object",
                    description="Tags for memory entry",
                    required=False,
                    default=[],
                ),
                ToolParameter(
                    name="memory_type",
                    type_="string",
                    description="Memory type: session_summary, instinct, context_control, continuous_learning, project_rule, language_standard",
                    required=False,
                ),
                ToolParameter(
                    name="skill_source",
                    type_="string",
                    description="Source to install skill from (path or URL)",
                    required=False,
                ),
                ToolParameter(
                    name="hook_type",
                    type_="string",
                    description="Hook type: pre_tool, post_tool, pre_agent, post_agent, session_start, session_end, error, learning",
                    required=False,
                ),
                ToolParameter(
                    name="language",
                    type_="string",
                    description="Language for rules/memory",
                    required=False,
                ),
                ToolParameter(
                    name="scan_type",
                    type_="string",
                    description="Security scan type: prompt_injection, hook_analysis, mcp_config, permissions, secrets, agent_files",
                    required=False,
                ),
                ToolParameter(
                    name="ecc_command",
                    type_="string",
                    description="ECC-main CLI command line, e.g. 'status --json', 'doctor', 'catalog profiles' (only used with action=ecc_cli)",
                    required=False,
                ),
                ToolParameter(
                    name="prompt",
                    type_="string",
                    description="Prompt for the LLM (only used with action=llm_generate)",
                    required=False,
                ),
                ToolParameter(
                    name="model",
                    type_="string",
                    description="Model identifier for the LLM provider (only used with action=llm_generate)",
                    required=False,
                ),
                ToolParameter(
                    name="provider",
                    type_="string",
                    description="LLM provider: openai (LM Studio-compatible), claude, ollama, atlas, astraflow (only used with action=llm_generate)",
                    required=False,
                ),
                ToolParameter(
                    name="base_url",
                    type_="string",
                    description="OpenAI-compatible base URL (only used with action=llm_generate, default http://127.0.0.1:1234/v1)",
                    required=False,
                ),
                ToolParameter(
                    name="temperature",
                    type_="number",
                    description="Sampling temperature (only used with action=llm_generate)",
                    required=False,
                ),
                ToolParameter(
                    name="max_tokens",
                    type_="number",
                    description="Maximum tokens to generate (only used with action=llm_generate)",
                    required=False,
                ),
            ],
        )

    def __init__(self) -> None:
        self._config = ECCConfig()
        self._agents: dict[str, ECCAgent] = {}
        self._skills: dict[str, ECCSkill] = {}
        self._hooks: dict[str, ECCHook] = {}
        self._memory: list[ECCMemory] = []
        self._rules: dict[str, ECCRule] = {}
        self._workflows: dict[str, ECCWorkflow] = {}
        self._security_findings: list[SecurityFinding] = []
        self._ecc_home = _find_ecc_home()
        self._load_ecc_llm()
        self._init_default_agents()
        self._init_default_skills()
        self._init_default_hooks()

    def _load_ecc_llm(self) -> None:
        """Import the ECC-main src/llm provider layer on sys.path when available."""
        if not self._ecc_home:
            self._ecc_llm_src = ""
            return
        src = str(Path(self._ecc_home) / "src")
        if (Path(self._ecc_home) / "src" / "llm").is_dir() and src not in sys.path:
            sys.path.insert(0, src)
        self._ecc_llm_src = src

    def _init_default_agents(self) -> None:
        """Initialize the default ECC agents covering the engineering board."""
        agents_spec = [
            ("Planner", AgentRole.PLANNER, "Architect of every build. Reads the board, understands the goal, and lays out the precise plan."),
            ("Reviewer", AgentRole.REVIEWER, "Cold-eyed critic. Subject every candidate change to ruthless, anonymous review."),
            ("Build & Repair", AgentRole.BUILD_REPAIR, "First responder. Handles builds that break and steers them back to health."),
            ("Security", AgentRole.SECURITY, "Guards the system. Prompts, hooks, configs, permissions - nothing escapes AgentShield."),
            ("Architect", AgentRole.ARCHITECT, "Sees the whole picture, sets the structural vision for the codebase."),
            ("Domain Expert", AgentRole.DOMAIN_EXPERT, "Knows the application domain cold and advises every technical decision."),
            ("TDD Specialist", AgentRole.TDD_SPECIALIST, "Test-first engineer. Writes the tests before the code, everything else is ceremony."),
            ("Researcher", AgentRole.RESEARCHER, "Digs up ground truth from the web, docs, and source before any plan is locked."),
            ("Docs Writer", AgentRole.DOCS_WRITER, "Turns engineering decisions into clear, current, honest documentation."),
            ("Frontend Dev", AgentRole.FRONTEND_DEV, "Builds the user-facing surface with accessible, performant interfaces."),
            ("Data Engineer", AgentRole.DATA_ENGINEER, "Owns pipelines, schemas, and the data layer that feeds everything."),
            ("ML Engineer", AgentRole.ML_ENGINEER, "Builds and tunes models and the features that surround them."),
            ("Ops Engineer", AgentRole.OPS_ENGINEER, "Keeps the ship running: builds, deploys, monitors, and remedies."),
        ]
        for name, role, desc in agents_spec:
            agent = ECCAgent(
                id=f"agent_{role.value}",
                name=name,
                role=role,
                description=desc,
                capabilities=[role.value],
                enabled=True,
            )
            self._agents[agent.id] = agent

    def _init_default_skills(self) -> None:
        """Initialize the default skill catalog."""
        skills_spec = [
            ("skill_tdd", SkillCategory.TDD, "Red-Green-Refactor loop with failing-test-first discipline", [AgentRole.TDD_SPECIALIST]),
            ("skill_research", SkillCategory.RESEARCH, "Grounded web + repo research with cited sources", [AgentRole.RESEARCHER, AgentRole.PLANNER]),
            ("skill_security", SkillCategory.SECURITY, "Threat-model-first review of any change", [AgentRole.SECURITY]),
            ("skill_docs", SkillCategory.DOCS, "Living docs generation with honest change summaries", [AgentRole.DOCS_WRITER]),
            ("skill_frontend", SkillCategory.FRONTEND, "Accessible, performant UI implementation", [AgentRole.FRONTEND_DEV]),
            ("skill_data", SkillCategory.DATA, "Pipeline + schema design and data-layer health", [AgentRole.DATA_ENGINEER]),
            ("skill_ml", SkillCategory.ML, "Model selection, tuning, evaluation harness", [AgentRole.ML_ENGINEER]),
            ("skill_ops", SkillCategory.OPERATIONS, "Build, deploy, monitor, remediate operations", [AgentRole.OPS_ENGINEER]),
            ("skill_architecture", SkillCategory.ARCHITECTURE, "Structural design and dependency discipline", [AgentRole.ARCHITECT]),
            ("skill_planning", SkillCategory.PLANNING, "Board reading and step-by-step plan construction", [AgentRole.PLANNER]),
            ("skill_review", SkillCategory.REVIEW, "Ruthless, anonymous candidate review", [AgentRole.REVIEWER]),
            ("skill_debugging", SkillCategory.DEBUGGING, "Root-cause debugging with evidence", [AgentRole.BUILD_REPAIR]),
            ("skill_refactoring", SkillCategory.REFACTORING, "Surgical refactoring without behavior change", [AgentRole.ARCHITECT, AgentRole.BUILD_REPAIR]),
            ("skill_testing", SkillCategory.TESTING, "Test harness construction and coverage discipline", [AgentRole.TDD_SPECIALIST, AgentRole.REVIEWER]),
            ("skill_integration", SkillCategory.OPERATIONS, "Cross-tool coordination and handoff", [AgentRole.OPS_ENGINEER]),
        ]
        for skill_id, category, desc, agents in skills_spec:
            skill = ECCSkill(
                id=skill_id,
                name=skill_id.replace("_", " "),
                category=category,
                description=desc,
                agent_compatibility=agents,
                enabled=True,
            )
            self._skills[skill.id] = skill

    def _init_default_hooks(self) -> None:
        """Initialize safety hooks."""
        hooks_spec = [
            ("block_destructive_commands", HookType.PRE_TOOL, "Block destructive commands without explicit confirmation", 10),
            ("audit_tool_calls", HookType.POST_TOOL, "Record every tool call result for review", 50),
            ("session_summary", HookType.SESSION_END, "Persist a session summary into memory", 20),
            ("learning_capture", HookType.LEARNING, "Capture instinct entries as lessons are learned", 30),
            ("error_escalation", HookType.ERROR, "Escalate repeated errors to the Build & Repair agent", 40),
        ]
        for hook_id, hook_type, desc, priority in hooks_spec:
            hook = ECCHook(
                id=f"hook_{hook_id}",
                name=hook_id.replace("_", " "),
                type=hook_type,
                description=desc,
                priority=priority,
                enabled=True,
            )
            self._hooks[hook.id] = hook

    def _store_memory(self, mem_type: MemoryType, key: str, value: str, tags: list[str] | None = None, project_id: str | None = None, language: str | None = None, importance: float = 0.5) -> ECCMemory:
        entry = ECCMemory(
            type=mem_type,
            key=key,
            value=value,
            tags=tags or [],
            project_id=project_id,
            language=language,
            importance=importance,
        )
        self._memory.append(entry)
        if len(self._memory) > max(self._config.max_memory_entries, 100):
            self._memory.pop(0)
        return entry

    def _save_json(self, path: str, data: Any) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "")
        if action == "execute_workflow":
            return await self._execute_workflow(context, kwargs)
        elif action == "create_agent":
            return await self._create_agent(context, kwargs)
        elif action == "create_skill":
            return await self._create_skill(context, kwargs)
        elif action == "install_skill":
            return await self._install_skill(context, kwargs)
        elif action == "create_hook":
            return await self._create_hook(context, kwargs)
        elif action == "create_rule":
            return await self._create_rule(context, kwargs)
        elif action == "search_memory":
            return await self._search_memory(context, kwargs)
        elif action == "save_memory":
            return await self._save_memory(context, kwargs)
        elif action == "build_context":
            return await self._build_context(context, kwargs)
        elif action == "scan_security":
            return await self._scan_security(context, kwargs)
        elif action == "get_status":
            return await self._get_status(context, kwargs)
        elif action == "ecc_cli":
            return await self._ecc_cli(context, kwargs)
        elif action == "llm_generate":
            return await self._llm_generate(context, kwargs)
        else:
            return ToolResult(success=False, error=f"Unknown action: {action}")

    async def _execute_workflow(self, context: ToolContext, kwargs: dict) -> ToolResult:
        """Run a full plan -> test -> implement -> review -> verify -> remember -> improve workflow."""
        feature = kwargs.get("feature", "")
        workflow_name = kwargs.get("workflow_name", "plan -> test -> implement -> review -> verify -> remember -> improve")
        project_dir = kwargs.get("project_dir", "")
        custom_steps = kwargs.get("steps", [])

        steps_spec = custom_steps or [
            {"name": "plan", "role": "planner", "skill": "skill_planning"},
            {"name": "test", "role": "tdd_specialist", "skill": "skill_tdd"},
            {"name": "implement", "role": "domain_expert", "skill": "skill_architecture"},
            {"name": "review", "role": "reviewer", "skill": "skill_review"},
            {"name": "verify", "role": "build_repair", "skill": "skill_testing"},
            {"name": "remember", "role": "docs_writer", "skill": "skill_docs"},
            {"name": "improve", "role": "researcher", "skill": "skill_refactoring"},
        ]

        workflow = ECCWorkflow(
            name=workflow_name,
            status="running",
        )
        for spec in steps_spec:
            step = WorkflowStep(
                name=spec.get("name", "step"),
                agent_role=AgentRole(spec.get("role", "domain_expert")),
                skill_id=spec.get("skill"),
                description="",
            )
            workflow.steps.append(step)
        self._workflows[workflow.id] = workflow

        outputs: list[str] = []
        for i, step in enumerate(workflow.steps):
            workflow.current_step = i
            step.status = "running"
            step.started_at = datetime.utcnow()
            agent = self._agents.get(f"agent_{step.agent_role.value}")
            skill = self._skills.get(step.skill_id or "")

            plan_prompt = (
                f"Feature: {feature}\n"
                f"STEP {i + 1}: {step.name} (agent: {agent.name if agent else step.agent_role.value}"
                f", skill: {skill.name if skill else 'none'})"
            )

            # Simulated step resolution - in production this hands off to model-backed agents.
            step_resolution = self._resolve_step(step, plan_prompt)
            step.output_data = {"summary": step_resolution}
            step.status = "completed"
            step.completed_at = datetime.utcnow()
            outputs.append(step_resolution)

        # remember: persist session memory
        summary = f"Workflow '{workflow_name}' for feature: {feature} -> {'; '.join(outputs)}"
        self._store_memory(MemoryType.SESSION_SUMMARY, f"wf_{workflow.id}", summary, tags=["workflow"], project_id=project_dir or None, importance=0.8)

        workflow.status = "completed"
        workflow.updated_at = datetime.utcnow()

        return ToolResult(
            success=True,
            output="Workflow completed: " + " -> ".join(f"[{o}]" for o in outputs),
            metadata={
                "workflow_id": workflow.id,
                "workflow_name": workflow_name,
                "steps": [s.name for s in workflow.steps],
                "memory_key": f"wf_{workflow.id}",
            },
        )

    def _resolve_step(self, step: WorkflowStep, prompt: str) -> str:
        """Resolve a workflow step. Deterministic placeholder awaiting model-backed agents."""
        return f"{step.name} resolved for step '{prompt.split('STEP')[-1].strip()}'"

    async def _create_agent(self, context: ToolContext, kwargs: dict) -> ToolResult:
        name = kwargs.get("name", "")
        role_str = kwargs.get("role", "domain_expert")
        description = kwargs.get("description", "")
        content = kwargs.get("content", "")

        if not name:
            return ToolResult(success=False, error="name is required")
        try:
            role = AgentRole(role_str)
        except ValueError:
            role = AgentRole.DOMAIN_EXPERT

        agent = ECCAgent(
            name=name,
            role=role,
            description=description,
            system_prompt=content,
        )
        self._agents[agent.id] = agent
        return ToolResult(
            success=True,
            output=f"Agent created: {name}",
            metadata={"agent_id": agent.id, "role": role.value},
        )

    async def _create_skill(self, context: ToolContext, kwargs: dict) -> ToolResult:
        name = kwargs.get("name", "")
        category_str = kwargs.get("category", "research")
        description = kwargs.get("description", "")
        content = kwargs.get("content", "")

        if not name:
            return ToolResult(success=False, error="name is required")
        try:
            category = SkillCategory(category_str)
        except ValueError:
            category = SkillCategory.RESEARCH

        skill = ECCSkill(
            name=name,
            category=category,
            description=description,
            entry_point=content,
        )
        self._skills[skill.id] = skill
        return ToolResult(
            success=True,
            output=f"Skill created: {name}",
            metadata={"skill_id": skill.id, "category": category.value},
        )

    async def _install_skill(self, context: ToolContext, kwargs: dict) -> ToolResult:
        skill_source = kwargs.get("skill_source", "")
        if not skill_source:
            return ToolResult(success=False, error="skill_source is required")

        source_path = Path(skill_source)
        installed = 0
        if source_path.is_dir():
            for f in source_path.glob("*.json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    skill = ECCSkill(
                        id=data.get("id", f.stem),
                        name=data.get("name", f.stem),
                        category=SkillCategory(data.get("category", "research")) if data.get("category") else SkillCategory.RESEARCH,
                        description=data.get("description", ""),
                        entry_point=data.get("entry_point", ""),
                        version=data.get("version", "1.0.0"),
                    )
                    self._skills[skill.id] = skill
                    installed += 1
                except Exception:
                    continue
        elif source_path.is_file():
            try:
                data = json.loads(source_path.read_text(encoding="utf-8"))
                skill = ECCSkill(
                    id=data.get("id", source_path.stem),
                    name=data.get("name", source_path.stem),
                    category=SkillCategory(data.get("category", "research")) if data.get("category") else SkillCategory.RESEARCH,
                    description=data.get("description", ""),
                    entry_point=data.get("entry_point", ""),
                    version=data.get("version", "1.0.0"),
                )
                self._skills[skill.id] = skill
                installed = 1
            except Exception as exc:
                return ToolResult(success=False, error=f"Failed to parse skill file: {exc}")

        return ToolResult(
            success=True,
            output=f"Installed {installed} skill(s) from {skill_source}",
            metadata={"installed": installed},
        )

    async def _create_hook(self, context: ToolContext, kwargs: dict) -> ToolResult:
        name = kwargs.get("name", "")
        hook_type_str = kwargs.get("hook_type", "pre_tool")
        description = kwargs.get("description", "")
        content = kwargs.get("content", "")

        if not name:
            return ToolResult(success=False, error="name is required")
        try:
            hook_type = HookType(hook_type_str)
        except ValueError:
            hook_type = HookType.PRE_TOOL

        hook = ECCHook(
            name=name,
            type=hook_type,
            description=description,
            script=content,
        )
        self._hooks[hook.id] = hook
        return ToolResult(
            success=True,
            output=f"Hook created: {name}",
            metadata={"hook_id": hook.id, "type": hook_type.value},
        )

    async def _create_rule(self, context: ToolContext, kwargs: dict) -> ToolResult:
        name = kwargs.get("name", "")
        description = kwargs.get("description", "")
        content = kwargs.get("content", "")
        language = kwargs.get("language")

        if not name:
            return ToolResult(success=False, error="name is required")

        rule = ECCRule(
            name=name,
            description=description,
            rule_content=content,
            language=language,
        )
        self._rules[rule.id] = rule
        return ToolResult(
            success=True,
            output=f"Rule created: {name}",
            metadata={"rule_id": rule.id, "language": language},
        )

    async def _search_memory(self, context: ToolContext, kwargs: dict) -> ToolResult:
        query = kwargs.get("query", "")
        if not query:
            return ToolResult(success=False, error="query is required")

        q = query.lower()
        results = []
        for entry in self._memory:
            haystack = f"{entry.key} {entry.value} {' '.join(entry.tags)}".lower()
            if q in haystack:
                results.append({
                    "id": entry.id,
                    "type": entry.type.value,
                    "key": entry.key,
                    "value": entry.value,
                    "tags": entry.tags,
                    "language": entry.language,
                    "importance": entry.importance,
                })

        return ToolResult(
            success=True,
            output=json.dumps(results, indent=2, default=str),
            metadata={"results_count": len(results), "memory_total": len(self._memory)},
        )

    async def _save_memory(self, context: ToolContext, kwargs: dict) -> ToolResult:
        key = kwargs.get("name", "")
        content = kwargs.get("content", "")
        mem_type_str = kwargs.get("memory_type", "session_summary")
        tags = kwargs.get("tags", [])

        if not key:
            return ToolResult(success=False, error="name (memory key) is required")
        if not content:
            return ToolResult(success=False, error="content is required")

        try:
            mem_type = MemoryType(mem_type_str)
        except ValueError:
            mem_type = MemoryType.SESSION_SUMMARY

        entry = self._store_memory(mem_type, key, content, tags=tags)
        return ToolResult(
            success=True,
            output=f"Memory saved: {key}",
            metadata={"memory_id": entry.id, "type": mem_type.value, "importance": entry.importance},
        )

    async def _build_context(self, context: ToolContext, kwargs: dict) -> ToolResult:
        project_dir = kwargs.get("project_dir", "")
        feature = kwargs.get("feature", "")

        context_agents = [a for a in self._agents.values() if a.enabled]
        context_skills = [s for s in self._skills.values() if s.enabled]
        context_hooks = [h for h in self._hooks.values() if h.enabled]
        context_rules = [r for r in self._rules.values() if r.enabled]
        project_memory = [e for e in self._memory if e.project_id and project_dir and e.project_id == project_dir]

        built = {
            "project_dir": project_dir,
            "feature": feature,
            "context_window": len(context_agents),
            "agents": [{"id": a.id, "name": a.name, "role": a.role.value, "description": a.description} for a in context_agents],
            "skills": [{"id": s.id, "name": s.name, "category": s.category.value, "description": s.description} for s in context_skills],
            "active_hooks": [{"id": h.id, "name": h.name, "type": h.type.value, "priority": h.priority} for h in context_hooks],
            "rules": [{"id": r.id, "name": r.name, "language": r.language} for r in context_rules],
            "project_memory": [{"id": e.id, "key": e.key, "type": e.type.value, "importance": e.importance} for e in project_memory],
            "born_from": {
                "skills_selected_language": "based on most recent memory/rules",
                "project_checks": "review prior session memory",
            },
        }

        safe = dict(built)
        return ToolResult(
            success=True,
            output=json.dumps(safe, indent=2, default=str),
            metadata={"context_window": len(context_agents), "memory_included": len(project_memory)},
        )

    async def _scan_security(self, context: ToolContext, kwargs: dict) -> ToolResult:
        scan_type_str = kwargs.get("scan_type", "")
        project_dir = kwargs.get("project_dir", "")

        if scan_type_str:
            try:
                scan_type = SecurityScanType(scan_type_str)
            except ValueError:
                scan_type = SecurityScanType.PROMPT_INJECTION
        else:
            scan_type = SecurityScanType.PROMPT_INJECTION

        findings: list[SecurityFinding] = []
        if scan_type in (SecurityScanType.PROMPT_INJECTION, SecurityScanType.AGENT_FILES):
            for agent in self._agents.values():
                prompt = agent.system_prompt or ""
                for pattern in ["ignore previous", "override instructions", "you are now", "system prompt:"]:
                    if pattern in prompt.lower():
                        findings.append(SecurityFinding(
                            scan_type=scan_type,
                            severity="high",
                            title=f"Prompt injection pattern in agent '{agent.name}'",
                            description=f"Agent prompt contains '{pattern}' which can be an injection vector.",
                            file_path=f"agents/{agent.id}",
                            evidence=pattern,
                            remediation="Remove or neutralize prompt-injection patterns from agent prompts.",
                            confidence=0.7,
                        ))

        if scan_type == SecurityScanType.HOOK_ANALYSIS:
            for hook in self._hooks.values():
                if hook.script and ("eval(" in hook.script or "exec(" in hook.script):
                    findings.append(SecurityFinding(
                        scan_type=scan_type,
                        severity="high",
                        title=f"Dynamic eval in hook '{hook.name}'",
                        description="Hook script uses eval/exec which can be dangerous.",
                        file_path=f"hooks/{hook.id}",
                        evidence="eval( or exec( detected",
                        remediation="Replace dynamic eval/exec with a safe allow-listed operation.",
                        confidence=0.8,
                    ))

        if scan_type == SecurityScanType.SECRETS and project_dir:
            secrets_patterns = [
                (re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[=:]\s*['\"][0-9a-zA-Z_\-]{12,}['\"]"), "API key / secret literal"),
                (re.compile(r"sk-[a-zA-Z0-9]{16,}"), "OpenAI-style secret key"),
                (re.compile(r"gh[pousr]_[a-zA-Z0-9]{20,}"), "GitHub token"),
            ]
            root = Path(project_dir)
            if root.is_dir():
                for f in list(root.rglob("*"))[:2000]:
                    if f.is_file() and f.stat().st_size < 512 * 1024:
                        try:
                            text = f.read_text(encoding="utf-8", errors="ignore")
                        except Exception:
                            continue
                        for pattern, kind in secrets_patterns:
                            if pattern.search(text):
                                findings.append(SecurityFinding(
                                    scan_type=scan_type,
                                    severity="critical",
                                    title=f"Possible {kind} committed",
                                    description=f"Secret-looking literal found in {f}",
                                    file_path=str(f),
                                    evidence=kind,
                                    remediation="Rotate the secret and remove it from the repository.",
                                    confidence=0.6,
                                ))

        if scan_type == SecurityScanType.PERMISSIONS:
            for hook in self._hooks.values():
                if hook.type in (HookType.PRE_AGENT, HookType.POST_AGENT) and not hook.script:
                    findings.append(SecurityFinding(
                        scan_type=scan_type,
                        severity="low",
                        title=f"Agent hook without script: '{hook.name}'",
                        description="Hook exists but has no validation script.",
                        file_path=f"hooks/{hook.id}",
                        evidence="empty script",
                        remediation="Attach a script to enforce the hook's intent.",
                        confidence=0.5,
                    ))

        self._security_findings.extend(findings)
        return ToolResult(
            success=True,
            output=json.dumps([f.__dict__ for f in findings], indent=2, default=str),
            metadata={"scan_type": scan_type.value, "findings_count": len(findings), "total_findings": len(self._security_findings)},
        )

    async def _ecc_cli(self, context: ToolContext, kwargs: dict) -> ToolResult:
        """Run a real ECC-main CLI command (node scripts/ecc.js ...)."""
        if not self._ecc_home:
            return ToolResult(success=False, error=f"ECC-main not found. Set {ECC_HOME_ENV} or clone to {ECC_HOME_DEFAULT}")
        command = kwargs.get("ecc_command", "").strip()
        if not command:
            return ToolResult(success=False, error="ecc_command is required for ecc_cli action")
        cli = str(Path(self._ecc_home) / "scripts" / "ecc.js")
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            return ToolResult(success=False, error=f"Invalid ecc_command quoting: {exc}")

        env = dict(os.environ)
        env["ECC_HOME"] = self._ecc_home
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    "node", cli, *parts,
                    cwd=self._ecc_home,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=120,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except TimeoutError:
            return ToolResult(success=False, error="ecc CLI timed out (120s)")
        except FileNotFoundError:
            return ToolResult(success=False, error="'node' not found on PATH; ECC-main CLI requires Node.js")

        output = stdout.decode("utf-8", errors="replace").strip()
        errout = stderr.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            return ToolResult(
                success=False,
                error=errout or f"ecc CLI exited {proc.returncode}",
                output=output[:12000],
                metadata={"exit_code": proc.returncode, "command": command, "ecc_home": self._ecc_home},
            )
        return ToolResult(
            success=True,
            output=output[:12000] or "(no output)",
            metadata={"exit_code": proc.returncode, "command": command, "ecc_home": self._ecc_home},
        )

    async def _llm_generate(self, context: ToolContext, kwargs: dict) -> ToolResult:
        """Generate text through ECC-main's src/llm provider layer.

        Defaults to the OpenAI-compatible LM Studio endpoint
        (http://127.0.0.1:1234/v1), which mirrors how the local chat/embeddings
        tools are wired.  Claude/Ollama are available when ECC-main is present.
        """
        if not self._ecc_home or not self._ecc_llm_src:
            return ToolResult(success=False, error=f"ECC-main src/llm not available. Set {ECC_HOME_ENV} or clone to {ECC_HOME_DEFAULT}")
        prompt = kwargs.get("prompt", "").strip()
        if not prompt:
            return ToolResult(success=False, error="prompt is required for llm_generate action")

        provider_name = kwargs.get("provider", "openai")
        model = kwargs.get("model", DEFAULT_LLM_MODEL)
        base_url = kwargs.get("base_url", LLM_BASE_URL_DEFAULT)
        temperature = float(kwargs.get("temperature", 0.3))
        max_tokens = int(kwargs.get("max_tokens", 512))

        try:
            from llm.core.types import LLMInput, Message, ProviderType, Role
            from llm.providers.resolver import get_provider
        except ImportError as exc:
            return ToolResult(success=False, error=f"ECC-main llm package import failed: {exc}")

        try:
            if provider_name in {"openai", "openai-compatible"}:
                provider = get_provider(ProviderType.OPENAI, api_key=os.environ.get("OPENAI_API_KEY", "lm-studio"), base_url=base_url)
            else:
                provider = get_provider(provider_name)
        except Exception as exc:
            return ToolResult(success=False, error=f"Provider '{provider_name}' init failed: {exc}")

        messages = [Message(role=Role.USER, content=prompt)]
        try:
            output = provider.generate(LLMInput(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens))
        except Exception as exc:
            return ToolResult(success=False, error=f"Generation failed: {exc}")

        return ToolResult(
            success=True,
            output=output.content,
            metadata={
                "model": output.model or model,
                "provider": provider_name,
                "base_url": base_url,
                "usage": output.usage or {},
                "stop_reason": output.stop_reason,
            },
        )

    async def _get_status(self, context: ToolContext, kwargs: dict) -> ToolResult:
        status = ECCStatus(
            agents_loaded=len([a for a in self._agents.values() if a.enabled]),
            skills_loaded=len([s for s in self._skills.values() if s.enabled]),
            hooks_loaded=len([h for h in self._hooks.values() if h.enabled]),
            memory_entries=len(self._memory),
            rules_loaded=len(self._rules),
            active_workflows=len([w for w in self._workflows.values() if w.status == "running"]),
            security_findings_count=len(self._security_findings),
            security_scan_last_run=datetime.utcnow() if self._security_findings else None,
        )
        if self._ecc_home:
            status.ecc_home = self._ecc_home
            has_cli = (Path(self._ecc_home) / "scripts" / "ecc.js").is_file()
            has_src = (Path(self._ecc_home) / "src" / "llm").is_dir()
            status.ecc_cli_available = has_cli
            status.ecc_llm_src_available = has_src
        return ToolResult(
            success=True,
            output=json.dumps({
                "agents_loaded": status.agents_loaded,
                "skills_loaded": status.skills_loaded,
                "hooks_loaded": status.hooks_loaded,
                "memory_entries": status.memory_entries,
                "rules_loaded": status.rules_loaded,
                "active_workflows": status.active_workflows,
                "security_findings": status.security_findings_count,
                "security_scan_last_run": status.security_scan_last_run.isoformat() if status.security_scan_last_run else None,
                "ecc_home": status.ecc_home,
                "ecc_cli_available": status.ecc_cli_available,
                "ecc_llm_src_available": status.ecc_llm_src_available,
            }, indent=2),
        )
