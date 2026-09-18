"""Structured output tool for Phase 4 - write findings with verdicts and validate against schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.tools.audit.models import Finding, Verdict
from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class StructuredOutputTool(Tool):
    """Phase 4: Structured output - write confirmed, needs_validation, rejected records to findings.json and validate against report-schema.json."""

    @property
    def name(self) -> str:
        return "structured_output"

    @property
    def description(self) -> str:
        return "Perform Phase 4 structured output: organize findings by verdict, validate against report-schema.json, produce final findings.json."

    @property
    def capabilities(self) -> list[str]:
        return ["structured_output", "schema_validation", "findings_organization"]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="findings_path",
                    type_="string",
                    description="Path to findings.json from candidate validation",
                    required=True,
                ),
                ToolParameter(
                    name="schema_path",
                    type_="string",
                    description="Path to report-schema.json for validation",
                    required=False,
                    default="",
                ),
                ToolParameter(
                    name="output_dir",
                    type_="string",
                    description="Directory to write structured output",
                    required=True,
                ),
                ToolParameter(
                    name="include_rejected",
                    type_="boolean",
                    description="Whether to include rejected findings in output",
                    required=False,
                    default=False,
                ),
            ],
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        findings_path = kwargs.get("findings_path", "")
        schema_path = kwargs.get("schema_path", "")
        output_dir = kwargs.get("output_dir", "")
        include_rejected = kwargs.get("include_rejected", False)

        if not findings_path or not output_dir:
            return ToolResult(success=False, error="findings_path and output_dir are required")

        # Load findings
        findings_data = json.loads(Path(findings_path).read_text(encoding="utf-8"))
        findings = self._load_findings(findings_data)

        # Load schema if provided
        schema = None
        if schema_path and Path(schema_path).exists():
            schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Organize findings by verdict
        organized = self._organize_by_verdict(findings, include_rejected)

        # Validate against schema if provided
        validation_results = []
        if schema:
            validation_results = self._validate_against_schema(organized, schema)

        # Write organized findings.json
        output_data = {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "confirmed": organized["confirmed"],
            "needs_validation": organized["needs_validation"],
            "rejected": organized["rejected"] if include_rejected else [],
            "summary": {
                "total_confirmed": len(organized["confirmed"]),
                "total_needs_validation": len(organized["needs_validation"]),
                "total_rejected": len(organized["rejected"]),
            },
        }

        (output_path / "findings.json").write_text(json.dumps(output_data, indent=2), encoding="utf-8")

        # Write validation results
        if validation_results:
            validation_output = {
                "version": "1.0",
                "validated_at": datetime.utcnow().isoformat() + "Z",
                "valid": all(r["valid"] for r in validation_results),
                "results": validation_results,
            }
            (output_path / "validation-results.json").write_text(json.dumps(validation_output, indent=2), encoding="utf-8")

        # Write individual verdict files for easy consumption
        self._write_verdict_files(organized, output_path, include_rejected)

        return ToolResult(
            success=True,
            output=f"Structured output complete. {len(organized['confirmed'])} confirmed, {len(organized['needs_validation'])} needs validation, {len(organized['rejected'])} rejected.",
            metadata={
                "confirmed_count": len(organized["confirmed"]),
                "needs_validation_count": len(organized["needs_validation"]),
                "rejected_count": len(organized["rejected"]),
                "validation_passed": all(r["valid"] for r in validation_results) if validation_results else True,
                "findings_json": str(output_path / "findings.json"),
            },
        )

    def _load_findings(self, data: dict) -> list[Finding]:
        """Load findings from JSON."""
        findings = []
        for f_data in data.get("findings", []):
            findings.append(
                Finding(
                    id=f_data.get("id", ""),
                    title=f_data.get("title", ""),
                    description=f_data.get("description", ""),
                    verdict=Verdict(f_data.get("verdict", "needs_validation")),
                    severity=f_data.get("severity", "medium"),
                    attack_class=f_data.get("attack_class", ""),
                    coverage_unit_id=f_data.get("coverage_unit_id", ""),
                    hunter_id=f_data.get("hunter_id", ""),
                    evidence=f_data.get("evidence", []),
                    steps_to_reproduce=f_data.get("steps_to_reproduce", []),
                    impact=f_data.get("impact", ""),
                    remediation=f_data.get("remediation", ""),
                    references=f_data.get("references", []),
                    confidence=f_data.get("confidence", 0.5),
                    verifier_id=f_data.get("verifier_id"),
                    verification_status=f_data.get("verification_status", "pending"),
                    verification_notes=f_data.get("verification_notes", ""),
                )
            )
        return findings

    def _organize_by_verdict(self, findings: list[Finding], include_rejected: bool) -> dict[str, list[dict]]:
        """Organize findings by verdict into JSON-serializable dicts."""
        confirmed = []
        needs_validation = []
        rejected = []

        for f in findings:
            finding_dict = {
                "id": f.id,
                "title": f.title,
                "description": f.description,
                "verdict": f.verdict.value,
                "severity": f.severity,
                "attack_class": f.attack_class,
                "coverage_unit_id": f.coverage_unit_id,
                "hunter_id": f.hunter_id,
                "evidence": f.evidence,
                "steps_to_reproduce": f.steps_to_reproduce,
                "impact": f.impact,
                "remediation": f.remediation,
                "references": f.references,
                "confidence": f.confidence,
                "verifier_id": f.verifier_id,
                "verification_status": f.verification_status,
                "verification_notes": f.verification_notes,
            }

            if f.verdict == Verdict.CONFIRMED:
                confirmed.append(finding_dict)
            elif f.verdict == Verdict.NEEDS_VALIDATION:
                needs_validation.append(finding_dict)
            elif f.verdict == Verdict.REJECTED:
                rejected.append(finding_dict)

        return {
            "confirmed": confirmed,
            "needs_validation": needs_validation,
            "rejected": rejected,
        }

    def _validate_against_schema(self, organized: dict, schema: dict) -> list[dict]:
        """Validate findings against JSON schema."""
        results = []

        # Simple validation - check required fields
        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})

        for verdict_type in ["confirmed", "needs_validation", "rejected"]:
            for finding in organized[verdict_type]:
                errors = []
                for field in required_fields:
                    if field not in finding:
                        errors.append(f"Missing required field: {field}")

                # Type checking
                for field, expected_type in properties.items():
                    if field in finding:
                        actual = finding[field]
                        if expected_type == "string" and not isinstance(actual, str):
                            errors.append(f"Field {field} should be string")
                        elif expected_type == "array" and not isinstance(actual, list):
                            errors.append(f"Field {field} should be array")
                        elif expected_type == "number" and not isinstance(actual, (int, float)):
                            errors.append(f"Field {field} should be number")

                results.append({
                    "finding_id": finding.get("id", "unknown"),
                    "verdict": verdict_type,
                    "valid": len(errors) == 0,
                    "errors": errors,
                })

        return results

    def _write_verdict_files(self, organized: dict, output_path: Path, include_rejected: bool) -> None:
        """Write separate files for each verdict category."""
        # Confirmed findings
        confirmed_data = {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "verdict": "confirmed",
            "findings": organized["confirmed"],
        }
        (output_path / "findings-confirmed.json").write_text(json.dumps(confirmed_data, indent=2), encoding="utf-8")

        # Needs validation
        nv_data = {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "verdict": "needs_validation",
            "findings": organized["needs_validation"],
        }
        (output_path / "findings-needs-validation.json").write_text(json.dumps(nv_data, indent=2), encoding="utf-8")

        # Rejected (if requested)
        if include_rejected:
            rejected_data = {
                "version": "1.0",
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "verdict": "rejected",
                "findings": organized["rejected"],
            }
            (output_path / "findings-rejected.json").write_text(json.dumps(rejected_data, indent=2), encoding="utf-8")


from datetime import datetime