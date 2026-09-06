"""Astra X Evaluation Framework.

Build and run benchmark suites that measure system-level intelligence:
general chat, tool selection, multi-step planning, reflection decisions,
memory retrieval, RAG, streaming, and error recovery.

Usage:
    python -m evals.runner                     # run all suites
    python -m evals.runner --suite chat        # single suite
    python -m evals.runner --json report.json  # export results
"""

from evals.base import BaseEval
from evals.metrics import MetricsTracker
from evals.models import Artifact, EvalCase, EvalResult
from evals.report import Report, ReportEntry

__all__ = [
    "Artifact",
    "BaseEval",
    "EvalCase",
    "EvalResult",
    "MetricsTracker",
    "Report",
    "ReportEntry",
]
