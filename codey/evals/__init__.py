"""Evaluation layer — golden-scenario review quality scoring (``codey eval``)."""

from codey.evals.runner import StubLLM, StubResponse, run_evals
from codey.evals.scenarios import SCENARIOS, EvalFile, EvalScenario, ExpectedIssue

__all__ = [
    "run_evals",
    "StubLLM",
    "StubResponse",
    "EvalFile",
    "EvalScenario",
    "ExpectedIssue",
    "SCENARIOS",
]
