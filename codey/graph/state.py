"""TypedDict state + reducers for the LangGraph review graph."""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from codey.agents.context import ReviewContext
from codey.agents.schemas import AgentReport, ReviewSummary

__all__ = ["ReviewState", "merge_reports", "initial_state"]


def merge_reports(
    left: dict[str, AgentReport] | None,
    right: dict[str, AgentReport] | None,
) -> dict[str, AgentReport]:
    """Reducer: merge two dicts of agent reports (right wins on key conflict)."""
    result: dict[str, AgentReport] = {}
    if left:
        result.update(left)
    if right:
        result.update(right)
    return result


class ReviewState(TypedDict, total=False):
    """State threaded through the review graph."""

    # Context shared by all nodes. Treated as read-only; nodes that need to
    # tweak it (e.g. inject the index summary) copy it first.
    context: ReviewContext

    # Agent reports accumulate via reducer.
    agent_reports: Annotated[dict[str, AgentReport], merge_reports]

    # Index summary populated by IndexAgent, consumed by CodeQualityAgent.
    index_summary: str

    # The final synthesized review (filled by the orchestrator at the end).
    final_review: ReviewSummary

    # Progress events for the progress emitter (parallel-safe via add reducer).
    progress: Annotated[list[str], add]

    # LLM object (primary), set at graph build time.
    primary_llm: object

    # Error tracking (parallel-safe).
    errors: Annotated[list[str], add]


def initial_state(
    ctx: ReviewContext,
    *,
    primary_llm: object | None = None,
) -> ReviewState:
    """Build the initial state for the review graph."""
    return ReviewState(
        context=ctx,
        agent_reports={},
        index_summary="",
        progress=[],
        errors=[],
        primary_llm=primary_llm,
    )
