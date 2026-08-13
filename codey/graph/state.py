"""TypedDict state + reducers for the LangGraph review graph."""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import add_messages

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

    # Context (immutable through the graph).
    context: ReviewContext

    # Agent reports accumulate via reducer.
    agent_reports: Annotated[dict[str, AgentReport], merge_reports]

    # Index summary populated by IndexAgent, consumed by CodeQualityAgent.
    index_summary: str

    # The final synthesized review (filled by the orchestrator at the end).
    final_review: ReviewSummary

    # LangGraph messages for supervisor subagent communication.
    messages: Annotated[list, add_messages]

    # Progress events for the progress emitter (parallel-safe via add reducer).
    progress: Annotated[list[str], add]

    # LLM objects (primary + summarizer), set at graph build time.
    primary_llm: object
    summarizer_llm: object

    # Error tracking (parallel-safe).
    errors: Annotated[list[str], add]


def initial_state(ctx: ReviewContext, *, primary_llm: object | None = None, summarizer_llm: object | None = None) -> ReviewState:
    """Build the initial state for the review graph."""
    return ReviewState(
        context=ctx,
        agent_reports={},
        index_summary="",
        progress=[],
        errors=[],
        messages=[],
        primary_llm=primary_llm,
        summarizer_llm=summarizer_llm,
    )