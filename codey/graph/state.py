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
    result: dict[str, AgentReport] = {}
    if left:
        result.update(left)
    if right:
        result.update(right)
    return result


class ReviewState(TypedDict, total=False):
    context: ReviewContext
    agent_reports: Annotated[dict[str, AgentReport], merge_reports]
    index_summary: str
    final_review: ReviewSummary
    progress: Annotated[list[str], add]
    primary_llm: object
    errors: Annotated[list[str], add]


def initial_state(ctx: ReviewContext, *, primary_llm: object | None = None) -> ReviewState:
    return ReviewState(
        context=ctx,
        agent_reports={},
        index_summary="",
        progress=[],
        errors=[],
        primary_llm=primary_llm,
    )
