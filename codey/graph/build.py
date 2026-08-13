"""LangGraph review graph assembly — supervisor with parallel fan-out.

Builds a StateGraph with:
  START -> index -> fan_out -> [security, code_quality, test] (parallel) -> codey -> END

The index agent runs first (populating index_summary). Then security,
code_quality, and test agents run in parallel (via Send). Finally the
codey (orchestrator) agent collects all reports and synthesises the final
ReviewSummary.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from codey.agents.code_quality_agent import run_code_quality_agent
from codey.agents.codey_agent import run_codey_agent
from codey.agents.context import ReviewContext
from codey.agents.index_agent import run_index_agent
from codey.agents.schemas import AgentReport, ReviewSummary
from codey.agents.security_agent import run_security_agent
from codey.agents.test_agent import run_test_agent
from codey.graph.state import ReviewState, initial_state

__all__ = ["build_graph", "run_review"]


def _index_node(state: ReviewState) -> dict[str, Any]:
    ctx: ReviewContext = state["context"]
    db = ctx.db
    if db is None:
        return {"progress": ["index skipped (no db)"], "errors": ["no db"]}
    primary_llm = state.get("primary_llm")
    report, index_summary = run_index_agent(ctx, db, primary_llm)
    return {
        "agent_reports": {"index": report},
        "index_summary": index_summary,
        "progress": [f"index: {report.status}"],
    }


def _security_node(state: ReviewState) -> dict[str, Any]:
    ctx: ReviewContext = state["context"]
    primary_llm = state.get("primary_llm")
    report = run_security_agent(ctx, db=ctx.db, llm=primary_llm)
    return {
        "agent_reports": {"security": report},
        "progress": [f"security: {report.status}"],
    }


def _code_quality_node(state: ReviewState) -> dict[str, Any]:
    ctx: ReviewContext = state["context"]
    # Inject the index summary from the prior index agent.
    ctx.index_summary = state.get("index_summary", "")
    primary_llm = state.get("primary_llm")
    report = run_code_quality_agent(ctx, db=ctx.db, llm=primary_llm)
    return {
        "agent_reports": {"code_quality": report},
        "progress": [f"code_quality: {report.status}"],
    }


def _test_node(state: ReviewState) -> dict[str, Any]:
    ctx: ReviewContext = state["context"]
    primary_llm = state.get("primary_llm")
    report = run_test_agent(ctx, db=ctx.db, llm=primary_llm)
    return {
        "agent_reports": {"test": report},
        "progress": [f"test: {report.status}"],
    }


def _codey_node(state: ReviewState) -> dict[str, Any]:
    ctx: ReviewContext = state["context"]
    primary_llm = state.get("primary_llm")
    reports: dict[str, AgentReport] = state.get("agent_reports", {})
    review = run_codey_agent(ctx, reports, primary_llm, repo_for_subagents=ctx.repo_path)
    return {
        "final_review": review,
        "progress": ["codey: synthesis complete"],
    }


def build_graph() -> Any:
    """Build and compile the LangGraph review graph."""
    graph = StateGraph(ReviewState)

    graph.add_node("index", _index_node)
    graph.add_node("security", _security_node)
    graph.add_node("code_quality", _code_quality_node)
    graph.add_node("test", _test_node)
    graph.add_node("codey", _codey_node)

    # Index runs first, then fan out to the three worker agents in parallel.
    graph.add_edge(START, "index")
    graph.add_edge("index", "security")
    graph.add_edge("index", "code_quality")
    graph.add_edge("index", "test")

    # All workers converge into the orchestrator.
    graph.add_edge("security", "codey")
    graph.add_edge("code_quality", "codey")
    graph.add_edge("test", "codey")
    graph.add_edge("codey", END)

    return graph.compile()


def run_review(
    ctx: ReviewContext,
    *,
    primary_llm: object | None = None,
    summarizer_llm: object | None = None,
    progress_callback=None,
) -> ReviewSummary:
    """Build the graph and execute the review pipeline.

    Returns the final ``ReviewSummary``.
    """
    graph = build_graph()
    state = initial_state(ctx, primary_llm=primary_llm, summarizer_llm=summarizer_llm)

    if progress_callback is not None:
        for chunk in graph.stream(state, stream_mode="updates"):
            progress_callback(chunk)
    else:
        state = graph.invoke(state)

    result = state if isinstance(state, dict) else dict(state)
    review: ReviewSummary | None = result.get("final_review")
    if review is None:
        # Fallback: synthesise manually if the orchestrator somehow didn't run.
        reports = result.get("agent_reports", {})
        review = run_codey_agent(ctx, reports, primary_llm, repo_for_subagents=ctx.repo_path)
    return review