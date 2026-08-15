"""LangGraph review graph assembly — DAG with parallel fan-out.

Builds a StateGraph with:
  START -> index -> fan_out -> [security, code_quality, test] (parallel) -> codey -> END

The index agent runs first (populating index_summary). Then security,
code_quality, and test agents run in parallel (edges from index). Finally the
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
from codey.agents.schemas import AgentReport, ReviewSummary, Severity
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
    try:
        report, index_summary = run_index_agent(ctx, db, primary_llm)
    except Exception as e:
        return {"progress": ["index: error"], "errors": [f"[index] {e}"]}
    return {
        "agent_reports": {"index": report},
        "index_summary": index_summary,
        "progress": [f"index: {report.status}"],
        "errors": ([f"[index] {report.error}"] if report.error else []),
    }


def _security_node(state: ReviewState) -> dict[str, Any]:
    ctx: ReviewContext = state["context"]
    primary_llm = state.get("primary_llm")
    try:
        report = run_security_agent(ctx, db=ctx.db, llm=primary_llm)
    except Exception as e:
        return {"progress": ["security: error"], "errors": [f"[security] {e}"]}
    return {
        "agent_reports": {"security": report},
        "progress": [f"security: {report.status}"],
        "errors": ([f"[security] {report.error}"] if report.error else []),
    }


def _code_quality_node(state: ReviewState) -> dict[str, Any]:
    ctx: ReviewContext = state["context"]
    # Inject the index summary from the prior index agent.
    ctx.index_summary = state.get("index_summary", "")
    primary_llm = state.get("primary_llm")
    try:
        report = run_code_quality_agent(ctx, db=ctx.db, llm=primary_llm)
    except Exception as e:
        return {"progress": ["code_quality: error"], "errors": [f"[code_quality] {e}"]}
    return {
        "agent_reports": {"code_quality": report},
        "progress": [f"code_quality: {report.status}"],
        "errors": ([f"[code_quality] {report.error}"] if report.error else []),
    }


def _test_node(state: ReviewState) -> dict[str, Any]:
    ctx: ReviewContext = state["context"]
    primary_llm = state.get("primary_llm")
    try:
        report = run_test_agent(ctx, db=ctx.db, llm=primary_llm)
    except Exception as e:
        return {"progress": ["test: error"], "errors": [f"[test] {e}"]}
    return {
        "agent_reports": {"test": report},
        "progress": [f"test: {report.status}"],
        "errors": ([f"[test] {report.error}"] if report.error else []),
    }


def _codey_node(state: ReviewState) -> dict[str, Any]:
    ctx: ReviewContext = state["context"]
    primary_llm = state.get("primary_llm")
    reports: dict[str, AgentReport] = state.get("agent_reports", {})
    try:
        review = run_codey_agent(
            ctx,
            reports,
            primary_llm,
            prior_errors=state.get("errors", []),
        )
    except Exception as e:
        # Final fallback: never leave the caller with no review object. The
        # verdict must never optimistically approve a failed synthesis.
        review = ReviewSummary(
            overall_severity=Severity.INFO,
            summary=f"Review synthesis failed: {e}",
            commit_hash=ctx.git_hash,
            commit_message=ctx.commit_message,
            agent_reports=reports,
            total_findings=sum(len(r.findings) for r in reports.values()),
            recommendation="request_changes",
            errors=[f"[codey] {e}"] + list(state.get("errors", [])),
        )
    return {
        "final_review": review,
        "progress": ["codey: synthesis complete"],
    }


def _register_agents() -> None:
    """Register the review agents in the declarative registry.

    Called once (idempotently) by ``build_graph``. Adding an agent to the
    review pipeline means adding an ``AgentSpec`` entry here — no graph
    wiring changes needed.
    """
    from codey.graph.registry import AgentSpec, register

    register(AgentSpec(
        name="index",
        label="Indexing repository",
        run=_index_node,
    ))
    register(AgentSpec(
        name="security",
        label="Running security analysis",
        run=_security_node,
        depends_on=("index",),
    ))
    register(AgentSpec(
        name="code_quality",
        label="Checking code quality",
        run=_code_quality_node,
        depends_on=("index",),
    ))
    register(AgentSpec(
        name="test",
        label="Running test suite",
        run=_test_node,
        depends_on=("index",),
    ))
    register(AgentSpec(
        name="codey",
        label="Synthesising final review",
        run=_codey_node,
        depends_on=("security", "code_quality", "test"),
    ))


def build_graph() -> Any:
    """Build and compile the LangGraph review graph from the agent registry.

    Topology is derived from the registry: the first agent with no
    dependencies is the entry point (index), agents fan out after their
    dependencies, and agents with no dependents connect to END (codey).
    """
    from codey.graph.registry import first_agent, get_specs, terminal_agents

    _register_agents()
    specs = get_specs()
    if not specs:
        raise RuntimeError("No agents registered — the review graph would be empty")

    graph = StateGraph(ReviewState)

    for spec in specs:
        graph.add_node(spec.name, spec.run)

    # Edges: dependency fan-out.
    for spec in specs:
        for dep in spec.depends_on:
            graph.add_edge(dep, spec.name)

    # Entry point: the first agent with no dependencies.
    entry = first_agent(specs)
    if entry is not None:
        graph.add_edge(START, entry.name)

    # Terminal nodes: agents nothing depends on.
    for name in terminal_agents(specs):
        graph.add_edge(name, END)

    return graph.compile()


def run_review(
    ctx: ReviewContext,
    *,
    primary_llm: object | None = None,
    progress_callback=None,
) -> ReviewSummary:
    """Build the graph and execute the review pipeline.

    Returns the final ``ReviewSummary``.
    """
    graph = build_graph()
    state = initial_state(ctx, primary_llm=primary_llm)

    final_state: ReviewState = state
    if progress_callback is not None:
        for mode, payload in graph.stream(state, stream_mode=["updates", "values"]):
            if mode == "updates":
                progress_callback(payload)
            elif mode == "values":
                final_state = payload
    else:
        final_state = graph.invoke(state)

    result = final_state if isinstance(final_state, dict) else dict(final_state)
    review: ReviewSummary | None = result.get("final_review")
    if review is None:
        reports = result.get("agent_reports", {})
        review = run_codey_agent(
            ctx,
            reports,
            primary_llm,
            prior_errors=result.get("errors") or [],
        )
    return review
