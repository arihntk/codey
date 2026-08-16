"""Tests for codey.graph — agent registry, state, and graph assembly."""

from __future__ import annotations

from codey.agents.context import ReviewContext
from codey.agents.schemas import AgentReport, ReviewSummary
from codey.graph.build import build_graph, run_review
from codey.graph.registry import (
    AgentSpec,
    first_agent,
    get_specs,
    ordered_agent_names,
    register,
    terminal_agents,
)
from codey.graph.state import initial_state, merge_reports


def _spec(name, deps=()):
    return AgentSpec(name=name, label=name, run=lambda s: {}, depends_on=deps)


def test_first_agent_picks_first_with_no_deps():
    specs = [_spec("a", ("z",)), _spec("b"), _spec("c")]
    assert first_agent(specs).name == "b"


def test_first_agent_none_when_all_have_deps():
    assert first_agent([_spec("a", ("x",))]) is None


def test_terminal_agents():
    specs = [_spec("index"), _spec("security", ("index",)), _spec("codey", ("security",))]
    assert terminal_agents(specs) == ["codey"]


def test_register_and_get_specs(monkeypatch):
    from codey.graph import registry

    monkeypatch.setattr(registry, "_SPECS", {})
    register(_spec("x"))
    assert [s.name for s in get_specs()] == ["x"]


def test_ordered_agent_names_after_build(monkeypatch):
    from codey.graph import registry

    monkeypatch.setattr(registry, "_SPECS", {})
    build_graph()  # populates the registry
    names = ordered_agent_names()
    assert names == ["index", "security", "code_quality", "test", "codey"]


def test_merge_reports_right_wins():
    left = {"a": AgentReport(agent="security", summary="old")}
    right = {"a": AgentReport(agent="security", summary="new"), "b": AgentReport(agent="index")}
    merged = merge_reports(left, right)
    assert merged["a"].summary == "new"
    assert "b" in merged


def test_merge_reports_none_handling():
    assert merge_reports(None, None) == {}
    assert merge_reports({"a": AgentReport(agent="security")}, None)["a"].agent == "security"


def test_initial_state_fields():
    ctx = ReviewContext(repo_path=None, git_hash="h", commit_message="m")
    s = initial_state(ctx, primary_llm="LLM")
    assert s["context"] is ctx
    assert s["agent_reports"] == {}
    assert s["index_summary"] == ""
    assert s["progress"] == []
    assert s["errors"] == []
    assert s["primary_llm"] == "LLM"


def test_build_graph_compiles():
    g = build_graph()
    assert g is not None


def test_run_review_no_llm(repo):
    from codey.cache.ast_cache import CacheDB

    db = CacheDB()
    ctx = ReviewContext(
        repo_path=repo,
        git_hash="HEAD",
        commit_message="m",
        changed_files=["main.py"],
        db=db,
        cache_repo_path=repo,
    )
    review = run_review(ctx, primary_llm=None)
    assert isinstance(review, ReviewSummary)
    assert set(review.agent_reports) == {"index", "security", "code_quality", "test"}
    assert review.recommendation in ("approve", "request_changes", "block")
    db.close()


def test_run_review_progress_callback(repo):
    from codey.cache.ast_cache import CacheDB

    db = CacheDB()
    ctx = ReviewContext(
        repo_path=repo,
        git_hash="HEAD",
        commit_message="m",
        changed_files=["main.py"],
        db=db,
        cache_repo_path=repo,
    )
    seen = []

    review = run_review(ctx, primary_llm=None, progress_callback=lambda chunk: seen.append(chunk))
    assert isinstance(review, ReviewSummary)
    assert len(seen) > 0
    db.close()
