"""Tests for codey.review.deps — dependent-file discovery and source loading."""

from __future__ import annotations

from codey.agents.context import ReviewContext
from codey.cache.ast_cache import CacheDB
from codey.index.callgraph import build_call_graph
from codey.index.indexer import index_repository
from codey.review.deps import (
    enrich_context,
    fetch_dependent_files,
    load_dependent_sources,
)
from tests.conftest import commit, init_repo


def test_fetch_dependent_files_empty(tmp_path):
    db = CacheDB()
    assert fetch_dependent_files(tmp_path, "h", db, []) == []
    db.close()


def test_fetch_dependent_files_cross_file(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    init_repo(r)
    (r / "a.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (r / "b.py").write_text("import a\n\ndef use():\n    return a.helper()\n", encoding="utf-8")
    commit(r, "init")

    db = CacheDB()
    ir = index_repository(r, db)
    build_call_graph(r, ir.git_hash, db)
    deps = fetch_dependent_files(r, ir.git_hash, db, ["a.py"])
    assert "b.py" in deps
    db.close()


def test_load_dependent_sources_truncates(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    (r / "big.py").write_text("x" * 9000, encoding="utf-8")
    (r / "missing.py")  # referenced but doesn't exist
    sources = load_dependent_sources(r, ["big.py", "missing.py"], max_chars=8000)
    assert "big.py" in sources
    assert len(sources["big.py"]) <= 8000 + len("\n... [truncated]")
    assert "missing.py" not in sources


def test_enrich_context_populates_fields(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    init_repo(r)
    (r / "a.py").write_text("x = 1\n", encoding="utf-8")
    commit(r, "init")

    db = CacheDB()
    ir = index_repository(r, db)
    ctx = ReviewContext(
        repo_path=r,
        git_hash=ir.git_hash,
        commit_message="m",
        changed_files=["a.py"],
    )
    enrich_context(ctx, db)
    assert "a.py" in ctx.file_sources
    db.close()
