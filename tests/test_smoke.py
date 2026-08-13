"""Smoke tests for codey — exercise the full pipeline without an LLM.

Run with: ``uv run pytest tests/ -v``
"""

from __future__ import annotations

import pytest


# Use a temp cache dir so tests don't pollute the real cache.
@pytest.fixture(autouse=True)
def _temp_cache(monkeypatch, tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("codey-cache")
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_dir))
    config_dir = tmp_path_factory.mktemp("codey-config")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_dir))

@pytest.fixture
def repo(tmp_path):
    """Create a tiny git repo with one Python file."""
    import subprocess

    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "main.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def sub(a, b):\n"
        "    return add(a, -b)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), check=True)
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=str(repo), check=True)
    # Add a second commit to get a diff.
    (repo / "main.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def sub(a, b):\n"
        "    return add(a, -b)\n\n"
        "def mul(a, b):\n"
        "    return a * b\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add mul function"], cwd=str(repo), check=True)
    return repo


class TestConfigProviders:
    def test_presets_exist(self):
        from codey.config.providers import PRESETS, get_preset

        assert len(PRESETS) == 5
        assert get_preset("openai") is not None
        assert get_preset("anthropic") is not None
        assert get_preset("deepseek") is not None
        assert get_preset("google") is not None
        assert get_preset("custom") is not None
        assert get_preset("nonexistent") is None

    def test_config_round_trip(self):
        from codey.config.store import Config, load_config, save_config

        cfg = Config(provider="openai", model="gpt-4.1", summarizer_model="gpt-4.1-mini")
        save_config(cfg)
        loaded = load_config()
        assert loaded == cfg
        assert loaded.is_complete()
        assert loaded.to_dict()["provider"] == "openai"


class TestCacheDB:
    def test_crud_round_trip(self):
        from codey.cache.ast_cache import (
            CacheDB,
            CallEdge,
            FileEntry,
            ImportEdge,
            SymbolRecord,
            hash_content,
        )

        db = CacheDB()
        db.upsert_index_run("/repo", "abc123", file_count=1)
        assert db.has_indexed_hash("/repo", "abc123")
        assert db.last_indexed_hash("/repo") == "abc123"

        entry = FileEntry("foo.py", "python", hash_content("x"), "{}", "[]", 0.0, 1)
        db.upsert_file_entry("/repo", "abc123", entry)
        got = db.get_file_entry("/repo", "abc123", "foo.py")
        assert got and got.rel_path == "foo.py"

        db.bulk_upsert_symbols("/repo", "abc123", [SymbolRecord("foo.py", "bar", "foo.bar", "function", 1, 2)])
        assert db.symbols_in_file("/repo", "abc123", "foo.py")[0].name == "bar"

        db.bulk_insert_call_edges("/repo", "abc123", [CallEdge("foo.py", "foo.baz", "bar", "foo.py", "foo.bar", 3)])
        assert db.callers_of("/repo", "abc123", "bar")[0].caller_qname == "foo.baz"

        db.bulk_insert_import_edges("/repo", "abc123", [ImportEdge("foo.py", "os", "path", "p", 1)])
        assert db.importers_of_module("/repo", "abc123", "os")[0].alias == "p"

        db.clear_run("/repo", "abc123")
        assert not db.has_indexed_hash("/repo", "abc123")
        db.close()


class TestShellTools:
    def test_grep_cat_ls_git(self, tmp_path):
        import subprocess

        from codey.tools.shell import run_cat, run_git, run_grep, run_ls

        (tmp_path / "foo.py").write_text("print('hello')\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=str(tmp_path), check=True)
        assert run_grep("hello", tmp_path).ok
        assert "hello" in run_cat("foo.py", tmp_path).stdout
        assert run_ls("", tmp_path).ok
        assert run_git(["status"], tmp_path).ok

    def test_git_allow_list(self, tmp_path):
        from codey.tools.shell import run_git

        result = run_git(["push"], tmp_path)
        assert not result.ok
        assert "Disallowed" in result.stderr

    def test_path_traversal_blocked(self, tmp_path):
        from codey.tools.shell import run_cat

        result = run_cat("../escape.py", tmp_path)
        assert not result.ok
        assert "escapes repo root" in result.stderr


class TestIndexer:
    def test_index_repository(self, repo):
        from codey.cache.ast_cache import CacheDB
        from codey.index.indexer import index_repository

        db = CacheDB()
        result = index_repository(repo, db)
        assert result.total_files > 0
        assert result.parsed_files > 0
        syms = db.all_symbols(str(repo.resolve()), result.git_hash)
        assert any(s.name == "add" for s in syms)

        # Second run reuses cache.
        result2 = index_repository(repo, db)
        assert result2.parsed_files == 0
        db.close()


class TestCallGraph:
    def test_build_and_reverse_deps(self, repo):
        from codey.cache.ast_cache import CacheDB
        from codey.index.callgraph import build_call_graph, reverse_dependencies
        from codey.index.indexer import index_repository

        db = CacheDB()
        ir = index_repository(repo, db)
        cg = build_call_graph(repo, ir.git_hash, db)
        assert cg.call_edges > 0
        deps = reverse_dependencies(repo, ir.git_hash, db, ["main.py"])
        # 'sub' calls 'add' — no separate files, so no cross-file deps here.
        assert isinstance(deps, list)
        db.close()


class TestSchemas:
    def test_agent_report_and_review(self):
        from codey.agents.schemas import (
            AgentReport,
            Finding,
            FindingCategory,
            ReviewSummary,
            Severity,
            aggregate_severity,
        )

        f1 = Finding(category=FindingCategory.SECURITY, severity=Severity.HIGH, title="SQLi")
        r = AgentReport(agent="security", summary="found 1", findings=[f1])
        assert r.finding_count() == 1
        review = ReviewSummary(
            overall_severity=aggregate_severity([f1]),
            summary="done", commit_hash="abc",
            agent_reports={"security": r},
            total_findings=1,
            recommendation="request_changes",
        )
        assert review.overall_severity == Severity.HIGH
        assert len(review.all_findings()) == 1


class TestChunking:
    def test_chunk_file_diff(self):
        from codey.cache.ast_cache import SymbolRecord
        from codey.review.chunking import chunk_file_diff

        diff = (
            "diff --git a/foo.py b/foo.py\n"
            "@@ -1,3 +1,5 @@\n"
            " def foo():\n"
            "-    return 1\n"
            "+    return 2\n"
            "+    \n"
        )
        symbols = [SymbolRecord("foo.py", "foo", "foo", "function", 1, 3)]
        chunks = chunk_file_diff("foo.py", diff, full_source="", symbols=symbols)
        assert len(chunks) >= 1
        assert chunks[0].symbol == "foo"
        assert chunks[0].symbol_kind == "function"


class TestPipelineNoLLM:
    def test_pipeline_smoke(self, repo):
        from codey.cache.ast_cache import CacheDB
        from codey.review.pipeline import run_pipeline

        db = CacheDB()
        result = run_pipeline(repo, db, primary_llm=None, summarizer_llm=None)
        assert result.review is not None
        assert result.review.recommendation in ("approve", "request_changes", "block")
        assert "test" in result.review.agent_reports or "security" in result.review.agent_reports
        assert len(result.review.agent_reports) == 4
        db.close()


class TestLLMFactory:
    def test_build_llm_no_config(self):
        from codey.config.store import ConfigError
        from codey.llm.factory import build_llm

        with pytest.raises(ConfigError):
            build_llm()


class TestEstimateTokens:
    def test_basic(self):
        from codey.llm.factory import estimate_tokens

        assert estimate_tokens("") == 1
        assert estimate_tokens("hello world") == 2
        assert estimate_tokens("x" * 100) == 25
