"""Tests for codey.index — indexer, symbols, languages, call graph."""

from __future__ import annotations

from pathlib import Path

from codey.cache.ast_cache import CacheDB
from codey.index import callgraph
from codey.index.indexer import (
    IndexResult,
    _content_hash,
    _walk_files,
    git_head_hash,
    index_repository,
    list_repo_files,
)
from codey.index.languages import detect_language
from codey.index.symbols import extract_symbols
from tests.conftest import commit, init_repo

# ---------------------------------------------------------------------------
# languages
# ---------------------------------------------------------------------------

def test_detect_language_known_extensions():
    assert detect_language(Path("a.py")) == "python"
    assert detect_language(Path("a.js")) == "javascript"
    assert detect_language(Path("a.tsx")) == "tsx"
    assert detect_language(Path("a.go")) == "go"
    assert detect_language(Path("a.rs")) == "rust"
    assert detect_language(Path("A.PY")) == "python"  # case-insensitive


def test_detect_language_unknown():
    assert detect_language(Path("a.unknownext")) is None


# ---------------------------------------------------------------------------
# symbols
# ---------------------------------------------------------------------------

def test_extract_symbols_python():
    from tree_sitter_language_pack import get_parser

    src = b"class Foo:\n    def bar(self):\n        pass\n\ndef top():\n    pass\n"
    tree = get_parser("python").parse(src)
    syms = extract_symbols(tree, "m.py", "python")
    names = {s.name for s in syms}
    qnames = {s.qualified_name for s in syms}
    assert names == {"Foo", "bar", "top"}
    assert "Foo.bar" in qnames
    assert "top" in qnames


def test_extract_symbols_generic_js():
    from tree_sitter_language_pack import get_parser

    src = b"function hello() {}\nclass C { method() {} }\n"
    tree = get_parser("javascript").parse(src)
    syms = extract_symbols(tree, "m.js", "javascript")
    names = {s.name for s in syms}
    assert "hello" in names
    assert "C" in names


def test_symbol_extractor_generic_fallback():
    from tree_sitter_language_pack import get_parser

    # Go: uses the generic extractor.
    src = b"package main\nfunc hello() {}\n"
    tree = get_parser("go").parse(src)
    syms = extract_symbols(tree, "m.go", "go")
    assert any(s.name == "hello" for s in syms)


# ---------------------------------------------------------------------------
# indexer
# ---------------------------------------------------------------------------

def test_git_head_hash(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    init_repo(r)
    (r / "x.py").write_text("x=1\n")
    commit(r, "init")
    assert git_head_hash(r) is not None
    assert git_head_hash(tmp_path / "not-a-repo") is None


def test_list_repo_files_respects_git(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    init_repo(r)
    (r / "a.py").write_text("x=1\n")
    (r / "ignored.py").write_text("x=1\n")
    (r / ".gitignore").write_text("ignored.py\n")
    commit(r, "init")
    files = list_repo_files(r)
    rels = {str(f.relative_to(r)) for f in files}
    assert "a.py" in rels
    assert "ignored.py" not in rels


def test_list_repo_files_falls_back_to_walk(tmp_path):
    # no git repo -> directory walk fallback
    d = tmp_path / "plain"
    d.mkdir()
    (d / "a.py").write_text("x=1\n")
    (d / "__pycache__").mkdir()
    (d / "__pycache__" / "c.pyc").write_bytes(b"")
    files = _walk_files(d)
    rels = {str(f.relative_to(d)) for f in files}
    assert "a.py" in rels
    assert all("__pycache__" not in str(f) for f in files)


def test_content_hash_deterministic():
    assert _content_hash("abc") == _content_hash("abc")
    assert _content_hash("abc") != _content_hash("abd")


def test_index_repository_indexes_and_reuses(repo):
    db = CacheDB()
    r1 = index_repository(repo, db)
    assert r1.total_files == 1
    assert r1.parsed_files == 1

    # Second run: same hash -> no-op.
    r2 = index_repository(repo, db)
    assert r2.parsed_files == 0
    assert r2.reused_files == 1

    # New commit changing the file -> reuse unchanged across commits.
    (repo / "main.py").write_text("def z():\n    return 1\n", encoding="utf-8")
    commit(repo, "change")
    r3 = index_repository(repo, db)
    assert r3.parsed_files == 1
    db.close()


def test_index_repository_uses_canonical_cache_key(repo):
    """cache_repo_path controls the cache key, decoupling it from the scan dir."""
    db = CacheDB()
    index_repository(repo, db, cache_repo_path=repo)
    canonical = str(repo.resolve())
    assert db.has_indexed_hash(canonical, git_head_hash(repo))
    db.close()


def test_index_repository_skips_unsupported_files(repo):
    db = CacheDB()
    (repo / "notes.md").write_text("# hi\n", encoding="utf-8")
    commit(repo, "add md")
    result = index_repository(repo, db, force=True)
    assert result.skipped_files >= 1
    db.close()


def test_index_result_dataclass_defaults():
    r = IndexResult(git_hash="h")
    assert r.total_files == 0
    assert r.changed_files == []


# ---------------------------------------------------------------------------
# callgraph
# ---------------------------------------------------------------------------

def test_module_name_for_path():
    repo = Path("/repo")
    assert callgraph._module_name_for_path(repo, repo / "codey" / "agents" / "x.py") == "codey.agents.x"
    assert callgraph._module_name_for_path(repo, repo / "pkg" / "__init__.py") == "pkg"
    assert callgraph._module_name_for_path(Path("/other"), repo / "x.py") == ""


def test_resolve_module_path(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "mod.py").write_text("x=1\n")
    init = tmp_path / "pkg2"
    init.mkdir()
    (init / "__init__.py").write_text("")
    assert callgraph.resolve_module_path(tmp_path, "pkg.mod") == pkg / "mod.py"
    assert callgraph.resolve_module_path(tmp_path, "pkg2") == init / "__init__.py"
    assert callgraph.resolve_module_path(tmp_path, "nope") is None


def test_extract_imports(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("import os\nfrom pathlib import Path\n", encoding="utf-8")
    edges = callgraph._extract_imports(f, "m.py")
    modules = {e.module for e in edges}
    assert modules == {"os", "pathlib"}


def test_build_call_graph_and_reverse_dependencies(tmp_path):
    """A cross-file dependency (b.py imports a.py) yields a reverse dep."""
    r = tmp_path / "r"
    r.mkdir()
    init_repo(r)
    (r / "a.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (r / "b.py").write_text("import a\n\ndef use():\n    return a.helper()\n", encoding="utf-8")
    commit(r, "init")

    db = CacheDB()
    ir = index_repository(r, db)
    cg = callgraph.build_call_graph(r, ir.git_hash, db)
    assert cg.files_processed >= 1

    # Changing a.py affects b.py (b imports a and calls a.helper).
    deps = callgraph.reverse_dependencies(r, ir.git_hash, db, [str(r / "a.py")])
    assert "b.py" in deps
    db.close()


def test_reverse_dependencies_no_bare_name_collisions(tmp_path):
    """Two unrelated classes with the same method name must not cross-link."""
    r = tmp_path / "r"
    r.mkdir()
    init_repo(r)
    (r / "a.py").write_text("class A:\n    def run(self):\n        pass\n", encoding="utf-8")
    (r / "b.py").write_text("class B:\n    def run(self):\n        pass\n", encoding="utf-8")
    commit(r, "init")

    db = CacheDB()
    ir = index_repository(r, db)
    callgraph.build_call_graph(r, ir.git_hash, db)
    deps = callgraph.reverse_dependencies(r, ir.git_hash, db, [str(r / "a.py")])
    assert "b.py" not in deps
    db.close()


def test_reverse_dependencies_empty_paths():
    db = CacheDB()
    assert callgraph.reverse_dependencies(Path("."), "h", db, []) == []
    db.close()
