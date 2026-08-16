"""Tests for codey.review.chunking — AST-aware diff chunking."""

from __future__ import annotations

from codey.agents.context import DiffChunk
from codey.cache.ast_cache import SymbolRecord
from codey.review.chunking import (
    _enclosing_symbol,
    _hunk_new_len,
    _merge_adjacent,
    _split_hunks,
    chunk_diff,
    chunk_file_diff,
)


def _sym(name, qname, kind, start, end):
    return SymbolRecord("f.py", name, qname, kind, start, end)


def test_chunk_file_diff_no_hunks_is_module():
    text = "diff --git a/f.py b/f.py\nindex 1..2 100644\n"
    chunks = chunk_file_diff("f.py", text)
    assert len(chunks) == 1
    assert chunks[0].symbol == "<module>"
    assert chunks[0].symbol_kind == "module"
    assert chunks[0].line_start == 0


def test_chunk_file_diff_maps_hunk_to_symbol():
    diff = (
        "diff --git a/f.py b/f.py\n"
        "@@ -1,3 +1,4 @@\n"
        " def foo():\n"
        "-    return 1\n"
        "+    return 2\n"
    )
    symbols = [_sym("foo", "foo", "function", 1, 3)]
    chunks = chunk_file_diff("f.py", diff, symbols=symbols)
    assert len(chunks) == 1
    assert chunks[0].symbol == "foo"
    assert chunks[0].symbol_kind == "function"


def test_chunk_file_diff_module_level_when_no_symbol_matches():
    diff = (
        "diff --git a/f.py b/f.py\n"
        "@@ -10,1 +10,1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
    )
    symbols = [_sym("foo", "foo", "function", 1, 3)]
    chunks = chunk_file_diff("f.py", diff, symbols=symbols)
    assert chunks[0].symbol == "<module>"


def test_split_hunks_single_and_multi():
    text = (
        "diff --git a/f.py b/f.py\n"
        "@@ -1,1 +1,1 @@\n"
        " a\n"
        "@@ -5,1 +5,1 @@\n"
        " b\n"
    )
    hunks = _split_hunks(text)
    assert [h[0] for h in hunks] == [1, 5]
    assert len(hunks) == 2


def test_split_hunks_handles_combined_merge_header():
    text = "@@@ -1,1 -1,1 +1,1 @@@\n a\n"
    hunks = _split_hunks(text)
    assert hunks[0][0] == 1


def test_hunk_new_len_counts_added_and_context_not_removed():
    lines = ["@@ -1,2 +1,3 @@\n", " ctx\n", "-removed\n", "+added\n"]
    assert _hunk_new_len(lines) == 2  # ctx + added


def test_enclosing_symbol_innermost_wins():
    symbols = [
        _sym("C", "C", "class", 1, 10),
        _sym("m", "C.m", "method", 3, 6),
    ]
    name, kind = _enclosing_symbol(4, 5, symbols)
    assert name == "C.m"
    assert kind == "method"


def test_enclosing_symbol_partial_overlap():
    symbols = [_sym("foo", "foo", "function", 1, 3)]
    assert _enclosing_symbol(2, 5, symbols) == ("foo", "function")
    assert _enclosing_symbol(10, 12, symbols) == ("<module>", "module")


def test_merge_adjacent_same_symbol():
    c1 = DiffChunk("f.py", "python", "foo", "function", "d1", 1, 5)
    c2 = DiffChunk("f.py", "python", "foo", "function", "d2", 6, 10)
    c3 = DiffChunk("f.py", "python", "bar", "function", "d3", 20, 30)
    merged = _merge_adjacent([c1, c2, c3])
    assert len(merged) == 2
    assert merged[0].line_start == 1
    assert merged[0].line_end == 10
    assert merged[0].diff_text == "d1\nd2"


def test_chunk_diff_multiple_files():
    diffs = {
        "a.py": "diff --git a/a.py b/a.py\n@@ -1,1 +1,1 @@\n x\n",
        "b.py": "diff --git a/b.py b/b.py\n@@ -1,1 +1,1 @@\n y\n",
    }
    chunks = chunk_diff(diffs)
    assert {c.file_path for c in chunks} == {"a.py", "b.py"}


def test_chunk_diff_uses_cached_symbols():
    from codey.cache.ast_cache import CacheDB

    db = CacheDB()
    db.upsert_index_run("/repo", "h1")
    db.bulk_upsert_symbols("/repo", "h1", [_sym("foo", "foo", "function", 1, 3)])
    diffs = {"f.py": "diff --git a/f.py b/f.py\n@@ -1,1 +1,1 @@\n x\n"}
    chunks = chunk_diff(diffs, db=db, repo_path="/repo", git_hash="h1")
    assert chunks[0].symbol == "foo"
    db.close()
