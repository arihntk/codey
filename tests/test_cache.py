"""Tests for codey.cache.ast_cache — the sqlite-backed index cache."""

from __future__ import annotations

import sqlite3
import threading

import pytest

from codey.cache.ast_cache import (
    CacheDB,
    CallEdge,
    FileEntry,
    ImportEdge,
    SymbolRecord,
    default_db_path,
)


def test_default_db_path_shape():
    # The autouse _isolate fixture points it at a temp cache dir.
    p = default_db_path()
    assert p.name == "codey.db"
    assert ".cache" in str(p).lower() or "codey-cache" in str(p)


def test_context_manager_closes():
    db = CacheDB()
    with db:
        db.upsert_index_run("/repo", "h1")
    # After __exit__ the underlying connection is closed.
    with pytest.raises(sqlite3.ProgrammingError):
        db.upsert_index_run("/repo", "h2")


def test_full_crud_round_trip():
    db = CacheDB()
    db.upsert_index_run("/repo", "h1", file_count=2)
    assert db.has_indexed_hash("/repo", "h1")
    assert db.last_indexed_hash("/repo") == "h1"

    entry = FileEntry("a.py", "python", "chash", 123.0, 42)
    db.upsert_file_entry("/repo", "h1", entry)
    got = db.get_file_entry("/repo", "h1", "a.py")
    assert got is not None
    assert got.rel_path == "a.py"
    assert got.language == "python"
    assert got.content_hash == "chash"
    assert got.mtime == 123.0
    assert got.byte_count == 42

    assert db.list_file_rel_paths("/repo", "h1") == ["a.py"]
    assert db.file_entry_hashes("/repo", "h1") == {"a.py": "chash"}
    assert [e.rel_path for e in db.list_file_entries("/repo", "h1")] == ["a.py"]

    # symbols
    syms = [SymbolRecord("a.py", "foo", "a.foo", "function", 1, 5)]
    db.bulk_upsert_symbols("/repo", "h1", syms)
    assert db.symbols_in_file("/repo", "h1", "a.py")[0].name == "foo"
    assert db.all_symbols("/repo", "h1")[0].qualified_name == "a.foo"

    # call edges
    db.bulk_insert_call_edges("/repo", "h1", [CallEdge("a.py", "a.foo", "bar", "b.py", "b.bar", 3)])
    callers = db.callers_of("/repo", "h1", "bar")
    assert callers[0].caller_qname == "a.foo"
    assert db.all_call_edges("/repo", "h1")[0].callee_qname == "b.bar"

    # import edges
    db.bulk_insert_import_edges("/repo", "h1", [ImportEdge("a.py", "os", "path", "p", 1)])
    assert db.importers_of_module("/repo", "h1", "os")[0].alias == "p"
    assert db.all_imports_for_modules("/repo", "h1", {"os"})[0].imported_name == "path"
    assert db.all_imports_for_modules("/repo", "h1", set()) == []

    db.clear_run("/repo", "h1")
    assert not db.has_indexed_hash("/repo", "h1")
    assert db.list_file_rel_paths("/repo", "h1") == []
    db.close()


def test_file_entry_upsert_updates_existing():
    db = CacheDB()
    db.upsert_index_run("/repo", "h1")
    db.upsert_file_entry("/repo", "h1", FileEntry("a.py", "python", "v1", 1.0, 1))
    db.upsert_file_entry("/repo", "h1", FileEntry("a.py", "python", "v2", 2.0, 2))
    got = db.get_file_entry("/repo", "h1", "a.py")
    assert got.content_hash == "v2"
    assert got.byte_count == 2
    db.close()


def test_has_call_edges_gates_on_call_and_import():
    db = CacheDB()
    db.upsert_index_run("/repo", "h1")
    assert db.has_call_edges("/repo", "h1") is False
    db.bulk_insert_import_edges("/repo", "h1", [ImportEdge("a.py", "os", None, None, 1)])
    assert db.has_call_edges("/repo", "h1") is True  # import-only still counts
    db.close()


def test_migration_drops_legacy_columns():
    db = CacheDB()
    db._conn.execute("ALTER TABLE file_entries ADD COLUMN ast_json TEXT NOT NULL DEFAULT '{}'")
    db._conn.execute("ALTER TABLE file_entries ADD COLUMN symbols_json TEXT NOT NULL DEFAULT '[]'")
    db._conn.commit()
    db.close()

    db2 = CacheDB()  # reopen triggers migration
    cols = {r[1] for r in db2._conn.execute("PRAGMA table_info(file_entries)")}
    assert "ast_json" not in cols
    assert "symbols_json" not in cols
    db2.close()


def test_threaded_access_is_serialized():
    """Concurrent reads/writes must not raise (lock serializes sqlite)."""
    db = CacheDB()
    db.upsert_index_run("/repo", "h1")
    errors = []

    def worker(i):
        try:
            db.upsert_file_entry("/repo", "h1", FileEntry(f"f{i}.py", "python", "c", 1.0, 1))
            db.get_file_entry("/repo", "h1", f"f{i}.py")
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(db.list_file_rel_paths("/repo", "h1")) == 20
    db.close()


def test_clear_run_removes_all_tables():
    db = CacheDB()
    db.upsert_index_run("/repo", "h1")
    db.upsert_file_entry("/repo", "h1", FileEntry("a.py", "python", "c", 1.0, 1))
    db.bulk_upsert_symbols("/repo", "h1", [SymbolRecord("a.py", "f", "a.f", "function", 1, 2)])
    db.bulk_insert_call_edges("/repo", "h1", [CallEdge("a.py", "a.f", "g", None, None, 1)])
    db.bulk_insert_import_edges("/repo", "h1", [ImportEdge("a.py", "os", None, None, 1)])
    db.clear_run("/repo", "h1")
    assert db.all_symbols("/repo", "h1") == []
    assert db.all_call_edges("/repo", "h1") == []
    assert db.all_imports_for_modules("/repo", "h1", {"os"}) == []
    db.close()


def test_symbols_upsert_on_conflict_updates():
    db = CacheDB()
    db.upsert_index_run("/repo", "h1")
    db.bulk_upsert_symbols("/repo", "h1", [SymbolRecord("a.py", "f", "a.f", "function", 1, 2)])
    db.bulk_upsert_symbols("/repo", "h1", [SymbolRecord("a.py", "f", "a.f", "method", 3, 9)])
    s = db.symbols_in_file("/repo", "h1", "a.py")
    assert len(s) == 1
    assert s[0].kind == "method"
    assert s[0].line_start == 3
    db.close()
