"""SQLite-backed AST/symbol cache keyed by git hash (WAL mode, thread-safe)."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

__all__ = ["CacheDB", "FileEntry", "default_db_path"]

DEFAULT_DB_PATH = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "codey" / "codey.db"


def default_db_path() -> Path:
    return DEFAULT_DB_PATH


@dataclass
class FileEntry:
    rel_path: str
    language: str
    content_hash: str
    mtime: float
    byte_count: int


@dataclass
class SymbolRecord:
    rel_path: str
    name: str
    qualified_name: str
    kind: str
    line_start: int
    line_end: int


@dataclass
class CallEdge:
    caller_path: str
    caller_qname: str
    callee_name: str
    callee_path: str | None
    callee_qname: str | None
    line: int


@dataclass
class ImportEdge:
    rel_path: str
    module: str
    imported_name: str | None
    alias: str | None
    line: int


def _fe(row) -> FileEntry:
    return FileEntry(row["rel_path"], row["language"], row["content_hash"], row["mtime"], row["byte_count"])


def _sym(row) -> SymbolRecord:
    return SymbolRecord(row["rel_path"], row["name"], row["qualified_name"], row["kind"],
                        row["line_start"], row["line_end"])


def _ce(row) -> CallEdge:
    return CallEdge(row["caller_path"], row["caller_qname"], row["callee_name"],
                    row["callee_path"], row["callee_qname"], row["line"])


def _ie(row) -> ImportEdge:
    return ImportEdge(row["rel_path"], row["module"], row["imported_name"], row["alias"], row["line"])


class CacheDB:
    """Thread-safe sqlite wrapper: the review graph runs parallel nodes on a
    thread pool, so every statement is serialized through a lock and the
    connection is opened with ``check_same_thread=False``."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        import threading

        path = Path(db_path) if db_path else DEFAULT_DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._migrate()

    # --- primitives ----

    def _execute(self, query: str, params: tuple | list = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(query, params)

    def _query(self, query: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(query, params))

    def _query_one(self, query: str, params: tuple | list = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(query, params).fetchone()

    def _executemany(self, query: str, params: Sequence) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.executemany(query, params)

    def _commit(self) -> None:
        with self._lock:
            self._conn.commit()

    # --- lifecycle ----

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> CacheDB:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    # --- schema ----

    def _migrate(self) -> None:
        schema = resources.files("codey.cache").joinpath("schema.sql").read_text("utf-8")
        self._conn.executescript(schema)
        self._drop_legacy_columns()
        self._commit()

    def _drop_legacy_columns(self) -> None:
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(file_entries)")}
        for legacy in ("ast_json", "symbols_json"):
            if legacy in cols:
                self._conn.execute(f"ALTER TABLE file_entries DROP COLUMN {legacy}")

    # --- index runs ----

    def upsert_index_run(self, repo_path: str, git_hash: str, file_count: int = 0) -> None:
        self._execute(
            """INSERT INTO index_runs (repo_path, git_hash, file_count) VALUES (?, ?, ?)
               ON CONFLICT (repo_path, git_hash) DO UPDATE SET file_count=excluded.file_count""",
            (repo_path, git_hash, file_count),
        )
        self._commit()

    def last_indexed_hash(self, repo_path: str) -> str | None:
        row = self._query_one(
            "SELECT git_hash FROM index_runs WHERE repo_path=? ORDER BY indexed_at DESC LIMIT 1",
            (repo_path,),
        )
        return row[0] if row else None

    def has_indexed_hash(self, repo_path: str, git_hash: str) -> bool:
        return self._query_one(
            "SELECT 1 FROM index_runs WHERE repo_path=? AND git_hash=? LIMIT 1", (repo_path, git_hash),
        ) is not None

    def has_call_edges(self, repo_path: str, git_hash: str) -> bool:
        if self._query_one("SELECT 1 FROM call_edges WHERE repo_path=? AND git_hash=? LIMIT 1", (repo_path, git_hash)):
            return True
        return self._query_one(
            "SELECT 1 FROM import_edges WHERE repo_path=? AND git_hash=? LIMIT 1", (repo_path, git_hash),
        ) is not None

    # --- file entries ----

    def upsert_file_entry(self, repo_path: str, git_hash: str, entry: FileEntry) -> None:
        self._execute(
            """INSERT INTO file_entries (repo_path, git_hash, rel_path, language, content_hash, mtime, byte_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (repo_path, git_hash, rel_path) DO UPDATE SET
                 language=excluded.language, content_hash=excluded.content_hash,
                 mtime=excluded.mtime, byte_count=excluded.byte_count""",
            (repo_path, git_hash, entry.rel_path, entry.language, entry.content_hash, entry.mtime, entry.byte_count),
        )
        self._commit()

    def get_file_entry(self, repo_path: str, git_hash: str, rel_path: str) -> FileEntry | None:
        row = self._query_one(
            "SELECT rel_path, language, content_hash, mtime, byte_count FROM file_entries "
            "WHERE repo_path=? AND git_hash=? AND rel_path=?",
            (repo_path, git_hash, rel_path),
        )
        return _fe(row) if row else None

    def list_file_entries(self, repo_path: str, git_hash: str) -> Iterable[FileEntry]:
        for row in self._query(
            "SELECT rel_path, language, content_hash, mtime, byte_count FROM file_entries "
            "WHERE repo_path=? AND git_hash=?",
            (repo_path, git_hash),
        ):
            yield _fe(row)

    def list_file_rel_paths(self, repo_path: str, git_hash: str) -> list[str]:
        return [r[0] for r in self._query(
            "SELECT rel_path FROM file_entries WHERE repo_path=? AND git_hash=?", (repo_path, git_hash),
        )]

    def file_entry_hashes(self, repo_path: str, git_hash: str) -> dict[str, str]:
        return {row["rel_path"]: row["content_hash"] for row in self._query(
            "SELECT rel_path, content_hash FROM file_entries WHERE repo_path=? AND git_hash=?",
            (repo_path, git_hash),
        )}

    # --- symbols ----

    def bulk_upsert_symbols(self, repo_path: str, git_hash: str, symbols: Sequence[SymbolRecord]) -> None:
        self._executemany(
            """INSERT INTO symbols (repo_path, git_hash, rel_path, name, qualified_name, kind, line_start, line_end)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (repo_path, git_hash, rel_path, qualified_name)
               DO UPDATE SET kind=excluded.kind, line_start=excluded.line_start, line_end=excluded.line_end""",
            [(repo_path, git_hash, s.rel_path, s.name, s.qualified_name, s.kind, s.line_start, s.line_end)
             for s in symbols],
        )
        self._commit()

    def symbols_in_file(self, repo_path: str, git_hash: str, rel_path: str) -> list[SymbolRecord]:
        return [_sym(r) for r in self._query(
            "SELECT rel_path, name, qualified_name, kind, line_start, line_end FROM symbols "
            "WHERE repo_path=? AND git_hash=? AND rel_path=?",
            (repo_path, git_hash, rel_path),
        )]

    def all_symbols(self, repo_path: str, git_hash: str) -> list[SymbolRecord]:
        return [_sym(r) for r in self._query(
            "SELECT rel_path, name, qualified_name, kind, line_start, line_end FROM symbols "
            "WHERE repo_path=? AND git_hash=?",
            (repo_path, git_hash),
        )]

    # --- call edges ----

    def bulk_insert_call_edges(self, repo_path: str, git_hash: str, edges: Sequence[CallEdge]) -> None:
        self._executemany(
            """INSERT INTO call_edges (repo_path, git_hash, caller_path, caller_qname, callee_name,
                                      callee_path, callee_qname, line)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [(repo_path, git_hash, e.caller_path, e.caller_qname, e.callee_name, e.callee_path, e.callee_qname, e.line)
             for e in edges],
        )
        self._commit()

    def callers_of(self, repo_path: str, git_hash: str, callee_name: str) -> list[CallEdge]:
        return [_ce(r) for r in self._query(
            "SELECT caller_path, caller_qname, callee_name, callee_path, callee_qname, line FROM call_edges "
            "WHERE repo_path=? AND git_hash=? AND callee_name=?",
            (repo_path, git_hash, callee_name),
        )]

    def all_call_edges(self, repo_path: str, git_hash: str) -> list[CallEdge]:
        return [_ce(r) for r in self._query(
            "SELECT caller_path, caller_qname, callee_name, callee_path, callee_qname, line FROM call_edges "
            "WHERE repo_path=? AND git_hash=?",
            (repo_path, git_hash),
        )]

    # --- import edges ----

    def bulk_insert_import_edges(self, repo_path: str, git_hash: str, edges: Sequence[ImportEdge]) -> None:
        self._executemany(
            """INSERT INTO import_edges (repo_path, git_hash, rel_path, module, imported_name, alias, line)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [(repo_path, git_hash, e.rel_path, e.module, e.imported_name, e.alias, e.line) for e in edges],
        )
        self._commit()

    def importers_of_module(self, repo_path: str, git_hash: str, module: str) -> list[ImportEdge]:
        return [_ie(r) for r in self._query(
            "SELECT rel_path, module, imported_name, alias, line FROM import_edges "
            "WHERE repo_path=? AND git_hash=? AND module=?",
            (repo_path, git_hash, module),
        )]

    def all_imports_for_modules(self, repo_path: str, git_hash: str, modules: set[str]) -> list[ImportEdge]:
        if not modules:
            return []
        placeholders = ",".join("?" * len(modules))
        rows = self._query(
            f"SELECT rel_path, module, imported_name, alias, line FROM import_edges "
            f"WHERE repo_path=? AND git_hash=? AND module IN ({placeholders})",
            (repo_path, git_hash, *modules),
        )
        return [_ie(r) for r in rows]

    def clear_run(self, repo_path: str, git_hash: str) -> None:
        for table in ("file_entries", "symbols", "call_edges", "import_edges", "index_runs"):
            self._execute(f"DELETE FROM {table} WHERE repo_path=? AND git_hash=?", (repo_path, git_hash))
        self._commit()
