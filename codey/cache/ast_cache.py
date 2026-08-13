"""SQLite-backed AST metadata cache keyed by git hash.

On review, the pipeline computes the current HEAD hash and asks the cache
which files have changed since the last indexed hash.  Only those files are
re-parsed; the rest reuse the stored AST/symbols.  All data is scoped by
absolute repo path so a single global DB can serve multiple repositories.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "CacheDB",
    "FileEntry",
    "cached_db_path",
    "default_db_path",
]

DEFAULT_DB_PATH = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "codey" / "codey.db"


def default_db_path() -> Path:
    return DEFAULT_DB_PATH


def cached_db_path() -> str:
    return str(DEFAULT_DB_PATH)


@dataclass
class FileEntry:
    """A stored file's AST metadata."""

    rel_path: str
    language: str
    content_hash: str
    ast_json: str
    symbols_json: str
    mtime: float
    byte_count: int


@dataclass
class SymbolRecord:
    """A stored symbol record for call-graph queries."""

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


class CacheDB:
    """Thin sqlite wrapper for the AST/symbol cache."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        path = Path(db_path) if db_path else DEFAULT_DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = path
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    # --- lifecycle ----

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "CacheDB":
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
        self._conn.commit()

    # --- index runs ----

    def upsert_index_run(self, repo_path: str, git_hash: str, file_count: int = 0) -> None:
        self._conn.execute(
            """INSERT INTO index_runs (repo_path, git_hash, file_count)
               VALUES (?, ?, ?)
               ON CONFLICT (repo_path, git_hash) DO UPDATE SET file_count=excluded.file_count""",
            (repo_path, git_hash, file_count),
        )
        self._conn.commit()

    def last_indexed_hash(self, repo_path: str) -> str | None:
        row = self._conn.execute(
            """SELECT git_hash FROM index_runs
               WHERE repo_path=? ORDER BY indexed_at DESC LIMIT 1""",
            (repo_path,),
        ).fetchone()
        return row[0] if row else None

    def has_indexed_hash(self, repo_path: str, git_hash: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM index_runs WHERE repo_path=? AND git_hash=? LIMIT 1",
            (repo_path, git_hash),
        ).fetchone()
        return row is not None

    # --- file entries ----

    def upsert_file_entry(
        self,
        repo_path: str,
        git_hash: str,
        entry: FileEntry,
    ) -> None:
        self._conn.execute(
            """INSERT INTO file_entries
                 (repo_path, git_hash, rel_path, language, content_hash,
                  ast_json, symbols_json, mtime, byte_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (repo_path, git_hash, rel_path) DO UPDATE SET
                 language=excluded.language,
                 content_hash=excluded.content_hash,
                 ast_json=excluded.ast_json,
                 symbols_json=excluded.symbols_json,
                 mtime=excluded.mtime,
                 byte_count=excluded.byte_count""",
            (
                repo_path,
                git_hash,
                entry.rel_path,
                entry.language,
                entry.content_hash,
                entry.ast_json,
                entry.symbols_json,
                entry.mtime,
                entry.byte_count,
            ),
        )
        self._conn.commit()

    def get_file_entry(
        self,
        repo_path: str,
        git_hash: str,
        rel_path: str,
    ) -> FileEntry | None:
        row = self._conn.execute(
            """SELECT rel_path, language, content_hash, ast_json, symbols_json,
                      mtime, byte_count
               FROM file_entries
               WHERE repo_path=? AND git_hash=? AND rel_path=?""",
            (repo_path, git_hash, rel_path),
        ).fetchone()
        if not row:
            return None
        return FileEntry(
            rel_path=row["rel_path"],
            language=row["language"],
            content_hash=row["content_hash"],
            ast_json=row["ast_json"],
            symbols_json=row["symbols_json"],
            mtime=row["mtime"],
            byte_count=row["byte_count"],
        )

    def list_file_entries(
        self,
        repo_path: str,
        git_hash: str,
    ) -> Iterable[FileEntry]:
        for row in self._conn.execute(
            """SELECT rel_path, language, content_hash, ast_json, symbols_json,
                      mtime, byte_count
               FROM file_entries
               WHERE repo_path=? AND git_hash=?""",
            (repo_path, git_hash),
        ):
            yield FileEntry(
                rel_path=row["rel_path"],
                language=row["language"],
                content_hash=row["content_hash"],
                ast_json=row["ast_json"],
                symbols_json=row["symbols_json"],
                mtime=row["mtime"],
                byte_count=row["byte_count"],
            )

    def list_file_rel_paths(self, repo_path: str, git_hash: str) -> list[str]:
        return [
            r[0]
            for r in self._conn.execute(
                "SELECT rel_path FROM file_entries WHERE repo_path=? AND git_hash=?",
                (repo_path, git_hash),
            )
        ]

    def file_entry_hashes(self, repo_path: str, git_hash: str) -> dict[str, str]:
        return {
            row["rel_path"]: row["content_hash"]
            for row in self._conn.execute(
                "SELECT rel_path, content_hash FROM file_entries WHERE repo_path=? AND git_hash=?",
                (repo_path, git_hash),
            )
        }

    # --- symbols ----

    def bulk_upsert_symbols(
        self,
        repo_path: str,
        git_hash: str,
        symbols: Sequence[SymbolRecord],
    ) -> None:
        rows = [
            (repo_path, git_hash, s.rel_path, s.name, s.qualified_name, s.kind, s.line_start, s.line_end)
            for s in symbols
        ]
        self._conn.executemany(
            """INSERT INTO symbols
                 (repo_path, git_hash, rel_path, name, qualified_name, kind,
                  line_start, line_end)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (repo_path, git_hash, rel_path, qualified_name)
               DO UPDATE SET kind=excluded.kind, line_start=excluded.line_start,
                             line_end=excluded.line_end""",
            rows,
        )
        self._conn.commit()

    def symbols_in_file(self, repo_path: str, git_hash: str, rel_path: str) -> list[SymbolRecord]:
        return [
            SymbolRecord(
                rel_path=row["rel_path"],
                name=row["name"],
                qualified_name=row["qualified_name"],
                kind=row["kind"],
                line_start=row["line_start"],
                line_end=row["line_end"],
            )
            for row in self._conn.execute(
                """SELECT rel_path, name, qualified_name, kind, line_start,
                          line_end
                   FROM symbols
                   WHERE repo_path=? AND git_hash=? AND rel_path=?""",
                (repo_path, git_hash, rel_path),
            )
        ]

    def all_symbols(self, repo_path: str, git_hash: str) -> list[SymbolRecord]:
        return [
            SymbolRecord(
                rel_path=row["rel_path"],
                name=row["name"],
                qualified_name=row["qualified_name"],
                kind=row["kind"],
                line_start=row["line_start"],
                line_end=row["line_end"],
            )
            for row in self._conn.execute(
                """SELECT rel_path, name, qualified_name, kind, line_start,
                          line_end
                   FROM symbols
                   WHERE repo_path=? AND git_hash=?""",
                (repo_path, git_hash),
            )
        ]

    # --- call edges ----

    def bulk_insert_call_edges(
        self,
        repo_path: str,
        git_hash: str,
        edges: Sequence[CallEdge],
    ) -> None:
        self._conn.executemany(
            """INSERT INTO call_edges
                 (repo_path, git_hash, caller_path, caller_qname, callee_name,
                  callee_path, callee_qname, line)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (repo_path, git_hash, e.caller_path, e.caller_qname, e.callee_name,
                 e.callee_path, e.callee_qname, e.line)
                for e in edges
            ],
        )
        self._conn.commit()

    def callers_of(self, repo_path: str, git_hash: str, callee_name: str) -> list[CallEdge]:
        return [
            CallEdge(
                caller_path=row["caller_path"],
                caller_qname=row["caller_qname"],
                callee_name=row["callee_name"],
                callee_path=row["callee_path"],
                callee_qname=row["callee_qname"],
                line=row["line"],
            )
            for row in self._conn.execute(
                """SELECT caller_path, caller_qname, callee_name, callee_path,
                          callee_qname, line
                   FROM call_edges
                   WHERE repo_path=? AND git_hash=? AND callee_name=?""",
                (repo_path, git_hash, callee_name),
            )
        ]

    # --- import edges ----

    def bulk_insert_import_edges(
        self,
        repo_path: str,
        git_hash: str,
        edges: Sequence[ImportEdge],
    ) -> None:
        self._conn.executemany(
            """INSERT INTO import_edges
                 (repo_path, git_hash, rel_path, module, imported_name, alias,
                  line)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (repo_path, git_hash, e.rel_path, e.module, e.imported_name,
                 e.alias, e.line)
                for e in edges
            ],
        )
        self._conn.commit()

    def importers_of_module(self, repo_path: str, git_hash: str, module: str) -> list[ImportEdge]:
        return [
            ImportEdge(
                rel_path=row["rel_path"],
                module=row["module"],
                imported_name=row["imported_name"],
                alias=row["alias"],
                line=row["line"],
            )
            for row in self._conn.execute(
                """SELECT rel_path, module, imported_name, alias, line
                   FROM import_edges
                   WHERE repo_path=? AND git_hash=? AND module=?""",
                (repo_path, git_hash, module),
            )
        ]

    def all_imports_for_modules(
        self,
        repo_path: str,
        git_hash: str,
        modules: set[str],
    ) -> list[ImportEdge]:
        """Return import edges for any of the given module strings."""
        if not modules:
            return []
        placeholders = ",".join("?" * len(modules))
        rows = self._conn.execute(
            f"""SELECT rel_path, module, imported_name, alias, line
                FROM import_edges
                WHERE repo_path=? AND git_hash=?
                  AND module IN ({placeholders})""",
            (repo_path, git_hash, *modules),
        )
        return [
            ImportEdge(
                rel_path=row["rel_path"],
                module=row["module"],
                imported_name=row["imported_name"],
                alias=row["alias"],
                line=row["line"],
            )
            for row in rows
        ]

    def all_call_edges(self, repo_path: str, git_hash: str) -> list[CallEdge]:
        return [
            CallEdge(
                caller_path=row["caller_path"],
                caller_qname=row["caller_qname"],
                callee_name=row["callee_name"],
                callee_path=row["callee_path"],
                callee_qname=row["callee_qname"],
                line=row["line"],
            )
            for row in self._conn.execute(
                """SELECT caller_path, caller_qname, callee_name, callee_path,
                          callee_qname, line
                   FROM call_edges
                   WHERE repo_path=? AND git_hash=?""",
                (repo_path, git_hash),
            )
        ]

    def clear_run(self, repo_path: str, git_hash: str) -> None:
        """Delete all data for a given (repo, git_hash) run."""
        self._conn.execute("DELETE FROM file_entries WHERE repo_path=? AND git_hash=?",
                           (repo_path, git_hash))
        self._conn.execute("DELETE FROM symbols WHERE repo_path=? AND git_hash=?",
                           (repo_path, git_hash))
        self._conn.execute("DELETE FROM call_edges WHERE repo_path=? AND git_hash=?",
                           (repo_path, git_hash))
        self._conn.execute("DELETE FROM import_edges WHERE repo_path=? AND git_hash=?",
                           (repo_path, git_hash))
        self._conn.execute("DELETE FROM index_runs WHERE repo_path=? AND git_hash=?",
                           (repo_path, git_hash))
        self._conn.commit()


@lru_cache(maxsize=1)
def hash_content(content: str) -> str:
    import hashlib

    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def to_symbols_json(symbols: Sequence[SymbolRecord]) -> str:
    return json.dumps(
        [{"name": s.name, "qn": s.qualified_name, "k": s.kind, "ls": s.line_start, "le": s.line_end} for s in symbols],
    )