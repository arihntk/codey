-- Codey AST/symbol metadata cache schema
-- Global DB at ~/.cache/codey/codey.db, keyed by absolute repo path

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- One row per (repo, git_hash) indexing run.
CREATE TABLE IF NOT EXISTS index_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_path       TEXT    NOT NULL,
    git_hash        TEXT    NOT NULL,
    indexed_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    file_count      INTEGER NOT NULL DEFAULT 0,
    UNIQUE (repo_path, git_hash)
);

-- One row per parsed file in a given index run.
CREATE TABLE IF NOT EXISTS file_entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_path       TEXT    NOT NULL,
    git_hash        TEXT    NOT NULL,
    rel_path        TEXT    NOT NULL,
    language        TEXT    NOT NULL,
    content_hash    TEXT    NOT NULL,
    mtime           REAL    NOT NULL DEFAULT 0,
    byte_count      INTEGER NOT NULL DEFAULT 0,
    UNIQUE (repo_path, git_hash, rel_path),
    FOREIGN KEY (repo_path, git_hash) REFERENCES index_runs (repo_path, git_hash) ON DELETE CASCADE
);

-- Symbol/call-graph edges for the Python reverse-dependency lookup.
CREATE TABLE IF NOT EXISTS symbols (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_path       TEXT    NOT NULL,
    git_hash        TEXT    NOT NULL,
    rel_path        TEXT    NOT NULL,
    name            TEXT    NOT NULL,
    qualified_name  TEXT    NOT NULL,
    kind            TEXT    NOT NULL,  -- function, class, method, import, etc.
    line_start      INTEGER NOT NULL,
    line_end        INTEGER NOT NULL,
    UNIQUE (repo_path, git_hash, rel_path, qualified_name),
    FOREIGN KEY (repo_path, git_hash) REFERENCES index_runs (repo_path, git_hash) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS call_edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_path       TEXT    NOT NULL,
    git_hash        TEXT    NOT NULL,
    caller_path     TEXT    NOT NULL,
    caller_qname    TEXT    NOT NULL,
    callee_name     TEXT    NOT NULL,
    callee_path     TEXT,             -- resolved target path, if known
    callee_qname    TEXT,             -- resolved target qual name, if known
    line            INTEGER NOT NULL,
    FOREIGN KEY (repo_path, git_hash) REFERENCES index_runs (repo_path, git_hash) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS import_edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_path       TEXT    NOT NULL,
    git_hash        TEXT    NOT NULL,
    rel_path        TEXT    NOT NULL,
    module          TEXT    NOT NULL,          -- imported module string
    imported_name   TEXT,                     -- specific name imported, or NULL for module
    alias           TEXT,                     -- local alias
    line            INTEGER NOT NULL,
    FOREIGN KEY (repo_path, git_hash) REFERENCES index_runs (repo_path, git_hash) ON DELETE CASCADE
);

-- Key-value metadata (last indexed hash per repo, misc settings).
CREATE TABLE IF NOT EXISTS meta (
    key     TEXT PRIMARY KEY,
    value   TEXT
);

CREATE INDEX IF NOT EXISTS idx_file_entries_path   ON file_entries (repo_path, rel_path);
CREATE INDEX IF NOT EXISTS idx_symbols_name        ON symbols (repo_path, git_hash, name);
CREATE INDEX IF NOT EXISTS idx_symbols_qname        ON symbols (repo_path, git_hash, qualified_name);
CREATE INDEX IF NOT EXISTS idx_call_edges_callee    ON call_edges (repo_path, git_hash, callee_name);
CREATE INDEX IF NOT EXISTS idx_import_edges_module ON import_edges (repo_path, git_hash, module);