"""AST-aware diff chunking — breaks a diff into function/class-level chunks.

Uses the cached symbol table to map diff hunks to their enclosing symbols
(function/class/method). A single diff with multiple hunks spanning different
functions becomes multiple DiffChunk objects, each tagged with the symbol it
modifies.
"""

from __future__ import annotations

import re
from pathlib import Path

from codey.agents.context import DiffChunk
from codey.cache.ast_cache import CacheDB, SymbolRecord
from codey.index.languages import detect_language

__all__ = ["chunk_diff", "chunk_file_diff"]

_HUNK_HEADER = re.compile(
    r"^@{2,3} (?:-\d+(?:,\d+)? )+?\+(\d+)(?:,(\d+))? @{2,3}"
)


def chunk_file_diff(
    file_path: str,
    diff_text: str,
    *,
    full_source: str = "",
    symbols: list[SymbolRecord] | None = None,
) -> list[DiffChunk]:
    """Chunk a single file's diff into granular DiffChunk objects by symbol.

    Args:
        file_path: relative path to the file.
        diff_text: raw diff text for this file.
        full_source: full file source (optional, for context).
        symbols: symbol records covering this file, if available.
    """
    language = detect_language(Path(file_path)) or "text"

    # Split into hunks via @@ headers.
    hunks: list[tuple[int, list[str]]] = _split_hunks(diff_text)

    if not hunks:
        return [DiffChunk(
            file_path=file_path,
            language=language,
            symbol="<module>",
            symbol_kind="module",
            diff_text=diff_text,
            line_start=0, line_end=0,
            full_source=full_source,
        )]

    chunks: list[DiffChunk] = []
    for new_start, hunk_lines in hunks:
        # Determine hunk extent (how many new lines this hunk covers).
        hunk_len = _hunk_new_len(hunk_lines)
        new_end = new_start + max(0, hunk_len - 1)

        # Map to enclosing symbol.
        symbol, kind = _enclosing_symbol(new_start, new_end, symbols or [])

        chunks.append(DiffChunk(
            file_path=file_path,
            language=language,
            symbol=symbol,
            symbol_kind=kind,
            diff_text="".join(hunk_lines),
            line_start=new_start,
            line_end=new_end,
            full_source=full_source,
        ))

    # Merge adjacent hunks for the same symbol to avoid redundancy.
    return _merge_adjacent(chunks)


def chunk_diff(
    diffs: dict[str, str],
    *,
    db: CacheDB | None = None,
    repo_path: str = "",
    git_hash: str = "",
    file_sources: dict[str, str] | None = None,
) -> list[DiffChunk]:
    """Chunk a full diff mapping {path -> diff_text} into DiffChunk objects."""
    file_sources = file_sources or {}
    all_chunks: list[DiffChunk] = []
    for path, text in diffs.items():
        symbols = None
        source = file_sources.get(path, "")
        if db and repo_path and git_hash:
            symbols = db.symbols_in_file(repo_path, git_hash, path)
        chunks = chunk_file_diff(path, text, full_source=source, symbols=symbols)
        all_chunks.extend(chunks)
    return all_chunks


# --- Helpers ----


def _split_hunks(diff_text: str) -> list[tuple[int, list[str]]]:
    """Split a file diff into (new_start_line, hunk_lines) pairs."""
    lines = diff_text.splitlines(keepends=True)
    hunks: list[tuple[int, list[str]]] = []
    current_start: int | None = None
    current_lines: list[str] = []

    for line in lines:
        m = _HUNK_HEADER.match(line)
        if m:
            if current_start is not None:
                hunks.append((current_start, current_lines))
            current_start = int(m.group(1))
            current_lines = [line]
        elif current_start is not None:
            current_lines.append(line)

    if current_start is not None:
        hunks.append((current_start, current_lines))

    return hunks


def _hunk_new_len(hunk_lines: list[str]) -> int:
    """Count the number of lines in the new file this hunk covers."""
    count = 0
    for line in hunk_lines:
        if line.startswith("@@") or line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("-"):
            continue
        # Lines that start with '+' or ' ' (context) count toward new file.
        count += 1
    return count


def _enclosing_symbol(
    line_start: int,
    line_end: int,
    symbols: list[SymbolRecord],
) -> tuple[str, str]:
    """Find the innermost symbol enclosing the line range [line_start, line_end]."""
    best: SymbolRecord | None = None
    for s in symbols:
        # Hunk is inside symbol if line_start >= s.line_start and line_end <= s.line_end.
        if s.line_start <= line_start and line_end <= s.line_end:
            if best is None or s.line_start > best.line_start:
                best = s
        # Partial overlap: hunk starts before symbol ends and ends at/after symbol start.
        elif line_start <= s.line_end and line_end >= s.line_start:
            if best is None or s.line_start > best.line_start:
                best = s
    if best:
        return best.qualified_name, best.kind
    return "<module>", "module"


def _merge_adjacent(chunks: list[DiffChunk]) -> list[DiffChunk]:
    """Merge chunks for the same file + symbol that are adjacent."""
    if len(chunks) <= 1:
        return chunks
    merged: list[DiffChunk] = [chunks[0]]
    for c in chunks[1:]:
        last = merged[-1]
        if (
            c.file_path == last.file_path
            and c.symbol == last.symbol
            and c.symbol_kind == last.symbol_kind
            and c.line_start <= last.line_end + 5
        ):
            merged[-1] = DiffChunk(
                file_path=last.file_path,
                language=last.language,
                symbol=last.symbol,
                symbol_kind=last.symbol_kind,
                diff_text=last.diff_text + "\n" + c.diff_text,
                line_start=min(last.line_start, c.line_start),
                line_end=max(last.line_end, c.line_end),
                full_source=last.full_source or c.full_source,
            )
        else:
            merged.append(c)
    return merged
