"""AST-aware diff chunking — break a diff into function/class-level chunks."""

from __future__ import annotations

import re
from pathlib import Path

from codey.agents.context import DiffChunk
from codey.cache.ast_cache import CacheDB, SymbolRecord
from codey.index.languages import detect_language

__all__ = ["chunk_diff", "chunk_file_diff"]

_HUNK_HEADER = re.compile(r"^@{2,3} (?:-\d+(?:,\d+)? )+?\+(\d+)(?:,(\d+))? @{2,3}")


def chunk_file_diff(
    file_path: str,
    diff_text: str,
    *,
    full_source: str = "",
    symbols: list[SymbolRecord] | None = None,
) -> list[DiffChunk]:
    language = detect_language(Path(file_path)) or "text"
    hunks: list[tuple[int, list[str]]] = _split_hunks(diff_text)

    if not hunks:
        return [DiffChunk(file_path, language, "<module>", "module", diff_text, 0, 0, full_source)]

    chunks: list[DiffChunk] = []
    for new_start, hunk_lines in hunks:
        hunk_len = _hunk_new_len(hunk_lines)
        new_end = new_start + max(0, hunk_len - 1)
        symbol, kind = _enclosing_symbol(new_start, new_end, symbols or [])
        chunks.append(DiffChunk(file_path, language, symbol, kind, "".join(hunk_lines),
                                new_start, new_end, full_source))

    return _merge_adjacent(chunks)


def chunk_diff(
    diffs: dict[str, str],
    *,
    db: CacheDB | None = None,
    repo_path: str = "",
    git_hash: str = "",
    file_sources: dict[str, str] | None = None,
) -> list[DiffChunk]:
    file_sources = file_sources or {}
    all_chunks: list[DiffChunk] = []
    for path, text in diffs.items():
        symbols = None
        source = file_sources.get(path, "")
        if db and repo_path and git_hash:
            symbols = db.symbols_in_file(repo_path, git_hash, path)
        all_chunks.extend(chunk_file_diff(path, text, full_source=source, symbols=symbols))
    return all_chunks


def _split_hunks(diff_text: str) -> list[tuple[int, list[str]]]:
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
    count = 0
    for line in hunk_lines:
        if line.startswith("@@") or line.startswith("---") or line.startswith("+++") or line.startswith("-"):
            continue
        count += 1
    return count


def _enclosing_symbol(line_start: int, line_end: int, symbols: list[SymbolRecord]) -> tuple[str, str]:
    best: SymbolRecord | None = None
    for s in symbols:
        if s.line_start <= line_start and line_end <= s.line_end:
            if best is None or s.line_start > best.line_start:
                best = s
        elif line_start <= s.line_end and line_end >= s.line_start:
            if best is None or s.line_start > best.line_start:
                best = s
    if best:
        return best.qualified_name, best.kind
    return "<module>", "module"


def _merge_adjacent(chunks: list[DiffChunk]) -> list[DiffChunk]:
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
                last.file_path, last.language, last.symbol, last.symbol_kind,
                last.diff_text + "\n" + c.diff_text,
                min(last.line_start, c.line_start), max(last.line_end, c.line_end),
                last.full_source or c.full_source,
            )
        else:
            merged.append(c)
    return merged
