"""Attach verbatim code evidence to findings, only when the location is real."""

from __future__ import annotations

from codey.agents.context import DiffChunk, ReviewContext
from codey.agents.schemas import Finding

__all__ = ["attach_evidence"]


def attach_evidence(findings: list[Finding], ctx: ReviewContext) -> None:
    """Fill missing ``evidence`` from the diff (or full source) for each finding.

    Evidence is attached ONLY when the finding's file/line matches real content —
    hallucinated paths/lines are left with empty evidence so they can be discarded.
    """
    if not findings:
        return
    chunks_by_file: dict[str, list[DiffChunk]] = {}
    for chunk in ctx.diff_chunks:
        chunks_by_file.setdefault(chunk.file_path, []).append(chunk)

    for f in findings:
        if f.evidence.strip() or not f.file_path:
            continue
        snippet = _extract_from_chunks(f, chunks_by_file) or _extract_from_source(f, ctx.file_sources)
        if snippet:
            f.evidence = snippet.strip()


def _extract_from_chunks(f: Finding, chunks_by_file: dict[str, list[DiffChunk]]) -> str | None:
    chunks = chunks_by_file.get(f.file_path)
    if not chunks:
        return None
    for chunk in chunks:
        if f.line_start is None:
            return chunk.diff_text[:1000]
        if chunk.line_start <= f.line_start <= chunk.line_end:
            return _window(chunk.diff_text, f.line_start, chunk.line_start)
    return None


def _extract_from_source(f: Finding, file_sources: dict[str, str]) -> str | None:
    source = file_sources.get(f.file_path)
    if not source:
        return None
    lines = source.splitlines()
    if f.line_start is None:
        return source[:600]
    if f.line_start < 1 or f.line_start > len(lines):
        return None
    start = max(0, f.line_start - 1 - 3)
    end = min(len(lines), (f.line_end or f.line_start or 1) + 3)
    window = lines[start:end]
    prefix = f"  L{start + 1}…" if start > 0 else ""
    return prefix + "\n".join(f"  {i + start + 1:4d} | {line}" for i, line in enumerate(window))


def _window(diff_text: str, target_line: int, chunk_start: int) -> str:
    lines = diff_text.splitlines()
    if not lines:
        return diff_text
    offset = max(0, target_line - chunk_start)
    lo = max(0, offset - 5)
    hi = min(len(lines), offset + 6)
    return "\n".join(lines[lo:hi])
