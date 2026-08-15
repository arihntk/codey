"""Utilities for attaching concrete code evidence to agent findings.

LLMs sometimes omit or hallucinate evidence.  These helpers extract
verbatim code snippets from the diff or full file source so every finding
is grounded in actual repository content — proving the agent did not
hallucinate.

Evidence is only attached when the finding's reported location actually
matches a diff chunk / source line. A hallucinated path or line number
NEVER gets plausible-looking but wrong evidence bolted on.
"""

from __future__ import annotations

from codey.agents.context import DiffChunk, ReviewContext
from codey.agents.schemas import Finding

__all__ = ["attach_evidence"]


def attach_evidence(findings: list[Finding], ctx: ReviewContext) -> None:
    """Mutate *findings* in place: fill in missing ``evidence`` from the diff.

    For each finding that has a ``file_path`` and ``line_start`` but an empty
    ``evidence`` string, locate the enclosing diff chunk and copy a relevant
    snippet of the diff text as evidence.  If no chunk matches, try a
    window of lines from the full file source.

    Evidence is attached ONLY when the finding's file/line matches real
    content — findings pointing at files or lines that don't exist are left
    with empty evidence so they can be discarded downstream.
    """
    if not findings:
        return

    # Index diff chunks by file for fast lookup.
    chunks_by_file: dict[str, list[DiffChunk]] = {}
    for chunk in ctx.diff_chunks:
        chunks_by_file.setdefault(chunk.file_path, []).append(chunk)

    file_sources: dict[str, str] = ctx.file_sources

    for f in findings:
        if f.evidence and f.evidence.strip():
            continue
        if not f.file_path:
            continue

        snippet = _extract_from_chunks(f, chunks_by_file)
        if not snippet:
            snippet = _extract_from_source(f, file_sources)
        if snippet:
            f.evidence = snippet.strip()


def _extract_from_chunks(
    f: Finding,
    chunks_by_file: dict[str, list[DiffChunk]],
) -> str | None:
    """Find the diff chunk enclosing the finding's line and return its text.

    Only a chunk that actually contains ``line_start`` (or a finding with no
    line info at all) yields evidence. Chunks merely in the same file do NOT
    qualify — attaching them would let a hallucinated line claim real code.
    """
    chunks = chunks_by_file.get(f.file_path)
    if not chunks:
        return None
    for chunk in chunks:
        if f.line_start is None:
            return chunk.diff_text[:1000]
        if chunk.line_start <= f.line_start <= chunk.line_end:
            return _window(chunk.diff_text, f.line_start, chunk.line_start)
    return None


def _extract_from_source(
    f: Finding,
    file_sources: dict[str, str],
) -> str | None:
    """Extract a code window from the full file source, only when the
    finding's line actually exists in that file."""
    source = file_sources.get(f.file_path)
    if not source:
        return None
    lines = source.splitlines()
    if f.line_start is None:
        return source[:600]
    if f.line_start < 1 or f.line_start > len(lines):
        return None
    # Window centered on the finding: 3 lines above, 4 below. lines is
    # 0-indexed, line_start is 1-based — the first shown line (1-based) is
    # line_start - 3, i.e. index line_start - 1 - 3.
    start = max(0, f.line_start - 1 - 3)
    end = min(len(lines), (f.line_end or f.line_start or 1) + 3)
    window = lines[start:end]
    prefix = f"  L{start + 1}…" if start > 0 else ""
    return prefix + "\n".join(f"  {i + start + 1:4d} | {line}" for i, line in enumerate(window))


def _window(diff_text: str, target_line: int, chunk_start: int) -> str:
    """Return a focused window of diff lines around *target_line*."""
    lines = diff_text.splitlines()
    if not lines:
        return diff_text
    # Heuristic: return lines within 5 of the target offset within the chunk.
    offset = max(0, target_line - chunk_start)
    lo = max(0, offset - 5)
    hi = min(len(lines), offset + 6)
    return "\n".join(lines[lo:hi])
