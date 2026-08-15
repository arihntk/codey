"""CodeQualityAgent — benchmarks the commit against existing codebase patterns.

Uses the index summary (architecture/design conventions from IndexAgent) plus
the diff chunks to ask the LLM whether the new code follows the established
patterns: naming, typing, docstring coverage, error handling, structure, etc.
"""

from __future__ import annotations

import json

from codey.agents.context import ReviewContext
from codey.agents.evidence import attach_evidence
from codey.agents.schemas import AgentReport, Finding, FindingCategory, Severity
from codey.llm.response import extract_text, response_tokens
from codey.llm.retry import invoke_with_retry

__all__ = ["run_code_quality_agent"]

_QUALITY_SYSTEM = (
    "You are a senior code quality reviewer. Given a diff and the codebase's "
    "architecture summary, evaluate the change against the existing conventions:\n"
    "1. Naming consistency (functions, classes, variables)\n"
    "2. Type hint coverage and style\n"
    "3. Docstring/comment coverage\n"
    "4. Error handling patterns\n"
    "5. Code duplication or unnecessary complexity\n"
    "6. Architectural alignment (does it follow the module structure?)\n\n"
    "For each issue, output a JSON object with:\n"
    "  severity, title, description, file_path, line_start, line_end, evidence,\n"
    "  recommendation, confidence\n\n"
    "CRITICAL: The 'evidence' field MUST contain a VERBATIM copy of the specific "
    "code line(s) from the diff that demonstrates the issue. Do not fabricate "
    "or paraphrase evidence — quote the actual code. Findings without verbatim "
    "evidence will be discarded.\n"
    "Use category 'code_quality'. If the code meets benchmarks, output an empty "
    "array []. Output only a JSON array."
)


def run_code_quality_agent(
    ctx: ReviewContext,
    db=None,
    llm: object | None = None,
) -> AgentReport:
    """Assess code quality of the diff against codebase benchmarks."""
    findings: list[Finding] = []
    token_usage = 0

    if not ctx.diff_chunks and not ctx.full_diff:
        return AgentReport(
            agent="code_quality",
            status="skipped",
            summary="No diff provided for quality analysis.",
            findings=[],
        )

    if llm is None:
        return AgentReport(
            agent="code_quality",
            status="skipped",
            summary="No LLM configured for quality analysis.",
            findings=[],
        )

    # Build context: architecture summary + diff chunks + dependent file snippets.
    diff_text = _build_diff_context(ctx)
    arch_context = ctx.index_summary or "(no architecture summary available)"

    report_error: str | None = None
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = invoke_with_retry(llm, [
            SystemMessage(content=_QUALITY_SYSTEM),
            HumanMessage(content=(
                f"## Codebase conventions\n{arch_context}\n\n"
                f"## Changes to review\n{diff_text}"
            )),
        ])
        raw = extract_text(response)
        token_usage = response_tokens(response, fallback_text=raw)
        findings, parse_error = _parse_llm_findings(raw)
        if parse_error:
            report_error = (report_error + "; " if report_error else "") + parse_error
        elif not _llm_output_is_parseable(raw):
            report_error = "LLM returned unparseable output (expected a JSON array of findings)"
    except Exception as e:
        report_error = str(e)

    if not findings and report_error is None:
        findings.append(Finding(
            category=FindingCategory.CODE_QUALITY,
            severity=Severity.INFO,
            title="Code meets quality benchmarks",
            description=(
                f"No quality issues detected. Reviewed {len(ctx.diff_chunks)} diff chunk(s) "
                f"across {len(ctx.changed_files)} file(s) against architecture conventions. "
                f"Checked {len(ctx.dependent_files)} dependent file(s)."
            ),
            evidence=(diff_text[:500] if diff_text else "No diff content to attach as evidence."),
            confidence=0.7,
        ))

    # Attach verbatim diff evidence to any LLM findings that lack it, then
    # ENFORCE the verbatim rule: LLM findings that still have no evidence are
    # discarded (the system prompt promises this). The synthetic INFO finding
    # (if any) always has evidence attached above.
    attach_evidence(findings, ctx)
    findings = [f for f in findings if f.evidence.strip()]

    finding_details = "; ".join(
        f"{f.title} [{f.severity.value}]"
        + (f" at {f.file_path}:{f.line_start}" if f.file_path else "")
        for f in findings
        if f.severity != Severity.INFO
    )
    summary = (
        f"Quality analysis of {len(ctx.changed_files)} file(s) "
        f"({len(ctx.diff_chunks)} diff chunks, {len(ctx.dependent_files)} dependent files). "
        f"Found {len(findings)} finding(s)."
    )
    if finding_details:
        summary += f" Issues: {finding_details}."
    if report_error is not None:
        summary += f" Warning: LLM analysis failed ({report_error})."

    return AgentReport(
        agent="code_quality",
        status="error" if report_error is not None else "completed",
        summary=summary,
        findings=findings,
        metadata={
            "diff_chunks": str(len(ctx.diff_chunks)),
            "dependent_files": str(len(ctx.dependent_files)),
        },
        token_usage=token_usage,
        error=report_error,
    )


def _llm_output_is_parseable(text: str) -> bool:
    """True when *text* parses as a JSON array (possibly empty)."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        stripped = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        json.loads(stripped)
        return True
    except json.JSONDecodeError:
        return False


def _build_diff_context(ctx: ReviewContext, *, max_chars: int = 16_000) -> str:
    """Build a compact text representation of the diff chunks for the LLM."""
    parts: list[str] = []
    total = 0
    for chunk in ctx.diff_chunks:
        header = (
            f"### {chunk.file_path} :: {chunk.symbol_kind} {chunk.symbol} "
            f"(L{chunk.line_start}-{chunk.line_end})"
        )
        body = chunk.diff_text
        entry = f"{header}\n```diff\n{body}\n```\n"
        if total + len(entry) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                entry = entry[:remaining] + "\n... [truncated]\n"
                parts.append(entry)
            break
        parts.append(entry)
        total += len(entry)

    # Include dependent file snippets if there's room.
    dep_budget = max_chars - total
    if dep_budget > 1000 and ctx.dependent_files:
        parts.append("\n## Dependent files (affected but not changed):\n")
        for dep in ctx.dependent_files[:10]:
            source = ctx.file_sources.get(dep, "")
            if source:
                snippet = source[:800]
                parts.append(f"### {dep}\n```python\n{snippet}\n```\n")
                dep_budget -= len(snippet) + 200
                if dep_budget < 200:
                    break

    return "\n".join(parts) if parts else ctx.full_diff[:max_chars]


def _parse_llm_findings(text: str) -> tuple[list[Finding], str | None]:
    """Parse LLM JSON output into Finding objects, defensively.

    Returns ``(findings, error)`` — malformed items (e.g. ``"severity":
    null``) are skipped rather than crashing the whole parse, and the count
    is returned as an error note. Unparseable top-level JSON yields
    ``([], None)``; the caller checks parseability separately.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [], None
    if not isinstance(data, list):
        return [], "LLM output was not a JSON array"
    findings: list[Finding] = []
    malformed = 0
    for item in data:
        try:
            sev_raw = item.get("severity", "info")
            if sev_raw is None:
                raise ValueError("severity is null")
            sev = Severity(str(sev_raw).lower())
            conf_raw = item.get("confidence", 0.7)
            conf = float(conf_raw) if conf_raw is not None else 0.7
            findings.append(Finding(
                category=FindingCategory.CODE_QUALITY,
                severity=sev,
                title=item.get("title", ""),
                description=item.get("description", ""),
                file_path=item.get("file_path"),
                line_start=item.get("line_start"),
                line_end=item.get("line_end"),
                evidence=item.get("evidence", ""),
                recommendation=item.get("recommendation", ""),
                confidence=conf,
            ))
        except (ValueError, TypeError):
            malformed += 1
    error = f"{malformed} malformed LLM finding(s) skipped" if malformed else None
    return findings, error
