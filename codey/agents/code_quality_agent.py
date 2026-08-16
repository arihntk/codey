"""CodeQualityAgent — benchmarks the commit against existing codebase patterns."""

from __future__ import annotations

from codey.agents.context import ReviewContext
from codey.agents.evidence import attach_evidence
from codey.agents.schemas import (
    AgentReport,
    Finding,
    FindingCategory,
    Severity,
    parse_llm_findings,
)
from codey.agents.schemas import (
    llm_output_is_parseable as _llm_output_is_parseable,
)
from codey.llm.response import extract_text, response_tokens
from codey.llm.retry import invoke_with_retry

__all__ = ["run_code_quality_agent"]

_QUALITY_SYSTEM = (
    "You are a senior code quality reviewer. Given a diff and the codebase's "
    "architecture summary, evaluate the change against existing conventions:\n"
    "1. Naming consistency\n2. Type hint coverage and style\n"
    "3. Docstring/comment coverage\n4. Error handling patterns\n"
    "5. Duplication or unnecessary complexity\n6. Architectural alignment\n\n"
    "For each issue output a JSON object with: severity, title, description, "
    "file_path, line_start, line_end, evidence, recommendation, confidence.\n\n"
    "CRITICAL: 'evidence' MUST be a VERBATIM copy of the specific code line(s) "
    "from the diff. Do not fabricate or paraphrase — findings without verbatim "
    "evidence will be discarded. Use category 'code_quality'. If the code meets "
    "benchmarks, output an empty array []. Output only a JSON array."
)


def _parse_llm_findings(text: str) -> tuple[list[Finding], str | None]:
    return parse_llm_findings(text, FindingCategory.CODE_QUALITY)


def run_code_quality_agent(ctx: ReviewContext, db=None, llm: object | None = None) -> AgentReport:
    findings: list[Finding] = []
    token_usage = 0

    if not ctx.diff_chunks and not ctx.full_diff:
        return AgentReport(agent="code_quality", status="skipped",
                           summary="No diff provided for quality analysis.", findings=[])
    if llm is None:
        return AgentReport(agent="code_quality", status="skipped",
                           summary="No LLM configured for quality analysis.", findings=[])

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
            report_error = parse_error
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

    # Enforce the verbatim rule: LLM findings without evidence are discarded.
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


def _build_diff_context(ctx: ReviewContext, *, max_chars: int = 16_000) -> str:
    parts: list[str] = []
    total = 0
    for chunk in ctx.diff_chunks:
        entry = (
            f"### {chunk.file_path} :: {chunk.symbol_kind} {chunk.symbol} "
            f"(L{chunk.line_start}-{chunk.line_end})\n```diff\n{chunk.diff_text}\n```\n"
        )
        if total + len(entry) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                parts.append(entry[:remaining] + "\n... [truncated]\n")
            break
        parts.append(entry)
        total += len(entry)

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
