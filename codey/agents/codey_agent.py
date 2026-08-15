"""CodeyAgent (orchestrator) — synthesises the final review.

After all worker agents have run in parallel, the orchestrator collects their
AgentReports, asks the LLM to write an executive summary, picks the overall
severity, and decides a recommendation (approve / request_changes / block).
"""

from __future__ import annotations

import json
from typing import Any

from codey.agents.context import ReviewContext
from codey.agents.schemas import (
    AgentReport,
    Finding,
    ReviewSummary,
    Severity,
    aggregate_severity,
)
from codey.llm.response import extract_text, response_tokens
from codey.llm.retry import invoke_with_retry

__all__ = ["run_codey_agent"]

_SYNTHESIS_SYSTEM = (
    "You are Codey, the lead code reviewer. You have been given structured reports "
    "from specialist agents (security, code quality, testing, indexing). Synthesise "
    "them into a final review. Your output must be a JSON object with:\n"
    '  "overall_severity": "critical|high|medium|low|info"\n'
    '  "summary": "2-3 paragraph executive summary in markdown"\n'
    '  "recommendation": "approve|request_changes|block"\n\n'
    "CRITICAL: Your summary MUST reference concrete evidence from the agent "
    "reports — cite specific findings, tool outputs, file paths, and test "
    "results to prove the review is grounded in actual analysis. Do not make "
    "claims that are not supported by the provided agent reports. "
    "Consider the severity and confidence of each finding. Don't repeat findings; "
    "instead provide an actionable executive assessment."
)


def run_codey_agent(
    ctx: ReviewContext,
    agent_reports: dict[str, AgentReport],
    primary_llm: object | None,
    *,
    prior_errors: list[str] | None = None,
) -> ReviewSummary:
    """Synthesise all agent reports into a final ReviewSummary.

    ``prior_errors`` (from the graph state) are propagated into the review's
    ``errors`` field along with any errors surfaced by the agents themselves,
    so a failing LLM call is never silently presented as a clean review.

    The verdict is never allowed to be optimistic: if any agent errored, the
    recommendation degrades to at most ``request_changes`` (an incomplete
    review cannot approve).
    """
    all_findings: list[Finding] = []
    for r in agent_reports.values():
        all_findings.extend(r.findings)

    overall = aggregate_severity(all_findings)

    # Collect structured errors: prior graph errors, agent-level errors, and
    # orchestrator synthesis errors — never silently dropped.
    errors: list[str] = list(prior_errors or [])
    for r in agent_reports.values():
        if r.status == "error" and r.error:
            errors.append(f"[{r.agent}] {r.error}")

    rec = _default_recommendation(all_findings, errors=errors)
    summary_text = _build_text_summary(ctx, agent_reports, overall, rec)

    synth_error: str | None = None
    if primary_llm is not None:
        llm_summary, llm_sev, llm_rec, _used, synth_error = _llm_synthesise(primary_llm, agent_reports, ctx)
        if llm_summary:
            summary_text = llm_summary
        # Trust the LLM's severity/recommendation when it produced valid
        # values AND the review is complete — the model sees the full context
        # a max() over severities cannot. Still degrade on errors.
        if synth_error is None:
            try:
                overall = Severity(llm_sev.lower())
            except (ValueError, AttributeError):
                pass
            if llm_rec in ("approve", "request_changes", "block"):
                rec = llm_rec
            if errors:
                rec = _degrade_recommendation(rec)
    if synth_error is not None:
        errors.append(f"[codey] summary synthesis failed: {synth_error}")

    review = ReviewSummary(
        overall_severity=overall,
        summary=summary_text,
        commit_hash=ctx.git_hash,
        commit_message=ctx.commit_message,
        files_reviewed=ctx.changed_files.copy(),
        dependent_files_checked=ctx.dependent_files.copy(),
        agent_reports=agent_reports,
        total_findings=len(all_findings),
        recommendation=rec,
        errors=errors,
        pruned_chunks=list(ctx.pruned_chunks),
    )
    return review


def _build_text_summary(
    ctx: ReviewContext,
    reports: dict[str, AgentReport],
    overall: Severity,
    rec: str,
) -> str:
    """Build a markdown summary without LLM if synthesis fails."""
    lines: list[str] = []
    lines.append(f"## Codey Review — {ctx.git_hash[:12] if ctx.git_hash else 'HEAD'}")
    lines.append("")
    lines.append(f"**Commit:** {ctx.commit_message[:200]}")
    lines.append(f"**Verdict:** {rec.replace('_', ' ').title()} ({overall.value})")
    lines.append(f"**Findings:** {sum(len(r.findings) for r in reports.values())} across {len(reports)} agent reports")
    if ctx.pruned_chunks:
        lines.append(
            f"**Coverage:** {len(ctx.pruned_chunks)} diff chunk(s) were pruned "
            f"to stay within the context budget: "
            f"{', '.join(ctx.pruned_chunks[:10])}"
            + (" …" if len(ctx.pruned_chunks) > 10 else "")
        )
    lines.append("")
    for name in ("index", "security", "code_quality", "test"):
        report = reports.get(name)
        if not report:
            continue
        lines.append(f"### {report.agent.title()} Agent ({report.status})")
        lines.append(f"> {report.summary}")
        if report.findings:
            for f in report.findings:
                if f.severity == Severity.INFO:
                    continue
                loc = f"{f.file_path}:{f.line_start}" if f.file_path else ""
                lines.append(f"- **[{f.severity.value.upper()}]** {f.title}" + (f" `{loc}`" if loc else ""))
                if f.recommendation:
                    lines.append(f"  - {f.recommendation}")
        lines.append("")
    lines.append("---")
    return "\n".join(lines)


def _default_recommendation(findings: list[Finding], *, errors: list[str] | None = None) -> str:
    """Determine recommendation based on highest-severity findings.

    An incomplete review can never approve: if any error occurred, the
    recommendation degrades to at most ``request_changes`` so a total LLM
    outage doesn't produce a green light.
    """
    base = "approve"
    if any(f.severity == Severity.CRITICAL for f in findings):
        base = "block"
    elif any(f.severity == Severity.HIGH and f.confidence >= 0.7 for f in findings):
        base = "request_changes"
    if errors:
        return _degrade_recommendation(base)
    return base


def _degrade_recommendation(rec: str) -> str:
    """Lower a recommendation to at most ``request_changes`` when the review
    is incomplete (errors occurred). Never approve an unverified review."""
    if rec == "approve":
        return "request_changes"
    return rec


def _llm_synthesise(
    llm: object,
    reports: dict[str, AgentReport],
    ctx: ReviewContext,
) -> tuple[str, str, str, int, str | None]:
    """Ask the LLM to write an executive summary + verdict.

    Returns ``(summary, severity, recommendation, token_usage, error)``.
    ``error`` is set when the LLM call itself failed; unparseable output
    falls back to the raw text and is reported as a parse error. The LLM's
    ``severity``/``recommendation`` are validated by the caller before use.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    # Serialize reports compactly.
    report_data: list[dict[str, Any]] = []
    for name, r in reports.items():
        report_data.append({
            "agent": r.agent,
            "status": r.status,
            "summary": r.summary,
            "findings": [
                {
                    "severity": f.severity.value,
                    "title": f.title,
                    "description": f.description[:500],
                    "file": f.file_path,
                    "line": f.line_start,
                    "evidence": f.evidence[:500] if f.evidence else "",
                    "confidence": f.confidence,
                    "recommendation": f.recommendation,
                }
                for f in r.findings
            ],
        })

    try:
        response = invoke_with_retry(llm, [
            SystemMessage(content=_SYNTHESIS_SYSTEM),
            HumanMessage(content=(
                f"Commit: {ctx.commit_message[:500]}\n"
                f"Hash: {ctx.git_hash}\n"
                f"Files: {', '.join(ctx.changed_files[:30])}\n\n"
                f"Agent reports:\n{json.dumps(report_data, indent=2)}"
            )),
        ])
        raw = extract_text(response)
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        usage = response_tokens(response, fallback_text=raw)
        try:
            obj = json.loads(text)
            summary = obj.get("summary", text)
            severity = str(obj.get("overall_severity", "")).lower()
            rec = str(obj.get("recommendation", "")).lower()
            return summary, severity, rec, usage, None
        except json.JSONDecodeError:
            # LLM output wasn't valid JSON — surface it instead of silently
            # using raw text as the summary.
            return text, "", "", usage, "LLM output was not valid JSON"
    except Exception as e:
        return "", "", "", 0, str(e)
