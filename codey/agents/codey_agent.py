"""CodeyAgent (orchestrator) — synthesises the final review."""

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
    severity_weight,
)
from codey.graph.registry import ordered_agent_names
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
    "claims not supported by the provided agent reports. Consider the severity "
    "and confidence of each finding. Don't repeat findings; provide an actionable "
    "executive assessment."
)

_REC_RANK = {"approve": 0, "request_changes": 1, "block": 2}

# Hard ceiling on findings kept from any single agent report. Whatever floods
# (a per-line lint rule, a noisy tool, a verbose LLM), the review stays focused.
_MAX_FINDINGS_PER_AGENT = 100

_AGENT_PRIORITY = {"security": 0, "code_quality": 1, "test": 2, "index": 3, "codey": 4}


def _rec_rank(rec: str) -> int:
    return _REC_RANK.get(rec, 0)


def _degrade_recommendation(rec: str) -> str:
    """Never approve an unverified review — degrade 'approve' to 'request_changes'."""
    return "request_changes" if rec == "approve" else rec


def _default_recommendation(findings: list[Finding], *, errors: list[str] | None = None) -> str:
    base = "approve"
    if any(f.severity == Severity.CRITICAL for f in findings):
        base = "block"
    elif any(f.severity == Severity.HIGH and f.confidence >= 0.7 for f in findings):
        base = "request_changes"
    return _degrade_recommendation(base) if errors else base


def _throttle_findings(report: AgentReport) -> AgentReport:
    """Cap a report's findings, keeping the highest-severity ones, and note the suppression."""
    if len(report.findings) <= _MAX_FINDINGS_PER_AGENT:
        return report
    ranked = sorted(report.findings, key=lambda f: (-severity_weight(f.severity), -f.confidence))
    kept = ranked[:_MAX_FINDINGS_PER_AGENT]
    suppressed = len(report.findings) - len(kept)
    note = f"{suppressed} low-priority finding(s) suppressed (cap {_MAX_FINDINGS_PER_AGENT})."
    summary = f"{report.summary} {note}" if report.summary else note
    return AgentReport(
        agent=report.agent, status=report.status, summary=summary,
        findings=kept, metadata=report.metadata, token_usage=report.token_usage,
        error=report.error,
    )


def _dedupe_across_reports(reports: dict[str, AgentReport]) -> dict[str, AgentReport]:
    """Collapse duplicate findings across agents by (category, file, line)."""
    seen: set[tuple[str, str, int]] = set()
    deduped: dict[str, AgentReport] = {}
    for name in sorted(reports, key=lambda n: _AGENT_PRIORITY.get(n, 99)):
        report = reports[name]
        kept = []
        for f in report.findings:
            if not f.file_path or f.line_start is None:
                kept.append(f)
                continue
            key = (f.category.value, f.file_path.lstrip("./"), f.line_start)
            if key in seen:
                continue
            seen.add(key)
            kept.append(f)
        if len(kept) == len(report.findings):
            deduped[name] = report
            continue
        deduped[name] = AgentReport(
            agent=report.agent, status=report.status, summary=report.summary,
            findings=kept, metadata=report.metadata, token_usage=report.token_usage,
            error=report.error,
        )
    return deduped


def run_codey_agent(
    ctx: ReviewContext,
    agent_reports: dict[str, AgentReport],
    primary_llm: object | None,
    *,
    prior_errors: list[str] | None = None,
) -> ReviewSummary:
    agent_reports = _dedupe_across_reports(agent_reports)
    agent_reports = {name: _throttle_findings(r) for name, r in agent_reports.items()}
    all_findings: list[Finding] = [f for r in agent_reports.values() for f in r.findings]
    overall = aggregate_severity(all_findings)

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
        # The LLM verdict can refine the deterministic one but NEVER make it
        # more optimistic (severity = max, recommendation = the stricter).
        if synth_error is None:
            try:
                llm_sev_parsed = Severity(llm_sev.lower())
                if severity_weight(llm_sev_parsed) > severity_weight(overall):
                    overall = llm_sev_parsed
            except (ValueError, AttributeError, TypeError):
                pass
            if llm_rec in ("approve", "request_changes", "block") and _rec_rank(llm_rec) > _rec_rank(rec):
                rec = llm_rec
            if errors:
                rec = _degrade_recommendation(rec)
    if synth_error is not None:
        errors.append(f"[codey] summary synthesis failed: {synth_error}")

    return ReviewSummary(
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


def _build_text_summary(ctx: ReviewContext, reports: dict[str, AgentReport], overall: Severity, rec: str) -> str:
    lines = [
        f"## Codey Review — {ctx.git_hash[:12] if ctx.git_hash else 'HEAD'}",
        "",
        f"**Commit:** {ctx.commit_message[:200]}",
        f"**Verdict:** {rec.replace('_', ' ').title()} ({overall.value})",
        f"**Findings:** {sum(len(r.findings) for r in reports.values())} across {len(reports)} agent reports",
    ]
    if ctx.pruned_chunks:
        lines.append(
            f"**Coverage:** {len(ctx.pruned_chunks)} diff chunk(s) were pruned "
            f"to stay within the context budget: {', '.join(ctx.pruned_chunks[:10])}"
            + (" …" if len(ctx.pruned_chunks) > 10 else "")
        )
    lines.append("")
    for name in ordered_agent_names():
        report = reports.get(name)
        if not report:
            continue
        lines.append(f"### {report.agent.title()} Agent ({report.status})")
        lines.append(f"> {report.summary}")
        notable = [f for f in report.findings if f.severity != Severity.INFO]
        for f in notable[:50]:
            loc = f"{f.file_path}:{f.line_start}" if f.file_path else ""
            lines.append(f"- **[{f.severity.value.upper()}]** {f.title}" + (f" `{loc}`" if loc else ""))
            if f.recommendation:
                lines.append(f"  - {f.recommendation}")
        if len(notable) > 50:
            lines.append(f"  - … {len(notable) - 50} more finding(s)")
        lines.append("")
    lines.append("---")
    return "\n".join(lines)


def _llm_synthesise(
    llm: object,
    reports: dict[str, AgentReport],
    ctx: ReviewContext,
) -> tuple[str, str, str, int, str | None]:
    """Returns ``(summary, severity, recommendation, token_usage, error)``."""
    from langchain_core.messages import HumanMessage, SystemMessage

    report_data: list[dict[str, Any]] = [
        {
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
        }
        for r in reports.values()
    ]

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
            return obj.get("summary", text), str(obj.get("overall_severity", "")).lower(), \
                str(obj.get("recommendation", "")).lower(), usage, None
        except json.JSONDecodeError:
            return text, "", "", usage, "LLM output was not valid JSON"
    except Exception as e:
        return "", "", "", 0, str(e)
