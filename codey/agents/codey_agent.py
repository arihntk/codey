"""CodeyAgent (orchestrator/supervisor) — synthesises the final review.

After all worker agents have run in parallel, the orchestrator collects their
AgentReports, asks the LLM to write an executive summary, picks the overall
severity, and decides a recommendation (approve / request_changes / block).

The orchestrator can also spawn retrieval subagents (via create_react_agent
with grep/cat/callgraph tools) to gather more information if any agent's
findings are ambiguous or incomplete.
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
from codey.llm.response import extract_text
from codey.llm.retry import invoke_with_retry
from codey.tools.shell import build_tools_for_agents

__all__ = ["run_codey_agent", "spawn_retrieval_subagent"]

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

_RETRIEVAL_SYSTEM = (
    "You are a retrieval subagent for the Codey review system. Use your tools "
    "(grep, cat, ls, git) to investigate the repository and answer the orchestrator's "
    "question. Be concise. If you can't find the answer, say so explicitly."
)


def run_codey_agent(
    ctx: ReviewContext,
    agent_reports: dict[str, AgentReport],
    primary_llm: object | None,
    *,
    repo_for_subagents=None,
) -> ReviewSummary:
    """Synthesise all agent reports into a final ReviewSummary."""

    all_findings: list[Finding] = []
    for r in agent_reports.values():
        all_findings.extend(r.findings)

    overall = aggregate_severity(all_findings)
    rec = _default_recommendation(all_findings)

    summary_text = _build_text_summary(ctx, agent_reports, overall, rec)

    if primary_llm is not None:
        llm_summary, _used = _llm_synthesise(primary_llm, agent_reports, ctx)
        if llm_summary:
            summary_text = llm_summary

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
    lines.append("*Run `codey review --report <agent>` to view a standalone report.*")
    return "\n".join(lines)


def _default_recommendation(findings: list[Finding]) -> str:
    """Determine recommendation based on highest-severity findings."""
    if any(f.severity == Severity.CRITICAL for f in findings):
        return "block"
    if any(f.severity == Severity.HIGH and f.confidence >= 0.7 for f in findings):
        return "request_changes"
    return "approve"


def _llm_synthesise(
    llm: object,
    reports: dict[str, AgentReport],
    ctx: ReviewContext,
) -> tuple[str, int]:
    """Ask the LLM to write an executive summary."""
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
        try:
            obj = json.loads(text)
            summary = obj.get("summary", text)
            return summary, len(raw) // 4
        except json.JSONDecodeError:
            return text, len(raw) // 4
    except Exception:
        return "", 0


def spawn_retrieval_subagent(
    repo,
    primary_llm: object,
    question: str,
) -> str:
    """Spawn a react-agent with grep/cat/ls/git tools to answer a follow-up question.

    Uses langgraph's create_react_agent pattern.
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langgraph.prebuilt import create_react_agent

        tools = build_tools_for_agents(repo)
        agent = create_react_agent(primary_llm, tools=tools)
        result = agent.invoke({
            "messages": [
                SystemMessage(content=_RETRIEVAL_SYSTEM),
                HumanMessage(content=question),
            ],
        })
        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            return extract_text(last)
    except Exception as e:
        return f"[retrieval subagent failed: {e}]"
    return ""
