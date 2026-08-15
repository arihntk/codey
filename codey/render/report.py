"""Rich-based terminal report rendering for the review summary.

Renders the final ReviewSummary as markdown in the terminal: header panel,
executive summary, findings table, and a per-agent overview table.
"""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from codey.agents.schemas import Finding, ReviewSummary, Severity, severity_weight

__all__ = [
    "render_review",
    "print_finding_table",
    "severity_style",
]

_SEVERITY_STYLE: dict[Severity, str] = {
    Severity.CRITICAL: "bold red on red",
    Severity.HIGH: "bold red",
    Severity.MEDIUM: "bold yellow",
    Severity.LOW: "cyan",
    Severity.INFO: "dim",
}

_SEVERITY_ICON: dict[Severity, str] = {
    Severity.CRITICAL: "x",
    Severity.HIGH: "!",
    Severity.MEDIUM: "-",
    Severity.LOW: "·",
    Severity.INFO: "i",
}

_REC_STYLE: dict[str, str] = {
    "approve": "bold green",
    "request_changes": "bold yellow",
    "block": "bold red",
}


def severity_style(s: Severity) -> str:
    return _SEVERITY_STYLE.get(s, "white")


def render_review(review: ReviewSummary, console: Console | None = None) -> None:
    """Render the final review summary in the terminal."""
    console = console or Console()

    # Header panel.
    header = Table(show_header=False, box=None, padding=(0, 1))
    header.add_column(style="dim")
    header.add_column(style="bold")
    header.add_row("Commit:", review.commit_message[:120] or "(no message)")
    header.add_row("Hash:", review.commit_hash[:12] if review.commit_hash else "HEAD")
    header.add_row("Files reviewed:", str(len(review.files_reviewed)))
    header.add_row("Dependent files checked:", str(len(review.dependent_files_checked)))
    header.add_row("Total findings:", str(review.total_findings))

    rec_style = _REC_STYLE.get(review.recommendation, "white")
    rec_text = review.recommendation.replace("_", " ").title()
    header.add_row("Recommendation:", Text(rec_text, style=rec_style))
    header.add_row("Overall severity:", Text(review.overall_severity.value.upper(),
                                             style=severity_style(review.overall_severity)))
    if review.errors:
        header.add_row("Errors:", Text(f"{len(review.errors)} agent error(s)", style="red"))

    # Coverage note when context-budget pruning dropped chunks.
    pruned = review.pruned_chunks
    if pruned:
        header.add_row(
            "Pruned:",
            Text(
                f"{len(pruned)} diff chunk(s) omitted for context budget "
                f"({', '.join(pruned[:5])}{' …' if len(pruned) > 5 else ''})",
                style="yellow",
            ),
        )

    console.print(Panel(header, title="[bold]Codey Review[/]", border_style="blue", padding=(1, 2)))

    # Surface structured errors prominently instead of hiding them.
    if review.errors:
        console.print()
        console.print("[bold red]Agent errors[/]")
        for err in review.errors:
            console.print(f"  [red]×[/] {err}")
        console.print()

    # Executive summary as markdown.
    console.print()
    console.print(Markdown(review.summary))
    console.print()

    # Findings table.
    print_finding_table(review.all_findings(), console=console)

    # Per-agent overview.
    console.print()
    agent_table = Table(title="Agent Reports", show_lines=True)
    agent_table.add_column("Agent", style="bold cyan")
    agent_table.add_column("Status", style="white")
    agent_table.add_column("Findings", justify="right")
    agent_table.add_column("Tokens", justify="right", style="dim")
    agent_table.add_column("Summary", ratio=1, overflow="fold")

    for name in ("index", "security", "code_quality", "test"):
        report = review.agent_reports.get(name)
        if not report:
            continue
        status_text = Text(
            report.status,
            style="bold red" if report.status == "error" else ("yellow" if report.status == "skipped" else "green"),
        )
        agent_table.add_row(
            report.agent,
            status_text,
            str(report.finding_count()),
            str(report.token_usage),
            Markdown(report.summary) if report.summary else Text("(no summary)", style="dim"),
        )
    console.print(agent_table)


def print_finding_table(findings: list[Finding], *, console: Console | None = None) -> None:
    """Render a sorted findings table."""
    console = console or Console()
    if not findings:
        console.print("[dim]No findings.[/]")
        return

    sorted_findings = sorted(findings, key=lambda f: -severity_weight(f.severity))
    table = Table(title="Findings", show_lines=True)
    table.add_column("Sev", style="bold", width=4)
    table.add_column("Category", style="cyan")
    table.add_column("Finding", style="white", overflow="fold")
    table.add_column("Location", style="dim")

    for f in sorted_findings:
        loc = f"{f.file_path}:{f.line_start}" if f.file_path and f.line_start else (f.file_path or "")
        sev_text = Text(f"[{_SEVERITY_ICON[f.severity]}]", style=severity_style(f.severity))
        table.add_row(
            sev_text,
            f.category.value.replace("_", " "),
            f.title,
            loc,
        )
    console.print(table)
