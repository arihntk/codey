"""Rich-based terminal report rendering + standalone report viewer.

Renders the final ReviewSummary as markdown in the terminal and offers
interactive viewing of individual agent standalone reports.
"""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import IntPrompt
from rich.table import Table
from rich.text import Text

from codey.agents.schemas import AgentReport, Finding, ReviewSummary, Severity, severity_weight

__all__ = [
    "render_review",
    "render_agent_report",
    "print_finding_table",
    "severity_style",
    "prompt_view_standalone",
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

    console.print(Panel(header, title="[bold]Codey Review[/]", border_style="blue", padding=(1, 2)))

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
    agent_table.add_column("Summary", style="dim", overflow="fold")

    for name in ("index", "security", "code_quality", "test"):
        report = review.agent_reports.get(name)
        if not report:
            continue
        agent_table.add_row(
            report.agent,
            report.status,
            str(report.finding_count()),
            str(report.token_usage),
            report.summary,
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


def _print_detailed_findings(findings: list[Finding], *, console: Console) -> None:
    """Print each finding with full description, evidence, and recommendation."""
    sorted_findings = sorted(findings, key=lambda f: -severity_weight(f.severity))
    for f in sorted_findings:
        sev_label = Text(
            f"[{_SEVERITY_ICON[f.severity]}] {f.severity.value.upper()}",
            style=severity_style(f.severity),
        )
        loc = ""
        if f.file_path and f.line_start:
            loc = f"  [dim]{f.file_path}:{f.line_start}[/]"
        console.print()
        console.print(sev_label, end="")
        console.print(f"  [cyan]{f.category.value.replace('_', ' ')}[/]", end="")
        console.print(f"  [bold]{f.title}[/]", end="")
        console.print(loc)
        if f.confidence < 1.0:
            console.print(f"  [dim]confidence: {f.confidence:.0%}[/]")
        if f.description:
            console.print(Markdown(f.description.strip()))
        if f.evidence:
            console.print()
            console.print("[dim]Evidence:[/]")
            console.print(Markdown(f"```\n{f.evidence.strip()}\n```"))
        if f.recommendation:
            console.print()
            console.print("[green]Recommendation:[/]")
            console.print(Markdown(f.recommendation.strip()))


def render_agent_report(report: AgentReport, console: Console | None = None) -> None:
    """Render a single agent's standalone report in full detail."""
    console = console or Console()
    title = f"{report.agent.title()} Agent Report"
    console.print(Panel(
        f"Status: [bold]{report.status}[/]\nFindings: {report.finding_count()}\nTokens: {report.token_usage}",
        title=f"[bold]{title}[/]",
        border_style="cyan",
    ))

    console.print()
    console.print("[bold]Summary[/]")
    console.print(Markdown(report.summary) if report.summary else "[dim](no summary)[/]")

    console.print()
    if report.findings:
        _print_detailed_findings(report.findings, console=console)
    else:
        console.print("[dim]No findings.[/]")

    if report.metadata:
        console.print()
        console.print("[bold]Metadata[/]")
        meta_table = Table(show_header=False)
        meta_table.add_column(style="dim")
        meta_table.add_column(style="white", overflow="fold")
        for k, v in report.metadata.items():
            val = v[:500] + ("..." if len(v) > 500 else "") if v else ""
            meta_table.add_row(k, val)
        console.print(meta_table)

    if report.error:
        console.print()
        console.print(f"[red]Error: {report.error}[/]")


def prompt_view_standalone(review: ReviewSummary, console: Console | None = None) -> None:
    """Interactive prompt to view individual agent standalone reports."""
    console = console or Console()
    agents = sorted(review.agent_reports.keys())
    if not agents:
        console.print("[dim]No agent reports available.[/]")
        return

    console.print()
    console.print("[bold]Standalone Reports[/]")
    for i, name in enumerate(agents, 1):
        report = review.agent_reports[name]
        console.print(f"  [cyan]{i}[/]. {name} ({report.finding_count()} findings, {report.status})")
    console.print("  [dim]0. Back[/]")

    choice = IntPrompt.ask(
        "[bold]View report number[/]",
        default=0,
        choices=[str(i) for i in range(len(agents) + 1)],
        console=console,
    )
    if choice == 0 or choice > len(agents):
        return
    chosen = agents[choice - 1]
    console.print()
    render_agent_report(review.agent_reports[chosen], console=console)
    console.print()
    # Allow chaining: prompt again.
    prompt_view_standalone(review, console=console)
