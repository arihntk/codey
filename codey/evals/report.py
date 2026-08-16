"""Eval report models (JSON-serializable) and Rich terminal rendering."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from codey.agents.schemas import Finding

__all__ = ["JudgeScore", "ScenarioScore", "AggregateScore", "EvalRunReport", "render_eval_report"]


class JudgeScore(BaseModel):
    groundedness: float = 0.0
    completeness: float = 0.0
    actionability: float = 0.0
    precision: float = 0.0
    overall: float = 0.0
    comment: str = ""


class ExpectedIssueOut(BaseModel):
    agent: str
    category: str
    severity: str
    file_path: str
    line_start: int | None = None
    line_end: int | None = None
    keywords: list[str] = Field(default_factory=list)
    expect_absent: bool = False


class ScenarioScore(BaseModel):
    id: str
    description: str
    status: Literal["completed", "error"] = "completed"
    error: str | None = None
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    severity_exact: float | None = None
    severity_within_one: float | None = None
    evidence_ratio: float | None = None
    evidence_grounded: float | None = None
    recommendation: str | None = None
    recommendation_match: bool | None = None
    coverage: float | None = None
    files_reviewed: int = 0
    changed_files: int = 0
    pruned_chunks: int = 0
    dependent_files_checked: list[str] = Field(default_factory=list)
    pipeline_checks: dict[str, bool] = Field(default_factory=dict)
    false_positives: list[Finding] = Field(default_factory=list)
    false_negatives: list[ExpectedIssueOut] = Field(default_factory=list)
    judge: JudgeScore | None = None
    notes: list[str] = Field(default_factory=list)
    duration_s: float = 0.0
    tokens: int = 0


class AggregateScore(BaseModel):
    scenarios_completed: int = 0
    scenarios_error: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    severity_exact: float | None = None
    severity_within_one: float | None = None
    evidence_ratio: float | None = None
    evidence_grounded: float | None = None
    recommendation_accuracy: float | None = None
    pipeline_checks_passed: float | None = None
    judge_overall: float | None = None
    tokens: int = 0
    duration_s: float = 0.0


class EvalRunReport(BaseModel):
    mode: str
    judge_enabled: bool
    scenario_count: int
    scenarios: list[ScenarioScore] = Field(default_factory=list)
    aggregate: AggregateScore = Field(default_factory=AggregateScore)
    duration_s: float = 0.0
    tokens: int = 0


def _pct(x: float | None) -> str:
    return f"{x * 100:.0f}%" if x is not None else "—"


def render_eval_report(report: EvalRunReport, console) -> None:
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    agg = report.aggregate
    header = Table(show_header=False, box=None, padding=(0, 1))
    header.add_column(style="dim")
    header.add_column(style="bold")
    header.add_row("Mode:", report.mode)
    header.add_row("Judge:", "enabled" if report.judge_enabled else "disabled")
    header.add_row("Scenarios:", str(report.scenario_count))
    header.add_row("Duration:", f"{report.duration_s:.1f}s")
    header.add_row("Tokens:", str(report.tokens))
    header.add_row(
        "Aggregate:",
        Text(
            f"P {_pct(agg.precision)}  R {_pct(agg.recall)}  F1 {agg.f1:.3f}  "
            f"({agg.tp} TP / {agg.fp} FP / {agg.fn} FN)",
            style="bold cyan",
        ),
    )
    if agg.severity_exact is not None:
        header.add_row("Severity (exact):", _pct(agg.severity_exact))
    if agg.recommendation_accuracy is not None:
        header.add_row("Recommendation accuracy:", _pct(agg.recommendation_accuracy))
    if agg.judge_overall is not None:
        header.add_row("Judge overall:", f"{agg.judge_overall:.2f}/5")
    console.print(Panel(header, title="[bold]Codey Evaluation[/]", border_style="blue", padding=(1, 2)))

    table = Table(title="Scenario Scores", expand=True, box=None)
    table.add_column("Scenario", ratio=3)
    table.add_column("Status")
    table.add_column("P", justify="right")
    table.add_column("R", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("FP", justify="right")
    table.add_column("FN", justify="right")
    table.add_column("SevAcc", justify="right")
    table.add_column("Rec", ratio=2)
    table.add_column("Evid%", justify="right")
    table.add_column("Judge", justify="right")
    table.add_column("Tok", justify="right")

    for s in report.scenarios:
        status_style = "bold red" if s.status == "error" else "green"
        rec_text = s.recommendation if s.recommendation else ""
        if s.recommendation_match is True:
            rec_text += " ✓"
        elif s.recommendation_match is False:
            rec_text += " ✗"
        table.add_row(
            s.id,
            Text(s.status, style=status_style),
            _pct(s.precision), _pct(s.recall), f"{s.f1:.3f}",
            str(s.fp), str(s.fn),
            _pct(s.severity_exact), rec_text,
            _pct(s.evidence_ratio),
            f"{s.judge.overall:.1f}" if s.judge else "—",
            str(s.tokens),
        )
    console.print()
    console.print(table)

    for s in report.scenarios:
        notes = []
        if s.error:
            notes.append(f"error: {s.error}")
        notes.extend(s.notes)
        if s.false_positives:
            notes.append(f"{len(s.false_positives)} false positive(s)")
        if s.false_negatives:
            notes.append(f"{len(s.false_negatives)} false negative(s)")
        if s.pipeline_checks and not all(s.pipeline_checks.values()):
            failed = [k for k, v in s.pipeline_checks.items() if not v]
            notes.append(f"pipeline check(s) failed: {', '.join(failed)}")
        if not notes:
            continue
        console.print()
        console.print(f"[bold cyan]{s.id}[/] — {'; '.join(notes)}")
        for f in s.false_positives[:10]:
            loc = f"{f.file_path}:{f.line_start}" if f.file_path else ""
            console.print(f"  [red]FP[/] [{f.severity.value}] {f.title} {loc}")
        for e in s.false_negatives[:10]:
            loc = f"{e.file_path}:{e.line_start}" if e.line_start else e.file_path
            console.print(f"  [yellow]FN[/] [{e.severity}] {e.agent} {e.category.value} {loc}")
        if s.judge and s.judge.comment:
            console.print(f"  [dim]judge:[/] {s.judge.comment[:200]}")

    console.print()
    agg_panel = Table(show_header=False, box=None, padding=(0, 1))
    agg_panel.add_column(style="dim")
    agg_panel.add_column(style="bold")
    agg_panel.add_row("Precision / Recall / F1:", f"{_pct(agg.precision)} / {_pct(agg.recall)} / {agg.f1:.3f}")
    agg_panel.add_row("TP / FP / FN:", f"{agg.tp} / {agg.fp} / {agg.fn}")
    if agg.severity_exact is not None:
        agg_panel.add_row("Severity exact / within-1:", f"{_pct(agg.severity_exact)} / {_pct(agg.severity_within_one)}")
    if agg.evidence_ratio is not None:
        agg_panel.add_row("Evidence attached:", _pct(agg.evidence_ratio))
    if agg.evidence_grounded is not None:
        agg_panel.add_row("Evidence grounded:", _pct(agg.evidence_grounded))
    if agg.recommendation_accuracy is not None:
        agg_panel.add_row("Recommendation accuracy:", _pct(agg.recommendation_accuracy))
    if agg.pipeline_checks_passed is not None:
        agg_panel.add_row("Pipeline checks:", _pct(agg.pipeline_checks_passed))
    if agg.judge_overall is not None:
        agg_panel.add_row("Judge overall:", f"{agg.judge_overall:.2f}/5")
    agg_panel.add_row("Total tokens:", str(agg.tokens))
    agg_panel.add_row("Total duration:", f"{agg.duration_s:.1f}s")
    console.print(Panel(agg_panel, title="[bold]Aggregate[/]", border_style="cyan"))
