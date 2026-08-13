"""Progress event emitter for live terminal updates during review.

Subscribes to LangGraph stream events and prints rich Status/Progress lines.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console

__all__ = ["ProgressEmitter", "make_callback"]

_AGENT_LABELS: dict[str, str] = {
    "index": "Indexing repository",
    "security": "Running security analysis",
    "code_quality": "Checking code quality",
    "test": "Running test suite",
    "codey": "Synthesising final review",
}


class ProgressEmitter:
    """Emits progress updates to the terminal using rich."""

    def __init__(self, console: Console | None = None, *, enabled: bool = True) -> None:
        self.console = console or Console(stderr=True)
        self.enabled = enabled
        self._steps: list[str] = []

    def emit(self, message: str, *, style: str = "cyan") -> None:
        if self.enabled:
            self.console.print(f"[{style}]›[/] {message}")

    def emit_agent_start(self, agent: str) -> None:
        label = _AGENT_LABELS.get(agent, agent)
        self.emit(label + "...", style="cyan")

    def emit_agent_done(self, agent: str, *, findings: int, status: str = "completed") -> None:
        label = _AGENT_LABELS.get(agent, agent)
        self.emit(f"{label}: {status} ({findings} finding(s))", style="green" if status == "completed" else "yellow")

    def emit_error(self, agent: str, error: str) -> None:
        self.emit(f"{agent} error: {error}", style="red")

    def done(self, message: str = "Review complete") -> None:
        self.emit(message, style="bold green")


def make_callback(emitter: ProgressEmitter):
    """Create a stream callback suitable for graph.stream()."""

    def callback(chunk: dict[str, Any]) -> None:
        for node_name, node_state in chunk.items():
            if node_name == "supervisor" or node_name == "codey":
                emitter.emit_agent_start("codey")
            elif node_name in _AGENT_LABELS:
                emitter.emit_agent_start(node_name)
                reports = node_state.get("agent_reports", {})
                report = reports.get(node_name)
                if report:
                    emitter.emit_agent_done(node_name, findings=len(report.findings), status=report.status)

    return callback
