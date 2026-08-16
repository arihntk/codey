"""Progress event emitter for live terminal updates during review."""

from __future__ import annotations

from typing import Any

from rich.console import Console

__all__ = ["ProgressEmitter", "make_callback"]

_FALLBACK_LABELS: dict[str, str] = {
    "index": "Indexing repository",
    "security": "Running security analysis",
    "code_quality": "Checking code quality",
    "test": "Running test suite",
    "codey": "Synthesising final review",
}


def _agent_labels() -> dict[str, str]:
    try:
        from codey.graph.registry import get_specs

        specs = get_specs()
        if specs:
            return {spec.name: spec.label for spec in specs}
    except Exception:
        pass
    return _FALLBACK_LABELS


class ProgressEmitter:
    def __init__(self, console: Console | None = None, *, enabled: bool = True) -> None:
        self.console = console or Console(stderr=True)
        self.enabled = enabled
        self._steps: list[str] = []

    def emit(self, message: str, *, style: str = "cyan") -> None:
        if self.enabled:
            self.console.print(f"[{style}]›[/] {message}")

    def emit_agent_start(self, agent: str) -> None:
        self.emit(_agent_labels().get(agent, agent) + "...", style="cyan")

    def emit_agent_done(self, agent: str, *, findings: int, status: str = "completed") -> None:
        label = _agent_labels().get(agent, agent)
        self.emit(f"{label}: {status} ({findings} finding(s))",
                  style="green" if status == "completed" else "yellow")

    def emit_error(self, agent: str, error: str) -> None:
        self.emit(f"{agent} error: {error}", style="red")

    def done(self, message: str = "Review complete") -> None:
        self.emit(message, style="bold green")


def make_callback(emitter: ProgressEmitter):
    def callback(chunk: dict[str, Any]) -> None:
        labels = _agent_labels()
        for node_name, node_state in chunk.items():
            if node_name == "codey":
                emitter.emit_agent_start("codey")
            elif node_name in labels:
                emitter.emit_agent_start(node_name)
                report = node_state.get("agent_reports", {}).get(node_name)
                if report:
                    emitter.emit_agent_done(node_name, findings=len(report.findings), status=report.status)

    return callback
