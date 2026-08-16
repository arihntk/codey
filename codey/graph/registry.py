"""Declarative agent registry — single source of truth for the review graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from codey.graph.state import ReviewState

__all__ = ["AgentSpec", "get_specs", "register", "first_agent", "terminal_agents", "ordered_agent_names"]

StateUpdate = dict

_FALLBACK_NAMES = ["index", "security", "code_quality", "test"]


@dataclass(frozen=True)
class AgentSpec:
    name: str
    label: str
    run: Callable[[ReviewState], StateUpdate]
    depends_on: tuple[str, ...] = field(default_factory=tuple)


# Populated by build.py to avoid circular imports at module load.
_SPECS: dict[str, AgentSpec] = {}


def register(spec: AgentSpec) -> AgentSpec:
    _SPECS[spec.name] = spec
    return spec


def get_specs() -> list[AgentSpec]:
    return list(_SPECS.values())


def first_agent(specs: list[AgentSpec]) -> AgentSpec | None:
    for spec in specs:
        if not spec.depends_on:
            return spec
    return None


def terminal_agents(specs: list[AgentSpec]) -> list[str]:
    depended_on = {dep for spec in specs for dep in spec.depends_on}
    return [spec.name for spec in specs if spec.name not in depended_on]


def ordered_agent_names() -> list[str]:
    names = [spec.name for spec in get_specs()]
    return names or list(_FALLBACK_NAMES)
