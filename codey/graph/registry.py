"""Declarative agent registry — the single source of truth for the review graph.

Adding an agent to the review pipeline = adding one ``AgentSpec`` entry here.
``build_graph`` (in :mod:`codey.graph.build`) reads this registry and wires
up the LangGraph nodes and edges automatically: each agent runs after its
declared dependencies, agents without dependents are the terminal nodes, and
the graph fans out / converges accordingly.

No changes to graph wiring are required when the set of agents changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from codey.graph.state import ReviewState

__all__ = ["AgentSpec", "get_specs", "register", "first_agent", "terminal_agents"]

StateUpdate = dict


@dataclass(frozen=True)
class AgentSpec:
    """Declaration of a review agent in the pipeline.

    Attributes:
        name: Unique node id used as the LangGraph node name.
        label: Human-readable label (used for progress display).
        run: Callable receiving the current ``ReviewState`` and returning a
            partial state update dict.
        depends_on: Agent names that must complete before this agent runs.
            The first agent in the registry with no dependencies becomes the
            graph entry point; agents with no dependents become terminal.
    """

    name: str
    label: str
    run: Callable[[ReviewState], StateUpdate]
    depends_on: tuple[str, ...] = field(default_factory=tuple)


# Registry populated by build.py to avoid circular imports at module load:
# the node functions live next to build_graph and register themselves here.
_SPECS: dict[str, AgentSpec] = {}


def register(spec: AgentSpec) -> AgentSpec:
    """Register an agent spec (called at graph build time)."""
    _SPECS[spec.name] = spec
    return spec


def get_specs() -> list[AgentSpec]:
    """Return all registered agent specs, in registration order."""
    return list(_SPECS.values())


def first_agent(specs: list[AgentSpec]) -> AgentSpec | None:
    """The graph entry point: the first spec with no dependencies."""
    for spec in specs:
        if not spec.depends_on:
            return spec
    return None


def terminal_agents(specs: list[AgentSpec]) -> list[str]:
    """Agent names that nothing depends on — these connect to END."""
    depended_on = {dep for spec in specs for dep in spec.depends_on}
    return [spec.name for spec in specs if spec.name not in depended_on]
