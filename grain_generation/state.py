"""Runtime state containers shared by the grain-generation modules."""

from dataclasses import dataclass
from typing import Any


@dataclass
class MeshState:
    parent_mesh: Any
    grain_mesh: Any
    cell_tags: Any
    facet_tags: Any
    dx: Any
    grain_tag: Any
    cap_tag: Any


@dataclass
class FieldState:
    ME: Any
    V_scalar: Any
    eta: Any
    eta_n: Any
    eta_outputs: list
    B: Any = None
    ce: Any = None


@dataclass
class EquationSystem:
    residual: Any
    jacobian: Any
    dt: Any = None


@dataclass
class SolverState:
    problem: Any
    periodic_constraints: dict
    time: float = 0.0
    step: int = 0


@dataclass
class GrainRunContext:
    config: Any
    mesh: MeshState
    fields: FieldState
    equations: EquationSystem | None = None
    solver: SolverState | None = None
