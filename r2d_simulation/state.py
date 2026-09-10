"""Shared state containers for the R2D workflow.

The containers are intentionally permissive: the existing solver creates
several related UFL/PETSc objects together, so modules can be connected
without copying or serializing those objects.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MeshState:
    mesh: Any = None
    cell_tags: Any = None
    facet_tags: Any = None
    dx: Any = None
    ds: Any = None
    dS: Any = None
    cell_markers: Any = None


@dataclass
class FiniteElementState:
    """Finite-element spaces and shared mixed-state UFL objects."""

    V_scalar: Any = None
    domain_markers: Any = None
    stats_dofs: dict[str, Any] = field(default_factory=dict)
    boundary_probe_dofs: dict[str, Any] = field(default_factory=dict)
    eta_regions: tuple[int, ...] = ()
    cathode_y_min: Any = None
    cathode_y_max: Any = None
    omega1_y_min: Any = None
    omega1_y_max: Any = None
    lifecycle_target_y: Any = None
    ME: Any = None
    n_grains: int = 0
    state: Any = None
    previous_state: Any = None
    previous_previous_state: Any = None
    older_state: Any = None
    components: Any = None
    previous_components: Any = None
    previous_previous_components: Any = None
    tests: Any = None
    trial: Any = None
    xi_ref: Any = None
    previous_state_array: Any = None
    eta_initial_values: Any = None


@dataclass
class FieldState:
    V_scalar: Any = None
    ME: Any = None
    fields: dict[str, Any] = field(default_factory=dict)
    scalar_outputs: dict[str, Any] = field(default_factory=dict)
    derived_outputs: dict[str, Any] = field(default_factory=dict)
    finite_element: FiniteElementState = field(default_factory=FiniteElementState)


@dataclass
class EquationSystem:
    residual: Any = None
    jacobian: Any = None
    forms: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverState:
    problem: Any = None
    profiler: Any = None
    status: Any = None


@dataclass
class SimulationConstants:
    """Mutable FEM constants shared by residuals and time integration."""

    dt: Any = None
    bdf_a0: Any = None
    bdf_a1: Any = None
    bdf_a2: Any = None
    current_density: Any = None
    applied_current: Any = None


@dataclass
class R2DContext:
    params: Any
    mesh: MeshState = field(default_factory=MeshState)
    fields: FieldState = field(default_factory=FieldState)
    equations: EquationSystem = field(default_factory=EquationSystem)
    solver: SolverState = field(default_factory=SolverState)
    constraints: Any = None
    evolution: Any = None
    constants: SimulationConstants = field(default_factory=SimulationConstants)
    diagnostics: Any = None
    output: Any = None
    runtime: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "MeshState", "FiniteElementState", "FieldState", "EquationSystem",
    "SolverState", "SimulationConstants", "R2DContext",
]
