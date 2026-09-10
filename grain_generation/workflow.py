"""Application workflow for grain-boundary generation."""

from .config import DEFAULT_MSH_FILE, GrainParams
from .evolution import run_context
from .state import GrainRunContext, SolverState
from .mesh import prepare_mesh
from .fields import create_fields
from .initialization import initialize_fields
from .constraints import build_constraints
from .equations import build_allen_cahn_system
from .solver import create_solver


def create_context(config=None):
    """Create the configuration context used by the modular workflow."""
    return GrainRunContext(config=config or GrainParams(), mesh=None, fields=None)


def run(config=None, *, msh_file=None, out_file=None):
    """Run one grain-generation job through the numerical runtime."""
    context = create_context(config)
    context.mesh = prepare_mesh(msh_file or DEFAULT_MSH_FILE, context.config)
    context.fields = create_fields(context.mesh, context.config)
    initialize_fields(context.mesh, context.fields, context.config)
    periodic_constraints = build_constraints(context.mesh, context.fields, context.config)
    context.equations = build_allen_cahn_system(context.mesh, context.fields, context.config)
    context.solver = create_solver(
        context.equations,
        context.fields,
        periodic_constraints,
        context.config,
    )
    context.solver = SolverState(context.solver, periodic_constraints)
    return run_context(
        context,
        output_file=out_file or "r2d_generate_garin.npz",
    )


__all__ = ["create_context", "run"]

