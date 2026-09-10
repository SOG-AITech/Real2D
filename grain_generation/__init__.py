"""Public API for the modular grain-boundary generation workflow."""

import os

# Set these before importing NumPy/PETSc/DOLFINx so each MPI rank keeps one
# numerical-library worker thread, matching the original command-line script.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from .config import DEFAULT_MSH_FILE, GrainParams


def create_context(config=None):
    from .workflow import create_context as create_workflow_context

    return create_workflow_context(config)


def run(config=None, *, msh_file=None, out_file=None):
    from .workflow import run as run_workflow

    return run_workflow(config, msh_file=msh_file, out_file=out_file)

__all__ = ["DEFAULT_MSH_FILE", "GrainParams", "create_context", "run"]
