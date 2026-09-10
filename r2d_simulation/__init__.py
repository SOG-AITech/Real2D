"""Modular implementation of the R2D charge-discharge simulation."""

import os

# Match the original monolithic entry point before NumPy/PETSc/DOLFINx load.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from .config import Params

__all__ = ["Params"]
