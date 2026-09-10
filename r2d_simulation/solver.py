"""Newton and linear-solver interfaces."""

from petsc4py import PETSc
import csv
from pathlib import Path
import time
import os
import tempfile

_VERBOSE_STAGE_LOGS = os.environ.get("R2D_STAGE_LOGS", "").lower() in ("1", "true", "yes", "on")
def mpi_stage_print(comm, message):
    if _VERBOSE_STAGE_LOGS or any(p in message for p in ("Newton", "problem.solve")):
        print(f"[rank {comm.rank}/{comm.size}] {message}", flush=True)
import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.fem.petsc import create_vector, assemble_vector, create_matrix, assemble_matrix, apply_lifting, set_bc
from .regions import collapse_mixed_subspace


def create_restricted_newton_problem(*, residual, jacobian, state, constraints,
                                     params):
    """Create the configured Newton problem from assembled workflow state."""
    comm = state.function_space.mesh.comm
    profiler = NewtonProfiler(params.newton_profile, params.newton_profile_file, comm)
    if params.newton_profile and comm.rank == 0:
        print(f"Newton profile enabled: {params.newton_profile_file}")
    jit_cache_dir = Path(tempfile.gettempdir()) / f"r2d_documented_fenics_jit_{os.getpid()}"
    jit_cache_dir.mkdir(parents=True, exist_ok=True)
    jit_options = {"cache_dir": str(jit_cache_dir), "timeout": 120}
    problem = RestrictedNewtonProblem(
        residual, jacobian, state,
        constraints.bcs, constraints.active_dofs,
        periodic_constraints=constraints.periodic_constraints,
        periodic_constraints_global=constraints.periodic_constraints_global,
        rtol=params.snes_rtol, atol=params.snes_atol, stol=params.snes_stol,
        max_it=params.snes_max_it, monitor=params.snes_monitor,
        profiler=profiler, linear_solver=params.linear_solver,
        linear_rtol=params.linear_rtol, linear_atol=params.linear_atol,
        linear_max_it=params.linear_max_it, jacobian_lag=params.jacobian_lag,
        reuse_jacobian_across_solves=params.reuse_jacobian_across_steps,
        fast_residual_bc=params.fast_residual_bc, jit_options=jit_options,
    )
    return problem, profiler


def build_solver_state(*, residual, jacobian, state, constraints, params):
    """Assembly-facing wrapper used by workflow orchestration."""
    return create_restricted_newton_problem(
        residual=residual, jacobian=jacobian, state=state,
        constraints=constraints, params=params)


def assign_constant(constant, value):
    try: constant.value = PETSc.ScalarType(value)
    except TypeError: constant.value[...] = PETSc.ScalarType(value)


class RestrictedNewtonStatus:
    """SNES-compatible status container used by the hand-written solver."""
    def __init__(self):
        self.reason = 0
        self.iterations = 0
        self.function_norm = float("nan")

    def getConvergedReason(self):
        return self.reason

    def getIterationNumber(self):
        return self.iterations

    def getFunctionNorm(self):
        return self.function_norm


class NewtonProfiler:
    """Low-overhead CSV profiler for the hand-written Newton loop."""

    fieldnames = (
        "accepted_step",
        "global_time_s",
        "phase",
        "dt_s",
        "retry",
        "newton_iteration",
        "alpha",
        "line_search_trials",
        "initial_residual",
        "previous_residual",
        "trial_residual",
        "step_norm",
        "residual_total_s",
        "residual_assemble_s",
        "residual_lifting_bc_s",
        "residual_projection_s",
        "jacobian_reused",
        "jacobian_age",
        "jacobian_total_s",
        "jacobian_assemble_s",
        "active_submatrix_s",
        "constraint_matmult_s",
        "linear_solve_s",
        "linear_iterations",
        "linear_reason",
        "solution_update_s",
        "solve_step_total_s",
        "line_search_total_s",
        "iteration_total_s",
        "time_error",
        "time_error_xi",
        "time_error_c",
        "time_adapt_factor",
        "time_adapt_dt_next",
        "time_adapt_reject_limit",
        "time_adapt_rejected",
    )

    def __init__(self, enabled: bool, path: str | Path | None, comm):
        self.enabled = bool(enabled)
        self.path = Path(path) if path is not None else None
        self.time_adapt_path = (
            self.path.with_name(f"{self.path.stem}_time_adapt{self.path.suffix}")
            if self.path is not None
            else None
        )
        self.comm = comm
        self.context = {
            "accepted_step": -1,
            "global_time_s": float("nan"),
            "phase": "",
            "dt_s": float("nan"),
            "retry": -1,
        }
        if self.enabled and self.path is not None and self.comm.rank == 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()
            if self.time_adapt_path is not None:
                with self.time_adapt_path.open("w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(
                        f,
                        fieldnames=(
                            "accepted_step",
                            "global_time_s",
                            "phase",
                            "dt_s",
                            "retry",
                            "time_error",
                            "time_error_xi",
                            "time_error_c",
                            "time_adapt_factor",
                            "time_adapt_dt_next",
                            "time_adapt_reject_limit",
                            "time_adapt_rejected",
                        ),
                    ).writeheader()

    def set_context(self, *, accepted_step, global_time_s, phase, dt_s, retry):
        self.context = {
            "accepted_step": int(accepted_step),
            "global_time_s": float(global_time_s),
            "phase": str(phase),
            "dt_s": float(dt_s),
            "retry": int(retry),
        }

    def write_iteration(self, row):
        if not self.enabled:
            return
        full_row = dict.fromkeys(self.fieldnames, "")
        full_row.update(self.context)
        full_row.update(row)
        if self.comm.rank == 0:
            if self.path is not None:
                with self.path.open("a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=self.fieldnames).writerow(full_row)
            print(
                "newton_profile "
                f"step={full_row['accepted_step']} "
                f"retry={full_row['retry']} "
                f"it={full_row['newton_iteration']} "
                f"alpha={full_row['alpha']:.3g} "
                f"res={full_row['trial_residual']:.3e} "
                f"J={full_row['jacobian_total_s']:.3f}s "
                f"reduce={full_row['constraint_matmult_s']:.3f}s "
                f"linear={full_row['linear_solve_s']:.3f}s "
                f"ls={full_row['line_search_total_s']:.3f}s "
                f"iter={full_row['iteration_total_s']:.3f}s"
            )

    def write_time_adapt(self, row):
        if not self.enabled:
            return
        full_row = dict.fromkeys(self.fieldnames, "")
        full_row.update(self.context)
        full_row.update(
            {
                "newton_iteration": 0,
                "alpha": "",
                "line_search_trials": "",
            }
        )
        full_row.update(row)
        if self.comm.rank == 0:
            if self.path is not None:
                with self.path.open("a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=self.fieldnames).writerow(full_row)
            if self.time_adapt_path is not None:
                with self.time_adapt_path.open("a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(
                        f,
                        fieldnames=(
                            "accepted_step",
                            "global_time_s",
                            "phase",
                            "dt_s",
                            "retry",
                            "time_error",
                            "time_error_xi",
                            "time_error_c",
                            "time_adapt_factor",
                            "time_adapt_dt_next",
                            "time_adapt_reject_limit",
                            "time_adapt_rejected",
                        ),
                    ).writerow(
                        {
                            "accepted_step": full_row["accepted_step"],
                            "global_time_s": full_row["global_time_s"],
                            "phase": full_row["phase"],
                            "dt_s": full_row["dt_s"],
                            "retry": full_row["retry"],
                            "time_error": full_row["time_error"],
                            "time_error_xi": full_row["time_error_xi"],
                            "time_error_c": full_row["time_error_c"],
                            "time_adapt_factor": full_row["time_adapt_factor"],
                            "time_adapt_dt_next": full_row["time_adapt_dt_next"],
                            "time_adapt_reject_limit": full_row["time_adapt_reject_limit"],
                            "time_adapt_rejected": full_row["time_adapt_rejected"],
                        }
                    )
            print(
                "time_adapt_profile "
                f"step={full_row['accepted_step']} "
                f"retry={full_row['retry']} "
                f"dt={float(full_row['dt_s']):.3e} "
                f"E={float(full_row['time_error']):.3e} "
                f"Exi={float(full_row['time_error_xi']):.3e} "
                f"Ec={float(full_row['time_error_c']):.3e} "
                f"factor={float(full_row['time_adapt_factor']):.3g} "
                f"dt_next={float(full_row['time_adapt_dt_next']):.3e} "
                f"rejected={full_row['time_adapt_rejected']}"
            )




def create_solver(*args, **kwargs):
    return RestrictedNewtonProblem(*args, **kwargs)


def nonlinear_solver_status(problem):
    solver = getattr(problem, "solver", None)
    if solver is None: return "unknown", "unknown", float("nan")
    try: reason = solver.getConvergedReason()
    except Exception: reason = "unknown"
    try: iterations = solver.getIterationNumber()
    except Exception: iterations = "unknown"
    try: residual = float(solver.getFunctionNorm())
    except Exception: residual = float("nan")
    return reason, iterations, residual


def nonlinear_solver_converged(reason):
    try: return int(reason) > 0
    except Exception: return False


__all__ = ["create_solver", "RestrictedNewtonProblem", "assign_constant", "RestrictedNewtonStatus", "NewtonProfiler", "build_solver_state",
           "nonlinear_solver_status", "nonlinear_solver_converged"]
class RestrictedNewtonProblem:
    """Newton solver with optional exact serial slave-master constraints.

    The original monolithic UFL residual and Jacobian are assembled on the full
    space.  Inactive dofs are removed from the Newton correction.  Periodic
    slave dofs are handled by a prolongation matrix P, solving
    ``P.T * A * P * du_r = P.T * r`` and prolonging ``du = P * du_r``.  This
    merges slave residual rows/columns into the master dofs instead of simply
    dropping the slave equations.

    Exact slave-master periodic constraints are kept serial.  MPI runs use
    owned global dof numbers for the active PETSc IS when no periodic
    constraints are present.
    """

    def __init__(
        self,
        F,
        J,
        u,
        bcs,
        active_dofs,
        periodic_constraints=None,
        periodic_constraints_global=None,
        *,
        rtol,
        atol,
        stol,
        max_it,
        monitor=False,
        profiler=None,
        linear_solver="lu",
        linear_rtol=1.0e-8,
        linear_atol=1.0e-12,
        linear_max_it=500,
        jacobian_lag=1,
        reuse_jacobian_across_solves=False,
        fast_residual_bc=False,
        jit_options=None,
    ):
        self.u = u
        self.comm = u.function_space.mesh.comm
        self.bcs = list(bcs)
        self.periodic_constraints = dict(periodic_constraints or {})
        self.periodic_constraints_global = dict(periodic_constraints_global or {})
        if self.comm.size > 1 and self.periodic_constraints and not self.periodic_constraints_global:
            raise RuntimeError("MPI periodic solve requires global periodic constraints.")
        self.periodic_slaves = np.asarray(
            sorted(self.periodic_constraints), dtype=np.int32
        )
        self.periodic_slaves_global = np.asarray(
            sorted(self.periodic_constraints_global), dtype=PETSc.IntType
        )
        self.F_form = fem.form(F, jit_options=jit_options)
        self.J_form = fem.form(J, jit_options=jit_options)
        try:
            self.b = create_vector(self.F_form)
        except TypeError:
            # Some DOLFINx versions expose create_vector(function_spaces)
            # rather than create_vector(linear_form). Assemble once to obtain a
            # compatible PETSc Vec, then zero it before every residual assembly.
            self.b = assemble_vector(self.F_form)
            with self.b.localForm() as loc_b:
                loc_b.set(0.0)
        self.A = create_matrix(self.J_form)
        active_input = np.asarray(active_dofs, dtype=np.int64)
        self.P = None
        self.P_is_distributed = False
        self.reduced_dofs = None
        self.reduced_is = None
        if self.comm.size == 1:
            self.active_dofs = active_input.astype(PETSc.IntType)
            self.active_local_dofs = self.active_dofs.astype(np.int32)
            self._build_constraint_prolongation()
            self.active_is = PETSc.IS().createGeneral(
                self.active_dofs, comm=self.comm
            )
            self.n_active_local = int(self.active_dofs.size)
            self.n_active_global = self.n_active_local
        else:
            local_size = self.u.x.petsc_vec.getLocalSize()
            active_local = active_input[
                (0 <= active_input) & (active_input < local_size)
            ]
            active_local = np.unique(active_local).astype(np.int32)
            index_map = self.u.function_space.dofmap.index_map
            index_map_bs = int(self.u.function_space.dofmap.index_map_bs)
            owned_size_from_map = int(index_map.size_local) * index_map_bs
            if owned_size_from_map != local_size:
                raise RuntimeError(
                    "RestrictedNewtonProblem expected PETSc local vector size to "
                    f"match the function-space owned dof map ({local_size} != "
                    f"{owned_size_from_map})."
                )
            # PETSc's mixed-space vector uses one scalar global numbering for
            # the local entries.  Keep the same numbering as the original
            # solver: the local PETSc ownership offset plus the local scalar
            # index.  The DOLFINx index-map block numbering is not the PETSc
            # scalar numbering used by the assembled matrix.
            ownership_start, _ownership_end = self.u.x.petsc_vec.getOwnershipRange()
            active_global = (
                int(ownership_start) + active_local.astype(np.int64)
            ).astype(PETSc.IntType)
            self.active_dofs = active_local.astype(PETSc.IntType)
            self.active_local_dofs = active_local
            self.active_global_dofs = active_global
            self.active_is = PETSc.IS().createGeneral(
                self.active_global_dofs, comm=self.comm
            )
            self.n_active_local = int(self.active_local_dofs.size)
            self.n_active_global = int(
                self.comm.allreduce(self.n_active_local, op=MPI.SUM)
            )
            self._build_distributed_constraint_prolongation()
        self.du = np.zeros_like(self.u.x.array)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.stol = float(stol)
        self.max_it = int(max_it)
        self.monitor = bool(monitor)
        self.profiler = profiler
        self.solver = RestrictedNewtonStatus()
        self.ksp = PETSc.KSP().create(self.comm)
        self.linear_solver = str(linear_solver).lower()
        self.linear_rtol = float(linear_rtol)
        self.linear_atol = float(linear_atol)
        self.linear_max_it = int(linear_max_it)
        self.jacobian_lag = max(1, int(jacobian_lag))
        self.reuse_jacobian_across_solves = bool(reuse_jacobian_across_solves)
        self.fast_residual_bc = bool(fast_residual_bc)
        self._cached_operator = None
        self._cached_aux_operator = None
        self._jacobian_age = self.jacobian_lag
        self._force_rebuild_jacobian = False
        self._configure_ksp()
        self.ksp.setFromOptions()

    def _configure_ksp(self):
        pc = self.ksp.getPC()
        mode = self.linear_solver
        if mode == "lu":
            self.ksp.setType("preonly")
            pc.setType("lu")
            return
        if mode == "mumps":
            self.ksp.setType("preonly")
            pc.setType("lu")
            try:
                pc.setFactorSolverType("mumps")
            except PETSc.Error:
                if self.comm.rank == 0:
                    print("linear_solver=mumps unavailable; falling back to plain LU.")
            return

        if mode.startswith("fgmres"):
            self.ksp.setType("fgmres")
        else:
            self.ksp.setType("gmres")
        self.ksp.setTolerances(
            rtol=self.linear_rtol,
            atol=self.linear_atol,
            max_it=self.linear_max_it,
        )
        if mode.endswith("_hypre"):
            pc.setType("hypre")
        elif mode.endswith("_gamg"):
            pc.setType("gamg")
        elif mode.endswith("_bjacobi"):
            pc.setType("bjacobi")
        elif mode.endswith("_jacobi"):
            pc.setType("jacobi")
        elif mode.endswith("_ilu"):
            pc.setType("ilu")
        else:
            pc.setType("ilu" if self.comm.size == 1 else "bjacobi")

    def _build_constraint_prolongation(self):
        self.P = None
        self.P_is_distributed = False
        self.reduced_dofs = self.active_dofs
        if not self.periodic_constraints:
            return

        active_set = set(int(d) for d in self.active_dofs)
        slave_set = set(int(d) for d in self.periodic_slaves)
        reduced_dofs = np.asarray(
            [int(d) for d in self.active_dofs if int(d) not in slave_set],
            dtype=PETSc.IntType,
        )
        reduced_index = {int(dof): i for i, dof in enumerate(reduced_dofs)}
        n_full = int(self.u.x.array.size)
        n_red = int(reduced_dofs.size)

        P = PETSc.Mat().createAIJ(
            size=(n_full, n_red),
            nnz=1,
            comm=self.u.function_space.mesh.comm,
        )
        for dof in reduced_dofs:
            P.setValue(int(dof), reduced_index[int(dof)], 1.0)
        for slave, master in self.periodic_constraints.items():
            slave = int(slave)
            master = int(master)
            if slave in active_set and master not in reduced_index:
                raise RuntimeError(
                    f"Periodic slave dof {slave} maps to inactive or slave master dof {master}."
                )
            if slave in active_set:
                P.setValue(slave, reduced_index[master], 1.0)
        P.assemble()

        self.P = P
        self.reduced_dofs = reduced_dofs

    def _build_distributed_constraint_prolongation(self):
        self.P = None
        self.P_is_distributed = False
        self.reduced_is = self.active_is
        if not self.periodic_constraints_global:
            return

        global_size = int(self.u.x.petsc_vec.getSize())
        local_size = int(self.u.x.petsc_vec.getLocalSize())
        active_global = np.asarray(self.active_global_dofs, dtype=np.int64)
        slave_set = set(int(d) for d in self.periodic_constraints_global)
        active_global_all = set(
            int(d)
            for part in self.comm.allgather(active_global.tolist())
            for d in part
        )
        reduced_global_local = np.asarray(
            [int(d) for d in active_global if int(d) not in slave_set],
            dtype=PETSc.IntType,
        )
        reduced_global_all = set(
            int(d)
            for part in self.comm.allgather(reduced_global_local.astype(np.int64).tolist())
            for d in part
        )
        for slave, master in self.periodic_constraints_global.items():
            if int(slave) in active_global_all and int(master) not in reduced_global_all:
                raise RuntimeError(
                    f"MPI periodic slave dof {slave} maps to inactive/slave master dof {master}."
                )

        try:
            P = PETSc.Mat().createAIJ(
                size=((local_size, global_size), (local_size, global_size)),
                nnz=1,
                comm=self.comm,
            )
        except Exception:
            P = PETSc.Mat().createAIJ(
                size=(global_size, global_size),
                nnz=1,
                comm=self.comm,
            )

        row_start, row_end = self.u.x.petsc_vec.getOwnershipRange()
        active_owned = set(int(d) for d in active_global)
        for row in range(int(row_start), int(row_end)):
            if row not in active_owned:
                continue
            if row in self.periodic_constraints_global:
                col = int(self.periodic_constraints_global[row])
            else:
                col = row
            P.setValue(row, col, 1.0)
        P.assemble()

        self.P = P
        self.P_is_distributed = True
        self.reduced_dofs = reduced_global_local
        self.reduced_is = PETSc.IS().createGeneral(
            reduced_global_local, comm=self.comm
        )

    def _apply_periodic_constraints(self):
        if self.P_is_distributed and self.P is not None:
            src = self.u.x.petsc_vec.duplicate()
            dst = self.u.x.petsc_vec.duplicate()
            self.u.x.petsc_vec.copy(src)
            self.P.mult(src, dst)
            dst.copy(self.u.x.petsc_vec)
            src.destroy()
            dst.destroy()
            self.u.x.scatter_forward()
            return
        if self.periodic_constraints:
            arr = self.u.x.array
            for slave, master in self.periodic_constraints.items():
                arr[slave] = arr[master]
        self.u.x.scatter_forward()

    def _assemble_residual(self):
        total_start = time.perf_counter()
        t0 = time.perf_counter()
        self._apply_periodic_constraints()
        constraint_apply_s = time.perf_counter() - t0
        with self.b.localForm() as loc_b:
            loc_b.set(0.0)
        t0 = time.perf_counter()
        try:
            assemble_vector(self.b, self.F_form)
        except TypeError:
            assembled = assemble_vector(self.F_form)
            self.b.array[:] = assembled.array_r
            assembled.destroy()
        residual_assemble_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        if not self.fast_residual_bc:
            apply_lifting(
                self.b,
                [self.J_form],
                [self.bcs],
                x0=[self.u.x.petsc_vec],
                alpha=-1.0,
            )
        self.b.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )
        set_bc(self.b, self.bcs, self.u.x.petsc_vec, -1.0)
        residual_lifting_bc_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        if self.P is None:
            r_active = self.b.getSubVector(self.active_is)
            norm = float(r_active.norm())
            self.b.restoreSubVector(self.active_is, r_active)
            residual_projection_s = time.perf_counter() - t0
            self._last_residual_timing = {
                "residual_total_s": time.perf_counter() - total_start,
                "residual_assemble_s": residual_assemble_s,
                "residual_lifting_bc_s": residual_lifting_bc_s + constraint_apply_s,
                "residual_projection_s": residual_projection_s,
            }
            return norm
        if self.P_is_distributed:
            b_projected = self.b.duplicate()
            self.P.multTranspose(self.b, b_projected)
            r_reduced = b_projected.getSubVector(self.reduced_is)
            norm = float(r_reduced.norm())
            b_projected.restoreSubVector(self.reduced_is, r_reduced)
            b_projected.destroy()
            residual_projection_s = time.perf_counter() - t0
            self._last_residual_timing = {
                "residual_total_s": time.perf_counter() - total_start,
                "residual_assemble_s": residual_assemble_s,
                "residual_lifting_bc_s": residual_lifting_bc_s + constraint_apply_s,
                "residual_projection_s": residual_projection_s,
            }
            return norm
        r_reduced = PETSc.Vec().createSeq(
            int(self.reduced_dofs.size), comm=self.u.function_space.mesh.comm
        )
        self.P.multTranspose(self.b, r_reduced)
        norm = float(r_reduced.norm())
        r_reduced.destroy()
        residual_projection_s = time.perf_counter() - t0
        self._last_residual_timing = {
            "residual_total_s": time.perf_counter() - total_start,
            "residual_assemble_s": residual_assemble_s,
            "residual_lifting_bc_s": residual_lifting_bc_s + constraint_apply_s,
            "residual_projection_s": residual_projection_s,
        }
        return norm

    def _assemble_jacobian(self):
        total_start = time.perf_counter()
        self._apply_periodic_constraints()
        t0 = time.perf_counter()
        self.A.zeroEntries()
        assemble_matrix(self.A, self.J_form, bcs=self.bcs)
        self.A.assemble()
        jacobian_assemble_s = time.perf_counter() - t0
        self._last_jacobian_timing = {
            "jacobian_total_s": time.perf_counter() - total_start,
            "jacobian_assemble_s": jacobian_assemble_s,
        }

    def _destroy_cached_operator(self):
        for attr in ("_cached_operator", "_cached_aux_operator"):
            obj = getattr(self, attr, None)
            if obj is not None:
                objs = obj if isinstance(obj, tuple) else (obj,)
                for item in objs:
                    try:
                        item.destroy()
                    except Exception:
                        pass
                setattr(self, attr, None)

    def reset_jacobian_cache(self):
        self._destroy_cached_operator()
        self._jacobian_age = self.jacobian_lag
        self._force_rebuild_jacobian = False

    def _solve_step(self, rebuild_jacobian=True):
        total_start = time.perf_counter()
        if rebuild_jacobian:
            self._destroy_cached_operator()
            self._assemble_jacobian()
            jacobian_reused = 0
            jacobian_age = 0
        else:
            self._last_jacobian_timing = {
                "jacobian_total_s": 0.0,
                "jacobian_assemble_s": 0.0,
            }
            jacobian_reused = 1
            jacobian_age = self._jacobian_age

        if self.P is None:
            t0 = time.perf_counter()
            if rebuild_jacobian or self._cached_operator is None:
                self._cached_operator = self.A.createSubMatrix(
                    self.active_is, self.active_is
                )
            A_active = self._cached_operator
            r_active = self.b.getSubVector(self.active_is)
            du_active = r_active.duplicate()
            active_submatrix_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            if rebuild_jacobian:
                self.ksp.setOperators(A_active)
            self.ksp.solve(r_active, du_active)
            linear_solve_s = time.perf_counter() - t0
            linear_iterations = int(self.ksp.getIterationNumber())
            linear_reason = int(self.ksp.getConvergedReason())
            t0 = time.perf_counter()
            self.du.fill(0.0)
            self.du[self.active_local_dofs] = du_active.array_r
            step_norm = float(du_active.norm())
            solution_update_s = time.perf_counter() - t0
            self.b.restoreSubVector(self.active_is, r_active)
            du_active.destroy()
            self._last_solve_timing = {
                **getattr(self, "_last_jacobian_timing", {}),
                "jacobian_reused": jacobian_reused,
                "jacobian_age": jacobian_age,
                "active_submatrix_s": active_submatrix_s,
                "constraint_matmult_s": 0.0,
                "linear_solve_s": linear_solve_s,
                "linear_iterations": linear_iterations,
                "linear_reason": linear_reason,
                "solution_update_s": solution_update_s,
                "solve_step_total_s": time.perf_counter() - total_start,
            }
            return step_norm

        if self.P_is_distributed:
            t0 = time.perf_counter()
            if rebuild_jacobian or self._cached_operator is None:
                if self.monitor:
                    mpi_stage_print(self.comm, "newton solve: before A*P")
                AP = self.A.matMult(self.P)
                if self.monitor:
                    mpi_stage_print(self.comm, "newton solve: before P.T*(A*P)")
                A_projected = self.P.transposeMatMult(AP)
                if self.monitor:
                    mpi_stage_print(self.comm, "newton solve: before reduced submatrix")
                A_reduced = A_projected.createSubMatrix(
                    self.reduced_is, self.reduced_is
                )
                self._cached_aux_operator = (AP, A_projected)
                self._cached_operator = A_reduced
            else:
                A_reduced = self._cached_operator
            b_projected = self.b.duplicate()
            if self.monitor:
                mpi_stage_print(self.comm, "newton solve: before P.T*b")
            self.P.multTranspose(self.b, b_projected)
            r_reduced = b_projected.getSubVector(self.reduced_is)
            du_reduced = r_reduced.duplicate()
            constraint_matmult_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            if rebuild_jacobian:
                self.ksp.setOperators(A_reduced)
            if self.monitor:
                mpi_stage_print(self.comm, "newton solve: before KSP solve")
            self.ksp.solve(r_reduced, du_reduced)
            if self.monitor:
                mpi_stage_print(self.comm, "newton solve: after KSP solve")
            linear_solve_s = time.perf_counter() - t0
            linear_iterations = int(self.ksp.getIterationNumber())
            linear_reason = int(self.ksp.getConvergedReason())
            t0 = time.perf_counter()
            du_projected = self.b.duplicate()
            with du_projected.localForm() as loc:
                loc.set(0.0)
            du_projected_reduced = du_projected.getSubVector(self.reduced_is)
            du_reduced.copy(du_projected_reduced)
            du_projected.restoreSubVector(self.reduced_is, du_projected_reduced)
            du_full = self.b.duplicate()
            self.P.mult(du_projected, du_full)
            self.du.fill(0.0)
            owned_size = int(du_full.getLocalSize())
            self.du[:owned_size] = du_full.array_r
            step_norm = float(du_reduced.norm())
            solution_update_s = time.perf_counter() - t0
            b_projected.restoreSubVector(self.reduced_is, r_reduced)
            b_projected.destroy()
            du_projected.destroy()
            du_full.destroy()
            du_reduced.destroy()
            self._last_solve_timing = {
                **getattr(self, "_last_jacobian_timing", {}),
                "jacobian_reused": jacobian_reused,
                "jacobian_age": jacobian_age,
                "active_submatrix_s": 0.0,
                "constraint_matmult_s": constraint_matmult_s,
                "linear_solve_s": linear_solve_s,
                "linear_iterations": linear_iterations,
                "linear_reason": linear_reason,
                "solution_update_s": solution_update_s,
                "solve_step_total_s": time.perf_counter() - total_start,
            }
            return step_norm

        t0 = time.perf_counter()
        if rebuild_jacobian or self._cached_operator is None:
            AP = self.A.matMult(self.P)
            A_reduced = self.P.transposeMatMult(AP)
            self._cached_aux_operator = AP
            self._cached_operator = A_reduced
        else:
            AP = self._cached_aux_operator
            A_reduced = self._cached_operator
        r_reduced = PETSc.Vec().createSeq(
            int(self.reduced_dofs.size), comm=self.u.function_space.mesh.comm
        )
        self.P.multTranspose(self.b, r_reduced)
        du_reduced = r_reduced.duplicate()
        constraint_matmult_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        if rebuild_jacobian:
            self.ksp.setOperators(A_reduced)
        self.ksp.solve(r_reduced, du_reduced)
        linear_solve_s = time.perf_counter() - t0
        linear_iterations = int(self.ksp.getIterationNumber())
        linear_reason = int(self.ksp.getConvergedReason())
        t0 = time.perf_counter()
        self.du.fill(0.0)
        self.du[self.reduced_dofs] = du_reduced.array_r
        for slave, master in self.periodic_constraints.items():
            self.du[int(slave)] = self.du[int(master)]
        step_norm = float(du_reduced.norm())
        solution_update_s = time.perf_counter() - t0
        r_reduced.destroy()
        du_reduced.destroy()
        self._last_solve_timing = {
            **getattr(self, "_last_jacobian_timing", {}),
            "jacobian_reused": jacobian_reused,
            "jacobian_age": jacobian_age,
            "active_submatrix_s": 0.0,
            "constraint_matmult_s": constraint_matmult_s,
            "linear_solve_s": linear_solve_s,
            "linear_iterations": linear_iterations,
            "linear_reason": linear_reason,
            "solution_update_s": solution_update_s,
            "solve_step_total_s": time.perf_counter() - total_start,
        }
        return step_norm

    def solve(self):
        self.solver.reason = 0
        self.solver.iterations = 0
        self.solver.function_norm = float("nan")
        if not self.reuse_jacobian_across_solves or self._cached_operator is None:
            self._destroy_cached_operator()
            self._jacobian_age = self.jacobian_lag
            self._force_rebuild_jacobian = False
        else:
            self._jacobian_age = min(
                self._jacobian_age, max(0, self.jacobian_lag - 1)
            )
        norm0 = self._assemble_residual()
        initial_residual_timing = dict(getattr(self, "_last_residual_timing", {}))
        self.solver.function_norm = norm0
        if self.monitor and self.comm.rank == 0:
            print(f"restricted_newton initial active_fnorm={norm0:.3e}", flush=True)
        if norm0 < self.atol:
            self.solver.reason = 2
            self.solver.iterations = 0
            return
        reference = max(norm0, 1.0)
        previous_norm = norm0
        old = self.u.x.array.copy()
        self._apply_periodic_constraints()
        old = self.u.x.array.copy()
        for it in range(1, self.max_it + 1):
            iteration_start = time.perf_counter()
            residual_before_step = previous_norm
            rebuild_jacobian = (
                self._cached_operator is None
                or self._force_rebuild_jacobian
                or self._jacobian_age >= self.jacobian_lag
            )
            if self.monitor and self.comm.rank == 0:
                print(
                    f"restricted_newton it={it} begin, "
                    f"rebuild_jacobian={int(rebuild_jacobian)}, "
                    f"active_fnorm={previous_norm:.3e}",
                    flush=True,
                )
            step_norm = self._solve_step(rebuild_jacobian=rebuild_jacobian)
            if rebuild_jacobian:
                self._jacobian_age = 1
                self._force_rebuild_jacobian = False
            else:
                self._jacobian_age += 1
            solve_timing = dict(getattr(self, "_last_solve_timing", {}))
            if step_norm < self.stol:
                self.solver.reason = 3
                self.solver.iterations = it
                self.solver.function_norm = previous_norm
                return

            accepted = False
            alpha = 1.0
            line_search_start = time.perf_counter()
            line_search_residual_timing = {}
            line_search_trials = 0
            trial_norm = float("nan")
            for _ in range(10):
                line_search_trials += 1
                self.u.x.array[:] = old - alpha * self.du
                self._apply_periodic_constraints()
                trial_norm = self._assemble_residual()
                line_search_residual_timing = dict(
                    getattr(self, "_last_residual_timing", {})
                )
                if trial_norm <= previous_norm or alpha <= 1.0e-3:
                    accepted = True
                    break
                alpha *= 0.5
            line_search_total_s = time.perf_counter() - line_search_start

            if not accepted:
                self.u.x.array[:] = old
                self._apply_periodic_constraints()
                self.solver.reason = -3
                self.solver.iterations = it
                self.solver.function_norm = previous_norm
                return

            old = self.u.x.array.copy()
            previous_norm = trial_norm
            if alpha < 1.0:
                self._force_rebuild_jacobian = True
            self.solver.function_norm = previous_norm
            self.solver.iterations = it
            if self.monitor and self.comm.rank == 0:
                print(
                    f"restricted_newton it={it}, alpha={alpha:.3g}, "
                    f"active_fnorm={previous_norm:.3e}, step={step_norm:.3e}"
                )
            if self.profiler is not None:
                profile_row = {
                    "newton_iteration": it,
                    "alpha": alpha,
                    "line_search_trials": line_search_trials,
                    "initial_residual": norm0,
                    "previous_residual": residual_before_step,
                    "trial_residual": trial_norm,
                    "step_norm": step_norm,
                    "line_search_total_s": line_search_total_s,
                    "iteration_total_s": time.perf_counter() - iteration_start,
                }
                profile_row.update(initial_residual_timing)
                profile_row.update(solve_timing)
                profile_row.update(line_search_residual_timing)
                self.profiler.write_iteration(profile_row)
            if previous_norm < self.atol or previous_norm / reference < self.rtol:
                self.solver.reason = 2
                return

        self.solver.reason = -2
        self.solver.iterations = self.max_it
        self.solver.function_norm = previous_norm
