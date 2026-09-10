"""Nonlinear periodic Newton solver used by grain annealing."""

import numpy as np
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import (
    NonlinearProblem,
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_matrix,
    create_vector,
    set_bc,
)


class PeriodicNewtonProblem:
    """Serial Newton solver that enforces eta slave-master periodic constraints."""

    def __init__(self, F, u, bcs, J, periodic_constraints, *, petsc_options=None):
        self.u = u
        self.comm = u.function_space.mesh.comm
        self._distributed_problem = None
        if self.comm.size > 1:
            options = dict(petsc_options or {})
            options["ksp_type"] = "gmres"
            options["pc_type"] = "jacobi"
            options["snes_type"] = "newtonls"
            options["snes_error_if_not_converged"] = False
            self._distributed_problem = NonlinearProblem(
                F, u, bcs=list(bcs), J=J, petsc_options=options,
                petsc_options_prefix="grain_",
            )
            self.solver = type("DistributedNewtonStatus", (), {})()
            return
        self.bcs = list(bcs)
        self.periodic_constraints = dict(periodic_constraints)
        self.F_form = fem.form(F)
        self.J_form = fem.form(J)
        try:
            self.b = create_vector(self.F_form)
        except TypeError:
            self.b = assemble_vector(self.F_form)
            with self.b.localForm() as loc_b:
                loc_b.set(0.0)
        self.A = create_matrix(self.J_form)
        options = dict(petsc_options or {})
        self.rtol = float(options.get("snes_rtol", 1.0e-7))
        self.atol = float(options.get("snes_atol", 1.0e-9))
        self.max_it = int(options.get("snes_max_it", 30))
        self.solver = type("PeriodicNewtonStatus", (), {})()
        self.solver.reason = 0
        self.solver.iterations = 0
        self.ksp = PETSc.KSP().create(self.comm)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")
        self.ksp.setFromOptions()
        self.full_dofs = np.arange(self.u.x.array.size, dtype=PETSc.IntType)
        self.du = np.zeros_like(self.u.x.array)
        self._build_constraint_prolongation()

    def _build_constraint_prolongation(self):
        slave_set = set(int(d) for d in self.periodic_constraints)
        reduced_dofs = np.asarray(
            [int(d) for d in self.full_dofs if int(d) not in slave_set],
            dtype=PETSc.IntType,
        )
        reduced_index = {int(dof): i for i, dof in enumerate(reduced_dofs)}
        n_full = int(self.u.x.array.size)
        n_red = int(reduced_dofs.size)
        P = PETSc.Mat().createAIJ(size=(n_full, n_red), nnz=1, comm=self.comm)
        for dof in reduced_dofs:
            P.setValue(int(dof), reduced_index[int(dof)], 1.0)
        for slave, master in self.periodic_constraints.items():
            if int(master) not in reduced_index:
                raise RuntimeError(
                    f"Periodic slave dof {slave} maps to inactive/slave master {master}."
                )
            P.setValue(int(slave), reduced_index[int(master)], 1.0)
        P.assemble()
        self.P = P
        self.reduced_dofs = reduced_dofs

    def _apply_periodic_constraints(self):
        arr = self.u.x.array
        for slave, master in self.periodic_constraints.items():
            arr[int(slave)] = arr[int(master)]
        self.u.x.scatter_forward()

    def _assemble_full_residual(self):
        self._apply_periodic_constraints()
        with self.b.localForm() as loc_b:
            loc_b.set(0.0)
        try:
            assemble_vector(self.b, self.F_form)
        except TypeError:
            assembled = assemble_vector(self.F_form)
            self.b.array[:] = assembled.array_r
            assembled.destroy()
        apply_lifting(self.b, [self.J_form], [self.bcs], x0=[self.u.x.petsc_vec], alpha=-1.0)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b, self.bcs, self.u.x.petsc_vec, -1.0)

    def _reduced_residual_norm(self):
        self._assemble_full_residual()
        r_reduced = PETSc.Vec().createSeq(int(self.reduced_dofs.size), comm=self.comm)
        self.P.multTranspose(self.b, r_reduced)
        norm = float(r_reduced.norm())
        r_reduced.destroy()
        return norm

    def _solve_step(self):
        self._apply_periodic_constraints()
        self.A.zeroEntries()
        assemble_matrix(self.A, self.J_form, bcs=self.bcs)
        self.A.assemble()
        AP = self.A.matMult(self.P)
        A_reduced = self.P.transposeMatMult(AP)
        r_reduced = PETSc.Vec().createSeq(int(self.reduced_dofs.size), comm=self.comm)
        self.P.multTranspose(self.b, r_reduced)
        du_reduced = r_reduced.duplicate()
        self.ksp.setOperators(A_reduced)
        self.ksp.solve(r_reduced, du_reduced)
        self.du.fill(0.0)
        self.du[self.reduced_dofs] = du_reduced.array_r
        for slave, master in self.periodic_constraints.items():
            self.du[int(slave)] = self.du[int(master)]
        step_norm = float(du_reduced.norm())
        AP.destroy()
        A_reduced.destroy()
        r_reduced.destroy()
        du_reduced.destroy()
        return step_norm

    def solve(self):
        if self._distributed_problem is not None:
            result = self._distributed_problem.solve()
            petsc_solver = self._distributed_problem.solver
            self.solver.reason = int(petsc_solver.getConvergedReason())
            self.solver.iterations = int(petsc_solver.getIterationNumber())
            if self.comm.rank == 0 and self.solver.reason <= 0:
                print(
                    f"Distributed SNES did not converge: reason={self.solver.reason}, "
                    f"iterations={self.solver.iterations}, "
                    f"function_norm={petsc_solver.getFunctionNorm():.3e}", flush=True,
                )
            return result
        self.solver.reason = 0
        self.solver.iterations = 0
        self._apply_periodic_constraints()
        norm0 = self._reduced_residual_norm()
        if norm0 < self.atol:
            self.solver.reason = 2
            return
        reference = max(norm0, 1.0)
        previous_norm = norm0
        old = self.u.x.array.copy()
        for it in range(1, self.max_it + 1):
            self._assemble_full_residual()
            step_norm = self._solve_step()
            alpha = 1.0
            accepted = False
            for _ in range(10):
                self.u.x.array[:] = old - alpha * self.du
                self._apply_periodic_constraints()
                trial_norm = self._reduced_residual_norm()
                if trial_norm <= previous_norm or alpha <= 1.0e-3:
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                self.u.x.array[:] = old
                self._apply_periodic_constraints()
                self.solver.reason = -3
                self.solver.iterations = it
                return
            old = self.u.x.array.copy()
            previous_norm = trial_norm
            self.solver.iterations = it
            if previous_norm < self.atol or previous_norm / reference < self.rtol:
                self.solver.reason = 2
                return
            if step_norm < 1.0e-12:
                self.solver.reason = 3
                return
        self.solver.reason = -2
        self.solver.iterations = self.max_it


__all__ = ["PeriodicNewtonProblem"]


def create_solver(equations, field_state, constraints, params):
    """Create the periodic Newton solver from a prepared equation system."""
    options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1.0e-7,
        "snes_atol": 1.0e-9,
        "snes_max_it": 30,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    return PeriodicNewtonProblem(
        equations.residual,
        field_state.eta,
        bcs=[],
        J=equations.jacobian,
        periodic_constraints=constraints,
        petsc_options=options,
    )
