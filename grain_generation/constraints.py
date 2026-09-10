"""Periodic and component boundary constraints."""

import numpy as np
from petsc4py import PETSc
from dolfinx import fem


def mixed_component_bc_from_scalar_dofs(ME, component, scalar_dofs, value):
    scalar_dofs = np.asarray(scalar_dofs, dtype=np.int32)
    if scalar_dofs.size == 0:
        return None
    _, submap = ME.sub(component).collapse()
    mixed_dofs = submap[scalar_dofs]
    return fem.dirichletbc(PETSc.ScalarType(value), mixed_dofs, ME.sub(component))


def build_lr_periodic_eta_constraints(msh, ME, n_grains, tol=1.0e-8):
    if msh.comm.size > 1:
        return {}
    V_scalar, _ = ME.sub(0).collapse()
    coords = V_scalar.tabulate_dof_coordinates()[:, :2]
    x_min = float(msh.geometry.x[:, 0].min())
    x_max = float(msh.geometry.x[:, 0].max())
    left = np.flatnonzero(np.isclose(coords[:, 0], x_min, atol=tol))
    right = np.flatnonzero(np.isclose(coords[:, 0], x_max, atol=tol))
    if left.size != right.size:
        raise RuntimeError(
            "Cannot build left-right periodic eta constraints: "
            f"left dofs={left.size}, right dofs={right.size}."
        )
    left_sorted = np.asarray(sorted(left, key=lambda dof: float(coords[dof, 1])), dtype=np.int32)
    right_sorted = np.asarray(sorted(right, key=lambda dof: float(coords[dof, 1])), dtype=np.int32)
    y_delta = np.abs(coords[left_sorted, 1] - coords[right_sorted, 1])
    max_y_delta = float(np.max(y_delta)) if y_delta.size else 0.0
    if max_y_delta > tol:
        raise RuntimeError(
            "Cannot build left-right periodic eta constraints: "
            f"max paired y mismatch={max_y_delta:.3e} exceeds tol={tol:.3e}. "
            "Regenerate the mesh with matching left/right boundary divisions, "
            "or increase the pairing tolerance only if this mismatch is purely roundoff."
        )
    constraints = {}
    for left_dof, right_dof in zip(left_sorted, right_sorted):
        for component in range(n_grains):
            _, submap = ME.sub(component).collapse()
            constraints[int(submap[right_dof])] = int(submap[left_dof])
    return constraints


__all__ = ["build_lr_periodic_eta_constraints", "mixed_component_bc_from_scalar_dofs"]


def build_constraints(mesh_state, field_state, params):
    """Build the eta periodic map used by the grain solver."""
    # The exact local slave/master elimination is only valid for the serial
    # PETSc layout.  The original runtime intentionally disabled this map in
    # MPI and let the distributed Newton system run without it.
    if mesh_state.grain_mesh.comm.size > 1:
        return {}
    return build_lr_periodic_eta_constraints(
        mesh_state.grain_mesh,
        field_state.ME,
        params.n_grains,
    )
