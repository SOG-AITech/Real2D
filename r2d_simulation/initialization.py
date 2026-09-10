"""Initial conditions for phase, grain, concentration, and potentials."""

import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from .config import OMEGA1_REGIONS, OMEGA2, OMEGA3
from .fields import (
    assign_component_from_expression,
    copy_component_from_array,
)
from .regions import collapse_mixed_subspace, global_inactive_only_dofs_from_markers
from .fields import load_frozen_grain_fields as _load_frozen_grain_fields
from .constitutive import eeq_from_soc_numpy


def initialize_xi_profile(X, y_top, params):
    depth = y_top - X[1]
    location = params.li_layer_thickness / params.length_scale
    width = params.xi_interface_width / params.length_scale
    if width <= 0.0:
        return np.where(depth < location, 1.0, 0.0).astype(PETSc.ScalarType)
    t = np.clip((depth - (location - 0.5 * width)) / width, 0.0, 1.0)
    smooth = t**3 * (6.0 * t**2 - 15.0 * t + 10.0)
    return (1.0 - smooth).astype(PETSc.ScalarType)


def load_frozen_grain_fields(V, gb_file):
    return _load_frozen_grain_fields(V, gb_file)


def initialize_state(msh, finite_element_state, params, stage=None):
    """Assign all initial fields and synchronize the history states."""
    ME = finite_element_state.ME
    w = finite_element_state.state
    w_n = finite_element_state.previous_state
    w_nm1 = finite_element_state.previous_previous_state
    w_nm2 = finite_element_state.older_state
    eta_initial_values = finite_element_state.eta_initial_values
    n_grains = finite_element_state.n_grains
    V_scalar = finite_element_state.V_scalar
    domain_markers = finite_element_state.domain_markers
    y_top = float(msh.comm.allreduce(float(msh.geometry.x[:, 1].max()), op=MPI.MAX))
    e_eq_init = eeq_from_soc_numpy(params.soc_init)
    phis_init_value = (
        params.li_side_potential + e_eq_init + params.initial_cathode_overpotential
        if params.phis_init is None else float(params.phis_init)
    )
    if msh.comm.rank == 0:
        print(
            f"initial potentials: phil={params.li_side_potential:.6g} V, "
            f"phis={phis_init_value:.6g} V Eeq(soc_init)={e_eq_init:.6g} V, "
            f"eta_a_guess={params.li_side_potential:.6g} V, "
            f"eta_c_guess={phis_init_value - params.li_side_potential - e_eq_init:.6g} V",
            flush=True,
        )
    if stage is not None: stage(msh.comm, "before assign xi")
    assign_component_from_expression(w, ME, 0, lambda X: initialize_xi_profile(X, y_top, params))
    if stage is not None: stage(msh.comm, "after assign xi")
    if stage is not None: stage(msh.comm, "before assign phil")
    assign_component_from_expression(
        w, ME, 1,
        lambda X: np.full(X.shape[1], params.li_side_potential, dtype=PETSc.ScalarType),
    )
    if stage is not None: stage(msh.comm, "after assign phil")
    if stage is not None: stage(msh.comm, "before assign phis")
    assign_component_from_expression(
        w, ME, 2,
        lambda X: np.full(X.shape[1], phis_init_value, dtype=PETSc.ScalarType),
    )
    if stage is not None: stage(msh.comm, "after assign phis")
    if stage is not None: stage(msh.comm, "before copy c")
    V_c, c_submap = collapse_mixed_subspace(ME, 3)
    c_values = np.full(c_submap.shape[0], PETSc.ScalarType(params.soc_init), dtype=PETSc.ScalarType)
    inactive = global_inactive_only_dofs_from_markers(
        V_c, domain_markers, OMEGA1_REGIONS + (OMEGA2,), (OMEGA3,)
    )
    if inactive.size:
        c_values[inactive] = PETSc.ScalarType(0.0)
    w.x.array[c_submap] = c_values.astype(w.x.array.dtype, copy=False)
    if stage is not None: stage(msh.comm, "after copy c")
    if stage is not None: stage(msh.comm, "before assign ux")
    assign_component_from_expression(
        w, ME, 4, lambda X: np.zeros(X.shape[1], dtype=PETSc.ScalarType)
    )
    if stage is not None: stage(msh.comm, "after assign ux")
    if stage is not None: stage(msh.comm, "before assign uy")
    assign_component_from_expression(
        w, ME, 5, lambda X: np.zeros(X.shape[1], dtype=PETSc.ScalarType)
    )
    if stage is not None: stage(msh.comm, "after assign uy")
    for i in range(n_grains):
        if stage is not None: stage(msh.comm, f"before copy eta{i + 1}")
        copy_component_from_array(w, ME, 6 + i, eta_initial_values[i])
        if stage is not None: stage(msh.comm, f"after copy eta{i + 1}")
    if stage is not None: stage(msh.comm, "before initial scatter")
    w.x.scatter_forward()
    if stage is not None: stage(msh.comm, "after initial scatter")
    for previous in (w_n, w_nm1, w_nm2):
        previous.x.array[:] = w.x.array
        previous.x.scatter_forward()
    if stage is not None: stage(msh.comm, "after old-state scatter")
    previous_state_array = w_n.x.array.copy()
    if stage is not None: stage(msh.comm, "before xi_ref collapse")
    xi_ref = w_n.sub(0).collapse()
    if stage is not None: stage(msh.comm, "after xi_ref collapse")
    finite_element_state.previous_state_array = previous_state_array
    finite_element_state.xi_ref = xi_ref
    return {
        "y_top": y_top,
        "e_eq_init": e_eq_init,
        "phis_init_value": phis_init_value,
        "previous_state_array": previous_state_array,
        "xi_ref": xi_ref,
    }


__all__ = ["initialize_xi_profile", "initialize_state", "load_frozen_grain_fields"]
