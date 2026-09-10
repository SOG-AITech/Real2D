"""Finite-element field construction and field transfer operations."""

import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx import fem
from mpi4py import MPI
from basix.ufl import element, mixed_element

from .config import GAMMA_A, GAMMA_C, OMEGA1, OMEGA1_REGIONS, OMEGA2, OMEGA3, ETA_REGIONS
from .state import FiniteElementState
from .regions import (
    boundary_dofs_from_tag,
    collapse_mixed_subspace,
    region_dofs_from_markers,
    cells_and_points,
)
from .mesh import resolve_here
from .constitutive import nearest_old_values


def create_scalar_space_state(msh, cell_tags, facet_tags, stage=None):
    """Create the scalar space and mesh-derived diagnostic dof maps."""
    if stage is not None:
        stage(msh.comm, "before scalar space/domain markers")
    V_scalar = fem.functionspace(msh, ("Lagrange", 1))
    domain_markers = cell_marker_array(msh, cell_tags)
    unknown_cell_markers = int(np.count_nonzero(domain_markers < 0))
    unknown_cell_markers_global = int(
        msh.comm.allreduce(unknown_cell_markers, op=MPI.SUM)
    )
    if msh.comm.rank == 0:
        print(
            "cell marker diagnostics: "
            f"unknown owned+ghost cell markers={unknown_cell_markers_global}",
            flush=True,
        )
    if stage is not None:
        stage(msh.comm, "after scalar space/domain markers")
        stage(msh.comm, "before stats_dofs")
    eta_regions = tuple(
        tag
        for tag in ETA_REGIONS
        if msh.comm.allreduce(int(np.any(cell_tags.values == tag)), op=MPI.MAX)
    ) or (OMEGA1,)
    stats_dofs = {
        "xi": region_dofs_from_markers(V_scalar, domain_markers, OMEGA1_REGIONS),
        "phil": region_dofs_from_markers(
            V_scalar, domain_markers, OMEGA1_REGIONS + (OMEGA2,)
        ),
        "phis": region_dofs_from_markers(V_scalar, domain_markers, (OMEGA2,)),
        "c": region_dofs_from_markers(V_scalar, domain_markers, (OMEGA3,)),
        "u": region_dofs_from_markers(
            V_scalar, domain_markers, OMEGA1_REGIONS + (OMEGA2, OMEGA3)
        ),
        "eta": region_dofs_from_markers(V_scalar, domain_markers, eta_regions),
    }
    from .diagnostics import region_y_extent

    cathode_y_min, cathode_y_max = region_y_extent(V_scalar, stats_dofs["c"])
    omega1_y_min, omega1_y_max = region_y_extent(V_scalar, stats_dofs["xi"])
    if stage is not None:
        stage(msh.comm, "after stats_dofs")
        stage(msh.comm, "before boundary_probe_dofs")
    y_probe_top = float(
        msh.comm.allreduce(float(msh.geometry.x[:, 1].max()), op=MPI.MAX)
    )
    y_probe_bottom = float(
        msh.comm.allreduce(float(msh.geometry.x[:, 1].min()), op=MPI.MIN)
    )
    boundary_probe_dofs = {
        "phil_gamma_a": boundary_dofs_from_tag(
            V_scalar, facet_tags, GAMMA_A,
            lambda X: np.isclose(X[1], y_probe_top),
        ),
        "phis_gamma_c": boundary_dofs_from_tag(
            V_scalar, facet_tags, GAMMA_C,
            lambda X: np.isclose(X[1], y_probe_bottom),
        ),
    }
    if stage is not None:
        stage(msh.comm, "after boundary_probe_dofs")
    return FiniteElementState(
        V_scalar=V_scalar,
        domain_markers=domain_markers,
        stats_dofs=stats_dofs,
        boundary_probe_dofs=boundary_probe_dofs,
        eta_regions=eta_regions,
        cathode_y_min=cathode_y_min,
        cathode_y_max=cathode_y_max,
        omega1_y_min=omega1_y_min,
        omega1_y_max=omega1_y_max,
        lifecycle_target_y=omega1_y_min,
    )


def create_mixed_state(msh, state, eta_initial_values, stage=None):
    """Create the shared mixed space, history Functions, and UFL components."""
    n_grains = int(np.asarray(eta_initial_values).shape[0])
    if stage is not None:
        stage(msh.comm, "before mixed space")
    P1 = element("Lagrange", msh.basix_cell(), 1)
    state.ME = fem.functionspace(msh, mixed_element([P1] * (6 + n_grains)))
    if stage is not None:
        stage(msh.comm, "after mixed space")
        stage(msh.comm, "before state functions")
    state.state = fem.Function(state.ME, name="state_xi_phil_phis_c_u_eta")
    state.previous_state = fem.Function(state.ME, name="state_old")
    state.previous_previous_state = fem.Function(state.ME, name="state_old_old")
    state.older_state = fem.Function(state.ME, name="state_old_old_old")
    if stage is not None:
        stage(msh.comm, "after state functions")
        stage(msh.comm, "before UFL split/test setup")
    state.components = ufl.split(state.state)
    state.previous_components = ufl.split(state.previous_state)
    state.previous_previous_components = ufl.split(state.previous_previous_state)
    state.tests = ufl.TestFunctions(state.ME)
    state.trial = ufl.TrialFunction(state.ME)
    state.n_grains = n_grains
    state.eta_initial_values = np.asarray(eta_initial_values)
    if stage is not None:
        stage(msh.comm, "after UFL split/test setup")
    return state


def grain_boundary_indicator_expr(etas, p, mask=1.0):
    overlap = sum(etas[i] * etas[j] for i in range(len(etas)) for j in range(i + 1, len(etas)))
    return mask * p.b_scale * ufl.min_value(p.b_clip, ufl.max_value(0.0, overlap))


def box_indicator_ufl(value, half_width, smoothing=0.0):
    if smoothing <= 0.0:
        return ufl.conditional(ufl.lt(abs(value), half_width), 1.0, 0.0)
    return 0.5 * (ufl.tanh((value + half_width) / smoothing) - ufl.tanh((value - half_width) / smoothing))


def grain_boundary_window_expr(etas, p, eta_clips=None):
    if eta_clips is None: eta_clips = etas
    return sum((1.0 - clip) * box_indicator_ufl(eta - 0.5, p.rho, p.gb_window_smoothing)
               for eta, clip in zip(etas, eta_clips))


def load_grain_arrays(V, gb_file):
    etas, B = load_frozen_grain_fields(V, gb_file)
    return np.vstack([eta.x.array.real.copy() for eta in etas]), B.x.array.real.copy()


def load_frozen_grain_fields(V, gb_file):
    data = np.load(resolve_here(gb_file), allow_pickle=True)
    eta_values, b_values = data["eta"], data["B"]
    ce_values = None
    if "extra" in data and "names" in data:
        names = [str(name) for name in data["names"]]
        if "ce" in names:
            ce_index = names.index("ce") - int(eta_values.shape[0]) - 1
            if 0 <= ce_index < data["extra"].shape[0]: ce_values = np.asarray(data["extra"][ce_index], dtype=np.float64)
    ndofs = V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts
    if b_values.shape[0] != ndofs:
        if "dof_coordinates" not in data:
            raise ValueError(f"GB npz has {b_values.shape[0]} dofs, but scalar space has {ndofs}, and the npz has no coordinates for remapping.")
        old_coords = np.asarray(data["dof_coordinates"], dtype=np.float64)
        new_coords = V.tabulate_dof_coordinates()[:, :2]
        new_top = new_coords[:, 1] > float(np.max(old_coords[:, 1])) + 1.0e-8
        default_eta = np.zeros(eta_values.shape[0], dtype=np.float64); default_eta[0] = 1.0
        eta_values = nearest_old_values(old_coords, eta_values, new_coords, default_eta)
        b_values = nearest_old_values(old_coords, b_values, new_coords, np.array(0.0))
        if ce_values is not None: ce_values = nearest_old_values(old_coords, ce_values, new_coords, np.array(1.0))
        eta_values[:, new_top] = default_eta[:, None]; b_values[new_top] = 0.0
        if ce_values is not None: ce_values[new_top] = 1.0
        if V.mesh.comm.rank == 0:
            print(f"Remapped frozen GB fields from {old_coords.shape[0]} old dofs to {ndofs} new dofs; {int(np.count_nonzero(new_top))} top-cap dofs set to B=0.")
    etas = []
    for i in range(eta_values.shape[0]):
        eta = fem.Function(V, name=f"eta{i + 1}"); eta.x.array[:] = eta_values[i].astype(eta.x.array.dtype); eta.x.scatter_forward(); etas.append(eta)
    B = fem.Function(V, name="B"); B.x.array[:] = b_values.astype(B.x.array.dtype); B.x.scatter_forward()
    return etas, B


def cell_marker_array(msh, cell_tags):
    tdim = msh.topology.dim
    index_map = msh.topology.index_map(tdim)
    num_owned = index_map.size_local
    num_cells = num_owned + index_map.num_ghosts
    V0 = fem.functionspace(msh, ("DG", 0))
    tag_fun = fem.Function(V0, name="cell_marker_array_tags")
    tag_fun.x.array[:] = -1.0
    dm = V0.dofmap.list
    arr = dm.array if hasattr(dm, "array") else np.asarray(dm)
    cell_dofs = arr.reshape((num_cells, -1))[:, 0].astype(np.int32)
    for cell, tag in zip(np.asarray(cell_tags.indices, dtype=np.int32), np.asarray(cell_tags.values, dtype=np.int32)):
        if 0 <= int(cell) < num_owned:
            tag_fun.x.array[cell_dofs[int(cell)]] = float(tag)
    tag_fun.x.scatter_forward()
    return np.rint(tag_fun.x.array[cell_dofs]).astype(np.int32)


def assign_component_from_expression(w, ME, component, expr):
    V_sub, submap = collapse_mixed_subspace(ME, component)
    fun = fem.Function(V_sub); fun.interpolate(expr); fun.x.scatter_forward()
    w.x.array[submap] = fun.x.array; w.x.scatter_forward()


def copy_component_from_array(w, ME, component, values):
    _, submap = collapse_mixed_subspace(ME, component)
    if values.shape[0] != submap.shape[0]:
        raise ValueError(f"Component {component} has {submap.shape[0]} dofs, but input has {values.shape[0]}.")
    w.x.array[submap] = values.astype(w.x.array.dtype); w.x.scatter_forward()


def assign_component_on_regions(w, ME, component, V_scalar, markers, region_ids, value):
    _, submap = collapse_mixed_subspace(ME, component)
    dofs = region_dofs_from_markers(V_scalar, markers, region_ids)
    if dofs.size: w.x.array[submap[dofs]] = PETSc.ScalarType(value); w.x.scatter_forward()


def clip_mixed_components(w, ME, components, lower=0.0, upper=1.0):
    for component in components:
        _, submap = collapse_mixed_subspace(ME, component)
        w.x.array[submap] = np.clip(w.x.array.real[submap], lower, upper).astype(w.x.array.dtype)
    w.x.scatter_forward()


def update_scalar_outputs(w, scalar_outputs, p=None, indices=None, component_maps=None):
    for i in (range(len(scalar_outputs)) if indices is None else indices):
        values = w.x.array[component_maps[i]] if component_maps is not None else w.sub(i).collapse().x.array
        scalar_outputs[i].x.array[:] = values.astype(scalar_outputs[i].x.array.dtype, copy=False)
        scalar_outputs[i].x.scatter_forward()


def update_interpolated(out, expr):
    points = out.function_space.element.interpolation_points
    if callable(points): points = points()
    out.interpolate(fem.Expression(expr, points)); out.x.scatter_forward()


def dg0_cell_dofs(V0):
    dm = V0.dofmap.list; arr = dm.array if hasattr(dm, "array") else np.asarray(dm)
    return arr.reshape((-1, 1))[:V0.mesh.topology.index_map(V0.mesh.topology.dim).size_local, 0].astype(np.int32)


def mask_dg0_to_regions(fun, cell_markers, valid_regions, fill_value=np.nan):
    if cell_markers is None: return
    if isinstance(valid_regions, (int, np.integer)): valid_regions = (int(valid_regions),)
    dofs = dg0_cell_dofs(fun.function_space); keep = np.isin(np.asarray(cell_markers[:dofs.size]), tuple(valid_regions))
    fun.x.array[dofs[~keep]] = PETSc.ScalarType(fill_value); fun.x.scatter_forward()


def copy_dg0_on_regions(target, source, cell_markers, valid_regions):
    if cell_markers is None: return
    if isinstance(valid_regions, (int, np.integer)): valid_regions = (int(valid_regions),)
    dofs = dg0_cell_dofs(target.function_space); keep = np.isin(np.asarray(cell_markers[:dofs.size]), tuple(valid_regions))
    target.x.array[dofs[keep]] = source.x.array[dofs[keep]]; target.x.scatter_forward()


def recover_dg0_to_p1(target, source_dg0, cell_markers, valid_regions):
    if cell_markers is None:
        target.x.array[:] = PETSc.ScalarType(np.nan); target.x.scatter_forward(); return
    if isinstance(valid_regions, (int, np.integer)): valid_regions = (int(valid_regions),)
    cells, _points, markers = cells_and_points(target.function_space, cell_markers)
    source_dofs = dg0_cell_dofs(source_dg0.function_space)
    sums = np.zeros(target.x.array.shape[0]); counts = np.zeros(target.x.array.shape[0])
    for idx in np.flatnonzero(np.isin(markers[:source_dofs.size], tuple(valid_regions))):
        value = source_dg0.x.array.real[source_dofs[idx]]
        if np.isfinite(value): sums[cells[idx]] += value; counts[cells[idx]] += 1.0
    values = np.full(target.x.array.shape[0], np.nan); valid = counts > 0; values[valid] = sums[valid] / counts[valid]
    target.x.array[:] = values.astype(target.x.array.dtype); target.x.scatter_forward()


__all__ = ["create_scalar_space_state", "create_mixed_state",
           "load_grain_arrays", "load_frozen_grain_fields", "copy_component_from_array",
           "grain_boundary_indicator_expr", "box_indicator_ufl", "grain_boundary_window_expr",
           "cell_marker_array",
           "assign_component_from_expression", "assign_component_on_regions", "update_scalar_outputs",
           "update_interpolated", "clip_mixed_components", "dg0_cell_dofs", "mask_dg0_to_regions",
           "copy_dg0_on_regions", "recover_dg0_to_p1"]
