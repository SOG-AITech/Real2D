"""Mixed eta fields, scalar outputs, and region-local field operations."""

import numpy as np
import ufl
from dolfinx import fem
from basix.ufl import element, mixed_element
from .state import FieldState


def create_fields(mesh_state, params):
    msh = mesh_state.grain_mesh
    P1 = element("Lagrange", msh.basix_cell(), 1)
    ME = fem.functionspace(msh, mixed_element([P1] * params.n_grains))
    V_scalar = fem.functionspace(msh, ("Lagrange", 1))
    eta = fem.Function(ME, name="eta")
    eta_n = fem.Function(ME, name="eta_old")
    eta_outputs = split_to_scalar_functions(eta, V_scalar, [f"eta{i + 1}" for i in range(params.n_grains)])
    return FieldState(ME, V_scalar, eta, eta_n, eta_outputs)


def normalize_eta_components(w, ME, n_grains):
    collapsed = []
    maps = []
    for i in range(n_grains):
        _, submap = ME.sub(i).collapse()
        collapsed.append(w.x.array[submap].copy())
        maps.append(submap)
    total = np.sum(np.vstack(collapsed), axis=0)
    total[total < 1.0e-12] = 1.0
    for values, submap in zip(collapsed, maps):
        w.x.array[submap] = values / total


def clip_eta_components(w):
    w.x.array[:] = np.clip(w.x.array.real, 0.0, 1.0).astype(w.x.array.dtype)
    w.x.scatter_forward()


def set_top_cap_single_grain(w, ME, p, y_cut):
    for i in range(p.n_grains):
        Vc, submap = ME.sub(i).collapse()
        coords = Vc.tabulate_dof_coordinates()[:, :2]
        cap = coords[:, 1] > y_cut
        if np.any(cap):
            values = w.x.array[submap].copy()
            values[cap] = 1.0 if i == 0 else 0.0
            w.x.array[submap] = values
    w.x.scatter_forward()


def set_region_single_grain(w, ME, cell_tags, tag, p):
    tdim = ME.mesh.topology.dim
    cells = cell_tags.find(tag)
    if len(cells) == 0:
        return
    for i in range(p.n_grains):
        Vc, submap = ME.sub(i).collapse()
        dofs = fem.locate_dofs_topological(Vc, tdim, cells)
        if len(dofs) == 0:
            continue
        values = w.x.array[submap].copy()
        values[dofs] = 1.0 if i == 0 else 0.0
        w.x.array[submap] = values
    w.x.scatter_forward()


def assign_component_from_expression(w, ME, component, expr):
    Vc, submap = ME.sub(component).collapse()
    f = fem.Function(Vc)
    f.interpolate(expr)
    w.x.array[submap] = f.x.array


def split_to_scalar_functions(w, V_out, names):
    scalar_outputs = []
    points = V_out.element.interpolation_points
    if callable(points):
        points = points()
    components = ufl.split(w)
    for i, name in enumerate(names):
        out = fem.Function(V_out, name=name)
        out.interpolate(fem.Expression(components[i], points))
        out.x.scatter_forward()
        scalar_outputs.append(out)
    return scalar_outputs


def update_scalar_functions(w, V_out, scalar_outputs):
    points = V_out.element.interpolation_points
    if callable(points):
        points = points()
    components = ufl.split(w)
    for i, out in enumerate(scalar_outputs):
        out.interpolate(fem.Expression(components[i], points))
        out.x.scatter_forward()


def clamp_scalar_outputs(functions):
    for fun in functions:
        fun.x.array[:] = np.clip(fun.x.array.real, 0.0, 1.0).astype(fun.x.array.dtype)
        fun.x.scatter_forward()


def set_top_cap_scalar_outputs(eta_outputs, p, y_cut):
    for i, fun in enumerate(eta_outputs):
        coords = fun.function_space.tabulate_dof_coordinates()[:, :2]
        cap = coords[:, 1] > y_cut
        if np.any(cap):
            fun.x.array[cap] = 1.0 if i == 0 else 0.0
            fun.x.scatter_forward()


def set_region_scalar_outputs(eta_outputs, cell_tags, tag, p):
    tdim = eta_outputs[0].function_space.mesh.topology.dim
    cells = cell_tags.find(tag)
    if len(cells) == 0:
        return
    for i, fun in enumerate(eta_outputs):
        dofs = fem.locate_dofs_topological(fun.function_space, tdim, cells)
        if len(dofs) == 0:
            continue
        fun.x.array[dofs] = 1.0 if i == 0 else 0.0
        fun.x.scatter_forward()


def zero_region_scalar(fun, cell_tags, tag):
    tdim = fun.function_space.mesh.topology.dim
    cells = cell_tags.find(tag)
    if len(cells) == 0:
        return
    dofs = fem.locate_dofs_topological(fun.function_space, tdim, cells)
    if len(dofs) > 0:
        fun.x.array[dofs] = 0.0
        fun.x.scatter_forward()


def region_dofs(V, cell_tags, tags):
    if isinstance(tags, (int, np.integer)):
        tags = (int(tags),)
    tdim = V.mesh.topology.dim
    pieces = []
    for tag in tags:
        cells = cell_tags.find(int(tag))
        if len(cells) > 0:
            pieces.append(fem.locate_dofs_topological(V, tdim, cells))
    if not pieces:
        return np.empty(0, dtype=np.int32)
    return np.unique(np.concatenate(pieces)).astype(np.int32)


def region_only_dofs(V, cell_tags, include_tags, exclude_tags):
    include = region_dofs(V, cell_tags, include_tags)
    exclude = region_dofs(V, cell_tags, exclude_tags)
    if include.size == 0 or exclude.size == 0:
        return include
    return np.setdiff1d(include, exclude, assume_unique=False).astype(np.int32)


def zero_region_only_scalar(fun, cell_tags, include_tags, exclude_tags):
    dofs = region_only_dofs(fun.function_space, cell_tags, include_tags, exclude_tags)
    if len(dofs) > 0:
        fun.x.array[dofs] = 0.0
        fun.x.scatter_forward()


def zero_cap_only_eta_outputs(eta_outputs, cell_tags, cap_tag, grain_tag):
    if cap_tag is None:
        return
    for fun in eta_outputs:
        zero_region_only_scalar(fun, cell_tags, (cap_tag,), grain_tag)


__all__ = [name for name in globals() if not name.startswith("_")]

__all__ = [
    "assign_component_from_expression", "clamp_scalar_outputs", "clip_eta_components",
    "normalize_eta_components", "region_dofs", "region_only_dofs",
    "set_region_single_grain", "set_top_cap_single_grain", "split_to_scalar_functions",
    "update_scalar_functions", "zero_cap_only_eta_outputs", "zero_region_only_scalar",
    "zero_region_scalar",
]
