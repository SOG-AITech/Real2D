"""Physical-region markers, masks, and region-local degrees of freedom."""

import numpy as np
from petsc4py import PETSc
from dolfinx import fem
try:
    from dolfinx.plot import vtk_mesh
except ImportError:
    vtk_mesh = None



def collapse_mixed_subspace(ME, component: int):
    """Return a collapsed mixed subspace and its NumPy parent-DOF map."""
    V_sub, submap = ME.sub(component).collapse()
    return V_sub, np.asarray(submap, dtype=np.int32).reshape(-1)


def region_dofs_from_markers(V, markers, region_ids):
    region_ids = set(region_ids)
    tdim = V.mesh.topology.dim
    num_cells = V.mesh.topology.index_map(tdim).size_local + V.mesh.topology.index_map(tdim).num_ghosts
    dofmap_list = V.dofmap.list
    dofmap_array = dofmap_list.array if hasattr(dofmap_list, "array") else np.asarray(dofmap_list)
    cells = dofmap_array.reshape((num_cells, -1)).astype(np.int32)
    chosen_cells = np.flatnonzero(np.isin(markers[:num_cells], list(region_ids)))
    if chosen_cells.size == 0:
        return np.empty(0, dtype=np.int32)
    return np.unique(cells[chosen_cells].reshape(-1)).astype(np.int32)


def global_union_region_dofs_from_markers(V, markers, region_ids):
    local_region_dofs = region_dofs_from_markers(V, markers, region_ids)
    index_map = V.dofmap.index_map
    local_size = index_map.size_local + index_map.num_ghosts
    local_all = np.arange(local_size, dtype=np.int32)
    local_region_global = np.asarray(index_map.local_to_global(local_region_dofs), dtype=np.int64)
    gathered = V.mesh.comm.allgather(local_region_global)
    global_region = np.unique(np.concatenate([part for part in gathered if part.size > 0])) if gathered else np.empty(0, dtype=np.int64)
    if global_region.size == 0:
        return np.empty(0, dtype=np.int32)
    local_all_global = np.asarray(index_map.local_to_global(local_all), dtype=np.int64)
    return local_all[np.isin(local_all_global, global_region)].astype(np.int32)


def global_inactive_only_dofs_from_markers(V, markers, inactive_region_ids, active_region_ids):
    inactive = global_union_region_dofs_from_markers(V, markers, inactive_region_ids)
    active = global_union_region_dofs_from_markers(V, markers, active_region_ids)
    if inactive.size == 0:
        return inactive
    return np.setdiff1d(inactive, active, assume_unique=False).astype(np.int32)


def inactive_only_dofs_from_markers(V, markers, inactive_region_ids, active_region_ids):
    inactive = region_dofs_from_markers(V, markers, inactive_region_ids)
    if inactive.size == 0:
        return inactive
    active = region_dofs_from_markers(V, markers, active_region_ids)
    return np.setdiff1d(inactive, active, assume_unique=False).astype(np.int32)


def component_region_dofs_from_markers(ME, component, markers, region_ids):
    region_ids = set(region_ids)
    tdim = ME.mesh.topology.dim
    num_cells = ME.mesh.topology.index_map(tdim).size_local + ME.mesh.topology.index_map(tdim).num_ghosts
    chosen_cells = np.flatnonzero(np.isin(markers[:num_cells], list(region_ids))).astype(np.int32)
    if chosen_cells.size == 0:
        return np.empty(0, dtype=np.int32)
    dofmap_list = ME.sub(component).dofmap.list
    dofmap_array = dofmap_list.array if hasattr(dofmap_list, "array") else np.asarray(dofmap_list)
    cells = dofmap_array.reshape((num_cells, -1)).astype(np.int32)
    return np.unique(cells[chosen_cells].reshape(-1)).astype(np.int32)


def component_inactive_only_dofs_from_markers(ME, component, markers, inactive_region_ids, active_region_ids):
    inactive = component_region_dofs_from_markers(ME, component, markers, inactive_region_ids)
    if inactive.size == 0:
        return inactive
    active = component_region_dofs_from_markers(ME, component, markers, active_region_ids)
    return np.setdiff1d(inactive, active, assume_unique=False).astype(np.int32)


def global_component_inactive_only_dofs_from_markers(ME, component, markers, inactive_region_ids, active_region_ids):
    V_sub, submap = collapse_mixed_subspace(ME, component)
    scalar_inactive = global_inactive_only_dofs_from_markers(V_sub, markers, inactive_region_ids, active_region_ids)
    if scalar_inactive.size == 0:
        return np.empty(0, dtype=np.int32)
    return submap[scalar_inactive].astype(np.int32)


def component_dirichlet_bc_from_scalar_dofs(ME, component, scalar_dofs, value=0.0):
    if scalar_dofs.size == 0:
        return None
    _, submap = collapse_mixed_subspace(ME, component)
    mixed_dofs = submap[np.asarray(scalar_dofs, dtype=np.int32)]
    return fem.dirichletbc(PETSc.ScalarType(value), mixed_dofs, ME.sub(component))


def component_dirichlet_bc_from_dofs(ME, component, dofs, value=0.0):
    if dofs.size == 0:
        return None
    return fem.dirichletbc(PETSc.ScalarType(value), np.asarray(dofs, dtype=np.int32), ME.sub(component))


def boundary_dofs_from_tag(V, facet_tags, tag, fallback_locator=None):
    if fallback_locator is not None:
        coords = V.tabulate_dof_coordinates()[:, : V.mesh.geometry.dim]
        keep = np.asarray(fallback_locator(coords.T), dtype=bool)
        return np.flatnonzero(keep).astype(np.int32)
    fdim = V.mesh.topology.dim - 1
    facets = facet_tags.find(tag)
    if len(facets) > 0:
        return fem.locate_dofs_topological(V, fdim, facets).astype(np.int32)
    return np.empty(0, dtype=np.int32)


def cells_and_points(V, markers=None, owned_only=False):
    if vtk_mesh is not None:
        topology, _cell_types, geometry = vtk_mesh(V)
        topology = np.asarray(topology, dtype=np.int64)
        geometry = np.asarray(geometry, dtype=np.float64)
        if topology.ndim == 1:
            cells, offset = [], 0
            while offset < topology.size:
                width = int(topology[offset]); offset += 1
                cells.append(np.asarray(topology[offset:offset + width], dtype=np.int32)); offset += width
            cells = np.asarray(cells, dtype=np.int32)
        else:
            cells = np.asarray(topology, dtype=np.int32)
        points = geometry[:, :2]
        num_cells = min(cells.shape[0], len(markers)) if markers is not None else cells.shape[0]
        cells = cells[:num_cells]
        cell_markers_out = markers[:num_cells] if markers is not None else None
    else:
        msh = V.mesh; tdim = msh.topology.dim; index_map = msh.topology.index_map(tdim)
        num_cells = index_map.size_local + (0 if owned_only else index_map.num_ghosts)
        dofmap_list = V.dofmap.list
        dofmap_array = dofmap_list.array if hasattr(dofmap_list, "array") else np.asarray(dofmap_list)
        offsets = getattr(dofmap_list, "offsets", None)
        if offsets is not None:
            offsets = np.asarray(offsets, dtype=np.int64); num_cells = min(num_cells, offsets.size - 1)
            cells = [np.asarray(dofmap_array[offsets[i]:offsets[i + 1]], dtype=np.int32) for i in range(num_cells)]
        else:
            cells = [np.asarray(V.dofmap.cell_dofs(i), dtype=np.int32) for i in range(num_cells)]
        points = V.tabulate_dof_coordinates()[:, :2]
        cell_markers_out = markers[:num_cells] if markers is not None else None
    widths = {cell.size for cell in cells}
    if len(widths) != 1 or next(iter(widths)) not in (3, 4):
        raise ValueError("Unsupported plotting cell dof count.")
    cells = np.asarray(cells, dtype=np.int32)
    if cells.shape[1] == 4:
        cp = points[cells]; centers = np.mean(cp, axis=1)
        cells = np.take_along_axis(cells, np.argsort(np.arctan2(cp[:, :, 1]-centers[:, None, 1], cp[:, :, 0]-centers[:, None, 0]), axis=1), axis=1)
    return cells, points, cell_markers_out


def restricted_cells(V, cell_markers=None, valid_regions=None, owned_only=False):
    cells, points, markers = cells_and_points(V, cell_markers, owned_only)
    if valid_regions is None or markers is None:
        return cells, points
    if isinstance(valid_regions, (int, np.integer)): valid_regions = (int(valid_regions),)
    return cells[np.isin(markers, list(valid_regions))], points


def restricted_dofs(V, cell_markers=None, valid_regions=None, owned_only=False):
    cells, _points, markers = cells_and_points(V, cell_markers, owned_only)
    if valid_regions is None or markers is None:
        return np.arange(V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts)
    if isinstance(valid_regions, (int, np.integer)): valid_regions = (int(valid_regions),)
    keep = np.isin(markers, list(valid_regions))
    return np.unique(cells[keep].reshape(-1)).astype(np.int32) if np.any(keep) else np.empty(0, dtype=np.int32)


def build_subsampled_triangles(V, cell_markers=None, valid_regions=None, order=5, owned_only=False):
    cells, points = restricted_cells(V, cell_markers, valid_regions, owned_only)
    if cells.size == 0:
        return (np.empty((0, 2)), np.empty((0, 3), dtype=np.int32), np.empty((0, 3), dtype=np.int32), np.empty((0, 3)))
    triangles = cells if cells.shape[1] == 3 else np.vstack((cells[:, [0, 1, 2]], cells[:, [0, 2, 3]])).astype(np.int32)
    bary, bary_id = [], {}
    for i in range(order + 1):
        for j in range(order + 1 - i):
            k = order - i - j
            bary_id[(i, j)] = len(bary)
            bary.append((i / order, j / order, k / order))
    bary = np.asarray(bary, dtype=np.float64)
    local_tris = []
    for i in range(order):
        for j in range(order - i):
            a, b, c = bary_id[(i, j)], bary_id[(i + 1, j)], bary_id[(i, j + 1)]
            local_tris.append((a, b, c))
            if j < order - i - 1:
                local_tris.append((b, bary_id[(i + 1, j + 1)], c))
    local_tris = np.asarray(local_tris, dtype=np.int32)
    sample_points, sample_cells, sample_parent_tris, sample_bary = [], [], [], []
    offset = 0
    for tri in triangles:
        new_points = bary @ points[tri]
        sample_points.append(new_points)
        sample_cells.append(local_tris + offset)
        sample_parent_tris.append(np.repeat(tri[None, :], len(bary), axis=0))
        sample_bary.append(bary)
        offset += len(bary)
    return np.vstack(sample_points), np.vstack(sample_cells), np.vstack(sample_parent_tris), np.vstack(sample_bary)


def cell_marker_array(msh, cell_tags):
    from .fields import cell_marker_array as build_cell_marker_array

    return build_cell_marker_array(msh, cell_tags)


__all__ = [
    "collapse_mixed_subspace", "region_dofs_from_markers",
    "global_union_region_dofs_from_markers", "global_inactive_only_dofs_from_markers",
    "inactive_only_dofs_from_markers", "component_region_dofs_from_markers",
    "component_inactive_only_dofs_from_markers", "global_component_inactive_only_dofs_from_markers",
    "component_dirichlet_bc_from_scalar_dofs", "component_dirichlet_bc_from_dofs",
    "boundary_dofs_from_tag", "cells_and_points", "restricted_dofs", "restricted_cells",
    "build_subsampled_triangles", "cell_marker_array",
]
