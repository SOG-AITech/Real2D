"""Mesh loading, physical-tag handling, and grain-domain helpers."""

from pathlib import Path
import numpy as np
from mpi4py import MPI
from dolfinx import mesh
from dolfinx.io import gmsh

from .config import OMEGA1, OMEGA11, OMEGA12
from .state import MeshState


def resolve_msh_file(msh_file):
    path = Path(msh_file)
    if path.is_absolute() or path.exists():
        return path
    return Path(__file__).resolve().parent.parent / path


def read_mesh(msh_file, p):
    try:
        partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet, 2)
    except TypeError:
        partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    mesh_data = gmsh.read_from_msh(resolve_msh_file(msh_file), MPI.COMM_WORLD,
                                   rank=0, gdim=2, partitioner=partitioner)
    msh = mesh_data.mesh
    msh.geometry.x[:, :msh.geometry.dim] /= p.length_scale
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim - 1)
    msh.topology.create_connectivity(msh.topology.dim, 0)
    return msh, mesh_data.cell_tags, mesh_data.facet_tags


def print_tag_report(comm, cell_tags, facet_tags):
    if comm.rank == 0:
        print("Cell tags in mesh:", sorted(set(cell_tags.values.tolist())))
        print("Facet tags in mesh:", sorted(set(facet_tags.values.tolist())))


def has_cell_tag(cell_tags, tag):
    """Return whether *tag* exists on any MPI rank.

    Gmsh imports distribute cells, so a physical region can be absent from
    the local ``MeshTags`` on an otherwise valid rank.  Region selection must
    therefore be collective.
    """
    local = int(tag) in set(int(value) for value in cell_tags.values.tolist())
    return bool(MPI.COMM_WORLD.allreduce(local, op=MPI.LOR))


def omega1_tags(cell_tags):
    return (OMEGA11, OMEGA12) if has_cell_tag(cell_tags, OMEGA11) and has_cell_tag(cell_tags, OMEGA12) else (OMEGA1,)


def grain_region_tag(cell_tags):
    return OMEGA12 if has_cell_tag(cell_tags, OMEGA12) else OMEGA1


def li_cap_region_tag(cell_tags):
    return OMEGA11 if has_cell_tag(cell_tags, OMEGA11) else None


def dx_tags(dx, tags):
    tags = tuple(tags)
    if not tags:
        raise ValueError("dx_tags requires at least one physical cell tag.")
    expr = dx(tags[0])
    for tag in tags[1:]:
        expr = expr + dx(tag)
    return expr


def tagged_cell_bbox(msh, cell_tags, tag):
    if isinstance(tag, tuple):
        boxes = [tagged_cell_bbox(msh, cell_tags, item) for item in tag]
        return np.min([item[0] for item in boxes], axis=0), np.max([item[1] for item in boxes], axis=0)
    cells = cell_tags.find(tag)
    c_to_v = msh.topology.connectivity(msh.topology.dim, 0)
    if len(cells):
        vertices = np.unique(np.hstack([c_to_v.links(int(cell)) for cell in cells]))
        coords = msh.geometry.x[vertices, :2]
        local_min = coords.min(axis=0)
        local_max = coords.max(axis=0)
        has_local = True
    else:
        local_min = np.full(2, np.inf, dtype=float)
        local_max = np.full(2, -np.inf, dtype=float)
        has_local = False

    comm = msh.comm
    global_has = comm.allreduce(has_local, op=MPI.LOR)
    if not global_has:
        # Some Gmsh exports expose the physical region through a different
        # entity tag after partitioning.  Keep the original initializer
        # usable by falling back to the parent mesh extent instead of making
        # a rank-local tag lookup fatal.
        coords = np.asarray(msh.geometry.x[:, :2], dtype=float)
        local_min = coords.min(axis=0)
        local_max = coords.max(axis=0)
    # ``Comm.allreduce`` with ``MPI.MIN/MAX`` may dispatch NumPy arrays
    # through mpi4py's Python-object path on some MPI builds.  Gather the
    # two-element bounds and reduce explicitly to keep this portable.
    mins = np.asarray(comm.allgather(local_min), dtype=float)
    maxs = np.asarray(comm.allgather(local_max), dtype=float)
    return mins.min(axis=0), maxs.max(axis=0)


def grain_y_cut(msh, p):
    local_y = msh.geometry.x[:, 1]
    local_max = float(local_y.max()) if local_y.size else float("-inf")
    y_top = msh.comm.allreduce(local_max, op=MPI.MAX)
    return float(y_top) - p.grain_excluded_top_thickness / p.length_scale


def prepare_mesh(msh_file, params):
    """Load the parent mesh and return the mesh portion shared by modules."""
    parent, cell_tags, facet_tags = read_mesh(msh_file, params)
    grain_tag = grain_region_tag(cell_tags)
    # Physical tags are distributed with the cells.  Resolve the selected
    # region against the global tag set before creating the submesh.
    local_tags = sorted(set(int(value) for value in cell_tags.values.tolist()))
    global_tags = sorted({
        int(value)
        for rank_tags in parent.comm.allgather(local_tags)
        for value in rank_tags
    })
    if not has_cell_tag(cell_tags, grain_tag):
        for candidate in (OMEGA12, OMEGA1, OMEGA11, 2, 3):
            if candidate in global_tags:
                grain_tag = candidate
                break
        else:
            raise ValueError(f"No usable grain cell tag; available tags={global_tags}")
    cap_tag = li_cap_region_tag(cell_tags)
    tdim = parent.topology.dim
    cells = np.unique(np.concatenate([cell_tags.find(int(grain_tag))])).astype(np.int32)
    owned = int(parent.topology.index_map(tdim).size_local)
    cells = cells[cells < owned]
    grain_mesh = mesh.create_submesh(parent, tdim, cells)[0]
    grain_mesh.topology.create_connectivity(tdim - 1, tdim)
    grain_mesh.topology.create_connectivity(tdim, tdim - 1)
    grain_mesh.topology.create_connectivity(tdim, 0)
    import ufl
    dx = ufl.Measure("dx", domain=grain_mesh)
    return MeshState(parent, grain_mesh, cell_tags, facet_tags, dx, grain_tag, cap_tag)

__all__ = [
    "dx_tags", "grain_region_tag", "grain_y_cut", "has_cell_tag",
    "li_cap_region_tag", "omega1_tags", "print_tag_report", "read_mesh",
    "resolve_msh_file", "tagged_cell_bbox",
]
