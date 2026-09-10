"""Mesh loading and measure construction."""

from pathlib import Path

from mpi4py import MPI
from dolfinx import mesh
from dolfinx.io import gmsh


def _stage(comm, message):
    import os

    verbose = os.environ.get("R2D_STAGE_LOGS", "").lower() in ("1", "true", "yes", "on")
    if verbose or "Newton" in message or "problem.solve" in message:
        print(f"[rank {comm.rank}/{comm.size}] {message}", flush=True)


def resolve_here(filename):
    path = Path(filename)
    if path.is_absolute() or path.exists():
        return path
    return Path(__file__).resolve().parent.parent / path


def read_mesh(msh_file, params):
    partitioner = mesh.create_cell_partitioner(
        mesh.GhostMode.shared_facet,
        max_facet_to_cell_links=2,
    )
    _stage(MPI.COMM_WORLD, "read_mesh: before gmsh.read_from_msh")
    mesh_data = gmsh.read_from_msh(
        resolve_here(msh_file), MPI.COMM_WORLD, rank=0, gdim=2, partitioner=partitioner
    )
    _stage(MPI.COMM_WORLD, "read_mesh: after gmsh.read_from_msh")
    msh = mesh_data.mesh
    msh.geometry.x[:, : msh.geometry.dim] /= params.length_scale
    _stage(msh.comm, "read_mesh: before connectivity tdim->0")
    msh.topology.create_connectivity(msh.topology.dim, 0)
    _stage(msh.comm, "read_mesh: before connectivity facet->cell")
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    _stage(msh.comm, "read_mesh: before connectivity cell->facet")
    msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim - 1)
    _stage(msh.comm, "read_mesh: done")
    return msh, mesh_data.cell_tags, mesh_data.facet_tags


def prepare_mesh(msh_file, params):
    return read_mesh(msh_file, params)


def dx_regions(dx_measure, tags):
    tags = tuple(tags)
    if not tags:
        raise ValueError("dx_regions requires at least one physical region tag.")
    result = dx_measure(tags[0])
    for tag in tags[1:]:
        result += dx_measure(tag)
    return result


__all__ = ["resolve_here", "read_mesh", "prepare_mesh", "dx_regions"]
