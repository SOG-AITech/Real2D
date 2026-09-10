"""Construction of mesh and finite-element state for an R2D run."""

import ufl
from mpi4py import MPI

from .config import OMEGA1_REGIONS
from .mesh import dx_regions, read_mesh
from .fields import create_scalar_space_state, create_mixed_state, load_grain_arrays
from .output import cell_tag_dg0_function
from .state import MeshState, R2DContext


def build_mesh_and_fields(*, msh_file, gb_file, params, mpi_stage_print):
    """Read the mesh and construct all shared FEM state objects."""
    mpi_stage_print(MPI.COMM_WORLD, "before read_mesh")
    msh, cell_tags, facet_tags = read_mesh(msh_file, params)
    mpi_stage_print(msh.comm, "after read_mesh")
    context = R2DContext(params=params,
        mesh=MeshState(mesh=msh, cell_tags=cell_tags, facet_tags=facet_tags))
    dx = ufl.Measure("dx", domain=msh, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    dS = ufl.Measure("dS", domain=msh, subdomain_data=facet_tags)
    context.mesh.dx, context.mesh.ds, context.mesh.dS = dx, ds, dS
    mpi_stage_print(msh.comm, "before cell_tag_dg0_function")
    cell_region_tag = cell_tag_dg0_function(msh, cell_tags)
    mpi_stage_print(msh.comm, "after cell_tag_dg0_function")
    dx_omega1 = dx_regions(dx, OMEGA1_REGIONS)
    fe = create_scalar_space_state(msh, cell_tags, facet_tags, stage=mpi_stage_print)
    context.fields.finite_element = fe
    context.mesh.cell_markers = fe.domain_markers
    context.fields.V_scalar = fe.V_scalar
    context.fields.fields.update({"stats_dofs": fe.stats_dofs,
        "boundary_probe_dofs": fe.boundary_probe_dofs})
    mpi_stage_print(msh.comm, "before load_grain_arrays")
    eta_initial_values, _ = load_grain_arrays(fe.V_scalar, gb_file)
    n_grains = int(eta_initial_values.shape[0])
    mpi_stage_print(msh.comm, f"after load_grain_arrays n_grains={n_grains}")
    mpi_stage_print(msh.comm, "before mixed space")
    create_mixed_state(msh, fe, eta_initial_values, stage=mpi_stage_print)
    context.fields.ME = fe.ME
    context.fields.fields.update({"state": fe.state, "previous_state": fe.previous_state,
        "previous_previous_state": fe.previous_previous_state, "older_state": fe.older_state,
        "eta_initial_values": fe.eta_initial_values})
    mpi_stage_print(msh.comm, "after mixed space")
    return {"context": context, "msh": msh, "cell_tags": cell_tags, "facet_tags": facet_tags,
        "dx": dx, "ds": ds, "dS": dS, "dx_omega1": dx_omega1,
        "cell_region_tag": cell_region_tag, "fe": fe, "V_scalar": fe.V_scalar,
        "domain_markers": fe.domain_markers, "stats_dofs": fe.stats_dofs,
        "boundary_probe_dofs": fe.boundary_probe_dofs, "eta_regions": fe.eta_regions,
        "dx_eta": dx_regions(dx, fe.eta_regions), "n_grains": n_grains,
        "eta_initial_values": eta_initial_values, "ME": fe.ME,
        "w": fe.state, "w_n": fe.previous_state, "w_nm1": fe.previous_previous_state,
        "w_nm2": fe.older_state}


__all__ = ["build_mesh_and_fields"]
