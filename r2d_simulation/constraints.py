"""Boundary and periodic constraint construction."""

import numpy as np
from dataclasses import dataclass
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
from .regions import (collapse_mixed_subspace,
                       global_inactive_only_dofs_from_markers)
from .config import GAMMA_A, GAMMA_C, OMEGA1_REGIONS, OMEGA2, OMEGA3


@dataclass
class ConstraintState:
    bcs: list
    inactive_component_dofs: dict
    periodic_constraints: dict
    periodic_constraints_global: dict
    periodic_constraint_counts: dict
    active_dofs: np.ndarray
    eliminated_dofs: np.ndarray
    periodic_slave_dofs: np.ndarray
    x_min: float
    x_max: float
    y_min: float


def build_constraint_state(*, msh, ME, V_scalar, domain_markers, facet_tags,
                           eta_regions, n_grains, params, y_top, w, w_n,
                           w_nm1, w_nm2, stage=None):
    """Assemble all strong, periodic, inactive, and mechanical gauge constraints."""
    def log(message):
        if stage is not None:
            stage(msh.comm, message)
    bcs = []
    log("before inactive BCs")
    specs = {
        "xi_eta": ((OMEGA2, OMEGA3), OMEGA1_REGIONS),
        "eta": (tuple(tag for tag in OMEGA1_REGIONS + (OMEGA2, OMEGA3)
                       if tag not in eta_regions), eta_regions),
        "phil": ((OMEGA3,), OMEGA1_REGIONS + (OMEGA2,)),
        "phis": (OMEGA1_REGIONS + (OMEGA3,), (OMEGA2,)),
        "c": (OMEGA1_REGIONS + (OMEGA2,), (OMEGA3,)),
    }
    inactive = {}
    scalar_global = {
        key: global_inactive_only_dofs_from_markers(V_scalar, domain_markers, inc, act)
        for key, (inc, act) in specs.items() if key in ("phil", "phis", "c")
    }
    from .regions import (component_inactive_only_dofs_from_markers,
                          component_dirichlet_bc_from_dofs)
    for component, key in ((0, "xi_eta"), (1, "phil"), (2, "phis"), (3, "c")):
        inc, act = specs[key]
        if component in (1, 2, 3):
            _, submap = collapse_mixed_subspace(ME, component)
            dofs = submap[scalar_global[key]].astype(np.int32)
        else:
            dofs = component_inactive_only_dofs_from_markers(ME, component, domain_markers, inc, act)
        inactive[component] = dofs
        bc = component_dirichlet_bc_from_dofs(ME, component, dofs, value=0.0)
        if bc is not None: bcs.append(bc)
    for i in range(n_grains):
        dofs = component_inactive_only_dofs_from_markers(ME, 6 + i, domain_markers, *specs["eta"])
        inactive[6 + i] = dofs
        bc = component_dirichlet_bc_from_dofs(ME, 6 + i, dofs, value=0.0)
        if bc is not None: bcs.append(bc)
    log(f"after inactive BCs count={len(bcs)}")
    log("before top xi BC")
    if params.enforce_top_xi_bc:
        top = facet_tags.find(GAMMA_A)
        if len(top) == 0:
            top = mesh.locate_entities_boundary(msh, msh.topology.dim - 1,
                                                lambda X: np.isclose(X[1], y_top))
        Vxi, _ = collapse_mixed_subspace(ME, 0)
        dofs = fem.locate_dofs_topological((ME.sub(0), Vxi), msh.topology.dim - 1, top)
        fun = fem.Function(Vxi); fun.x.array[:] = 1.0; fun.x.scatter_forward()
        bcs.append(fem.dirichletbc(fun, dofs, ME.sub(0)))
    x_min = float(msh.comm.allreduce(float(msh.geometry.x[:, 0].min()), op=MPI.MIN))
    x_max = float(msh.comm.allreduce(float(msh.geometry.x[:, 0].max()), op=MPI.MAX))
    y_min = float(msh.comm.allreduce(float(msh.geometry.x[:, 1].min()), op=MPI.MIN))
    bottom = facet_tags.find(GAMMA_C)
    if len(bottom) == 0:
        bottom = mesh.locate_entities_boundary(msh, msh.topology.dim - 1,
                                                lambda X: np.isclose(X[1], y_min))
    Vuy, _ = collapse_mixed_subspace(ME, 5)
    duy = fem.locate_dofs_topological((ME.sub(5), Vuy), msh.topology.dim - 1, bottom)
    uy0 = fem.Function(Vuy); uy0.x.array[:] = 0.0; uy0.x.scatter_forward()
    bcs.append(fem.dirichletbc(uy0, duy, ME.sub(5)))
    log("before periodic constraints")
    periodic = {}; periodic_global = {}; counts = {}
    if params.mechanics_periodic_lr:
        if msh.comm.size > 1:
            periodic_global, counts = build_lr_periodic_constraints_global(msh, ME, n_grains)
            for fun in (w, w_n, w_nm1, w_nm2): apply_global_periodic_constraints_to_function(fun, periodic_global)
        else:
            periodic, counts = build_lr_periodic_constraints(msh, ME, n_grains)
            for s, master in periodic.items():
                for fun in (w, w_n, w_nm1, w_nm2): fun.x.array[s] = fun.x.array[master]
            for fun in (w, w_n, w_nm1, w_nm2): fun.x.scatter_forward()
    log("after periodic constraints")
    xmid = 0.5 * (x_min + x_max)
    Vux, uxmap = collapse_mixed_subspace(ME, 4)
    coords = Vux.tabulate_dof_coordinates()[:, :2]; owned = int(Vux.dofmap.index_map.size_local)
    ids = np.flatnonzero(np.isclose(coords[:owned, 1], y_min, atol=1e-10))
    local = (float("inf"), msh.comm.rank, -1) if ids.size == 0 else (float(np.min(np.abs(coords[ids, 0] - xmid))), msh.comm.rank, int(ids[np.argmin(np.abs(coords[ids, 0] - xmid))]))
    best = min(msh.comm.allgather(local), key=lambda x: (x[0], x[1]))
    dofs = np.asarray([int(uxmap[best[2]])], dtype=np.int32) if msh.comm.rank == best[1] else np.empty(0, dtype=np.int32)
    bc = component_dirichlet_bc_from_dofs(ME, 4, dofs, value=0.0)
    if bc is not None: bcs.append(bc)
    log(f"after ux gauge BC count={len(bcs)}")
    eliminated = [inactive.get(0, np.empty(0, dtype=np.int32))] + [inactive.get(6+i, np.empty(0, dtype=np.int32)) for i in range(n_grains)]
    eliminated = np.unique(np.concatenate(eliminated)).astype(np.int32) if eliminated else np.empty(0, dtype=np.int32)
    all_dofs = np.arange(int(w.x.petsc_vec.getLocalSize()), dtype=np.int32)
    eliminated_owned = eliminated[(0 <= eliminated) & (eliminated < all_dofs.size)]
    active = np.setdiff1d(all_dofs, eliminated_owned, assume_unique=False)
    slaves = np.fromiter(periodic.keys() if msh.comm.size == 1 else periodic_global.keys(), dtype=np.int64)
    return ConstraintState(bcs, inactive, periodic, periodic_global, counts, active, eliminated, slaves, x_min, x_max, y_min)


def build_lr_periodic_component_constraints(msh, ME, components, label, tol=1.0e-8):
    V, _ = collapse_mixed_subspace(ME, int(tuple(components)[0]))
    coords = V.tabulate_dof_coordinates()[:, :2]
    x_min = float(msh.comm.allreduce(float(msh.geometry.x[:, 0].min()), op=MPI.MIN))
    x_max = float(msh.comm.allreduce(float(msh.geometry.x[:, 0].max()), op=MPI.MAX))
    left = np.flatnonzero(np.isclose(coords[:, 0], x_min, atol=tol)); right = np.flatnonzero(np.isclose(coords[:, 0], x_max, atol=tol))
    if left.size != right.size: raise RuntimeError(f"Cannot build left-right periodic {label} constraints: left dofs={left.size}, right dofs={right.size}.")
    left = np.asarray(sorted(left, key=lambda i: float(coords[i, 1])), dtype=np.int32)
    right = np.asarray(sorted(right, key=lambda i: float(coords[i, 1])), dtype=np.int32)
    if np.max(np.abs(coords[left, 1] - coords[right, 1])) if left.size else 0.0 > tol:
        raise RuntimeError(f"Cannot build left-right periodic {label} constraints: paired y mismatch exceeds tol.")
    constraints = {}
    for l, r in zip(left, right):
        for component in components:
            _, submap = collapse_mixed_subspace(ME, component)
            constraints[int(submap[r])] = int(submap[l])
    return constraints


def build_lr_periodic_constraints(msh, ME, n_grains, tol=1.0e-8):
    constraints = {}; counts = {}
    for components, label in (((0,), "xi"), (tuple(range(6, 6 + n_grains)), "eta"), ((4, 5), "mechanics")):
        group = build_lr_periodic_component_constraints(msh, ME, components, label, tol)
        overlap = set(constraints).intersection(group)
        if overlap: raise RuntimeError(f"Duplicate periodic slave dofs while adding {label}: {sorted(overlap)[:5]}")
        constraints.update(group); counts[label] = len(group)
    return constraints, counts


def global_dofs_from_local(V, local_dofs):
    local_dofs = np.asarray(local_dofs, dtype=np.int64)
    index_map = V.dofmap.index_map
    block_size = int(V.dofmap.index_map_bs)
    try:
        local_start = int(index_map.local_range[0])
    except TypeError:
        local_start = int(index_map.local_range()[0])
    if block_size == 1:
        return local_start + local_dofs
    blocks = (local_dofs // block_size).astype(np.int32)
    offsets = (local_dofs % block_size).astype(np.int64)
    return (local_start + blocks.astype(np.int64)) * block_size + offsets


def build_lr_periodic_component_constraints_global(msh, ME, components, label, tol=1.0e-8):
    local_x_min = float(np.min(msh.geometry.x[:, 0]))
    local_x_max = float(np.max(msh.geometry.x[:, 0]))
    x_min = float(msh.comm.allreduce(local_x_min, op=MPI.MIN))
    x_max = float(msh.comm.allreduce(local_x_max, op=MPI.MAX))
    constraints = {}
    counts = {}
    for component in tuple(components):
        V_scalar, submap = collapse_mixed_subspace(ME, component)
        coords = V_scalar.tabulate_dof_coordinates()[:, :2]
        local_dofs = np.arange(coords.shape[0], dtype=np.int32)
        parent_local = np.asarray(submap, dtype=np.int64)[local_dofs]
        owned_size = int(ME.dofmap.index_map.size_local) * int(ME.dofmap.index_map_bs)
        owned = parent_local < owned_size
        parent_global = global_dofs_from_local(ME, parent_local[owned])
        coords = coords[owned]
        side_records = []
        for y, dof in zip(coords[np.isclose(coords[:, 0], x_min, atol=tol), 1],
                           parent_global[np.isclose(coords[:, 0], x_min, atol=tol)]):
            side_records.append(("L", float(y), int(dof)))
        for y, dof in zip(coords[np.isclose(coords[:, 0], x_max, atol=tol), 1],
                           parent_global[np.isclose(coords[:, 0], x_max, atol=tol)]):
            side_records.append(("R", float(y), int(dof)))
        gathered = [record for part in msh.comm.allgather(side_records) for record in part]
        left = sorted((record for record in gathered if record[0] == "L"), key=lambda record: record[1])
        right = sorted((record for record in gathered if record[0] == "R"), key=lambda record: record[1])
        if len(left) != len(right):
            raise RuntimeError(
                f"Cannot build MPI periodic {label} component {component}: "
                f"left dofs={len(left)}, right dofs={len(right)}."
            )
        if left:
            max_y_delta = max(abs(l[1] - r[1]) for l, r in zip(left, right))
            if max_y_delta > tol:
                raise RuntimeError(
                    f"Cannot build MPI periodic {label} component {component}: "
                    f"max paired y mismatch={max_y_delta:.3e} exceeds tol={tol:.3e}."
                )
        for left_record, right_record in zip(left, right):
            constraints[right_record[2]] = left_record[2]
        counts[int(component)] = len(right)
    if msh.comm.rank == 0:
        print(
            f"periodic global {label}: paired {sum(counts.values())} slave dofs "
            f"across {len(counts)} component(s).", flush=True
        )
    return constraints, counts


def build_lr_periodic_constraints_global(msh, ME, n_grains, tol=1.0e-8):
    constraints = {}
    counts = {}
    for components, label in (
        ((0,), "xi"),
        (tuple(range(6, 6 + n_grains)), "eta"),
        ((4, 5), "mechanics"),
    ):
        group, group_counts = build_lr_periodic_component_constraints_global(
            msh, ME, components, label, tol
        )
        overlap = set(constraints).intersection(group)
        if overlap:
            raise RuntimeError(
                f"Duplicate MPI periodic slave dofs while adding {label}: {sorted(overlap)[:5]}"
            )
        constraints.update(group)
        counts[label] = sum(group_counts.values())
    return constraints, counts


def build_lr_periodic_displacement_constraints(msh, ME, tol=1.0e-8):
    """Build backward-compatible mechanics-only left-right constraints."""
    return build_lr_periodic_component_constraints(
        msh, ME, (4, 5), "mechanics", tol=tol
    )


def apply_global_periodic_constraints_to_function(fun, constraints_global):
    constraints_global = dict(constraints_global or {})
    if not constraints_global: fun.x.scatter_forward(); return
    vec = fun.x.petsc_vec; comm = fun.function_space.mesh.comm
    global_size, local_size = int(vec.getSize()), int(vec.getLocalSize()); row_start, row_end = vec.getOwnershipRange()
    P = PETSc.Mat().createAIJ(size=((local_size, global_size), (local_size, global_size)), nnz=1, comm=comm)
    for row in range(int(row_start), int(row_end)): P.setValue(row, int(constraints_global.get(row, row)), 1.0)
    P.assemble(); src, dst = vec.duplicate(), vec.duplicate(); vec.copy(src); P.mult(src, dst); dst.copy(vec); fun.x.scatter_forward()
    src.destroy(); dst.destroy(); P.destroy()


__all__ = ["ConstraintState", "build_constraint_state",
           "build_lr_periodic_constraints", "build_lr_periodic_constraints_global",
           "build_lr_periodic_component_constraints_global", "global_dofs_from_local",
           "build_lr_periodic_displacement_constraints",
           "apply_global_periodic_constraints_to_function"]
