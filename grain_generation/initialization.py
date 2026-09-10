from __future__ import annotations

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from .config import OMEGA1
from .mesh import tagged_cell_bbox


def comsol_step_initial_xi_ufl(msh, p):
    x = ufl.SpatialCoordinate(msh)
    local_y = msh.geometry.x[:, 1]
    local_y_max = float(local_y.max()) if local_y.size else float("-inf")
    y_top = msh.comm.allreduce(local_y_max, op=MPI.MAX)
    depth = float(y_top) - x[1]
    location = p.grain_excluded_top_thickness / p.length_scale
    width = p.xi_interface_width / p.length_scale
    if width <= 0.0:
        return ufl.conditional(ufl.lt(depth, location), 1.0, 0.0)
    t = ufl.max_value(0.0, ufl.min_value(1.0, (depth - location + 0.5 * width) / width))
    return 1.0 - t**3 * (6.0 * t**2 - 15.0 * t + 10.0)


def comsol_step_initial_xi_np(y, y_top, p):
    depth = y_top - y
    location = p.grain_excluded_top_thickness / p.length_scale
    width = p.xi_interface_width / p.length_scale
    if width <= 0.0:
        return (depth < location).astype(np.float64)
    t = np.clip((depth - location + 0.5 * width) / width, 0.0, 1.0)
    return 1.0 - t**3 * (6.0 * t**2 - 15.0 * t + 10.0)


def build_columnar_grain_initializers(msh, cell_tags, p, tag=OMEGA1, y_cut=None):
    (x_min, y_min), (x_max, y_max) = tagged_cell_bbox(msh, cell_tags, tag)
    if y_cut is not None:
        y_max = min(y_max, y_cut)
    width = max(p.interface_width * (x_max - x_min), 1.0e-8)
    base = np.linspace(x_min, x_max, p.n_grains + 1)

    def make_expr(i):
        def expr(X):
            left = base[i] if i else x_min - 4.0 * width
            right = base[i + 1] if i + 1 < p.n_grains else x_max + 4.0 * width
            value = 0.5 * (1.0 + np.tanh((X[0] - left) / width))
            value *= 0.5 * (1.0 + np.tanh((right - X[0]) / width))
            return value.astype(PETSc.ScalarType)
        return expr

    return [make_expr(i) for i in range(p.n_grains)]


def build_partition_random_initializers(msh, cell_tags, p, tag=OMEGA1, y_cut=None):
    (x_min, y_min), (x_max, y_max) = tagged_cell_bbox(msh, cell_tags, tag)
    if y_cut is not None:
        y_max = min(y_max, y_cut)
    rng = np.random.default_rng(p.seed)
    count = max(int(p.partition_seed_count), p.n_grains)
    seed_x = rng.uniform(x_min, x_max, count)
    seed_y = rng.uniform(y_min, y_max, count)
    labels = rng.integers(0, p.n_grains, count)
    labels[:p.n_grains] = np.arange(p.n_grains)
    rng.shuffle(labels)
    width = max(float(p.partition_transition_width), 1.0e-8)

    def make_expr(i):
        def expr(X):
            dist = (X[0][:, None] - seed_x) ** 2 + (X[1][:, None] - seed_y) ** 2
            scores = np.full((X[0].size, p.n_grains), np.inf)
            for grain in range(p.n_grains):
                members = labels == grain
                if np.any(members):
                    scores[:, grain] = np.min(dist[:, members], axis=1)
            scores = -scores / (2.0 * width * width)
            scores -= np.max(scores, axis=1, keepdims=True)
            weights = np.exp(scores)
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-300)
            return weights[:, i].astype(PETSc.ScalarType)
        return expr

    return [make_expr(i) for i in range(p.n_grains)]


def build_comsol_random_initializers(msh, cell_tags, p, tag=OMEGA1, y_cut=None):
    return build_partition_random_initializers(msh, cell_tags, p, tag, y_cut)


def build_grain_initializers(msh, cell_tags, p, tag=OMEGA1, y_cut=None):
    if p.initializer == "partition_random":
        return build_partition_random_initializers(msh, cell_tags, p, tag, y_cut)
    if p.initializer == "comsol_random":
        return build_comsol_random_initializers(msh, cell_tags, p, tag, y_cut)
    if p.initializer == "columnar":
        return build_columnar_grain_initializers(msh, cell_tags, p, tag, y_cut)
    raise ValueError(f"Unknown initializer {p.initializer!r}")


def initialize_fields(mesh_state, field_state, params):
    from .fields import assign_component_from_expression, normalize_eta_components
    initializers = build_grain_initializers(mesh_state.parent_mesh, mesh_state.cell_tags, params, mesh_state.grain_tag)
    for i, initializer in enumerate(initializers):
        assign_component_from_expression(field_state.eta, field_state.ME, i, initializer)
    normalize_eta_components(field_state.eta, field_state.ME, params.n_grains)
    field_state.eta.x.scatter_forward()
    field_state.eta_n.x.array[:] = field_state.eta.x.array
