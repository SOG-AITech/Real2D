from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri


    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    mtri = None
    HAS_MATPLOTLIB = False
import numpy as np
import ufl
from basix.ufl import element, mixed_element
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_matrix,
    create_vector,
    set_bc,
)
from dolfinx.io import gmsh


# Gmsh physical tags.  New initialization meshes may split the old Omega1 into
# a top initial Li cap and a lower initial grain region.  The production R2D
# solve should later merge those two tags back into one physical Omega1.
OMEGA1 = 1  # Legacy Li metal / solid electrolyte phase-field region
OMEGA11 = 101  # Initialization-only top Li cap
OMEGA12 = 102  # Initialization-only grain-generation region
OMEGA2 = 2  # SE-carbon mixture region
OMEGA3 = 3  # Cathode particle region

DEFAULT_MSH_FILE = "r2d_irregular_3_34.msh"


@dataclass(frozen=True)
class GrainParams:
    # Number of solid electrolyte grains represented by order parameters eta_i.
    n_grains: int = 10

    # Geometry scale. Mesh coordinates are nondimensionalized as x_hat = x / L.
    length_scale: float = 50.0e-6  # [m]

    # Allen-Cahn annealing parameters from the COMSOL/PRL setup.
    L_phi: float = 1.5e-8  # [m*s/kg], grain phase-field mobility
    W_phi: float = 1.025e7  # [J/m^3], multiwell barrier height
    k_phi: float = 12.8e-7  # [J/m], gradient coefficient

    # Time integration. These are physical seconds before nondimensionalization.
    time_scale: float = 1.0  # [s]
    dt: float = 10.0  # [s], matches COMSOL range(0,10,3600)
    t_end: float = 3000.0  # [s], full COMSOL-like grain annealing time

    # Initial columnar grain seeds.
    seed: int = 32
    initializer: str = "partition_random"  # "partition_random", "comsol_random", or "columnar"
    grain_excluded_top_thickness: float = 5.0e-6  # [m], initial Li cap above SE grains.
    xi_interface_width: float = 1.78e-6  # [m], COMSOL step1 smoothing width 2*dx with dx=8.9e-7 m.
    interface_width: float = 0.015  # nondimensional tanh transition width
    boundary_wiggle: float = 0.025  # nondimensional horizontal waviness
    boundary_drift: float = 0.035  # nondimensional linear tilt amplitude
    comsol_random_grid: int = 128  # smooth value-noise lattice used to mimic COMSOL rn_i
    partition_seed_count: int = 55
    partition_transition_width: float = 0.018

    # Weakly constrain eta fields outside Omega1 because this debug script uses
    # full-domain mixed spaces while the grain equations only live in Omega1.
    inactive_penalty: float = 1.0e-8
    clip_eta_after_solve: bool = True

    # Clamp only for post-processing B. The PDE itself remains smooth.
    b_clip: float = 0.2
    b_scale: float = 5.0
    rho: float = 0.2
    gb_window_smoothing: float = 0.02
    preview_interval: float = 500.0  # [s], save grain-boundary PNG previews during annealing.


def resolve_msh_file(msh_file: str | Path) -> Path:
    path = Path(msh_file)
    if path.is_absolute() or path.exists():
        return path
    return Path(__file__).resolve().parent / path


def read_mesh(msh_file: str | Path, p: GrainParams):
    # Explicit shared-facet partitioning is required for reliable MPI mesh
    # distribution (the lifecycle solver uses the same setting).
    try:
        partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet, 2)
    except TypeError:
        # Compatibility with releases where the second argument was not yet
        # exposed. Shared-facet ghosts are essential here: without them each
        # MPI rank drops cells along its partition boundary, which appears as
        # the white diagonal seams in the gathered PNGs.
        partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    mesh_data = gmsh.read_from_msh(
        resolve_msh_file(msh_file),
        MPI.COMM_WORLD,
        rank=0,
        gdim=2,
        partitioner=partitioner,
    )
    msh = mesh_data.mesh
    msh.geometry.x[:, : msh.geometry.dim] /= p.length_scale
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim - 1)
    msh.topology.create_connectivity(msh.topology.dim, 0)
    return msh, cell_tags, facet_tags


def print_tag_report(comm, cell_tags, facet_tags):
    if comm.rank != 0:
        return
    print("Cell tags in mesh:", sorted(set(cell_tags.values.tolist())))
    print("Facet tags in mesh:", sorted(set(facet_tags.values.tolist())))


def has_cell_tag(cell_tags, tag: int) -> bool:
    return tag in set(cell_tags.values.tolist())


def omega1_tags(cell_tags):
    """Tags that together form the original Omega1 phase-field domain."""
    if has_cell_tag(cell_tags, OMEGA11) and has_cell_tag(cell_tags, OMEGA12):
        return (OMEGA11, OMEGA12)
    return (OMEGA1,)


def grain_region_tag(cell_tags):
    """Region used to generate SE grains before the R2D solve starts."""
    return OMEGA12 if has_cell_tag(cell_tags, OMEGA12) else OMEGA1


def li_cap_region_tag(cell_tags):
    """Initialization-only Li cap, if the mesh explicitly provides it."""
    return OMEGA11 if has_cell_tag(cell_tags, OMEGA11) else None


def dx_tags(dx, tags):
    tags = tuple(tags)
    if not tags:
        raise ValueError("dx_tags requires at least one physical cell tag.")
    expr = dx(tags[0])
    for tag in tags[1:]:
        expr = expr + dx(tag)
    return expr


def tagged_cell_bbox(msh, cell_tags, tag: int):
    """Bounding box of one physical cell region."""
    if isinstance(tag, tuple):
        mins = []
        maxs = []
        for one_tag in tag:
            mn, mx = tagged_cell_bbox(msh, cell_tags, int(one_tag))
            mins.append(mn)
            maxs.append(mx)
        return np.min(np.vstack(mins), axis=0), np.max(np.vstack(maxs), axis=0)

    tdim = msh.topology.dim
    c_to_v = msh.topology.connectivity(tdim, 0)
    cells = cell_tags.find(tag)
    if len(cells) == 0:
        raise ValueError(f"No cells found for physical tag {tag}.")

    vertices = np.unique(np.hstack([c_to_v.links(int(cell)) for cell in cells]))
    coords = msh.geometry.x[vertices, :2]
    return coords.min(axis=0), coords.max(axis=0)


def grain_y_cut(msh, p: GrainParams):
    """Top of the grain-generation window, excluding the added Li cap."""
    return float(msh.geometry.x[:, 1].max()) - p.grain_excluded_top_thickness / p.length_scale


def comsol_step_initial_xi_ufl(msh, p: GrainParams):
    """COMSOL step1(H-y): from 1 to 0, location=5 um, smooth=2*dx."""
    x = ufl.SpatialCoordinate(msh)
    y_top = float(msh.geometry.x[:, 1].max())
    depth = y_top - x[1]
    location = p.grain_excluded_top_thickness / p.length_scale
    width = p.xi_interface_width / p.length_scale
    if width <= 0.0:
        return ufl.conditional(ufl.lt(depth, location), 1.0, 0.0)
    t = ufl.max_value(
        0.0,
        ufl.min_value(1.0, (depth - (location - 0.5 * width)) / width),
    )
    smooth = t**3 * (6.0 * t**2 - 15.0 * t + 10.0)
    return 1.0 - smooth


def comsol_step_initial_xi_np(y, y_top, p: GrainParams):
    """Numpy version of COMSOL step1(H-y) for sampled PNG previews."""
    depth = y_top - y
    location = p.grain_excluded_top_thickness / p.length_scale
    width = p.xi_interface_width / p.length_scale
    if width <= 0.0:
        return (depth < location).astype(np.float64)
    t = np.clip((depth - (location - 0.5 * width)) / width, 0.0, 1.0)
    smooth = t**3 * (6.0 * t**2 - 15.0 * t + 10.0)
    return 1.0 - smooth


def build_columnar_grain_initializers(
    msh, cell_tags, p: GrainParams, tag: int = OMEGA1, y_cut: float | None = None
):
    """Create COMSOL-like columnar eta_i initializers.

    The earlier Voronoi initializer creates radial/fan-shaped domains. The PRL
    figure and the COMSOL-exported setup are closer to nearly vertical grains
    with slightly curved boundaries, so each eta_i is initialized as a smooth
    stripe between two perturbed vertical boundary curves.
    """
    (x_min, y_min), (x_max, y_max) = tagged_cell_bbox(msh, cell_tags, tag)
    if y_cut is not None:
        y_max = min(y_max, y_cut)
    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span <= 0.0 or y_span <= 0.0:
        raise ValueError("Mesh bounding box is degenerate; cannot seed grains.")

    rng = np.random.default_rng(p.seed)
    base = np.linspace(x_min, x_max, p.n_grains + 1)
    spacing = x_span / p.n_grains

    internal_jitter = rng.uniform(-0.18 * spacing, 0.18 * spacing, p.n_grains - 1)
    base[1:-1] += internal_jitter
    base[1:-1] = np.sort(base[1:-1])

    phases = rng.uniform(0.0, 2.0 * np.pi, p.n_grains + 1)
    freqs = rng.uniform(1.2, 3.0, p.n_grains + 1)
    amps = rng.uniform(0.45, 1.0, p.n_grains + 1) * p.boundary_wiggle * x_span
    drifts = rng.uniform(-1.0, 1.0, p.n_grains + 1) * p.boundary_drift * x_span
    width = max(p.interface_width * x_span, 1.0e-8)

    def boundary_curve(k: int, y):
        if k == 0:
            return np.full_like(y, x_min - 4.0 * width)
        if k == p.n_grains:
            return np.full_like(y, x_max + 4.0 * width)

        yn = (y - y_min) / y_span
        curve = (
            base[k]
            + amps[k] * np.sin(2.0 * np.pi * freqs[k] * yn + phases[k])
            + drifts[k] * (yn - 0.5)
        )
        left_limit = base[k - 1] + 0.25 * spacing
        right_limit = base[k + 1] - 0.25 * spacing
        return np.clip(curve, left_limit, right_limit)

    def smooth_step(s):
        return 0.5 * (1.0 + np.tanh(s / width))

    def make_expr(i: int):
        def expr(X):
            x = X[0]
            y = X[1]
            left = boundary_curve(i, y)
            right = boundary_curve(i + 1, y)
            value = smooth_step(x - left) * smooth_step(right - x)
            return value.astype(PETSc.ScalarType)

        return expr

    return [make_expr(i) for i in range(p.n_grains)]


def build_comsol_random_initializers(
    msh, cell_tags, p: GrainParams, tag: int = OMEGA1, y_cut: float | None = None
):
    """Approximate the rn/an random initializer exported from COMSOL.

    COMSOL defines random functions rn1...rn9 and analytic functions an1...an10:
      eta_i(x,y) = 1 - an_i(x/W, y/H)
    The an_i definitions allocate each point to the first successful random
    bin, and an10 receives the leftover points. A coarse random lattice is used
    here so the initial grains are nuclei/patches rather than node-scale noise.
    """
    (x_min, y_min), (x_max, y_max) = tagged_cell_bbox(msh, cell_tags, tag)
    if y_cut is not None:
        y_max = min(y_max, y_cut)
    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span <= 0.0 or y_span <= 0.0:
        raise ValueError("Mesh bounding box is degenerate; cannot seed grains.")

    rng = np.random.default_rng(p.seed)
    ngrid = p.comsol_random_grid

    # COMSOL's rn_i are independent seeded random functions with mean 1 and
    # uniformrange = 10, 9, ..., 2. We use U(1-range/2, 1+range/2), which gives
    # the same threshold probabilities implied by floor(min(rn_i + offset_i, 1)).
    random_fields = [
        rng.uniform(1.0 - r / 2.0, 1.0 + r / 2.0, size=(ngrid, ngrid))
        for r in range(10, 1, -1)
    ]

    def smooth_random(field, x, y):
        """Bilinearly interpolate rn_i(x,y) instead of snapping to grid cells."""
        gx = np.clip((x - x_min) / x_span * (ngrid - 1), 0.0, ngrid - 1.0)
        gy = np.clip((y - y_min) / y_span * (ngrid - 1), 0.0, ngrid - 1.0)
        ix0 = np.floor(gx).astype(np.int64)
        iy0 = np.floor(gy).astype(np.int64)
        ix1 = np.minimum(ix0 + 1, ngrid - 1)
        iy1 = np.minimum(iy0 + 1, ngrid - 1)
        tx = gx - ix0
        ty = gy - iy0
        f00 = field[ix0, iy0]
        f10 = field[ix1, iy0]
        f01 = field[ix0, iy1]
        f11 = field[ix1, iy1]
        return (
            (1.0 - tx) * (1.0 - ty) * f00
            + tx * (1.0 - ty) * f10
            + (1.0 - tx) * ty * f01
            + tx * ty * f11
        )

    def labels_at_points(x, y):
        assigned = np.zeros_like(x, dtype=bool)
        labels = np.full_like(x, p.n_grains - 1, dtype=np.int64)
        offsets = np.array([4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0])

        for i, offset in enumerate(offsets):
            rn = smooth_random(random_fields[i], x, y)
            # floor(min(rn + offset, 1)) is 0 only when rn + offset < 1.
            eta_i_is_one = (~assigned) & (rn + offset < 1.0)
            labels[eta_i_is_one] = i
            assigned |= eta_i_is_one

        return labels

    def make_expr(i: int):
        def expr(X):
            labels = labels_at_points(X[0], X[1])
            value = np.where(labels == i, 1.0, 0.0)
            return value.astype(PETSc.ScalarType)

        return expr

    return [make_expr(i) for i in range(p.n_grains)]


def build_partition_random_initializers(
    msh, cell_tags, p: GrainParams, tag: int = OMEGA1, y_cut: float | None = None
):
    """Continuous random partition initializer with connected grain boundaries."""
    (x_min, y_min), (x_max, y_max) = tagged_cell_bbox(msh, cell_tags, tag)
    if y_cut is not None:
        y_max = min(y_max, y_cut)
    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span <= 0.0 or y_span <= 0.0:
        raise ValueError("Mesh bounding box is degenerate; cannot seed grains.")

    rng = np.random.default_rng(p.seed)
    n_seeds = max(int(p.partition_seed_count), p.n_grains)
    seed_x = rng.uniform(x_min, x_max, n_seeds)
    seed_y = rng.uniform(y_min, y_max, n_seeds)
    seed_label = rng.integers(0, p.n_grains, n_seeds)
    seed_label[: p.n_grains] = np.arange(p.n_grains)
    rng.shuffle(seed_label)
    width = max(float(p.partition_transition_width), 1.0e-8)

    def soft_membership(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        dx = x[:, None] - seed_x[None, :]
        dy = y[:, None] - seed_y[None, :]
        dist2 = dx * dx + dy * dy
        scores = np.full((x.size, p.n_grains), np.inf, dtype=np.float64)
        for i in range(p.n_grains):
            member = seed_label == i
            if np.any(member):
                scores[:, i] = np.min(dist2[:, member], axis=1)
        scores = -scores / (2.0 * width * width)
        scores -= np.max(scores, axis=1, keepdims=True)
        weights = np.exp(scores)
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        weights_sum[weights_sum < 1.0e-300] = 1.0
        return weights / weights_sum

    def make_expr(i: int):
        def expr(X):
            values = soft_membership(X[0], X[1])[:, i]
            return values.astype(PETSc.ScalarType)

        return expr

    return [make_expr(i) for i in range(p.n_grains)]


def build_grain_initializers(
    msh, cell_tags, p: GrainParams, tag: int = OMEGA1, y_cut: float | None = None
):
    """Select the initial grain generator."""
    if p.initializer == "partition_random":
        return build_partition_random_initializers(msh, cell_tags, p, tag=tag, y_cut=y_cut)
    if p.initializer == "comsol_random":
        return build_comsol_random_initializers(msh, cell_tags, p, tag=tag, y_cut=y_cut)
    if p.initializer == "columnar":
        return build_columnar_grain_initializers(msh, cell_tags, p, tag=tag, y_cut=y_cut)
    raise ValueError(
        f"Unknown initializer {p.initializer!r}; use 'partition_random', 'comsol_random', or 'columnar'."
    )


def normalize_eta_components(w, ME, n_grains: int):
    """Normalize nodal eta values so sum_i eta_i is approximately one."""
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
    """Keep eta values in the physical range used by COMSOL's bounded solve."""
    w.x.array[:] = np.clip(w.x.array.real, 0.0, 1.0).astype(w.x.array.dtype)
    w.x.scatter_forward()


def set_top_cap_single_grain(w, ME, p: GrainParams, y_cut: float):
    """Keep the added top cap out of the grain-boundary model."""
    for i in range(p.n_grains):
        Vc, submap = ME.sub(i).collapse()
        coords = Vc.tabulate_dof_coordinates()[:, :2]
        cap = coords[:, 1] > y_cut
        if np.any(cap):
            values = w.x.array[submap].copy()
            values[cap] = 1.0 if i == 0 else 0.0
            w.x.array[submap] = values
    w.x.scatter_forward()


def set_region_single_grain(w, ME, cell_tags, tag: int, p: GrainParams):
    """Set eta=(1,0,...,0) on all dofs belonging to one cell region."""
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


def assign_component_from_expression(w, ME, component: int, expr):
    """Interpolate an initializer into one component of a mixed function."""
    Vc, submap = ME.sub(component).collapse()
    f = fem.Function(Vc)
    f.interpolate(expr)
    w.x.array[submap] = f.x.array


def split_to_scalar_functions(w, V_out, names):
    """Project mixed eta components to one common scalar output space.

    Do not copy collapsed-subspace arrays directly into the plotting/output
    space. Different collapsed spaces can have different dof orderings, which
    makes a correct columnar field look like a fan-shaped artifact in plots.
    """
    scalar_outputs = []
    points = V_out.element.interpolation_points
    if callable(points):
        points = points()
    components = ufl.split(w)
    for i, name in enumerate(names):
        out = fem.Function(V_out, name=name)
        expr = fem.Expression(components[i], points)
        out.interpolate(expr)
        out.x.scatter_forward()
        scalar_outputs.append(out)
    return scalar_outputs


def update_scalar_functions(w, V_out, scalar_outputs):
    """Refresh scalar eta outputs by interpolation from the mixed solution."""
    points = V_out.element.interpolation_points
    if callable(points):
        points = points()
    components = ufl.split(w)
    for i, out in enumerate(scalar_outputs):
        expr = fem.Expression(components[i], points)
        out.interpolate(expr)
        out.x.scatter_forward()


def clamp_scalar_outputs(functions):
    """Clamp post-processing copies so saved eta_i stay in [0, 1]."""
    for fun in functions:
        fun.x.array[:] = np.clip(fun.x.array.real, 0.0, 1.0).astype(fun.x.array.dtype)
        fun.x.scatter_forward()


def set_top_cap_scalar_outputs(eta_outputs, p: GrainParams, y_cut: float):
    for i, fun in enumerate(eta_outputs):
        coords = fun.function_space.tabulate_dof_coordinates()[:, :2]
        cap = coords[:, 1] > y_cut
        if np.any(cap):
            fun.x.array[cap] = 1.0 if i == 0 else 0.0
            fun.x.scatter_forward()


def set_region_scalar_outputs(eta_outputs, cell_tags, tag: int, p: GrainParams):
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


def zero_region_scalar(fun, cell_tags, tag: int):
    tdim = fun.function_space.mesh.topology.dim
    cells = cell_tags.find(tag)
    if len(cells) == 0:
        return
    dofs = fem.locate_dofs_topological(fun.function_space, tdim, cells)
    if len(dofs) > 0:
        fun.x.array[dofs] = 0.0
        fun.x.scatter_forward()


def region_dofs(V, cell_tags, tags):
    """Return scalar/component dofs touching any cell in the supplied tags."""
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
    """Dofs in include_tags that are not shared with exclude_tags cells.

    Continuous P1 dofs on an interface are shared by both neighboring regions.
    Removing the excluded-region dofs keeps those interface traces free.
    """
    include = region_dofs(V, cell_tags, include_tags)
    exclude = region_dofs(V, cell_tags, exclude_tags)
    if include.size == 0:
        return include
    if exclude.size == 0:
        return include
    return np.setdiff1d(include, exclude, assume_unique=False).astype(np.int32)


def zero_region_only_scalar(fun, cell_tags, include_tags, exclude_tags):
    dofs = region_only_dofs(fun.function_space, cell_tags, include_tags, exclude_tags)
    if len(dofs) > 0:
        fun.x.array[dofs] = 0.0
        fun.x.scatter_forward()


def zero_cap_only_eta_outputs(eta_outputs, cell_tags, cap_tag, grain_tag):
    """Hide eta inside the initialization-only Li cap without cutting the interface."""
    if cap_tag is None:
        return
    for fun in eta_outputs:
        zero_region_only_scalar(fun, cell_tags, (cap_tag,), grain_tag)


def smooth_box_indicator_ufl(value, half_width, smoothing):
    eps = max(float(smoothing), 1.0e-12)
    return 1.0 / (1.0 + ufl.exp((abs(value) - half_width) / eps))


def interpolate_comsol_ce(
    msh,
    V_out,
    eta_outputs,
    cell_tags,
    grain_tag,
    cap_tag,
    p: GrainParams,
):
    """COMSOL-style ce = xi + (1-xi)^2 * GB window for annealing previews."""
    eta_exprs = eta_outputs
    eta_clips = [ufl.max_value(0.0, ufl.min_value(1.0, eta_i)) for eta_i in eta_exprs]
    gb_window = sum(
        (1.0 - eta_clip_i)
        * smooth_box_indicator_ufl(eta_i - 0.5, p.rho, p.gb_window_smoothing)
        for eta_i, eta_clip_i in zip(eta_exprs, eta_clips)
    )
    if cap_tag is not None:
        xi_init = comsol_step_initial_xi_ufl(msh, p)
    else:
        xi_init = 0.0
    ce_expr = xi_init + (1.0 - xi_init) ** 2 * gb_window
    ce_fun = fem.Function(V_out, name="ce")
    points = V_out.element.interpolation_points
    if callable(points):
        points = points()
    ce_fun.interpolate(fem.Expression(ce_expr, points))
    if cap_tag is not None:
        cap_dofs = region_only_dofs(V_out, cell_tags, (cap_tag,), grain_tag)
        if cap_dofs.size:
            ce_fun.x.array[cap_dofs] = 1.0
    ce_fun.x.scatter_forward()
    return ce_fun


def mixed_component_bc_from_scalar_dofs(ME, component: int, scalar_dofs, value: float):
    """Dirichlet BC on one mixed component from scalar-space dof indices."""
    scalar_dofs = np.asarray(scalar_dofs, dtype=np.int32)
    if scalar_dofs.size == 0:
        return None
    _, submap = ME.sub(component).collapse()
    mixed_dofs = submap[scalar_dofs]
    return fem.dirichletbc(PETSc.ScalarType(value), mixed_dofs, ME.sub(component))


def build_lr_periodic_eta_constraints(msh, ME, n_grains: int, tol=1.0e-8):
    """Pair right-side eta_i dofs to left-side dofs with the same y."""
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
    left_sorted = np.asarray(
        sorted(left, key=lambda dof: float(coords[dof, 1])), dtype=np.int32
    )
    right_sorted = np.asarray(
        sorted(right, key=lambda dof: float(coords[dof, 1])), dtype=np.int32
    )
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


class PeriodicNewtonProblem:
    """Serial Newton solver that enforces eta slave-master periodic constraints."""

    def __init__(
        self,
        F,
        u,
        bcs,
        J,
        periodic_constraints,
        *,
        petsc_options=None,
    ):
        self.u = u
        self.comm = u.function_space.mesh.comm
        self._distributed_problem = None
        if self.comm.size > 1:
            options = dict(petsc_options or {})
            # The serial defaults (preonly+LU) are not a portable distributed
            # configuration.  Use a genuinely parallel Krylov solve here.
            options["ksp_type"] = "gmres"
            options["pc_type"] = "jacobi"
            options["snes_type"] = "newtonls"
            options["snes_error_if_not_converged"] = False
            self._distributed_problem = NonlinearProblem(
                F,
                u,
                bcs=list(bcs),
                J=J,
                petsc_options=options,
                petsc_options_prefix="grain_",
            )
            self.solver = type("DistributedNewtonStatus", (), {})()
            return
        self.bcs = list(bcs)
        self.periodic_constraints = dict(periodic_constraints)
        self.F_form = fem.form(F)
        self.J_form = fem.form(J)
        try:
            self.b = create_vector(self.F_form)
        except TypeError:
            self.b = assemble_vector(self.F_form)
            with self.b.localForm() as loc_b:
                loc_b.set(0.0)
        self.A = create_matrix(self.J_form)
        options = dict(petsc_options or {})
        self.rtol = float(options.get("snes_rtol", 1.0e-7))
        self.atol = float(options.get("snes_atol", 1.0e-9))
        self.max_it = int(options.get("snes_max_it", 30))
        self.solver = type("PeriodicNewtonStatus", (), {})()
        self.solver.reason = 0
        self.solver.iterations = 0
        self.ksp = PETSc.KSP().create(self.comm)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")
        self.ksp.setFromOptions()
        self.full_dofs = np.arange(self.u.x.array.size, dtype=PETSc.IntType)
        self.du = np.zeros_like(self.u.x.array)
        self._build_constraint_prolongation()

    def _build_constraint_prolongation(self):
        slave_set = set(int(d) for d in self.periodic_constraints)
        reduced_dofs = np.asarray(
            [int(d) for d in self.full_dofs if int(d) not in slave_set],
            dtype=PETSc.IntType,
        )
        reduced_index = {int(dof): i for i, dof in enumerate(reduced_dofs)}
        n_full = int(self.u.x.array.size)
        n_red = int(reduced_dofs.size)
        P = PETSc.Mat().createAIJ(size=(n_full, n_red), nnz=1, comm=self.comm)
        for dof in reduced_dofs:
            P.setValue(int(dof), reduced_index[int(dof)], 1.0)
        for slave, master in self.periodic_constraints.items():
            if int(master) not in reduced_index:
                raise RuntimeError(
                    f"Periodic slave dof {slave} maps to inactive/slave master {master}."
                )
            P.setValue(int(slave), reduced_index[int(master)], 1.0)
        P.assemble()
        self.P = P
        self.reduced_dofs = reduced_dofs

    def _apply_periodic_constraints(self):
        arr = self.u.x.array
        for slave, master in self.periodic_constraints.items():
            arr[int(slave)] = arr[int(master)]
        self.u.x.scatter_forward()

    def _assemble_full_residual(self):
        self._apply_periodic_constraints()
        with self.b.localForm() as loc_b:
            loc_b.set(0.0)
        try:
            assemble_vector(self.b, self.F_form)
        except TypeError:
            assembled = assemble_vector(self.F_form)
            self.b.array[:] = assembled.array_r
            assembled.destroy()
        apply_lifting(
            self.b,
            [self.J_form],
            [self.bcs],
            x0=[self.u.x.petsc_vec],
            alpha=-1.0,
        )
        self.b.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )
        set_bc(self.b, self.bcs, self.u.x.petsc_vec, -1.0)

    def _reduced_residual_norm(self):
        self._assemble_full_residual()
        r_reduced = PETSc.Vec().createSeq(
            int(self.reduced_dofs.size), comm=self.comm
        )
        self.P.multTranspose(self.b, r_reduced)
        norm = float(r_reduced.norm())
        r_reduced.destroy()
        return norm

    def _solve_step(self):
        self._apply_periodic_constraints()
        self.A.zeroEntries()
        assemble_matrix(self.A, self.J_form, bcs=self.bcs)
        self.A.assemble()
        AP = self.A.matMult(self.P)
        A_reduced = self.P.transposeMatMult(AP)
        r_reduced = PETSc.Vec().createSeq(
            int(self.reduced_dofs.size), comm=self.comm
        )
        self.P.multTranspose(self.b, r_reduced)
        du_reduced = r_reduced.duplicate()
        self.ksp.setOperators(A_reduced)
        self.ksp.solve(r_reduced, du_reduced)
        self.du.fill(0.0)
        self.du[self.reduced_dofs] = du_reduced.array_r
        for slave, master in self.periodic_constraints.items():
            self.du[int(slave)] = self.du[int(master)]
        step_norm = float(du_reduced.norm())
        AP.destroy()
        A_reduced.destroy()
        r_reduced.destroy()
        du_reduced.destroy()
        return step_norm

    def solve(self):
        if self._distributed_problem is not None:
            result = self._distributed_problem.solve()
            petsc_solver = self._distributed_problem.solver
            self.solver.reason = int(petsc_solver.getConvergedReason())
            self.solver.iterations = int(petsc_solver.getIterationNumber())
            if self.comm.rank == 0 and self.solver.reason <= 0:
                print(
                    f"Distributed SNES did not converge: reason={self.solver.reason}, "
                    f"iterations={self.solver.iterations}, "
                    f"function_norm={petsc_solver.getFunctionNorm():.3e}",
                    flush=True,
                )
            return result
        self.solver.reason = 0
        self.solver.iterations = 0
        self._apply_periodic_constraints()
        norm0 = self._reduced_residual_norm()
        if norm0 < self.atol:
            self.solver.reason = 2
            return
        reference = max(norm0, 1.0)
        previous_norm = norm0
        old = self.u.x.array.copy()
        for it in range(1, self.max_it + 1):
            self._assemble_full_residual()
            step_norm = self._solve_step()
            alpha = 1.0
            accepted = False
            for _ in range(10):
                self.u.x.array[:] = old - alpha * self.du
                self._apply_periodic_constraints()
                trial_norm = self._reduced_residual_norm()
                if trial_norm <= previous_norm or alpha <= 1.0e-3:
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                self.u.x.array[:] = old
                self._apply_periodic_constraints()
                self.solver.reason = -3
                self.solver.iterations = it
                return
            old = self.u.x.array.copy()
            previous_norm = trial_norm
            self.solver.iterations = it
            if previous_norm < self.atol or previous_norm / reference < self.rtol:
                self.solver.reason = 2
                return
            if step_norm < 1.0e-12:
                self.solver.reason = 3
                return
        self.solver.reason = -2
        self.solver.iterations = self.max_it


def interpolate_grain_boundary_indicator(
    msh,
    etas,
    V_out,
    p: GrainParams,
    y_cut=None,
    cell_tags=None,
    cap_tag=None,
    grain_tag=None,
):
    """Compute B = 5 * min(0.2, max(0, sum_{i<j} eta_i eta_j))."""
    b_raw = sum(
        etas[i] * etas[j]
        for i in range(p.n_grains)
        for j in range(i + 1, p.n_grains)
    )
    b_expr = p.b_scale * ufl.min_value(p.b_clip, ufl.max_value(0.0, b_raw))
    b_fun = fem.Function(V_out, name="grain_boundary_B")
    points = V_out.element.interpolation_points
    if callable(points):
        points = points()
    expr = fem.Expression(b_expr, points)
    b_fun.interpolate(expr)
    b_fun.x.scatter_forward()
    return b_fun


def interpolate_grain_boundary_indicator_from_outputs(
    msh,
    eta_outputs,
    V_out,
    p: GrainParams,
):
    """Compute B from the same post-processed eta fields used by ce."""
    return interpolate_grain_boundary_indicator(
        msh,
        eta_outputs,
        V_out,
        p,
    )


def make_triangulation(V, cell_tags=None, tag=None):
    """Build a triangulation whose node numbering matches Function.x.array."""
    cells, points, mask = cells_and_dof_points(V, cell_tags=cell_tags, tag=tag)
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], cells)
    if mask is not None:
        triangulation.set_mask(mask)
    return triangulation


def _gather_preview_payload(comm, points, cells, *values):
    """Gather partition-local preview arrays and renumber cell connectivity."""
    if comm.size == 1:
        return points, cells, values
    gathered = comm.gather((points, cells, values), root=0)
    if comm.rank != 0:
        return None
    # Values may be supplied as a tuple by callers; normalize once so the
    # gather accumulator always has the same arity as each rank's payload.
    values = tuple(values)
    all_points, all_cells, all_values = [], [], [[] for _ in values]
    offset = 0
    for pts, cls, vals in gathered:
        if pts.size == 0:
            continue
        all_points.append(pts)
        all_cells.append(cls + offset)
        vals = tuple(vals)
        for i, val in enumerate(vals):
            if i >= len(all_values):
                all_values.append([])
            all_values[i].append(val)
        offset += len(pts)
    if not all_points:
        return np.empty((0, 2)), np.empty((0, 3), dtype=np.int32), tuple(np.empty(0) for _ in all_values)
    return np.vstack(all_points), np.vstack(all_cells), tuple(
        np.concatenate(parts) for parts in all_values
    )


def save_field_png(V, field, filename, title, cmap="viridis", cell_tags=None, tag=None):
    """Save one scalar finite-element field as a PNG preview."""
    if not HAS_MATPLOTLIB:
        if V.mesh.comm.size != 1:
            return
        svg_name = str(Path(filename).with_suffix(".svg"))
        save_scalar_svg(
            V,
            field.x.array.real,
            svg_name,
            title,
            palette=cmap,
            cell_tags=cell_tags,
            tag=tag,
        )
        return

    cells, points, mask = cells_and_dof_points(V, cell_tags=cell_tags, tag=tag)
    if mask is not None:
        cells = cells[~mask]
    gathered = _gather_preview_payload(V.mesh.comm, points, cells, field.x.array.real)
    if gathered is None:
        return
    points, cells, (values,) = gathered
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], cells)

    fig, ax = plt.subplots(figsize=(7.2, 8.4), dpi=180)
    color = ax.tripcolor(triangulation, values, shading="gouraud", cmap=cmap)
    ax.set_aspect("equal")
    ax.set_xlabel("x / L")
    ax.set_ylabel("y / L")
    ax.set_title(title)
    fig.colorbar(color, ax=ax)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def build_subsampled_triangles(V, cell_tags=None, tag=None, order: int = 5):
    """Subdivide preview triangles so discontinuous expressions are sampled inside cells."""
    triangles, points, mask = cells_and_dof_points(V, cell_tags=cell_tags, tag=tag)
    if mask is not None:
        triangles = triangles[~mask]

    # Barycentric grid on the reference triangle.  Duplicating points per parent
    # triangle keeps the implementation simple and avoids cross-cell smoothing.
    bary = []
    bary_id = {}
    for i in range(order + 1):
        for j in range(order + 1 - i):
            k = order - i - j
            bary_id[(i, j)] = len(bary)
            bary.append((i / order, j / order, k / order))
    bary = np.asarray(bary, dtype=np.float64)

    local_tris = []
    for i in range(order):
        for j in range(order - i):
            a = bary_id[(i, j)]
            b = bary_id[(i + 1, j)]
            c = bary_id[(i, j + 1)]
            local_tris.append((a, b, c))
            if j < order - i - 1:
                d = bary_id[(i + 1, j + 1)]
                local_tris.append((b, d, c))
    local_tris = np.asarray(local_tris, dtype=np.int32)

    all_points = []
    all_cells = []
    all_parent = []
    all_bary = []
    offset = 0
    for tri in triangles:
        tri_points = points[tri]
        new_points = bary @ tri_points
        all_points.append(new_points)
        all_cells.append(local_tris + offset)
        all_parent.append(np.full(len(bary), len(all_parent), dtype=np.int32))
        all_bary.append(bary)
        offset += len(bary)

    if not all_points:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 3), dtype=np.float64),
        )

    return (
        np.vstack(all_points),
        np.vstack(all_cells),
        triangles,
        np.vstack(all_bary),
    )


def save_comsol_sampled_b_ce_pngs(
    V,
    eta_outputs,
    p: GrainParams,
    b_filename,
    ce_filename,
    b_title,
    ce_title,
    cell_tags=None,
    tag=None,
):
    """Save COMSOL-like B and ce previews by evaluating formulas inside cells."""
    if not HAS_MATPLOTLIB:
        return

    sample_points, sample_cells, parent_tris, sample_bary = build_subsampled_triangles(
        V, cell_tags=cell_tags, tag=tag, order=5
    )
    if sample_points.size == 0:
        # This is a collective preview routine: empty MPI ranks must still
        # participate in gather, otherwise rank 0 waits forever after writing
        # only the initial image.
        gathered = _gather_preview_payload(
            V.mesh.comm,
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 3), dtype=np.int32),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
        if gathered is None or gathered[0].size == 0:
            return

    def smooth_box_indicator_np(value, half_width, smoothing):
        eps = max(float(smoothing), 1.0e-12)
        z = np.clip((np.abs(value) - half_width) / eps, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(z))

    eta_node_values = np.vstack([eta_i.x.array.real for eta_i in eta_outputs])
    eta_samples = []
    samples_per_parent = sample_bary.shape[0] // parent_tris.shape[0]
    for parent_index, tri in enumerate(parent_tris):
        bary_slice = sample_bary[
            parent_index * samples_per_parent : (parent_index + 1) * samples_per_parent
        ]
        eta_vertices = eta_node_values[:, tri]
        eta_samples.append(bary_slice @ eta_vertices.T)
    eta_samples = np.vstack(eta_samples)

    pair_sum = np.zeros(eta_samples.shape[0], dtype=np.float64)
    for i in range(p.n_grains):
        for j in range(i + 1, p.n_grains):
            pair_sum += eta_samples[:, i] * eta_samples[:, j]
    b_values = p.b_scale * np.minimum(p.b_clip, np.maximum(0.0, pair_sum))

    eta_clip = np.clip(eta_samples, 0.0, 1.0)
    gb_window = np.sum(
        (1.0 - eta_clip)
        * smooth_box_indicator_np(eta_samples - 0.5, p.rho, p.gb_window_smoothing),
        axis=1,
    )
    # In MPI each rank sees a different local y_max.  Using that value makes
    # the cap contribution jump between partitions, creating the solid red
    # blocks in ce.  The mesh is nondimensionalized by L and its global top is
    # y/L=2.4, so use the same global value on every rank for the preview.
    xi_init = comsol_step_initial_xi_np(sample_points[:, 1], 2.4, p)
    ce_values = xi_init + (1.0 - xi_init) ** 2 * gb_window

    gathered = _gather_preview_payload(
        V.mesh.comm, sample_points, sample_cells, b_values, ce_values
    )
    if gathered is None:
        return
    sample_points, sample_cells, gathered_values = gathered
    b_values, ce_values = gathered_values[0], gathered_values[1]

    for values, filename, title, cmap, clim in (
        (b_values, b_filename, b_title, "inferno", (0.0, 1.0)),
        (ce_values, ce_filename, ce_title, "turbo", (-1.16e-3, 1.0)),
    ):
        triangulation = mtri.Triangulation(sample_points[:, 0], sample_points[:, 1], sample_cells)
        fig, ax = plt.subplots(figsize=(7.2, 8.4), dpi=180)
        color = ax.tripcolor(
            triangulation,
            values,
            shading="flat",
            cmap=cmap,
            vmin=clim[0],
            vmax=clim[1],
        )
        ax.set_aspect("equal")
        ax.set_xlabel("x / L")
        ax.set_ylabel("y / L")
        ax.set_title(title)
        fig.colorbar(color, ax=ax)
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)


def save_grain_label_png(
    V, eta_outputs, filename="grain_labels.png", cell_tags=None, tag=None
):
    """Save argmax_i eta_i as a quick grain-domain preview."""
    if V.mesh.comm.size != 1 and not HAS_MATPLOTLIB:
        return

    cells, points, mask = cells_and_dof_points(V, cell_tags=cell_tags, tag=tag)
    if mask is not None:
        cells = cells[~mask]
    eta_values = np.vstack([eta_i.x.array.real for eta_i in eta_outputs])
    gathered = _gather_preview_payload(V.mesh.comm, points, cells, *eta_values)
    if gathered is None:
        return
    points, cells, eta_parts = gathered
    eta_values = np.vstack(eta_parts)
    labels = np.argmax(eta_values, axis=0)

    if not HAS_MATPLOTLIB:
        svg_name = str(Path(filename).with_suffix(".svg"))
        save_label_svg(
            V,
            labels,
            svg_name,
            "Grain labels: argmax eta_i",
            cell_tags=cell_tags,
            tag=tag,
        )
        return

    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], cells)

    fig, ax = plt.subplots(figsize=(7.2, 8.4), dpi=180)
    color = ax.tripcolor(triangulation, labels, shading="flat", cmap="tab20")
    ax.set_aspect("equal")
    ax.set_xlabel("x / L")
    ax.set_ylabel("y / L")
    ax.set_title("Grain labels: argmax eta_i")
    fig.colorbar(color, ax=ax, ticks=range(len(eta_outputs)))
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def save_boundary_preview_png(
    V, eta_outputs, filename="grain_boundary_preview.png", cell_tags=None, tag=None
):
    """Save an intentionally high-contrast grain-boundary preview.

    COMSOL's physical B uses overlap eta_i eta_j, which can be very small on a
    coarse mesh. For visual checking, 1 - max_i(eta_i) highlights transition
    bands even when the physical B image looks dark.
    """
    if V.mesh.comm.size != 1 and not HAS_MATPLOTLIB:
        return

    cells, points, mask = cells_and_dof_points(V, cell_tags=cell_tags, tag=tag)
    if mask is not None:
        cells = cells[~mask]
    eta_values = np.vstack([eta_i.x.array.real for eta_i in eta_outputs])
    gathered = _gather_preview_payload(V.mesh.comm, points, cells, *eta_values)
    if gathered is None:
        return
    points, cells, eta_parts = gathered
    eta_values = np.vstack(eta_parts)
    preview = 1.0 - np.max(eta_values, axis=0)
    if np.max(preview) > 0.0:
        preview = preview / np.max(preview)

    if not HAS_MATPLOTLIB:
        svg_name = str(Path(filename).with_suffix(".svg"))
        save_scalar_svg(
            V,
            preview,
            svg_name,
            "Visual grain-boundary preview: 1 - max eta_i",
            palette="Reds",
            cell_tags=cell_tags,
            tag=tag,
        )
        return

    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], cells)

    fig, ax = plt.subplots(figsize=(7.2, 8.4), dpi=180)
    color = ax.tripcolor(triangulation, preview, shading="gouraud", cmap="Reds")
    ax.set_aspect("equal")
    ax.set_xlabel("x / L")
    ax.set_ylabel("y / L")
    ax.set_title("Visual grain-boundary preview: 1 - max eta_i")
    fig.colorbar(color, ax=ax)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def save_timed_previews(
    V,
    eta_outputs,
    b_fun,
    ce_fun,
    p: GrainParams,
    t: float,
    cell_tags=None,
    tag=None,
    b_tag=None,
    ce_tag=None,
    output_dir="grain_boundary_previews",
):
    """Save grain-boundary snapshots at one annealing time."""
    time_label = f"t{int(round(t)):04d}s"
    output_dir = Path(output_dir)
    if V.mesh.comm.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    V.mesh.comm.barrier()
    save_comsol_sampled_b_ce_pngs(
        V,
        eta_outputs,
        p,
        b_filename=str(output_dir / f"grain_boundary_B_{time_label}.png"),
        ce_filename=str(output_dir / f"grain_boundary_ce_{time_label}.png"),
        b_title=f"Grain-boundary indicator B at t={t:.0f} s",
        ce_title=f"COMSOL-style ce during grain annealing at t={t:.0f} s",
    )
    save_grain_label_png(
        V,
        eta_outputs,
        filename=str(output_dir / f"grain_labels_{time_label}.png"),
    )
    save_boundary_preview_png(
        V,
        eta_outputs,
        filename=str(output_dir / f"grain_boundary_preview_{time_label}.png"),
    )


def cells_and_dof_points(V, cell_tags=None, tag=None):
    """Return cell dofs and coordinates using the same numbering as field values.

    Matplotlib's tripcolor expects the triangle connectivity to index the same
    array used for scalar values. Using plot.vtk_mesh(V) can reorder points for
    visualization, while Function.x.array is ordered by finite-element dofs.
    That mismatch produces the fan/radiating artifact seen in the preview.
    """
    msh = V.mesh
    tdim = msh.topology.dim
    index_map = msh.topology.index_map(tdim)
    # Gather each MPI partition's owned cells exactly once. Ghost cells have
    # local dof indices too, but including them duplicates/overlays geometry in
    # the rank-0 preview (the solve itself still uses ghost values).
    num_cells = index_map.size_local
    points = V.tabulate_dof_coordinates()[:, :2]
    cells = np.asarray([V.dofmap.cell_dofs(cell) for cell in range(num_cells)], dtype=np.int32)
    if num_cells == 0:
        # Some MPI ranks may own no cells in the selected grain region.
        # Return well-shaped empty arrays so gather/plot code can proceed.
        ndofs = int(getattr(V.element, "space_dimension", 3))
        return np.empty((0, ndofs), dtype=np.int32), points, None

    # Basix dof order on quadrilateral cells is not guaranteed to be geometric
    # counter-clockwise order. Sort each cell's dofs by angle before plotting;
    # otherwise the preview can contain artificial crossed triangles.
    cell_points = points[cells]
    centers = np.mean(cell_points, axis=1)
    angles = np.arctan2(
        cell_points[:, :, 1] - centers[:, None, 1],
        cell_points[:, :, 0] - centers[:, None, 0],
    )
    order = np.argsort(angles, axis=1)
    cells = np.take_along_axis(cells, order, axis=1)

    cell_mask = None
    if cell_tags is not None and tag is not None:
        selected = np.zeros(num_cells, dtype=bool)
        tag_tuple = tag if isinstance(tag, tuple) else (tag,)
        tagged_pieces = [cell_tags.find(int(one_tag)) for one_tag in tag_tuple]
        tagged_cells = (
            np.concatenate(tagged_pieces)
            if any(len(piece) > 0 for piece in tagged_pieces)
            else np.empty(0, dtype=np.int32)
        )
        tagged_cells = tagged_cells[tagged_cells < num_cells]
        selected[tagged_cells] = True
        cell_mask = ~selected

    # The mesh contains another tagged block below the grain window.  Filter
    # it only for visualization; the MPI solve still uses the complete field.
    # Test every vertex so no lower cell can leak through the gathered PNG.
    if cells.size:
        lower = np.any(points[cells, 1] < 1.5 - 1.0e-10, axis=1)
        cell_mask = lower if cell_mask is None else np.logical_or(cell_mask, lower)

    if cells.shape[1] == 3:
        triangles = cells
        triangle_mask = cell_mask
    elif cells.shape[1] == 4:
        # Matplotlib only supports triangular cells. The gmsh battery mesh uses
        # bilinear quads, so split each quad into two triangles for previewing.
        triangles = np.vstack(
            (
                cells[:, [0, 1, 2]],
                cells[:, [0, 2, 3]],
            )
        ).astype(np.int32)
        triangle_mask = (
            np.concatenate((cell_mask, cell_mask)) if cell_mask is not None else None
        )
    else:
        raise ValueError(
            "Preview plotting only supports P1 triangles or Q1 quadrilaterals; "
            f"got {cells.shape[1]} dofs per cell."
        )

    return triangles, points, triangle_mask


def svg_project(points, width=900.0, margin=36.0):
    """Project nondimensional mesh coordinates into an SVG viewport."""
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    span_x = max(x_max - x_min, 1.0e-12)
    span_y = max(y_max - y_min, 1.0e-12)
    scale = (width - 2.0 * margin) / span_x
    height = span_y * scale + 2.0 * margin

    xy = np.empty_like(points)
    xy[:, 0] = margin + (points[:, 0] - x_min) * scale
    xy[:, 1] = height - margin - (points[:, 1] - y_min) * scale
    return xy, width, height


def scalar_color(value, vmin, vmax, palette):
    """Small dependency-free colormaps used when matplotlib is unavailable."""
    if vmax <= vmin:
        s = 0.0
    else:
        s = float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))

    if palette == "Reds":
        r = 255
        g = int(248 - 210 * s)
        b = int(240 - 225 * s)
    elif palette == "inferno":
        r = int(12 + 240 * s)
        g = int(7 + 120 * s**1.8)
        b = int(60 * (1.0 - s) + 15 * s)
    else:
        r = int(45 + 30 * s)
        g = int(75 + 155 * s)
        b = int(120 + 60 * (1.0 - s))
    return f"rgb({r},{g},{b})"


def save_scalar_svg(
    V, values, filename, title, palette="viridis", cell_tags=None, tag=None
):
    """Dependency-free SVG scalar preview for environments without matplotlib."""
    cells, points, mask = cells_and_dof_points(V, cell_tags=cell_tags, tag=tag)
    if mask is not None:
        cells = cells[~mask]
    xy, width, height = svg_project(points)
    cell_values = np.mean(values[cells], axis=1)
    vmin = float(np.nanmin(cell_values))
    vmax = float(np.nanmax(cell_values))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
        f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="24" font-size="18" font-family="sans-serif">{title}</text>',
    ]
    for tri, value in zip(cells, cell_values):
        coords = " ".join(f"{xy[k, 0]:.2f},{xy[k, 1]:.2f}" for k in tri)
        color = scalar_color(value, vmin, vmax, palette)
        parts.append(f'<polygon points="{coords}" fill="{color}" stroke="{color}" stroke-width="0.2"/>')
    parts.append("</svg>")
    Path(filename).write_text("\n".join(parts), encoding="utf-8")


def save_label_svg(V, labels, filename, title, cell_tags=None, tag=None):
    """Dependency-free SVG grain-label preview."""
    cells, points, mask = cells_and_dof_points(V, cell_tags=cell_tags, tag=tag)
    if mask is not None:
        cells = cells[~mask]
    xy, width, height = svg_project(points)
    palette = [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#59A14F",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
        f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="24" font-size="18" font-family="sans-serif">{title}</text>',
    ]
    for tri in cells:
        label = int(np.bincount(labels[tri]).argmax())
        coords = " ".join(f"{xy[k, 0]:.2f},{xy[k, 1]:.2f}" for k in tri)
        color = palette[label % len(palette)]
        parts.append(f'<polygon points="{coords}" fill="{color}" stroke="{color}" stroke-width="0.2"/>')
    parts.append("</svg>")
    Path(filename).write_text("\n".join(parts), encoding="utf-8")


def save_final_npz(filename, V_out, eta_outputs, b_fun, p: GrainParams, t: float, extra_outputs=()):
    """Save final grain fields in a compact format for the next FEniCSx solver.

    This assumes the next script uses the same mesh and the same first-order
    scalar function space. The arrays can then be copied directly into matching
    dolfinx Function.x.array buffers.
    """
    eta_values = np.vstack([eta_i.x.array.real for eta_i in eta_outputs])
    extra_values = [fun.x.array.real.copy() for fun in extra_outputs]
    extra_names = [fun.name for fun in extra_outputs]
    comm = V_out.mesh.comm
    payload = comm.gather(
        (V_out.tabulate_dof_coordinates()[:, :2], eta_values, b_fun.x.array.real.copy(), extra_values),
        root=0,
    )
    if comm.rank != 0:
        return
    coords = np.vstack([part[0] for part in payload])
    eta_values = np.hstack([part[1] for part in payload])
    b_values = np.concatenate([part[2] for part in payload])
    extra_values = [np.concatenate([part[3][i] for part in payload]) for i in range(len(extra_values))]
    np.savez(
        filename,
        t=np.array(t, dtype=np.float64),
        length_scale=np.array(p.length_scale, dtype=np.float64),
        dof_coordinates=coords,
        eta=eta_values,
        B=b_values,
        extra=np.vstack(extra_values) if extra_values else np.empty((0, eta_values.shape[1])),
        names=np.array([eta_i.name for eta_i in eta_outputs] + [b_fun.name] + extra_names),
    )


def main(
    msh_file: str = DEFAULT_MSH_FILE,
    out_file: str = "grain_boundaries_periodic_120um.bp",
):
    p = GrainParams()
    parent_msh, cell_tags, facet_tags = read_mesh(msh_file, p)
    print_tag_report(parent_msh.comm, cell_tags, facet_tags)
    tdim = parent_msh.topology.dim

    omega1_domain_tags = omega1_tags(cell_tags)
    grain_tag = (grain_region_tag(cell_tags),)
    # Mesh exports differ in whether the initialization region is tagged 102
    # or folded into another physical region. Select a usable cell tag before
    # constructing the submesh instead of failing on a stale default tag.
    if len(cell_tags.find(int(grain_tag[0]))) == 0:
        available = set(int(v) for v in cell_tags.values.tolist())
        for candidate in (OMEGA12, OMEGA1, OMEGA11, OMEGA2, OMEGA3):
            if candidate in available and len(cell_tags.find(candidate)) > 0:
                grain_tag = (candidate,)
                break
    cap_tag = li_cap_region_tag(cell_tags)
    grain_cells = np.unique(
        np.concatenate([cell_tags.find(int(tag)) for tag in grain_tag])
    ).astype(np.int32)
    # In DOLFINx 0.11, create_submesh requires owned cells only. Passing ghost
    # cell indices can trigger "Index owner change detected" during MPI setup.
    owned_cell_count = int(parent_msh.topology.index_map(tdim).size_local)
    grain_cells = grain_cells[grain_cells < owned_cell_count]
    # Keep the physical 102 MeshTag in MPI. A coordinate cut here clips the
    # square region and creates the saw-tooth lower edge visible in PNGs.
    if grain_cells.size == 0 and parent_msh.comm.size == 1:
        available = sorted(set(int(v) for v in cell_tags.values.tolist()))
        raise ValueError(
            f"No cells found for grain_tag={grain_tag}; available cell tags={available}."
        )

    submesh_result = mesh.create_submesh(parent_msh, tdim, grain_cells)
    msh = submesh_result[0]
    msh.topology.create_connectivity(tdim - 1, tdim)
    msh.topology.create_connectivity(tdim, tdim - 1)
    msh.topology.create_connectivity(tdim, 0)
    dx = ufl.Measure("dx", domain=msh)
    grain_dx = dx
    if parent_msh.comm.rank == 0:
        print(
            "Grain initialization regions: "
            f"omega1_tags={omega1_domain_tags}, grain_tag={grain_tag}, "
            f"cap_tag={cap_tag}, submesh_cells={grain_cells.size}"
        )

    # Mixed space for eta1...eta10 on the grain submesh only. These fields
    # describe solid electrolyte grains; their overlap after annealing is B.
    P1 = element("Lagrange", msh.basix_cell(), 1)
    ME = fem.functionspace(msh, mixed_element([P1] * p.n_grains))
    V_scalar = fem.functionspace(msh, ("Lagrange", 1))

    eta = fem.Function(ME, name="eta")
    eta_n = fem.Function(ME, name="eta_old")
    etas = ufl.split(eta)
    etas_n = ufl.split(eta_n)
    tests = ufl.TestFunctions(ME)
    deta = ufl.TrialFunction(ME)

    # Grain seeds. "comsol_random" follows the rn/an initializer in text.m;
    # "columnar" is an artificial visual helper for nearly vertical grains.
    with eta.x.petsc_vec.localForm() as loc:
        loc.set(0.0)
    for i, initializer in enumerate(
        build_grain_initializers(
            parent_msh,
            cell_tags,
            p,
            tag=grain_tag,
            y_cut=None,
        )
    ):
        assign_component_from_expression(eta, ME, i, initializer)
    normalize_eta_components(eta, ME, p.n_grains)
    eta.x.scatter_forward()
    # Exact slave/master elimination is serial-only. MPI uses the distributed
    # nonlinear solver selected in PeriodicNewtonProblem.
    periodic_constraints = (
        build_lr_periodic_eta_constraints(msh, ME, p.n_grains)
        if msh.comm.size == 1
        else {}
    )
    if msh.comm.size > 1 and msh.comm.rank == 0:
        print(
            "MPI mode: using distributed DOLFINx Newton solve; "
            "the serial slave-master periodic reduction is disabled."
        )
    for slave, master in periodic_constraints.items():
        eta.x.array[int(slave)] = eta.x.array[int(master)]
    eta.x.scatter_forward()
    if msh.comm.rank == 0:
        print(
            "COMSOL-style grain annealing periodic BCs: "
            f"left-right eta_i tie enabled, slave dofs={len(periodic_constraints)}."
        )
    if p.clip_eta_after_solve:
        clip_eta_components(eta)
        for slave, master in periodic_constraints.items():
            eta.x.array[int(slave)] = eta.x.array[int(master)]
        eta.x.scatter_forward()
    eta_n.x.array[:] = eta.x.array

    dt_hat = fem.Constant(msh, PETSc.ScalarType(p.dt / p.time_scale))

    # Nondimensional coefficients after x_hat = x/L and t_hat = t/t0.
    ac_grad = p.time_scale * p.L_phi * p.k_phi / (p.length_scale**2)
    ac_chem = p.time_scale * p.L_phi * p.W_phi

    # COMSOL equation form for each eta_i:
    #   d eta_i / dt =
    #       L_phi*k_phi*Delta(eta_i)
    #       - L_phi*(-W_phi*eta_i + W_phi*eta_i^3
    #                + 2 W_phi eta_i sum_{j != i} eta_j^2)
    #
    # Weak backward-Euler residual on the grain submesh:
    #   int (eta_i-eta_i_n)/dt * v_i dx
    # + int ac_grad grad(eta_i).grad(v_i) dx
    # + int ac_chem*(-eta_i + eta_i^3 + 2 eta_i sum eta_j^2) * v_i dx = 0
    #
    # Mechanical coupling and xi coupling are omitted in this preprocessing
    # version. The goal here is only to generate a stable grain-boundary field.
    F_eta = 0
    for i in range(p.n_grains):
        eta_i = etas[i]
        eta_i_n = etas_n[i]
        v_i = tests[i]
        cross = sum(etas[j] ** 2 for j in range(p.n_grains) if j != i)
        dF_deta_i = -eta_i + eta_i**3 + 2.0 * eta_i * cross
        F_eta += (
            (eta_i - eta_i_n) / dt_hat * v_i * grain_dx
            + ac_grad * ufl.dot(ufl.grad(eta_i), ufl.grad(v_i)) * grain_dx
            + ac_chem * dF_deta_i * v_i * grain_dx
        )

    F_total = F_eta
    J = ufl.derivative(F_total, eta, deta)

    bcs = []

    problem = PeriodicNewtonProblem(
        F_total,
        eta,
        bcs=bcs,
        J=J,
        periodic_constraints=periodic_constraints,
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1.0e-7,
            "snes_atol": 1.0e-9,
            "snes_max_it": 30,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    V_out = V_scalar
    eta_outputs = split_to_scalar_functions(
        eta, V_out, [f"eta{i + 1}" for i in range(p.n_grains)]
    )
    if p.clip_eta_after_solve:
        clamp_scalar_outputs(eta_outputs)
    b_fun = interpolate_grain_boundary_indicator_from_outputs(
        msh,
        eta_outputs,
        V_out,
        p,
    )
    ce_fun = interpolate_comsol_ce(
        msh, V_out, eta_outputs, None, None, None, p
    )
    outputs = eta_outputs + [b_fun, ce_fun]

    def refresh_outputs():
        update_scalar_functions(eta, V_out, eta_outputs)
        if p.clip_eta_after_solve:
            clamp_scalar_outputs(eta_outputs)
        refreshed_b = interpolate_grain_boundary_indicator_from_outputs(
            msh,
            eta_outputs,
            V_out,
            p,
        )
        refreshed_ce = interpolate_comsol_ce(
            msh, V_out, eta_outputs, None, None, None, p
        )
        outputs[-2].x.array[:] = refreshed_b.x.array
        outputs[-2].x.scatter_forward()
        outputs[-1].x.array[:] = refreshed_ce.x.array
        outputs[-1].x.scatter_forward()

    t = 0.0
    refresh_outputs()
    preview_dir = Path("grain_boundary_previews")
    if msh.comm.rank == 0:
        preview_dir.mkdir(parents=True, exist_ok=True)
    msh.comm.barrier()

    # Save the initial map separately. This isolates seeding/plotting mistakes
    # from possible Allen-Cahn evolution issues.
    save_grain_label_png(
        V_out,
        eta_outputs,
        filename=str(preview_dir / "initial_grain_labels.png"),
        # V_out is already the Omega12 submesh; do not apply parent-mesh
        # MeshTags indices to it (those indices cause checkerboard masks).
    )
    save_boundary_preview_png(
        V_out,
        eta_outputs,
        filename=str(preview_dir / "initial_grain_boundary_preview.png"),
        # Submesh contains only the grain region.
    )
    save_timed_previews(
        V_out,
        eta_outputs,
        outputs[-2],
        outputs[-1],
        p,
        t,
        output_dir=preview_dir,
        # Submesh contains only the grain region.
    )

    step = 0
    next_preview_t = p.preview_interval
    while t < p.t_end - 1.0e-14:
        problem.solve()
        reason = int(problem.solver.reason)
        its = int(problem.solver.iterations)
        if reason <= 0:
            raise RuntimeError(
                f"Grain annealing SNES failed at t={t:g}, "
                f"iterations={its}, reason={reason}"
            )

        eta.x.scatter_forward()
        if p.clip_eta_after_solve:
            clip_eta_components(eta)
        for slave, master in periodic_constraints.items():
            eta.x.array[int(slave)] = eta.x.array[int(master)]
        eta.x.scatter_forward()
        eta_n.x.array[:] = eta.x.array

        t += p.dt
        step += 1
        if t >= next_preview_t - 0.5 * p.dt:
            refresh_outputs()
            save_timed_previews(
                V_out,
                eta_outputs,
                outputs[-2],
                outputs[-1],
                p,
                t,
                output_dir=preview_dir,
                # Submesh contains only the grain region.
            )
            next_preview_t += p.preview_interval

        if msh.comm.rank == 0:
            print(
                f"step={step}, t={t:.4e} s, SNES iterations={its}, "
                f"reason={reason}"
            )

    refresh_outputs()
    save_final_npz(
        "r2d_120um_r3_comsol_layout1.npz",
        V_out,
        eta_outputs,
        outputs[-2],
        p,
        t,
        extra_outputs=(outputs[-1],),
    )

    # PNG previews for quick verification without ParaView.
    save_comsol_sampled_b_ce_pngs(
        V_out,
        eta_outputs,
        p,
        b_filename=str(preview_dir / "grain_boundary_B.png"),
        ce_filename=str(preview_dir / "grain_boundary_ce.png"),
        b_title="Grain-boundary indicator B",
        ce_title="COMSOL-style ce during grain annealing",
        # Submesh contains only the grain region.
    )
    save_grain_label_png(V_out, eta_outputs, filename=str(preview_dir / "grain_labels.png"))
    save_boundary_preview_png(V_out, eta_outputs, filename=str(preview_dir / "grain_boundary_preview.png"))
    if msh.comm.rank == 0:
        print(
            "Saved previews: grain_boundary_B.png, grain_labels.png, "
            "grain_boundary_preview.png"
        )


if __name__ == "__main__":
    main()
