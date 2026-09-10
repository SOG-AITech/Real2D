from __future__ import annotations
from pathlib import Path
import numpy as np
from .config import GrainParams
from .initialization import comsol_step_initial_xi_np
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


def save_initial_outputs(V, eta_outputs, b_fun, ce_fun, p, output_dir="grain_boundary_previews"):
    """Write the initial preview set for a prepared grain state."""
    output_dir = Path(output_dir)
    if V.mesh.comm.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    V.mesh.comm.barrier()
    save_grain_label_png(
        V, eta_outputs,
        filename=str(output_dir / "initial_grain_labels.png"),
    )
    save_boundary_preview_png(
        V, eta_outputs,
        filename=str(output_dir / "initial_grain_boundary_preview.png"),
    )
    save_timed_previews(
        V, eta_outputs, b_fun, ce_fun, p, 0.0, output_dir=output_dir,
    )


def save_final_outputs(
    filename, V, eta_outputs, b_fun, ce_fun, p, t,
    output_dir="grain_boundary_previews",
):
    """Write the final NPZ and the final visual preview set."""
    output_dir = Path(output_dir)
    save_final_npz(
        filename, V, eta_outputs, b_fun, p, t, extra_outputs=(ce_fun,)
    )
    save_comsol_sampled_b_ce_pngs(
        V, eta_outputs, p,
        b_filename=str(output_dir / "grain_boundary_B.png"),
        ce_filename=str(output_dir / "grain_boundary_ce.png"),
        b_title="Grain-boundary indicator B",
        ce_title="COMSOL-style ce during grain annealing",
    )
    save_grain_label_png(
        V, eta_outputs, filename=str(output_dir / "grain_labels.png")
    )
    save_boundary_preview_png(
        V, eta_outputs, filename=str(output_dir / "grain_boundary_preview.png")
    )


