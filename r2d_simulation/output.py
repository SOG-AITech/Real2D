"""PNG, XDMF, NPZ, and snapshot output interfaces."""

import math
from pathlib import Path
from dataclasses import asdict, dataclass
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem

from .config import OMEGA1, OMEGA11, OMEGA12, OMEGA1_REGIONS, OMEGA2, OMEGA3, Params
from .regions import (
    build_subsampled_triangles,
    region_dofs_from_markers,
    restricted_dofs,
    cells_and_points,
)


@dataclass
class OutputSchedule:
    """Simulation-time thresholds for periodic output snapshots."""

    png_interval: float
    xdmf_interval: float
    field_interval: float

    def __post_init__(self):
        self.next_png_time = self.png_interval if self.png_interval > 0.0 else math.inf
        self.next_xdmf_time = self.xdmf_interval if self.xdmf_interval > 0.0 else math.inf
        self.next_field_time = self.field_interval if self.field_interval > 0.0 else math.inf

    def due(self, output_kind, time_s):
        next_name, interval = {
            "png": ("next_png_time", self.png_interval),
            "xdmf": ("next_xdmf_time", self.xdmf_interval),
            "field": ("next_field_time", self.field_interval),
        }[output_kind]
        next_time = getattr(self, next_name)
        if time_s + 1.0e-12 < next_time:
            return False
        while next_time <= time_s + 1.0e-12:
            next_time += interval
        setattr(self, next_name, next_time)
        return True


def initialize_output_state(*, msh, ME, V_scalar, n_grains, scalar_names=None):
    """Create scalar output Functions and their mixed-to-scalar maps."""
    from .regions import collapse_mixed_subspace
    names = scalar_names or ["xi", "phil", "phis", "c", "ux", "uy"] + [f"eta{i + 1}" for i in range(n_grains)]
    scalar_outputs = []
    component_maps = []
    for component, name in enumerate(names):
        V_component, submap = collapse_mixed_subspace(ME, component)
        scalar_outputs.append(fem.Function(V_component, name=name))
        component_maps.append(np.asarray(submap, dtype=np.int64))
    derived_names = (
        "ce", "gb_window", "hxi", "sigma_eff", "reaction_li", "deposition_drive",
        "xi_source_weight", "xit", "overp_li", "overp_c", "eeq", "hydrostatic_stress",
        "overp_mech", "u_magnitude", "liion_source", "dfmechdxi", "eeq_eff",
        "hydrostatic_stress_cathode", "overp_mech_cathode", "stress_flux_drive",
        "hydrostatic_stress_cathode_smooth", "hydrostatic_stress_omega23_smooth",
        "heta", "E1", "nu1", "hydrostatic_stress_omega1_smooth",
        "hydrostatic_stress_omega12_smooth",
    )
    derived_outputs = tuple(fem.Function(V_scalar, name=name) for name in derived_names)
    return {"scalar_outputs": scalar_outputs, "component_maps": component_maps,
            "derived_outputs": derived_outputs}


def refresh_output_fields(state, *, full=True, indices=None):
    """Refresh scalar and derived output functions from a workflow state."""
    update_scalars = state["update_scalar_outputs"]
    update_scalars(
        state["w"], state["scalar_outputs"], state["params"],
        indices=indices, component_maps=state["component_maps"],
    )
    if full:
        state["update_derived"]()


def save_scheduled_output(*, schedule, kind, time_s, output_state, out_dir,
                          V_scalar, scalar_outputs, derived_outputs, B,
                          domain_markers, params, cycle=None, phase=None,
                          step=None):
    """Refresh and write one due PNG, XDMF, or field snapshot."""
    if not schedule.due(kind, time_s):
        return False
    refresh_output_fields(output_state)
    label = f"t_{time_s:010.3f}s"
    if kind == "png":
        save_png_outputs(out_dir, label, V_scalar, scalar_outputs,
                         derived_outputs, B, domain_markers, params)
    elif kind == "xdmf":
        save_sampled_outputs(out_dir, label, V_scalar, scalar_outputs,
                             derived_outputs, B, domain_markers, params)
    elif kind == "field":
        metadata = {"time_s": time_s}
        if step is not None:
            metadata["step"] = step
        if cycle is not None:
            metadata["cycle"] = cycle
        if phase is not None:
            metadata["phase"] = phase
        save_field_snapshot(
            Path(out_dir) / "fields" / f"{label}.npz", V_scalar,
            scalar_outputs, derived_outputs, B, domain_markers, params,
            variables=params.field_output_variables,
            metadata=metadata,
        )
    else:
        raise ValueError(f"Unknown scheduled output kind: {kind}")
    return True


def save_phase_output(*, output_state, out_dir, label, V_scalar,
                      scalar_outputs, derived_outputs, B, domain_markers,
                      params):
    """Write the standard end-of-phase PNG and optional XDMF snapshot."""
    refresh_output_fields(output_state)
    save_png_outputs(out_dir, label, V_scalar, scalar_outputs,
                     derived_outputs, B, domain_markers, params)
    if params.xdmf_interval > 0.0:
        save_sampled_outputs(out_dir, label, V_scalar, scalar_outputs,
                             derived_outputs, B, domain_markers, params)


def finalize_outputs(*, output_state, out_dir, V_scalar, scalar_outputs,
                     derived_outputs, B, domain_markers, params,
                     initial_outputs, extra_outputs=(), rank=0):
    """Refresh and write all final simulation artifacts."""
    output_state["update_scalar_outputs"](
        output_state["w"], scalar_outputs, params,
        component_maps=output_state["component_maps"])
    output_state["update_derived"]()
    save_png_outputs(out_dir, "final", V_scalar, scalar_outputs,
                     derived_outputs, B, domain_markers, params)
    if params.xdmf_interval > 0.0:
        save_sampled_outputs(out_dir, "final", V_scalar, scalar_outputs,
                             derived_outputs, B, domain_markers, params)
    save_delta_outputs(out_dir, V_scalar, initial_outputs, scalar_outputs,
                       B, domain_markers)
    save_final_npz(params.final_npz_file, V_scalar, scalar_outputs,
                   derived_outputs, params, extra_outputs=extra_outputs)
    if rank == 0:
        print(f"已保存循环输出到: {out_dir}")
        print(f"已保存诊断 CSV: {params.diagnostics_file}")
        print(f"已保存最终状态 NPZ: {params.final_npz_file}")


def make_phase_output_callback(*, output_state, out_dir, V_scalar,
                               scalar_outputs, derived_outputs, B,
                               domain_markers, params):
    """Bind standard end-of-phase output to the evolution driver."""
    def finish_phase(phase, _state):
        save_phase_output(output_state=output_state, out_dir=out_dir,
            label=f"cycle{phase.cycle_index:03d}_after_{phase.phase_name}",
            V_scalar=V_scalar, scalar_outputs=scalar_outputs,
            derived_outputs=derived_outputs, B=B,
            domain_markers=domain_markers, params=params)
    return finish_phase
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

def cellwise_output_function(msh, cell_tags, values, name):
    V0 = fem.functionspace(msh, ("DG", 0))
    out = fem.Function(V0, name=name)
    tdim = msh.topology.dim
    index_map = msh.topology.index_map(tdim)
    num_owned = index_map.size_local
    num_cells = num_owned + index_map.num_ghosts
    out.x.array[:] = -1.0
    dm = V0.dofmap.list
    arr = dm.array if hasattr(dm, "array") else np.asarray(dm)
    cell_dofs = arr.reshape((num_cells, -1))[:, 0].astype(np.int32)
    lookup = {int(k): float(v) for k, v in values.items()}
    for cell, tag in zip(np.asarray(cell_tags.indices, dtype=np.int32), np.asarray(cell_tags.values, dtype=np.int32)):
        if 0 <= int(cell) < num_owned:
            out.x.array[cell_dofs[int(cell)]] = lookup.get(int(tag), -1.0)
    out.x.scatter_forward()
    return out


def cell_tag_dg0_function(msh, cell_tags, name="cell_region_tag"):
    tags = sorted({int(v) for v in np.asarray(cell_tags.values, dtype=np.int32)})
    return cellwise_output_function(msh, cell_tags, {tag: tag for tag in tags}, name)


def xdmf_region_fields(msh, cell_tags):
    tags = sorted({int(v) for v in np.asarray(cell_tags.values, dtype=np.int32)})
    raw = cellwise_output_function(msh, cell_tags, {tag: tag for tag in tags}, "cell_region_raw")
    plot = cellwise_output_function(msh, cell_tags, {OMEGA1: 1.0, OMEGA11: 1.0, OMEGA12: 1.0, OMEGA2: 2.0, OMEGA3: 3.0}, "plot_region")
    return (raw, plot,
            cellwise_output_function(msh, cell_tags, {OMEGA1: 1.0, OMEGA11: 1.0, OMEGA12: 1.0}, "omega1_mask"),
            cellwise_output_function(msh, cell_tags, {OMEGA2: 1.0}, "omega2_mask"),
            cellwise_output_function(msh, cell_tags, {OMEGA3: 1.0}, "omega3_mask"))


def point_region_mask_function(V, cell_markers, valid_regions, name):
    out = fem.Function(V, name=name)
    out.x.array[:] = PETSc.ScalarType(-1.0)
    dofs = region_dofs_from_markers(V, cell_markers, valid_regions)
    if dofs.size:
        out.x.array[dofs] = PETSc.ScalarType(1.0)
    out.x.scatter_forward()
    return out


def xdmf_point_region_fields(V, cell_markers):
    return (point_region_mask_function(V, cell_markers, OMEGA1_REGIONS, "omega1_point_mask"),
            point_region_mask_function(V, cell_markers, (OMEGA2,), "omega2_point_mask"),
            point_region_mask_function(V, cell_markers, (OMEGA3,), "omega3_point_mask"))


def mask_function_to_regions(fun, V, cell_markers, valid_regions, fill_value=np.nan):
    valid = restricted_dofs(V, cell_markers, valid_regions)
    keep = np.zeros(fun.x.array.shape[0], dtype=bool); keep[valid] = True
    fun.x.array[~keep] = PETSc.ScalarType(fill_value); fun.x.scatter_forward()


def xdmf_plot_field(source, name, cell_markers, valid_regions):
    out = fem.Function(source.function_space, name=name)
    out.x.array[:] = source.x.array; out.x.scatter_forward()
    mask_function_to_regions(out, out.function_space, cell_markers, valid_regions)
    return out


def xdmf_normalized_plot_field(source, name, cell_markers, valid_regions, power=1.0):
    out = fem.Function(source.function_space, name=name)
    dofs = restricted_dofs(source.function_space, cell_markers, valid_regions)
    local_max = float(np.nanmax(np.abs(source.x.array.real[dofs]))) if dofs.size else 0.0
    global_max = float(source.function_space.mesh.comm.allreduce(local_max, op=MPI.MAX))
    if global_max <= 0.0 or not math.isfinite(global_max): out.x.array[:] = PETSc.ScalarType(0.0)
    else:
        values = np.clip(source.x.array.real / global_max, 0.0, 1.0)
        out.x.array[:] = (values ** float(power)).astype(out.x.array.dtype)
    out.x.scatter_forward(); mask_function_to_regions(out, out.function_space, cell_markers, valid_regions)
    return out


def xdmf_eta_gb_plot_field(scalar_outputs, name, cell_markers, p, power=1.0,
                           xi_cutoff=None, xi_suppression_power=None):
    etas = [fun for fun in scalar_outputs if fun.name.startswith("eta")]
    if not etas:
        return None
    out = fem.Function(etas[0].function_space, name=name)
    values = np.zeros_like(out.x.array.real)
    clipped = [np.clip(fun.x.array.real, 0.0, 1.0) for fun in etas]
    for i in range(len(clipped)):
        for j in range(i + 1, len(clipped)):
            values += clipped[i] * clipped[j]
    values = p.b_scale * np.minimum(p.b_clip, np.maximum(0.0, values))
    scalars = {fun.name: fun for fun in scalar_outputs}
    xi_fun = scalars.get("xi")
    if xi_fun is not None:
        xi = np.clip(xi_fun.x.array.real, 0.0, 1.0)
        if xi_cutoff is not None:
            values = np.where(xi < float(xi_cutoff), values, 0.0)
        if xi_suppression_power is not None:
            values *= (1.0 - xi) ** float(xi_suppression_power)
    out.x.array[:] = values.astype(out.x.array.dtype)
    out.x.scatter_forward()
    return xdmf_normalized_plot_field(out, name, cell_markers, OMEGA1_REGIONS, power)


def composite_region_field(msh, V, cell_markers, xi_fun, B_fun):
    V0 = fem.functionspace(msh, ("DG", 0))
    out = fem.Function(V0, name="composite_region_field")
    nlocal = msh.topology.index_map(msh.topology.dim).size_local
    values = np.zeros(nlocal, dtype=np.float64)
    cells, _points, markers = cells_and_points(V, cell_markers)
    xi_values, B_values = xi_fun.x.array.real, B_fun.x.array.real
    dofs = restricted_dofs(V, cell_markers, OMEGA1_REGIONS)
    local_max = float(np.nanmax(np.abs(B_values[dofs]))) if dofs.size else 0.0
    Bmax = float(msh.comm.allreduce(local_max, op=MPI.MAX))
    threshold = 0.35 * Bmax if Bmax > 0.0 and math.isfinite(Bmax) else math.inf
    for cell in range(nlocal):
        marker = int(markers[cell])
        code = 1.0 if marker in OMEGA1_REGIONS else 2.0 if marker == OMEGA2 else 3.0 if marker == OMEGA3 else 0.0
        if marker in OMEGA1_REGIONS:
            cdofs = cells[cell]
            if float(np.mean(xi_values[cdofs])) >= 0.5: code = 4.0
            elif float(np.max(np.abs(B_values[cdofs]))) >= threshold: code = 5.0
        values[cell] = code
    out.x.array[:nlocal] = values.astype(out.x.array.dtype)
    out.x.scatter_forward()
    return out


def xdmf_plot_fields(scalar_outputs, derived_outputs, B, cell_markers, p, diagnostic_outputs=()):
    scalars = {fun.name: fun for fun in scalar_outputs}
    derived = {fun.name: fun for fun in derived_outputs}
    fields = []
    if "xi" in scalars: fields.append(xdmf_plot_field(scalars["xi"], "xi_plot", cell_markers, OMEGA1_REGIONS))
    if "ce" in derived: fields.append(xdmf_plot_field(derived["ce"], "ce_plot", cell_markers, OMEGA1_REGIONS))
    fields.extend((xdmf_plot_field(B, "B_plot", cell_markers, OMEGA1_REGIONS),
                   xdmf_normalized_plot_field(B, "B_norm_plot", cell_markers, OMEGA1_REGIONS),
                   xdmf_normalized_plot_field(B, "B_sharp_plot", cell_markers, OMEGA1_REGIONS, 3.0)))
    for name, kwargs in (("eta_gb_plot", {}), ("eta_gb_no_li_plot", {"xi_cutoff": .001}),
                         ("eta_gb_effective_plot", {"xi_cutoff": .5, "xi_suppression_power": 2.0}),
                         ("eta_gb_sharp_plot", {"power": 3.0})):
        field = xdmf_eta_gb_plot_field(scalar_outputs, name, cell_markers, p, **kwargs)
        if field is not None: fields.append(field)
    region_map = {"hydrostatic_stress": ("hydrostatic_stress_omega1_plot", OMEGA1_REGIONS),
                  "hydrostatic_stress_cathode": ("hydrostatic_stress_cathode_plot", (OMEGA3,)),
                  "hydrostatic_stress_omega1_smooth": ("hydrostatic_stress_omega1_smooth_plot", OMEGA1_REGIONS),
                  "hydrostatic_stress_omega12_smooth": ("hydrostatic_stress_omega12_smooth_plot", OMEGA1_REGIONS + (OMEGA2,)),
                  "hydrostatic_stress_cathode_smooth": ("hydrostatic_stress_cathode_smooth_plot", (OMEGA3,)),
                  "hydrostatic_stress_omega23_smooth": ("hydrostatic_stress_omega23_smooth_plot", (OMEGA2, OMEGA3))}
    for key, (name, regions) in region_map.items():
        if key in derived: fields.append(xdmf_plot_field(derived[key], name, cell_markers, regions))
    return tuple(fields)


def _gather_png_payload(comm, *arrays):
    if comm.size == 1:
        return arrays
    gathered = comm.gather(arrays, root=0)
    if comm.rank != 0:
        return None
    payload = [[] for _ in arrays]; offset = 0
    for chunk in gathered:
        if not chunk or chunk[0].size == 0:
            continue
        payload[0].append(chunk[0])
        if len(chunk) > 1: payload[1].append(chunk[1] + offset)
        for i in range(2, len(chunk)): payload[i].append(chunk[i])
        offset += chunk[0].shape[0]
    result = []
    for i, parts in enumerate(payload):
        if not parts:
            result.append(np.empty((0, 2), dtype=np.float64) if i == 0 else np.empty((0, 3), dtype=np.int32) if i == 1 else np.empty(0, dtype=np.float64))
        else:
            result.append(np.vstack(parts) if i < 2 else np.concatenate(parts))
    return tuple(result)


def field_output_variable_set(variables):
    if variables is None:
        return {"all"}
    if isinstance(variables, str):
        variables = tuple(part.strip() for part in variables.split(","))
    names = {str(name).strip() for name in variables if str(name).strip()}
    return names or {"all"}


def triangulate_plot_cells(cells):
    cells = np.asarray(cells)
    if cells.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    return cells.astype(np.int32, copy=False) if cells.shape[1] == 3 else np.vstack((cells[:, [0, 1, 2]], cells[:, [0, 2, 3]])).astype(np.int32)


def safe_xdmf_field_prefix(prefix):
    return str(prefix).replace(".", "p").replace("-", "m").replace(":", "_")


def write_sampled_xdmf(path, grid_name, points, cells, point_values, cell_values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    def rows(array):
        array = np.asarray(array)
        if array.ndim == 1: return " ".join(f"{float(v):.17g}" for v in array)
        if np.issubdtype(array.dtype, np.integer): return "\n".join(" ".join(str(int(v)) for v in row) for row in array)
        return "\n".join(" ".join(f"{float(v):.17g}" for v in row) for row in array)
    name = safe_xdmf_field_prefix(grid_name)
    point_attr = f"{name}_sampled_point"
    cell_attr = f"{name}_sampled_cell"
    xml = f'''<?xml version="1.0" ?>
<Xdmf Version="3.0"><Domain><Grid Name="{name}_sampled" GridType="Uniform">
<Topology TopologyType="Triangle" NumberOfElements="{cells.shape[0]}"><DataItem Dimensions="{cells.shape[0]} 3" NumberType="Int" Format="XML">{rows(cells)}</DataItem></Topology>
<Geometry GeometryType="XY"><DataItem Dimensions="{points.shape[0]} 2" NumberType="Float" Precision="8" Format="XML">{rows(points)}</DataItem></Geometry>
<Attribute Name="{point_attr}" AttributeType="Scalar" Center="Node"><DataItem Dimensions="{point_values.shape[0]}" NumberType="Float" Precision="8" Format="XML">{rows(point_values)}</DataItem></Attribute>
<Attribute Name="{cell_attr}" AttributeType="Scalar" Center="Cell"><DataItem Dimensions="{cell_values.shape[0]}" NumberType="Float" Precision="8" Format="XML">{rows(cell_values)}</DataItem></Attribute>
</Grid></Domain></Xdmf>'''
    path.write_text(xml, encoding="utf-8")


def save_field_png(V, field, filename, title, cmap="viridis", symmetric=False,
                   vmin=None, vmax=None, overlay=None, cell_markers=None,
                   valid_regions=None, fill_outside=None):
    if not HAS_MATPLOTLIB: return
    filename = Path(filename); filename.parent.mkdir(parents=True, exist_ok=True)
    cells, points, markers = cells_and_points(V, cell_markers, owned_only=True)
    if valid_regions is not None and markers is not None:
        regions = (valid_regions,) if isinstance(valid_regions, (int, np.integer)) else tuple(valid_regions)
        keep = np.isin(markers, regions)
        if fill_outside is None: cells = cells[keep]
    if fill_outside is not None and valid_regions is not None and markers is not None:
        dofs = np.unique(cells[np.isin(markers, regions)].reshape(-1))
        values = field.x.array.real.copy(); values[np.setdiff1d(np.arange(values.size), dofs)] = fill_outside
    else: values = field.x.array.real.copy()
    triangles = cells if cells.size and cells.shape[1] == 3 else np.vstack((cells[:, [0,1,2]], cells[:, [0,2,3]])).astype(np.int32) if cells.size else np.empty((0,3), dtype=np.int32)
    overlay_values = overlay.x.array.real.copy() if overlay is not None else np.empty(0, dtype=np.float64)
    clim_values = values[np.unique(cells.reshape(-1))] if cells.size else np.empty(0)
    payload = _gather_png_payload(V.mesh.comm, points, triangles, values, overlay_values, clim_values)
    if payload is None or payload[0].size == 0 or payload[1].size == 0: return
    points, triangles, values, overlay_values, clim_values = payload
    fig, ax = plt.subplots(figsize=(7, 8), dpi=180)
    tri = mtri.Triangulation(points[:,0], points[:,1], triangles)
    finite = clim_values[np.isfinite(clim_values)]
    if symmetric:
        lim = max(float(np.max(np.abs(finite))) if finite.size else 0.0, 1e-30); vmin, vmax = -lim, lim
    try:
        low = float(np.min(finite)) if vmin is None and finite.size else vmin
        high = float(np.max(finite)) if vmax is None and finite.size else vmax
        if low is None or high is None or finite.size == 0: raise ValueError("no finite values to plot")
        if np.isclose(low, high):
            pad = max(abs(low), 1.0) * 1.0e-9; low -= pad; high += pad
        color = ax.tricontourf(tri, values, levels=np.linspace(low, high, 96), cmap=cmap,
                               vmin=low, vmax=high, extend="both", antialiased=False)
    except Exception:
        color = ax.tripcolor(tri, values, shading="gouraud", cmap=cmap,
                             edgecolors="none", linewidth=0.0, antialiased=False)
        if vmin is not None and vmax is not None: color.set_clim(vmin, vmax)
    if overlay is not None:
        ax.tricontour(tri, overlay_values, levels=[0.35], colors="black", linewidths=.95, alpha=.90)
    ax.set_aspect("equal"); ax.set_xlabel("x / L"); ax.set_ylabel("y / L"); ax.set_title(title)
    fig.colorbar(color, ax=ax); fig.tight_layout(); fig.savefig(filename); plt.close(fig)


def save_sampled_field_png(
    V, field, filename, title, cmap="turbo", vmin=0.0, vmax=1.0,
    cell_markers=None, valid_regions=None, order=5,
):
    if not HAS_MATPLOTLIB:
        return
    comm = V.mesh.comm
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    sample_points, sample_cells, parent_tris, sample_bary = build_subsampled_triangles(
        V, cell_markers=cell_markers, valid_regions=valid_regions,
        order=order, owned_only=True,
    )
    if sample_points.size == 0:
        gathered = _gather_png_payload(comm, sample_points, sample_cells, np.empty(0, dtype=np.float64))
        if gathered is None:
            return
        sample_points, sample_cells, sample_values = gathered
        if sample_points.size == 0 or sample_cells.size == 0:
            return
    else:
        values = field.x.array.real
        samples_per_parent = sample_bary.shape[0] // parent_tris.shape[0]
        sample_values = []
        for parent_index in range(parent_tris.shape[0]):
            bary_slice = sample_bary[parent_index * samples_per_parent:(parent_index + 1) * samples_per_parent]
            tri = parent_tris[parent_index * samples_per_parent]
            sample_values.append(bary_slice @ values[tri])
        sample_values = np.concatenate(sample_values)
        gathered = _gather_png_payload(comm, sample_points, sample_cells, sample_values)
        if gathered is None:
            return
        sample_points, sample_cells, sample_values = gathered
    triangulation = mtri.Triangulation(sample_points[:, 0], sample_points[:, 1], sample_cells)
    fig, ax = plt.subplots(figsize=(7.2, 8.4), dpi=180)
    color = ax.tripcolor(triangulation, sample_values, shading="flat", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xlabel("x / L")
    ax.set_ylabel("y / L")
    ax.set_title(title)
    fig.colorbar(color, ax=ax)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def save_sampled_interface_indicator_png(
    V, xi_fun, eta_funs, p, filename, title, cmap="turbo", vmin=0.0,
    vmax=1.0, cell_markers=None, valid_regions=None, order=5,
):
    if not HAS_MATPLOTLIB:
        return
    comm = V.mesh.comm
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    sample_points, sample_cells, parent_tris, sample_bary = build_subsampled_triangles(
        V, cell_markers=cell_markers, valid_regions=valid_regions,
        order=order, owned_only=True,
    )
    if sample_points.size == 0:
        gathered = _gather_png_payload(comm, sample_points, sample_cells, np.empty(0, dtype=np.float64))
        if gathered is None:
            return
        sample_points, sample_cells, values = gathered
        if sample_points.size == 0 or sample_cells.size == 0:
            return
    else:
        def sample_scalar(values):
            return np.sum(sample_bary * values[parent_tris], axis=1)

        xi_sample = sample_scalar(xi_fun.x.array.real)
        eta_samples = np.vstack([sample_scalar(eta.x.array.real) for eta in eta_funs]).T
        eta_clip = np.clip(eta_samples, 0.0, 1.0)
        gb_window = np.sum(
            (1.0 - eta_clip) * (np.abs(eta_samples - 0.5) < p.rho), axis=1
        )
        values = xi_sample + (1.0 - xi_sample) ** 2 * gb_window
        gathered = _gather_png_payload(comm, sample_points, sample_cells, values)
        if gathered is None:
            return
        sample_points, sample_cells, values = gathered
    triangulation = mtri.Triangulation(sample_points[:, 0], sample_points[:, 1], sample_cells)
    fig, ax = plt.subplots(figsize=(7.2, 8.4), dpi=180)
    color = ax.tripcolor(triangulation, values, shading="flat", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xlabel("x / L")
    ax.set_ylabel("y / L")
    ax.set_title(title)
    fig.colorbar(color, ax=ax)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def _sampled_scalar_data(V, field, cell_markers, valid_regions, order):
    points, cells, parent_tris, bary = build_subsampled_triangles(
        V, cell_markers=cell_markers, valid_regions=valid_regions,
        order=order, owned_only=True,
    )
    if points.size:
        values = np.sum(bary * field.x.array.real[parent_tris], axis=1)
        cell_values = np.mean(values[cells], axis=1)
    else:
        values = np.empty(0, dtype=np.float64)
        cell_values = np.empty(0, dtype=np.float64)
    gathered = _gather_png_payload(V.mesh.comm, points, cells, values, cell_values)
    return gathered


def save_sampled_field_xdmf(V, field, filename, grid_name, cell_markers=None,
                            valid_regions=None, order=5):
    gathered = _sampled_scalar_data(V, field, cell_markers, valid_regions, order)
    if gathered is None:
        return
    points, cells, values, cell_values = gathered
    if points.size:
        write_sampled_xdmf(filename, grid_name, points, cells, values, cell_values)


def save_sampled_interface_indicator_xdmf(V, xi_fun, eta_funs, p, filename,
                                          grid_name, cell_markers=None,
                                          valid_regions=None, order=5):
    comm = V.mesh.comm
    points, cells, parent_tris, bary = build_subsampled_triangles(
        V, cell_markers=cell_markers, valid_regions=valid_regions,
        order=order, owned_only=True,
    )
    if points.size:
        xi = np.sum(bary * xi_fun.x.array.real[parent_tris], axis=1)
        eta = np.vstack([
            np.sum(bary * fun.x.array.real[parent_tris], axis=1) for fun in eta_funs
        ]).T
        gb = np.sum((1.0 - np.clip(eta, 0.0, 1.0)) * (np.abs(eta - 0.5) < p.rho), axis=1)
        values = xi + (1.0 - xi) ** 2 * gb
        cell_values = np.mean(values[cells], axis=1)
    else:
        values = np.empty(0, dtype=np.float64)
        cell_values = np.empty(0, dtype=np.float64)
    gathered = _gather_png_payload(comm, points, cells, values, cell_values)
    if gathered is None:
        return
    points, cells, values, cell_values = gathered
    if points.size:
        write_sampled_xdmf(filename, grid_name, points, cells, values, cell_values)


def save_final_npz(path, V, scalar_outputs, derived_outputs, params, extra_outputs=()):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {fun.name: fun.x.array.real.copy()
              for fun in list(scalar_outputs) + list(derived_outputs) + list(extra_outputs)}
    arrays["dof_coordinates"] = V.tabulate_dof_coordinates()[:, :2]
    arrays["params"] = np.array(str(asdict(params)))
    np.savez(path, **arrays)


def save_delta_outputs(out_dir, V, initial_outputs, final_outputs, B, cell_markers):
    # Delta preview output is intentionally disabled in the existing runner.
    return


def _output_groups(cell_markers):
    return {
        "xi": OMEGA1_REGIONS,
        "phil": OMEGA1_REGIONS + (OMEGA2,),
        "phis": (OMEGA2,), "c": (OMEGA3,),
    }


def save_png_outputs(out_dir, prefix, V, scalar_outputs, derived_outputs, B, cell_markers, p):
    out_dir = Path(out_dir) / "png" / str(prefix)
    xi_fun, phil_fun, phis_fun, c_fun = scalar_outputs[:4]
    reg_xi = OMEGA1_REGIONS
    reg_phil = OMEGA1_REGIONS + (OMEGA2,)
    reg_phis = (OMEGA2,)
    reg_cathode = (OMEGA3,)
    save_field_png(V, xi_fun, out_dir / "xi.png", f"{prefix}: xi", vmin=0.0, vmax=1.0,
                   overlay=B, cell_markers=cell_markers, valid_regions=reg_xi)
    eta_funs = [fun for fun in scalar_outputs[6:] if fun.name.startswith("eta")]
    if eta_funs:
        save_sampled_interface_indicator_png(
            V, xi_fun, eta_funs, p, out_dir / "interface_indicator.png",
            f"{prefix}: interface indicator", cmap="turbo", vmin=0.0, vmax=1.0,
            cell_markers=cell_markers, valid_regions=reg_xi,
        )
    for field, name, title, regions in (
        (phil_fun, "phil.png", "phi_l [V]", reg_phil),
        (phis_fun, "phis.png", "phi_s [V]", reg_phis),
        (c_fun, "soc.png", "particle SOC", reg_cathode),
    ):
        save_field_png(V, field, out_dir / name, f"{prefix}: {title}",
                       cell_markers=cell_markers, valid_regions=regions)
    save_field_png(V, c_fun, out_dir / "soc_full.png", f"{prefix}: particle SOC, full domain",
                   vmin=0.0, vmax=1.0, cell_markers=cell_markers,
                   valid_regions=reg_cathode, fill_outside=0.0)
    fields = {fun.name: fun for fun in derived_outputs}
    stress_fields = (
        ("hydrostatic_stress", "stress_o1_raw.png", "Omega1 stress [Pa]", reg_xi),
        ("hydrostatic_stress_omega1_smooth", "stress_o1.png", "Omega1 stress [Pa]", reg_xi),
        ("hydrostatic_stress_omega12_smooth", "stress_o12.png", "Omega1+Omega2 stress [Pa]", OMEGA1_REGIONS + (OMEGA2,)),
        ("hydrostatic_stress_cathode_smooth", "stress_cathode.png", "cathode stress [Pa]", reg_cathode),
        ("hydrostatic_stress_omega23_smooth", "stress_o23.png", "Omega2+Omega3 stress [Pa]", (OMEGA2, OMEGA3)),
    )
    for key, name, title, regions in stress_fields:
        field = fields.get(key)
        if field is not None:
            save_field_png(V, field, out_dir / name, f"{prefix}: {title}",
                           cmap="coolwarm", symmetric=True,
                           overlay=B if regions == reg_xi else None,
                           cell_markers=cell_markers, valid_regions=regions)
    if prefix in ("initial", "final", "after_charge", "after_discharge"):
        save_field_png(V, B, out_dir / ("initial_B.png" if prefix == "initial" else "gb_indicator.png"),
                       f"{prefix}: GB indicator B", cmap="magma", vmin=0.0, vmax=1.0,
                       overlay=B, cell_markers=cell_markers, valid_regions=reg_xi)


def save_sampled_outputs(out_dir, prefix, V, scalar_outputs, derived_outputs, B, cell_markers, p):
    out_dir = Path(out_dir) / "sampled_xdmf" / str(prefix)
    xi_fun, phil_fun, phis_fun, c_fun = scalar_outputs[:4]
    reg_xi = OMEGA1_REGIONS
    save_sampled_field_xdmf(V, xi_fun, out_dir / "xi_sampled.xdmf", "xi", cell_markers, reg_xi)
    save_sampled_field_xdmf(V, phil_fun, out_dir / "phil_sampled.xdmf", "phil", cell_markers, OMEGA1_REGIONS + (OMEGA2,))
    save_sampled_field_xdmf(V, phis_fun, out_dir / "phis_sampled.xdmf", "phis", cell_markers, (OMEGA2,))
    save_sampled_field_xdmf(V, c_fun, out_dir / "c_soc_sampled.xdmf", "c_soc", cell_markers, (OMEGA3,))
    eta_funs = [fun for fun in scalar_outputs[6:] if fun.name.startswith("eta")]
    if eta_funs:
        save_sampled_interface_indicator_xdmf(V, xi_fun, eta_funs, p,
                                              out_dir / "interface_indicator_sampled.xdmf",
                                              "interface_indicator", cell_markers, reg_xi)
    fields = {fun.name: fun for fun in derived_outputs}
    for key, stem, regions in (
        ("hydrostatic_stress", "hydrostatic_stress_omega1", reg_xi),
        ("hydrostatic_stress_omega1_smooth", "hydrostatic_stress_omega1_smooth", reg_xi),
        ("hydrostatic_stress_omega12_smooth", "hydrostatic_stress_omega12_smooth", OMEGA1_REGIONS + (OMEGA2,)),
        ("hydrostatic_stress_cathode_smooth", "hydrostatic_stress_cathode_smooth", (OMEGA3,)),
        ("hydrostatic_stress_omega23_smooth", "hydrostatic_stress_omega23_smooth", (OMEGA2, OMEGA3)),
    ):
        field = fields.get(key)
        if field is not None:
            save_sampled_field_xdmf(V, field, out_dir / f"{stem}_sampled.xdmf", stem,
                                    cell_markers, regions)
    save_sampled_field_xdmf(V, B, out_dir / "B_sampled.xdmf", "B", cell_markers, reg_xi)


def _png_style_field_payload(
    V,
    field,
    cell_markers,
    valid_regions=None,
    fill_outside=None,
    overlay=None,
):
    """Return the same plotting payload used by save_field_png()."""
    comm = V.mesh.comm
    values = field.x.array.real.copy()
    cells_all, points, markers = cells_and_points(V, cell_markers, owned_only=True)
    if fill_outside is None:
        if valid_regions is None or markers is None:
            cells = cells_all
        else:
            regions = (
                (valid_regions,)
                if isinstance(valid_regions, (int, np.integer))
                else tuple(valid_regions)
            )
            cells = cells_all[np.isin(markers, regions)]
        plot_dofs = (
            np.unique(cells.reshape(-1)).astype(np.int32)
            if cells.size
            else np.empty(0, dtype=np.int32)
        )
        clim_values = values[plot_dofs].copy() if plot_dofs.size else np.empty(0)
    else:
        if valid_regions is not None and markers is not None:
            regions = (
                (valid_regions,)
                if isinstance(valid_regions, (int, np.integer))
                else tuple(valid_regions)
            )
            keep_cells = np.isin(markers, regions)
            valid_dofs = (
                np.unique(cells_all[keep_cells].reshape(-1))
                if np.any(keep_cells)
                else np.empty(0, dtype=np.int32)
            )
            invalid = np.ones(values.shape[0], dtype=bool)
            invalid[valid_dofs] = False
            values[invalid] = float(fill_outside)
            clim_values = values[valid_dofs].copy()
        else:
            clim_values = values.copy()
        cells = cells_all
    triangles = triangulate_plot_cells(cells)
    overlay_values = (
        overlay.x.array.real.copy() if overlay is not None else np.empty(0, dtype=np.float64)
    )
    gathered = _gather_png_payload(comm, points, triangles, values, overlay_values, clim_values)
    if gathered is None:
        return None
    points, triangles, values, overlay_values, clim_values = gathered
    if triangles.size == 0:
        return None
    return {
        "points": points,
        "cells": triangles.astype(np.int32, copy=False),
        "values": values,
        "overlay_values": overlay_values,
        "clim_values": clim_values,
    }


def _png_style_ce_payload(V, xi_fun, eta_funs, p: Params, cell_markers, valid_regions, order: int = 5):
    """Return the same sampled interface-indicator payload used by the PNG output."""
    comm = V.mesh.comm
    sample_points, sample_cells, parent_tris, sample_bary = build_subsampled_triangles(
        V,
        cell_markers=cell_markers,
        valid_regions=valid_regions,
        order=order,
        owned_only=True,
    )
    if sample_points.size == 0:
        gathered = _gather_png_payload(
            comm, sample_points, sample_cells, np.empty(0, dtype=np.float64)
        )
        if gathered is None:
            return None
        sample_points, sample_cells, ce_values = gathered
    else:
        def sample_scalar(values):
            return np.sum(sample_bary * values[parent_tris], axis=1)

        def box_indicator_np(value, half_width):
            return (np.abs(value) < half_width).astype(np.float64)

        xi_sample = sample_scalar(xi_fun.x.array.real)
        eta_samples = np.vstack([sample_scalar(eta_i.x.array.real) for eta_i in eta_funs]).T
        eta_clip = np.clip(eta_samples, 0.0, 1.0)
        gb_window = np.sum(
            (1.0 - eta_clip) * box_indicator_np(eta_samples - 0.5, p.rho),
            axis=1,
        )
        ce_values = xi_sample + (1.0 - xi_sample) ** 2 * gb_window
        gathered = _gather_png_payload(comm, sample_points, sample_cells, ce_values)
        if gathered is None:
            return None
        sample_points, sample_cells, ce_values = gathered
    if sample_points.size == 0 or sample_cells.size == 0:
        return None
    return {
        "points": sample_points,
        "cells": sample_cells.astype(np.int32, copy=False),
        "values": ce_values,
        "overlay_values": np.empty(0, dtype=np.float64),
        "clim_values": ce_values,
    }


def _png_style_regions_for_field(name):
    reg_xi = OMEGA1_REGIONS
    reg_phil = OMEGA1_REGIONS + (OMEGA2,)
    reg_phis = (OMEGA2,)
    reg_cathode = (OMEGA3,)
    if name == "xi" or name == "B" or name.startswith("eta"):
        return reg_xi
    if name == "phil":
        return reg_phil
    if name == "phis":
        return reg_phis
    if name == "c":
        return reg_cathode
    if name in ("eeq", "eeq_eff", "hydrostatic_stress_cathode", "overp_mech_cathode", "stress_flux_drive", "hydrostatic_stress_cathode_smooth"):
        return reg_cathode
    if name in ("hydrostatic_stress", "hydrostatic_stress_omega1_smooth", "overp_mech", "reaction_li", "deposition_drive", "xi_source_weight", "xit", "overp_li", "liion_source", "dfmechdxi", "heta", "E1", "nu1", "gb_window", "hxi", "sigma_eff"):
        return reg_xi
    if name in ("hydrostatic_stress_omega12_smooth",):
        return OMEGA1_REGIONS + (OMEGA2,)
    if name in ("hydrostatic_stress_omega23_smooth",):
        return (OMEGA2, OMEGA3)
    return None


def save_field_snapshot(
    path,
    V,
    scalar_outputs,
    derived_outputs,
    B,
    cell_markers,
    params,
    variables=("all",),
    extra_outputs=(),
    metadata=None,
):
    """Save one PNG-style software plotting snapshot.

    Each variable is stored as `<name>_points`, `<name>_cells`, and
    `<name>_values`. These arrays are the same payload the current PNG code
    uses, including region filtering and the special sampled ce calculation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    requested = field_output_variable_set(variables)
    save_all = "all" in requested or "*" in requested
    fields = {fun.name: fun for fun in list(scalar_outputs) + list(derived_outputs)}
    fields["B"] = B
    if "u" in requested and "u_magnitude" in fields:
        requested.add("u_magnitude")
    field_names = sorted(fields) if save_all else [name for name in sorted(fields) if name in requested]
    eta_funs = [fun for fun in scalar_outputs[6:] if fun.name.startswith("eta")]

    arrays = {
        "field_names": np.asarray(field_names, dtype=str),
        "plot_payload": np.array("png_style"),
        "params": np.array(str(asdict(params))),
    }
    saved_names = []
    skipped_names = []
    for name in field_names:
        if name == "ce":
            payload = _png_style_ce_payload(V, scalar_outputs[0], eta_funs, params, cell_markers, OMEGA1_REGIONS)
        else:
            overlay = B if name in ("xi", "B", "hydrostatic_stress", "hydrostatic_stress_omega1_smooth") else None
            payload = _png_style_field_payload(
                V,
                fields[name],
                cell_markers,
                valid_regions=_png_style_regions_for_field(name),
                fill_outside=0.0 if name == "c_soc_full" else None,
                overlay=overlay,
            )
        if payload is None:
            skipped_names.append(name)
            continue
        saved_names.append(name)
        arrays[f"{name}_points"] = payload["points"]
        arrays[f"{name}_cells"] = payload["cells"]
        arrays[f"{name}_values"] = payload["values"]
        arrays[f"{name}_clim_values"] = payload["clim_values"]
        arrays[f"{name}_overlay_values"] = payload["overlay_values"]
    arrays["field_names"] = np.asarray(saved_names, dtype=str)
    arrays["skipped_field_names"] = np.asarray(skipped_names, dtype=str)
    if metadata:
        for key, value in metadata.items():
            arrays[f"meta_{key}"] = np.array(value)
    np.savez(path, **arrays)


def save_latest_state(
    reason, w, w_n, scalar_outputs, derived_outputs, params, scalar_component_maps,
    V_scalar, B, domain_markers, out_dir, msh, update_scalar_outputs_fn,
    update_derived_fn, save_png_outputs_fn, save_final_npz_fn,
    extra_output_fields_fn,
):
    """Persist the last accepted nonlinear state and its diagnostic previews."""
    w.x.array[:] = w_n.x.array
    w.x.scatter_forward()
    update_scalar_outputs_fn(
        w, scalar_outputs, params, component_maps=scalar_component_maps,
    )
    update_derived_fn()
    latest_npz = Path(params.final_npz_file).with_name("latest_state.npz")
    save_final_npz_fn(
        latest_npz, V_scalar, scalar_outputs, derived_outputs,
        params, extra_outputs=extra_output_fields_fn(),
    )
    save_png_outputs_fn(
        out_dir, "latest", V_scalar, scalar_outputs, derived_outputs,
        B, domain_markers, params,
    )
    if msh.comm.rank == 0:
        print(f"已保存最后一个成功步状态: {latest_npz}")
        print(f"保存原因: {reason}")


def make_latest_state_saver(
    w, w_n, scalar_outputs, derived_outputs, params, scalar_component_maps,
    V_scalar, B, domain_markers, out_dir, msh, update_scalar_outputs_fn,
    update_derived_fn, save_png_outputs_fn, save_final_npz_fn,
    extra_output_fields_fn,
):
    """Bind the current simulation objects to the latest-state writer."""
    def save(reason):
        return save_latest_state(
            reason, w, w_n, scalar_outputs, derived_outputs, params,
            scalar_component_maps, V_scalar, B, domain_markers, out_dir, msh,
            update_scalar_outputs_fn, update_derived_fn, save_png_outputs_fn,
            save_final_npz_fn, extra_output_fields_fn,
        )

    return save


def derived_output_fields(hydro_omega1_dg0, hydro_cathode_dg0, hydro_omega2_dg0,
                          hydro_omega12_dg0, hydro_omega23_dg0):
    """Return the additional DG0 fields included in final state output."""
    return (hydro_omega1_dg0, hydro_cathode_dg0, hydro_omega2_dg0,
            hydro_omega12_dg0, hydro_omega23_dg0)

__all__ = ["OutputSchedule", "make_latest_state_saver", "save_final_npz", "save_png_outputs", "save_sampled_outputs", "save_delta_outputs", "save_phase_output", "make_phase_output_callback", "finalize_outputs",
           "save_field_png", "save_sampled_field_png", "save_sampled_interface_indicator_png",
           "save_sampled_field_xdmf", "save_sampled_interface_indicator_xdmf",
           "field_output_variable_set", "triangulate_plot_cells", "save_latest_state",
           "derived_output_fields"]
