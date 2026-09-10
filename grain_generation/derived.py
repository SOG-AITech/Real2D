from __future__ import annotations

import ufl
from dolfinx import fem

from .initialization import comsol_step_initial_xi_ufl


def smooth_box_indicator_ufl(value, half_width, smoothing):
    eps = max(float(smoothing), 1.0e-12)
    return 1.0 / (1.0 + ufl.exp((abs(value) - half_width) / eps))


def interpolate_comsol_ce(msh, V_out, eta_outputs, cell_tags, grain_tag, cap_tag, p):
    eta_clips = [ufl.max_value(0.0, ufl.min_value(1.0, eta_i)) for eta_i in eta_outputs]
    gb_window = sum(
        (1.0 - eta_clip) * smooth_box_indicator_ufl(eta_i - 0.5, p.rho, p.gb_window_smoothing)
        for eta_i, eta_clip in zip(eta_outputs, eta_clips)
    )
    xi_init = comsol_step_initial_xi_ufl(msh, p) if cap_tag is not None else 0.0
    ce_expr = xi_init + (1.0 - xi_init) ** 2 * gb_window
    ce_fun = fem.Function(V_out, name="ce")
    points = V_out.element.interpolation_points
    if callable(points):
        points = points()
    ce_fun.interpolate(fem.Expression(ce_expr, points))
    # ``V_out`` lives on the grain submesh, while ``cell_tags`` belongs to
    # the parent mesh.  Their cell indices are not interchangeable.  The
    # grain submesh already excludes the cap region, so there are no cap dofs
    # to overwrite here.
    ce_fun.x.scatter_forward()
    return ce_fun


def interpolate_grain_boundary_indicator(msh, etas, V_out, p, y_cut=None, cell_tags=None, cap_tag=None, grain_tag=None):
    b_raw = sum(etas[i] * etas[j] for i in range(p.n_grains) for j in range(i + 1, p.n_grains))
    b_expr = p.b_scale * ufl.min_value(p.b_clip, ufl.max_value(0.0, b_raw))
    b_fun = fem.Function(V_out, name="grain_boundary_B")
    points = V_out.element.interpolation_points
    if callable(points):
        points = points()
    b_fun.interpolate(fem.Expression(b_expr, points))
    b_fun.x.scatter_forward()
    return b_fun


def interpolate_grain_boundary_indicator_from_outputs(msh, eta_outputs, V_out, p):
    return interpolate_grain_boundary_indicator(msh, eta_outputs, V_out, p)


def update_derived_fields(mesh_state, field_state, params):
    field_state.B = interpolate_grain_boundary_indicator_from_outputs(
        mesh_state.grain_mesh, field_state.eta_outputs, field_state.V_scalar, params
    )
    field_state.ce = interpolate_comsol_ce(
        mesh_state.grain_mesh, field_state.V_scalar, field_state.eta_outputs,
        mesh_state.cell_tags, mesh_state.grain_tag, mesh_state.cap_tag, params
    )
    return field_state
