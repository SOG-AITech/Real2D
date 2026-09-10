"""UFL equation and weak-form construction."""

import ufl

from .config import GAMMA_A, GAMMA_C, GAMMA_S, OMEGA2, OMEGA3, GAMMA_S_OMEGA2_SIDE, GAMMA_S_OMEGA3_SIDE
from .constitutive import strain, xi_anisotropic_gradient_term
from .constitutive import (
    clip01_ufl, eeq_from_soc, h, hp, mechanical_energy_2d,
    plane_stress_hydrostatic_stress, plane_stress_tensor,
)
from .fields import grain_boundary_indicator_expr, grain_boundary_window_expr


def bdf_time_derivative(u, u_n, u_nm1, a0, a1, a2, dt):
    return (a0 * u + a1 * u_n + a2 * u_nm1) / dt


def build_auxiliary_fields(*, p, n_grains, xi, xi_ref, phil, phis, c, etas,
                           xi_n, xi_nm1, v_l, v_s, v_c, u_vec,
                           cell_region_tag, dS, dt, bdf_a0, bdf_a1, bdf_a2,
                           stage=None):
    """Build shared UFL material, electrochemical, and interface expressions."""
    if stage is not None:
        stage("before UFL auxiliary fields")
    xi_clip = clip01_ufl(xi)
    xi_ref_clip = clip01_ufl(xi_ref)
    Xi = ufl.variable(xi_clip)
    f_well = Xi**2 * (1.0 - Xi) ** 2
    df_dxi = ufl.diff(f_well, Xi)
    hxi = h(xi_clip)
    hxi0 = h(xi_ref_clip)
    hXi = h(Xi)
    eta_clips = [clip01_ufl(eta_i) for eta_i in etas]
    h_etas = [h(eta_clip_i) for eta_clip_i in eta_clips]
    B_expr = grain_boundary_indicator_expr(etas, p)
    gb_window = grain_boundary_window_expr(etas, p, eta_clips)
    sum_eta_sq = sum(eta_i * eta_i for eta_i in etas)
    gb_pair_sum = sum(
        etas[i] * etas[j]
        for i in range(n_grains) for j in range(n_grains) if j != i
    )
    ce_expr = ufl.conditional(ufl.gt(xi, 0.5), xi, gb_window)
    I2 = ufl.Identity(2)
    eps_u = strain(u_vec)
    heta = sum(h_etas)
    E1 = p.E_li * hxi + p.E_se * (1.0 - hxi)
    nu1 = p.nu_li * hxi + p.nu_se * (1.0 - hxi)
    eig1 = 0.0 * I2
    eig2 = 0.0 * I2
    eig3 = (p.cathode_swelling_scale * p.omega_cathode * p.c_li_max
            * (c - p.soc_ref) / 3.0) * I2
    E1_var = p.E_li * hXi + p.E_se * (1.0 - hXi)
    nu1_var = p.nu_li * hXi + p.nu_se * (1.0 - hXi)
    eig1_var = 0.0 * I2
    eps_eff1 = eps_u - eig1
    eps_eff1_var = eps_u - eig1_var
    sigma1 = plane_stress_tensor(E1, nu1, eps_u, eig1)
    sigma2 = plane_stress_tensor(p.E_mix, p.nu_mix, eps_u, eig2)
    sigma3 = plane_stress_tensor(p.E_cathode, p.nu_cathode, eps_u, eig3)
    fmech1 = mechanical_energy_2d(E1_var, nu1_var, eps_eff1_var)
    xi_clip_derivative = ufl.conditional(
        ufl.And(ufl.ge(xi, 0.0), ufl.le(xi, 1.0)), 1.0, 0.0
    )
    dfmech_dxi = ufl.diff(fmech1, Xi) * xi_clip_derivative
    dfmech_detas = [0.0 for _ in range(n_grains)]
    hydro1 = plane_stress_hydrostatic_stress(E1, nu1, eps_u, eig1)
    hydro2 = plane_stress_hydrostatic_stress(p.E_mix, p.nu_mix, eps_u, eig2)
    hydro3 = plane_stress_hydrostatic_stress(p.E_cathode, p.nu_cathode, eps_u, eig3)
    overp_mech_volts = hydro1 * p.omega_li / p.F
    overp_mech_c_volts = hydro3 * p.omega_cathode / p.F
    sigmaS = p.sigmal + p.sigmaS_stress_coeff * hydro1
    sigma_eff = p.sigmae * hxi + sigmaS * heta + p.sigma_phi * B_expr
    if stage is not None:
        stage("after UFL auxiliary fields")
        stage("before electrochemical interface expressions")
    theta_eps = 1.0e-4
    theta = ufl.max_value(theta_eps, ufl.min_value(1.0 - theta_eps, c))
    e_eq = eeq_from_soc(theta)
    overp_c_volts = phis - phil - e_eq - overp_mech_c_volts
    overp_c = overp_c_volts / p.Vt
    i0_c = p.i0_c_ref * (2.0 * theta) ** p.alpha * (2.0 * (1.0 - theta)) ** (1.0 - p.alpha)
    i_cathode = i0_c * (ufl.exp(-p.alpha * overp_c) - ufl.exp((1.0 - p.alpha) * overp_c))
    phil_omega2_s = trace_on_region(phil, cell_region_tag, OMEGA2)
    phis_omega2_s = trace_on_region(phis, cell_region_tag, OMEGA2)
    v_l_omega2_s = trace_on_region(v_l, cell_region_tag, OMEGA2)
    v_s_omega2_s = trace_on_region(v_s, cell_region_tag, OMEGA2)
    c_omega3_s = trace_on_region(c, cell_region_tag, OMEGA3)
    v_c_omega3_s = trace_on_region(v_c, cell_region_tag, OMEGA3)
    e_eq_omega3_s = trace_on_region(e_eq, cell_region_tag, OMEGA3)
    overp_mech_c_omega3_s = trace_on_region(overp_mech_c_volts, cell_region_tag, OMEGA3)
    theta_s = ufl.max_value(theta_eps, ufl.min_value(1.0 - theta_eps, c_omega3_s))
    overp_c_volts_s = phis_omega2_s - phil_omega2_s - e_eq_omega3_s - overp_mech_c_omega3_s
    overp_c_s = overp_c_volts_s / p.Vt
    i0_c_s = p.i0_c_ref * (2.0 * theta_s) ** p.alpha * (2.0 * (1.0 - theta_s)) ** (1.0 - p.alpha)
    i_cathode_s = i0_c_s * (ufl.exp(-p.alpha * overp_c_s) - ufl.exp((1.0 - p.alpha) * overp_c_s))
    overp_li_volts = phil - overp_mech_volts
    overp_li = overp_li_volts / p.Vt
    reaction_li_raw = ufl.exp((1.0 - p.alpha) * overp_li) - ufl.min_value(ce_expr, 1.0) * ufl.exp(-p.alpha * overp_li)
    deposition_drive = -reaction_li_raw
    xi_source_weight = hp(xi_clip)
    localized_reaction_li = xi_source_weight * deposition_drive
    xit_expr = bdf_time_derivative(xi, xi_n, xi_nm1, bdf_a0, bdf_a1, bdf_a2, dt)
    liion_source_expr = p.F * (1.0 / p.omega_li) * xit_expr
    cathode_phil_term, cathode_phis_term, cathode_c_term = cathode_interface_flux_terms(
        i_cathode_s, v_l_omega2_s, v_s_omega2_s, v_c_omega3_s, dS, dt, p
    )
    if stage is not None:
        stage("after electrochemical interface expressions")
    return {
        "xi_clip": xi_clip, "xi_ref_clip": xi_ref_clip, "Xi": Xi,
        "f_well": f_well, "df_dxi": df_dxi, "hxi": hxi,
        "hxi0": hxi0, "hXi": hXi, "eta_clips": eta_clips,
        "h_etas": h_etas, "B_expr": B_expr, "gb_window": gb_window,
        "sum_eta_sq": sum_eta_sq, "gb_pair_sum": gb_pair_sum,
        "ce_expr": ce_expr, "I2": I2, "eps_u": eps_u, "heta": heta,
        "E1": E1, "nu1": nu1, "eig1": eig1, "eig2": eig2,
        "eig3": eig3, "eig1_var": eig1_var,
        "eps_eff1": eps_eff1, "E1_var": E1_var, "nu1_var": nu1_var,
        "eps_eff1_var": eps_eff1_var, "sigma1": sigma1,
        "sigma2": sigma2, "sigma3": sigma3, "fmech1": fmech1,
        "xi_clip_derivative": xi_clip_derivative,
        "dfmech_dxi": dfmech_dxi, "dfmech_detas": dfmech_detas,
        "hydro1": hydro1, "hydro2": hydro2, "hydro3": hydro3,
        "overp_mech_volts": overp_mech_volts,
        "overp_mech_c_volts": overp_mech_c_volts,
        "Sh": hydro1, "sigmaS": sigmaS, "sigma_eff": sigma_eff,
        "theta_eps": theta_eps, "theta": theta, "e_eq": e_eq,
        "e_eq_eff": overp_mech_c_volts, "overp_c_volts": overp_c_volts,
        "overp_c": overp_c, "i0_c": i0_c, "i_cathode": i_cathode,
        "phil_omega2_s": phil_omega2_s,
        "phis_omega2_s": phis_omega2_s,
        "v_l_omega2_s": v_l_omega2_s,
        "v_s_omega2_s": v_s_omega2_s,
        "c_omega3_s": c_omega3_s, "v_c_omega3_s": v_c_omega3_s,
        "e_eq_omega3_s": e_eq_omega3_s,
        "overp_mech_c_omega3_s": overp_mech_c_omega3_s,
        "electrolyte_side": GAMMA_S_OMEGA2_SIDE,
        "cathode_side": GAMMA_S_OMEGA3_SIDE,
        "theta_s": theta_s, "overp_c_volts_s": overp_c_volts_s,
        "overp_c_s": overp_c_s, "i0_c_s": i0_c_s,
        "i_cathode_s": i_cathode_s,
        "overp_li_volts": overp_li_volts, "overp_li": overp_li,
        "reaction_li_raw": reaction_li_raw,
        "deposition_drive": deposition_drive,
        "xi_source_weight": xi_source_weight,
        "localized_reaction_li": localized_reaction_li,
        "xit_expr": xit_expr, "liion_source_expr": liion_source_expr,
        "cathode_phil_term": cathode_phil_term,
        "cathode_phis_term": cathode_phis_term,
        "cathode_c_term": cathode_c_term,
    }


def side_is_region(cell_region_tag, region_id, side):
    return ufl.conditional(
        ufl.lt(abs(cell_region_tag(side) - float(region_id)), 0.5),
        1.0,
        0.0,
    )


def build_context_auxiliary_fields(*, params, n_grains, state_parts,
                                   finite_element_state, context,
                                   cell_region_tag, dS, stage):
    """Build constitutive fields from the shared state context."""
    auxiliary = build_auxiliary_fields(
        p=params, n_grains=n_grains, xi=state_parts["xi"],
        xi_ref=finite_element_state.xi_ref, xi_n=state_parts["xi_n"],
        xi_nm1=state_parts["xi_nm1"], phil=state_parts["phil"], phis=state_parts["phis"],
        c=state_parts["c"], etas=state_parts["etas"], v_l=state_parts["v_l"],
        v_s=state_parts["v_s"], v_c=state_parts["v_c"], u_vec=state_parts["u_vec"],
        cell_region_tag=cell_region_tag, dS=dS, dt=context.constants.dt,
        bdf_a0=context.constants.bdf_a0, bdf_a1=context.constants.bdf_a1,
        bdf_a2=context.constants.bdf_a2, stage=stage)
    context.equations.forms["auxiliary"] = auxiliary
    return auxiliary


def trace_on_region(expr, cell_region_tag, region_id):
    return (
        side_is_region(cell_region_tag, region_id, "+") * expr("+")
        + side_is_region(cell_region_tag, region_id, "-") * expr("-")
    )


def build_jacobian(residual, electrochemical_residual, mechanics_residual,
                   grain_residual, unknown, direction, n_grains, mode):
    """Build the configured full or block-approximate Jacobian."""
    if mode == "full":
        return ufl.derivative(residual, unknown, direction)

    direction_components = ufl.split(direction)
    zero_direction = 0 * direction_components[0]

    def block_direction(active_components):
        active_components = set(active_components)
        return ufl.as_vector(
            [
                direction_components[i]
                if i in active_components
                else zero_direction
                for i in range(6 + n_grains)
            ]
        )

    electro_xi_block = (0, 1, 2, 3)
    mechanics_block = (4, 5)
    eta_block = tuple(range(6, 6 + n_grains))
    return (
        ufl.derivative(
            electrochemical_residual,
            unknown,
            block_direction(electro_xi_block),
        )
        + ufl.derivative(
            mechanics_residual,
            unknown,
            block_direction(mechanics_block),
        )
        + ufl.derivative(
            grain_residual,
            unknown,
            block_direction(eta_block),
        )
    )


def assemble_equation_system(*, blocks, unknown, direction, n_grains,
                             jacobian_mode, context=None, stage=None):
    """Combine weak-form blocks and construct the configured Jacobian."""
    F_xi, F_phil, F_phis, F_c, F_eta, F_mech = blocks
    total = F_xi + F_phil + F_phis + F_c + F_eta + F_mech
    if stage is not None:
        stage("before F_total/J")
    jacobian = build_jacobian(total, F_xi + F_phil + F_phis + F_c,
        F_mech, F_eta, unknown, direction, n_grains, jacobian_mode)
    if context is not None:
        context.equations.residual = total
        context.equations.jacobian = jacobian
        context.equations.forms.update(dict(zip(
            ("xi", "phil", "phis", "c", "eta", "mechanics"), blocks)))
    if stage is not None:
        stage("after F_total/J")
    return total, jacobian


def cathode_interface_flux_terms(i_s, v_l_s, v_s_s, v_c_s, dS, dt_expr, p):
    scale_c = 1.0 / (p.F * p.c_li_max * p.length_scale)
    return (
        i_s * v_l_s * dS(GAMMA_S),
        -i_s * v_s_s * dS(GAMMA_S),
        -scale_c * i_s * v_c_s * dS(GAMMA_S),
    )


def build_residual_blocks(
    *, p, n_grains, xi, xi_n, xi_nm1, phil, phis, c, c_n, c_nm1,
    etas, etas_n, etas_nm1, dfmech_detas, v_xi, v_l, v_s, v_c, v_etas, v_u,
    dx_omega1, dx_eta, dx, ds, dt, bdf_a0, bdf_a1, bdf_a2, df_dxi,
    dfmech_dxi, sum_eta_sq, localized_reaction_li, sigma_eff,
    liion_source_expr, cathode_phil_term, cathode_phis_term,
    cathode_c_term, i_app, hydro3, sigma1, sigma2, sigma3,
):
    """Construct the original weak residual blocks from shared UFL objects."""
    def time_derivative(value, previous, previous_previous):
        return bdf_time_derivative(
            value, previous, previous_previous, bdf_a0, bdf_a1, bdf_a2, dt
        )

    F_xi = (
        time_derivative(xi, xi_n, xi_nm1) / p.L_sigma * v_xi * dx_omega1
        + xi_anisotropic_gradient_term(xi, v_xi, p) * dx_omega1
        + p.W_b * df_dxi * v_xi * dx_omega1
        + dfmech_dxi * v_xi * dx_omega1
        + 2.0 * p.W_b * xi * sum_eta_sq * v_xi * dx_omega1
        - p.xi_mobility_scale
        * (p.L_eta / p.L_sigma)
        * localized_reaction_li
        * v_xi
        * dx_omega1
    )

    F_phil = (
        (sigma_eff / p.length_scale)
        * ufl.dot(ufl.grad(phil), ufl.grad(v_l))
        * dx_omega1
        + (p.sse / p.length_scale)
        * ufl.dot(ufl.grad(phil), ufl.grad(v_l))
        * dx(OMEGA2)
        + p.length_scale * liion_source_expr * v_l * dx_omega1
        + cathode_phil_term
    )

    F_phis = (
        (p.sigma_cathode / p.length_scale)
        * ufl.dot(ufl.grad(phis), ufl.grad(v_s))
        * dx(OMEGA2)
        + cathode_phis_term
        - i_app * v_s * ds(GAMMA_C)
    )

    F_c = (
        time_derivative(c, c_n, c_nm1) * v_c * dx(OMEGA3)
        + (p.D_li / p.length_scale**2)
        * ufl.dot(ufl.grad(c), ufl.grad(v_c))
        * dx(OMEGA3)
        - (p.D_li * p.omega_cathode / (p.R * p.T * p.length_scale**2))
        * c
        * ufl.dot(ufl.grad(hydro3), ufl.grad(v_c))
        * dx(OMEGA3)
        + cathode_c_term
    )

    eta_grad_coeff = p.eta_mobility_scale * p.L_gb * p.k_gb / (
        p.length_scale**2
    )
    eta_chem_coeff = p.eta_mobility_scale * p.L_gb * p.W_gb
    F_eta = 0
    for i in range(n_grains):
        eta_i = etas[i]
        v_eta_i = v_etas[i]
        dfmech_deta_i = dfmech_detas[i]
        cross = sum(etas[j] ** 2 for j in range(n_grains) if j != i)
        dF_deta_i = -eta_i + eta_i**3 + 2.0 * eta_i * cross
        F_eta += (
            time_derivative(eta_i, etas_n[i], etas_nm1[i])
            * v_eta_i
            * dx_eta
            + eta_grad_coeff
            * ufl.dot(ufl.grad(eta_i), ufl.grad(v_eta_i))
            * dx_eta
            - p.eta_mobility_scale
            * p.L_gb
            * p.kappa_gb_cross
            / (p.length_scale**2)
            * 2.0
            * sum(
                ufl.dot(ufl.grad(etas[j]), ufl.grad(v_eta_i))
                for j in range(n_grains)
                if j != i
            )
            * dx_eta
            + eta_chem_coeff * dF_deta_i * v_eta_i * dx_eta
            + p.eta_mobility_scale
            * p.L_gb
            * 2.0
            * p.W_gb
            * xi**2
            * eta_i
            * v_eta_i
            * dx_eta
            + p.eta_mobility_scale
            * p.L_gb
            * dfmech_deta_i
            * v_eta_i
            * dx_eta
        )

    anode_traction = ufl.as_vector((0.0, p.anode_compressive_load))
    F_mech = p.mechanics_residual_scale * (
        ufl.inner(sigma1, strain(v_u)) * dx_omega1
        + ufl.inner(sigma2, strain(v_u)) * dx(OMEGA2)
        + ufl.inner(sigma3, strain(v_u)) * dx(OMEGA3)
        - ufl.dot(anode_traction, v_u) * ds(GAMMA_A)
    )
    return F_xi, F_phil, F_phis, F_c, F_eta, F_mech


def assemble_context_equations(*, params, n_grains, state_parts, auxiliary,
                               context, dx_omega1, dx_eta, dx, ds, stage):
    """Connect shared state and constitutive terms to residual/Jacobian forms."""
    s = state_parts
    c = context.constants
    stage("before residual blocks")
    blocks = build_residual_blocks(
        p=params, n_grains=n_grains, xi=s["xi"], xi_n=s["xi_n"], xi_nm1=s["xi_nm1"],
        phil=s["phil"], phis=s["phis"], c=s["c"], c_n=s["c_n"], c_nm1=s["c_nm1"],
        etas=s["etas"], etas_n=s["etas_n"], etas_nm1=s["etas_nm1"],
        dfmech_detas=auxiliary["dfmech_detas"], v_xi=s["v_xi"], v_l=s["v_l"],
        v_s=s["v_s"], v_c=s["v_c"], v_etas=s["v_etas"], v_u=s["v_u"],
        dx_omega1=dx_omega1, dx_eta=dx_eta, dx=dx, ds=ds,
        dt=c.dt, bdf_a0=c.bdf_a0, bdf_a1=c.bdf_a1, bdf_a2=c.bdf_a2,
        df_dxi=auxiliary["df_dxi"], dfmech_dxi=auxiliary["dfmech_dxi"],
        sum_eta_sq=auxiliary["sum_eta_sq"], localized_reaction_li=auxiliary["localized_reaction_li"],
        sigma_eff=auxiliary["sigma_eff"], liion_source_expr=auxiliary["liion_source_expr"],
        cathode_phil_term=auxiliary["cathode_phil_term"], cathode_phis_term=auxiliary["cathode_phis_term"],
        cathode_c_term=auxiliary["cathode_c_term"], i_app=c.applied_current,
        hydro3=auxiliary["hydro3"], sigma1=auxiliary["sigma1"],
        sigma2=auxiliary["sigma2"], sigma3=auxiliary["sigma3"])
    stage("after residual blocks")
    total, jacobian = assemble_equation_system(
        blocks=blocks, unknown=s["state"], direction=s["dw"], n_grains=n_grains,
        jacobian_mode=params.jacobian_mode, context=context,
        stage=stage)
    return {"blocks": blocks, "residual": total, "jacobian": jacobian}


__all__ = [
    "bdf_time_derivative", "cathode_interface_flux_terms",
    "build_jacobian", "build_residual_blocks", "side_is_region", "trace_on_region",
    "assemble_equation_system",
    "assemble_context_equations",
]
