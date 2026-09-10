"""Diagnostics, field statistics, and CSV reporting."""

import math
from pathlib import Path
import csv
from dataclasses import dataclass
import numpy as np
from mpi4py import MPI
from dolfinx import fem


def field_stats(fun, dofs=None):
    comm = fun.function_space.mesh.comm
    owned_size = fun.function_space.dofmap.index_map.size_local
    if dofs is None: values = fun.x.array.real[:owned_size]
    else:
        dofs = np.asarray(dofs, dtype=np.int64); dofs = dofs[(dofs >= 0) & (dofs < owned_size)]
        values = fun.x.array.real[dofs]
    if values.size:
        local_min, local_max, local_sum, local_count = float(values.min()), float(values.max()), float(values.sum()), int(values.size)
    else: local_min, local_max, local_sum, local_count = math.inf, -math.inf, 0.0, 0
    count = int(comm.allreduce(local_count, op=MPI.SUM))
    if count == 0: return float("nan"), float("nan"), float("nan")
    return (float(comm.allreduce(local_min, op=MPI.MIN)),
            float(comm.allreduce(local_max, op=MPI.MAX)),
            float(comm.allreduce(local_sum, op=MPI.SUM)) / count)


def dendrite_tip_y(fun, dofs=None, threshold=0.5):
    V = fun.function_space; comm = V.mesh.comm; owned_size = V.dofmap.index_map.size_local
    if dofs is None: dofs = np.arange(owned_size, dtype=np.int64)
    else:
        dofs = np.asarray(dofs, dtype=np.int64); dofs = dofs[(dofs >= 0) & (dofs < owned_size)]
    if dofs.size:
        active = dofs[fun.x.array.real[dofs] >= float(threshold)]
        local_tip = float(np.min(V.tabulate_dof_coordinates()[:owned_size][active, 1])) if active.size else math.inf
    else: local_tip = math.inf
    tip = float(comm.allreduce(local_tip, op=MPI.MIN))
    return tip if math.isfinite(tip) else float("nan")


def append_csv(path, row, write_header=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: writer.writeheader()
        writer.writerow(row)


def region_y_extent(V, dofs):
    comm = V.mesh.comm; owned_size = V.dofmap.index_map.size_local
    dofs = np.asarray(dofs, dtype=np.int64); dofs = dofs[(dofs >= 0) & (dofs < owned_size)]
    if dofs.size:
        coords = V.tabulate_dof_coordinates()[:owned_size]
        local_min, local_max = float(np.min(coords[dofs, 1])), float(np.max(coords[dofs, 1]))
    else: local_min, local_max = math.inf, -math.inf
    low = float(comm.allreduce(local_min, op=MPI.MIN)); high = float(comm.allreduce(local_max, op=MPI.MAX))
    return (low if math.isfinite(low) else float("nan"), high if math.isfinite(high) else float("nan"))


def constant_scalar_value(constant):
    return float(np.asarray(constant.value).reshape(-1)[0])


def assemble_total(expr, comm=None):
    value = float(fem.assemble_scalar(fem.form(expr)))
    return float(comm.allreduce(value, op=MPI.SUM)) if comm is not None else value


def assemble_form_total(form, comm=None):
    value = float(fem.assemble_scalar(form))
    return float(comm.allreduce(value, op=MPI.SUM)) if comm is not None else value


def create_diagnostic_context(initial=None, **values):
    """Build the mutable diagnostic namespace used by write_diagnostics."""
    context = dict(initial or {})
    context.update(values)
    return context


def update_diagnostic_state(context, values=None, **updates):
    """Update mutable phase/cutoff fields used by diagnostic reporting."""
    context.update(values or {})
    context.update(updates)
    return context


@dataclass
class CutoffState:
    terminal_voltage: float
    soc_cutoff: float
    soc_cutoff_value: float
    voltage_hit: bool
    soc_hit: bool
    dendrite_tip_y: float
    lifecycle_hit: bool
    reached: bool
    reason: str


def evaluate_cutoff_state(*, phase, cutoff_voltage, params, comm,
                          phil_boundary_stats, phis_boundary_stats,
                          cathode_stats, xi_function, xi_dofs, target_y):
    """Evaluate voltage, SOC, and lifecycle termination after an accepted step."""
    phil_min = phil_boundary_stats[0]
    phis_max = phis_boundary_stats[1]
    c_min, c_max = cathode_stats[0], cathode_stats[1]
    voltage = phis_max - phil_min
    voltage_local = voltage >= cutoff_voltage if phase == "charge" else voltage <= cutoff_voltage
    soc_limit = params.soc_min if phase == "charge" else params.soc_max
    soc_value = c_min if phase == "charge" else c_max
    soc_local = c_min <= soc_limit if phase == "charge" else c_max >= soc_limit
    voltage_hit = bool(comm.allreduce(int(voltage_local), op=MPI.MAX))
    soc_hit = bool(comm.allreduce(int(soc_local), op=MPI.MAX))
    tip = dendrite_tip_y(xi_function, xi_dofs, params.lifecycle_xi_threshold)
    lifecycle_hit = bool(
        params.lifecycle_stop_enabled
        and math.isfinite(tip)
        and math.isfinite(target_y)
        and tip <= target_y + params.lifecycle_target_margin
    )
    reached = voltage_hit or soc_hit or lifecycle_hit
    reason = (
        "lifecycle" if lifecycle_hit else
        "voltage+soc" if voltage_hit and soc_hit else
        "voltage" if voltage_hit else
        "soc" if soc_hit else ""
    )
    return CutoffState(voltage, soc_limit, soc_value, voltage_hit, soc_hit,
                       tip, lifecycle_hit, reached, reason)


__all__ = ["field_stats", "dendrite_tip_y", "append_csv", "region_y_extent",
           "constant_scalar_value", "assemble_total", "assemble_form_total",
           "create_diagnostic_context", "update_diagnostic_state",
           "CutoffState", "evaluate_cutoff_state", "write_diagnostics"]

def write_diagnostics(context, step, time_s, phase, dt_used):
    xi_min, xi_max, xi_mean = context['field_stats'](context['scalar_outputs'][0], context['stats_dofs']["xi"])
    c_min, c_max, c_mean = context['field_stats'](context['scalar_outputs'][3], context['stats_dofs']["c"])
    phil_min, phil_max, phil_mean = context['field_stats'](context['scalar_outputs'][1], context['stats_dofs']["phil"])
    phis_min, phis_max, phis_mean = context['field_stats'](context['scalar_outputs'][2], context['stats_dofs']["phis"])
    phil_gamma_a_min, phil_gamma_a_max, phil_gamma_a_mean = context['field_stats'](
        context['scalar_outputs'][1], context['boundary_probe_dofs']["phil_gamma_a"]
    )
    phis_gamma_c_min, phis_gamma_c_max, phis_gamma_c_mean = context['field_stats'](
        context['scalar_outputs'][2], context['boundary_probe_dofs']["phis_gamma_c"]
    )
    boundary_voltage_mean = phis_gamma_c_mean - phil_gamma_a_mean
    boundary_voltage = phis_gamma_c_max - phil_gamma_a_min
    overp_li_min, overp_li_max, overp_li_mean = context['field_stats'](
        context['derived_outputs'][8], context['stats_dofs']["xi"]
    )
    li_metal_mol_m2 = context['assemble_form_total'](context['li_metal_form'], context['msh'].comm)
    cathode_li_mol_m2 = context['assemble_form_total'](context['cathode_li_form'], context['msh'].comm)
    li_metal_gain_mol_m2 = li_metal_mol_m2 - context['xi_initial_mol_m2']
    cathode_li_delta_mol_m2 = cathode_li_mol_m2 - context['c_initial_mol_m2']
    cathode_li_loss_mol_m2 = -cathode_li_delta_mol_m2
    li_balance_error_mol_m2 = li_metal_gain_mol_m2 + cathode_li_delta_mol_m2
    denom = context['max'](context['abs'](li_metal_gain_mol_m2), context['abs'](cathode_li_delta_mol_m2), 1.0e-30)
    loss_denom = context['max'](context['abs'](cathode_li_loss_mol_m2), 1.0e-30)
    # 即时守恒检测：这些量直接来自当前时间步方程中的通量/源项。
    # 网格坐标使用 x/L，所以线积分需要乘 L，面积积分需要乘 L^2，
    # 才能得到二维单位厚度下的物理量。
    gamma_s_current_A_m = context['assemble_form_total'](context['gamma_s_current_form'], context['msh'].comm)
    gamma_s_reaction_current_avg_A_m2 = gamma_s_current_A_m / context['gamma_s_length_scale']
    # COMSOL liion.bei1.er1.iloc uses the opposite sign convention from
    # the paper i_s used above for the cathode BV expression.
    gamma_s_iloc_equiv_A_m = -gamma_s_current_A_m
    gamma_s_iloc_equiv_avg_A_m2 = (
        gamma_s_iloc_equiv_A_m / context['gamma_s_length_scale']
    )
    c_boundary_source_avg_soc_s = (
        -gamma_s_reaction_current_avg_A_m2 / (context['p'].F * context['p'].c_li_max * context['p'].length_scale)
    )
    gamma_c_current_A_m = context['assemble_form_total'](context['gamma_c_current_form'], context['msh'].comm)
    omega1_xi_current_A_m = context['assemble_form_total'](context['omega1_xi_current_form'], context['msh'].comm)
    overp_c_gamma_s_avg = context['assemble_form_total'](context['overp_c_gamma_s_form'], context['msh'].comm) / context['gamma_s_length_scale']
    overp_c_mean = overp_c_gamma_s_avg
    external_charge_current_A_m = -gamma_c_current_A_m
    xi_plus_gamma_s_A_m = omega1_xi_current_A_m + gamma_s_current_A_m

    if context['p'].light_diagnostics:
        u_dofs = context['stats_dofs']["u"]
        ux_values = context['scalar_outputs'][4].x.array.real[u_dofs]
        uy_values = context['scalar_outputs'][5].x.array.real[u_dofs]
        if ux_values.size == 0:
            u_mag_max = context['float']("nan")
        else:
            u_mag_max = context['float'](context['np'].sqrt(context['np'].max(ux_values * ux_values + uy_values * uy_values)))
        row = {
            "step": step,
            "time_s": time_s,
            "dt_s": dt_used,
            "cycle": context['current_cycle_index'],
            "phase": phase,
            "phase_elapsed_s": context['current_phase_elapsed'],
            "phase_duration_s": context['current_phase_duration'],
            "current_density_A_m2": context['constant_scalar_value'](context['current_density']),
            "cutoff_voltage_V": context['current_cutoff_voltage'],
            "soc_cutoff": context['current_soc_cutoff'],
            "cutoff_reached": context['current_cutoff_reached'],
            "cutoff_reason": context['current_cutoff_reason'],
            "boundary_voltage_probe_V": boundary_voltage,
            "boundary_voltage_probe_delta_V": boundary_voltage - context['boundary_voltage_initial'],
            "boundary_voltage_mean_probe_V": boundary_voltage_mean,
            "phil_gamma_a_min_V": phil_gamma_a_min,
            "phil_gamma_a_max_V": phil_gamma_a_max,
            "phil_gamma_a_mean_V": phil_gamma_a_mean,
            "phis_gamma_c_min_V": phis_gamma_c_min,
            "phis_gamma_c_max_V": phis_gamma_c_max,
            "phis_gamma_c_mean_V": phis_gamma_c_mean,
            "overp_c_mean_V": overp_c_mean,
            "xi_min": xi_min,
            "xi_max": xi_max,
            "xi_mean": xi_mean,
            "dendrite_tip_y": context['current_dendrite_tip_y'],
            "lifecycle_target_y": context['lifecycle_target_y'],
            "lifecycle_target_margin": context['p'].lifecycle_target_margin,
            "cathode_top_y": context['cathode_y_max'],
            "lifecycle_stop_reached": context['current_lifecycle_stop_reached'],
            "c_min": c_min,
            "c_max": c_max,
            "c_mean": c_mean,
            "phil_mean": phil_mean,
            "phis_mean": phis_mean,
            "u_magnitude_max_m": u_mag_max * context['p'].length_scale,
            "li_metal_mol_m2": li_metal_mol_m2,
            "cathode_li_mol_m2": cathode_li_mol_m2,
            "li_metal_gain_mol_m2": li_metal_gain_mol_m2,
            "cathode_li_delta_mol_m2": cathode_li_delta_mol_m2,
            "cathode_li_loss_mol_m2": cathode_li_loss_mol_m2,
            "li_gain_to_cathode_loss_ratio": li_metal_gain_mol_m2 / loss_denom,
            "li_balance_error_mol_m2": li_balance_error_mol_m2,
            "li_balance_rel_error": li_balance_error_mol_m2 / denom,
            "gamma_s_current_A_m": gamma_s_current_A_m,
            "gamma_s_reaction_current_avg_A_m2": gamma_s_reaction_current_avg_A_m2,
            "gamma_s_iloc_equiv_A_m": gamma_s_iloc_equiv_A_m,
            "gamma_s_iloc_equiv_avg_A_m2": gamma_s_iloc_equiv_avg_A_m2,
            "gamma_c_current_A_m": gamma_c_current_A_m,
            "external_charge_current_A_m": external_charge_current_A_m,
            "omega1_xi_current_A_m": omega1_xi_current_A_m,
            "xi_plus_gamma_s_A_m": xi_plus_gamma_s_A_m,
        }
        if context['msh'].comm.rank == 0:
            context['append_csv'](context['p'].diagnostics_file, row, write_header=(step == 0))
        return row
    ux_min, ux_max, ux_mean = context['field_stats'](context['scalar_outputs'][4], context['stats_dofs']["u"])
    uy_min, uy_max, uy_mean = context['field_stats'](context['scalar_outputs'][5], context['stats_dofs']["u"])
    reaction_li_min, reaction_li_max, reaction_li_mean = context['field_stats'](
        context['derived_outputs'][4], context['stats_dofs']["xi"]
    )
    deposition_drive_min, deposition_drive_max, deposition_drive_mean = context['field_stats'](
        context['derived_outputs'][5], context['stats_dofs']["xi"]
    )
    xi_source_weight_min, xi_source_weight_max, xi_source_weight_mean = context['field_stats'](
        context['derived_outputs'][6], context['stats_dofs']["xi"]
    )
    hydro1_min, hydro1_max, hydro1_mean = context['field_stats'](
        context['derived_outputs'][11], context['stats_dofs']["xi"]
    )
    overp_mech_min, overp_mech_max, overp_mech_mean = context['field_stats'](
        context['derived_outputs'][12], context['stats_dofs']["xi"]
    )
    dfmechdxi_min, dfmechdxi_max, dfmechdxi_mean = context['field_stats'](
        context['derived_outputs'][15], context['stats_dofs']["xi"]
    )
    heta_min, heta_max, heta_mean = context['field_stats'](context['derived_outputs'][22], context['stats_dofs']["xi"])
    E1_min, E1_max, E1_mean = context['field_stats'](context['derived_outputs'][23], context['stats_dofs']["xi"])
    nu1_min, nu1_max, nu1_mean = context['field_stats'](context['derived_outputs'][24], context['stats_dofs']["xi"])
    u_mag_min, u_mag_max, u_mag_mean = context['field_stats'](context['derived_outputs'][13])
    eeq_min, eeq_max, eeq_mean = context['field_stats'](context['derived_outputs'][10], context['stats_dofs']["c"])
    eta_c_mean = phis_mean - phil_mean - eeq_mean
    gamma_s_avg = lambda expr: context['assemble_total'](
        expr * context['p'].length_scale * context['dS'](context['GAMMA_S']), context['msh'].comm
    ) / context['gamma_s_length_scale']
    gamma_s_rms = lambda expr: context['math'].sqrt(context['max'](gamma_s_avg(expr * expr), 0.0))
    gamma_s_l8 = lambda expr: context['max'](gamma_s_avg((expr * expr) ** 4), 0.0) ** 0.125
    c_gamma_s_trace = context['c_omega3_s']
    eeq_gamma_s_trace = context['e_eq_omega3_s']
    mech_c_gamma_s_trace = context['overp_mech_c_omega3_s']
    eta_c_gamma_s_trace = context['overp_c_volts_s']
    i_s_gamma_s_trace = context['i_cathode_s']
    phil_gamma_s_trace = context['phil_omega2_s']
    phis_gamma_s_trace = context['phis_omega2_s']
    phil_gamma_s_plus_avg = context['assemble_total'](
        context['phil']("+") * context['p'].length_scale * context['dS'](context['GAMMA_S']), context['msh'].comm
    ) / context['gamma_s_length_scale']
    phil_gamma_s_minus_avg = context['assemble_total'](
        context['phil']("-") * context['p'].length_scale * context['dS'](context['GAMMA_S']), context['msh'].comm
    ) / context['gamma_s_length_scale']
    phil_gamma_s_avgtrace_avg = context['assemble_total'](
        context['phil'](context['electrolyte_side']) * context['p'].length_scale * context['dS'](context['GAMMA_S']), context['msh'].comm
    ) / context['gamma_s_length_scale']
    phis_gamma_s_plus_avg = context['assemble_total'](
        context['phis']("+") * context['p'].length_scale * context['dS'](context['GAMMA_S']), context['msh'].comm
    ) / context['gamma_s_length_scale']
    phis_gamma_s_minus_avg = context['assemble_total'](
        context['phis']("-") * context['p'].length_scale * context['dS'](context['GAMMA_S']), context['msh'].comm
    ) / context['gamma_s_length_scale']
    phis_gamma_s_avgtrace_avg = context['assemble_total'](
        context['phis'](context['electrolyte_side']) * context['p'].length_scale * context['dS'](context['GAMMA_S']), context['msh'].comm
    ) / context['gamma_s_length_scale']
    c_gamma_s_plus_avg = context['assemble_total'](
        context['c']("+") * context['p'].length_scale * context['dS'](context['GAMMA_S']), context['msh'].comm
    ) / context['gamma_s_length_scale']
    c_gamma_s_minus_avg = context['assemble_total'](
        context['c']("-") * context['p'].length_scale * context['dS'](context['GAMMA_S']), context['msh'].comm
    ) / context['gamma_s_length_scale']
    c_gamma_s_avgtrace_avg = context['assemble_total'](
        context['c'](context['cathode_side']) * context['p'].length_scale * context['dS'](context['GAMMA_S']), context['msh'].comm
    ) / context['gamma_s_length_scale']
    eeq_gamma_s_plus_avg = context['assemble_total'](
        context['e_eq']("+") * context['p'].length_scale * context['dS'](context['GAMMA_S']), context['msh'].comm
    ) / context['gamma_s_length_scale']
    eeq_gamma_s_minus_avg = context['assemble_total'](
        context['e_eq']("-") * context['p'].length_scale * context['dS'](context['GAMMA_S']), context['msh'].comm
    ) / context['gamma_s_length_scale']
    mech_c_gamma_s_plus_avg = context['assemble_total'](
        context['overp_mech_c_volts']("+") * context['p'].length_scale * context['dS'](context['GAMMA_S']), context['msh'].comm
    ) / context['gamma_s_length_scale']
    mech_c_gamma_s_minus_avg = context['assemble_total'](
        context['overp_mech_c_volts']("-") * context['p'].length_scale * context['dS'](context['GAMMA_S']), context['msh'].comm
    ) / context['gamma_s_length_scale']
    c_gamma_s_rms = gamma_s_rms(c_gamma_s_trace)
    c_gamma_s_l8 = gamma_s_l8(c_gamma_s_trace)
    c_gamma_s_dev_rms = gamma_s_rms(c_gamma_s_trace - c_gamma_s_avgtrace_avg)
    eeq_gamma_s_rms_V = gamma_s_rms(eeq_gamma_s_trace)
    eeq_gamma_s_l8_V = gamma_s_l8(eeq_gamma_s_trace)
    eeq_gamma_s_dev_rms_V = gamma_s_rms(eeq_gamma_s_trace - eeq_gamma_s_minus_avg)
    mech_c_gamma_s_rms_V = gamma_s_rms(mech_c_gamma_s_trace)
    mech_c_gamma_s_l8_V = gamma_s_l8(mech_c_gamma_s_trace)
    mech_c_gamma_s_dev_rms_V = gamma_s_rms(
        mech_c_gamma_s_trace - mech_c_gamma_s_minus_avg
    )
    eta_c_gamma_s_rms_V = gamma_s_rms(eta_c_gamma_s_trace)
    eta_c_gamma_s_l8_V = gamma_s_l8(eta_c_gamma_s_trace)
    eta_c_gamma_s_dev_rms_V = gamma_s_rms(
        eta_c_gamma_s_trace - overp_c_gamma_s_avg
    )
    i_s_gamma_s_rms_A_m2 = gamma_s_rms(i_s_gamma_s_trace)
    i_s_gamma_s_l8_A_m2 = gamma_s_l8(i_s_gamma_s_trace)
    i_s_gamma_s_dev_rms_A_m2 = gamma_s_rms(
        i_s_gamma_s_trace - gamma_s_reaction_current_avg_A_m2
    )
    phil_gamma_s_dev_rms_V = gamma_s_rms(
        phil_gamma_s_trace - phil_gamma_s_avgtrace_avg
    )
    phis_gamma_s_dev_rms_V = gamma_s_rms(
        phis_gamma_s_trace - phis_gamma_s_avgtrace_avg
    )
    row = {
        "step": step,
        "time_s": time_s,
        "dt_s": dt_used,
        "cycle": context['current_cycle_index'],
        "phase": phase,
        "phase_elapsed_s": context['current_phase_elapsed'],
        "phase_duration_s": context['current_phase_duration'],
        "cutoff_voltage_V": context['current_cutoff_voltage'],
        "soc_cutoff": context['current_soc_cutoff'],
        "cutoff_reached": context['current_cutoff_reached'],
        "voltage_cutoff_reached": context['current_voltage_cutoff_reached'],
        "soc_cutoff_reached": context['current_soc_cutoff_reached'],
        "cutoff_reason": context['current_cutoff_reason'],
        "current_density_A_m2": context['constant_scalar_value'](context['current_density']),
        "boundary_voltage_probe_V": boundary_voltage,
        "boundary_voltage_probe_delta_V": boundary_voltage - context['boundary_voltage_initial'],
        "boundary_voltage_mean_probe_V": boundary_voltage_mean,
        "phil_gamma_a_min_V": phil_gamma_a_min,
        "phil_gamma_a_max_V": phil_gamma_a_max,
        "phil_gamma_a_mean_V": phil_gamma_a_mean,
        "phis_gamma_c_min_V": phis_gamma_c_min,
        "phis_gamma_c_max_V": phis_gamma_c_max,
        "phis_gamma_c_mean_V": phis_gamma_c_mean,
        "eeq_min_V": eeq_min,
        "eeq_max_V": eeq_max,
        "eeq_mean_V": eeq_mean,
        "eta_c_mean_no_mech_V": eta_c_mean,
        "hydro1_min_Pa": hydro1_min,
        "hydro1_max_Pa": hydro1_max,
        "hydro1_mean_Pa": hydro1_mean,
        "overp_mech_min_V": overp_mech_min,
        "overp_mech_max_V": overp_mech_max,
        "overp_mech_mean_V": overp_mech_mean,
        "dfmechdxi_min_J_m3": dfmechdxi_min,
        "dfmechdxi_max_J_m3": dfmechdxi_max,
        "dfmechdxi_mean_J_m3": dfmechdxi_mean,
        "heta_min": heta_min,
        "heta_max": heta_max,
        "heta_mean": heta_mean,
        "E1_min_Pa": E1_min,
        "E1_max_Pa": E1_max,
        "E1_mean_Pa": E1_mean,
        "nu1_min": nu1_min,
        "nu1_max": nu1_max,
        "nu1_mean": nu1_mean,
        "u_magnitude_min": u_mag_min,
        "u_magnitude_max": u_mag_max,
        "u_magnitude_mean": u_mag_mean,
        "u_magnitude_max_m": u_mag_max * context['p'].length_scale,
        "ux_min": ux_min,
        "ux_max": ux_max,
        "ux_mean": ux_mean,
        "uy_min": uy_min,
        "uy_max": uy_max,
        "uy_mean": uy_mean,
        "reaction_li_min": reaction_li_min,
        "reaction_li_max": reaction_li_max,
        "reaction_li_mean": reaction_li_mean,
        "deposition_drive_min": deposition_drive_min,
        "deposition_drive_max": deposition_drive_max,
        "deposition_drive_mean": deposition_drive_mean,
        "xi_source_weight_min": xi_source_weight_min,
        "xi_source_weight_max": xi_source_weight_max,
        "xi_source_weight_mean": xi_source_weight_mean,
        "overp_c_mean_V": overp_c_mean,
        "overp_c_gamma_s_avg_V": overp_c_gamma_s_avg,
        "phil_gamma_s_plus_avg_V": phil_gamma_s_plus_avg,
        "phil_gamma_s_minus_avg_V": phil_gamma_s_minus_avg,
        "phil_gamma_s_avgtrace_avg_V": phil_gamma_s_avgtrace_avg,
        "phis_gamma_s_plus_avg_V": phis_gamma_s_plus_avg,
        "phis_gamma_s_minus_avg_V": phis_gamma_s_minus_avg,
        "phis_gamma_s_avgtrace_avg_V": phis_gamma_s_avgtrace_avg,
        "c_gamma_s_plus_avg": c_gamma_s_plus_avg,
        "c_gamma_s_minus_avg": c_gamma_s_minus_avg,
        "c_gamma_s_avgtrace_avg": c_gamma_s_avgtrace_avg,
        "c_gamma_s_rms": c_gamma_s_rms,
        "c_gamma_s_l8": c_gamma_s_l8,
        "c_gamma_s_dev_rms": c_gamma_s_dev_rms,
        "eeq_gamma_s_plus_avg_V": eeq_gamma_s_plus_avg,
        "eeq_gamma_s_minus_avg_V": eeq_gamma_s_minus_avg,
        "eeq_gamma_s_rms_V": eeq_gamma_s_rms_V,
        "eeq_gamma_s_l8_V": eeq_gamma_s_l8_V,
        "eeq_gamma_s_dev_rms_V": eeq_gamma_s_dev_rms_V,
        "mech_c_gamma_s_plus_avg_V": mech_c_gamma_s_plus_avg,
        "mech_c_gamma_s_minus_avg_V": mech_c_gamma_s_minus_avg,
        "mech_c_gamma_s_rms_V": mech_c_gamma_s_rms_V,
        "mech_c_gamma_s_l8_V": mech_c_gamma_s_l8_V,
        "mech_c_gamma_s_dev_rms_V": mech_c_gamma_s_dev_rms_V,
        "eta_c_gamma_s_rms_V": eta_c_gamma_s_rms_V,
        "eta_c_gamma_s_l8_V": eta_c_gamma_s_l8_V,
        "eta_c_gamma_s_dev_rms_V": eta_c_gamma_s_dev_rms_V,
        "i_s_gamma_s_rms_A_m2": i_s_gamma_s_rms_A_m2,
        "i_s_gamma_s_l8_A_m2": i_s_gamma_s_l8_A_m2,
        "i_s_gamma_s_dev_rms_A_m2": i_s_gamma_s_dev_rms_A_m2,
        "phil_gamma_s_dev_rms_V": phil_gamma_s_dev_rms_V,
        "phis_gamma_s_dev_rms_V": phis_gamma_s_dev_rms_V,
        "xi_min": xi_min,
        "xi_max": xi_max,
        "xi_mean": xi_mean,
        "c_min": c_min,
        "c_max": c_max,
        "c_mean": c_mean,
        "phil_min": phil_min,
        "phil_max": phil_max,
        "phil_mean": phil_mean,
        "phil_mean_delta_V": phil_mean - context['phil_initial_mean'],
        "phis_min": phis_min,
        "phis_max": phis_max,
        "phis_mean": phis_mean,
        "phis_mean_delta_V": phis_mean - context['phis_initial_mean'],
        "li_metal_mol_m2": li_metal_mol_m2,
        "cathode_li_mol_m2": cathode_li_mol_m2,
        "li_metal_gain_mol_m2": li_metal_gain_mol_m2,
        "cathode_li_delta_mol_m2": cathode_li_delta_mol_m2,
        "cathode_li_loss_mol_m2": cathode_li_loss_mol_m2,
        "li_gain_to_cathode_loss_ratio": li_metal_gain_mol_m2 / loss_denom,
        "li_balance_error_mol_m2": li_balance_error_mol_m2,
        "li_balance_rel_error": li_balance_error_mol_m2 / denom,
        "gamma_s_current_A_m": gamma_s_current_A_m,
        "gamma_s_reaction_current_avg_A_m2": gamma_s_reaction_current_avg_A_m2,
        "gamma_s_iloc_equiv_A_m": gamma_s_iloc_equiv_A_m,
        "gamma_s_iloc_equiv_avg_A_m2": gamma_s_iloc_equiv_avg_A_m2,
        "c_boundary_source_avg_soc_s": c_boundary_source_avg_soc_s,
        "external_charge_current_A_m": external_charge_current_A_m,
        "omega1_xi_current_A_m": omega1_xi_current_A_m,
    }
    if context['msh'].comm.rank == 0:
        context['append_csv'](context['p'].diagnostics_file, row, write_header=(step == 0))
    console_row = context['dict'](row)
    console_row.update(
        {
            "gamma_c_current_A_m": gamma_c_current_A_m,
            "xi_plus_gamma_s_A_m": xi_plus_gamma_s_A_m,
        }
    )
    return console_row


def write_diagnostic_row(context_factory, step, time_s, phase, dt_used):
    """Evaluate a mutable context and write one diagnostics row."""
    return write_diagnostics(context_factory(), step, time_s, phase, dt_used)
