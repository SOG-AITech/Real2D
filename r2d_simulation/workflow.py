"""Top-level orchestration for the R2D simulation.

The workflow owns the public execution boundary.  Numerical stages still
receive the original keyword arguments unchanged, so this layer does not
alter equations, solver settings, or output behavior.
"""

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
import builtins
import math
import os
from typing import Any

import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc

from .config import (
    DEFAULT_GB_FILE, DEFAULT_MSH_FILE, GAMMA_A, GAMMA_C,
    GAMMA_S, GAMMA_S_OMEGA2_SIDE, GAMMA_S_OMEGA3_SIDE, OMEGA1,
    OMEGA1_REGIONS, OMEGA2, OMEGA3,
    configure_run_params,
)
from .constitutive import xi_anisotropic_gradient_term
from .mesh import dx_regions
from .initialization import initialize_state
from .regions import collapse_mixed_subspace
from .fields import (
    clip_mixed_components,
    update_scalar_outputs,
)
from .constraints import build_constraint_state
from .diagnostics import (
    append_csv, assemble_total, assemble_form_total, constant_scalar_value, dendrite_tip_y,
    field_stats, region_y_extent, write_diagnostics, create_diagnostic_context,
    update_diagnostic_state,
    evaluate_cutoff_state,
)
from .output import (
    OutputSchedule, field_output_variable_set,
    make_latest_state_saver, save_delta_outputs,
    save_field_snapshot, save_final_npz, save_png_outputs,
    save_sampled_outputs,
    derived_output_fields,
    initialize_output_state,
    refresh_output_fields,
    save_scheduled_output,
    make_phase_output_callback, finalize_outputs,
)
from .solver import (
    assign_constant, nonlinear_solver_status,
    mpi_stage_print,
    build_solver_state,
)
from .equations import (
    build_auxiliary_fields, build_context_auxiliary_fields,
    assemble_context_equations,
)
from .derived import update_derived_fields, update_diagnostic_derived_fields
from .evolution import (
    estimate_state_time_error,
    next_dt_after_accept,
    reject_time_step, time_error_history_ready, time_step_decision,
    set_bdf_coefficients as modular_set_bdf_coefficients,
    solve_with_retries,
)
from .evolution_loop import (
    EvolutionRuntimeContext, advance_history,
    solve_trial_step,
    run_evolution_loop,
)
from .evolution_callbacks import build_attempt_step
from .cycling import build_phase_schedule, phase_dt_max
from .state import R2DContext
from .setup import build_mesh_and_fields
from .state_builder import split_state
from .simulation_setup import attach_mesh_field_setup


@dataclass
class SimulationRequest:
    """Original call arguments carried unchanged across the workflow boundary."""

    args: tuple[Any, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)

    def call(self, function):
        return function(*self.args, **self.options)


def prepare_request(*args, **kwargs) -> SimulationRequest:
    """Retain the established main() calling convention exactly."""
    return SimulationRequest(tuple(args), dict(kwargs))


_ORIGINAL_PRINT = builtins.print
_PROGRESS_ONLY_ENABLED = False
_QUIET_OUTPUT_ENABLED = False
_SAVED_STDOUT_FD = None
_SAVED_STDERR_FD = None
_DEVNULL_FD = None


def set_progress_only_terminal(enabled: bool, quiet: bool = False):
    global _PROGRESS_ONLY_ENABLED, _QUIET_OUTPUT_ENABLED
    global _SAVED_STDOUT_FD, _SAVED_STDERR_FD, _DEVNULL_FD
    _QUIET_OUTPUT_ENABLED = bool(quiet)
    enabled = bool(enabled)
    if enabled and not _PROGRESS_ONLY_ENABLED:
        _SAVED_STDOUT_FD = os.dup(1)
        _DEVNULL_FD = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_DEVNULL_FD, 1)
        builtins.print = lambda *args, **kwargs: None
        _PROGRESS_ONLY_ENABLED = True
    elif not enabled and _PROGRESS_ONLY_ENABLED:
        os.dup2(_SAVED_STDOUT_FD, 1)
        if _SAVED_STDERR_FD is not None:
            os.dup2(_SAVED_STDERR_FD, 2)
        os.close(_SAVED_STDOUT_FD)
        if _SAVED_STDERR_FD is not None:
            os.close(_SAVED_STDERR_FD)
        os.close(_DEVNULL_FD)
        _SAVED_STDOUT_FD = None
        _SAVED_STDERR_FD = None
        _DEVNULL_FD = None
        builtins.print = _ORIGINAL_PRINT
        _PROGRESS_ONLY_ENABLED = False


def progress_print(*args, **kwargs):
    if _QUIET_OUTPUT_ENABLED:
        return
    if _PROGRESS_ONLY_ENABLED and _SAVED_STDOUT_FD is not None:
        text = kwargs.get("sep", " ").join(str(arg) for arg in args) + kwargs.get("end", "\n")
        os.write(_SAVED_STDOUT_FD, text.encode(errors="replace"))
    else:
        _ORIGINAL_PRINT(*args, **kwargs)


def run_simulation(
    msh_file: str = DEFAULT_MSH_FILE,
    gb_file: str = DEFAULT_GB_FILE,
    charge_time: float | None = None,
    discharge_time: float | None = None,
    cycle_count: int | None = None,
    charge_cutoff_voltage: float | None = None,
    discharge_cutoff_voltage: float | None = None,
    dt: float | None = None,
    cathode_reaction_mode: str = "interface",
    li_side_potential: float | None = None,
    dt_min: float | None = None,
    soc_init: float | None = None,
    soc_min: float | None = None,
    soc_max: float | None = None,
    D_li: float | None = None,
    i0_ref_li: float | None = None,
    i0_c_ref: float | None = None,
    c_li_max: float | None = None,
    temperature: float | None = None,
    initial_cathode_overpotential: float | None = None,
    phis_init: float | None = None,
    charge_current_sign: float | None = None,
    discharge_current_sign: float | None = None,
    current_abs: float | None = None,
    E_li: float | None = None,
    nu_li: float | None = None,
    E_se: float | None = None,
    nu_se: float | None = None,
    E_cathode: float | None = None,
    nu_cathode: float | None = None,
    anode_compressive_load: float | None = None,
    cathode_swelling_scale: float | None = None,
    gamma: float | None = None,
    k_xi: float | None = None,
    W_xi: float | None = None,
    L_gb: float | None = None,
    W_gb: float | None = None,
    k_gb: float | None = None,
    li_layer_thickness: float | None = None,
    enforce_top_xi_bc: bool | None = None,
    mechanics_periodic_lr: bool | None = None,
    dt_max: float | None = None,
    dt_max_charge: float | None = None,
    dt_max_discharge: float | None = None,
    dt_growth: float | None = None,
    dt_shrink: float | None = None,
    fixed_initial_steps: int | None = None,
    fixed_initial_dt: float | None = None,
    time_adapt_tol_max: float | None = None,
    time_adapt_tol_min: float | None = None,
    time_adapt_safety: float | None = None,
    time_adapt_rho_abs: float | None = None,
    time_adapt_rho_rel: float | None = None,
    time_adapt_factor_min: float | None = None,
    time_adapt_factor_max: float | None = None,
    time_adapt_reject: bool | None = None,
    time_adapt_reject_factor: float | None = None,
    time_adapt_target_fraction: float | None = None,
    retry_recovery_steps: int | None = None,
    retry_recovery_growth: float | None = None,
    snes_rtol: float | None = None,
    snes_atol: float | None = None,
    snes_stol: float | None = None,
    snes_max_it: int | None = None,
    snes_monitor: bool | None = None,
    png_interval: float | None = None,
    xdmf_interval: float | None = None,
    field_output_interval_s: float | None = None,
    field_output_variables: tuple[str, ...] | str | None = None,
    diagnostics_interval: int | None = None,
    progress_only: bool | None = None,
    quiet: bool | None = None,
    clip_eta_after_solve: bool | None = None,
    mechanics_residual_scale: float | None = None,
    newton_profile: bool | None = None,
    newton_profile_file: str | None = None,
    linear_solver: str | None = None,
    linear_rtol: float | None = None,
    linear_atol: float | None = None,
    linear_max_it: int | None = None,
    jacobian_lag: int | None = None,
    jacobian_mode: str | None = None,
    reuse_jacobian_across_steps: bool | None = None,
    fast_residual_bc: bool | None = None,
    light_diagnostics: bool | None = None,
    extrapolate_initial_guess: bool | None = None,
    extrapolate_max_factor: float | None = None,
    lifecycle_stop_enabled: bool | None = None,
    lifecycle_xi_threshold: float | None = None,
    lifecycle_target_margin: float | None = None,
):
    p = configure_run_params(cathode_reaction_mode, locals(), field_output_variable_set)
    set_progress_only_terminal(p.progress_only or p.quiet, quiet=p.quiet)


    out_dir = Path(p.preview_dir)
    if MPI.COMM_WORLD.rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = Path(p.diagnostics_file)
        if diagnostics.exists():
            diagnostics.unlink()

    setup = build_mesh_and_fields(msh_file=msh_file, gb_file=gb_file,
        params=p, mpi_stage_print=mpi_stage_print)
    attach_mesh_field_setup(setup["context"], setup)
    context = setup["context"]
    msh = setup["msh"]
    cell_tags, facet_tags = setup["cell_tags"], setup["facet_tags"]
    dx, ds, dS, dx_omega1, dx_eta = setup["dx"], setup["ds"], setup["dS"], setup["dx_omega1"], setup["dx_eta"]
    cell_region_tag = setup["cell_region_tag"]
    finite_element_state = context.fields.finite_element
    V_scalar, domain_markers = setup["V_scalar"], setup["domain_markers"]
    stats_dofs, boundary_probe_dofs = setup["stats_dofs"], setup["boundary_probe_dofs"]
    eta_regions, n_grains = setup["eta_regions"], setup["n_grains"]
    eta_initial_values = setup["eta_initial_values"]
    ME = setup["ME"]
    w, w_n, w_nm1, w_nm2 = setup["w"], setup["w_n"], setup["w_nm1"], setup["w_nm2"]
    cathode_y_min, cathode_y_max = finite_element_state.cathode_y_min, finite_element_state.cathode_y_max
    omega1_y_min, omega1_y_max = finite_element_state.omega1_y_min, finite_element_state.omega1_y_max
    lifecycle_target_y = finite_element_state.lifecycle_target_y    # Boundary probe values are diagnostics only, matching COMSOL Boundary Probe style.
    # They do not impose Dirichlet conditions on phil or phis.

    mpi_stage_print(msh.comm, "before UFL split/test setup")
    state_parts = split_state(finite_element_state)
    (components, components_n, components_nm1, tests, xi, phil, phis, c, ux, uy,
     xi_n, phil_n, phis_n, c_n, ux_n, uy_n, xi_nm1, phil_nm1, phis_nm1,
     c_nm1, ux_nm1, uy_nm1, etas, etas_n, etas_nm1, v_xi, v_l, v_s, v_c,
     v_ux, v_uy, v_etas, u_vec, v_u, dw) = (
        state_parts[key] for key in (
            "components", "components_n", "components_nm1", "tests", "xi", "phil", "phis", "c", "ux", "uy",
            "xi_n", "phil_n", "phis_n", "c_n", "ux_n", "uy_n", "xi_nm1", "phil_nm1", "phis_nm1", "c_nm1", "ux_nm1", "uy_nm1",
            "etas", "etas_n", "etas_nm1", "v_xi", "v_l", "v_s", "v_c", "v_ux", "v_uy", "v_etas", "u_vec", "v_u", "dw"))
    context.fields.fields.update(
        {
            "components": components,
            "previous_components": components_n,
            "previous_previous_components": components_nm1,
            "tests": tests,
            "trial": dw,
            "u_vector": u_vec,
            "displacement_test": v_u,
        }
    )
    mpi_stage_print(msh.comm, "after UFL split/test setup")

    # ------------------------------------------------------------------
    #
    #
    # ------------------------------------------------------------------
    initialization = initialize_state(msh, finite_element_state, p, stage=mpi_stage_print)
    y_top = initialization["y_top"]
    e_eq_init = initialization["e_eq_init"]
    phis_init_value = initialization["phis_init_value"]
    w_prev_array = initialization["previous_state_array"]
    xi_ref = initialization["xi_ref"]
    context.fields.fields["xi_ref"] = xi_ref

    context.constants.dt = fem.Constant(msh, PETSc.ScalarType(p.dt))
    context.constants.bdf_a0 = fem.Constant(msh, PETSc.ScalarType(1.0))
    context.constants.bdf_a1 = fem.Constant(msh, PETSc.ScalarType(-1.0))
    context.constants.bdf_a2 = fem.Constant(msh, PETSc.ScalarType(0.0))
    context.constants.current_density = fem.Constant(
        msh, PETSc.ScalarType(-p.current_abs)
    )
    context.constants.applied_current = fem.Constant(
        msh, PETSc.ScalarType(-p.current_abs)
    )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    auxiliary = build_context_auxiliary_fields(
        params=p, n_grains=n_grains, state_parts=state_parts,
        finite_element_state=finite_element_state, context=context,
        cell_region_tag=cell_region_tag, dS=dS,
        stage=lambda message: mpi_stage_print(msh.comm, message))
    equation_state = assemble_context_equations(
        params=p, n_grains=n_grains, state_parts=state_parts, auxiliary=auxiliary,
        context=context, dx_omega1=dx_omega1, dx_eta=dx_eta, dx=dx, ds=ds,
        stage=lambda message: mpi_stage_print(msh.comm, message))
    F_total, J = equation_state["residual"], equation_state["jacobian"]
    e_eq = auxiliary["e_eq"]
    e_eq_omega3_s = auxiliary["e_eq_omega3_s"]
    electrolyte_side = auxiliary["electrolyte_side"]
    i_cathode_s = auxiliary["i_cathode_s"]
    overp_c_volts_s = auxiliary["overp_c_volts_s"]
    overp_mech_c_omega3_s = auxiliary["overp_mech_c_omega3_s"]
    overp_mech_c_volts = auxiliary["overp_mech_c_volts"]
    phil_omega2_s = auxiliary["phil_omega2_s"]
    phis_omega2_s = auxiliary["phis_omega2_s"]
    c_omega3_s = auxiliary["c_omega3_s"]
    cathode_side = auxiliary["cathode_side"]
    constraints = build_constraint_state(
        msh=msh, ME=ME, V_scalar=V_scalar, domain_markers=domain_markers,
        facet_tags=facet_tags, eta_regions=eta_regions, n_grains=n_grains,
        params=p, y_top=y_top, w=w, w_n=w_n, w_nm1=w_nm1, w_nm2=w_nm2,
        stage=mpi_stage_print,
    )
    context.constraints = constraints
    bcs = constraints.bcs
    inactive_component_dofs = constraints.inactive_component_dofs
    periodic_constraints = constraints.periodic_constraints
    periodic_constraints_global = constraints.periodic_constraints_global
    periodic_constraint_counts = constraints.periodic_constraint_counts
    active_dofs = constraints.active_dofs
    eliminated_dofs = constraints.eliminated_dofs
    periodic_slave_dofs = constraints.periodic_slave_dofs
    x_min, x_max, y_min = constraints.x_min, constraints.x_max, constraints.y_min
    # No Dirichlet/gauge BC is imposed for phil.
    # In text.m, init1.phil = -0.1 is an initial value only. The physical outer
    # boundaries of the phil equation are natural Neumann boundaries; Gamma_s
    # reaction flux is already included through cathode_interface_flux_terms().
    # In the fully coupled system, the common potential level is closed by the
    # xi equation, Jcell=(F/Omega_Li)*(xi^{n+1}-xi^n)/dt, and the phis current BC.


    if p.mechanics_periodic_lr:
        if msh.comm.rank == 0:
            total_periodic_dofs = (
                len(periodic_constraints_global)
                if msh.comm.size > 1
                else len(periodic_constraints)
            )
            print(
                "COMSOL-style left-right periodic BCs enabled for xi, eta_i, "
                "and mechanics; "
                + ", ".join(
                    f"{name} slave dofs={count}"
                    for name, count in periodic_constraint_counts.items()
                )
                + f", total slave dofs={total_periodic_dofs}."
            )
    # Numerical gauge for the horizontal rigid-body mode. With left-right
    # periodic displacement constraints, ux -> ux + constant is still free in
    # the discrete elastic block; fixing one bottom midpoint dof only defines
    # the reference displacement.
    if msh.comm.rank == 0:
        print(
            "mechanical BCs: traction (0,-20 MPa) on Gamma_a, "
            "uy=0 on Gamma_c, ux=0 gauge at bottom midpoint; "
            + (
                "Gamma_b left/right are periodic."
                if p.mechanics_periodic_lr
                else "Gamma_b lateral sides are natural traction-free."
            )
        )

    # Stage-1 optimization:
    # keep the exact same monolithic residual/Jacobian, but remove fixed
    # inactive dofs from the Newton linear solve.
    owned_local_size = int(w.x.petsc_vec.getLocalSize())
    all_local_dofs = np.arange(owned_local_size, dtype=np.int32)
    eliminated_owned_dofs = eliminated_dofs[
        (0 <= eliminated_dofs) & (eliminated_dofs < owned_local_size)
    ]
    active_dofs_global_count = int(msh.comm.allreduce(active_dofs.size, op=MPI.SUM))
    eliminated_dofs_global_count = int(
        msh.comm.allreduce(eliminated_owned_dofs.size, op=MPI.SUM)
    )
    full_dofs_global_count = int(msh.comm.allreduce(all_local_dofs.size, op=MPI.SUM))
    if msh.comm.rank == 0:
        print(
            "stage1 restricted Newton: "
            f"active owned dofs={active_dofs_global_count}, "
            f"eliminated owned xi/eta inactive dofs={eliminated_dofs_global_count}, "
            f"periodic displacement slave dofs={periodic_slave_dofs.size}, "
            f"full owned dofs={full_dofs_global_count}, "
            f"jacobian_mode={p.jacobian_mode}"
            ,
            flush=True,
        )
    problem, newton_profiler = build_solver_state(
        residual=F_total, jacobian=J, state=w, constraints=constraints, params=p,
    )
    context.solver.problem = problem
    context.solver.profiler = newton_profiler
    mpi_stage_print(msh.comm, "before output function collapse")
    B = fem.Function(V_scalar, name="B")
    output_state = initialize_output_state(
        msh=msh, ME=ME, V_scalar=V_scalar, n_grains=n_grains,
    )
    scalar_outputs = output_state["scalar_outputs"]
    scalar_component_maps = output_state["component_maps"]
    mpi_stage_print(msh.comm, "after output function collapse")
    primary_scalar_indices = tuple(range(6))
    mpi_stage_print(msh.comm, "before initial scalar output update")
    update_scalar_outputs(w, scalar_outputs, p, component_maps=scalar_component_maps)
    mpi_stage_print(msh.comm, "after initial scalar output update")
    initial_outputs = []
    for fun in scalar_outputs:
        copy_fun = fem.Function(fun.function_space, name=f"initial_{fun.name}")
        copy_fun.x.array[:] = fun.x.array.real
        copy_fun.x.scatter_forward()
        initial_outputs.append(copy_fun)

    derived_outputs = output_state["derived_outputs"]
    V_dg0 = fem.functionspace(msh, ("DG", 0))
    hydro_omega1_dg0 = fem.Function(V_dg0, name="hydrostatic_stress_omega1_DG0")
    hydro_cathode_dg0 = fem.Function(V_dg0, name="hydrostatic_stress_cathode_DG0")
    hydro_omega2_dg0 = fem.Function(V_dg0, name="hydrostatic_stress_omega2_DG0")
    hydro_omega12_dg0 = fem.Function(V_dg0, name="hydrostatic_stress_omega12_DG0")
    hydro_omega23_dg0 = fem.Function(V_dg0, name="hydrostatic_stress_omega23_DG0")
    derived_exprs = (
        auxiliary["ce_expr"],
        auxiliary["gb_window"],
        auxiliary["hxi"],
        auxiliary["sigma_eff"],
        auxiliary["reaction_li_raw"],
        auxiliary["deposition_drive"],
        auxiliary["xi_source_weight"],
        auxiliary["xit_expr"],
        auxiliary["overp_li_volts"],
        auxiliary["overp_c_volts"],
        auxiliary["e_eq"],
        auxiliary["hydro1"],
        auxiliary["overp_mech_volts"],
        ufl.sqrt(ux * ux + uy * uy),
        auxiliary["liion_source_expr"],
        auxiliary["dfmech_dxi"],
        auxiliary["e_eq_eff"],
        auxiliary["hydro3"],
        auxiliary["overp_mech_c_volts"],
        (p.D_li * p.omega_cathode / (p.R * p.T * p.length_scale**2))
        * c
        * ufl.sqrt(ufl.dot(ufl.grad(auxiliary["hydro3"]), ufl.grad(auxiliary["hydro3"]))),
        0.0,
        0.0,
        auxiliary["heta"],
        auxiliary["E1"],
        auxiliary["nu1"],
        0.0,
        0.0,
    )
    omega1_derived_indices = (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 22, 23, 24)
    omega12_derived_indices = (26,)
    recovery_derived_indices = (20, 21, 25, 26)
    omega3_derived_indices = (9, 10, 16, 17, 18, 19, 20)
    light_diagnostic_derived_indices = (8,)
    full_diagnostic_derived_indices = (4, 5, 6, 8, 10, 11, 12, 13, 15, 22, 23, 24)

    update_derived = partial(
        update_derived_fields,
        derived_outputs,
        derived_exprs,
        B,
        auxiliary["B_expr"],
        domain_markers,
        hydro_omega1_dg0,
        hydro_cathode_dg0,
        hydro_omega2_dg0,
        hydro_omega12_dg0,
        hydro_omega23_dg0,
        auxiliary["hydro1"],
        auxiliary["hydro2"],
        auxiliary["hydro3"],
        V_scalar,
        recovery_derived_indices,
        omega1_derived_indices,
        omega12_derived_indices,
        omega3_derived_indices,
    )
    output_runtime_state = {
        "w": w, "scalar_outputs": scalar_outputs, "params": p,
        "component_maps": scalar_component_maps,
        "update_scalar_outputs": update_scalar_outputs,
        "update_derived": update_derived,
    }
    mpi_stage_print(msh.comm, "before initial derived update")
    refresh_output_fields(output_runtime_state)
    mpi_stage_print(msh.comm, "after initial derived update")
    mpi_stage_print(msh.comm, "before initial png outputs")
    save_png_outputs(
        out_dir,
        "initial",
        V_scalar,
        scalar_outputs,
        derived_outputs,
        B,
        domain_markers,
        p,
    )
    mpi_stage_print(msh.comm, "after initial png outputs")
    if p.xdmf_interval > 0.0:
        save_sampled_outputs(
            out_dir,
            "initial",
            V_scalar,
            scalar_outputs,
            derived_outputs,
            B,
            domain_markers,
            p,
        )
    li_metal_form = fem.form((xi / p.omega_li) * p.length_scale**2 * dx_omega1)
    cathode_li_form = fem.form(p.c_li_max * c * p.length_scale**2 * dx(OMEGA3))
    gamma_s_length_form = fem.form(p.length_scale * dS(GAMMA_S))
    gamma_s_current_form = fem.form(auxiliary["i_cathode_s"] * p.length_scale * dS(GAMMA_S))
    gamma_c_current_form = fem.form(
        context.constants.applied_current * p.length_scale * ds(GAMMA_C)
    )
    omega1_xi_current_form = fem.form(
        (p.F / p.omega_li) * auxiliary["xit_expr"] * p.length_scale**2 * dx_omega1
    )
    overp_c_gamma_s_form = fem.form(auxiliary["overp_c_volts_s"] * p.length_scale * dS(GAMMA_S))

    xi_initial_mol_m2 = assemble_form_total(li_metal_form, msh.comm)
    c_initial_mol_m2 = assemble_form_total(cathode_li_form, msh.comm)
    gamma_s_length_m = assemble_form_total(gamma_s_length_form, msh.comm)
    gamma_s_length_scale = max(abs(gamma_s_length_m), 1.0e-30)
    _, _, phil_initial_mean = field_stats(scalar_outputs[1], stats_dofs["phil"])
    _, _, phis_initial_mean = field_stats(scalar_outputs[2], stats_dofs["phis"])
    cell_voltage_initial = phis_initial_mean - phil_initial_mean
    (
        phil_gamma_a_initial_min,
        phil_gamma_a_initial_max,
        phil_gamma_a_initial_mean,
    ) = field_stats(
        scalar_outputs[1], boundary_probe_dofs["phil_gamma_a"]
    )
    (
        phis_gamma_c_initial_min,
        phis_gamma_c_initial_max,
        phis_gamma_c_initial_mean,
    ) = field_stats(
        scalar_outputs[2], boundary_probe_dofs["phis_gamma_c"]
    )
    boundary_voltage_initial = phis_gamma_c_initial_max - phil_gamma_a_initial_min
    current_cycle_index = 0
    current_phase_elapsed = 0.0
    current_phase_duration = 0.0
    current_cutoff_voltage = float("nan")
    current_soc_cutoff = float("nan")
    current_cutoff_reached = False
    current_voltage_cutoff_reached = False
    current_soc_cutoff_reached = False
    current_cutoff_reason = ""

    current_dendrite_tip_y = dendrite_tip_y(
        scalar_outputs[0], stats_dofs["xi"], p.lifecycle_xi_threshold
    )
    current_lifecycle_stop_reached = False
    diagnostic_context = create_diagnostic_context({
        "GAMMA_S": GAMMA_S,
        "abs": abs,
        "append_csv": append_csv,
        "assemble_form_total": assemble_form_total,
        "assemble_total": assemble_total,
        "boundary_probe_dofs": boundary_probe_dofs,
        "boundary_voltage_initial": boundary_voltage_initial,
        "c": c,
        "c_initial_mol_m2": c_initial_mol_m2,
        "c_omega3_s": c_omega3_s,
        "cathode_li_form": cathode_li_form,
        "cathode_side": cathode_side,
        "cathode_y_max": cathode_y_max,
        "constant_scalar_value": constant_scalar_value,
        "current_cutoff_reached": current_cutoff_reached,
        "current_cutoff_reason": current_cutoff_reason,
        "current_cutoff_voltage": current_cutoff_voltage,
        "current_cycle_index": current_cycle_index,
        "current_dendrite_tip_y": current_dendrite_tip_y,
        "current_density": context.constants.current_density,
        "current_lifecycle_stop_reached": current_lifecycle_stop_reached,
        "current_phase_duration": current_phase_duration,
        "current_phase_elapsed": current_phase_elapsed,
        "current_soc_cutoff": current_soc_cutoff,
        "current_soc_cutoff_reached": current_soc_cutoff_reached,
        "current_voltage_cutoff_reached": current_voltage_cutoff_reached,
        "dS": dS,
        "derived_outputs": derived_outputs,
        "dict": dict,
        "e_eq": e_eq,
        "e_eq_omega3_s": e_eq_omega3_s,
        "electrolyte_side": electrolyte_side,
        "field_stats": field_stats,
        "float": float,
        "gamma_c_current_form": gamma_c_current_form,
        "gamma_s_current_form": gamma_s_current_form,
        "gamma_s_length_scale": gamma_s_length_scale,
        "i_cathode_s": i_cathode_s,
        "li_metal_form": li_metal_form,
        "lifecycle_target_y": lifecycle_target_y,
        "math": math,
        "max": max,
        "msh": msh,
        "np": np,
        "omega1_xi_current_form": omega1_xi_current_form,
        "overp_c_gamma_s_form": overp_c_gamma_s_form,
        "overp_c_volts_s": overp_c_volts_s,
        "overp_mech_c_omega3_s": overp_mech_c_omega3_s,
        "overp_mech_c_volts": overp_mech_c_volts,
        "p": p,
        "phil": phil,
        "phil_initial_mean": phil_initial_mean,
        "phil_omega2_s": phil_omega2_s,
        "phis": phis,
        "phis_initial_mean": phis_initial_mean,
        "phis_omega2_s": phis_omega2_s,
        "scalar_outputs": scalar_outputs,
        "stats_dofs": stats_dofs,
        "xi_initial_mol_m2": xi_initial_mol_m2,
    })

    save_latest_state = make_latest_state_saver(
        w, w_n, scalar_outputs, derived_outputs, p, scalar_component_maps,
        V_scalar, B, domain_markers, out_dir, msh, update_scalar_outputs,
        update_derived, save_png_outputs, save_final_npz,
        lambda: derived_output_fields(
            hydro_omega1_dg0, hydro_cathode_dg0, hydro_omega2_dg0,
            hydro_omega12_dg0, hydro_omega23_dg0,
        ),
    )

    mpi_stage_print(msh.comm, "before initial diagnostics")
    write_diagnostics(diagnostic_context, 0, 0.0, "initial", p.dt)
    mpi_stage_print(msh.comm, "after initial diagnostics")

    phases = build_phase_schedule(p)
    _, xi_map = collapse_mixed_subspace(ME, 0)
    _, c_map = collapse_mixed_subspace(ME, 3)
    eta_components = tuple(range(6, 6 + n_grains))
    output_schedule = OutputSchedule(
        p.png_interval, p.xdmf_interval, p.field_output_interval_s
    )
    context.output = output_schedule
    context.diagnostics = diagnostic_context
    context.evolution = EvolutionRuntimeContext(
        params=p, state=w, previous_state=w_n,
        previous_previous_state=w_nm1, older_state=w_nm2,
        previous_state_array=w_prev_array, constants=context.constants,
        problem=problem, profiler=newton_profiler,
        output_state=output_runtime_state, diagnostics=diagnostic_context,
        output_schedule=output_schedule, save_latest_state=save_latest_state,
        comm=msh.comm, model={
            "ME": ME, "V_scalar": V_scalar, "xi_map": xi_map,
            "c_map": c_map, "eta_components": eta_components,
            "scalar_outputs": scalar_outputs, "derived_outputs": derived_outputs,
            "B": B, "domain_markers": domain_markers,
            "stats_dofs": stats_dofs, "boundary_probe_dofs": boundary_probe_dofs,
            "update_derived": update_derived,
            "periodic_constraints": periodic_constraints,
            "clip_mixed_components": clip_mixed_components,
            "out_dir": out_dir, "lifecycle_target_y": lifecycle_target_y,
            "primary_scalar_indices": primary_scalar_indices,
            "light_diagnostic_derived_indices": light_diagnostic_derived_indices,
            "full_diagnostic_derived_indices": full_diagnostic_derived_indices,
        },
        mesh_context=context.mesh, field_context=context.fields,
        equation_context=context.equations, constraint_context=context.constraints,
        output_context=output_runtime_state, diagnostic_context=diagnostic_context,
    )
    if p.field_output_interval_s > 0.0:
        refresh_output_fields(output_runtime_state)
        save_field_snapshot(
            Path(out_dir) / "fields" / "t_000000.000s.npz",
            V_scalar,
            scalar_outputs,
            derived_outputs,
            B,
            domain_markers,
            p,
            variables=p.field_output_variables,
            metadata={"time_s": 0.0, "step": 0, "phase": "initial"},
        )
    runtime = context.evolution
    runtime.model["set_current"] = lambda value: (assign_constant(context.constants.current_density, value), assign_constant(context.constants.applied_current, value))
    runtime.callbacks.update({
        "solve_trial_step": solve_trial_step,
        "set_bdf_coefficients": modular_set_bdf_coefficients,
        "solve_with_retries": solve_with_retries,
        "assign_constant": assign_constant,
        "nonlinear_status": nonlinear_solver_status,
        "estimate_state_time_error": estimate_state_time_error,
        "time_error_history_ready": time_error_history_ready,
        "time_step_decision": time_step_decision,
        "reject_time_step": reject_time_step,
        "next_dt_after_accept": next_dt_after_accept,
        "refresh_output_fields": refresh_output_fields,
        "update_diagnostic_derived_fields": update_diagnostic_derived_fields,
        "evaluate_cutoff_state": evaluate_cutoff_state,
        "field_stats": field_stats,
        "update_diagnostic_state": update_diagnostic_state,
        "write_diagnostics": write_diagnostics,
        "save_scheduled_output": save_scheduled_output,
        "advance_history": advance_history,
        "progress_print": progress_print,
    })
    attempt_step = build_attempt_step(runtime)
    runtime.callbacks["attempt_step"] = attempt_step
    finish_phase = make_phase_output_callback(output_state=output_runtime_state,
        out_dir=out_dir, V_scalar=V_scalar, scalar_outputs=scalar_outputs,
        derived_outputs=derived_outputs, B=B, domain_markers=domain_markers, params=p)

    run_evolution_loop(phases=phases, state={"set_current": runtime.model["set_current"], "runtime": runtime},
        params=p, phase_dt_max=phase_dt_max, advance_step=runtime.callbacks["attempt_step"],
        finish_phase=finish_phase, finish_all=lambda _state: None)
    finalize_outputs(output_state=output_runtime_state, out_dir=out_dir,
        V_scalar=V_scalar, scalar_outputs=scalar_outputs,
        derived_outputs=derived_outputs, B=B, domain_markers=domain_markers,
        params=p, initial_outputs=initial_outputs,
        extra_outputs=derived_output_fields(
            hydro_omega1_dg0, hydro_cathode_dg0, hydro_omega2_dg0,
            hydro_omega12_dg0, hydro_omega23_dg0), rank=msh.comm.rank)


def execute(request: SimulationRequest):
    """Run the simulation implementation owned by this workflow module."""
    return request.call(run_simulation)


def run(*args, **kwargs):
    request = prepare_request(*args, **kwargs)
    return execute(request)


__all__ = [
    "DEFAULT_GB_FILE",
    "DEFAULT_MSH_FILE",
    "SimulationRequest",
    "prepare_request",
    "run_simulation",
    "execute",
    "run",
]
