"""Model-specific callbacks used by the generic evolution loop."""

from __future__ import annotations


def build_attempt_step(runtime):
    """Create the complete model callback for one trial/accepted step."""
    model = runtime.model
    operations = runtime.callbacks
    solve_trial_step = operations["solve_trial_step"]
    set_bdf_coefficients = operations["set_bdf_coefficients"]
    solve_with_retries = operations["solve_with_retries"]
    assign_constant = operations["assign_constant"]
    nonlinear_status = operations["nonlinear_status"]
    estimate_state_time_error = operations["estimate_state_time_error"]
    time_error_history_ready = operations["time_error_history_ready"]
    time_step_decision = operations["time_step_decision"]
    reject_time_step = operations["reject_time_step"]
    next_dt_after_accept = operations["next_dt_after_accept"]
    refresh_output_fields = operations["refresh_output_fields"]
    update_diagnostic_derived_fields = operations["update_diagnostic_derived_fields"]
    evaluate_cutoff_state = operations["evaluate_cutoff_state"]
    field_stats = operations["field_stats"]
    update_diagnostic_state = operations["update_diagnostic_state"]
    write_diagnostics = operations["write_diagnostics"]
    save_scheduled_output = operations["save_scheduled_output"]
    advance_history = operations["advance_history"]
    update_derived = model["update_derived"]
    diagnostic_context = runtime.diagnostics
    output_runtime_state = runtime.output_state
    output_schedule = runtime.output_schedule
    out_dir = model["out_dir"]
    V_scalar = model["V_scalar"]
    scalar_outputs = model["scalar_outputs"]
    derived_outputs = model["derived_outputs"]
    B = model["B"]
    domain_markers = model["domain_markers"]
    params = runtime.params
    mesh_comm = runtime.comm
    state = runtime.state
    previous_state = runtime.previous_state
    previous_previous_state = runtime.previous_previous_state
    older_state = runtime.older_state
    previous_state_array = runtime.previous_state_array
    problem = runtime.problem
    profiler = runtime.profiler
    constants = runtime.constants
    save_latest_state = runtime.save_latest_state
    c_map = model["c_map"]
    xi_map = model["xi_map"]
    stats_dofs = model["stats_dofs"]
    boundary_probe_dofs = model["boundary_probe_dofs"]
    lifecycle_target_y = model["lifecycle_target_y"]
    light_diagnostic_derived_indices = model["light_diagnostic_derived_indices"]
    full_diagnostic_derived_indices = model["full_diagnostic_derived_indices"]
    clip_indices = model["primary_scalar_indices"]
    progress_print = operations["progress_print"]
    def attempt_step(phase, clock):
        result = solve_trial_step(runtime, phase, accepted_step=clock["total_step"] + 1,
            global_time=clock["time_s"], set_bdf_coefficients=set_bdf_coefficients,
            solve_with_retries=solve_with_retries, assign_constant=assign_constant,
            nonlinear_status=nonlinear_status)
        dt_used = result["dt_used"]
        error = error_xi = error_c = float("nan")
        have_history = time_error_history_ready(phase.accepted_steps,
            phase.previous_accepted_dt, phase.previous_previous_accepted_dt)
        if have_history:
            error, error_xi, error_c = estimate_state_time_error(
                state, previous_state, previous_previous_state, older_state,
                xi_map, c_map, stats_dofs["xi"], stats_dofs["c"], V_scalar,
                params, mesh_comm, dt_used, phase.previous_accepted_dt,
                phase.previous_previous_accepted_dt)
            factor, dt_next, reject_limit, reject, rejected_dt = time_step_decision(
                params, dt_used, phase.current_dt_max, error)
            profiler.write_time_adapt({"time_error": error, "time_error_xi": error_xi,
                "time_error_c": error_c, "time_adapt_factor": factor,
                "time_adapt_dt_next": dt_next, "time_adapt_reject_limit": reject_limit,
                "time_adapt_rejected": int(reject)})
            if reject:
                next_dt = reject_time_step(params=params, phase_name=phase.phase_name,
                    time_s=clock["time_s"], dt_used=dt_used, time_error=error,
                    error_xi=error_xi, error_c=error_c, rejected_dt=rejected_dt,
                    current_dt_max=phase.current_dt_max, previous_dt=phase.previous_accepted_dt,
                    accepted_steps=phase.accepted_steps, state=state,
                    previous_state=previous_state, problem=problem,
                    dt_constant=constants.dt,
                    bdf_coefficients=(constants.bdf_a0, constants.bdf_a1, constants.bdf_a2),
                    assign_constant=assign_constant, scatter_forward=state.x.scatter_forward,
                    save_latest_state=save_latest_state, comm=mesh_comm)
                return {"accepted": False, "dt_trial": next_dt}
        next_dt, next_recovery = next_dt_after_accept(
            params, phase.accepted_steps + 1, phase.fixed_dt, dt_used,
            phase.current_dt_max, result["retries_used"], have_history, error,
            phase.retry_recovery_remaining)
        phase_elapsed_next = phase.phase_elapsed + dt_used
        refresh_output_fields(output_runtime_state, full=False, indices=clip_indices)
        update_diagnostic_derived_fields(update_derived, params.light_diagnostics,
            light_diagnostic_derived_indices, full_diagnostic_derived_indices)
        cutoff = evaluate_cutoff_state(
            phase=phase.phase_name, cutoff_voltage=phase.cutoff_voltage, params=params,
            comm=mesh_comm,
            phil_boundary_stats=field_stats(scalar_outputs[1], boundary_probe_dofs["phil_gamma_a"]),
            phis_boundary_stats=field_stats(scalar_outputs[2], boundary_probe_dofs["phis_gamma_c"]),
            cathode_stats=field_stats(scalar_outputs[3], stats_dofs["c"]),
            xi_function=scalar_outputs[0], xi_dofs=stats_dofs["xi"], target_y=lifecycle_target_y)
        step = clock["total_step"] + 1
        time_next = clock["time_s"] + dt_used
        if step % params.diagnostics_interval == 0:
            update_diagnostic_state(diagnostic_context, {
                "current_cycle_index": phase.cycle_index, "current_phase_elapsed": phase_elapsed_next,
                "current_phase_duration": phase.duration, "current_cutoff_voltage": phase.cutoff_voltage,
                "current_soc_cutoff": cutoff.soc_cutoff, "current_cutoff_reached": cutoff.reached,
                "current_voltage_cutoff_reached": cutoff.voltage_hit, "current_soc_cutoff_reached": cutoff.soc_hit,
                "current_cutoff_reason": cutoff.reason, "current_lifecycle_stop_reached": cutoff.lifecycle_hit,
                "current_dendrite_tip_y": cutoff.dendrite_tip_y})
            write_diagnostics(diagnostic_context, step, time_next, phase.phase_name, dt_used)
        for kind in ("png", "xdmf", "field"):
            save_scheduled_output(schedule=output_schedule, kind=kind, time_s=time_next,
                output_state=output_runtime_state, out_dir=out_dir, V_scalar=V_scalar,
                scalar_outputs=scalar_outputs, derived_outputs=derived_outputs, B=B,
                domain_markers=domain_markers, params=params, cycle=phase.cycle_index,
                phase=phase.phase_name, step=step)
        if mesh_comm.rank == 0:
            progress_print(f"cycle={phase.cycle_index}, {phase.phase_name}: accepted_step={phase.accepted_steps + 1}, phase_t={phase_elapsed_next:.3f} s, global_t={time_next:.3f} s, dt={dt_used:.4f} s, V={cutoff.terminal_voltage:.6g} V, cutoff={phase.cutoff_voltage:.6g} V, E_time={error:.3e}, SNES reason={result['snes_reason']}, its={result['snes_iterations']}")
        advance_history(state=state, previous_state=previous_state,
            previous_previous_state=previous_previous_state, older_state=older_state,
            previous_state_array=previous_state_array, scatter_forward=state.x.scatter_forward)
        if cutoff.reached and mesh_comm.rank == 0:
            print(f"cycle={phase.cycle_index}, {phase.phase_name}: cutoff reached, reason={cutoff.reason}, V={cutoff.terminal_voltage:.6g} V, V_cutoff={phase.cutoff_voltage:.6g} V")
        return {"accepted": True, "dt_used": dt_used, "dt_trial": next_dt,
                "retry_recovery_remaining": next_recovery, "stop_phase": cutoff.reached,
                "lifecycle_stop": cutoff.lifecycle_hit}
    return attempt_step
