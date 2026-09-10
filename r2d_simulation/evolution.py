"""Accepted-step evolution and adaptive-time-step interfaces."""

import math
import numpy as np
from mpi4py import MPI


def run_time_evolution(*, phases, phase_runner):
    """Run phase sequence through a supplied phase-step callback."""
    for phase in phases:
        if phase_runner(phase) is False:
            break


def time_error_step_factor(error_value, params):
    if math.isfinite(error_value) and error_value > 0.0:
        target = max(params.time_adapt_tol_min, params.time_adapt_target_fraction * params.time_adapt_tol_max)
        factor = params.time_adapt_safety * (target / max(error_value, 1.0e-300)) ** (1.0 / 3.0)
    else:
        factor = params.time_adapt_factor_max
    return min(max(factor, params.time_adapt_factor_min), params.time_adapt_factor_max)


def next_dt_from_time_error(error_value, dt_used, current_dt_max, params):
    factor = time_error_step_factor(error_value, params)
    return min(max(float(dt_used) * factor, params.dt_min), current_dt_max)


def local_error_component(current, previous, previous_previous, older, dt_np1, dt_n, dt_nm1, params, comm):
    current = np.asarray(current, dtype=np.float64)
    if current.size == 0 or dt_np1 <= 0.0 or dt_n <= 0.0 or dt_nm1 <= 0.0:
        local = np.array([0.0, 0.0], dtype=np.float64)
    else:
        ratio = float(dt_np1) / float(dt_n)
        d0 = (current - previous) / float(dt_np1)
        d1 = (previous - previous_previous) / float(dt_n)
        d2 = (previous_previous - older) / float(dt_nm1)
        tau = (float(dt_np1) ** 2 / 6.0) * (d0 - (1.0 + ratio) * d1 + ratio * d2)
        scale = params.time_adapt_rho_abs + params.time_adapt_rho_rel * np.maximum(np.abs(current), np.abs(previous))
        scaled = tau / np.maximum(scale, 1.0e-300)
        local = np.array([float(np.dot(scaled, scaled)), float(scaled.size)], dtype=np.float64)
    global_pair = np.zeros(2, dtype=np.float64)
    comm.Allreduce(local, global_pair, op=MPI.SUM)
    return math.sqrt(float(global_pair[0]) / float(global_pair[1])) if global_pair[1] > 0 else 0.0


def estimate_time_error(current_xi, previous_xi, previous_previous_xi, older_xi,
                        current_c, previous_c, previous_previous_c, older_c,
                        dt_np1, dt_n, dt_nm1, params, comm):
    err_xi = local_error_component(current_xi, previous_xi, previous_previous_xi, older_xi, dt_np1, dt_n, dt_nm1, params, comm)
    err_c = local_error_component(current_c, previous_c, previous_previous_c, older_c, dt_np1, dt_n, dt_nm1, params, comm)
    return max(err_xi, err_c), err_xi, err_c


def estimate_state_time_error(
    current, previous, previous_previous, older,
    xi_map, c_map, xi_dofs, c_dofs, scalar_space, params, comm,
    dt_np1, dt_n, dt_nm1,
):
    """Estimate the local time error directly from mixed-state Functions."""
    owned_size = scalar_space.dofmap.index_map.size_local

    def component_error(component_map, scalar_dofs):
        if scalar_dofs is None:
            scalar_dofs = np.empty(0, dtype=np.int64)
        else:
            scalar_dofs = np.asarray(scalar_dofs, dtype=np.int64)
        scalar_dofs = scalar_dofs[(scalar_dofs >= 0) & (scalar_dofs < owned_size)]
        mixed_dofs = component_map[scalar_dofs.astype(np.int32, copy=False)] if scalar_dofs.size else np.empty(0, dtype=np.int32)
        u0 = np.asarray(current.x.array[mixed_dofs].real, dtype=np.float64)
        u1 = np.asarray(previous.x.array[mixed_dofs].real, dtype=np.float64)
        u2 = np.asarray(previous_previous.x.array[mixed_dofs].real, dtype=np.float64)
        u3 = np.asarray(older.x.array[mixed_dofs].real, dtype=np.float64)
        if u0.size == 0 or dt_np1 <= 0.0 or dt_n <= 0.0 or dt_nm1 <= 0.0:
            pair = np.array([0.0, 0.0], dtype=np.float64)
        else:
            ratio = float(dt_np1) / float(dt_n)
            d0 = (u0 - u1) / float(dt_np1)
            d1 = (u1 - u2) / float(dt_n)
            d2 = (u2 - u3) / float(dt_nm1)
            tau = (float(dt_np1) ** 2 / 6.0) * (d0 - (1.0 + ratio) * d1 + ratio * d2)
            scale = params.time_adapt_rho_abs + params.time_adapt_rho_rel * np.maximum(np.abs(u0), np.abs(u1))
            scaled = tau / np.maximum(scale, 1.0e-300)
            pair = np.array([float(np.dot(scaled, scaled)), float(scaled.size)], dtype=np.float64)
        global_pair = np.zeros(2, dtype=np.float64)
        comm.Allreduce(pair, global_pair, op=MPI.SUM)
        return math.sqrt(float(global_pair[0]) / float(global_pair[1])) if global_pair[1] > 0 else 0.0

    err_xi = component_error(xi_map, xi_dofs)
    err_c = component_error(c_map, c_dofs)
    return max(err_xi, err_c), err_xi, err_c


def set_bdf_coefficients(dt_used, previous_dt, use_bdf2, assign, a0, a1, a2):
    """Set the BDF1/BDF2 coefficients used by the transient residual."""
    if use_bdf2 and previous_dt is not None and previous_dt > 0.0:
        ratio = float(dt_used) / float(previous_dt)
        assign(a0, (1.0 + 2.0 * ratio) / (1.0 + ratio))
        assign(a1, -(1.0 + ratio))
        assign(a2, (ratio * ratio) / (1.0 + ratio))
    else:
        assign(a0, 1.0)
        assign(a1, -1.0)
        assign(a2, 0.0)


def solve_with_retries(
    *, params, phase_name, dt_used, previous_dt, accepted_steps,
    accepted_step, global_time, state, previous_state, previous_state_array,
    periodic_constraints, problem, profiler, dt_constant, bdf_coefficients,
    assign_constant, nonlinear_status, scatter_forward, save_latest_state,
    comm,
):
    """Solve one trial step, shrinking ``dt`` and retrying on failure.

    The callbacks keep this evolution routine independent of the concrete
    DOLFINx field container and output implementation.  The retry semantics
    match the original workflow loop exactly.
    """
    dt_trial = float(dt_used)
    retries_used = 0
    a0, a1, a2 = bdf_coefficients
    set_bdf_coefficients(
        dt_trial, previous_dt, accepted_steps > 0,
        assign_constant, a0, a1, a2,
    )

    for retry in range(params.max_retries_per_step):
        use_extrapolated_guess = (
            params.extrapolate_initial_guess
            and retry == 0
            and accepted_steps > 0
            and previous_dt is not None
            and previous_dt > 0.0
            and params.extrapolate_max_factor > 0.0
        )
        if use_extrapolated_guess:
            factor = min(
                params.extrapolate_max_factor,
                max(0.0, dt_trial / previous_dt),
            )
            state.x.array[:] = (
                previous_state.x.array
                + factor * (previous_state.x.array - previous_state_array)
            )
            for slave, master in periodic_constraints.items():
                state.x.array[slave] = state.x.array[master]
        else:
            state.x.array[:] = previous_state.x.array
        scatter_forward()
        try:
            profiler.set_context(
                accepted_step=accepted_step,
                global_time_s=global_time,
                phase=phase_name,
                dt_s=dt_trial,
                retry=retry,
            )
            problem.solve()
            scatter_forward()
            snes_reason, snes_its, snes_residual = nonlinear_status(problem)
            if isinstance(snes_reason, (int, np.integer)) and snes_reason <= 0:
                raise RuntimeError(
                    f"SNES did not converge, reason={snes_reason}, "
                    f"its={snes_its}, fnorm={snes_residual:.3e}"
                )
            return dt_trial, retries_used, snes_reason, snes_its, snes_residual
        except Exception as exc:
            retries_used += 1
            problem.reset_jacobian_cache()
            if comm.rank == 0:
                print(
                    f"retry at t={global_time:.3e}s with dt={dt_trial:.3e}s "
                    f"after: {exc}"
                )
            dt_trial *= params.dt_shrink
            if dt_trial < params.dt_min:
                save_latest_state(
                    f"{phase_name} phase failed: dt dropped below "
                    f"dt_min={params.dt_min:g} s"
                )
                raise RuntimeError(
                    f"{phase_name} phase failed: dt dropped below "
                    f"dt_min={params.dt_min:g} s"
                )
            assign_constant(dt_constant, dt_trial)
            set_bdf_coefficients(
                dt_trial, previous_dt, accepted_steps > 0,
                assign_constant, a0, a1, a2,
            )

    save_latest_state(
        f"{phase_name} phase could not find a converged adaptive step."
    )
    raise RuntimeError(
        f"{phase_name} phase could not find a converged adaptive step."
    )


def time_error_history_ready(accepted_steps, previous_dt, previous_previous_dt):
    return bool(
        accepted_steps >= 2
        and previous_dt is not None
        and previous_previous_dt is not None
        and previous_dt > 0.0
        and previous_previous_dt > 0.0
    )


def time_step_decision(params, dt_used, current_dt_max, time_error):
    """Compute the adaptive-step diagnostics and accept/reject decision."""
    factor = time_error_step_factor(time_error, params)
    dt_next = min(
        max(float(dt_used) * factor, params.dt_min), current_dt_max
    )
    reject_limit = params.time_adapt_tol_max * max(
        1.0, params.time_adapt_reject_factor
    )
    rejected = bool(
        params.time_adapt_reject
        and math.isfinite(time_error)
        and time_error > reject_limit
    )
    rejected_dt = None
    if rejected:
        rejected_dt = next_dt_from_time_error(
            time_error, dt_used, current_dt_max, params
        )
        if rejected_dt >= dt_used * (1.0 - 1.0e-12):
            rejected_dt = max(params.dt_min, dt_used * params.time_adapt_factor_min)
    return factor, dt_next, reject_limit, rejected, rejected_dt


def reject_time_step(
    *, params, phase_name, time_s, dt_used, time_error, error_xi, error_c,
    rejected_dt, current_dt_max, previous_dt, accepted_steps, state, previous_state, problem,
    dt_constant, bdf_coefficients, assign_constant, scatter_forward,
    save_latest_state, comm,
):
    """Restore the last accepted state and prepare a smaller retry step."""
    if rejected_dt < params.dt_min or dt_used <= params.dt_min * (1.0 + 1.0e-12):
        save_latest_state(
            f"{phase_name} phase failed: local time error {time_error:.3e} "
            f"cannot be reduced below dt_min={params.dt_min:g} s"
        )
        raise RuntimeError(
            f"{phase_name} phase failed: local time error {time_error:.3e} "
            f"cannot be reduced below dt_min={params.dt_min:g} s"
        )
    problem.reset_jacobian_cache()
    state.x.array[:] = previous_state.x.array
    scatter_forward()
    if comm.rank == 0:
        print(
            f"time-adapt reject at t={time_s:.3e}s: "
            f"E={time_error:.3e} (xi={error_xi:.3e}, c={error_c:.3e}) "
            f"> reject_limit={params.time_adapt_tol_max * max(1.0, params.time_adapt_reject_factor):.3e}; "
            f"dt {dt_used:.3e} -> {rejected_dt:.3e}; retrying",
            flush=True,
        )
    dt_trial = min(max(rejected_dt, params.dt_min), current_dt_max)
    assign_constant(dt_constant, dt_trial)
    a0, a1, a2 = bdf_coefficients
    set_bdf_coefficients(
        dt_trial, previous_dt, accepted_steps > 0,
        assign_constant, a0, a1, a2,
    )
    return dt_trial


def next_dt_after_accept(
    params, accepted_steps, fixed_dt, dt_used, current_dt_max, retries_used,
    have_error_history, time_error, retry_recovery_remaining,
):
    """Choose the next trial dt and updated retry-recovery counter."""
    if accepted_steps < params.fixed_initial_steps:
        return fixed_dt, retry_recovery_remaining
    if retries_used > 0:
        retry_recovery_remaining = params.retry_recovery_steps
        dt_trial = min(max(dt_used, params.dt_min), current_dt_max)
    elif have_error_history and math.isfinite(time_error):
        dt_trial = next_dt_from_time_error(
            time_error, dt_used, current_dt_max, params
        )
        if retry_recovery_remaining > 0:
            recovery_cap = min(
                current_dt_max,
                max(params.dt_min, dt_used * params.retry_recovery_growth),
            )
            dt_trial = min(dt_trial, recovery_cap)
            retry_recovery_remaining -= 1
    else:
        growth = params.time_adapt_factor_max
        if retry_recovery_remaining > 0:
            growth = min(growth, params.retry_recovery_growth)
            retry_recovery_remaining -= 1
        dt_trial = min(max(dt_used * growth, params.dt_min), current_dt_max)
    return dt_trial, retry_recovery_remaining


def advance(*args, **kwargs):
    from .workflow import run

    return run(*args, **kwargs)


__all__ = [
    "advance", "estimate_time_error", "estimate_state_time_error",
    "next_dt_from_time_error", "time_error_step_factor", "local_error_component",
    "set_bdf_coefficients", "solve_with_retries",
    "time_error_history_ready", "time_step_decision", "reject_time_step",
    "next_dt_after_accept",
]
