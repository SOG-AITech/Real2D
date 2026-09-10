"""High-level phase/time-loop orchestration.

The numerical step remains supplied by the workflow callback.  This module
owns the phase traversal, zero-duration phases, and the stop contract so the
workflow does not have to implement another scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass
class EvolutionContext:
    """Shared mutable clock for the complete charge/discharge evolution."""

    phases: Iterable[tuple[Any, ...]]
    state: dict[str, Any]
    total_step: int = 0
    global_time: float = 0.0
    lifecycle_terminated: bool = False


@dataclass
class EvolutionRuntimeContext:
    """All mutable FEM, time-integration, diagnostics, and output state.

    The context deliberately stores callbacks for model-specific operations;
    the evolution driver never reaches into workflow locals directly.
    """

    params: Any
    state: Any
    previous_state: Any
    previous_previous_state: Any
    older_state: Any
    previous_state_array: Any
    constants: Any
    problem: Any
    profiler: Any
    output_state: dict[str, Any]
    diagnostics: dict[str, Any]
    output_schedule: Any
    save_latest_state: Callable[..., Any]
    comm: Any
    total_step: int = 0
    global_time: float = 0.0
    lifecycle_terminated: bool = False
    current_phase: Any = None
    model: dict[str, Any] = field(default_factory=dict)
    # References to the aggregate context sections.  Keeping these on the
    # runtime object makes the callback boundary explicit and avoids growing
    # closures over workflow locals as more operations are extracted.
    mesh_context: Any = None
    field_context: Any = None
    equation_context: Any = None
    constraint_context: Any = None
    output_context: Any = None
    diagnostic_context: Any = None
    callbacks: dict[str, Any] = field(default_factory=dict)

    def clock(self) -> dict[str, Any]:
        return {"total_step": self.total_step, "time_s": self.global_time}

    def sync_clock(self, clock: dict[str, Any]) -> None:
        self.total_step = int(clock.get("total_step", self.total_step))
        self.global_time = float(clock.get("time_s", self.global_time))


@dataclass
class PhaseState:
    cycle_index: int
    phase_name: str
    duration: float
    current_value: float
    cutoff_voltage: float
    current_dt_max: float
    fixed_dt: float
    dt_trial: float
    phase_elapsed: float = 0.0
    accepted_steps: int = 0
    previous_accepted_dt: float | None = None
    previous_previous_accepted_dt: float | None = None
    retry_recovery_remaining: int = 0
    total_step: int = 0
    global_time: float = 0.0
    lifecycle_terminated: bool = False


def initialize_phase(phase, *, params, phase_dt_max):
    """Create the mutable state for one charge/discharge phase."""
    cycle_index, phase_name, duration, current_value, cutoff_voltage = phase
    current_dt_max = phase_dt_max(params, phase_name)
    fixed_dt = params.dt if params.fixed_initial_dt is None else params.fixed_initial_dt
    fixed_dt = min(max(fixed_dt, params.dt_min), current_dt_max)
    dt_trial = (
        fixed_dt if params.fixed_initial_steps > 0
        else min(max(params.dt, params.dt_min), current_dt_max)
    )
    return PhaseState(cycle_index, phase_name, duration, current_value,
                      cutoff_voltage, current_dt_max, fixed_dt, dt_trial)


def accept_step(state: PhaseState, *, dt_used, retries_used, time_error,
                have_error_history, params, next_dt_after_accept):
    """Advance phase clocks after an accepted nonlinear solve."""
    state.accepted_steps += 1
    state.phase_elapsed += float(dt_used)
    state.dt_trial, state.retry_recovery_remaining = next_dt_after_accept(
        params, state.accepted_steps, state.fixed_dt, dt_used, state.current_dt_max,
        retries_used, have_error_history, time_error,
        state.retry_recovery_remaining,
    )
    state.previous_previous_accepted_dt = state.previous_accepted_dt
    state.previous_accepted_dt = float(dt_used)
    return state


def trial_dt(state: PhaseState) -> float:
    """Return the next bounded trial step for the active phase."""
    return min(float(state.dt_trial), max(0.0, state.duration - state.phase_elapsed))


def solve_trial_step(runtime: EvolutionRuntimeContext, phase: PhaseState, *,
                     accepted_step, global_time, set_bdf_coefficients,
                     solve_with_retries, assign_constant, nonlinear_status):
    """Prepare BDF coefficients and solve one phase trial with retries."""
    dt_used = trial_dt(phase)
    constants = runtime.constants
    assign_constant(constants.dt, dt_used)
    set_bdf_coefficients(
        dt_used, phase.previous_accepted_dt, phase.accepted_steps > 0,
        assign_constant, constants.bdf_a0, constants.bdf_a1, constants.bdf_a2,
    )
    result = solve_with_retries(
        params=runtime.params,
        phase_name=phase.phase_name,
        dt_used=dt_used,
        previous_dt=phase.previous_accepted_dt,
        accepted_steps=phase.accepted_steps,
        accepted_step=accepted_step,
        global_time=global_time,
        state=runtime.state,
        previous_state=runtime.previous_state,
        previous_state_array=runtime.previous_state_array,
        periodic_constraints=runtime.model["periodic_constraints"],
        problem=runtime.problem,
        profiler=runtime.profiler,
        dt_constant=constants.dt,
        bdf_coefficients=(constants.bdf_a0, constants.bdf_a1, constants.bdf_a2),
        assign_constant=assign_constant,
        nonlinear_status=nonlinear_status,
        scatter_forward=runtime.state.x.scatter_forward,
        save_latest_state=runtime.save_latest_state,
        comm=runtime.comm,
    )
    dt_solved, retries, reason, iterations, residual = result
    if runtime.params.clip_eta_after_solve:
        runtime.model["clip_mixed_components"](
            runtime.state, runtime.model["ME"], runtime.model["eta_components"], 0.0, 1.0
        )
    runtime.state.x.scatter_forward()
    return {
        "dt_used": dt_solved,
        "retries_used": retries,
        "snes_reason": reason,
        "snes_iterations": iterations,
        "snes_residual": residual,
    }


def advance_history(*, state, previous_state, previous_previous_state, older_state,
                    previous_state_array, scatter_forward):
    """Shift accepted mixed states after diagnostics have been written."""
    previous_state_array[:] = previous_state.x.array
    older_state.x.array[:] = previous_previous_state.x.array
    older_state.x.scatter_forward()
    previous_previous_state.x.array[:] = previous_state.x.array
    previous_previous_state.x.scatter_forward()
    previous_state.x.array[:] = state.x.array
    previous_state.x.scatter_forward()
    return previous_state_array


def phase_should_stop(*, lifecycle_terminated, phase_result=False):
    """Return whether the phase scheduler should stop after a phase."""
    return bool(lifecycle_terminated or phase_result)


def run_evolution_loop(*, phases, state, params, phase_dt_max,
                       advance_step=None, finish_phase=None, finish_all=None,
                       callbacks=None):
    """Own complete phase and accepted-step loop control.

    ``advance_step`` performs one solver attempt and all accepted-step side
    effects, returning a mapping with ``accepted`` and optional ``dt_trial``,
    ``stop_phase`` and ``lifecycle_stop`` values. Rejected adaptive trials do
    not advance either simulation clock.
    """
    if callbacks is not None:
        state["set_current"] = callbacks.set_current
        advance_step = callbacks.attempt_step
        finish_phase = callbacks.finish_phase
        finish_all = callbacks.finish_all
    if advance_step is None or finish_phase is None or finish_all is None:
        raise TypeError("run_evolution_loop requires evolution callbacks")
    state.setdefault("total_step", 0)
    state.setdefault("global_time", 0.0)
    state.setdefault("lifecycle_terminated", False)
    evolution = EvolutionContext(phases=phases, state=state,
        total_step=int(state["total_step"]), global_time=float(state["global_time"]),
        lifecycle_terminated=bool(state["lifecycle_terminated"]))
    state.setdefault("evolution", evolution)
    for raw_phase in iter_scheduled_phases(phases):
        phase = initialize_phase(raw_phase, params=params, phase_dt_max=phase_dt_max)
        state["phase"] = phase
        state["set_current"](phase.current_value)
        def attempt(current_phase, clock_state):
            return advance_step(current_phase, clock_state)

        def finish(current_phase, clock):
            evolution.total_step = int(clock.get("total_step", evolution.total_step))
            evolution.global_time = float(clock.get("time_s", evolution.global_time))
            if current_phase.lifecycle_terminated:
                evolution.lifecycle_terminated = True
                state["lifecycle_terminated"] = True

        clock = {"total_step": evolution.total_step, "time_s": evolution.global_time}
        run_phase_iterations(phase, global_clock=clock, attempt_step=attempt,
                             finish_phase=finish)
        evolution.total_step = int(clock["total_step"])
        evolution.global_time = float(clock["time_s"])
        state["total_step"] = evolution.total_step
        state["global_time"] = evolution.global_time
        runtime = state.get("runtime")
        if runtime is not None:
            runtime.sync_clock(clock)
            runtime.lifecycle_terminated = bool(state["lifecycle_terminated"])
        finish_phase(phase, state)
        if phase_should_stop(lifecycle_terminated=state["lifecycle_terminated"]):
            break
    finish_all(state)
    return state


def run_phase_iterations(phase: PhaseState, *, global_clock, attempt_step,
                        finish_phase):
    """Execute a phase's accepted-step loop in the evolution module.

    ``attempt_step`` performs exactly one solve/adaptive-error attempt and
    returns an outcome mapping. Rejected attempts update only ``dt_trial``;
    accepted attempts advance phase/global clocks here. The callback is
    responsible for model-specific fields, diagnostics, and output side effects.
    """
    while phase.phase_elapsed < phase.duration - 1.0e-15:
        outcome = attempt_step(phase, global_clock)
        phase.dt_trial = float(outcome["dt_trial"])
        if not outcome.get("accepted", False):
            continue
        dt = float(outcome["dt_used"])
        phase.phase_elapsed += dt
        phase.accepted_steps += 1
        phase.previous_previous_accepted_dt = phase.previous_accepted_dt
        phase.previous_accepted_dt = dt
        phase.retry_recovery_remaining = int(
            outcome.get("retry_recovery_remaining", phase.retry_recovery_remaining)
        )
        global_clock["time_s"] += dt
        global_clock["total_step"] += 1
        if outcome.get("lifecycle_stop", False):
            phase.lifecycle_terminated = True
            break
        if outcome.get("stop_phase", False):
            break
    return phase


def iter_scheduled_phases(phases):
    """Yield positive-duration phases, preserving the schedule order."""
    for phase in phases:
        if len(phase) >= 3 and float(phase[2]) <= 0.0:
            continue
        yield phase


__all__ = ["EvolutionContext", "EvolutionRuntimeContext", "PhaseState", "initialize_phase", "accept_step", "trial_dt",
           "solve_trial_step",
           "advance_history", "phase_should_stop",
           "iter_scheduled_phases", "run_evolution_loop",
           "run_phase_iterations"]
