"""Charge/discharge cycle orchestration."""


def build_phase_schedule(params):
    """Return the charge/discharge phases in the established order."""
    phases = []
    for cycle_index in range(1, params.cycle_count + 1):
        if params.charge_time > 0.0:
            phases.append(
                (
                    cycle_index,
                    "charge",
                    params.charge_time,
                    params.charge_current_sign * params.current_abs,
                    params.charge_cutoff_voltage,
                )
            )
        if params.discharge_time > 0.0:
            phases.append(
                (
                    cycle_index,
                    "discharge",
                    params.discharge_time,
                    params.discharge_current_sign * params.current_abs,
                    params.discharge_cutoff_voltage,
                )
            )
    return phases


def phase_dt_max(params, phase_name):
    """Return the configured maximum step for a charge/discharge phase."""
    result = params.dt_max
    if phase_name == "charge" and params.dt_max_charge is not None:
        result = min(result, float(params.dt_max_charge))
    elif phase_name == "discharge" and params.dt_max_discharge is not None:
        result = min(result, float(params.dt_max_discharge))
    return result

def run_cycles(*args, **kwargs):
    from .workflow import run

    return run(*args, **kwargs)

__all__ = ["build_phase_schedule", "phase_dt_max", "run_cycles"]
