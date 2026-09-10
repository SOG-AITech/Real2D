"""Command-line entry point for the R2D simulation."""

import argparse

from .config import DEFAULT_GB_FILE, DEFAULT_MSH_FILE


def _parser():
    parser = argparse.ArgumentParser(
        description="R2D periodic mechanics charge-discharge cycling experiment."
    )
    parser.add_argument("--msh", default=DEFAULT_MSH_FILE)
    parser.add_argument("--gb", default=DEFAULT_GB_FILE)
    options = {
        "--t-end": ("charge_time", float),
        "--discharge-time": ("discharge_time", float),
        "--cycles": ("cycle_count", int),
        "--charge-cutoff-voltage": ("charge_cutoff_voltage", float),
        "--discharge-cutoff-voltage": ("discharge_cutoff_voltage", float),
        "--dt": ("dt", float), "--dt-min": ("dt_min", float),
        "--li-side-potential": ("li_side_potential", float),
        "--soc-init": ("soc_init", float), "--soc-min": ("soc_min", float),
        "--soc-max": ("soc_max", float), "--D-li": ("D_li", float),
        "--i0-ref-li": ("i0_ref_li", float), "--i0-c-ref": ("i0_c_ref", float),
        "--c-li-max": ("c_li_max", float), "--temperature": ("temperature", float),
        "--initial-cathode-overpotential": ("initial_cathode_overpotential", float),
        "--phis-init": ("phis_init", float),
        "--charge-current-sign": ("charge_current_sign", float),
        "--discharge-current-sign": ("discharge_current_sign", float),
        "--current-density": ("current_abs", float),
        "--E-li": ("E_li", float), "--nu-li": ("nu_li", float),
        "--E-se": ("E_se", float), "--nu-se": ("nu_se", float),
        "--E-cathode": ("E_cathode", float), "--nu-cathode": ("nu_cathode", float),
        "--anode-compressive-load": ("anode_compressive_load", float),
        "--cathode-swelling-scale": ("cathode_swelling_scale", float),
        "--gamma": ("gamma", float), "--k-xi": ("k_xi", float),
        "--W-xi": ("W_xi", float), "--L-gb": ("L_gb", float),
        "--W-gb": ("W_gb", float), "--k-gb": ("k_gb", float),
        "--mechanics-residual-scale": ("mechanics_residual_scale", float),
        "--li-layer-thickness": ("li_layer_thickness", float),
        "--dt-max": ("dt_max", float), "--dt-max-charge": ("dt_max_charge", float),
        "--dt-max-discharge": ("dt_max_discharge", float),
        "--dt-growth": ("dt_growth", float), "--dt-shrink": ("dt_shrink", float),
        "--fixed-initial-steps": ("fixed_initial_steps", int),
        "--fixed-initial-dt": ("fixed_initial_dt", float),
        "--time-adapt-tol-max": ("time_adapt_tol_max", float),
        "--time-adapt-tol-min": ("time_adapt_tol_min", float),
        "--time-adapt-safety": ("time_adapt_safety", float),
        "--time-adapt-rho-abs": ("time_adapt_rho_abs", float),
        "--time-adapt-rho-rel": ("time_adapt_rho_rel", float),
        "--time-adapt-factor-min": ("time_adapt_factor_min", float),
        "--time-adapt-factor-max": ("time_adapt_factor_max", float),
        "--time-adapt-reject-factor": ("time_adapt_reject_factor", float),
        "--time-adapt-target-fraction": ("time_adapt_target_fraction", float),
        "--retry-recovery-steps": ("retry_recovery_steps", int),
        "--retry-recovery-growth": ("retry_recovery_growth", float),
        "--snes-rtol": ("snes_rtol", float), "--snes-atol": ("snes_atol", float),
        "--snes-stol": ("snes_stol", float), "--snes-max-it": ("snes_max_it", int),
        "--png-interval": ("png_interval", float),
        "--xdmf-interval": ("xdmf_interval", float),
        "--field-output-interval": ("field_output_interval_s", float),
        "--field-output-variables": ("field_output_variables", str),
        "--diagnostics-interval": ("diagnostics_interval", int),
        "--newton-profile-file": ("newton_profile_file", str),
        "--linear-rtol": ("linear_rtol", float), "--linear-atol": ("linear_atol", float),
        "--linear-max-it": ("linear_max_it", int), "--jacobian-lag": ("jacobian_lag", int),
        "--extrapolate-max-factor": ("extrapolate_max_factor", float),
        "--lifecycle-xi-threshold": ("lifecycle_xi_threshold", float),
        "--lifecycle-target-margin": ("lifecycle_target_margin", float),
    }
    for flag, (dest, kind) in options.items():
        choices = (-1.0, 1.0) if flag in ("--charge-current-sign", "--discharge-current-sign") else None
        parser.add_argument(flag, dest=dest, type=kind, choices=choices, default=None)
    parser.add_argument("--cathode-reaction-mode", choices=("interface",), default="interface")
    parser.add_argument("--linear-solver", choices=("lu", "mumps", "gmres_ilu", "fgmres_ilu", "gmres_hypre", "fgmres_hypre", "gmres_gamg", "fgmres_gamg", "gmres_bjacobi", "fgmres_bjacobi", "gmres_jacobi", "fgmres_jacobi"), default=None)
    parser.add_argument("--jacobian-mode", choices=("full", "block"), default=None)
    parser.add_argument("--progress-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--full-diagnostics", action="store_true")
    parser.add_argument("--no-top-xi-bc", action="store_true")
    parser.add_argument("--no-mechanics-periodic", action="store_true")
    parser.add_argument("--no-time-adapt-reject", action="store_true")
    parser.add_argument("--no-snes-monitor", action="store_true")
    parser.add_argument("--snes-monitor", action="store_true")
    parser.add_argument("--no-eta-clip", action="store_true")
    parser.add_argument("--no-newton-profile", action="store_true")
    parser.add_argument("--newton-profile", action="store_true")
    parser.add_argument("--reuse-jacobian-across-steps", action="store_true")
    parser.add_argument("--fast-residual-bc", action="store_true")
    parser.add_argument("--safe-residual-bc", action="store_true")
    parser.add_argument("--no-extrapolate-initial-guess", action="store_true")
    parser.add_argument("--no-lifecycle-stop", action="store_true")
    return parser


def main(*args, **kwargs):
    from .workflow import run

    if args or kwargs:
        return run(*args, **kwargs)
    parsed = vars(_parser().parse_args())
    msh_file = parsed.pop("msh")
    gb_file = parsed.pop("gb")
    parsed["quiet"] = parsed.pop("quiet")
    parsed.pop("verbose")
    parsed["light_diagnostics"] = not parsed.pop("full_diagnostics")
    parsed["enforce_top_xi_bc"] = not parsed.pop("no_top_xi_bc")
    parsed["mechanics_periodic_lr"] = not parsed.pop("no_mechanics_periodic")
    parsed["time_adapt_reject"] = not parsed.pop("no_time_adapt_reject")
    monitor = parsed.pop("snes_monitor")
    no_monitor = parsed.pop("no_snes_monitor")
    parsed["snes_monitor"] = False if no_monitor else (True if monitor else None)
    parsed["clip_eta_after_solve"] = not parsed.pop("no_eta_clip")
    profile = parsed.pop("newton_profile")
    no_profile = parsed.pop("no_newton_profile")
    parsed["newton_profile"] = False if no_profile else (True if profile else None)
    safe_residual = parsed.pop("safe_residual_bc")
    fast_residual = parsed.pop("fast_residual_bc")
    parsed["fast_residual_bc"] = False if safe_residual else (True if fast_residual else None)
    parsed["extrapolate_initial_guess"] = not parsed.pop("no_extrapolate_initial_guess")
    parsed["lifecycle_stop_enabled"] = not parsed.pop("no_lifecycle_stop")
    return run(msh_file=msh_file, gb_file=gb_file, **parsed)


if __name__ == "__main__":
    main()


__all__ = ["main"]
