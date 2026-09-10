"""Simulation parameters, region tags, and equilibrium-potential data."""

from dataclasses import dataclass
from dataclasses import replace
import sys

import numpy as np


def configure_run_params(cathode_reaction_mode, overrides, parse_field_variables):
    """Build Params from run() keyword overrides, preserving legacy coercions."""
    if cathode_reaction_mode != "interface":
        raise ValueError("Only cathode_reaction_mode='interface' is implemented in this cycle script.")
    p = Params(cathode_reaction_mode=cathode_reaction_mode)
    direct = {
        "charge_time": float, "discharge_time": float, "charge_cutoff_voltage": float,
        "discharge_cutoff_voltage": float, "dt": float, "dt_min": float,
        "li_side_potential": float, "D_li": float, "i0_ref_li": float,
        "i0_c_ref": float, "c_li_max": float, "temperature": float,
        "initial_cathode_overpotential": float, "phis_init": float,
        "E_li": float, "nu_li": float, "E_cathode": float, "nu_cathode": float,
        "anode_compressive_load": float, "cathode_swelling_scale": float,
        "gamma": float, "k_xi": float, "W_xi": float, "L_gb": float,
        "W_gb": float, "k_gb": float, "dt_max": float,
        "dt_max_charge": float, "dt_max_discharge": float, "dt_growth": float,
        "dt_shrink": float, "fixed_initial_dt": float, "time_adapt_tol_max": float,
        "time_adapt_tol_min": float, "time_adapt_safety": float,
        "time_adapt_rho_abs": float, "time_adapt_rho_rel": float,
        "time_adapt_factor_min": float, "time_adapt_factor_max": float,
        "snes_rtol": float, "snes_atol": float, "snes_stol": float,
        "mechanics_residual_scale": float, "linear_rtol": float, "linear_atol": float,
        "extrapolate_max_factor": float, "lifecycle_xi_threshold": float,
        "lifecycle_target_margin": float,
    }
    for key, cast in direct.items():
        if overrides.get(key) is not None:
            attr = {"temperature": "T"}.get(key, key)
            p = replace(p, **{attr: cast(overrides[key])})
    integer = {"cycle_count": (1, None), "fixed_initial_steps": (0, None),
               "diagnostics_interval": (1, None), "snes_max_it": (None, None),
               "linear_max_it": (None, None), "jacobian_lag": (1, None),
               "retry_recovery_steps": (0, None)}
    for key, (minimum, _) in integer.items():
        if overrides.get(key) is not None:
            value = int(overrides[key])
            if minimum is not None: value = max(minimum, value)
            p = replace(p, **{key: value})
    booleans = ("enforce_top_xi_bc", "mechanics_periodic_lr", "time_adapt_reject",
        "snes_monitor", "progress_only", "quiet", "clip_eta_after_solve", "newton_profile",
        "reuse_jacobian_across_steps", "fast_residual_bc", "light_diagnostics",
        "extrapolate_initial_guess", "lifecycle_stop_enabled")
    for key in booleans:
        if overrides.get(key) is not None: p = replace(p, **{key: bool(overrides[key])})
    if overrides.get("soc_min") is not None or overrides.get("soc_max") is not None:
        lo = float(overrides["soc_min"]) if overrides.get("soc_min") is not None else p.soc_min
        hi = float(overrides["soc_max"]) if overrides.get("soc_max") is not None else p.soc_max
        p = replace(p, soc_min=lo, soc_max=hi, soc_ref=0.5 * (lo + hi))
    if overrides.get("soc_init") is not None:
        p = replace(p, soc_init=float(np.clip(overrides["soc_init"], p.soc_min, p.soc_max)))
    signs = {}
    for key in ("charge_current_sign", "discharge_current_sign"):
        if overrides.get(key) is not None: signs[key] = 1.0 if float(overrides[key]) >= 0 else -1.0
    if signs: p = replace(p, **signs)
    couples = {}
    if overrides.get("E_se") is not None:
        couples.update(E_se=float(overrides["E_se"]), E_mix=float(overrides["E_se"]))
    if overrides.get("nu_se") is not None:
        couples.update(nu_se=float(overrides["nu_se"]), nu_mix=float(overrides["nu_se"]))
    if couples: p = replace(p, **couples)
    bounded = {
        "current_abs": ("current_abs", lambda x: abs(float(x))),
        "li_layer_thickness": ("li_layer_thickness", lambda x: max(0.0, float(x))),
        "time_adapt_reject_factor": ("time_adapt_reject_factor", lambda x: max(1.0, float(x))),
        "time_adapt_target_fraction": ("time_adapt_target_fraction", lambda x: min(max(float(x), 1.0e-12), 1.0)),
        "retry_recovery_growth": ("retry_recovery_growth", lambda x: max(1.0, float(x))),
        "png_interval": ("png_interval", lambda x: max(0.0, float(x))),
        "xdmf_interval": ("xdmf_interval", lambda x: max(0.0, float(x))),
        "field_output_interval_s": ("field_output_interval_s", lambda x: max(0.0, float(x))),
        "extrapolate_max_factor": ("extrapolate_max_factor", lambda x: max(0.0, float(x))),
    }
    for key, (attr, cast) in bounded.items():
        if overrides.get(key) is not None: p = replace(p, **{attr: cast(overrides[key])})
    if overrides.get("field_output_variables") is not None:
        p = replace(p, field_output_variables=tuple(parse_field_variables(overrides["field_output_variables"])))
    for key in ("linear_solver", "jacobian_mode"):
        if overrides.get(key) is not None:
            value = str(overrides[key]).lower()
            if key == "jacobian_mode" and value not in ("full", "block"):
                raise ValueError("jacobian_mode must be 'full' or 'block'.")
            p = replace(p, **{key: value})
    if overrides.get("newton_profile_file") is not None:
        p = replace(p, newton_profile_file=str(overrides["newton_profile_file"]))
    if p.progress_only: p = replace(p, snes_monitor=False, newton_profile=False)
    elif p.quiet: p = replace(p, snes_monitor=False)
    return p


# Gmsh physical regions.
OMEGA1 = 1
OMEGA11 = 101
OMEGA12 = 102
OMEGA1_REGIONS = (OMEGA1, OMEGA11, OMEGA12)
ETA_REGIONS = OMEGA1_REGIONS
OMEGA2 = 2
OMEGA3 = 3

GAMMA_A = 11
GAMMA_B = 12
GAMMA_C = 13
GAMMA_L = 14
GAMMA_S = 15

DEFAULT_MSH_FILE = "r2d_irregular_3_34.msh"
DEFAULT_GB_FILE = "r2d_irregular_3_34.npz"
GAMMA_S_OMEGA2_SIDE = "+"
GAMMA_S_OMEGA3_SIDE = "-"

#   eta_c = phis - phil - Eeq(theta) - V_Li+,m*sigma_h/F
E_EQ_TABLE = np.array(
    [
        [0.2228930343076256, 4.256817954840526],
        [0.23718770939025557, 4.2212385803217725],
        [0.2503742701253948, 4.198216215024365],
        [0.2635608308605341, 4.184354581468334],
        [0.2767473915956734, 4.175555457558853],
        [0.28993395233081265, 4.169287588472648],
        [0.3031205130659519, 4.163501863162304],
        [0.3163070738010912, 4.156631314356272],
        [0.3294936345362305, 4.145300935623516],
        [0.3426801952713697, 4.130836622347658],
        [0.355866756006509, 4.113841054248525],
        [0.3690533167416483, 4.09395262349422],
        [0.38223987747678756, 4.0746668724597415],
        [0.3954264382119268, 4.056104337089057],
        [0.4086129989470661, 4.037903409550268],
        [0.4217995596822054, 4.021944450569238],
        [0.43498612041734463, 4.007287279783036],
        [0.44817268115248393, 3.9945104697226936],
        [0.4613592418876232, 3.9798050845589046],
        [0.47454580262276247, 3.96497916345115],
        [0.4877323633579017, 3.9507559220632222],
        [0.500918924093041, 3.9348451774597786],
        [0.5141054848281803, 3.918090681248576],
        [0.5272920455633195, 3.901215649093408],
        [0.5404786062984588, 3.884581688826171],
        [0.5536651670335981, 3.8661396893994517],
        [0.5668517277687374, 3.850108408852042],
        [0.5800382885038766, 3.834920879912391],
        [0.593224849239016, 3.819612815028774],
        [0.6064114099741551, 3.806233325248605],
        [0.6195979707092945, 3.795023482459815],
        [0.6327845314444338, 3.7852600709986106],
        [0.6459710921795729, 3.77646094708913],
        [0.6591576529147123, 3.7660948559080984],
        [0.6723442136498515, 3.7569341241667216],
        [0.6855307743849908, 3.748376072145172],
        [0.6987173351201301, 3.7407823076753464],
        [0.7119038958552694, 3.7321037197098312],
        [0.7250904565904086, 3.724148347408109],
        [0.738277017325548, 3.7154697594425943],
        [0.7514635780606872, 3.7046215244857006],
        [0.7646501387958264, 3.69582240057622],
        [0.7778366995309657, 3.686541132890878],
        [0.791023260266105, 3.6770187933176044],
        [0.8042098210012443, 3.6672553818564],
        [0.8173963817363835, 3.6556839312357132],
        [0.8305829424715228, 3.643871408727096],
        [0.8437695032066621, 3.633505317546064],
        [0.8569560639418013, 3.6226570825891704],
        [0.8701426246769406, 3.6130142070719313],
        [0.8833291854120799, 3.603009723722796],
        [0.8965157461472192, 3.592643632541764],
        [0.9097023068823584, 3.585049868071939],
        [0.9228888676174977, 3.5790230708736646],
        [0.9351335311572699, 3.5724538619275457],
        [0.9505178520149323, 3.5661257248693574],
        [0.9624485498229155, 3.559857855783152],
        [0.9756351105580549, 3.5525051632012574],
        [0.9888216712931941, 3.541536392300398],
        [0.998240643246865, 3.488540755603573],
        [0.9994965061740211, 3.3640592684056174],
        [1.0, 3.0],
    ],
    dtype=np.float64,
)

@dataclass(frozen=True)
class Params:
    length_scale: float = 50.0e-6
    dt: float = 0.01
    charge_time: float = sys.float_info.max
    discharge_time: float = sys.float_info.max
    cycle_count: int = 7
    lifecycle_stop_enabled: bool = True
    lifecycle_xi_threshold: float = 0.5
    lifecycle_target_margin: float = 1.0e-9
    charge_cutoff_voltage: float = 4.25
    discharge_cutoff_voltage: float = 3.56
    dt_min: float = 1.0e-5
    dt_max: float = 50.0
    dt_max_charge: float | None = 25.0
    dt_max_discharge: float | None = 50.0
    dt_growth: float = 1.5
    dt_shrink: float = 0.4
    fixed_initial_steps: int = 1
    fixed_initial_dt: float | None = 0.01
    time_adapt_tol_max: float = 8.0
    time_adapt_tol_min: float = 1.0
    time_adapt_safety: float = 0.9
    time_adapt_rho_abs: float = 1.0e-10
    time_adapt_rho_rel: float = 1.0e-2
    time_adapt_factor_min: float = 0.5
    time_adapt_factor_max: float = 3.0
    time_adapt_reject: bool = True
    time_adapt_reject_factor: float = 3.0
    time_adapt_target_fraction: float = 0.4
    retry_recovery_steps: int = 2
    retry_recovery_growth: float = 1.2
    max_retries_per_step: int = 5
    preview_dir: str = "r2d"
    diagnostics_file: str = "r2d/diagnostics.csv"
    final_npz_file: str = "r2d/final_state.npz"
    png_interval: float = 500
    xdmf_interval: float = 0.0
    field_output_interval_s: float = 500.0
    field_output_variables: tuple[str, ...] = ("all",)
    diagnostics_interval: int = 1
    progress_only: bool = False
    quiet: bool = False
    newton_profile: bool = True
    newton_profile_file: str = "r2d/newton_profile.csv"
    linear_solver: str = "mumps"
    linear_rtol: float = 1.0e-8
    linear_atol: float = 1.0e-12
    linear_max_it: int = 500
    jacobian_lag: int = 8
    jacobian_mode: str = "block"
    reuse_jacobian_across_steps: bool = False
    fast_residual_bc: bool = True
    light_diagnostics: bool = True
    extrapolate_initial_guess: bool = True
    extrapolate_max_factor: float = 1.0
    li_layer_thickness: float = 5.0e-6
    xi_interface_width: float = 1.78e-6
    enforce_top_xi_bc: bool = True
    R: float = 8.314462618
    T: float = 298.0
    F: float = 96485.33212
    alpha: float = 0.5
    current_abs: float = 10.0
    charge_current_sign: float = 1.0
    discharge_current_sign: float = -1.0
    li_side_potential: float = -0.1
    sigmae: float = 1.0e7
    sigmal: float = 2.2
    sigmaS_stress_coeff: float = 9.0e-10
    sse: float = 1.85
    sigma_cathode: float = 0.17
    sigma_phi: float = 0.10
    E_li: float = 7.8e9
    nu_li: float = 0.381
    E_se: float = 20.0e9
    nu_se: float = 0.257
    E_mix: float = 20.0e9
    nu_mix: float = 0.257
    E_cathode: float = 177.5e9
    nu_cathode: float = 0.253
    beta_li: float = 0.333056
    omega_cathode: float = 3.5e-6
    cathode_swelling_scale: float = 1.0
    mechanics_residual_scale: float = 1.0e-10
    anode_compressive_load: float = -20.0e6
    mechanics_periodic_lr: bool = True
    gamma: float = 0.6
    interface_dx: float = 8.9e-7
    xi_anisotropy_delta: float = 0.082
    xi_anisotropy_omega: float = 4.0
    omega_li_m: float = 13.08e-6
    c_li_metal: float = 76.4e3
    k_xi: float = 3.2e-6
    W_xi: float = 4.0e6
    M_li: float = 7.0e-3
    rho_li: float = 535.0
    i0_ref_li: float = 7.81
    xi_mobility_scale: float = 1.0
    c_li_max: float = 50.06e3
    c_li_ref: float = 29.1e3
    c_li_init: float = 50.06e3
    D_li: float = 5.0e-13
    i0_c_ref: float = 7.4
    soc_min: float = 0.222
    soc_max: float = 0.942
    soc_ref: float = 0.5 * (soc_min + soc_max)
    soc_init: float = 0.942
    initial_cathode_overpotential: float = 0.0
    phis_init: float | None = None
    cathode_reaction_mode: str = "interface"
    L_gb: float = 1.5e-10
    W_gb: float = 1.0e7
    k_gb: float = 12.8e-7
    kappa_gb_cross: float = 0.0
    eta_mobility_scale: float = 1.0
    clip_eta_after_solve: bool = True
    b_clip: float = 0.2
    b_scale: float = 5.0
    rho: float = 0.2
    gb_window_smoothing: float = 0.02
    snes_rtol: float = 1.0e-10
    snes_atol: float = 1.0e-2
    snes_stol: float = 1.0e-10
    snes_max_it: int = 30
    snes_monitor: bool = False

    @property
    def omega_li(self) -> float:
        return self.omega_li_m

    @property
    def kappa0(self) -> float:
        return self.k_xi

    @property
    def W_b(self) -> float:
        return self.W_xi

    @property
    def L_eta(self) -> float:
        return self.i0_ref_li * self.omega_li / (6.0 * self.interface_dx * self.F)

    @property
    def L_sigma(self) -> float:
        return self.omega_li * self.L_eta / (self.R * self.T)

    @property
    def Vt(self) -> float:
        return self.R * self.T / self.F


__all__ = [
    "DEFAULT_GB_FILE", "DEFAULT_MSH_FILE", "E_EQ_TABLE", "ETA_REGIONS",
    "GAMMA_A", "GAMMA_B", "GAMMA_C", "GAMMA_L", "GAMMA_S",
    "GAMMA_S_OMEGA2_SIDE", "GAMMA_S_OMEGA3_SIDE", "OMEGA1",
    "OMEGA11", "OMEGA12", "OMEGA1_REGIONS", "OMEGA2", "OMEGA3",
    "Params", "configure_run_params",
]
