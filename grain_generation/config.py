"""Configuration and physical region tags for grain generation."""

from dataclasses import dataclass


OMEGA1 = 1
OMEGA11 = 101
OMEGA12 = 102
OMEGA2 = 2
OMEGA3 = 3
DEFAULT_MSH_FILE = "r2d_irregular_3_34.msh"


@dataclass(frozen=True)
class GrainParams:
    n_grains: int = 10
    length_scale: float = 50.0e-6
    L_phi: float = 1.5e-8
    W_phi: float = 1.025e7
    k_phi: float = 12.8e-7
    time_scale: float = 1.0
    dt: float = 10.0
    t_end: float = 3000.0
    seed: int = 32
    initializer: str = "partition_random"
    grain_excluded_top_thickness: float = 5.0e-6
    xi_interface_width: float = 1.78e-6
    interface_width: float = 0.015
    boundary_wiggle: float = 0.025
    boundary_drift: float = 0.035
    comsol_random_grid: int = 128
    partition_seed_count: int = 55
    partition_transition_width: float = 0.018
    inactive_penalty: float = 1.0e-8
    clip_eta_after_solve: bool = True
    b_clip: float = 0.2
    b_scale: float = 5.0
    rho: float = 0.2
    gb_window_smoothing: float = 0.02
    preview_interval: float = 500.0

__all__ = [
    "DEFAULT_MSH_FILE", "OMEGA1", "OMEGA11", "OMEGA12", "OMEGA2", "OMEGA3",
    "GrainParams",
]
