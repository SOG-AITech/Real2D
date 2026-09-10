#保持原来的 monolithic UFL 方程、残差、Jacobian 不变，只在 Newton 线性求解时把 
# xi/eta_i 在非 Ω1 区域那些本来就被固定为 0 的自由度剔除掉。
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import csv
import argparse
import builtins
import math
import os
import tempfile
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import ufl
from basix.ufl import element, mixed_element
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh
try:
    from dolfinx.plot import vtk_mesh
except ImportError:
    vtk_mesh = None
from dolfinx.fem.petsc import (
    NonlinearProblem,
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_matrix,
    create_vector,
    set_bc,
)
from dolfinx.io import gmsh, XDMFFile

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
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        text = sep.join(str(arg) for arg in args) + end
        os.write(_SAVED_STDOUT_FD, text.encode(errors="replace"))
    else:
        _ORIGINAL_PRINT(*args, **kwargs)


VERBOSE_STAGE_LOGS = os.environ.get("R2D_STAGE_LOGS", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
IMPORTANT_STAGE_LOG_PATTERNS = (
    "Newton",
    "problem.solve",
)


def mpi_stage_print(comm, message):
    if not VERBOSE_STAGE_LOGS and not any(
        pattern in message for pattern in IMPORTANT_STAGE_LOG_PATTERNS
    ):
        return
    print(f"[rank {comm.rank}/{comm.size}] {message}", flush=True)


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.collections import PolyCollection

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    mtri = None
    PolyCollection = None
    HAS_MATPLOTLIB = False


# Gmsh 物理区域编号，与 battery_prlsi25_regions_coarse.geo/.msh 保持一致。
OMEGA1 = 1  # Li-SE/金属锂与固态电解质复合区域：求解 xi、phil、eta_i 和力学位移。
OMEGA11 = 101  # 初始化专用：顶部 Li 层，主求解时并回 Omega1。
OMEGA12 = 102  # 初始化专用：下方晶粒区域，主求解时并回 Omega1。
# In the production solve Omega11 and Omega12 are one continuous Omega1
# domain. Their shared facet is an internal mesh interface, not a physical
# boundary, and must not be treated as inactive/active trace.
OMEGA1_REGIONS = (OMEGA1, OMEGA11, OMEGA12)
ETA_REGIONS = OMEGA1_REGIONS
OMEGA2 = 2  # SE-carbon 区域：求解 phil、phis 和力学位移。
OMEGA3 = 3  # NMC 正极颗粒区域：求解归一化锂浓度 c 和力学位移；不在这里求解 phis。

GAMMA_A = 11  # 顶部锂侧边界；当前用于探针和初始锂层位置，不作为 phil 的 Dirichlet 边界。
GAMMA_B = 12  # 外侧绝缘边界标签。
GAMMA_C = 13  # 底部集流体边界；恒流电流从这里进入 phis 方程。
GAMMA_L = 14  # 侧边界标签。
GAMMA_S = 15  # Ω2-Ω3 正极颗粒界面；Butler-Volmer 反应通量 i_s 在这里耦合 phil、phis、c。

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
    length_scale: float = 50.0e-6  # [m]
    dt: float = 0.01  # [s]
    charge_time: float = sys.float_info.max
    discharge_time: float = sys.float_info.max
    cycle_count: int = 7
    lifecycle_stop_enabled: bool = True
    lifecycle_xi_threshold: float = 0.5
    lifecycle_target_margin: float = 1.0e-9  # Tiny normalized y/L margin below the mesh Omega1/Omega2 interface.
    charge_cutoff_voltage: float = 4.25
    discharge_cutoff_voltage: float = 3.56
    dt_min: float = 1.0e-5  # [s]
    dt_max: float = 50.0 # [s], production default keeps the validated fixed step.
    dt_max_charge: float | None = 25.0  # [s], charge maximum step cap.
    dt_max_discharge: float | None = 50.0  # [s], discharge maximum step cap.
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
    png_interval: float = 500 # [s], 0 disables time-interval PNG snapshots.
    xdmf_interval: float = 0.0 # [s], 0 disables time-interval XDMF snapshots.
    field_output_interval_s: float = 500.0  # [s], 0 disables software-plot NPZ snapshots.
    field_output_variables: tuple[str, ...] = ("all",)
    diagnostics_interval: int = 1  # accepted steps; 1 keeps one CSV row per accepted step.
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

    #
    li_layer_thickness: float = 5.0e-6  # [m]
    xi_interface_width: float = 1.78e-6  # [m], COMSOL step1 smoothing width 2*dx with dx=8.9e-7 m.
    enforce_top_xi_bc: bool = True

    R: float = 8.314462618  # [J/mol/K]
    T: float = 298.0  # [K], COMSOL text.m room-temperature setting.
    F: float = 96485.33212  # [C/mol]
    alpha: float = 0.5

    current_abs: float = 10.0  # [A/m^2]
    charge_current_sign: float = 1.0
    discharge_current_sign: float = -1.0
    li_side_potential: float = -0.1  # [V], COMSOL text.m liion/init1.phil; not a Dirichlet BC.

    sigmae: float = 1.0e7
    sigmal: float = 2.2
    sigmaS_stress_coeff: float = 9.0e-10  # [S/(m Pa)], COMSOL 9e-9 mS/(cm Pa)
    sse: float = 1.85
    sigma_cathode: float = 0.17
    sigma_phi: float = 0.10

    #
    E_li: float = 7.8e9  # [Pa]
    nu_li: float = 0.381
    E_se: float = 20.0e9  # [Pa]
    nu_se: float = 0.257
    E_mix: float = 20.0e9  # [Pa]
    nu_mix: float = 0.257
    E_cathode: float = 177.5e9  # [Pa]
    nu_cathode: float = 0.253
    #
    #   eps_Li = beta_li * h(xi) * I.
    #   13.08e-6 [m^3/mol] * 76.4e3 [mol/m^3] / 3 ~= 0.333.
    beta_li: float = 0.333056
    omega_cathode: float = 3.5e-6  # [m^3/mol], V_Li+,m = 3.5 cm^3/mol.
    cathode_swelling_scale: float = 1.0
    mechanics_residual_scale: float = 1.0e-10
    anode_compressive_load: float = -20.0e6  # [Pa], y-traction on Gamma_a.
    mechanics_periodic_lr: bool = True

    gamma: float = 0.6  # [J/m^2]
    interface_dx: float = 8.9e-7  # [m]
    xi_anisotropy_delta: float = 0.082
    xi_anisotropy_omega: float = 4.0
    omega_li_m: float = 13.08e-6
    c_li_metal: float = 76.4e3
    k_xi: float = 3.2e-6
    W_xi: float = 4.0e6
    M_li: float = 7.0e-3
    rho_li: float = 535.0
    i0_ref_li: float = 7.81  # [A/m^2]
    xi_mobility_scale: float = 1.0

    #
    #   n_cathode [mol/m^2] = int_Omega3 c_li_max * c * dV.
    #
    c_li_max: float = 50.06e3  # [mol/m^3]
    c_li_ref: float = 29.1e3  # [mol/m^3]
    c_li_init: float = 50.06e3  # [mol/m^3]
    D_li: float = 5.0e-13  # [m^2/s]
    i0_c_ref: float = 7.4  # [A/m^2]
    soc_min: float = 0.222
    soc_max: float = 0.942
    # COMSOL text.m uses cmo_ref = c_li_max*(socmax+socmin)/2 for cathode
    # hygroscopic swelling.
    soc_ref: float = 0.5 * (soc_min + soc_max)
    soc_init: float = 0.942
    # Start the transient from near-equilibrium cathode kinetics.  COMSOL runs
    # a CurrentDistributionInitialization step before the transient; without
    # that stationary initialization, a hard-coded 0.1 V cathode overpotential
    # drives a Gamma_s BV current hundreds of times larger than the applied
    # current and makes the first Newton step very fragile.
    initial_cathode_overpotential: float = 0.0
    # Initial guesses for the electric potentials. They seed the first
    # Newton solve only; subsequent eta values are always recomputed from
    # the solved fields.
    phis_init: float | None = None  # [V]
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
    snes_stol: float = 1.0e-10  # SNES step-size tolerance; PETSc default is commonly 1e-8.
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


def resolve_here(filename: str | Path) -> Path:
    path = Path(filename)
    if path.is_absolute() or path.exists():
        return path
    return Path(__file__).resolve().parent / path


# 读取 Gmsh 网格以及体/边界物理标签。
def read_mesh(msh_file: str | Path, p: Params):
    partitioner = mesh.create_cell_partitioner(
        mesh.GhostMode.shared_facet,
        max_facet_to_cell_links=2,
    )
    mpi_stage_print(MPI.COMM_WORLD, "read_mesh: before gmsh.read_from_msh")
    mesh_data = gmsh.read_from_msh(
        resolve_here(msh_file), MPI.COMM_WORLD, rank=0, gdim=2, partitioner=partitioner
    )
    mpi_stage_print(MPI.COMM_WORLD, "read_mesh: after gmsh.read_from_msh")
    msh = mesh_data.mesh
    msh.geometry.x[:, : msh.geometry.dim] /= p.length_scale
    mpi_stage_print(msh.comm, "read_mesh: before connectivity tdim->0")
    msh.topology.create_connectivity(msh.topology.dim, 0)
    mpi_stage_print(msh.comm, "read_mesh: before connectivity facet->cell")
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    mpi_stage_print(msh.comm, "read_mesh: before connectivity cell->facet")
    msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim - 1)
    mpi_stage_print(msh.comm, "read_mesh: done")
    return msh, mesh_data.cell_tags, mesh_data.facet_tags


def h(z):
    return z**3 * (6.0 * z**2 - 15.0 * z + 10.0)


def hp(z):
    return 30.0 * z**2 * (1.0 - z) ** 2


def clip01_ufl(z):
    return ufl.max_value(0.0, ufl.min_value(1.0, z))


def xi_anisotropic_gradient_term(xi, v_xi, p: Params):
    grad_xi = ufl.grad(xi)
    xi_x = grad_xi[0]
    xi_y = grad_xi[1]
    grad_norm = ufl.sqrt(xi_x * xi_x + xi_y * xi_y)
    xi_x_for_angle = ufl.conditional(ufl.gt(grad_norm, 1.0e-14), xi_x, 1.0)
    xi_y_for_angle = ufl.conditional(ufl.gt(grad_norm, 1.0e-14), xi_y, 0.0)
    theta = ufl.atan2(xi_y_for_angle, xi_x_for_angle)
    K = 1.0 + p.xi_anisotropy_delta * ufl.cos(p.xi_anisotropy_omega * theta)
    Ktheta = -p.xi_anisotropy_omega * p.xi_anisotropy_delta * ufl.sin(
        p.xi_anisotropy_omega * theta
    )
    grad_v = ufl.grad(v_xi)
    flux_x = K * K * xi_x - K * Ktheta * xi_y
    flux_y = K * K * xi_y + K * Ktheta * xi_x
    return (p.kappa0 / p.length_scale**2) * (flux_x * grad_v[0] + flux_y * grad_v[1])


def initialize_xi_profile(X, y_top, p: Params):
    """Initialize xi from 1 to 0 across the configured transition width."""
    depth = y_top - X[1]
    location = p.li_layer_thickness / p.length_scale
    width = p.xi_interface_width / p.length_scale
    if width <= 0.0:
        return np.where(depth < location, 1.0, 0.0).astype(PETSc.ScalarType)
    t = np.clip((depth - (location - 0.5 * width)) / width, 0.0, 1.0)
    smooth = t**3 * (6.0 * t**2 - 15.0 * t + 10.0)
    return (1.0 - smooth).astype(PETSc.ScalarType)


def strain(u):
    return ufl.sym(ufl.grad(u))


def plane_stress_tensor(E, nu, eps_tensor, eigenstrain_tensor):
    """COMSOL 2D Solid Mechanics thickness formulation: plane stress."""
    eps_eff = eps_tensor - eigenstrain_tensor
    c = E / (1.0 - nu**2)
    sigma_xx = c * (eps_eff[0, 0] + nu * eps_eff[1, 1])
    sigma_yy = c * (eps_eff[1, 1] + nu * eps_eff[0, 0])
    sigma_xy = E / (1.0 + nu) * eps_eff[0, 1]
    return ufl.as_tensor(((sigma_xx, sigma_xy), (sigma_xy, sigma_yy)))


def plane_stress_hydrostatic_stress(E, nu, eps_tensor, eigenstrain_tensor):
    sigma = plane_stress_tensor(E, nu, eps_tensor, eigenstrain_tensor)
    return (sigma[0, 0] + sigma[1, 1]) / 3.0


def smooth_li_profile(y, y_interface, width):
    return 0.5 * (1.0 + np.tanh((y - y_interface) / width))


def sharp_li_profile(y, y_interface):
    return np.where(y >= y_interface, 1.0, 0.0)


def piecewise_linear_ufl(x, table: np.ndarray):
    x0, y0 = table[0]
    x1, y1 = table[1]
    result = y0 + (y1 - y0) / (x1 - x0) * (x - x0)
    for i in range(len(table) - 1, 0, -1):
        xa, ya = table[i - 1]
        xb, yb = table[i]
        value = ya + (yb - ya) / (xb - xa) * (x - xa)
        result = ufl.conditional(ufl.le(x, xb), value, result)
    return result


def pchip_slopes(table: np.ndarray) -> np.ndarray:
    x = table[:, 0]
    y = table[:, 1]
    h_seg = np.diff(x)
    delta = np.diff(y) / h_seg
    slopes = np.zeros_like(y)

    for i in range(1, len(y) - 1):
        if delta[i - 1] == 0.0 or delta[i] == 0.0 or np.sign(delta[i - 1]) != np.sign(delta[i]):
            slopes[i] = 0.0
        else:
            w1 = 2.0 * h_seg[i] + h_seg[i - 1]
            w2 = h_seg[i] + 2.0 * h_seg[i - 1]
            slopes[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    slopes[0] = ((2.0 * h_seg[0] + h_seg[1]) * delta[0] - h_seg[0] * delta[1]) / (
        h_seg[0] + h_seg[1]
    )
    if np.sign(slopes[0]) != np.sign(delta[0]):
        slopes[0] = 0.0
    elif np.sign(delta[0]) != np.sign(delta[1]) and abs(slopes[0]) > abs(3.0 * delta[0]):
        slopes[0] = 3.0 * delta[0]

    slopes[-1] = ((2.0 * h_seg[-1] + h_seg[-2]) * delta[-1] - h_seg[-1] * delta[-2]) / (
        h_seg[-1] + h_seg[-2]
    )
    if np.sign(slopes[-1]) != np.sign(delta[-1]):
        slopes[-1] = 0.0
    elif np.sign(delta[-1]) != np.sign(delta[-2]) and abs(slopes[-1]) > abs(3.0 * delta[-1]):
        slopes[-1] = 3.0 * delta[-1]

    return slopes


E_EQ_SLOPES = pchip_slopes(E_EQ_TABLE)


def smooth_table_ufl(x, table: np.ndarray, slopes: np.ndarray):
    xa, ya = table[0]
    xb, yb = table[1]
    ma = slopes[0]
    mb = slopes[1]
    h_seg = xb - xa
    t = (x - xa) / h_seg
    result = (
        (2.0 * t**3 - 3.0 * t**2 + 1.0) * ya
        + (t**3 - 2.0 * t**2 + t) * h_seg * ma
        + (-2.0 * t**3 + 3.0 * t**2) * yb
        + (t**3 - t**2) * h_seg * mb
    )
    for i in range(len(table) - 1, 0, -1):
        xa, ya = table[i - 1]
        xb, yb = table[i]
        ma = slopes[i - 1]
        mb = slopes[i]
        h_seg = xb - xa
        t = (x - xa) / h_seg
        value = (
            (2.0 * t**3 - 3.0 * t**2 + 1.0) * ya
            + (t**3 - 2.0 * t**2 + t) * h_seg * ma
            + (-2.0 * t**3 + 3.0 * t**2) * yb
            + (t**3 - t**2) * h_seg * mb
        )
        result = ufl.conditional(ufl.le(x, xb), value, result)
    return result


def smooth_table_numpy(theta: float | np.ndarray, table: np.ndarray, slopes: np.ndarray):
    x = table[:, 0]
    y = table[:, 1]
    theta_arr = np.asarray(theta, dtype=np.float64)
    theta_clamped = np.clip(theta_arr, x[0], x[-1])
    idx = np.searchsorted(x, theta_clamped, side="right") - 1
    idx = np.clip(idx, 0, len(x) - 2)
    xa = x[idx]
    xb = x[idx + 1]
    ya = y[idx]
    yb = y[idx + 1]
    ma = slopes[idx]
    mb = slopes[idx + 1]
    h_seg = xb - xa
    t = (theta_clamped - xa) / h_seg
    value = (
        (2.0 * t**3 - 3.0 * t**2 + 1.0) * ya
        + (t**3 - 2.0 * t**2 + t) * h_seg * ma
        + (-2.0 * t**3 + 3.0 * t**2) * yb
        + (t**3 - t**2) * h_seg * mb
    )
    return float(value) if np.isscalar(theta) else value


def eeq_from_soc(theta):
    return smooth_table_ufl(theta, E_EQ_TABLE, E_EQ_SLOPES)


def eeq_from_soc_numpy(theta: float) -> float:
    return float(smooth_table_numpy(theta, E_EQ_TABLE, E_EQ_SLOPES))


def nearest_old_values(old_coords, old_values, new_coords, default_values, tol=1.0e-10):
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(old_coords)
        dist, idx = tree.query(new_coords, k=1)
        out = old_values[..., idx]
        far = dist > tol
    except ModuleNotFoundError:
        idx = np.empty(new_coords.shape[0], dtype=np.int64)
        dist = np.empty(new_coords.shape[0], dtype=np.float64)
        chunk = 512
        for start in range(0, new_coords.shape[0], chunk):
            stop = min(start + chunk, new_coords.shape[0])
            diff = new_coords[start:stop, None, :] - old_coords[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            local_idx = np.argmin(d2, axis=1)
            idx[start:stop] = local_idx
            dist[start:stop] = np.sqrt(d2[np.arange(stop - start), local_idx])
        out = old_values[..., idx]
        far = dist > tol
    if np.any(far):
        out = np.array(out, copy=True)
        out[..., far] = default_values[..., None]
    return out


def load_frozen_grain_fields(V, gb_file: str | Path):
    data = np.load(resolve_here(gb_file), allow_pickle=True)
    eta_values = data["eta"]
    b_values = data["B"]
    ce_values = None
    if "extra" in data and "names" in data:
        names = [str(name) for name in data["names"]]
        if "ce" in names:
            ce_index = names.index("ce") - int(eta_values.shape[0]) - 1
            if ce_index >= 0 and ce_index < data["extra"].shape[0]:
                ce_values = np.asarray(data["extra"][ce_index], dtype=np.float64)

    ndofs = V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts
    if b_values.shape[0] != ndofs:
        if "dof_coordinates" not in data:
            raise ValueError(
                f"GB npz has {b_values.shape[0]} dofs, but scalar space has {ndofs}, "
                "and the npz has no coordinates for remapping."
            )
        old_coords = np.asarray(data["dof_coordinates"], dtype=np.float64)
        new_coords = V.tabulate_dof_coordinates()[:, :2]
        old_ymax = float(np.max(old_coords[:, 1]))
        new_top = new_coords[:, 1] > old_ymax + 1.0e-8
        default_eta = np.zeros(eta_values.shape[0], dtype=np.float64)
        default_eta[0] = 1.0
        default_B = np.array(0.0, dtype=np.float64)
        default_ce = np.array(1.0, dtype=np.float64)
        eta_values = nearest_old_values(old_coords, eta_values, new_coords, default_eta)
        b_values = nearest_old_values(old_coords, b_values, new_coords, default_B)
        if ce_values is not None:
            ce_values = nearest_old_values(old_coords, ce_values, new_coords, default_ce)
        eta_values[:, new_top] = default_eta[:, None]
        b_values[new_top] = 0.0
        if ce_values is not None:
            ce_values[new_top] = 1.0
        if V.mesh.comm.rank == 0:
            print(
                "Remapped frozen GB fields from "
                f"{old_coords.shape[0]} old dofs to {ndofs} new dofs; "
                f"{int(np.count_nonzero(new_top))} top-cap dofs set to B=0."
            )

    etas = []
    for i in range(eta_values.shape[0]):
        eta_i = fem.Function(V, name=f"eta{i + 1}")
        eta_i.x.array[:] = eta_values[i].astype(eta_i.x.array.dtype)
        eta_i.x.scatter_forward()
        etas.append(eta_i)

    B = fem.Function(V, name="B")
    B.x.array[:] = b_values.astype(B.x.array.dtype)
    B.x.scatter_forward()
    return etas, B


def load_grain_arrays(V, gb_file: str | Path):
    """Load eta_i arrays, remapping old GB files onto a slightly changed mesh."""
    etas, B = load_frozen_grain_fields(V, gb_file)
    eta_values = np.vstack([eta_i.x.array.real.copy() for eta_i in etas])
    return eta_values, B.x.array.real.copy()


def grain_boundary_indicator_expr(etas, p: Params, mask=1.0):
    gb_overlap = sum(
        etas[i] * etas[j]
        for i in range(len(etas))
        for j in range(i + 1, len(etas))
    )
    return mask * p.b_scale * ufl.min_value(p.b_clip, ufl.max_value(0.0, gb_overlap))


def box_indicator_ufl(value, half_width, smoothing=0.0):
    if smoothing <= 0.0:
        return ufl.conditional(ufl.lt(abs(value), half_width), 1.0, 0.0)
    return 0.5 * (
        ufl.tanh((value + half_width) / smoothing)
        - ufl.tanh((value - half_width) / smoothing)
    )


def grain_boundary_window_expr(etas, p: Params, eta_clips=None):
    if eta_clips is None:
        eta_clips = etas
    return sum(
        (1.0 - eta_clip_i)
        * box_indicator_ufl(eta_i - 0.5, p.rho, p.gb_window_smoothing)
        for eta_i, eta_clip_i in zip(etas, eta_clips)
    )


def collapse_mixed_subspace(ME, component: int):
    """Return a collapsed mixed subspace and a NumPy parent-DOF map.

    DOLFINx 0.11 may expose the map from ``collapse()`` as a Python list,
    whereas earlier releases returned a NumPy array.
    """
    V_sub, submap = ME.sub(component).collapse()
    # DOLFINx 0.11 may wrap the map in a one-element list.  Flattening keeps
    # the one-dimensional parent-DOF map expected by NumPy indexing.
    return V_sub, np.asarray(submap, dtype=np.int32).reshape(-1)


def copy_component_from_array(w, ME, component: int, values):
    _, submap = collapse_mixed_subspace(ME, component)
    if values.shape[0] != submap.shape[0]:
        raise ValueError(
            f"Component {component} has {submap.shape[0]} dofs, but eta input has {values.shape[0]}."
        )
    w.x.array[submap] = values.astype(w.x.array.dtype)


def assign_component_from_expression(w, ME, component: int, expr):
    V_sub, submap = collapse_mixed_subspace(ME, component)
    fun = fem.Function(V_sub)
    fun.interpolate(expr)
    fun.x.scatter_forward()
    w.x.array[submap] = fun.x.array


def assign_component_on_regions(w, ME, component: int, V_scalar, markers, region_ids, value):
    """Assign one mixed component on cells selected by physical-region tags."""
    _, submap = collapse_mixed_subspace(ME, component)
    dofs = region_dofs_from_markers(V_scalar, markers, region_ids)
    if dofs.size > 0:
        w.x.array[submap[dofs]] = PETSc.ScalarType(value)
        w.x.scatter_forward()


def clip_mixed_components(w, ME, components, lower=0.0, upper=1.0):
    """Clip selected mixed-function components in-place."""
    for component in components:
        _, submap = collapse_mixed_subspace(ME, component)
        w.x.array[submap] = np.clip(w.x.array.real[submap], lower, upper).astype(
            w.x.array.dtype
        )
    w.x.scatter_forward()


def update_scalar_outputs(
    w,
    scalar_outputs,
    p: Params | None = None,
    indices=None,
    component_maps=None,
):
    update_indices = range(len(scalar_outputs)) if indices is None else indices
    for i in update_indices:
        out = scalar_outputs[i]
        if component_maps is None:
            values = w.sub(i).collapse().x.array
        else:
            values = w.x.array[component_maps[i]]
        out.x.array[:] = values.astype(out.x.array.dtype, copy=False)
        out.x.scatter_forward()


def update_interpolated(out, expr):
    points = out.function_space.element.interpolation_points
    if callable(points):
        points = points()
    out.interpolate(fem.Expression(expr, points))
    out.x.scatter_forward()


def dg0_cell_dofs(V0):
    msh = V0.mesh
    tdim = msh.topology.dim
    num_cells = msh.topology.index_map(tdim).size_local
    dofmap_list = V0.dofmap.list
    dofmap_array = (
        dofmap_list.array if hasattr(dofmap_list, "array") else np.asarray(dofmap_list)
    )
    return dofmap_array.reshape((-1, 1))[:num_cells, 0].astype(np.int32)


def mask_dg0_to_regions(fun, cell_markers, valid_regions, fill_value=np.nan):
    if cell_markers is None:
        return
    if isinstance(valid_regions, (int, np.integer)):
        valid_regions = (int(valid_regions),)
    regions = tuple(int(v) for v in valid_regions)
    cell_dofs = dg0_cell_dofs(fun.function_space)
    local_markers = np.asarray(cell_markers[: cell_dofs.size], dtype=np.int32)
    keep = np.isin(local_markers, regions)
    fun.x.array[cell_dofs[~keep]] = PETSc.ScalarType(fill_value)
    fun.x.scatter_forward()


def copy_dg0_on_regions(target, source, cell_markers, valid_regions):
    if cell_markers is None:
        return
    if isinstance(valid_regions, (int, np.integer)):
        valid_regions = (int(valid_regions),)
    regions = tuple(int(v) for v in valid_regions)
    cell_dofs = dg0_cell_dofs(target.function_space)
    local_markers = np.asarray(cell_markers[: cell_dofs.size], dtype=np.int32)
    keep = np.isin(local_markers, regions)
    target.x.array[cell_dofs[keep]] = source.x.array[cell_dofs[keep]]
    target.x.scatter_forward()


def recover_dg0_to_p1(target, source_dg0, cell_markers, valid_regions):
    if cell_markers is None:
        target.x.array[:] = PETSc.ScalarType(np.nan)
        target.x.scatter_forward()
        return
    if isinstance(valid_regions, (int, np.integer)):
        valid_regions = (int(valid_regions),)
    regions = tuple(int(v) for v in valid_regions)
    cells, _points, markers = cells_and_points(target.function_space, cell_markers)
    source_dofs = dg0_cell_dofs(source_dg0.function_space)
    keep = np.isin(markers[: source_dofs.size], regions)
    sums = np.zeros(target.x.array.shape[0], dtype=np.float64)
    counts = np.zeros(target.x.array.shape[0], dtype=np.float64)
    source_values = source_dg0.x.array.real
    for cell_idx in np.flatnonzero(keep):
        value = source_values[source_dofs[cell_idx]]
        if not np.isfinite(value):
            continue
        cell_dofs = cells[cell_idx]
        sums[cell_dofs] += value
        counts[cell_dofs] += 1.0
    values = np.full(target.x.array.shape[0], np.nan, dtype=np.float64)
    valid = counts > 0.0
    values[valid] = sums[valid] / counts[valid]
    target.x.array[:] = values.astype(target.x.array.dtype)
    target.x.scatter_forward()


def mask_function_to_regions(fun, V, cell_markers, valid_regions, fill_value=np.nan):
    """Set dofs outside valid cell regions to fill_value for visualization output."""
    valid = restricted_dofs(V, cell_markers, valid_regions)
    keep = np.zeros(fun.x.array.shape[0], dtype=bool)
    keep[valid] = True
    fun.x.array[~keep] = PETSc.ScalarType(fill_value)
    fun.x.scatter_forward()


def cell_marker_array(msh, cell_tags):
    tdim = msh.topology.dim
    index_map = msh.topology.index_map(tdim)
    num_owned = index_map.size_local
    num_cells = num_owned + index_map.num_ghosts

    # cell_tags normally stores owned entities.  The inactive-dof logic below
    # deliberately inspects owned+ghost cells, so ghost tags must be scattered
    # first; otherwise partition-boundary dofs can be misclassified as
    # inactive-only on some MPI sizes.
    V0 = fem.functionspace(msh, ("DG", 0))
    tag_fun = fem.Function(V0, name="cell_marker_array_tags")
    tag_fun.x.array[:] = -1.0

    dofmap_list = V0.dofmap.list
    dofmap_array = (
        dofmap_list.array if hasattr(dofmap_list, "array") else np.asarray(dofmap_list)
    )
    cell_dofs = dofmap_array.reshape((num_cells, -1))[:, 0].astype(np.int32)
    for cell, tag in zip(
        np.asarray(cell_tags.indices, dtype=np.int32),
        np.asarray(cell_tags.values, dtype=np.int32),
    ):
        if 0 <= int(cell) < num_owned:
            tag_fun.x.array[cell_dofs[int(cell)]] = float(tag)
    tag_fun.x.scatter_forward()

    markers = np.rint(tag_fun.x.array[cell_dofs]).astype(np.int32)
    return markers


def cellwise_output_function(msh, cell_tags, values, name):
    """Create a DG0 cell field for ParaView Threshold/Color By operations."""
    V0 = fem.functionspace(msh, ("DG", 0))
    out = fem.Function(V0, name=name)
    tdim = msh.topology.dim
    index_map = msh.topology.index_map(tdim)
    num_owned = index_map.size_local
    num_cells = num_owned + index_map.num_ghosts
    out.x.array[:] = -1.0
    dofmap_list = V0.dofmap.list
    dofmap_array = (
        dofmap_list.array if hasattr(dofmap_list, "array") else np.asarray(dofmap_list)
    )
    cell_dofs = dofmap_array.reshape((num_cells, -1))[:, 0].astype(np.int32)
    tag_to_value = {int(k): float(v) for k, v in values.items()}
    for cell, tag in zip(np.asarray(cell_tags.indices, dtype=np.int32), np.asarray(cell_tags.values, dtype=np.int32)):
        if 0 <= int(cell) < num_owned:
            out.x.array[cell_dofs[int(cell)]] = tag_to_value.get(int(tag), -1.0)
    out.x.scatter_forward()
    return out


def cell_tag_dg0_function(msh, cell_tags, name="cell_region_tag"):
    """DG0 cell field carrying the raw physical cell tag for UFL side selection."""
    raw_tags = sorted({int(v) for v in np.asarray(cell_tags.values, dtype=np.int32)})
    return cellwise_output_function(msh, cell_tags, {tag: tag for tag in raw_tags}, name)


def xdmf_region_fields(msh, cell_tags):
    """Cell-wise helper fields that make multi-layer ParaView plots easier."""
    raw_tags = sorted({int(v) for v in np.asarray(cell_tags.values, dtype=np.int32)})
    raw_region = cellwise_output_function(msh, cell_tags, {tag: tag for tag in raw_tags}, "cell_region_raw")
    plot_region = cellwise_output_function(
        msh,
        cell_tags,
        {
            OMEGA1: 1.0,
            OMEGA11: 1.0,
            OMEGA12: 1.0,
            OMEGA2: 2.0,
            OMEGA3: 3.0,
        },
        "plot_region",
    )
    omega1_mask = cellwise_output_function(
        msh, cell_tags, {OMEGA1: 1.0, OMEGA11: 1.0, OMEGA12: 1.0}, "omega1_mask"
    )
    omega2_mask = cellwise_output_function(msh, cell_tags, {OMEGA2: 1.0}, "omega2_mask")
    omega3_mask = cellwise_output_function(msh, cell_tags, {OMEGA3: 1.0}, "omega3_mask")
    return (raw_region, plot_region, omega1_mask, omega2_mask, omega3_mask)


def point_region_mask_function(V, cell_markers, valid_regions, name):
    """Create a P1 point mask so ParaView Threshold keeps point-field arrays."""
    out = fem.Function(V, name=name)
    out.x.array[:] = PETSc.ScalarType(-1.0)
    dofs = region_dofs_from_markers(V, cell_markers, valid_regions)
    if dofs.size > 0:
        out.x.array[dofs] = PETSc.ScalarType(1.0)
    out.x.scatter_forward()
    return out


def xdmf_point_region_fields(V, cell_markers):
    """Point-wise masks for Threshold while retaining point arrays such as stress."""
    omega1 = point_region_mask_function(V, cell_markers, OMEGA1_REGIONS, "omega1_point_mask")
    omega2 = point_region_mask_function(V, cell_markers, (OMEGA2,), "omega2_point_mask")
    omega3 = point_region_mask_function(V, cell_markers, (OMEGA3,), "omega3_point_mask")
    return (omega1, omega2, omega3)


def xdmf_plot_field(source, name, cell_markers, valid_regions):
    """Copy a point field and mask it to the regions used in ParaView plots."""
    out = fem.Function(source.function_space, name=name)
    out.x.array[:] = source.x.array
    out.x.scatter_forward()
    mask_function_to_regions(out, out.function_space, cell_markers, valid_regions)
    return out


def xdmf_normalized_plot_field(source, name, cell_markers, valid_regions, power=1.0):
    """Region-masked normalized point field for robust visualization thresholds."""
    out = fem.Function(source.function_space, name=name)
    dofs = restricted_dofs(source.function_space, cell_markers, valid_regions)
    local_max = 0.0
    if dofs.size > 0:
        local_max = float(np.nanmax(np.abs(source.x.array.real[dofs])))
    global_max = float(source.function_space.mesh.comm.allreduce(local_max, op=MPI.MAX))
    if global_max <= 0.0 or not math.isfinite(global_max):
        out.x.array[:] = PETSc.ScalarType(0.0)
    else:
        normalized = np.clip(source.x.array.real / global_max, 0.0, 1.0)
        if power != 1.0:
            normalized = normalized**float(power)
        out.x.array[:] = normalized.astype(out.x.array.dtype)
    out.x.scatter_forward()
    mask_function_to_regions(out, out.function_space, cell_markers, valid_regions)
    return out


def xdmf_eta_gb_plot_field(
    scalar_outputs,
    name,
    cell_markers,
    p: Params,
    power=1.0,
    xi_cutoff: float | None = None,
    xi_suppression_power: float | None = None,
):
    """Normalized evolving GB indicator reconstructed from current eta_i fields."""
    etas = [fun for fun in scalar_outputs if fun.name.startswith("eta")]
    if not etas:
        return None
    out = fem.Function(etas[0].function_space, name=name)
    values = np.zeros_like(out.x.array.real)
    clipped = [np.clip(fun.x.array.real, 0.0, 1.0) for fun in etas]
    for i in range(len(clipped)):
        for j in range(i + 1, len(clipped)):
            values += clipped[i] * clipped[j]
    values = p.b_scale * np.minimum(p.b_clip, np.maximum(0.0, values))
    if xi_cutoff is not None or xi_suppression_power is not None:
        scalars = {fun.name: fun for fun in scalar_outputs}
        xi_fun = scalars.get("xi")
        if xi_fun is not None:
            xi_values = np.clip(xi_fun.x.array.real, 0.0, 1.0)
            if xi_cutoff is not None:
                values = np.where(xi_values < float(xi_cutoff), values, 0.0)
            if xi_suppression_power is not None:
                values = values * (1.0 - xi_values) ** float(xi_suppression_power)
    out.x.array[:] = values.astype(out.x.array.dtype)
    out.x.scatter_forward()
    return xdmf_normalized_plot_field(
        out, name, cell_markers, OMEGA1_REGIONS, power=power
    )


def composite_region_field(msh, V, cell_markers, xi_fun, B_fun):
    """Single DG0 helper field for quick paper-style ParaView coloring.

    Codes:
      1 = Omega1 electrolyte/grain background
      2 = Omega2 matrix background
      3 = Omega3 cathode particles
      4 = Li metal, xi high
      5 = grain boundary center band, normalized B high

    This field does not change the solve.  It only makes a one-field view in
    ParaView possible, avoiding manual multi-layer Threshold/Contour setup.
    """
    V0 = fem.functionspace(msh, ("DG", 0))
    out = fem.Function(V0, name="composite_region_field")
    tdim = msh.topology.dim
    num_local_cells = msh.topology.index_map(tdim).size_local
    values = np.zeros(num_local_cells, dtype=np.float64)

    cells, _points, markers = cells_and_points(V, cell_markers)
    xi_values = xi_fun.x.array.real
    B_values = B_fun.x.array.real
    omega1_dofs = restricted_dofs(V, cell_markers, OMEGA1_REGIONS)
    local_Bmax = 0.0
    if omega1_dofs.size > 0:
        local_Bmax = float(np.nanmax(np.abs(B_values[omega1_dofs])))
    Bmax = float(msh.comm.allreduce(local_Bmax, op=MPI.MAX))
    B_threshold = 0.35 * Bmax if Bmax > 0.0 and math.isfinite(Bmax) else math.inf

    for cell in range(num_local_cells):
        marker = int(markers[cell])
        if marker in OMEGA1_REGIONS:
            code = 1.0
        elif marker == OMEGA2:
            code = 2.0
        elif marker == OMEGA3:
            code = 3.0
        else:
            code = 0.0

        if marker in OMEGA1_REGIONS:
            dofs = cells[cell]
            xi_mean = float(np.mean(xi_values[dofs]))
            B_max_cell = float(np.max(np.abs(B_values[dofs])))
            if xi_mean >= 0.5:
                code = 4.0
            elif B_max_cell >= B_threshold:
                code = 5.0
        values[cell] = code

    out.x.array[:num_local_cells] = values.astype(out.x.array.dtype)
    out.x.scatter_forward()
    return out


def xdmf_plot_fields(scalar_outputs, derived_outputs, B, cell_markers, p: Params, diagnostic_outputs=()):
    """Point-wise visualization fields that avoid ParaView partial-array Threshold issues."""
    scalars = {fun.name: fun for fun in scalar_outputs}
    derived = {fun.name: fun for fun in derived_outputs}
    fields = []
    if "xi" in scalars:
        fields.append(xdmf_plot_field(scalars["xi"], "xi_plot", cell_markers, OMEGA1_REGIONS))
    if "ce" in derived:
        fields.append(xdmf_plot_field(derived["ce"], "ce_plot", cell_markers, OMEGA1_REGIONS))
    fields.append(xdmf_plot_field(B, "B_plot", cell_markers, OMEGA1_REGIONS))
    fields.append(xdmf_normalized_plot_field(B, "B_norm_plot", cell_markers, OMEGA1_REGIONS))
    fields.append(
        xdmf_normalized_plot_field(
            B, "B_sharp_plot", cell_markers, OMEGA1_REGIONS, power=3.0
        )
    )
    eta_gb = xdmf_eta_gb_plot_field(
        scalar_outputs, "eta_gb_plot", cell_markers, p, power=1.0
    )
    if eta_gb is not None:
        fields.append(eta_gb)
    eta_gb_no_li = xdmf_eta_gb_plot_field(
        scalar_outputs,
        "eta_gb_no_li_plot",
        cell_markers,
        p,
        power=1.0,
        xi_cutoff=0.001,
    )
    if eta_gb_no_li is not None:
        fields.append(eta_gb_no_li)
    eta_gb_effective = xdmf_eta_gb_plot_field(
        scalar_outputs,
        "eta_gb_effective_plot",
        cell_markers,
        p,
        power=1.0,
        xi_cutoff=0.5,
        xi_suppression_power=2.0,
    )
    if eta_gb_effective is not None:
        fields.append(eta_gb_effective)
    eta_gb_sharp = xdmf_eta_gb_plot_field(
        scalar_outputs, "eta_gb_sharp_plot", cell_markers, p, power=3.0
    )
    if eta_gb_sharp is not None:
        fields.append(eta_gb_sharp)
    if "hydrostatic_stress" in derived:
        fields.append(
            xdmf_plot_field(
                derived["hydrostatic_stress"],
                "hydrostatic_stress_omega1_plot",
                cell_markers,
                OMEGA1_REGIONS,
            )
        )
    if "hydrostatic_stress_cathode" in derived:
        fields.append(
            xdmf_plot_field(
                derived["hydrostatic_stress_cathode"],
                "hydrostatic_stress_cathode_plot",
                cell_markers,
                (OMEGA3,),
            )
        )
    if "hydrostatic_stress_omega1_smooth" in derived:
        fields.append(
            xdmf_plot_field(
                derived["hydrostatic_stress_omega1_smooth"],
                "hydrostatic_stress_omega1_smooth_plot",
                cell_markers,
                OMEGA1_REGIONS,
            )
        )
    if "hydrostatic_stress_omega12_smooth" in derived:
        fields.append(
            xdmf_plot_field(
                derived["hydrostatic_stress_omega12_smooth"],
                "hydrostatic_stress_omega12_smooth_plot",
                cell_markers,
                OMEGA1_REGIONS + (OMEGA2,),
            )
        )
    if "hydrostatic_stress_cathode_smooth" in derived:
        fields.append(
            xdmf_plot_field(
                derived["hydrostatic_stress_cathode_smooth"],
                "hydrostatic_stress_cathode_smooth_plot",
                cell_markers,
                (OMEGA3,),
            )
        )
    if "hydrostatic_stress_omega23_smooth" in derived:
        fields.append(
            xdmf_plot_field(
                derived["hydrostatic_stress_omega23_smooth"],
                "hydrostatic_stress_omega23_smooth_plot",
                cell_markers,
                (OMEGA2, OMEGA3),
            )
        )
    return tuple(fields)


def dx_regions(dx_measure, tags):
    """Return a single integration measure over several physical cell tags."""
    tags = tuple(tags)
    total = dx_measure(tags[0])
    for tag in tags[1:]:
        total = total + dx_measure(tag)
    return total


def region_dofs_from_markers(V, markers, region_ids):
    region_ids = set(region_ids)
    tdim = V.mesh.topology.dim
    num_cells = V.mesh.topology.index_map(tdim).size_local + V.mesh.topology.index_map(tdim).num_ghosts
    dofmap_list = V.dofmap.list
    dofmap_array = dofmap_list.array if hasattr(dofmap_list, "array") else np.asarray(dofmap_list)
    cells = dofmap_array.reshape((num_cells, -1)).astype(np.int32)
    chosen_cells = np.flatnonzero(np.isin(markers[:num_cells], list(region_ids)))
    if chosen_cells.size == 0:
        return np.empty(0, dtype=np.int32)
    return np.unique(cells[chosen_cells].reshape(-1)).astype(np.int32)


def global_union_region_dofs_from_markers(V, markers, region_ids):
    """Region dofs with MPI-consistent ownership-independent classification.

    A continuous P1 dof on a partition boundary can be seen as active on one
    rank and inactive on another.  Build the union in global dof numbering,
    then map that decision back to every rank's local owned+ghost numbering.
    """
    local_region_dofs = region_dofs_from_markers(V, markers, region_ids)
    index_map = V.dofmap.index_map
    local_size = index_map.size_local + index_map.num_ghosts
    local_all = np.arange(local_size, dtype=np.int32)
    local_region_global = np.asarray(
        index_map.local_to_global(local_region_dofs), dtype=np.int64
    )
    gathered = V.mesh.comm.allgather(local_region_global)
    if gathered:
        global_region = np.unique(
            np.concatenate([part for part in gathered if part.size > 0])
        )
    else:
        global_region = np.empty(0, dtype=np.int64)
    if global_region.size == 0:
        return np.empty(0, dtype=np.int32)
    local_all_global = np.asarray(index_map.local_to_global(local_all), dtype=np.int64)
    return local_all[np.isin(local_all_global, global_region)].astype(np.int32)


def global_inactive_only_dofs_from_markers(
    V, markers, inactive_region_ids, active_region_ids
):
    inactive = global_union_region_dofs_from_markers(V, markers, inactive_region_ids)
    active = global_union_region_dofs_from_markers(V, markers, active_region_ids)
    if inactive.size == 0:
        return inactive
    return np.setdiff1d(inactive, active, assume_unique=False).astype(np.int32)


def inactive_only_dofs_from_markers(V, markers, inactive_region_ids, active_region_ids):
    """Scalar dofs that belong only to inactive cells.

    Continuous P1 dofs on interfaces are shared by active and inactive cells.
    Removing all active-cell dofs keeps those interface traces free while
    strongly fixing the nonphysical interior of inactive regions.
    """
    inactive = region_dofs_from_markers(V, markers, inactive_region_ids)
    if inactive.size == 0:
        return inactive
    active = region_dofs_from_markers(V, markers, active_region_ids)
    return np.setdiff1d(inactive, active, assume_unique=False).astype(np.int32)


def component_region_dofs_from_markers(ME, component, markers, region_ids):
    """Mixed-subspace dofs on cells selected by physical-region tags."""
    region_ids = set(region_ids)
    tdim = ME.mesh.topology.dim
    num_cells = (
        ME.mesh.topology.index_map(tdim).size_local
        + ME.mesh.topology.index_map(tdim).num_ghosts
    )
    chosen_cells = np.flatnonzero(
        np.isin(markers[:num_cells], list(region_ids))
    ).astype(np.int32)
    if chosen_cells.size == 0:
        return np.empty(0, dtype=np.int32)
    dofmap_list = ME.sub(component).dofmap.list
    dofmap_array = (
        dofmap_list.array if hasattr(dofmap_list, "array") else np.asarray(dofmap_list)
    )
    cells = dofmap_array.reshape((num_cells, -1)).astype(np.int32)
    return np.unique(cells[chosen_cells].reshape(-1)).astype(np.int32)


def component_inactive_only_dofs_from_markers(
    ME, component, markers, inactive_region_ids, active_region_ids
):
    """Mixed-subspace dofs that belong only to inactive cells."""
    inactive = component_region_dofs_from_markers(
        ME, component, markers, inactive_region_ids
    )
    if inactive.size == 0:
        return inactive
    active = component_region_dofs_from_markers(
        ME, component, markers, active_region_ids
    )
    return np.setdiff1d(inactive, active, assume_unique=False).astype(np.int32)


def global_component_inactive_only_dofs_from_markers(
    ME, component, markers, inactive_region_ids, active_region_ids
):
    """Mixed component inactive dofs using a global union on the collapsed space."""
    V_sub, submap = collapse_mixed_subspace(ME, component)
    scalar_inactive = global_inactive_only_dofs_from_markers(
        V_sub, markers, inactive_region_ids, active_region_ids
    )
    if scalar_inactive.size == 0:
        return np.empty(0, dtype=np.int32)
    return submap[scalar_inactive].astype(np.int32)


def component_dirichlet_bc_from_scalar_dofs(ME, component, scalar_dofs, value=0.0):
    """Create a Dirichlet BC on one mixed component from collapsed scalar dofs."""
    if scalar_dofs.size == 0:
        return None
    _, submap = collapse_mixed_subspace(ME, component)
    mixed_dofs = submap[np.asarray(scalar_dofs, dtype=np.int32)]
    return fem.dirichletbc(PETSc.ScalarType(value), mixed_dofs, ME.sub(component))


def component_dirichlet_bc_from_dofs(ME, component, dofs, value=0.0):
    """Create a Dirichlet BC on one mixed component from subspace dofs."""
    if dofs.size == 0:
        return None
    return fem.dirichletbc(
        PETSc.ScalarType(value), np.asarray(dofs, dtype=np.int32), ME.sub(component)
    )


def build_lr_periodic_component_constraints(msh, ME, components, label, tol=1.0e-8):
    """Pair right-side mixed-component dofs to left-side dofs with the same y."""
    first_component = int(tuple(components)[0])
    V_scalar, _ = collapse_mixed_subspace(ME, first_component)
    coords = V_scalar.tabulate_dof_coordinates()[:, :2]
    x_min = float(msh.comm.allreduce(float(msh.geometry.x[:, 0].min()), op=MPI.MIN))
    x_max = float(msh.comm.allreduce(float(msh.geometry.x[:, 0].max()), op=MPI.MAX))
    left = np.flatnonzero(np.isclose(coords[:, 0], x_min, atol=tol))
    right = np.flatnonzero(np.isclose(coords[:, 0], x_max, atol=tol))
    if left.size != right.size:
        raise RuntimeError(
            f"Cannot build left-right periodic {label} constraints: "
            f"left dofs={left.size}, right dofs={right.size}."
        )

    left_sorted = np.asarray(
        sorted(left, key=lambda dof: float(coords[dof, 1])), dtype=np.int32
    )
    right_sorted = np.asarray(
        sorted(right, key=lambda dof: float(coords[dof, 1])), dtype=np.int32
    )
    y_delta = np.abs(coords[left_sorted, 1] - coords[right_sorted, 1])
    max_y_delta = float(np.max(y_delta)) if y_delta.size else 0.0
    if max_y_delta > tol:
        raise RuntimeError(
            f"Cannot build left-right periodic {label} constraints: "
            f"max paired y mismatch={max_y_delta:.3e} exceeds tol={tol:.3e}. "
            "Regenerate the mesh with matching left/right boundary divisions, "
            "or increase the pairing tolerance only if this mismatch is purely roundoff."
        )
    constraints = {}
    for left_dof, right_dof in zip(left_sorted, right_sorted):
        for component in components:
            _, submap = collapse_mixed_subspace(ME, component)
            constraints[int(submap[right_dof])] = int(submap[left_dof])
    return constraints


def build_lr_periodic_constraints(msh, ME, n_grains, tol=1.0e-8):
    """Left-right periodic constraints for xi, eta_i, and mechanics."""
    constraints = {}
    periodic_groups = (
        ((0,), "xi"),
        (tuple(range(6, 6 + n_grains)), "eta"),
        ((4, 5), "mechanics"),
    )
    counts = {}
    for components, label in periodic_groups:
        group_constraints = build_lr_periodic_component_constraints(
            msh, ME, components, label, tol=tol
        )
        overlap = set(constraints).intersection(group_constraints)
        if overlap:
            raise RuntimeError(
                f"Duplicate periodic slave dofs while adding {label}: "
                f"{sorted(overlap)[:5]}"
            )
        constraints.update(group_constraints)
        counts[label] = len(group_constraints)
    return constraints, counts


def global_dofs_from_local(V, local_dofs):
    local_dofs = np.asarray(local_dofs, dtype=np.int64)
    index_map = V.dofmap.index_map
    bs = int(V.dofmap.index_map_bs)
    try:
        local_range_start = int(index_map.local_range[0])
    except TypeError:
        local_range_start = int(index_map.local_range()[0])
    if bs == 1:
        return local_range_start + local_dofs
    blocks = (local_dofs // bs).astype(np.int32)
    offsets = (local_dofs % bs).astype(np.int64)
    return (local_range_start + blocks.astype(np.int64)) * bs + offsets


def build_lr_periodic_component_constraints_global(msh, ME, components, label, tol=1.0e-8):
    """MPI-safe global right-to-left periodic dof map for mixed components."""
    local_x_min = float(np.min(msh.geometry.x[:, 0]))
    local_x_max = float(np.max(msh.geometry.x[:, 0]))
    x_min = float(msh.comm.allreduce(local_x_min, op=MPI.MIN))
    x_max = float(msh.comm.allreduce(local_x_max, op=MPI.MAX))
    constraints = {}
    counts = {}

    for component in tuple(components):
        V_scalar, submap = collapse_mixed_subspace(ME, component)
        coords = V_scalar.tabulate_dof_coordinates()[:, :2]
        scalar_local = np.arange(coords.shape[0], dtype=np.int32)
        parent_local = np.asarray(submap, dtype=np.int64)[scalar_local]
        owned_size = int(ME.dofmap.index_map.size_local) * int(ME.dofmap.index_map_bs)
        owned = parent_local < owned_size
        parent_local = parent_local[owned]
        coords = coords[owned]
        parent_global = global_dofs_from_local(ME, parent_local)

        side_records = []
        left_mask = np.isclose(coords[:, 0], x_min, atol=tol)
        right_mask = np.isclose(coords[:, 0], x_max, atol=tol)
        for y, g in zip(coords[left_mask, 1], parent_global[left_mask]):
            side_records.append(("L", float(y), int(g)))
        for y, g in zip(coords[right_mask, 1], parent_global[right_mask]):
            side_records.append(("R", float(y), int(g)))

        gathered = [rec for part in msh.comm.allgather(side_records) for rec in part]
        left = sorted((rec for rec in gathered if rec[0] == "L"), key=lambda rec: rec[1])
        right = sorted((rec for rec in gathered if rec[0] == "R"), key=lambda rec: rec[1])
        if len(left) != len(right):
            raise RuntimeError(
                f"Cannot build MPI periodic {label} component {component}: "
                f"left dofs={len(left)}, right dofs={len(right)}."
            )
        if left:
            max_y_delta = max(abs(l[1] - r[1]) for l, r in zip(left, right))
            if max_y_delta > tol:
                raise RuntimeError(
                    f"Cannot build MPI periodic {label} component {component}: "
                    f"max paired y mismatch={max_y_delta:.3e} exceeds tol={tol:.3e}."
                )
        for l, r in zip(left, right):
            constraints[int(r[2])] = int(l[2])
        counts[int(component)] = len(right)
    if msh.comm.rank == 0:
        print(
            f"periodic global {label}: paired {sum(counts.values())} slave dofs "
            f"across {len(counts)} component(s).",
            flush=True,
        )
    return constraints, counts


def build_lr_periodic_constraints_global(msh, ME, n_grains, tol=1.0e-8):
    """Global periodic constraints for MPI projection."""
    constraints = {}
    counts = {}
    periodic_groups = (
        ((0,), "xi"),
        (tuple(range(6, 6 + n_grains)), "eta"),
        ((4, 5), "mechanics"),
    )
    for components, label in periodic_groups:
        group_constraints, component_counts = build_lr_periodic_component_constraints_global(
            msh, ME, components, label, tol=tol
        )
        overlap = set(constraints).intersection(group_constraints)
        if overlap:
            raise RuntimeError(
                f"Duplicate MPI periodic slave dofs while adding {label}: "
                f"{sorted(overlap)[:5]}"
            )
        constraints.update(group_constraints)
        counts[label] = sum(component_counts.values())
    return constraints, counts


def apply_global_periodic_constraints_to_function(fun, constraints_global):
    """Project a distributed function onto global slave=master constraints.

    This is used for MPI initial/old states before the nonlinear solve starts.
    The Newton correction projection handles updates later, but the residual
    also depends on old-time states, so w_n and w_nm1 must be made periodic too.
    """
    constraints_global = dict(constraints_global or {})
    if not constraints_global:
        fun.x.scatter_forward()
        return

    comm = fun.function_space.mesh.comm
    vec = fun.x.petsc_vec
    global_size = int(vec.getSize())
    local_size = int(vec.getLocalSize())
    row_start, row_end = vec.getOwnershipRange()

    try:
        P = PETSc.Mat().createAIJ(
            size=((local_size, global_size), (local_size, global_size)),
            nnz=1,
            comm=comm,
        )
    except Exception:
        P = PETSc.Mat().createAIJ(size=(global_size, global_size), nnz=1, comm=comm)

    for row in range(int(row_start), int(row_end)):
        col = int(constraints_global.get(row, row))
        P.setValue(row, col, 1.0)
    P.assemble()

    src = vec.duplicate()
    dst = vec.duplicate()
    vec.copy(src)
    P.mult(src, dst)
    dst.copy(vec)
    fun.x.scatter_forward()
    src.destroy()
    dst.destroy()
    P.destroy()


def build_lr_periodic_displacement_constraints(msh, ME, tol=1.0e-8):
    """Backward-compatible mechanics-only left-right periodic constraints."""
    return build_lr_periodic_component_constraints(
        msh, ME, (4, 5), "mechanics", tol=tol
    )


def boundary_dofs_from_tag(V, facet_tags, tag, fallback_locator=None):
    """Return scalar-space dofs on a tagged boundary for COMSOL-like boundary probes.

    This is diagnostics only. It does not impose a Dirichlet boundary condition.
    """
    if fallback_locator is not None:
        # Probe dofs are diagnostics only.  On highly partitioned meshes,
        # topology-based boundary lookup can put ranks on noticeably different
        # paths before later vector scatters.  Coordinate filtering is enough
        # for P1 scalar boundary probes and stays purely local.
        coords = V.tabulate_dof_coordinates()[:, : V.mesh.geometry.dim]
        keep = np.asarray(fallback_locator(coords.T), dtype=bool)
        if keep.size != coords.shape[0]:
            raise RuntimeError(
                f"Boundary fallback for tag {tag} returned {keep.size} flags for "
                f"{coords.shape[0]} dofs."
            )
        return np.flatnonzero(keep).astype(np.int32)

    fdim = V.mesh.topology.dim - 1
    facets = facet_tags.find(tag)
    if len(facets) > 0:
        return fem.locate_dofs_topological(V, fdim, facets).astype(np.int32)
    return np.empty(0, dtype=np.int32)

def cells_and_points(V, markers=None, owned_only: bool = False):
    if vtk_mesh is not None:
        topology, _cell_types, geometry = vtk_mesh(V)
        topology = np.asarray(topology, dtype=np.int64)
        geometry = np.asarray(geometry, dtype=np.float64)
        if topology.ndim == 1:
            cells = []
            offset = 0
            while offset < topology.size:
                width = int(topology[offset])
                offset += 1
                cells.append(np.asarray(topology[offset : offset + width], dtype=np.int32))
                offset += width
            cells = np.asarray(cells, dtype=np.int32)
        else:
            cells = np.asarray(topology, dtype=np.int32)
        points = geometry[:, :2]
        if markers is not None:
            num_cells = min(cells.shape[0], len(markers))
        else:
            num_cells = cells.shape[0]
        cells = cells[:num_cells]
        cell_markers_out = markers[:num_cells] if markers is not None else None
    else:
        msh = V.mesh
        tdim = msh.topology.dim
        index_map = msh.topology.index_map(tdim)
        num_cells = index_map.size_local
        if not owned_only:
            num_cells += index_map.num_ghosts
        dofmap_list = V.dofmap.list
        dofmap_array = dofmap_list.array if hasattr(dofmap_list, "array") else np.asarray(dofmap_list)
        offsets = getattr(dofmap_list, "offsets", None)
        if offsets is not None:
            offsets = np.asarray(offsets, dtype=np.int64)
            total_cells = offsets.size - 1
            if num_cells > total_cells:
                num_cells = total_cells
            cells = [
                np.asarray(dofmap_array[offsets[i] : offsets[i + 1]], dtype=np.int32)
                for i in range(num_cells)
            ]
        else:
            cell_dofs_fn = getattr(V.dofmap, "cell_dofs", None)
            if cell_dofs_fn is None:
                raise RuntimeError("Plotting requires either vtk_mesh() or cell_dofs().")
            cells = [np.asarray(cell_dofs_fn(i), dtype=np.int32) for i in range(num_cells)]
        points = V.tabulate_dof_coordinates()[:, :2]
        cell_markers_out = markers[:num_cells] if markers is not None else None
    cell_widths = {cell.size for cell in cells}
    if len(cell_widths) != 1:
        raise ValueError(f"Mixed plotting cell widths are not supported: {sorted(cell_widths)}")
    cells = np.asarray(cells, dtype=np.int32)
    if cells.shape[1] not in (3, 4):
        raise ValueError(f"Unsupported plotting cell dof count {cells.shape[1]}.")
    if cells.shape[1] == 4:
        # Match the grain-initialization preview path: Basix/Gmsh dof order on
        # quads is not necessarily geometric order, and splitting unsorted
        # quads into triangles creates artificial broken/diagonal bands.
        cell_points = points[cells]
        centers = np.mean(cell_points, axis=1)
        angles = np.arctan2(
            cell_points[:, :, 1] - centers[:, None, 1],
            cell_points[:, :, 0] - centers[:, None, 0],
        )
        order = np.argsort(angles, axis=1)
        cells = np.take_along_axis(cells, order, axis=1)
    return cells, points, cell_markers_out


def restricted_cells(V, cell_markers=None, valid_regions=None, owned_only: bool = False):
    cells, points, markers = cells_and_points(V, cell_markers, owned_only=owned_only)
    if valid_regions is None or markers is None:
        return cells, points
    if isinstance(valid_regions, (int, np.integer)):
        valid_regions = (int(valid_regions),)
    keep = np.isin(markers, list(valid_regions))
    return cells[keep], points


def restricted_dofs(V, cell_markers=None, valid_regions=None, owned_only: bool = False):
    cells, _points, markers = cells_and_points(V, cell_markers, owned_only=owned_only)
    if valid_regions is None or markers is None:
        return np.arange(V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts)
    if isinstance(valid_regions, (int, np.integer)):
        valid_regions = (int(valid_regions),)
    keep = np.isin(markers, list(valid_regions))
    if not np.any(keep):
        return np.empty(0, dtype=np.int32)
    return np.unique(cells[keep].reshape(-1)).astype(np.int32)


def build_subsampled_triangles(V, cell_markers=None, valid_regions=None, order: int = 5, owned_only: bool = False):
    cells, points = restricted_cells(V, cell_markers, valid_regions, owned_only=owned_only)
    if cells.size == 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 3), dtype=np.float64),
        )
    triangles = cells if cells.shape[1] == 3 else np.vstack(
        (cells[:, [0, 1, 2]], cells[:, [0, 2, 3]])
    ).astype(np.int32)

    bary = []
    bary_id = {}
    for i in range(order + 1):
        for j in range(order + 1 - i):
            k = order - i - j
            bary_id[(i, j)] = len(bary)
            bary.append((i / order, j / order, k / order))
    bary = np.asarray(bary, dtype=np.float64)

    local_tris = []
    for i in range(order):
        for j in range(order - i):
            a = bary_id[(i, j)]
            b = bary_id[(i + 1, j)]
            c = bary_id[(i, j + 1)]
            local_tris.append((a, b, c))
            if j < order - i - 1:
                d = bary_id[(i + 1, j + 1)]
                local_tris.append((b, d, c))
    local_tris = np.asarray(local_tris, dtype=np.int32)

    sample_points = []
    sample_cells = []
    sample_parent_tris = []
    sample_bary = []
    offset = 0
    for tri in triangles:
        tri_points = points[tri]
        new_points = bary @ tri_points
        sample_points.append(new_points)
        sample_cells.append(local_tris + offset)
        sample_parent_tris.append(np.repeat(tri[None, :], len(bary), axis=0))
        sample_bary.append(bary)
        offset += len(bary)

    return (
        np.vstack(sample_points),
        np.vstack(sample_cells),
        np.vstack(sample_parent_tris),
        np.vstack(sample_bary),
    )


def _gather_png_payload(comm, *arrays):
    if comm.size == 1:
        return arrays
    gathered = comm.gather(arrays, root=0)
    if comm.rank != 0:
        return None
    payload = [[] for _ in arrays]
    point_offset = 0
    for chunk in gathered:
        if not chunk:
            continue
        pts = chunk[0]
        if pts.size == 0:
            continue
        payload[0].append(pts)
        if len(chunk) > 1:
            payload[1].append(chunk[1] + point_offset)
        for idx in range(2, len(chunk)):
            payload[idx].append(chunk[idx])
        point_offset += pts.shape[0]
    result = []
    for idx, parts in enumerate(payload):
        if not parts:
            if idx == 0:
                result.append(np.empty((0, 2), dtype=np.float64))
            elif idx == 1:
                result.append(np.empty((0, 3), dtype=np.int32))
            else:
                result.append(np.empty(0, dtype=np.float64))
            continue
        result.append(np.vstack(parts) if idx < 2 else np.concatenate(parts))
    return tuple(result)


def save_sampled_field_png(
    V,
    field,
    filename,
    title,
    cmap="turbo",
    vmin=0.0,
    vmax=1.0,
    cell_markers=None,
    valid_regions=None,
    order: int = 5,
):
    if not HAS_MATPLOTLIB:
        return
    comm = V.mesh.comm
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    sample_points, sample_cells, parent_tris, sample_bary = build_subsampled_triangles(
        V,
        cell_markers=cell_markers,
        valid_regions=valid_regions,
        order=order,
        owned_only=True,
    )
    if sample_points.size == 0:
        gathered = _gather_png_payload(
            comm, sample_points, sample_cells, np.empty(0, dtype=np.float64)
        )
        if gathered is None:
            return
        sample_points, sample_cells, sample_values = gathered
        if sample_points.size == 0 or sample_cells.size == 0:
            return
        triangulation = mtri.Triangulation(
            sample_points[:, 0], sample_points[:, 1], sample_cells
        )
        fig, ax = plt.subplots(figsize=(7.2, 8.4), dpi=180)
        color = ax.tripcolor(
            triangulation,
            sample_values,
            shading="flat",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("x / L")
        ax.set_ylabel("y / L")
        ax.set_title(title)
        fig.colorbar(color, ax=ax)
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)
        return

    values = field.x.array.real
    samples_per_parent = sample_bary.shape[0] // parent_tris.shape[0]
    sample_values = []
    for parent_index in range(parent_tris.shape[0]):
        bary_slice = sample_bary[
            parent_index * samples_per_parent : (parent_index + 1) * samples_per_parent
        ]
        tri = parent_tris[parent_index * samples_per_parent]
        sample_values.append(bary_slice @ values[tri])
    sample_values = np.concatenate(sample_values)
    gathered = _gather_png_payload(comm, sample_points, sample_cells, sample_values)
    if gathered is None:
        return
    sample_points, sample_cells, sample_values = gathered

    triangulation = mtri.Triangulation(sample_points[:, 0], sample_points[:, 1], sample_cells)
    fig, ax = plt.subplots(figsize=(7.2, 8.4), dpi=180)
    color = ax.tripcolor(
        triangulation,
        sample_values,
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x / L")
    ax.set_ylabel("y / L")
    ax.set_title(title)
    fig.colorbar(color, ax=ax)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def save_sampled_interface_indicator_png(
    V,
    xi_fun,
    eta_funs,
    p: Params,
    filename,
    title,
    cmap="turbo",
    vmin=0.0,
    vmax=1.0,
    cell_markers=None,
    valid_regions=None,
    order: int = 5,
):
    if not HAS_MATPLOTLIB:
        return
    comm = V.mesh.comm
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    sample_points, sample_cells, parent_tris, sample_bary = build_subsampled_triangles(
        V,
        cell_markers=cell_markers,
        valid_regions=valid_regions,
        order=order,
        owned_only=True,
    )
    if sample_points.size == 0:
        gathered = _gather_png_payload(
            comm, sample_points, sample_cells, np.empty(0, dtype=np.float64)
        )
        if gathered is None:
            return
        sample_points, sample_cells, ce_values = gathered
        if sample_points.size == 0 or sample_cells.size == 0:
            return
        triangulation = mtri.Triangulation(
            sample_points[:, 0], sample_points[:, 1], sample_cells
        )
        fig, ax = plt.subplots(figsize=(7.2, 8.4), dpi=180)
        color = ax.tripcolor(
            triangulation,
            ce_values,
            shading="flat",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("x / L")
        ax.set_ylabel("y / L")
        ax.set_title(title)
        fig.colorbar(color, ax=ax)
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)
        return

    def sample_scalar(values):
        return np.sum(sample_bary * values[parent_tris], axis=1)

    def box_indicator_np(value, half_width):
        return (np.abs(value) < half_width).astype(np.float64)

    xi_sample = sample_scalar(xi_fun.x.array.real)
    eta_samples = np.vstack([sample_scalar(eta_i.x.array.real) for eta_i in eta_funs]).T
    eta_clip = np.clip(eta_samples, 0.0, 1.0)
    gb_window = np.sum(
        (1.0 - eta_clip)
        * box_indicator_np(eta_samples - 0.5, p.rho),
        axis=1,
    )
    ce_values = xi_sample + (1.0 - xi_sample) ** 2 * gb_window
    gathered = _gather_png_payload(comm, sample_points, sample_cells, ce_values)
    if gathered is None:
        return
    sample_points, sample_cells, ce_values = gathered

    triangulation = mtri.Triangulation(sample_points[:, 0], sample_points[:, 1], sample_cells)
    fig, ax = plt.subplots(figsize=(7.2, 8.4), dpi=180)
    color = ax.tripcolor(
        triangulation,
        ce_values,
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x / L")
    ax.set_ylabel("y / L")
    ax.set_title(title)
    fig.colorbar(color, ax=ax)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def safe_xdmf_field_prefix(prefix):
    """Keep time-stamped PNG prefixes usable as ParaView array names."""
    return str(prefix).replace(".", "p").replace("-", "m").replace(":", "_")


def write_sampled_xdmf(path, grid_name, points, cells, point_values, cell_values):
    """Write a sampled triangle mesh as a standalone XDMF file for ParaView."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def rows(array):
        array = np.asarray(array)
        if array.ndim == 1:
            return " ".join(f"{float(v):.17g}" for v in array)
        if np.issubdtype(array.dtype, np.integer):
            return "\n".join(" ".join(str(int(v)) for v in row) for row in array)
        return "\n".join(" ".join(f"{float(v):.17g}" for v in row) for row in array)

    name = safe_xdmf_field_prefix(grid_name)
    point_attr = f"{name}_sampled_point"
    cell_attr = f"{name}_sampled_cell"
    xml = f'''<?xml version="1.0" ?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="{name}_sampled" GridType="Uniform">
      <Topology TopologyType="Triangle" NumberOfElements="{cells.shape[0]}">
        <DataItem Dimensions="{cells.shape[0]} 3" NumberType="Int" Format="XML">
          {rows(cells)}
        </DataItem>
      </Topology>
      <Geometry GeometryType="XY">
        <DataItem Dimensions="{points.shape[0]} 2" NumberType="Float" Precision="8" Format="XML">
          {rows(points)}
        </DataItem>
      </Geometry>
      <Attribute Name="{point_attr}" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{point_values.shape[0]}" NumberType="Float" Precision="8" Format="XML">
          {rows(point_values)}
        </DataItem>
      </Attribute>
      <Attribute Name="{cell_attr}" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions="{cell_values.shape[0]}" NumberType="Float" Precision="8" Format="XML">
          {rows(cell_values)}
        </DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
'''
    path.write_text(xml, encoding="utf-8")


def save_sampled_field_xdmf(
    V,
    field,
    filename,
    grid_name,
    cell_markers=None,
    valid_regions=None,
    order: int = 5,
):
    comm = V.mesh.comm
    sample_points, sample_cells, parent_tris, sample_bary = build_subsampled_triangles(
        V,
        cell_markers=cell_markers,
        valid_regions=valid_regions,
        order=order,
        owned_only=True,
    )
    if sample_points.size == 0:
        gathered = _gather_png_payload(
            comm,
            sample_points,
            sample_cells,
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
        if gathered is None:
            return
        sample_points, sample_cells, sample_values, cell_values = gathered
    else:
        sample_values = np.sum(sample_bary * field.x.array.real[parent_tris], axis=1)
        cell_values = np.mean(sample_values[sample_cells], axis=1)
        gathered = _gather_png_payload(comm, sample_points, sample_cells, sample_values, cell_values)
        if gathered is None:
            return
        sample_points, sample_cells, sample_values, cell_values = gathered
    if sample_points.size == 0:
        return
    write_sampled_xdmf(filename, grid_name, sample_points, sample_cells, sample_values, cell_values)


def save_sampled_interface_indicator_xdmf(
    V,
    xi_fun,
    eta_funs,
    p: Params,
    filename,
    grid_name,
    cell_markers=None,
    valid_regions=None,
    order: int = 5,
):
    comm = V.mesh.comm
    sample_points, sample_cells, parent_tris, sample_bary = build_subsampled_triangles(
        V,
        cell_markers=cell_markers,
        valid_regions=valid_regions,
        order=order,
        owned_only=True,
    )
    if sample_points.size == 0:
        gathered = _gather_png_payload(
            comm,
            sample_points,
            sample_cells,
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
        if gathered is None:
            return
        sample_points, sample_cells, ce_values, ce_cell_values = gathered
    else:
        def sample_scalar(values):
            return np.sum(sample_bary * values[parent_tris], axis=1)

        xi_sample = sample_scalar(xi_fun.x.array.real)
        eta_samples = np.vstack([sample_scalar(eta_i.x.array.real) for eta_i in eta_funs]).T
        eta_clip = np.clip(eta_samples, 0.0, 1.0)
        gb_window = np.sum(
            (1.0 - eta_clip)
            * (np.abs(eta_samples - 0.5) < p.rho).astype(np.float64),
            axis=1,
        )
        ce_values = xi_sample + (1.0 - xi_sample) ** 2 * gb_window
        ce_cell_values = np.mean(ce_values[sample_cells], axis=1)
        gathered = _gather_png_payload(comm, sample_points, sample_cells, ce_values, ce_cell_values)
        if gathered is None:
            return
        sample_points, sample_cells, ce_values, ce_cell_values = gathered
    if sample_points.size == 0:
        return
    write_sampled_xdmf(filename, grid_name, sample_points, sample_cells, ce_values, ce_cell_values)


def save_field_png(
    V,
    field,
    filename,
    title,
    cmap="viridis",
    symmetric=False,
    vmin=None,
    vmax=None,
    overlay=None,
    cell_markers=None,
    valid_regions=None,
    fill_outside=None,
):
    if not HAS_MATPLOTLIB:
        return
    comm = V.mesh.comm
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    values = field.x.array.real.copy()
    cells_all, points, markers = cells_and_points(V, cell_markers, owned_only=True)
    if fill_outside is None:
        if valid_regions is None or markers is None:
            cells = cells_all
        else:
            regions = (
                (valid_regions,)
                if isinstance(valid_regions, (int, np.integer))
                else tuple(valid_regions)
            )
            cells = cells_all[np.isin(markers, regions)]
        plot_dofs = (
            np.unique(cells.reshape(-1)).astype(np.int32)
            if cells.size
            else np.empty(0, dtype=np.int32)
        )
        clim_values = values[plot_dofs].copy() if plot_dofs.size else np.empty(0)
    else:
        if valid_regions is not None and markers is not None:
            regions = (
                (valid_regions,)
                if isinstance(valid_regions, (int, np.integer))
                else tuple(valid_regions)
            )
            keep_cells = np.isin(markers, regions)
            valid_dofs = (
                np.unique(cells_all[keep_cells].reshape(-1))
                if np.any(keep_cells)
                else np.empty(0, dtype=np.int32)
            )
            invalid = np.ones(values.shape[0], dtype=bool)
            invalid[valid_dofs] = False
            values[invalid] = float(fill_outside)
            clim_values = values[valid_dofs].copy()
        else:
            clim_values = values.copy()
        cells = cells_all
    triangles = (
        cells
        if cells.size and cells.shape[1] == 3
        else (
            np.vstack((cells[:, [0, 1, 2]], cells[:, [0, 2, 3]])).astype(np.int32)
            if cells.size
            else np.empty((0, 3), dtype=np.int32)
        )
    )
    overlay_values = np.empty(0, dtype=np.float64)
    if overlay is not None:
        overlay_values = overlay.x.array.real.copy()
    gathered = _gather_png_payload(
        comm,
        points,
        triangles,
        values,
        overlay_values,
        clim_values,
    )
    if gathered is None:
        return
    points, triangles, values, overlay_values, clim_values = gathered
    if triangles.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7.0, 8.0), dpi=180)
    if symmetric:
        finite_values = clim_values[np.isfinite(clim_values)]
        vmax = max(float(np.max(np.abs(finite_values))) if finite_values.size else 0.0, 1.0e-30)
        vmin = -vmax
    tri = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    finite_values = clim_values[np.isfinite(clim_values)]
    try:
        if finite_values.size == 0:
            raise ValueError("no finite values to plot")
        if vmin is None:
            vmin_plot = float(np.min(finite_values))
        else:
            vmin_plot = float(vmin)
        if vmax is None:
            vmax_plot = float(np.max(finite_values))
        else:
            vmax_plot = float(vmax)
        if np.isclose(vmin_plot, vmax_plot):
            pad = max(abs(vmin_plot), 1.0) * 1.0e-9
            vmin_plot -= pad
            vmax_plot += pad
        levels = np.linspace(vmin_plot, vmax_plot, 96)
        color = ax.tricontourf(
            tri,
            values,
            levels=levels,
            cmap=cmap,
            vmin=vmin_plot,
            vmax=vmax_plot,
            extend="both",
            antialiased=False,
        )
        ax.set_xlim(float(points[:, 0].min()), float(points[:, 0].max()))
        ax.set_ylim(float(points[:, 1].min()), float(points[:, 1].max()))
    except Exception:
        color = ax.tripcolor(
            tri,
            values,
            shading="gouraud",
            cmap=cmap,
            edgecolors="none",
            linewidth=0.0,
            antialiased=False,
        )
        if vmin is not None and vmax is not None:
            color.set_clim(vmin, vmax)
    if overlay is not None:
        ax.tricontour(
            tri,
            overlay_values,
            levels=[0.35],
            colors="black",
            linewidths=0.95,
            alpha=0.90,
        )
    ax.set_aspect("equal")
    ax.set_xlabel("x / L")
    ax.set_ylabel("y / L")
    ax.set_title(title)
    fig.colorbar(color, ax=ax)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


# 保存主要预览图。PNG 仅用于快速查看，定量后处理建议使用 XDMF/ParaView。
def save_png_outputs(out_dir, prefix, V, scalar_outputs, derived_outputs, B, cell_markers, p: Params):
    xi_fun, phil_fun, phis_fun, c_fun, ux_fun, uy_fun = scalar_outputs[:6]
    (
        ce_fun,
        gb_window_fun,
        hxi_fun,
        sigma_fun,
        reaction_li_fun,
        deposition_drive_fun,
        xi_source_weight_fun,
        xit_fun,
        overp_li_fun,
        overp_c_fun,
        eeq_fun,
        hydro_fun,
        overp_mech_fun,
        u_mag_fun,
        liion_source_fun,
        dfmechdxi_fun,
        eeq_eff_fun,
        hydro_cathode_fun,
        overp_mech_c_fun,
        stress_flux_drive_fun,
        hydro_cathode_smooth_fun,
        hydro_omega23_smooth_fun,
        heta_fun,
        E1_fun,
        nu1_fun,
        hydro_omega1_smooth_fun,
        hydro_omega12_smooth_fun,
    ) = derived_outputs
    out_dir = Path(out_dir) / "png" / str(prefix)
    reg_xi = OMEGA1_REGIONS
    reg_phil = OMEGA1_REGIONS + (OMEGA2,)
    reg_phis = (OMEGA2,)
    reg_cathode = (OMEGA3,)
    eta_funs = [fun for fun in scalar_outputs[6:] if fun.name.startswith("eta")]
    save_field_png(V, xi_fun, out_dir / "xi.png", f"{prefix}: xi", vmin=0.0, vmax=1.0, overlay=B, cell_markers=cell_markers, valid_regions=reg_xi )
    save_sampled_interface_indicator_png(V, xi_fun, eta_funs, p, out_dir / "interface_indicator.png", f"{prefix}: interface indicator", cmap="turbo", vmin=0.0, vmax=1.0, cell_markers=cell_markers, valid_regions=reg_xi )
    save_field_png(V, phil_fun, out_dir / "phil.png", f"{prefix}: phi_l [V]", cell_markers=cell_markers, valid_regions=reg_phil )
    save_field_png(V, phis_fun, out_dir / "phis.png", f"{prefix}: phi_s [V]", cell_markers=cell_markers, valid_regions=reg_phis )
    save_field_png(V, c_fun, out_dir / "soc.png", f"{prefix}: particle SOC", cell_markers=cell_markers, valid_regions=reg_cathode )
    save_field_png(V, c_fun, out_dir / "soc_full.png", f"{prefix}: particle SOC, full domain", vmin=0.0, vmax=1.0, cell_markers=cell_markers, valid_regions=reg_cathode, fill_outside=0.0)
    eeq_eff_smooth_fun = fem.Function(eeq_eff_fun.function_space, name="eeq_eff_smooth")
    eeq_eff_smooth_fun.x.array[:] = (
        hydro_cathode_smooth_fun.x.array.real * p.omega_cathode / p.F
    ).astype(eeq_eff_smooth_fun.x.array.dtype)
    eeq_eff_smooth_fun.x.scatter_forward()
    save_field_png(V, eeq_eff_smooth_fun, out_dir / "stress_overpotential.png", f"{prefix}: stress overpotential [V]", cmap="coolwarm", symmetric=True, cell_markers=cell_markers, valid_regions=reg_cathode )
    save_field_png(V, hydro_fun, out_dir / "stress_o1_raw.png", f"{prefix}: Omega1 stress [Pa]", cmap="coolwarm", symmetric=True, overlay=B, cell_markers=cell_markers, valid_regions=reg_xi )
    save_field_png(V, hydro_omega1_smooth_fun, out_dir / "stress_o1.png", f"{prefix}: Omega1 stress [Pa]", cmap="coolwarm", symmetric=True, overlay=B, cell_markers=cell_markers, valid_regions=reg_xi )
    save_field_png(V, hydro_omega12_smooth_fun, out_dir / "stress_o12.png", f"{prefix}: Omega1+Omega2 stress [Pa]", cmap="coolwarm", symmetric=True, overlay=B, cell_markers=cell_markers, valid_regions=OMEGA1_REGIONS + (OMEGA2,) )
    save_field_png(V, hydro_cathode_smooth_fun, out_dir / "stress_cathode.png", f"{prefix}: cathode stress [Pa]", cmap="coolwarm", symmetric=True, cell_markers=cell_markers, valid_regions=reg_cathode )
    save_field_png(V, hydro_omega23_smooth_fun, out_dir / "stress_o23.png", f"{prefix}: Omega2+Omega3 stress [Pa]", cmap="coolwarm", symmetric=True, cell_markers=cell_markers, valid_regions=(OMEGA2, OMEGA3) )
    if prefix == "initial":
        save_field_png(V, B, out_dir / "initial_B.png", "initial: GB indicator B", cmap="magma", vmin=0.0, vmax=1.0, overlay=B, cell_markers=cell_markers, valid_regions=reg_xi )
    elif prefix in ("final", "after_charge", "after_discharge"):
        save_field_png(V, B, out_dir / "gb_indicator.png", f"{prefix}: GB indicator B", cmap="magma", vmin=0.0, vmax=1.0, overlay=B, cell_markers=cell_markers, valid_regions=reg_xi )


def save_sampled_outputs(out_dir, prefix, V, scalar_outputs, derived_outputs, B, cell_markers, p: Params):
    xi_fun, phil_fun, phis_fun, c_fun, ux_fun, uy_fun = scalar_outputs[:6]
    (
        ce_fun,
        gb_window_fun,
        hxi_fun,
        sigma_fun,
        reaction_li_fun,
        deposition_drive_fun,
        xi_source_weight_fun,
        xit_fun,
        overp_li_fun,
        overp_c_fun,
        eeq_fun,
        hydro_fun,
        overp_mech_fun,
        u_mag_fun,
        liion_source_fun,
        dfmechdxi_fun,
        eeq_eff_fun,
        hydro_cathode_fun,
        overp_mech_c_fun,
        stress_flux_drive_fun,
        hydro_cathode_smooth_fun,
        hydro_omega23_smooth_fun,
        heta_fun,
        E1_fun,
        nu1_fun,
        hydro_omega1_smooth_fun,
        hydro_omega12_smooth_fun,
    ) = derived_outputs
    out_dir = Path(out_dir) / "sampled_xdmf" / prefix
    reg_xi = OMEGA1_REGIONS
    reg_phil = OMEGA1_REGIONS + (OMEGA2,)
    reg_phis = (OMEGA2,)
    reg_cathode = (OMEGA3,)
    eta_funs = [fun for fun in scalar_outputs[6:] if fun.name.startswith("eta")]

    save_sampled_field_xdmf(V, xi_fun, out_dir / "xi_sampled.xdmf", "xi", cell_markers=cell_markers, valid_regions=reg_xi )
    save_sampled_interface_indicator_xdmf(V, xi_fun, eta_funs, p, out_dir / "interface_indicator_sampled.xdmf", "interface_indicator", cell_markers=cell_markers, valid_regions=reg_xi )
    save_sampled_field_xdmf(V, phil_fun, out_dir / "phil_sampled.xdmf", "phil", cell_markers=cell_markers, valid_regions=reg_phil )
    save_sampled_field_xdmf(V, phis_fun, out_dir / "phis_sampled.xdmf", "phis", cell_markers=cell_markers, valid_regions=reg_phis )
    save_sampled_field_xdmf(V, c_fun, out_dir / "c_soc_sampled.xdmf", "c_soc", cell_markers=cell_markers, valid_regions=reg_cathode )
    c_full_fun = fem.Function(c_fun.function_space, name=f"{prefix}_c_soc_full")
    c_full_fun.x.array[:] = c_fun.x.array
    valid_c = restricted_dofs(c_fun.function_space, cell_markers, reg_cathode)
    keep_c = np.zeros(c_full_fun.x.array.shape[0], dtype=bool)
    keep_c[valid_c] = True
    c_full_fun.x.array[~keep_c] = PETSc.ScalarType(0.0)
    c_full_fun.x.scatter_forward()
    save_sampled_field_xdmf(V, c_full_fun, out_dir / "c_soc_full_sampled.xdmf", "c_soc_full", cell_markers=cell_markers, valid_regions=None )
    eeq_eff_smooth_fun = fem.Function(eeq_eff_fun.function_space, name="eeq_eff_smooth")
    eeq_eff_smooth_fun.x.array[:] = (
        hydro_cathode_smooth_fun.x.array.real * p.omega_cathode / p.F
    ).astype(eeq_eff_smooth_fun.x.array.dtype)
    eeq_eff_smooth_fun.x.scatter_forward()
    save_sampled_field_xdmf(V, eeq_eff_smooth_fun, out_dir / "eeq_eff_sampled.xdmf", "eeq_eff", cell_markers=cell_markers, valid_regions=reg_cathode )
    save_sampled_field_xdmf(V, hydro_fun, out_dir / "hydrostatic_stress_omega1_sampled.xdmf", "hydrostatic_stress_omega1", cell_markers=cell_markers, valid_regions=reg_xi )
    save_sampled_field_xdmf(V, hydro_omega1_smooth_fun, out_dir / "hydrostatic_stress_omega1_smooth_sampled.xdmf", "hydrostatic_stress_omega1_smooth", cell_markers=cell_markers, valid_regions=reg_xi )
    save_sampled_field_xdmf(V, hydro_omega12_smooth_fun, out_dir / "hydrostatic_stress_omega12_smooth_sampled.xdmf", "hydrostatic_stress_omega12_smooth", cell_markers=cell_markers, valid_regions=OMEGA1_REGIONS + (OMEGA2,) )
    save_sampled_field_xdmf(V, hydro_cathode_smooth_fun, out_dir / "hydrostatic_stress_cathode_smooth_sampled.xdmf", "hydrostatic_stress_cathode_smooth", cell_markers=cell_markers, valid_regions=reg_cathode )
    save_sampled_field_xdmf(V, hydro_omega23_smooth_fun, out_dir / "hydrostatic_stress_omega23_smooth_sampled.xdmf", "hydrostatic_stress_omega23_smooth", cell_markers=cell_markers, valid_regions=(OMEGA2, OMEGA3) )
    if prefix == "initial":
        save_sampled_field_xdmf(V, B, out_dir / "B_sampled.xdmf", "B", cell_markers=cell_markers, valid_regions=reg_xi )
    elif prefix in ("final", "after_charge", "after_discharge"):
        save_sampled_field_xdmf(V, B, out_dir / "B_sampled.xdmf", "B", cell_markers=cell_markers, valid_regions=reg_xi )


def save_delta_outputs(out_dir, V, initial_outputs, final_outputs, B, cell_markers):
    # PNG delta previews are disabled in this loop-friendly runner.
    return


def append_csv(path, row, write_header=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def field_stats(fun, dofs=None):
    comm = fun.function_space.mesh.comm
    owned_size = fun.function_space.dofmap.index_map.size_local
    if dofs is None:
        values = fun.x.array.real[:owned_size]
    else:
        dofs = np.asarray(dofs, dtype=np.int64)
        dofs = dofs[(dofs >= 0) & (dofs < owned_size)]
        values = fun.x.array.real[dofs]
    if values.size == 0:
        local_min = math.inf
        local_max = -math.inf
        local_sum = 0.0
        local_count = 0
    else:
        local_min = float(values.min())
        local_max = float(values.max())
        local_sum = float(values.sum())
        local_count = int(values.size)
    global_count = int(comm.allreduce(local_count, op=MPI.SUM))
    if global_count == 0:
        return float("nan"), float("nan"), float("nan")
    global_min = float(comm.allreduce(local_min, op=MPI.MIN))
    global_max = float(comm.allreduce(local_max, op=MPI.MAX))
    global_sum = float(comm.allreduce(local_sum, op=MPI.SUM))
    return global_min, global_max, global_sum / global_count



def dendrite_tip_y(fun, dofs=None, threshold=0.5):
    """Return the lowest y/L coordinate where xi >= threshold."""
    V = fun.function_space
    comm = V.mesh.comm
    owned_size = V.dofmap.index_map.size_local
    if dofs is None:
        dofs = np.arange(owned_size, dtype=np.int64)
    else:
        dofs = np.asarray(dofs, dtype=np.int64)
        dofs = dofs[(dofs >= 0) & (dofs < owned_size)]
    if dofs.size == 0:
        local_tip = math.inf
    else:
        vals = fun.x.array.real[dofs]
        active = dofs[vals >= float(threshold)]
        if active.size == 0:
            local_tip = math.inf
        else:
            coords = V.tabulate_dof_coordinates()[:owned_size]
            local_tip = float(np.min(coords[active, 1]))
    tip = float(comm.allreduce(local_tip, op=MPI.MIN))
    return tip if math.isfinite(tip) else float("nan")


def region_y_extent(V, dofs):
    """Return min/max y/L coordinates for a region dof set."""
    comm = V.mesh.comm
    owned_size = V.dofmap.index_map.size_local
    dofs = np.asarray(dofs, dtype=np.int64)
    dofs = dofs[(dofs >= 0) & (dofs < owned_size)]
    if dofs.size == 0:
        local_min = math.inf
        local_max = -math.inf
    else:
        coords = V.tabulate_dof_coordinates()[:owned_size]
        local_min = float(np.min(coords[dofs, 1]))
        local_max = float(np.max(coords[dofs, 1]))
    global_min = float(comm.allreduce(local_min, op=MPI.MIN))
    global_max = float(comm.allreduce(local_max, op=MPI.MAX))
    if not math.isfinite(global_min):
        global_min = float("nan")
    if not math.isfinite(global_max):
        global_max = float("nan")
    return global_min, global_max

def constant_scalar_value(constant):
    value = constant.value
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def assemble_total(expr, comm=None):
    local_value = float(fem.assemble_scalar(fem.form(expr)))
    return float(comm.allreduce(local_value, op=MPI.SUM)) if comm is not None else local_value


def assemble_form_total(form, comm=None):
    local_value = float(fem.assemble_scalar(form))
    return float(comm.allreduce(local_value, op=MPI.SUM)) if comm is not None else local_value


def save_final_npz(path, V, scalar_outputs, derived_outputs, params, extra_outputs=()):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for fun in list(scalar_outputs) + list(derived_outputs) + list(extra_outputs):
        arrays[fun.name] = fun.x.array.real.copy()
    arrays["dof_coordinates"] = V.tabulate_dof_coordinates()[:, :2]
    arrays["params"] = np.array(str(asdict(params)))
    np.savez(path, **arrays)


def _field_output_variable_set(variables):
    if variables is None:
        return {"all"}
    if isinstance(variables, str):
        variables = tuple(part.strip() for part in variables.split(","))
    names = {str(name).strip() for name in variables if str(name).strip()}
    return names or {"all"}


def _triangulate_plot_cells(cells):
    if cells.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    return (
        cells
        if cells.shape[1] == 3
        else np.vstack((cells[:, [0, 1, 2]], cells[:, [0, 2, 3]])).astype(np.int32)
    )


def _png_style_field_payload(
    V,
    field,
    cell_markers,
    valid_regions=None,
    fill_outside=None,
    overlay=None,
):
    """Return the same plotting payload used by save_field_png()."""
    comm = V.mesh.comm
    values = field.x.array.real.copy()
    cells_all, points, markers = cells_and_points(V, cell_markers, owned_only=True)
    if fill_outside is None:
        if valid_regions is None or markers is None:
            cells = cells_all
        else:
            regions = (
                (valid_regions,)
                if isinstance(valid_regions, (int, np.integer))
                else tuple(valid_regions)
            )
            cells = cells_all[np.isin(markers, regions)]
        plot_dofs = (
            np.unique(cells.reshape(-1)).astype(np.int32)
            if cells.size
            else np.empty(0, dtype=np.int32)
        )
        clim_values = values[plot_dofs].copy() if plot_dofs.size else np.empty(0)
    else:
        if valid_regions is not None and markers is not None:
            regions = (
                (valid_regions,)
                if isinstance(valid_regions, (int, np.integer))
                else tuple(valid_regions)
            )
            keep_cells = np.isin(markers, regions)
            valid_dofs = (
                np.unique(cells_all[keep_cells].reshape(-1))
                if np.any(keep_cells)
                else np.empty(0, dtype=np.int32)
            )
            invalid = np.ones(values.shape[0], dtype=bool)
            invalid[valid_dofs] = False
            values[invalid] = float(fill_outside)
            clim_values = values[valid_dofs].copy()
        else:
            clim_values = values.copy()
        cells = cells_all
    triangles = _triangulate_plot_cells(cells)
    overlay_values = (
        overlay.x.array.real.copy() if overlay is not None else np.empty(0, dtype=np.float64)
    )
    gathered = _gather_png_payload(comm, points, triangles, values, overlay_values, clim_values)
    if gathered is None:
        return None
    points, triangles, values, overlay_values, clim_values = gathered
    if triangles.size == 0:
        return None
    return {
        "points": points,
        "cells": triangles.astype(np.int32, copy=False),
        "values": values,
        "overlay_values": overlay_values,
        "clim_values": clim_values,
    }


def _png_style_ce_payload(V, xi_fun, eta_funs, p: Params, cell_markers, valid_regions, order: int = 5):
    """Return the same sampled interface-indicator payload used by the PNG output."""
    comm = V.mesh.comm
    sample_points, sample_cells, parent_tris, sample_bary = build_subsampled_triangles(
        V,
        cell_markers=cell_markers,
        valid_regions=valid_regions,
        order=order,
        owned_only=True,
    )
    if sample_points.size == 0:
        gathered = _gather_png_payload(
            comm, sample_points, sample_cells, np.empty(0, dtype=np.float64)
        )
        if gathered is None:
            return None
        sample_points, sample_cells, ce_values = gathered
    else:
        def sample_scalar(values):
            return np.sum(sample_bary * values[parent_tris], axis=1)

        def box_indicator_np(value, half_width):
            return (np.abs(value) < half_width).astype(np.float64)

        xi_sample = sample_scalar(xi_fun.x.array.real)
        eta_samples = np.vstack([sample_scalar(eta_i.x.array.real) for eta_i in eta_funs]).T
        eta_clip = np.clip(eta_samples, 0.0, 1.0)
        gb_window = np.sum(
            (1.0 - eta_clip) * box_indicator_np(eta_samples - 0.5, p.rho),
            axis=1,
        )
        ce_values = xi_sample + (1.0 - xi_sample) ** 2 * gb_window
        gathered = _gather_png_payload(comm, sample_points, sample_cells, ce_values)
        if gathered is None:
            return None
        sample_points, sample_cells, ce_values = gathered
    if sample_points.size == 0 or sample_cells.size == 0:
        return None
    return {
        "points": sample_points,
        "cells": sample_cells.astype(np.int32, copy=False),
        "values": ce_values,
        "overlay_values": np.empty(0, dtype=np.float64),
        "clim_values": ce_values,
    }


def _png_style_regions_for_field(name):
    reg_xi = OMEGA1_REGIONS
    reg_phil = OMEGA1_REGIONS + (OMEGA2,)
    reg_phis = (OMEGA2,)
    reg_cathode = (OMEGA3,)
    if name == "xi" or name == "B" or name.startswith("eta"):
        return reg_xi
    if name == "phil":
        return reg_phil
    if name == "phis":
        return reg_phis
    if name == "c":
        return reg_cathode
    if name in ("eeq", "eeq_eff", "hydrostatic_stress_cathode", "overp_mech_cathode", "stress_flux_drive", "hydrostatic_stress_cathode_smooth"):
        return reg_cathode
    if name in ("hydrostatic_stress", "hydrostatic_stress_omega1_smooth", "overp_mech", "reaction_li", "deposition_drive", "xi_source_weight", "xit", "overp_li", "liion_source", "dfmechdxi", "heta", "E1", "nu1", "gb_window", "hxi", "sigma_eff"):
        return reg_xi
    if name in ("hydrostatic_stress_omega12_smooth",):
        return OMEGA1_REGIONS + (OMEGA2,)
    if name in ("hydrostatic_stress_omega23_smooth",):
        return (OMEGA2, OMEGA3)
    return None


def save_field_snapshot(
    path,
    V,
    scalar_outputs,
    derived_outputs,
    B,
    cell_markers,
    params,
    variables=("all",),
    extra_outputs=(),
    metadata=None,
):
    """Save one PNG-style software plotting snapshot.

    Each variable is stored as `<name>_points`, `<name>_cells`, and
    `<name>_values`. These arrays are the same payload the current PNG code
    uses, including region filtering and the special sampled ce calculation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    requested = _field_output_variable_set(variables)
    save_all = "all" in requested or "*" in requested
    fields = {fun.name: fun for fun in list(scalar_outputs) + list(derived_outputs)}
    fields["B"] = B
    if "u" in requested and "u_magnitude" in fields:
        requested.add("u_magnitude")
    field_names = sorted(fields) if save_all else [name for name in sorted(fields) if name in requested]
    eta_funs = [fun for fun in scalar_outputs[6:] if fun.name.startswith("eta")]

    arrays = {
        "field_names": np.asarray(field_names, dtype=str),
        "plot_payload": np.array("png_style"),
        "params": np.array(str(asdict(params))),
    }
    saved_names = []
    skipped_names = []
    for name in field_names:
        if name == "ce":
            payload = _png_style_ce_payload(V, scalar_outputs[0], eta_funs, params, cell_markers, OMEGA1_REGIONS)
        else:
            overlay = B if name in ("xi", "B", "hydrostatic_stress", "hydrostatic_stress_omega1_smooth") else None
            payload = _png_style_field_payload(
                V,
                fields[name],
                cell_markers,
                valid_regions=_png_style_regions_for_field(name),
                fill_outside=0.0 if name == "c_soc_full" else None,
                overlay=overlay,
            )
        if payload is None:
            skipped_names.append(name)
            continue
        saved_names.append(name)
        arrays[f"{name}_points"] = payload["points"]
        arrays[f"{name}_cells"] = payload["cells"]
        arrays[f"{name}_values"] = payload["values"]
        arrays[f"{name}_clim_values"] = payload["clim_values"]
        arrays[f"{name}_overlay_values"] = payload["overlay_values"]
    arrays["field_names"] = np.asarray(saved_names, dtype=str)
    arrays["skipped_field_names"] = np.asarray(skipped_names, dtype=str)
    if metadata:
        for key, value in metadata.items():
            arrays[f"meta_{key}"] = np.array(value)
    np.savez(path, **arrays)


def assign_constant(constant, value: float):
    try:
        constant.value = PETSc.ScalarType(value)
    except TypeError:
        constant.value[...] = PETSc.ScalarType(value)


def nonlinear_solver_status(problem):
    solver = getattr(problem, "solver", None)
    if solver is None:
        return "unknown", "unknown", float("nan")
    try:
        reason = solver.getConvergedReason()
    except Exception:
        reason = "unknown"
    try:
        iterations = solver.getIterationNumber()
    except Exception:
        iterations = "unknown"
    try:
        residual = float(solver.getFunctionNorm())
    except Exception:
        residual = float("nan")
    return reason, iterations, residual


def nonlinear_solver_converged(reason):
    try:
        return int(reason) > 0
    except Exception:
        return False


class NewtonProfiler:
    """Low-overhead CSV profiler for the hand-written Newton loop."""

    fieldnames = (
        "accepted_step",
        "global_time_s",
        "phase",
        "dt_s",
        "retry",
        "newton_iteration",
        "alpha",
        "line_search_trials",
        "initial_residual",
        "previous_residual",
        "trial_residual",
        "step_norm",
        "residual_total_s",
        "residual_assemble_s",
        "residual_lifting_bc_s",
        "residual_projection_s",
        "jacobian_reused",
        "jacobian_age",
        "jacobian_total_s",
        "jacobian_assemble_s",
        "active_submatrix_s",
        "constraint_matmult_s",
        "linear_solve_s",
        "linear_iterations",
        "linear_reason",
        "solution_update_s",
        "solve_step_total_s",
        "line_search_total_s",
        "iteration_total_s",
        "time_error",
        "time_error_xi",
        "time_error_c",
        "time_adapt_factor",
        "time_adapt_dt_next",
        "time_adapt_reject_limit",
        "time_adapt_rejected",
    )

    def __init__(self, enabled: bool, path: str | Path | None, comm):
        self.enabled = bool(enabled)
        self.path = Path(path) if path is not None else None
        self.time_adapt_path = (
            self.path.with_name(f"{self.path.stem}_time_adapt{self.path.suffix}")
            if self.path is not None
            else None
        )
        self.comm = comm
        self.context = {
            "accepted_step": -1,
            "global_time_s": float("nan"),
            "phase": "",
            "dt_s": float("nan"),
            "retry": -1,
        }
        if self.enabled and self.path is not None and self.comm.rank == 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()
            if self.time_adapt_path is not None:
                with self.time_adapt_path.open("w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(
                        f,
                        fieldnames=(
                            "accepted_step",
                            "global_time_s",
                            "phase",
                            "dt_s",
                            "retry",
                            "time_error",
                            "time_error_xi",
                            "time_error_c",
                            "time_adapt_factor",
                            "time_adapt_dt_next",
                            "time_adapt_reject_limit",
                            "time_adapt_rejected",
                        ),
                    ).writeheader()

    def set_context(self, *, accepted_step, global_time_s, phase, dt_s, retry):
        self.context = {
            "accepted_step": int(accepted_step),
            "global_time_s": float(global_time_s),
            "phase": str(phase),
            "dt_s": float(dt_s),
            "retry": int(retry),
        }

    def write_iteration(self, row):
        if not self.enabled:
            return
        full_row = dict.fromkeys(self.fieldnames, "")
        full_row.update(self.context)
        full_row.update(row)
        if self.comm.rank == 0:
            if self.path is not None:
                with self.path.open("a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=self.fieldnames).writerow(full_row)
            print(
                "newton_profile "
                f"step={full_row['accepted_step']} "
                f"retry={full_row['retry']} "
                f"it={full_row['newton_iteration']} "
                f"alpha={full_row['alpha']:.3g} "
                f"res={full_row['trial_residual']:.3e} "
                f"J={full_row['jacobian_total_s']:.3f}s "
                f"reduce={full_row['constraint_matmult_s']:.3f}s "
                f"linear={full_row['linear_solve_s']:.3f}s "
                f"ls={full_row['line_search_total_s']:.3f}s "
                f"iter={full_row['iteration_total_s']:.3f}s"
            )

    def write_time_adapt(self, row):
        if not self.enabled:
            return
        full_row = dict.fromkeys(self.fieldnames, "")
        full_row.update(self.context)
        full_row.update(
            {
                "newton_iteration": 0,
                "alpha": "",
                "line_search_trials": "",
            }
        )
        full_row.update(row)
        if self.comm.rank == 0:
            if self.path is not None:
                with self.path.open("a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=self.fieldnames).writerow(full_row)
            if self.time_adapt_path is not None:
                with self.time_adapt_path.open("a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(
                        f,
                        fieldnames=(
                            "accepted_step",
                            "global_time_s",
                            "phase",
                            "dt_s",
                            "retry",
                            "time_error",
                            "time_error_xi",
                            "time_error_c",
                            "time_adapt_factor",
                            "time_adapt_dt_next",
                            "time_adapt_reject_limit",
                            "time_adapt_rejected",
                        ),
                    ).writerow(
                        {
                            "accepted_step": full_row["accepted_step"],
                            "global_time_s": full_row["global_time_s"],
                            "phase": full_row["phase"],
                            "dt_s": full_row["dt_s"],
                            "retry": full_row["retry"],
                            "time_error": full_row["time_error"],
                            "time_error_xi": full_row["time_error_xi"],
                            "time_error_c": full_row["time_error_c"],
                            "time_adapt_factor": full_row["time_adapt_factor"],
                            "time_adapt_dt_next": full_row["time_adapt_dt_next"],
                            "time_adapt_reject_limit": full_row["time_adapt_reject_limit"],
                            "time_adapt_rejected": full_row["time_adapt_rejected"],
                        }
                    )
            print(
                "time_adapt_profile "
                f"step={full_row['accepted_step']} "
                f"retry={full_row['retry']} "
                f"dt={float(full_row['dt_s']):.3e} "
                f"E={float(full_row['time_error']):.3e} "
                f"Exi={float(full_row['time_error_xi']):.3e} "
                f"Ec={float(full_row['time_error_c']):.3e} "
                f"factor={float(full_row['time_adapt_factor']):.3g} "
                f"dt_next={float(full_row['time_adapt_dt_next']):.3e} "
                f"rejected={full_row['time_adapt_rejected']}"
            )


class RestrictedNewtonStatus:
    """Small SNES-like status object used by nonlinear_solver_status()."""

    def __init__(self):
        self.reason = 0
        self.iterations = 0
        self.function_norm = float("nan")

    def getConvergedReason(self):
        return self.reason

    def getIterationNumber(self):
        return self.iterations

    def getFunctionNorm(self):
        return self.function_norm


class RestrictedNewtonProblem:
    """Newton solver with optional exact serial slave-master constraints.

    The original monolithic UFL residual and Jacobian are assembled on the full
    space.  Inactive dofs are removed from the Newton correction.  Periodic
    slave dofs are handled by a prolongation matrix P, solving
    ``P.T * A * P * du_r = P.T * r`` and prolonging ``du = P * du_r``.  This
    merges slave residual rows/columns into the master dofs instead of simply
    dropping the slave equations.

    Exact slave-master periodic constraints are kept serial.  MPI runs use
    owned global dof numbers for the active PETSc IS when no periodic
    constraints are present.
    """

    def __init__(
        self,
        F,
        J,
        u,
        bcs,
        active_dofs,
        periodic_constraints=None,
        periodic_constraints_global=None,
        *,
        rtol,
        atol,
        stol,
        max_it,
        monitor=False,
        profiler=None,
        linear_solver="lu",
        linear_rtol=1.0e-8,
        linear_atol=1.0e-12,
        linear_max_it=500,
        jacobian_lag=1,
        reuse_jacobian_across_solves=False,
        fast_residual_bc=False,
        jit_options=None,
    ):
        self.u = u
        self.comm = u.function_space.mesh.comm
        self.bcs = list(bcs)
        self.periodic_constraints = dict(periodic_constraints or {})
        self.periodic_constraints_global = dict(periodic_constraints_global or {})
        if self.comm.size > 1 and self.periodic_constraints and not self.periodic_constraints_global:
            raise RuntimeError("MPI periodic solve requires global periodic constraints.")
        self.periodic_slaves = np.asarray(
            sorted(self.periodic_constraints), dtype=np.int32
        )
        self.periodic_slaves_global = np.asarray(
            sorted(self.periodic_constraints_global), dtype=PETSc.IntType
        )
        self.F_form = fem.form(F, jit_options=jit_options)
        self.J_form = fem.form(J, jit_options=jit_options)
        try:
            self.b = create_vector(self.F_form)
        except TypeError:
            # Some DOLFINx versions expose create_vector(function_spaces)
            # rather than create_vector(linear_form). Assemble once to obtain a
            # compatible PETSc Vec, then zero it before every residual assembly.
            self.b = assemble_vector(self.F_form)
            with self.b.localForm() as loc_b:
                loc_b.set(0.0)
        self.A = create_matrix(self.J_form)
        active_input = np.asarray(active_dofs, dtype=np.int64)
        self.P = None
        self.P_is_distributed = False
        self.reduced_dofs = None
        self.reduced_is = None
        if self.comm.size == 1:
            self.active_dofs = active_input.astype(PETSc.IntType)
            self.active_local_dofs = self.active_dofs.astype(np.int32)
            self._build_constraint_prolongation()
            self.active_is = PETSc.IS().createGeneral(
                self.active_dofs, comm=self.comm
            )
            self.n_active_local = int(self.active_dofs.size)
            self.n_active_global = self.n_active_local
        else:
            local_size = self.u.x.petsc_vec.getLocalSize()
            active_local = active_input[
                (0 <= active_input) & (active_input < local_size)
            ]
            active_local = np.unique(active_local).astype(np.int32)
            index_map = self.u.function_space.dofmap.index_map
            index_map_bs = int(self.u.function_space.dofmap.index_map_bs)
            owned_size_from_map = int(index_map.size_local) * index_map_bs
            if owned_size_from_map != local_size:
                raise RuntimeError(
                    "RestrictedNewtonProblem expected PETSc local vector size to "
                    f"match the function-space owned dof map ({local_size} != "
                    f"{owned_size_from_map})."
                )
            ownership_start, _ownership_end = self.u.x.petsc_vec.getOwnershipRange()
            active_global = (int(ownership_start) + active_local.astype(np.int64)).astype(
                PETSc.IntType
            )
            self.active_dofs = active_local.astype(PETSc.IntType)
            self.active_local_dofs = active_local
            self.active_global_dofs = active_global
            self.active_is = PETSc.IS().createGeneral(
                self.active_global_dofs, comm=self.comm
            )
            self.n_active_local = int(self.active_local_dofs.size)
            self.n_active_global = int(
                self.comm.allreduce(self.n_active_local, op=MPI.SUM)
            )
            self._build_distributed_constraint_prolongation()
        self.du = np.zeros_like(self.u.x.array)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.stol = float(stol)
        self.max_it = int(max_it)
        self.monitor = bool(monitor)
        self.profiler = profiler
        self.solver = RestrictedNewtonStatus()
        self.ksp = PETSc.KSP().create(self.comm)
        self.linear_solver = str(linear_solver).lower()
        self.linear_rtol = float(linear_rtol)
        self.linear_atol = float(linear_atol)
        self.linear_max_it = int(linear_max_it)
        self.jacobian_lag = max(1, int(jacobian_lag))
        self.reuse_jacobian_across_solves = bool(reuse_jacobian_across_solves)
        self.fast_residual_bc = bool(fast_residual_bc)
        self._cached_operator = None
        self._cached_aux_operator = None
        self._jacobian_age = self.jacobian_lag
        self._force_rebuild_jacobian = False
        self._configure_ksp()
        self.ksp.setFromOptions()

    def _configure_ksp(self):
        pc = self.ksp.getPC()
        mode = self.linear_solver
        if mode == "lu":
            self.ksp.setType("preonly")
            pc.setType("lu")
            return
        if mode == "mumps":
            self.ksp.setType("preonly")
            pc.setType("lu")
            try:
                pc.setFactorSolverType("mumps")
            except PETSc.Error:
                if self.comm.rank == 0:
                    print("linear_solver=mumps unavailable; falling back to plain LU.")
            return

        if mode.startswith("fgmres"):
            self.ksp.setType("fgmres")
        else:
            self.ksp.setType("gmres")
        self.ksp.setTolerances(
            rtol=self.linear_rtol,
            atol=self.linear_atol,
            max_it=self.linear_max_it,
        )
        if mode.endswith("_hypre"):
            pc.setType("hypre")
        elif mode.endswith("_gamg"):
            pc.setType("gamg")
        elif mode.endswith("_bjacobi"):
            pc.setType("bjacobi")
        elif mode.endswith("_jacobi"):
            pc.setType("jacobi")
        elif mode.endswith("_ilu"):
            pc.setType("ilu")
        else:
            pc.setType("ilu" if self.comm.size == 1 else "bjacobi")

    def _build_constraint_prolongation(self):
        self.P = None
        self.P_is_distributed = False
        self.reduced_dofs = self.active_dofs
        if not self.periodic_constraints:
            return

        active_set = set(int(d) for d in self.active_dofs)
        slave_set = set(int(d) for d in self.periodic_slaves)
        reduced_dofs = np.asarray(
            [int(d) for d in self.active_dofs if int(d) not in slave_set],
            dtype=PETSc.IntType,
        )
        reduced_index = {int(dof): i for i, dof in enumerate(reduced_dofs)}
        n_full = int(self.u.x.array.size)
        n_red = int(reduced_dofs.size)

        P = PETSc.Mat().createAIJ(
            size=(n_full, n_red),
            nnz=1,
            comm=self.u.function_space.mesh.comm,
        )
        for dof in reduced_dofs:
            P.setValue(int(dof), reduced_index[int(dof)], 1.0)
        for slave, master in self.periodic_constraints.items():
            slave = int(slave)
            master = int(master)
            if slave in active_set and master not in reduced_index:
                raise RuntimeError(
                    f"Periodic slave dof {slave} maps to inactive or slave master dof {master}."
                )
            if slave in active_set:
                P.setValue(slave, reduced_index[master], 1.0)
        P.assemble()

        self.P = P
        self.reduced_dofs = reduced_dofs

    def _build_distributed_constraint_prolongation(self):
        self.P = None
        self.P_is_distributed = False
        self.reduced_is = self.active_is
        if not self.periodic_constraints_global:
            return

        global_size = int(self.u.x.petsc_vec.getSize())
        local_size = int(self.u.x.petsc_vec.getLocalSize())
        active_global = np.asarray(self.active_global_dofs, dtype=np.int64)
        slave_set = set(int(d) for d in self.periodic_constraints_global)
        active_global_all = set(
            int(d)
            for part in self.comm.allgather(active_global.tolist())
            for d in part
        )
        reduced_global_local = np.asarray(
            [int(d) for d in active_global if int(d) not in slave_set],
            dtype=PETSc.IntType,
        )
        reduced_global_all = set(
            int(d)
            for part in self.comm.allgather(reduced_global_local.astype(np.int64).tolist())
            for d in part
        )
        for slave, master in self.periodic_constraints_global.items():
            if int(slave) in active_global_all and int(master) not in reduced_global_all:
                raise RuntimeError(
                    f"MPI periodic slave dof {slave} maps to inactive/slave master dof {master}."
                )

        try:
            P = PETSc.Mat().createAIJ(
                size=((local_size, global_size), (local_size, global_size)),
                nnz=1,
                comm=self.comm,
            )
        except Exception:
            P = PETSc.Mat().createAIJ(
                size=(global_size, global_size),
                nnz=1,
                comm=self.comm,
            )

        row_start, row_end = self.u.x.petsc_vec.getOwnershipRange()
        active_owned = set(int(d) for d in active_global)
        for row in range(int(row_start), int(row_end)):
            if row not in active_owned:
                continue
            if row in self.periodic_constraints_global:
                col = int(self.periodic_constraints_global[row])
            else:
                col = row
            P.setValue(row, col, 1.0)
        P.assemble()

        self.P = P
        self.P_is_distributed = True
        self.reduced_dofs = reduced_global_local
        self.reduced_is = PETSc.IS().createGeneral(
            reduced_global_local, comm=self.comm
        )

    def _apply_periodic_constraints(self):
        if self.P_is_distributed and self.P is not None:
            src = self.u.x.petsc_vec.duplicate()
            dst = self.u.x.petsc_vec.duplicate()
            self.u.x.petsc_vec.copy(src)
            self.P.mult(src, dst)
            dst.copy(self.u.x.petsc_vec)
            src.destroy()
            dst.destroy()
            self.u.x.scatter_forward()
            return
        if self.periodic_constraints:
            arr = self.u.x.array
            for slave, master in self.periodic_constraints.items():
                arr[slave] = arr[master]
        self.u.x.scatter_forward()

    def _assemble_residual(self):
        total_start = time.perf_counter()
        t0 = time.perf_counter()
        self._apply_periodic_constraints()
        constraint_apply_s = time.perf_counter() - t0
        with self.b.localForm() as loc_b:
            loc_b.set(0.0)
        t0 = time.perf_counter()
        try:
            assemble_vector(self.b, self.F_form)
        except TypeError:
            assembled = assemble_vector(self.F_form)
            self.b.array[:] = assembled.array_r
            assembled.destroy()
        residual_assemble_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        if not self.fast_residual_bc:
            apply_lifting(
                self.b,
                [self.J_form],
                [self.bcs],
                x0=[self.u.x.petsc_vec],
                alpha=-1.0,
            )
        self.b.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )
        set_bc(self.b, self.bcs, self.u.x.petsc_vec, -1.0)
        residual_lifting_bc_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        if self.P is None:
            r_active = self.b.getSubVector(self.active_is)
            norm = float(r_active.norm())
            self.b.restoreSubVector(self.active_is, r_active)
            residual_projection_s = time.perf_counter() - t0
            self._last_residual_timing = {
                "residual_total_s": time.perf_counter() - total_start,
                "residual_assemble_s": residual_assemble_s,
                "residual_lifting_bc_s": residual_lifting_bc_s + constraint_apply_s,
                "residual_projection_s": residual_projection_s,
            }
            return norm
        if self.P_is_distributed:
            b_projected = self.b.duplicate()
            self.P.multTranspose(self.b, b_projected)
            r_reduced = b_projected.getSubVector(self.reduced_is)
            norm = float(r_reduced.norm())
            b_projected.restoreSubVector(self.reduced_is, r_reduced)
            b_projected.destroy()
            residual_projection_s = time.perf_counter() - t0
            self._last_residual_timing = {
                "residual_total_s": time.perf_counter() - total_start,
                "residual_assemble_s": residual_assemble_s,
                "residual_lifting_bc_s": residual_lifting_bc_s + constraint_apply_s,
                "residual_projection_s": residual_projection_s,
            }
            return norm
        r_reduced = PETSc.Vec().createSeq(
            int(self.reduced_dofs.size), comm=self.u.function_space.mesh.comm
        )
        self.P.multTranspose(self.b, r_reduced)
        norm = float(r_reduced.norm())
        r_reduced.destroy()
        residual_projection_s = time.perf_counter() - t0
        self._last_residual_timing = {
            "residual_total_s": time.perf_counter() - total_start,
            "residual_assemble_s": residual_assemble_s,
            "residual_lifting_bc_s": residual_lifting_bc_s + constraint_apply_s,
            "residual_projection_s": residual_projection_s,
        }
        return norm

    def _assemble_jacobian(self):
        total_start = time.perf_counter()
        self._apply_periodic_constraints()
        t0 = time.perf_counter()
        self.A.zeroEntries()
        assemble_matrix(self.A, self.J_form, bcs=self.bcs)
        self.A.assemble()
        jacobian_assemble_s = time.perf_counter() - t0
        self._last_jacobian_timing = {
            "jacobian_total_s": time.perf_counter() - total_start,
            "jacobian_assemble_s": jacobian_assemble_s,
        }

    def _destroy_cached_operator(self):
        for attr in ("_cached_operator", "_cached_aux_operator"):
            obj = getattr(self, attr, None)
            if obj is not None:
                objs = obj if isinstance(obj, tuple) else (obj,)
                for item in objs:
                    try:
                        item.destroy()
                    except Exception:
                        pass
                setattr(self, attr, None)

    def reset_jacobian_cache(self):
        self._destroy_cached_operator()
        self._jacobian_age = self.jacobian_lag
        self._force_rebuild_jacobian = False

    def _solve_step(self, rebuild_jacobian=True):
        total_start = time.perf_counter()
        if rebuild_jacobian:
            self._destroy_cached_operator()
            self._assemble_jacobian()
            jacobian_reused = 0
            jacobian_age = 0
        else:
            self._last_jacobian_timing = {
                "jacobian_total_s": 0.0,
                "jacobian_assemble_s": 0.0,
            }
            jacobian_reused = 1
            jacobian_age = self._jacobian_age

        if self.P is None:
            t0 = time.perf_counter()
            if rebuild_jacobian or self._cached_operator is None:
                self._cached_operator = self.A.createSubMatrix(
                    self.active_is, self.active_is
                )
            A_active = self._cached_operator
            r_active = self.b.getSubVector(self.active_is)
            du_active = r_active.duplicate()
            active_submatrix_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            if rebuild_jacobian:
                self.ksp.setOperators(A_active)
            self.ksp.solve(r_active, du_active)
            linear_solve_s = time.perf_counter() - t0
            linear_iterations = int(self.ksp.getIterationNumber())
            linear_reason = int(self.ksp.getConvergedReason())
            t0 = time.perf_counter()
            self.du.fill(0.0)
            self.du[self.active_local_dofs] = du_active.array_r
            step_norm = float(du_active.norm())
            solution_update_s = time.perf_counter() - t0
            self.b.restoreSubVector(self.active_is, r_active)
            du_active.destroy()
            self._last_solve_timing = {
                **getattr(self, "_last_jacobian_timing", {}),
                "jacobian_reused": jacobian_reused,
                "jacobian_age": jacobian_age,
                "active_submatrix_s": active_submatrix_s,
                "constraint_matmult_s": 0.0,
                "linear_solve_s": linear_solve_s,
                "linear_iterations": linear_iterations,
                "linear_reason": linear_reason,
                "solution_update_s": solution_update_s,
                "solve_step_total_s": time.perf_counter() - total_start,
            }
            return step_norm

        if self.P_is_distributed:
            t0 = time.perf_counter()
            if rebuild_jacobian or self._cached_operator is None:
                if self.monitor:
                    mpi_stage_print(self.comm, "newton solve: before A*P")
                AP = self.A.matMult(self.P)
                if self.monitor:
                    mpi_stage_print(self.comm, "newton solve: before P.T*(A*P)")
                A_projected = self.P.transposeMatMult(AP)
                if self.monitor:
                    mpi_stage_print(self.comm, "newton solve: before reduced submatrix")
                A_reduced = A_projected.createSubMatrix(
                    self.reduced_is, self.reduced_is
                )
                self._cached_aux_operator = (AP, A_projected)
                self._cached_operator = A_reduced
            else:
                A_reduced = self._cached_operator
            b_projected = self.b.duplicate()
            if self.monitor:
                mpi_stage_print(self.comm, "newton solve: before P.T*b")
            self.P.multTranspose(self.b, b_projected)
            r_reduced = b_projected.getSubVector(self.reduced_is)
            du_reduced = r_reduced.duplicate()
            constraint_matmult_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            if rebuild_jacobian:
                self.ksp.setOperators(A_reduced)
            if self.monitor:
                mpi_stage_print(self.comm, "newton solve: before KSP solve")
            self.ksp.solve(r_reduced, du_reduced)
            if self.monitor:
                mpi_stage_print(self.comm, "newton solve: after KSP solve")
            linear_solve_s = time.perf_counter() - t0
            linear_iterations = int(self.ksp.getIterationNumber())
            linear_reason = int(self.ksp.getConvergedReason())
            t0 = time.perf_counter()
            du_projected = self.b.duplicate()
            with du_projected.localForm() as loc:
                loc.set(0.0)
            du_projected_reduced = du_projected.getSubVector(self.reduced_is)
            du_reduced.copy(du_projected_reduced)
            du_projected.restoreSubVector(self.reduced_is, du_projected_reduced)
            du_full = self.b.duplicate()
            self.P.mult(du_projected, du_full)
            self.du.fill(0.0)
            owned_size = int(du_full.getLocalSize())
            self.du[:owned_size] = du_full.array_r
            step_norm = float(du_reduced.norm())
            solution_update_s = time.perf_counter() - t0
            b_projected.restoreSubVector(self.reduced_is, r_reduced)
            b_projected.destroy()
            du_projected.destroy()
            du_full.destroy()
            du_reduced.destroy()
            self._last_solve_timing = {
                **getattr(self, "_last_jacobian_timing", {}),
                "jacobian_reused": jacobian_reused,
                "jacobian_age": jacobian_age,
                "active_submatrix_s": 0.0,
                "constraint_matmult_s": constraint_matmult_s,
                "linear_solve_s": linear_solve_s,
                "linear_iterations": linear_iterations,
                "linear_reason": linear_reason,
                "solution_update_s": solution_update_s,
                "solve_step_total_s": time.perf_counter() - total_start,
            }
            return step_norm

        t0 = time.perf_counter()
        if rebuild_jacobian or self._cached_operator is None:
            AP = self.A.matMult(self.P)
            A_reduced = self.P.transposeMatMult(AP)
            self._cached_aux_operator = AP
            self._cached_operator = A_reduced
        else:
            AP = self._cached_aux_operator
            A_reduced = self._cached_operator
        r_reduced = PETSc.Vec().createSeq(
            int(self.reduced_dofs.size), comm=self.u.function_space.mesh.comm
        )
        self.P.multTranspose(self.b, r_reduced)
        du_reduced = r_reduced.duplicate()
        constraint_matmult_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        if rebuild_jacobian:
            self.ksp.setOperators(A_reduced)
        self.ksp.solve(r_reduced, du_reduced)
        linear_solve_s = time.perf_counter() - t0
        linear_iterations = int(self.ksp.getIterationNumber())
        linear_reason = int(self.ksp.getConvergedReason())
        t0 = time.perf_counter()
        self.du.fill(0.0)
        self.du[self.reduced_dofs] = du_reduced.array_r
        for slave, master in self.periodic_constraints.items():
            self.du[int(slave)] = self.du[int(master)]
        step_norm = float(du_reduced.norm())
        solution_update_s = time.perf_counter() - t0
        r_reduced.destroy()
        du_reduced.destroy()
        self._last_solve_timing = {
            **getattr(self, "_last_jacobian_timing", {}),
            "jacobian_reused": jacobian_reused,
            "jacobian_age": jacobian_age,
            "active_submatrix_s": 0.0,
            "constraint_matmult_s": constraint_matmult_s,
            "linear_solve_s": linear_solve_s,
            "linear_iterations": linear_iterations,
            "linear_reason": linear_reason,
            "solution_update_s": solution_update_s,
            "solve_step_total_s": time.perf_counter() - total_start,
        }
        return step_norm

    def solve(self):
        self.solver.reason = 0
        self.solver.iterations = 0
        self.solver.function_norm = float("nan")
        if not self.reuse_jacobian_across_solves or self._cached_operator is None:
            self._destroy_cached_operator()
            self._jacobian_age = self.jacobian_lag
            self._force_rebuild_jacobian = False
        else:
            self._jacobian_age = min(
                self._jacobian_age, max(0, self.jacobian_lag - 1)
            )
        norm0 = self._assemble_residual()
        initial_residual_timing = dict(getattr(self, "_last_residual_timing", {}))
        self.solver.function_norm = norm0
        if self.monitor and self.comm.rank == 0:
            print(f"restricted_newton initial active_fnorm={norm0:.3e}", flush=True)
        if norm0 < self.atol:
            self.solver.reason = 2
            self.solver.iterations = 0
            return
        reference = max(norm0, 1.0)
        previous_norm = norm0
        old = self.u.x.array.copy()
        self._apply_periodic_constraints()
        old = self.u.x.array.copy()
        for it in range(1, self.max_it + 1):
            iteration_start = time.perf_counter()
            residual_before_step = previous_norm
            rebuild_jacobian = (
                self._cached_operator is None
                or self._force_rebuild_jacobian
                or self._jacobian_age >= self.jacobian_lag
            )
            if self.monitor and self.comm.rank == 0:
                print(
                    f"restricted_newton it={it} begin, "
                    f"rebuild_jacobian={int(rebuild_jacobian)}, "
                    f"active_fnorm={previous_norm:.3e}",
                    flush=True,
                )
            step_norm = self._solve_step(rebuild_jacobian=rebuild_jacobian)
            if rebuild_jacobian:
                self._jacobian_age = 1
                self._force_rebuild_jacobian = False
            else:
                self._jacobian_age += 1
            solve_timing = dict(getattr(self, "_last_solve_timing", {}))
            if step_norm < self.stol:
                self.solver.reason = 3
                self.solver.iterations = it
                self.solver.function_norm = previous_norm
                return

            accepted = False
            alpha = 1.0
            line_search_start = time.perf_counter()
            line_search_residual_timing = {}
            line_search_trials = 0
            trial_norm = float("nan")
            for _ in range(10):
                line_search_trials += 1
                self.u.x.array[:] = old - alpha * self.du
                self._apply_periodic_constraints()
                trial_norm = self._assemble_residual()
                line_search_residual_timing = dict(
                    getattr(self, "_last_residual_timing", {})
                )
                if trial_norm <= previous_norm or alpha <= 1.0e-3:
                    accepted = True
                    break
                alpha *= 0.5
            line_search_total_s = time.perf_counter() - line_search_start

            if not accepted:
                self.u.x.array[:] = old
                self._apply_periodic_constraints()
                self.solver.reason = -3
                self.solver.iterations = it
                self.solver.function_norm = previous_norm
                return

            old = self.u.x.array.copy()
            previous_norm = trial_norm
            if alpha < 1.0:
                self._force_rebuild_jacobian = True
            self.solver.function_norm = previous_norm
            self.solver.iterations = it
            if self.monitor and self.comm.rank == 0:
                print(
                    f"restricted_newton it={it}, alpha={alpha:.3g}, "
                    f"active_fnorm={previous_norm:.3e}, step={step_norm:.3e}"
                )
            if self.profiler is not None:
                profile_row = {
                    "newton_iteration": it,
                    "alpha": alpha,
                    "line_search_trials": line_search_trials,
                    "initial_residual": norm0,
                    "previous_residual": residual_before_step,
                    "trial_residual": trial_norm,
                    "step_norm": step_norm,
                    "line_search_total_s": line_search_total_s,
                    "iteration_total_s": time.perf_counter() - iteration_start,
                }
                profile_row.update(initial_residual_timing)
                profile_row.update(solve_timing)
                profile_row.update(line_search_residual_timing)
                self.profiler.write_iteration(profile_row)
            if previous_norm < self.atol or previous_norm / reference < self.rtol:
                self.solver.reason = 2
                return

        self.solver.reason = -2
        self.solver.iterations = self.max_it
        self.solver.function_norm = previous_norm


def cathode_interface_flux_terms(
    i_s, v_l_s, v_s_s, v_c_s, dS, dt_expr, p: Params
):
    """Gamma_s 上的界面反应弱形式项。

    i_s follows the paper convention:
        i_s = i0,c*(exp(-alpha*eta/Vt) - exp((1-alpha)*eta/Vt)).
    The normal n on Gamma_s points from Omega2 to Omega3. Therefore i_s > 0
    means Li enters the cathode particle, while charge has i_s < 0.
    """
    scale_c = 1.0 / (p.F * p.c_li_max * p.length_scale)
    phil_term = i_s * v_l_s * dS(GAMMA_S)
    phis_term = -i_s * v_s_s * dS(GAMMA_S)
    c_term = -scale_c * i_s * v_c_s * dS(GAMMA_S)
    return phil_term, phis_term, c_term


def main(
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
    if cathode_reaction_mode != "interface":
        raise ValueError("Only cathode_reaction_mode='interface' is implemented in this cycle script.")
    p = Params(cathode_reaction_mode=cathode_reaction_mode)
    if charge_time is not None:
        p = replace(p, charge_time=float(charge_time))
    if discharge_time is not None:
        p = replace(p, discharge_time=float(discharge_time))
    if cycle_count is not None:
        p = replace(p, cycle_count=max(1, int(cycle_count)))
    if charge_cutoff_voltage is not None:
        p = replace(p, charge_cutoff_voltage=float(charge_cutoff_voltage))
    if discharge_cutoff_voltage is not None:
        p = replace(p, discharge_cutoff_voltage=float(discharge_cutoff_voltage))
    if dt is not None:
        p = replace(p, dt=float(dt))
    if dt_min is not None:
        p = replace(p, dt_min=float(dt_min))
    if li_side_potential is not None:
        p = replace(p, li_side_potential=float(li_side_potential))
    if soc_min is not None or soc_max is not None:
        new_soc_min = float(soc_min) if soc_min is not None else p.soc_min
        new_soc_max = float(soc_max) if soc_max is not None else p.soc_max
        p = replace(p, soc_min=new_soc_min, soc_max=new_soc_max, soc_ref=0.5 * (new_soc_min + new_soc_max))
    if soc_init is not None:
        p = replace(p, soc_init=float(np.clip(soc_init, p.soc_min, p.soc_max)))
    if D_li is not None:
        p = replace(p, D_li=float(D_li))
    if i0_ref_li is not None:
        p = replace(p, i0_ref_li=float(i0_ref_li))
    if i0_c_ref is not None:
        p = replace(p, i0_c_ref=float(i0_c_ref))
    if c_li_max is not None:
        p = replace(p, c_li_max=float(c_li_max))
    if temperature is not None:
        p = replace(p, T=float(temperature))
    if initial_cathode_overpotential is not None:
        p = replace(p, initial_cathode_overpotential=float(initial_cathode_overpotential))
    if phis_init is not None:
        p = replace(p, phis_init=float(phis_init))
    if charge_current_sign is not None:
        sign = 1.0 if float(charge_current_sign) >= 0.0 else -1.0
        p = replace(p, charge_current_sign=sign)
    if discharge_current_sign is not None:
        sign = 1.0 if float(discharge_current_sign) >= 0.0 else -1.0
        p = replace(p, discharge_current_sign=sign)
    if current_abs is not None:
        p = replace(p, current_abs=abs(float(current_abs)))
    if E_li is not None:
        p = replace(p, E_li=float(E_li))
    if nu_li is not None:
        p = replace(p, nu_li=float(nu_li))
    if E_se is not None:
        p = replace(p, E_se=float(E_se), E_mix=float(E_se))
    if nu_se is not None:
        p = replace(p, nu_se=float(nu_se), nu_mix=float(nu_se))
    if E_cathode is not None:
        p = replace(p, E_cathode=float(E_cathode))
    if nu_cathode is not None:
        p = replace(p, nu_cathode=float(nu_cathode))
    if anode_compressive_load is not None:
        p = replace(p, anode_compressive_load=float(anode_compressive_load))
    if cathode_swelling_scale is not None:
        p = replace(p, cathode_swelling_scale=float(cathode_swelling_scale))
    if gamma is not None:
        p = replace(p, gamma=float(gamma))
    if k_xi is not None:
        p = replace(p, k_xi=float(k_xi))
    if W_xi is not None:
        p = replace(p, W_xi=float(W_xi))
    if L_gb is not None:
        p = replace(p, L_gb=float(L_gb))
    if W_gb is not None:
        p = replace(p, W_gb=float(W_gb))
    if k_gb is not None:
        p = replace(p, k_gb=float(k_gb))
    if li_layer_thickness is not None:
        p = replace(p, li_layer_thickness=max(0.0, float(li_layer_thickness)))
    if enforce_top_xi_bc is not None:
        p = replace(p, enforce_top_xi_bc=bool(enforce_top_xi_bc))
    if mechanics_periodic_lr is not None:
        p = replace(p, mechanics_periodic_lr=bool(mechanics_periodic_lr))
    if dt_max is not None:
        p = replace(p, dt_max=float(dt_max))
    if dt_max_charge is not None:
        p = replace(p, dt_max_charge=float(dt_max_charge))
    if dt_max_discharge is not None:
        p = replace(p, dt_max_discharge=float(dt_max_discharge))
    if dt_growth is not None:
        p = replace(p, dt_growth=float(dt_growth))
    if dt_shrink is not None:
        p = replace(p, dt_shrink=float(dt_shrink))
    if fixed_initial_steps is not None:
        p = replace(p, fixed_initial_steps=max(0, int(fixed_initial_steps)))
    if fixed_initial_dt is not None:
        p = replace(p, fixed_initial_dt=float(fixed_initial_dt))
    if time_adapt_tol_max is not None:
        p = replace(p, time_adapt_tol_max=float(time_adapt_tol_max))
    if time_adapt_tol_min is not None:
        p = replace(p, time_adapt_tol_min=float(time_adapt_tol_min))
    if time_adapt_safety is not None:
        p = replace(p, time_adapt_safety=float(time_adapt_safety))
    if time_adapt_rho_abs is not None:
        p = replace(p, time_adapt_rho_abs=float(time_adapt_rho_abs))
    if time_adapt_rho_rel is not None:
        p = replace(p, time_adapt_rho_rel=float(time_adapt_rho_rel))
    if time_adapt_factor_min is not None:
        p = replace(p, time_adapt_factor_min=float(time_adapt_factor_min))
    if time_adapt_factor_max is not None:
        p = replace(p, time_adapt_factor_max=float(time_adapt_factor_max))
    if time_adapt_reject is not None:
        p = replace(p, time_adapt_reject=bool(time_adapt_reject))
    if time_adapt_reject_factor is not None:
        p = replace(p, time_adapt_reject_factor=max(1.0, float(time_adapt_reject_factor)))
    if time_adapt_target_fraction is not None:
        p = replace(p, time_adapt_target_fraction=min(max(float(time_adapt_target_fraction), 1.0e-12), 1.0))
    if retry_recovery_steps is not None:
        p = replace(p, retry_recovery_steps=max(0, int(retry_recovery_steps)))
    if retry_recovery_growth is not None:
        p = replace(p, retry_recovery_growth=max(1.0, float(retry_recovery_growth)))
    if snes_rtol is not None:
        p = replace(p, snes_rtol=float(snes_rtol))
    if snes_atol is not None:
        p = replace(p, snes_atol=float(snes_atol))
    if snes_stol is not None:
        p = replace(p, snes_stol=float(snes_stol))
    if snes_max_it is not None:
        p = replace(p, snes_max_it=int(snes_max_it))
    if snes_monitor is not None:
        p = replace(p, snes_monitor=bool(snes_monitor))
    if png_interval is not None:
        p = replace(p, png_interval=max(0.0, float(png_interval)))
    if xdmf_interval is not None:
        p = replace(p, xdmf_interval=max(0.0, float(xdmf_interval)))
    if field_output_interval_s is not None:
        p = replace(p, field_output_interval_s=max(0.0, float(field_output_interval_s)))
    if field_output_variables is not None:
        p = replace(p, field_output_variables=tuple(_field_output_variable_set(field_output_variables)))
    if diagnostics_interval is not None:
        p = replace(p, diagnostics_interval=max(1, int(diagnostics_interval)))
    if progress_only is not None:
        p = replace(p, progress_only=bool(progress_only))
    if quiet is not None:
        p = replace(p, quiet=bool(quiet))
    if clip_eta_after_solve is not None:
        p = replace(p, clip_eta_after_solve=bool(clip_eta_after_solve))
    if mechanics_residual_scale is not None:
        p = replace(p, mechanics_residual_scale=float(mechanics_residual_scale))
    if newton_profile is not None:
        p = replace(p, newton_profile=bool(newton_profile))
    if newton_profile_file is not None:
        p = replace(p, newton_profile_file=str(newton_profile_file))
    if linear_solver is not None:
        p = replace(p, linear_solver=str(linear_solver).lower())
    if linear_rtol is not None:
        p = replace(p, linear_rtol=float(linear_rtol))
    if linear_atol is not None:
        p = replace(p, linear_atol=float(linear_atol))
    if linear_max_it is not None:
        p = replace(p, linear_max_it=int(linear_max_it))
    if jacobian_lag is not None:
        p = replace(p, jacobian_lag=max(1, int(jacobian_lag)))
    if jacobian_mode is not None:
        jacobian_mode = str(jacobian_mode).lower()
        if jacobian_mode not in ("full", "block"):
            raise ValueError("jacobian_mode must be 'full' or 'block'.")
        p = replace(p, jacobian_mode=jacobian_mode)
    if reuse_jacobian_across_steps is not None:
        p = replace(p, reuse_jacobian_across_steps=bool(reuse_jacobian_across_steps))
    if fast_residual_bc is not None:
        p = replace(p, fast_residual_bc=bool(fast_residual_bc))
    if light_diagnostics is not None:
        p = replace(p, light_diagnostics=bool(light_diagnostics))
    if extrapolate_initial_guess is not None:
        p = replace(p, extrapolate_initial_guess=bool(extrapolate_initial_guess))
    if extrapolate_max_factor is not None:
        p = replace(p, extrapolate_max_factor=max(0.0, float(extrapolate_max_factor)))
    if lifecycle_stop_enabled is not None:
        p = replace(p, lifecycle_stop_enabled=bool(lifecycle_stop_enabled))
    if lifecycle_xi_threshold is not None:
        p = replace(p, lifecycle_xi_threshold=float(lifecycle_xi_threshold))
    if lifecycle_target_margin is not None:
        p = replace(p, lifecycle_target_margin=float(lifecycle_target_margin))

    if p.progress_only:
        p = replace(p, snes_monitor=False, newton_profile=False)
    elif p.quiet:
        p = replace(p, snes_monitor=False)
    set_progress_only_terminal(p.progress_only or p.quiet, quiet=p.quiet)

    out_dir = Path(p.preview_dir)
    if MPI.COMM_WORLD.rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = Path(p.diagnostics_file)
        if diagnostics.exists():
            diagnostics.unlink()

    mpi_stage_print(MPI.COMM_WORLD, "before read_mesh")
    msh, cell_tags, facet_tags = read_mesh(msh_file, p)
    mpi_stage_print(msh.comm, "after read_mesh")
    dx = ufl.Measure("dx", domain=msh, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    dS = ufl.Measure("dS", domain=msh, subdomain_data=facet_tags)
    mpi_stage_print(msh.comm, "before cell_tag_dg0_function")
    cell_region_tag = cell_tag_dg0_function(msh, cell_tags)
    mpi_stage_print(msh.comm, "after cell_tag_dg0_function")
    dx_omega1 = dx_regions(dx, OMEGA1_REGIONS)
    eta_regions = tuple(
        tag
        for tag in ETA_REGIONS
        if msh.comm.allreduce(int(np.any(cell_tags.values == tag)), op=MPI.MAX)
    )
    if not eta_regions:
        eta_regions = (OMEGA1,)
    dx_eta = dx_regions(dx, eta_regions)

    mpi_stage_print(msh.comm, "before scalar space/domain markers")
    V_scalar = fem.functionspace(msh, ("Lagrange", 1))
    domain_markers = cell_marker_array(msh, cell_tags)
    unknown_cell_markers = int(np.count_nonzero(domain_markers < 0))
    unknown_cell_markers_global = int(
        msh.comm.allreduce(unknown_cell_markers, op=MPI.SUM)
    )
    if msh.comm.rank == 0:
        print(
            "cell marker diagnostics: "
            f"unknown owned+ghost cell markers={unknown_cell_markers_global}",
            flush=True,
        )
    mpi_stage_print(msh.comm, "after scalar space/domain markers")
    mpi_stage_print(msh.comm, "before stats_dofs")
    stats_dofs = {
        "xi": region_dofs_from_markers(V_scalar, domain_markers, OMEGA1_REGIONS),
        "phil": region_dofs_from_markers(V_scalar, domain_markers, OMEGA1_REGIONS + (OMEGA2,)),
        "phis": region_dofs_from_markers(V_scalar, domain_markers, (OMEGA2,)),
        "c": region_dofs_from_markers(V_scalar, domain_markers, (OMEGA3,)),
        "u": region_dofs_from_markers(V_scalar, domain_markers, OMEGA1_REGIONS + (OMEGA2, OMEGA3)),
        "eta": region_dofs_from_markers(V_scalar, domain_markers, eta_regions),
    }
    cathode_y_min, cathode_y_max = region_y_extent(V_scalar, stats_dofs["c"])
    omega1_y_min, omega1_y_max = region_y_extent(V_scalar, stats_dofs["xi"])
    lifecycle_target_y = omega1_y_min
    mpi_stage_print(msh.comm, "after stats_dofs")
    mpi_stage_print(msh.comm, "before boundary_probe_dofs")
    y_probe_top = float(
        msh.comm.allreduce(float(msh.geometry.x[:, 1].max()), op=MPI.MAX)
    )
    y_probe_bottom = float(
        msh.comm.allreduce(float(msh.geometry.x[:, 1].min()), op=MPI.MIN)
    )
    boundary_probe_dofs = {
        "phil_gamma_a": boundary_dofs_from_tag(
            V_scalar,
            facet_tags,
            GAMMA_A,
            lambda X: np.isclose(X[1], y_probe_top),
        ),
        "phis_gamma_c": boundary_dofs_from_tag(
            V_scalar,
            facet_tags,
            GAMMA_C,
            lambda X: np.isclose(X[1], y_probe_bottom),
        ),
    }
    mpi_stage_print(msh.comm, "after boundary_probe_dofs")
    # Boundary probe values are diagnostics only, matching COMSOL Boundary Probe style.
    # They do not impose Dirichlet conditions on phil or phis.

    mpi_stage_print(msh.comm, "before load_grain_arrays")
    eta_initial_values, _ = load_grain_arrays(V_scalar, gb_file)
    n_grains = int(eta_initial_values.shape[0])
    mpi_stage_print(msh.comm, f"after load_grain_arrays n_grains={n_grains}")

    mpi_stage_print(msh.comm, "before mixed space")
    P1 = element("Lagrange", msh.basix_cell(), 1)
    ME = fem.functionspace(msh, mixed_element([P1] * (6 + n_grains)))
    mpi_stage_print(msh.comm, "after mixed space")
    mpi_stage_print(msh.comm, "before state functions")
    w = fem.Function(ME, name="state_xi_phil_phis_c_u_eta")
    w_n = fem.Function(ME, name="state_old")
    w_nm1 = fem.Function(ME, name="state_old_old")
    w_nm2 = fem.Function(ME, name="state_old_old_old")
    mpi_stage_print(msh.comm, "after state functions")
    mpi_stage_print(msh.comm, "before UFL split/test setup")
    components = ufl.split(w)
    components_n = ufl.split(w_n)
    components_nm1 = ufl.split(w_nm1)
    xi, phil, phis, c, ux, uy = components[:6]
    xi_n, phil_n, phis_n, c_n, ux_n, uy_n = components_n[:6]
    xi_nm1, phil_nm1, phis_nm1, c_nm1, ux_nm1, uy_nm1 = components_nm1[:6]
    etas = components[6:]
    etas_n = components_n[6:]
    etas_nm1 = components_nm1[6:]
    u_vec = ufl.as_vector((ux, uy))
    tests = ufl.TestFunctions(ME)
    v_xi, v_l, v_s, v_c, v_ux, v_uy = tests[:6]
    v_etas = tests[6:]
    v_u = ufl.as_vector((v_ux, v_uy))
    dw = ufl.TrialFunction(ME)
    mpi_stage_print(msh.comm, "after UFL split/test setup")

    # ------------------------------------------------------------------
    #
    #
    # ------------------------------------------------------------------
    y_top = float(msh.comm.allreduce(float(msh.geometry.x[:, 1].max()), op=MPI.MAX))
    li_cap_bottom_y = y_top - p.li_layer_thickness / p.length_scale
    e_eq_init = eeq_from_soc_numpy(p.soc_init)
    phis_init_value = (
        p.li_side_potential + e_eq_init + p.initial_cathode_overpotential
        if p.phis_init is None
        else float(p.phis_init)
    )
    eta_a_init = p.li_side_potential
    eta_c_init = phis_init_value - p.li_side_potential - e_eq_init
    if msh.comm.rank == 0:
        print(
            f"initial potentials: phil={p.li_side_potential:.6g} V, "
            f"phis={phis_init_value:.6g} V "
            f"Eeq(soc_init)={e_eq_init:.6g} V, "
            f"eta_a_guess={eta_a_init:.6g} V, "
            f"eta_c_guess={eta_c_init:.6g} V"
        , flush=True)
    mpi_stage_print(msh.comm, "before assign xi")
    assign_component_from_expression(
        w, ME, 0, lambda X: initialize_xi_profile(X, y_top, p)
    )
    mpi_stage_print(msh.comm, "after assign xi")
    mpi_stage_print(msh.comm, "before assign phil")
    assign_component_from_expression(
        w,
        ME,
        1,
        lambda X: np.full(X.shape[1], p.li_side_potential, dtype=PETSc.ScalarType),
    )
    mpi_stage_print(msh.comm, "after assign phil")
    mpi_stage_print(msh.comm, "before assign phis")
    assign_component_from_expression(
        w, ME, 2, lambda X: np.full(X.shape[1], phis_init_value, dtype=PETSc.ScalarType)
    )
    mpi_stage_print(msh.comm, "after assign phis")
    # c 只在 Omega3 正极颗粒中有物理意义；Omega1/Omega2 中设为 0，
    mpi_stage_print(msh.comm, "before copy c")
    V_c_init, c_submap_init = collapse_mixed_subspace(ME, 3)
    c_initial_values = np.full(
        c_submap_init.shape[0],
        PETSc.ScalarType(p.soc_init),
        dtype=PETSc.ScalarType,
    )
    c_inactive_only_dofs = global_inactive_only_dofs_from_markers(
        V_c_init,
        domain_markers,
        OMEGA1_REGIONS + (OMEGA2,),
        (OMEGA3,),
    )
    if c_inactive_only_dofs.size > 0:
        c_initial_values[c_inactive_only_dofs] = PETSc.ScalarType(0.0)
    w.x.array[c_submap_init] = c_initial_values.astype(w.x.array.dtype, copy=False)
    mpi_stage_print(msh.comm, "after copy c")
    mpi_stage_print(msh.comm, "before assign ux")
    assign_component_from_expression(
        w, ME, 4, lambda X: np.zeros(X.shape[1], dtype=PETSc.ScalarType)
    )
    mpi_stage_print(msh.comm, "after assign ux")
    mpi_stage_print(msh.comm, "before assign uy")
    assign_component_from_expression(
        w, ME, 5, lambda X: np.zeros(X.shape[1], dtype=PETSc.ScalarType)
    )
    mpi_stage_print(msh.comm, "after assign uy")
    for i in range(n_grains):
        mpi_stage_print(msh.comm, f"before copy eta{i + 1}")
        copy_component_from_array(w, ME, 6 + i, eta_initial_values[i])
        mpi_stage_print(msh.comm, f"after copy eta{i + 1}")
    mpi_stage_print(msh.comm, "before initial scatter")
    w.x.scatter_forward()
    mpi_stage_print(msh.comm, "after initial scatter")
    w_n.x.array[:] = w.x.array
    w_n.x.scatter_forward()
    w_nm1.x.array[:] = w.x.array
    w_nm1.x.scatter_forward()
    w_nm2.x.array[:] = w.x.array
    w_nm2.x.scatter_forward()
    mpi_stage_print(msh.comm, "after old-state scatter")
    w_prev_array = w_n.x.array.copy()
    mpi_stage_print(msh.comm, "before xi_ref collapse")
    xi_ref = w_n.sub(0).collapse()
    mpi_stage_print(msh.comm, "after xi_ref collapse")

    dt_const = fem.Constant(msh, PETSc.ScalarType(p.dt))
    bdf_a0 = fem.Constant(msh, PETSc.ScalarType(1.0))
    bdf_a1 = fem.Constant(msh, PETSc.ScalarType(-1.0))
    bdf_a2 = fem.Constant(msh, PETSc.ScalarType(0.0))
    current_density = fem.Constant(msh, PETSc.ScalarType(-p.current_abs))
    i_app = fem.Constant(msh, PETSc.ScalarType(-p.current_abs))

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    mpi_stage_print(msh.comm, "before UFL auxiliary fields")
    # COMSOL text.m defines Xi = max(0, min(1, xi)) and uses Xi in the
    # phase-field interpolation and double-well terms. The unknown remains xi;
    # this auxiliary clipped value prevents h'(xi) and g'(xi) from amplifying
    # small overshoots outside [0, 1].
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
    # COMSOL's grain PDE is selected on the SE-grain subdomain, not on the
    # initialization Li cap.  B is computed from eta but eta itself is active
    # only on eta_regions below.
    B_expr = grain_boundary_indicator_expr(etas, p)
    gb_window = grain_boundary_window_expr(etas, p, eta_clips)
    sum_eta_sq = sum(eta_i * eta_i for eta_i in etas)
    gb_pair_sum = sum(
        etas[i] * etas[j]
        for i in range(n_grains)
        for j in range(n_grains)
        if j != i
    )
    # Hard COMSOL-style switch for electron concentration: Li-side electron
    # concentration follows xi, while the SE side only conducts through GBs.
    ce_expr = ufl.conditional(ufl.gt(xi, 0.5), xi, gb_window)

    I2 = ufl.Identity(2)
    eps_u = strain(u_vec)
    # Mechanics only distinguishes Li metal from the SSE matrix.  Grain
    # boundaries and grain interiors are the same elastic material, so heta is
    # not used in E/nu; otherwise small gaps in the eta partition make Omega1
    # spuriously soft.
    heta = sum(h_etas)
    E1 = p.E_li * hxi + p.E_se * (1.0 - hxi)
    nu1 = p.nu_li * hxi + p.nu_se * (1.0 - hxi)
    #
    #   Omega1/Omega2 follow the paper/PDF mechanics model without Li
    #   phase-field eigenstrain; only the cathode concentration field swells.
    #   eps_eig,3 = V_Li+,m * c_li_max * (c-soc_ref) / 3 * I.
    eig1 = 0.0 * I2
    eig2 = 0.0 * I2
    eig3 = (
        p.cathode_swelling_scale
        * p.omega_cathode
        * p.c_li_max
        * (c - p.soc_ref)
        / 3.0
    ) * I2
    E1_var = p.E_li * hXi + p.E_se * (1.0 - hXi)
    nu1_var = p.nu_li * hXi + p.nu_se * (1.0 - hXi)
    eig1_var = 0.0 * I2
    eps_eff1 = eps_u - eig1
    eps_eff1_var = eps_u - eig1_var
    sigma1 = plane_stress_tensor(E1, nu1, eps_u, eig1)
    sigma2 = plane_stress_tensor(p.E_mix, p.nu_mix, eps_u, eig2)
    sigma3 = plane_stress_tensor(p.E_cathode, p.nu_cathode, eps_u, eig3)
    # COMSOL text.m defines the mechanical energy source used in dfmechdxi
    # explicitly from the 2D elastic strain components:
    # 1/2*(E/(1-nu^2)*(e11^2+nu*e11*e22)
    #    +E/(1-nu^2)*(e22^2+nu*e11*e22)
    #    +E/(1+2*nu)*2*e12^2)
    def fmech_2d(E_expr, nu_expr, eps_eff_expr):
        e11 = eps_eff_expr[0, 0]
        e22 = eps_eff_expr[1, 1]
        e12 = eps_eff_expr[0, 1]
        return 0.5 * (
            E_expr / (1.0 - nu_expr**2) * (e11**2 + nu_expr * e11 * e22)
            + E_expr / (1.0 - nu_expr**2) * (e22**2 + nu_expr * e11 * e22)
            + E_expr / (1.0 + 2.0 * nu_expr) * 2.0 * e12**2
        )

    fmech1 = fmech_2d(E1_var, nu1_var, eps_eff1_var)
    xi_clip_derivative = ufl.conditional(
        ufl.And(ufl.ge(xi, 0.0), ufl.le(xi, 1.0)),
        1.0,
        0.0,
    )
    dfmech_dxi = ufl.diff(fmech1, Xi) * xi_clip_derivative
    dfmech_detas = [0.0 for _ in range(n_grains)]
    hydro1 = plane_stress_hydrostatic_stress(E1, nu1, eps_u, eig1)
    hydro2 = plane_stress_hydrostatic_stress(p.E_mix, p.nu_mix, eps_u, eig2)
    hydro3 = plane_stress_hydrostatic_stress(p.E_cathode, p.nu_cathode, eps_u, eig3)
    overp_mech_volts = hydro1 * p.omega_li / p.F
    overp_mech_c_volts = hydro3 * p.omega_cathode / p.F

    # COMSOL uses hxi = h(Xi) with Xi=max(0,min(1,xi)) for material
    # interpolation. The unknown xi itself is still used in mass balance and
    # in ce below, matching the lower-case xi expressions in text.m.
    #
    # Paper SI Eq. S10 in Omega1.  The COMSOL text.m hard
    # if(hxi>0.5) truncation creates a 0 -> sigma_Li jump on the
    # Omega11/Omega12 initialization split line and stalls the phil Newton
    # residual.  The continuous S10 form removes that numerical interface
    # artifact while keeping the COMSOL stress correction to sigma_SE.
    Sh = hydro1
    sigmaS = p.sigmal + p.sigmaS_stress_coeff * Sh
    sigma_eff = (
        p.sigmae * hxi
        + sigmaS * heta
        + p.sigma_phi * B_expr
    )
    mpi_stage_print(msh.comm, "after UFL auxiliary fields")

    #
    #   (2 theta)^alpha [2(1-theta)]^(1-alpha).
    mpi_stage_print(msh.comm, "before electrochemical interface expressions")
    theta_eps = 1.0e-4
    theta = ufl.max_value(theta_eps, ufl.min_value(1.0 - theta_eps, c))
    e_eq = eeq_from_soc(theta)
    e_eq_eff = overp_mech_c_volts
    i0_c = p.i0_c_ref * (2.0 * theta) ** p.alpha * (2.0 * (1.0 - theta)) ** (
        1.0 - p.alpha
    )
    overp_c_volts = phis - phil - e_eq - e_eq_eff
    overp_c = overp_c_volts / p.Vt
    i_cathode = i0_c * (
        ufl.exp(-p.alpha * overp_c) - ufl.exp((1.0 - p.alpha) * overp_c)
    )

    def side_is_region(region_id, side):
        return ufl.conditional(
            ufl.lt(abs(cell_region_tag(side) - float(region_id)), 0.5),
            1.0,
            0.0,
        )

    def trace_on_region(expr, region_id):
        plus_is_region = side_is_region(region_id, "+")
        minus_is_region = side_is_region(region_id, "-")
        return plus_is_region * expr("+") + minus_is_region * expr("-")

    phil_omega2_s = trace_on_region(phil, OMEGA2)
    phis_omega2_s = trace_on_region(phis, OMEGA2)
    v_l_omega2_s = trace_on_region(v_l, OMEGA2)
    v_s_omega2_s = trace_on_region(v_s, OMEGA2)
    c_omega3_s = trace_on_region(c, OMEGA3)
    v_c_omega3_s = trace_on_region(v_c, OMEGA3)
    e_eq_omega3_s = trace_on_region(e_eq, OMEGA3)
    overp_mech_c_omega3_s = trace_on_region(overp_mech_c_volts, OMEGA3)

    electrolyte_side = GAMMA_S_OMEGA2_SIDE
    cathode_side = GAMMA_S_OMEGA3_SIDE
    theta_s = ufl.max_value(
        theta_eps,
        ufl.min_value(1.0 - theta_eps, c_omega3_s),
    )
    overp_c_volts_s = (
        phis_omega2_s
        - phil_omega2_s
        - e_eq_omega3_s
        - overp_mech_c_omega3_s
    )
    overp_c_s = overp_c_volts_s / p.Vt
    i0_c_s = p.i0_c_ref * (2.0 * theta_s) ** p.alpha * (2.0 * (1.0 - theta_s)) ** (
        1.0 - p.alpha
    )
    i_cathode_s = i0_c_s * (
        ufl.exp(-p.alpha * overp_c_s) - ufl.exp((1.0 - p.alpha) * overp_c_s)
    )

    #
    # The phase-field anode reaction uses the Li/Li+ reference, so its
    # equilibrium potential is 0 V.  The cathode interface reaction below
    # is the branch that subtracts Eeq(c/cmax).
    overp_li_volts = phil - overp_mech_volts
    overp_li = overp_li_volts / p.Vt
    reaction_li_raw = ufl.exp((1.0 - p.alpha) * overp_li) - ufl.min_value(
        ce_expr, 1.0
    ) * ufl.exp(-p.alpha * overp_li)
    # Paper S13 uses -L_eta*(exp((1-alpha)*eta/Vt) - ce*exp(-alpha*eta/Vt));
    # therefore negative anode overpotential during charge should increase xi.
    deposition_drive = -reaction_li_raw
    xi_source_weight = hp(xi_clip)
    localized_reaction_li = xi_source_weight * deposition_drive
    def bdf_time_derivative(u, u_n, u_nm1):
        return (bdf_a0 * u + bdf_a1 * u_n + bdf_a2 * u_nm1) / dt_const

    xit_expr = bdf_time_derivative(xi, xi_n, xi_nm1)
    liion_source_expr = p.F * (1.0 / p.omega_li) * xit_expr
    cathode_phil_term, cathode_phis_term, cathode_c_term = cathode_interface_flux_terms(
        i_cathode_s, v_l_omega2_s, v_s_omega2_s, v_c_omega3_s, dS, dt_const, p
    )
    mpi_stage_print(msh.comm, "after electrochemical interface expressions")

    # ------------------------------------------------------------------
    #
    # ------------------------------------------------------------------

    #
    #   (xi^{n+1}-xi^n)/dt
    #   = -L_xi^sigma * deltaG/dxi
    #     - L_eta*h'(Xi)*BV_anode.
    #
    mpi_stage_print(msh.comm, "before residual blocks")
    F_xi = (
        bdf_time_derivative(xi, xi_n, xi_nm1) / p.L_sigma * v_xi * dx_omega1
        + xi_anisotropic_gradient_term(xi, v_xi, p) * dx_omega1
        + p.W_b * df_dxi * v_xi * dx_omega1
        + dfmech_dxi * v_xi * dx_omega1
        # COMSOL final xi source uses 2*W_b*xi*sum_i eta_i^2 with raw xi/eta.
        + 2.0 * p.W_b * xi * sum_eta_sq * v_xi * dx_omega1
        - p.xi_mobility_scale
        * (p.L_eta / p.L_sigma)
        * localized_reaction_li
        * v_xi
        * dx_omega1
    )

    F_phil = (
        (sigma_eff / p.length_scale) * ufl.dot(ufl.grad(phil), ufl.grad(v_l)) * dx_omega1
        + (p.sse / p.length_scale) * ufl.dot(ufl.grad(phil), ufl.grad(v_l)) * dx(OMEGA2)
        + p.length_scale * liion_source_expr * v_l * dx_omega1
        + cathode_phil_term
    )

    #
    #   div(-sigma_s grad(phi_s)) = 0.
    #
    F_phis = (
        (p.sigma_cathode / p.length_scale) * ufl.dot(ufl.grad(phis), ufl.grad(v_s)) * dx(OMEGA2)
        + cathode_phis_term
        - i_app * v_s * ds(GAMMA_C)
    )

    #
    #   dc/dt = div(D grad c)
    #           - div(D*c*V_Li+,m/(RT) grad(sigma_h))
    #
    F_c = (
        bdf_time_derivative(c, c_n, c_nm1) * v_c * dx(OMEGA3)
        + (p.D_li / p.length_scale**2)
        * ufl.dot(ufl.grad(c), ufl.grad(v_c))
        * dx(OMEGA3)
        - (p.D_li * p.omega_cathode / (p.R * p.T * p.length_scale**2))
        * c
        * ufl.dot(ufl.grad(hydro3), ufl.grad(v_c))
        * dx(OMEGA3)
        + cathode_c_term
    )

    #
    #   deta_i/dt = -L_phi^sigma * deltaG/deta_i.
    #
    eta_grad_coeff = p.eta_mobility_scale * p.L_gb * p.k_gb / (p.length_scale**2)
    eta_chem_coeff = p.eta_mobility_scale * p.L_gb * p.W_gb
    F_eta = 0
    for i in range(n_grains):
        eta_i = etas[i]
        eta_i_n = etas_n[i]
        v_eta_i = v_etas[i]
        dfmech_deta_i = dfmech_detas[i]
        cross = sum(etas[j] ** 2 for j in range(n_grains) if j != i)
        # COMSOL final eta equation:
        # -L_phi*(-W_phi*eta_i + W_phi*eta_i^3
        #         + 2*W_phi*eta_i*sum_{j!=i} eta_j^2
        #         + 2*W_phi*eta_i*xi^2 + dfmechdeta_i).
        dF_deta_i = -eta_i + eta_i**3 + 2.0 * eta_i * cross
        F_eta += (
            bdf_time_derivative(eta_i, eta_i_n, etas_nm1[i]) * v_eta_i * dx_eta
            + eta_grad_coeff * ufl.dot(ufl.grad(eta_i), ufl.grad(v_eta_i)) * dx_eta
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

    #
    #   div(sigma)=0.
    #
    anode_traction = ufl.as_vector((0.0, p.anode_compressive_load))
    F_mech = p.mechanics_residual_scale * (
        ufl.inner(sigma1, strain(v_u)) * dx_omega1
        + ufl.inner(sigma2, strain(v_u)) * dx(OMEGA2)
        + ufl.inner(sigma3, strain(v_u)) * dx(OMEGA3)
        - ufl.dot(anode_traction, v_u) * ds(GAMMA_A)
    )
    mpi_stage_print(msh.comm, "after residual blocks")

    # Inactive variables are handled by strong Dirichlet constraints below.
    # This avoids the tiny inactive-variable penalty block that can make the Jacobian
    # ill-conditioned.
    mpi_stage_print(msh.comm, "before F_total/J")
    F_total = F_xi + F_phil + F_phis + F_c + F_eta + F_mech
    if p.jacobian_mode == "full":
        J = ufl.derivative(F_total, w, dw)
    else:
        dw_components = ufl.split(dw)
        zero_direction = 0 * dw_components[0]

        def block_direction(active_components):
            active_components = set(active_components)
            return ufl.as_vector(
                [
                    dw_components[i] if i in active_components else zero_direction
                    for i in range(6 + n_grains)
                ]
            )

        # Approximate COMSOL-style segregated Jacobian:
        # keep the physical residual F_total unchanged, but assemble only the
        # dominant within-block derivatives.  The strongest coupling in this
        # model is xi <-> phil through the anode BV source and the xi time
        # source in F_phil, so xi must stay in the electrochemical block.
        electro_xi_block = (0, 1, 2, 3)  # xi, phil, phis, c
        mech_block = (4, 5)  # ux, uy
        eta_block = tuple(range(6, 6 + n_grains))
        J = (
            ufl.derivative(
                F_xi + F_phil + F_phis + F_c,
                w,
                block_direction(electro_xi_block),
            )
            + ufl.derivative(F_mech, w, block_direction(mech_block))
            + ufl.derivative(F_eta, w, block_direction(eta_block))
        )
    mpi_stage_print(msh.comm, "after F_total/J")

    mpi_stage_print(msh.comm, "before inactive BCs")
    bcs = []
    inactive_region_specs = {
        "xi_eta": ((OMEGA2, OMEGA3), OMEGA1_REGIONS),
        "eta": (
            tuple(tag for tag in OMEGA1_REGIONS + (OMEGA2, OMEGA3) if tag not in eta_regions),
            eta_regions,
        ),
        "phil": ((OMEGA3,), OMEGA1_REGIONS + (OMEGA2,)),
        "phis": (OMEGA1_REGIONS + (OMEGA3,), (OMEGA2,)),
        "c": (OMEGA1_REGIONS + (OMEGA2,), (OMEGA3,)),
    }
    inactive_component_dofs = {}
    scalar_global_inactive_dofs = {
        key: global_inactive_only_dofs_from_markers(
            V_scalar, domain_markers, inactive_regions, active_regions
        )
        for key, (inactive_regions, active_regions) in inactive_region_specs.items()
        if key in ("phil", "phis", "c")
    }
    for component, key in ((0, "xi_eta"), (1, "phil"), (2, "phis"), (3, "c")):
        inactive_regions, active_regions = inactive_region_specs[key]
        if component in (1, 2, 3):
            _, submap = collapse_mixed_subspace(ME, component)
            scalar_inactive = scalar_global_inactive_dofs[key]
            inactive_dofs = submap[scalar_inactive].astype(np.int32)
        else:
            inactive_dofs = component_inactive_only_dofs_from_markers(
                ME, component, domain_markers, inactive_regions, active_regions
            )
        inactive_component_dofs[component] = inactive_dofs
        bc = component_dirichlet_bc_from_dofs(ME, component, inactive_dofs, value=0.0)
        if bc is not None:
            bcs.append(bc)
    for i in range(n_grains):
        component = 6 + i
        inactive_regions, active_regions = inactive_region_specs["eta"]
        inactive_dofs = component_inactive_only_dofs_from_markers(
            ME, component, domain_markers, inactive_regions, active_regions
        )
        inactive_component_dofs[component] = inactive_dofs
        bc = component_dirichlet_bc_from_dofs(ME, component, inactive_dofs, value=0.0)
        if bc is not None:
            bcs.append(bc)
    mpi_stage_print(msh.comm, f"after inactive BCs count={len(bcs)}")

    mpi_stage_print(msh.comm, "before top xi BC")
    if p.enforce_top_xi_bc:
        top_facets = facet_tags.find(GAMMA_A)
        if len(top_facets) == 0:
            top_facets = mesh.locate_entities_boundary(
                msh, msh.topology.dim - 1, lambda X: np.isclose(X[1], y_top)
            )
        V_xi, _ = collapse_mixed_subspace(ME, 0)
        dofs_xi_top = fem.locate_dofs_topological(
            (ME.sub(0), V_xi), msh.topology.dim - 1, top_facets
        )
        xi_top_fun = fem.Function(V_xi)
        xi_top_fun.x.array[:] = 1.0
        xi_top_fun.x.scatter_forward()
        bcs.append(fem.dirichletbc(xi_top_fun, dofs_xi_top, ME.sub(0)))
    mpi_stage_print(msh.comm, f"after top xi BC count={len(bcs)}")

    x_min = float(msh.comm.allreduce(float(msh.geometry.x[:, 0].min()), op=MPI.MIN))
    x_max = float(msh.comm.allreduce(float(msh.geometry.x[:, 0].max()), op=MPI.MAX))
    # No Dirichlet/gauge BC is imposed for phil.
    # In text.m, init1.phil = -0.1 is an initial value only. The physical outer
    # boundaries of the phil equation are natural Neumann boundaries; Gamma_s
    # reaction flux is already included through cathode_interface_flux_terms().
    # In the fully coupled system, the common potential level is closed by the
    # xi equation, Jcell=(F/Omega_Li)*(xi^{n+1}-xi^n)/dt, and the phis current BC.


    y_min = float(msh.comm.allreduce(float(msh.geometry.x[:, 1].min()), op=MPI.MIN))
    bottom_facets = facet_tags.find(GAMMA_C)
    if len(bottom_facets) == 0:
        bottom_facets = mesh.locate_entities_boundary(
            msh, msh.topology.dim - 1, lambda X: np.isclose(X[1], y_min)
        )

    # Roller support on Gamma_c: normal displacement is zero, uy=0.
    V_uy, _ = collapse_mixed_subspace(ME, 5)
    dofs_uy_bottom = fem.locate_dofs_topological(
        (ME.sub(5), V_uy), msh.topology.dim - 1, bottom_facets
    )
    uy_zero = fem.Function(V_uy)
    uy_zero.x.array[:] = 0.0
    uy_zero.x.scatter_forward()
    bcs.append(fem.dirichletbc(uy_zero, dofs_uy_bottom, ME.sub(5)))
    mpi_stage_print(msh.comm, f"after uy bottom BC count={len(bcs)}")

    mpi_stage_print(msh.comm, "before periodic constraints")
    periodic_constraints = {}
    periodic_constraints_global = {}
    periodic_constraint_counts = {}
    if p.mechanics_periodic_lr:
        if msh.comm.size > 1:
            periodic_constraints_global, periodic_constraint_counts = (
                build_lr_periodic_constraints_global(msh, ME, n_grains)
            )
            apply_global_periodic_constraints_to_function(w, periodic_constraints_global)
            apply_global_periodic_constraints_to_function(w_n, periodic_constraints_global)
            apply_global_periodic_constraints_to_function(w_nm1, periodic_constraints_global)
            apply_global_periodic_constraints_to_function(w_nm2, periodic_constraints_global)
        else:
            periodic_constraints, periodic_constraint_counts = (
                build_lr_periodic_constraints(msh, ME, n_grains)
            )
            for slave, master in periodic_constraints.items():
                w.x.array[slave] = w.x.array[master]
                w_n.x.array[slave] = w_n.x.array[master]
                w_nm1.x.array[slave] = w_nm1.x.array[master]
                w_nm2.x.array[slave] = w_nm2.x.array[master]
            w.x.scatter_forward()
            w_n.x.scatter_forward()
            w_nm1.x.scatter_forward()
            w_nm2.x.scatter_forward()
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
    mpi_stage_print(msh.comm, "after periodic constraints")

    # Numerical gauge for the horizontal rigid-body mode. With left-right
    # periodic displacement constraints, ux -> ux + constant is still free in
    # the discrete elastic block; fixing one bottom midpoint dof only defines
    # the reference displacement.
    x_mid = 0.5 * (x_min + x_max)
    V_ux_gauge, ux_gauge_submap = collapse_mixed_subspace(ME, 4)
    ux_coords = V_ux_gauge.tabulate_dof_coordinates()[:, :2]
    ux_owned = int(V_ux_gauge.dofmap.index_map.size_local)
    bottom_owned = np.flatnonzero(
        np.isclose(ux_coords[:ux_owned, 1], y_min, atol=1.0e-10)
    ).astype(np.int32)
    if bottom_owned.size > 0:
        local_idx = int(
            bottom_owned[
                np.argmin(np.abs(ux_coords[bottom_owned, 0] - x_mid))
            ]
        )
        local_best = (
            float(abs(ux_coords[local_idx, 0] - x_mid)),
            int(msh.comm.rank),
            local_idx,
        )
    else:
        local_best = (float("inf"), int(msh.comm.rank), -1)
    global_best = min(msh.comm.allgather(local_best), key=lambda item: (item[0], item[1]))
    if global_best[2] < 0:
        raise RuntimeError("Could not locate a bottom owned ux dof for the gauge BC.")
    if msh.comm.rank == global_best[1]:
        dofs_ux_bottom_mid = np.asarray(
            [int(ux_gauge_submap[int(global_best[2])])], dtype=np.int32
        )
    else:
        dofs_ux_bottom_mid = np.empty(0, dtype=np.int32)
    bc_ux_bottom_mid = component_dirichlet_bc_from_dofs(
        ME, 4, dofs_ux_bottom_mid, value=0.0
    )
    if bc_ux_bottom_mid is not None:
        bcs.append(bc_ux_bottom_mid)
    mpi_stage_print(msh.comm, f"after ux gauge BC count={len(bcs)}")

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

    jit_cache_dir = Path(tempfile.gettempdir()) / f"r2d_documented_fenics_jit_{os.getpid()}"
    jit_cache_dir.mkdir(parents=True, exist_ok=True)
    jit_options = {"cache_dir": str(jit_cache_dir), "timeout": 120}

    # Stage-1 optimization:
    # keep the exact same monolithic residual/Jacobian, but remove fixed
    # inactive dofs from the Newton linear solve.
    eliminated_dofs = [inactive_component_dofs.get(0, np.empty(0, dtype=np.int32))]
    for i in range(n_grains):
        eliminated_dofs.append(
            inactive_component_dofs.get(6 + i, np.empty(0, dtype=np.int32))
        )
    periodic_slave_dofs = np.fromiter(
        periodic_constraints.keys()
        if msh.comm.size == 1
        else periodic_constraints_global.keys(),
        dtype=np.int64,
    )
    eliminated_dofs = (
        np.unique(np.concatenate(eliminated_dofs)).astype(np.int32)
        if eliminated_dofs
        else np.empty(0, dtype=np.int32)
    )
    owned_local_size = int(w.x.petsc_vec.getLocalSize())
    all_local_dofs = np.arange(owned_local_size, dtype=np.int32)
    eliminated_owned_dofs = eliminated_dofs[
        (0 <= eliminated_dofs) & (eliminated_dofs < owned_local_size)
    ]
    active_dofs = np.setdiff1d(
        all_local_dofs, eliminated_owned_dofs, assume_unique=False
    )
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
    newton_profiler = NewtonProfiler(
        p.newton_profile,
        p.newton_profile_file,
        msh.comm,
    )
    if p.newton_profile and msh.comm.rank == 0:
        print(f"Newton profile enabled: {p.newton_profile_file}")

    problem = RestrictedNewtonProblem(
        F_total,
        J,
        w,
        bcs,
        active_dofs,
        periodic_constraints=periodic_constraints,
        periodic_constraints_global=periodic_constraints_global,
        rtol=p.snes_rtol,
        atol=p.snes_atol,
        stol=p.snes_stol,
        max_it=p.snes_max_it,
        monitor=p.snes_monitor,
        profiler=newton_profiler,
        linear_solver=p.linear_solver,
        linear_rtol=p.linear_rtol,
        linear_atol=p.linear_atol,
        linear_max_it=p.linear_max_it,
        jacobian_lag=p.jacobian_lag,
        reuse_jacobian_across_solves=p.reuse_jacobian_across_steps,
        fast_residual_bc=p.fast_residual_bc,
        jit_options=jit_options,
    )
    mpi_stage_print(msh.comm, "before output function collapse")
    B = fem.Function(V_scalar, name="B")
    scalar_outputs = []
    scalar_component_maps = []
    scalar_names = ["xi", "phil", "phis", "c", "ux", "uy"] + [
        f"eta{i + 1}" for i in range(n_grains)
    ]
    for i, name in enumerate(scalar_names):
        Vi, submap = collapse_mixed_subspace(ME, i)
        scalar_outputs.append(fem.Function(Vi, name=name))
        scalar_component_maps.append(np.asarray(submap, dtype=np.int64))
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

    derived_outputs = tuple(
        fem.Function(V_scalar, name=name)
        for name in (
            "ce",
            "gb_window",
            "hxi",
            "sigma_eff",
            "reaction_li",
            "deposition_drive",
            "xi_source_weight",
            "xit",
            "overp_li",
            "overp_c",
            "eeq",
            "hydrostatic_stress",
            "overp_mech",
            "u_magnitude",
            "liion_source",
            "dfmechdxi",
            "eeq_eff",
            "hydrostatic_stress_cathode",
            "overp_mech_cathode",
            "stress_flux_drive",
            "hydrostatic_stress_cathode_smooth",
            "hydrostatic_stress_omega23_smooth",
            "heta",
            "E1",
            "nu1",
            "hydrostatic_stress_omega1_smooth",
            "hydrostatic_stress_omega12_smooth",
        )
    )
    V_dg0 = fem.functionspace(msh, ("DG", 0))
    hydro_omega1_dg0 = fem.Function(V_dg0, name="hydrostatic_stress_omega1_DG0")
    hydro_cathode_dg0 = fem.Function(V_dg0, name="hydrostatic_stress_cathode_DG0")
    hydro_omega2_dg0 = fem.Function(V_dg0, name="hydrostatic_stress_omega2_DG0")
    hydro_omega12_dg0 = fem.Function(V_dg0, name="hydrostatic_stress_omega12_DG0")
    hydro_omega23_dg0 = fem.Function(V_dg0, name="hydrostatic_stress_omega23_DG0")
    derived_exprs = (
        ce_expr,
        gb_window,
        hxi,
        sigma_eff,
        reaction_li_raw,
        deposition_drive,
        xi_source_weight,
        xit_expr,
        overp_li_volts,
        overp_c_volts,
        e_eq,
        hydro1,
        overp_mech_volts,
        ufl.sqrt(ux * ux + uy * uy),
        liion_source_expr,
        dfmech_dxi,
        e_eq_eff,
        hydro3,
        overp_mech_c_volts,
        (p.D_li * p.omega_cathode / (p.R * p.T * p.length_scale**2))
        * c
        * ufl.sqrt(ufl.dot(ufl.grad(hydro3), ufl.grad(hydro3))),
        0.0,
        0.0,
        heta,
        E1,
        nu1,
        0.0,
        0.0,
    )
    omega1_derived_indices = (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 22, 23, 24)
    omega12_derived_indices = (26,)
    recovery_derived_indices = (20, 21, 25, 26)
    omega3_derived_indices = (9, 10, 16, 17, 18, 19, 20)
    light_diagnostic_derived_indices = (8,)
    full_diagnostic_derived_indices = (4, 5, 6, 8, 10, 11, 12, 13, 15, 22, 23, 24)

    def extra_output_fields():
        return (
            hydro_omega1_dg0,
            hydro_cathode_dg0,
            hydro_omega2_dg0,
            hydro_omega12_dg0,
            hydro_omega23_dg0,
        )

    def update_derived(indices=None, include_b=True):
        if include_b:
            update_interpolated(B, B_expr)
        update_indices = range(len(derived_outputs)) if indices is None else indices
        for idx in update_indices:
            if idx in recovery_derived_indices:
                continue
            update_interpolated(derived_outputs[idx], derived_exprs[idx])
        if indices is None or 11 in update_indices or 17 in update_indices:
            update_interpolated(hydro_omega1_dg0, hydro1)
            mask_dg0_to_regions(hydro_omega1_dg0, domain_markers, tuple(OMEGA1_REGIONS))
            update_interpolated(hydro_cathode_dg0, hydro3)
            mask_dg0_to_regions(hydro_cathode_dg0, domain_markers, (OMEGA3,))
            update_interpolated(hydro_omega2_dg0, hydro2)
            hydro_omega12_dg0.x.array[:] = hydro_omega1_dg0.x.array
            copy_dg0_on_regions(hydro_omega12_dg0, hydro_omega2_dg0, domain_markers, (OMEGA2,))
            mask_dg0_to_regions(hydro_omega12_dg0, domain_markers, OMEGA1_REGIONS + (OMEGA2,))
            hydro_omega23_dg0.x.array[:] = hydro_cathode_dg0.x.array
            copy_dg0_on_regions(hydro_omega23_dg0, hydro_omega2_dg0, domain_markers, (OMEGA2,))
            mask_dg0_to_regions(hydro_omega23_dg0, domain_markers, (OMEGA2, OMEGA3))
            recover_dg0_to_p1(derived_outputs[20], hydro_cathode_dg0, domain_markers, (OMEGA3,))
            recover_dg0_to_p1(derived_outputs[21], hydro_omega23_dg0, domain_markers, (OMEGA2, OMEGA3))
            recover_dg0_to_p1(derived_outputs[25], hydro_omega1_dg0, domain_markers, tuple(OMEGA1_REGIONS))
            recover_dg0_to_p1(derived_outputs[26], hydro_omega12_dg0, domain_markers, OMEGA1_REGIONS + (OMEGA2,))
        for idx in update_indices:
            if idx in omega1_derived_indices:
                mask_function_to_regions(
                    derived_outputs[idx], V_scalar, domain_markers, OMEGA1_REGIONS
                )
            elif idx in omega3_derived_indices:
                mask_function_to_regions(
                    derived_outputs[idx], V_scalar, domain_markers, (OMEGA3,)
                )
            elif idx in omega12_derived_indices:
                mask_function_to_regions(
                    derived_outputs[idx], V_scalar, domain_markers, OMEGA1_REGIONS + (OMEGA2,)
                )
            elif idx == 22:
                mask_function_to_regions(
                    derived_outputs[idx], V_scalar, domain_markers, (OMEGA2, OMEGA3)
                )
    def update_diagnostic_derived():
        indices = (
            light_diagnostic_derived_indices
            if p.light_diagnostics
            else full_diagnostic_derived_indices
        )
        update_derived(indices, include_b=False)

    mpi_stage_print(msh.comm, "before initial derived update")
    update_derived()
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
    gamma_s_current_form = fem.form(i_cathode_s * p.length_scale * dS(GAMMA_S))
    gamma_c_current_form = fem.form(i_app * p.length_scale * ds(GAMMA_C))
    omega1_xi_current_form = fem.form(
        (p.F / p.omega_li) * xit_expr * p.length_scale**2 * dx_omega1
    )
    overp_c_gamma_s_form = fem.form(overp_c_volts_s * p.length_scale * dS(GAMMA_S))

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

    def write_diag(step, time_s, phase, dt_used):
        xi_min, xi_max, xi_mean = field_stats(scalar_outputs[0], stats_dofs["xi"])
        c_min, c_max, c_mean = field_stats(scalar_outputs[3], stats_dofs["c"])
        phil_min, phil_max, phil_mean = field_stats(scalar_outputs[1], stats_dofs["phil"])
        phis_min, phis_max, phis_mean = field_stats(scalar_outputs[2], stats_dofs["phis"])
        phil_gamma_a_min, phil_gamma_a_max, phil_gamma_a_mean = field_stats(
            scalar_outputs[1], boundary_probe_dofs["phil_gamma_a"]
        )
        phis_gamma_c_min, phis_gamma_c_max, phis_gamma_c_mean = field_stats(
            scalar_outputs[2], boundary_probe_dofs["phis_gamma_c"]
        )
        boundary_voltage_mean = phis_gamma_c_mean - phil_gamma_a_mean
        boundary_voltage = phis_gamma_c_max - phil_gamma_a_min
        overp_li_min, overp_li_max, overp_li_mean = field_stats(
            derived_outputs[8], stats_dofs["xi"]
        )
        li_metal_mol_m2 = assemble_form_total(li_metal_form, msh.comm)
        cathode_li_mol_m2 = assemble_form_total(cathode_li_form, msh.comm)
        li_metal_gain_mol_m2 = li_metal_mol_m2 - xi_initial_mol_m2
        cathode_li_delta_mol_m2 = cathode_li_mol_m2 - c_initial_mol_m2
        cathode_li_loss_mol_m2 = -cathode_li_delta_mol_m2
        li_balance_error_mol_m2 = li_metal_gain_mol_m2 + cathode_li_delta_mol_m2
        denom = max(abs(li_metal_gain_mol_m2), abs(cathode_li_delta_mol_m2), 1.0e-30)
        loss_denom = max(abs(cathode_li_loss_mol_m2), 1.0e-30)
        # 即时守恒检测：这些量直接来自当前时间步方程中的通量/源项。
        # 网格坐标使用 x/L，所以线积分需要乘 L，面积积分需要乘 L^2，
        # 才能得到二维单位厚度下的物理量。
        gamma_s_current_A_m = assemble_form_total(gamma_s_current_form, msh.comm)
        gamma_s_reaction_current_avg_A_m2 = gamma_s_current_A_m / gamma_s_length_scale
        # COMSOL liion.bei1.er1.iloc uses the opposite sign convention from
        # the paper i_s used above for the cathode BV expression.
        gamma_s_iloc_equiv_A_m = -gamma_s_current_A_m
        gamma_s_iloc_equiv_avg_A_m2 = (
            gamma_s_iloc_equiv_A_m / gamma_s_length_scale
        )
        c_boundary_source_avg_soc_s = (
            -gamma_s_reaction_current_avg_A_m2 / (p.F * p.c_li_max * p.length_scale)
        )
        gamma_c_current_A_m = assemble_form_total(gamma_c_current_form, msh.comm)
        omega1_xi_current_A_m = assemble_form_total(omega1_xi_current_form, msh.comm)
        overp_c_gamma_s_avg = assemble_form_total(overp_c_gamma_s_form, msh.comm) / gamma_s_length_scale
        overp_c_mean = overp_c_gamma_s_avg
        external_charge_current_A_m = -gamma_c_current_A_m
        xi_plus_gamma_s_A_m = omega1_xi_current_A_m + gamma_s_current_A_m

        if p.light_diagnostics:
            u_dofs = stats_dofs["u"]
            ux_values = scalar_outputs[4].x.array.real[u_dofs]
            uy_values = scalar_outputs[5].x.array.real[u_dofs]
            if ux_values.size == 0:
                u_mag_max = float("nan")
            else:
                u_mag_max = float(np.sqrt(np.max(ux_values * ux_values + uy_values * uy_values)))
            row = {
                "step": step,
                "time_s": time_s,
                "dt_s": dt_used,
                "cycle": current_cycle_index,
                "phase": phase,
                "phase_elapsed_s": current_phase_elapsed,
                "phase_duration_s": current_phase_duration,
                "current_density_A_m2": constant_scalar_value(current_density),
                "cutoff_voltage_V": current_cutoff_voltage,
                "soc_cutoff": current_soc_cutoff,
                "cutoff_reached": current_cutoff_reached,
                "cutoff_reason": current_cutoff_reason,
                "boundary_voltage_probe_V": boundary_voltage,
                "boundary_voltage_probe_delta_V": boundary_voltage - boundary_voltage_initial,
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
                "dendrite_tip_y": current_dendrite_tip_y,
                "lifecycle_target_y": lifecycle_target_y,
                "lifecycle_target_margin": p.lifecycle_target_margin,
                "cathode_top_y": cathode_y_max,
                "lifecycle_stop_reached": current_lifecycle_stop_reached,
                "c_min": c_min,
                "c_max": c_max,
                "c_mean": c_mean,
                "phil_mean": phil_mean,
                "phis_mean": phis_mean,
                "u_magnitude_max_m": u_mag_max * p.length_scale,
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
            if msh.comm.rank == 0:
                append_csv(p.diagnostics_file, row, write_header=(step == 0))
            return row
        ux_min, ux_max, ux_mean = field_stats(scalar_outputs[4], stats_dofs["u"])
        uy_min, uy_max, uy_mean = field_stats(scalar_outputs[5], stats_dofs["u"])
        reaction_li_min, reaction_li_max, reaction_li_mean = field_stats(
            derived_outputs[4], stats_dofs["xi"]
        )
        deposition_drive_min, deposition_drive_max, deposition_drive_mean = field_stats(
            derived_outputs[5], stats_dofs["xi"]
        )
        xi_source_weight_min, xi_source_weight_max, xi_source_weight_mean = field_stats(
            derived_outputs[6], stats_dofs["xi"]
        )
        hydro1_min, hydro1_max, hydro1_mean = field_stats(
            derived_outputs[11], stats_dofs["xi"]
        )
        overp_mech_min, overp_mech_max, overp_mech_mean = field_stats(
            derived_outputs[12], stats_dofs["xi"]
        )
        dfmechdxi_min, dfmechdxi_max, dfmechdxi_mean = field_stats(
            derived_outputs[15], stats_dofs["xi"]
        )
        heta_min, heta_max, heta_mean = field_stats(derived_outputs[22], stats_dofs["xi"])
        E1_min, E1_max, E1_mean = field_stats(derived_outputs[23], stats_dofs["xi"])
        nu1_min, nu1_max, nu1_mean = field_stats(derived_outputs[24], stats_dofs["xi"])
        u_mag_min, u_mag_max, u_mag_mean = field_stats(derived_outputs[13])
        eeq_min, eeq_max, eeq_mean = field_stats(derived_outputs[10], stats_dofs["c"])
        eta_c_mean = phis_mean - phil_mean - eeq_mean
        gamma_s_avg = lambda expr: assemble_total(
            expr * p.length_scale * dS(GAMMA_S), msh.comm
        ) / gamma_s_length_scale
        gamma_s_rms = lambda expr: math.sqrt(max(gamma_s_avg(expr * expr), 0.0))
        gamma_s_l8 = lambda expr: max(gamma_s_avg((expr * expr) ** 4), 0.0) ** 0.125
        c_gamma_s_trace = c_omega3_s
        eeq_gamma_s_trace = e_eq_omega3_s
        mech_c_gamma_s_trace = overp_mech_c_omega3_s
        eta_c_gamma_s_trace = overp_c_volts_s
        i_s_gamma_s_trace = i_cathode_s
        phil_gamma_s_trace = phil_omega2_s
        phis_gamma_s_trace = phis_omega2_s
        phil_gamma_s_plus_avg = assemble_total(
            phil("+") * p.length_scale * dS(GAMMA_S), msh.comm
        ) / gamma_s_length_scale
        phil_gamma_s_minus_avg = assemble_total(
            phil("-") * p.length_scale * dS(GAMMA_S), msh.comm
        ) / gamma_s_length_scale
        phil_gamma_s_avgtrace_avg = assemble_total(
            phil(electrolyte_side) * p.length_scale * dS(GAMMA_S), msh.comm
        ) / gamma_s_length_scale
        phis_gamma_s_plus_avg = assemble_total(
            phis("+") * p.length_scale * dS(GAMMA_S), msh.comm
        ) / gamma_s_length_scale
        phis_gamma_s_minus_avg = assemble_total(
            phis("-") * p.length_scale * dS(GAMMA_S), msh.comm
        ) / gamma_s_length_scale
        phis_gamma_s_avgtrace_avg = assemble_total(
            phis(electrolyte_side) * p.length_scale * dS(GAMMA_S), msh.comm
        ) / gamma_s_length_scale
        c_gamma_s_plus_avg = assemble_total(
            c("+") * p.length_scale * dS(GAMMA_S), msh.comm
        ) / gamma_s_length_scale
        c_gamma_s_minus_avg = assemble_total(
            c("-") * p.length_scale * dS(GAMMA_S), msh.comm
        ) / gamma_s_length_scale
        c_gamma_s_avgtrace_avg = assemble_total(
            c(cathode_side) * p.length_scale * dS(GAMMA_S), msh.comm
        ) / gamma_s_length_scale
        eeq_gamma_s_plus_avg = assemble_total(
            e_eq("+") * p.length_scale * dS(GAMMA_S), msh.comm
        ) / gamma_s_length_scale
        eeq_gamma_s_minus_avg = assemble_total(
            e_eq("-") * p.length_scale * dS(GAMMA_S), msh.comm
        ) / gamma_s_length_scale
        mech_c_gamma_s_plus_avg = assemble_total(
            overp_mech_c_volts("+") * p.length_scale * dS(GAMMA_S), msh.comm
        ) / gamma_s_length_scale
        mech_c_gamma_s_minus_avg = assemble_total(
            overp_mech_c_volts("-") * p.length_scale * dS(GAMMA_S), msh.comm
        ) / gamma_s_length_scale
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
            "cycle": current_cycle_index,
            "phase": phase,
            "phase_elapsed_s": current_phase_elapsed,
            "phase_duration_s": current_phase_duration,
            "cutoff_voltage_V": current_cutoff_voltage,
            "soc_cutoff": current_soc_cutoff,
            "cutoff_reached": current_cutoff_reached,
            "voltage_cutoff_reached": current_voltage_cutoff_reached,
            "soc_cutoff_reached": current_soc_cutoff_reached,
            "cutoff_reason": current_cutoff_reason,
            "current_density_A_m2": constant_scalar_value(current_density),
            "boundary_voltage_probe_V": boundary_voltage,
            "boundary_voltage_probe_delta_V": boundary_voltage - boundary_voltage_initial,
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
            "u_magnitude_max_m": u_mag_max * p.length_scale,
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
            "phil_mean_delta_V": phil_mean - phil_initial_mean,
            "phis_min": phis_min,
            "phis_max": phis_max,
            "phis_mean": phis_mean,
            "phis_mean_delta_V": phis_mean - phis_initial_mean,
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
        if msh.comm.rank == 0:
            append_csv(p.diagnostics_file, row, write_header=(step == 0))
        console_row = dict(row)
        console_row.update(
            {
                "gamma_c_current_A_m": gamma_c_current_A_m,
                "xi_plus_gamma_s_A_m": xi_plus_gamma_s_A_m,
            }
        )
        return console_row

    def save_latest_state(reason):
        w.x.array[:] = w_n.x.array
        w.x.scatter_forward()
        update_scalar_outputs(w, scalar_outputs, p, component_maps=scalar_component_maps)
        update_derived()
        latest_npz = Path(p.final_npz_file).with_name("latest_state.npz")
        save_final_npz(
            latest_npz,
            V_scalar,
            scalar_outputs,
            derived_outputs,
            p,
            extra_outputs=extra_output_fields(),
        )
        save_png_outputs(
            out_dir,
            "latest",
            V_scalar,
            scalar_outputs,
            derived_outputs,
            B,
            domain_markers,
            p,
        )
        if msh.comm.rank == 0:
            print(f"已保存最后一个成功步状态: {latest_npz}")
            print(f"保存原因: {reason}")

    current_dendrite_tip_y = dendrite_tip_y(scalar_outputs[0], stats_dofs["xi"], p.lifecycle_xi_threshold)
    def set_bdf_coefficients(dt_used, previous_dt, use_bdf2):
        if use_bdf2 and previous_dt is not None and previous_dt > 0.0:
            r = float(dt_used) / float(previous_dt)
            assign_constant(bdf_a0, (1.0 + 2.0 * r) / (1.0 + r))
            assign_constant(bdf_a1, -(1.0 + r))
            assign_constant(bdf_a2, (r * r) / (1.0 + r))
        else:
            assign_constant(bdf_a0, 1.0)
            assign_constant(bdf_a1, -1.0)
            assign_constant(bdf_a2, 0.0)

    current_lifecycle_stop_reached = False

    mpi_stage_print(msh.comm, "before initial diagnostics")
    write_diag(0, 0.0, "initial", p.dt)
    mpi_stage_print(msh.comm, "after initial diagnostics")

    lifecycle_terminated = False
    phases = []
    for cycle_index in range(1, p.cycle_count + 1):
        if p.charge_time > 0.0:
            phases.append(
                (
                    cycle_index,
                    "charge",
                    p.charge_time,
                    p.charge_current_sign * p.current_abs,
                    p.charge_cutoff_voltage,
                )
            )
        if p.discharge_time > 0.0:
            phases.append(
                (
                    cycle_index,
                    "discharge",
                    p.discharge_time,
                    p.discharge_current_sign * p.current_abs,
                    p.discharge_cutoff_voltage,
                )
            )
    total_step = 0
    time_s = 0.0
    next_png_time = p.png_interval if p.png_interval > 0.0 else math.inf
    next_xdmf_time = p.xdmf_interval if p.xdmf_interval > 0.0 else math.inf
    next_field_time = (
        p.field_output_interval_s if p.field_output_interval_s > 0.0 else math.inf
    )
    if p.field_output_interval_s > 0.0:
        update_scalar_outputs(w, scalar_outputs, p, component_maps=scalar_component_maps)
        update_derived()
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
    _, xi_map = collapse_mixed_subspace(ME, 0)
    _, c_map = collapse_mixed_subspace(ME, 3)
    eta_components = tuple(range(6, 6 + n_grains))

    def _local_error_component(component_map, scalar_dofs, dt_np1, dt_n, dt_nm1):
        owned_scalar_size = V_scalar.dofmap.index_map.size_local
        if scalar_dofs is None:
            scalar_dofs = np.empty(0, dtype=np.int64)
        else:
            scalar_dofs = np.asarray(scalar_dofs, dtype=np.int64)
        scalar_dofs = scalar_dofs[(scalar_dofs >= 0) & (scalar_dofs < owned_scalar_size)]
        if scalar_dofs.size == 0:
            mixed_dofs = np.empty(0, dtype=np.int32)
        else:
            mixed_dofs = component_map[scalar_dofs.astype(np.int32, copy=False)]
        u_np1 = np.asarray(w.x.array[mixed_dofs].real, dtype=np.float64)
        u_n = np.asarray(w_n.x.array[mixed_dofs].real, dtype=np.float64)
        u_nm1 = np.asarray(w_nm1.x.array[mixed_dofs].real, dtype=np.float64)
        u_nm2 = np.asarray(w_nm2.x.array[mixed_dofs].real, dtype=np.float64)
        if u_np1.size == 0 or dt_np1 <= 0.0 or dt_n <= 0.0 or dt_nm1 <= 0.0:
            local_pair = np.array([0.0, 0.0], dtype=np.float64)
        else:
            r = float(dt_np1) / float(dt_n)
            d0 = (u_np1 - u_n) / float(dt_np1)
            d1 = (u_n - u_nm1) / float(dt_n)
            d2 = (u_nm1 - u_nm2) / float(dt_nm1)
            tau = (float(dt_np1) ** 2 / 6.0) * (d0 - (1.0 + r) * d1 + r * d2)
            scale = p.time_adapt_rho_abs + p.time_adapt_rho_rel * np.maximum(
                np.abs(u_np1), np.abs(u_n)
            )
            scaled = tau / np.maximum(scale, 1.0e-300)
            local_pair = np.array([float(np.dot(scaled, scaled)), float(scaled.size)], dtype=np.float64)
        global_pair = np.zeros(2, dtype=np.float64)
        msh.comm.Allreduce(local_pair, global_pair, op=MPI.SUM)
        global_sum = float(global_pair[0])
        global_count = int(global_pair[1])
        if global_count <= 0:
            return 0.0
        return math.sqrt(global_sum / float(global_count))

    def estimate_time_error(dt_np1, dt_n, dt_nm1):
        err_xi = _local_error_component(xi_map, stats_dofs["xi"], dt_np1, dt_n, dt_nm1)
        err_c = _local_error_component(c_map, stats_dofs["c"], dt_np1, dt_n, dt_nm1)
        return max(err_xi, err_c), err_xi, err_c

    def time_error_step_factor(error_value):
        if math.isfinite(error_value) and error_value > 0.0:
            target = max(
                p.time_adapt_tol_min,
                p.time_adapt_target_fraction * p.time_adapt_tol_max,
            )
            factor = p.time_adapt_safety * (target / max(error_value, 1.0e-300)) ** (1.0 / 3.0)
        else:
            factor = p.time_adapt_factor_max
        return min(max(factor, p.time_adapt_factor_min), p.time_adapt_factor_max)

    def next_dt_from_time_error(error_value, dt_used, current_dt_max):
        factor = time_error_step_factor(error_value)
        return min(max(float(dt_used) * factor, p.dt_min), current_dt_max)

    for cycle_index, phase_name, duration, current_value, cutoff_voltage in phases:
        if duration <= 0.0:
            continue
        assign_constant(current_density, current_value)
        assign_constant(i_app, current_value)

        phase_elapsed = 0.0
        accepted_steps = 0
        previous_accepted_dt = None
        previous_previous_accepted_dt = None
        retry_recovery_remaining = 0
        current_dt_max = p.dt_max
        if phase_name == "charge" and p.dt_max_charge is not None:
            current_dt_max = min(current_dt_max, float(p.dt_max_charge))
        elif phase_name == "discharge" and p.dt_max_discharge is not None:
            current_dt_max = min(current_dt_max, float(p.dt_max_discharge))
        fixed_dt = p.dt if p.fixed_initial_dt is None else p.fixed_initial_dt
        fixed_dt = min(max(fixed_dt, p.dt_min), current_dt_max)
        dt_trial = fixed_dt if p.fixed_initial_steps > 0 else min(max(p.dt, p.dt_min), current_dt_max)
        while phase_elapsed < duration - 1.0e-15:
            dt_used = min(dt_trial, duration - phase_elapsed)
            assign_constant(dt_const, dt_used)
            set_bdf_coefficients(dt_used, previous_accepted_dt, accepted_steps > 0)
            trial_success = False

            retries_used = 0
            for _retry in range(p.max_retries_per_step):
                use_extrapolated_guess = (
                    p.extrapolate_initial_guess
                    and _retry == 0
                    and accepted_steps > 0
                    and previous_accepted_dt is not None
                    and previous_accepted_dt > 0.0
                    and p.extrapolate_max_factor > 0.0
                )
                if use_extrapolated_guess:
                    factor = min(
                        p.extrapolate_max_factor,
                        max(0.0, dt_used / previous_accepted_dt),
                    )
                    w.x.array[:] = w_n.x.array + factor * (w_n.x.array - w_prev_array)
                    for slave, master in periodic_constraints.items():
                        w.x.array[slave] = w.x.array[master]
                else:
                    w.x.array[:] = w_n.x.array
                w.x.scatter_forward()
                try:
                    newton_profiler.set_context(
                        accepted_step=total_step + 1,
                        global_time_s=time_s,
                        phase=phase_name,
                        dt_s=dt_used,
                        retry=_retry,
                    )
                    problem.solve()
                    w.x.scatter_forward()
                    snes_reason, snes_its, snes_residual = nonlinear_solver_status(problem)
                    if isinstance(snes_reason, (int, np.integer)) and snes_reason <= 0:
                        raise RuntimeError(
                            f"SNES did not converge, reason={snes_reason}, "
                            f"its={snes_its}, fnorm={snes_residual:.3e}"
                        )
                    trial_success = True
                    break
                except Exception as exc:
                    retries_used += 1
                    problem.reset_jacobian_cache()
                    if msh.comm.rank == 0:
                        print(
                            f"retry at t={time_s:.3e}s with dt={dt_used:.3e}s "
                            f"after: {exc}"
                        )
                    dt_used *= p.dt_shrink
                    if dt_used < p.dt_min:
                        save_latest_state(
                            f"{phase_name} phase failed: dt dropped below dt_min={p.dt_min:g} s"
                        )
                        raise RuntimeError(
                            f"{phase_name} phase failed: dt dropped below dt_min={p.dt_min:g} s"
                        )
                    assign_constant(dt_const, dt_used)
                    set_bdf_coefficients(dt_used, previous_accepted_dt, accepted_steps > 0)

            if not trial_success:
                save_latest_state(
                    f"{phase_name} phase could not find a converged adaptive step."
                )
                raise RuntimeError(f"{phase_name} phase could not find a converged adaptive step.")

            # 不再对 xi 和 c 做求解后的硬裁剪。
            # 硬裁剪 xi/c 不是弱形式的一部分，会改变 xi/c 的总量，直接破坏
            # “正极 Li 损失 = Omega1 Li 金属增加”的积分守恒检测。
            # 但 COMSOL 模型对 eta1...eta10 设置了 0/1 上下限；这里只裁剪
            # eta_i，用于防止长时间耦合后晶界相场漂出物理范围。
            if p.clip_eta_after_solve:
                clip_mixed_components(w, ME, eta_components, 0.0, 1.0)
            w.x.scatter_forward()

            time_error = float("nan")
            time_error_xi = float("nan")
            time_error_c = float("nan")
            time_adapt_factor = float("nan")
            time_adapt_dt_next = float("nan")
            reject_limit = p.time_adapt_tol_max * max(1.0, p.time_adapt_reject_factor)
            have_error_history = (
                accepted_steps >= 2
                and previous_accepted_dt is not None
                and previous_previous_accepted_dt is not None
                and previous_accepted_dt > 0.0
                and previous_previous_accepted_dt > 0.0
            )
            if have_error_history:
                time_error, time_error_xi, time_error_c = estimate_time_error(
                    dt_used, previous_accepted_dt, previous_previous_accepted_dt
                )
                time_adapt_factor = time_error_step_factor(time_error)
                time_adapt_dt_next = min(max(float(dt_used) * time_adapt_factor, p.dt_min), current_dt_max)
                will_reject = bool(
                    p.time_adapt_reject
                    and math.isfinite(time_error)
                    and time_error > reject_limit
                )
                newton_profiler.write_time_adapt(
                    {
                        "time_error": time_error,
                        "time_error_xi": time_error_xi,
                        "time_error_c": time_error_c,
                        "time_adapt_factor": time_adapt_factor,
                        "time_adapt_dt_next": time_adapt_dt_next,
                        "time_adapt_reject_limit": reject_limit,
                        "time_adapt_rejected": int(will_reject),
                    }
                )
                if will_reject:
                    next_rejected_dt = next_dt_from_time_error(time_error, dt_used, current_dt_max)
                    if next_rejected_dt >= dt_used * (1.0 - 1.0e-12):
                        next_rejected_dt = max(p.dt_min, dt_used * p.time_adapt_factor_min)
                    if next_rejected_dt < p.dt_min or dt_used <= p.dt_min * (1.0 + 1.0e-12):
                        save_latest_state(
                            f"{phase_name} phase failed: local time error {time_error:.3e} "
                            f"cannot be reduced below dt_min={p.dt_min:g} s"
                        )
                        raise RuntimeError(
                            f"{phase_name} phase failed: local time error {time_error:.3e} "
                            f"cannot be reduced below dt_min={p.dt_min:g} s"
                        )
                    problem.reset_jacobian_cache()
                    w.x.array[:] = w_n.x.array
                    w.x.scatter_forward()
                    if msh.comm.rank == 0:
                        print(
                            f"time-adapt reject at t={time_s:.3e}s: "
                            f"E={time_error:.3e} (xi={time_error_xi:.3e}, c={time_error_c:.3e}) "
                            f"> reject_limit={reject_limit:.3e}; "
                            f"dt {dt_used:.3e} -> {next_rejected_dt:.3e}; retrying",
                            flush=True,
                        )
                    dt_trial = min(max(next_rejected_dt, p.dt_min), current_dt_max)
                    assign_constant(dt_const, dt_trial)
                    set_bdf_coefficients(dt_trial, previous_accepted_dt, accepted_steps > 0)
                    continue

            accepted_steps += 1
            total_step += 1
            phase_elapsed += dt_used
            time_s += dt_used
            in_fixed_window = accepted_steps < p.fixed_initial_steps
            if in_fixed_window:
                dt_trial = fixed_dt
            else:
                if retries_used > 0:
                    retry_recovery_remaining = p.retry_recovery_steps
                    dt_trial = min(max(dt_used, p.dt_min), current_dt_max)
                elif have_error_history and math.isfinite(time_error):
                    error_dt_trial = next_dt_from_time_error(time_error, dt_used, current_dt_max)
                    if retry_recovery_remaining > 0:
                        recovery_cap = min(current_dt_max, max(p.dt_min, dt_used * p.retry_recovery_growth))
                        dt_trial = min(error_dt_trial, recovery_cap)
                        retry_recovery_remaining -= 1
                    else:
                        dt_trial = error_dt_trial
                else:
                    growth = p.time_adapt_factor_max
                    if retry_recovery_remaining > 0:
                        growth = min(growth, p.retry_recovery_growth)
                        retry_recovery_remaining -= 1
                    dt_trial = min(max(dt_used * growth, p.dt_min), current_dt_max)

            update_scalar_outputs(
                w,
                scalar_outputs,
                p,
                indices=primary_scalar_indices,
                component_maps=scalar_component_maps,
            )
            update_diagnostic_derived()
            phil_cutoff_min, phil_cutoff_max, phil_cutoff_mean = field_stats(
                scalar_outputs[1], boundary_probe_dofs["phil_gamma_a"]
            )
            phis_cutoff_min, phis_cutoff_max, phis_cutoff_mean = field_stats(
                scalar_outputs[2], boundary_probe_dofs["phis_gamma_c"]
            )
            c_cutoff_min, c_cutoff_max, c_cutoff_mean = field_stats(
                scalar_outputs[3], stats_dofs["c"]
            )
            terminal_voltage = phis_cutoff_max - phil_cutoff_min
            voltage_cutoff_hit_local = (
                terminal_voltage >= cutoff_voltage
                if phase_name == "charge"
                else terminal_voltage <= cutoff_voltage
            )
            soc_cutoff = p.soc_min if phase_name == "charge" else p.soc_max
            soc_cutoff_value = (
                c_cutoff_min if phase_name == "charge" else c_cutoff_max
            )
            soc_cutoff_hit_local = (
                c_cutoff_min <= soc_cutoff
                if phase_name == "charge"
                else c_cutoff_max >= soc_cutoff
            )
            voltage_cutoff_hit = bool(
                msh.comm.allreduce(int(voltage_cutoff_hit_local), op=MPI.MAX)
            )
            soc_cutoff_hit = bool(
                msh.comm.allreduce(int(soc_cutoff_hit_local), op=MPI.MAX)
            )
            current_dendrite_tip_y = dendrite_tip_y(
                scalar_outputs[0], stats_dofs["xi"], p.lifecycle_xi_threshold
            )
            lifecycle_stop_hit = bool(
                p.lifecycle_stop_enabled
                and math.isfinite(current_dendrite_tip_y)
                and math.isfinite(lifecycle_target_y)
                and current_dendrite_tip_y <= lifecycle_target_y + p.lifecycle_target_margin
            )
            cutoff_hit = voltage_cutoff_hit or soc_cutoff_hit or lifecycle_stop_hit
            cutoff_reason = (
                "lifecycle"
                if lifecycle_stop_hit
                else "voltage+soc"
                if voltage_cutoff_hit and soc_cutoff_hit
                else "voltage"
                if voltage_cutoff_hit
                else "soc"
                if soc_cutoff_hit
                else ""
            )
            write_diagnostics = (total_step % p.diagnostics_interval) == 0
            current_cycle_index = cycle_index
            current_phase_elapsed = phase_elapsed
            current_phase_duration = duration
            current_cutoff_voltage = cutoff_voltage
            current_soc_cutoff = soc_cutoff
            current_cutoff_reached = cutoff_hit
            current_voltage_cutoff_reached = voltage_cutoff_hit
            current_soc_cutoff_reached = soc_cutoff_hit
            current_cutoff_reason = cutoff_reason
            current_lifecycle_stop_reached = lifecycle_stop_hit
            diag_row = (
                write_diag(total_step, time_s, phase_name, dt_used)
                if write_diagnostics
                else None
            )
            if time_s + 1.0e-12 >= next_png_time:
                update_scalar_outputs(w, scalar_outputs, p, component_maps=scalar_component_maps)
                update_derived()
                save_png_outputs(
                    out_dir,
                    f"t_{time_s:010.3f}s",
                    V_scalar,
                    scalar_outputs,
                    derived_outputs,
                    B,
                    domain_markers,
                    p,
                )
                while next_png_time <= time_s + 1.0e-12:
                    next_png_time += p.png_interval
            if time_s + 1.0e-12 >= next_xdmf_time:
                update_scalar_outputs(w, scalar_outputs, p, component_maps=scalar_component_maps)
                update_derived()
                save_sampled_outputs(
                    out_dir,
                    f"t_{time_s:010.3f}s",
                    V_scalar,
                    scalar_outputs,
                    derived_outputs,
                    B,
                    domain_markers,
                    p,
                )
                while next_xdmf_time <= time_s + 1.0e-12:
                    next_xdmf_time += p.xdmf_interval
            if time_s + 1.0e-12 >= next_field_time:
                update_scalar_outputs(w, scalar_outputs, p, component_maps=scalar_component_maps)
                update_derived()
                save_field_snapshot(
                    Path(out_dir) / "fields" / f"t_{time_s:010.3f}s.npz",
                    V_scalar,
                    scalar_outputs,
                    derived_outputs,
                    B,
                    domain_markers,
                    p,
                    variables=p.field_output_variables,
                    metadata={
                        "time_s": time_s,
                        "step": total_step,
                        "cycle": cycle_index,
                        "phase": phase_name,
                    },
                )
                while next_field_time <= time_s + 1.0e-12:
                    next_field_time += p.field_output_interval_s
            if msh.comm.rank == 0:
                if p.progress_only:
                    message = (
                        f"cycle={cycle_index}, {phase_name}: accepted_step={accepted_steps}, "
                        f"phase_t={phase_elapsed:.3f} s, "
                        f"global_t={time_s:.3f} s, dt={dt_used:.4f} s, "
                        f"V={terminal_voltage:.6g} V, cutoff={cutoff_voltage:.6g} V, "
                        f"E_time={time_error:.3e}, "
                        f"SNES reason={snes_reason}, its={snes_its}"
                    )
                else:
                    message = (
                        f"cycle={cycle_index}, {phase_name}: accepted_step={accepted_steps}, "
                        f"phase_t={phase_elapsed:.3f} s, "
                        f"global_t={time_s:.3f} s, dt={dt_used:.4f} s, "
                        f"I={current_value:.3g} A/m^2, "
                        f"V={terminal_voltage:.6g} V, cutoff={cutoff_voltage:.6g} V, "
                        f"E_time={time_error:.3e} (xi={time_error_xi:.3e}, c={time_error_c:.3e}), "
                        f"SNES reason={snes_reason}, its={snes_its}, "
                        f"fnorm={snes_residual:.3e}"
                    )
                progress_print(message)

            # 诊断写完后再更新旧解；否则 (xi-xi_n)/dt 会被错误地算成 0。
            w_prev_array[:] = w_n.x.array
            w_nm2.x.array[:] = w_nm1.x.array
            w_nm2.x.scatter_forward()
            w_nm1.x.array[:] = w_n.x.array
            w_nm1.x.scatter_forward()
            w_n.x.array[:] = w.x.array
            w_n.x.scatter_forward()
            previous_previous_accepted_dt = previous_accepted_dt
            previous_accepted_dt = dt_used

            if cutoff_hit:
                if lifecycle_stop_hit:
                    lifecycle_terminated = True
                if msh.comm.rank == 0:
                    print(
                        f"cycle={cycle_index}, {phase_name}: cutoff reached, "
                        f"reason={cutoff_reason}, V={terminal_voltage:.6g} V, "
                        f"V_cutoff={cutoff_voltage:.6g} V, "
                        f"SOC_extreme={soc_cutoff_value:.6g}, "
                        f"SOC_cutoff={soc_cutoff:.6g}, "
                        f"dendrite_tip_y={current_dendrite_tip_y:.6g}, "
                        f"lifecycle_target_y={lifecycle_target_y:.6g}"
                    )
                break

        update_scalar_outputs(w, scalar_outputs, p, component_maps=scalar_component_maps)
        update_derived()
        save_png_outputs(
            out_dir,
            f"cycle{cycle_index:03d}_after_{phase_name}",
            V_scalar,
            scalar_outputs,
            derived_outputs,
            B,
            domain_markers,
            p,
        )
        if p.xdmf_interval > 0.0:
            save_sampled_outputs(
                out_dir,
                f"cycle{cycle_index:03d}_after_{phase_name}",
                V_scalar,
                scalar_outputs,
                derived_outputs,
                B,
                domain_markers,
                p,
            )

        if lifecycle_terminated:
            if msh.comm.rank == 0:
                print("lifecycle stop reached; ending all remaining phases.")
            break

    update_scalar_outputs(w, scalar_outputs, p, component_maps=scalar_component_maps)
    update_derived()
    save_png_outputs(
        out_dir,
        "final",
        V_scalar,
        scalar_outputs,
        derived_outputs,
        B,
        domain_markers,
        p,
    )
    if p.xdmf_interval > 0.0:
        save_sampled_outputs(
            out_dir,
            "final",
            V_scalar,
            scalar_outputs,
            derived_outputs,
            B,
            domain_markers,
            p,
        )
    save_delta_outputs(out_dir, V_scalar, initial_outputs, scalar_outputs, B, domain_markers)
    save_final_npz(
        p.final_npz_file,
        V_scalar,
        scalar_outputs,
        derived_outputs,
        p,
        extra_outputs=extra_output_fields(),
    )

    if msh.comm.rank == 0:
        print(f"已保存循环输出到: {out_dir}")
        print(f"已保存诊断 CSV: {p.diagnostics_file}")
        print(f"已保存最终状态 NPZ: {p.final_npz_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="R2D periodic mechanics charge-discharge cycling experiment."
    )
    parser.add_argument("--msh", default=DEFAULT_MSH_FILE)
    parser.add_argument("--gb", default=DEFAULT_GB_FILE)
    parser.add_argument("--t-end", type=float, default=None, help="Maximum charge time per cycle in seconds.")
    parser.add_argument("--discharge-time", type=float, default=None, help="Maximum discharge time per cycle in seconds.")
    parser.add_argument("--cycles", type=int, default=None, help="Number of charge-discharge cycles.")
    parser.add_argument("--charge-cutoff-voltage", type=float, default=None, help="Stop charge when terminal voltage reaches this value [V].")
    parser.add_argument("--discharge-cutoff-voltage", type=float, default=None, help="Stop discharge when terminal voltage reaches this value [V].")
    parser.add_argument("--cathode-reaction-mode", choices=("interface",), default="interface")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--dt-min", type=float, default=None)
    parser.add_argument("--li-side-potential", type=float, default=None)
    parser.add_argument("--soc-init", type=float, default=None)
    parser.add_argument("--soc-min", type=float, default=None)
    parser.add_argument("--soc-max", type=float, default=None)
    parser.add_argument("--D-li", dest="D_li", type=float, default=None)
    parser.add_argument("--i0-ref-li", type=float, default=None)
    parser.add_argument("--i0-c-ref", type=float, default=None)
    parser.add_argument("--c-li-max", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--initial-cathode-overpotential", type=float, default=None)
    parser.add_argument("--phis-init", type=float, default=None)
    parser.add_argument(
        "--charge-current-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=None,
        help="Sign for charge current; default is +1.",
    )
    parser.add_argument(
        "--discharge-current-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=None,
        help="Sign for discharge current; default is -1.",
    )
    parser.add_argument("--current-density", type=float, default=None, help="Absolute applied current density [A/m^2].")
    parser.add_argument("--E-li", type=float, default=None)
    parser.add_argument("--nu-li", type=float, default=None)
    parser.add_argument("--E-se", type=float, default=None)
    parser.add_argument("--nu-se", type=float, default=None)
    parser.add_argument("--E-cathode", type=float, default=None)
    parser.add_argument("--nu-cathode", type=float, default=None)
    parser.add_argument("--anode-compressive-load", type=float, default=None)
    parser.add_argument("--cathode-swelling-scale", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--k-xi", type=float, default=None)
    parser.add_argument("--W-xi", type=float, default=None)
    parser.add_argument("--L-gb", type=float, default=None)
    parser.add_argument("--W-gb", type=float, default=None)
    parser.add_argument("--k-gb", type=float, default=None)
    parser.add_argument("--mechanics-residual-scale", type=float, default=None)
    parser.add_argument("--li-layer-thickness", type=float, default=None, help="Optional initial Li layer thickness [m]; default is 0 in cycle runs.")
    parser.add_argument("--no-top-xi-bc", action="store_true", help="Disable xi=1 Dirichlet condition on the top Gamma_a boundary.")
    parser.add_argument(
        "--no-mechanics-periodic",
        action="store_true",
        help="Disable left-right displacement MPC. Use this for MPI runs until exact parallel MPC is implemented.",
    )
    parser.add_argument("--dt-max", type=float, default=None, help="Maximum adaptive time step [s].")
    parser.add_argument(
        "--dt-max-charge",
        type=float,
        default=None,
        help="Maximum adaptive time step during charge [s]. Defaults to the Params charge cap.",
    )
    parser.add_argument(
        "--dt-max-discharge",
        type=float,
        default=None,
        help="Maximum adaptive time step during discharge [s]. Defaults to the Params discharge cap.",
    )
    parser.add_argument("--dt-growth", type=float, default=None, help="Legacy option; normal accepted-step growth is controlled by the local time-error estimator in this copy.")
    parser.add_argument("--dt-shrink", type=float, default=None, help="Shrink factor after a failed nonlinear trial.")
    parser.add_argument(
        "--fixed-initial-steps",
        type=int,
        default=None,
        help="Keep the first N accepted steps at a fixed trial dt before adaptive growth starts.",
    )
    parser.add_argument(
        "--fixed-initial-dt",
        type=float,
        default=None,
        help="Fixed trial dt [s] used during --fixed-initial-steps; defaults to --dt/the Params dt.",
    )
    parser.add_argument("--time-adapt-tol-max", type=float, default=None, help="Reject accepted Newton trial if max(E_xi,E_c) exceeds this local time-error limit.")
    parser.add_argument("--time-adapt-tol-min", type=float, default=None, help="Lower bound for the local time-error target.")
    parser.add_argument("--time-adapt-safety", type=float, default=None, help="Safety factor in dt_new = safety*(tol/E)^(1/3)*dt.")
    parser.add_argument("--time-adapt-rho-abs", type=float, default=None, help="Absolute scale in the weighted local time-error norm.")
    parser.add_argument("--time-adapt-rho-rel", type=float, default=None, help="Relative scale in the weighted local time-error norm.")
    parser.add_argument("--time-adapt-factor-min", type=float, default=None, help="Minimum multiplicative dt factor from the local time-error controller.")
    parser.add_argument("--time-adapt-factor-max", type=float, default=None, help="Maximum multiplicative dt factor from the local time-error controller.")
    parser.add_argument("--time-adapt-reject-factor", type=float, default=None, help="Reject only when E_time exceeds reject_factor * time_adapt_tol_max; mild overshoots are accepted and shrink the next dt.")
    parser.add_argument("--time-adapt-target-fraction", type=float, default=None, help="Target error fraction of tol_max used in dt_new = safety*(target/E)^(1/3)*dt.")
    parser.add_argument("--retry-recovery-steps", type=int, default=None, help="Number of accepted steps after a Newton retry where dt growth is capped.")
    parser.add_argument("--retry-recovery-growth", type=float, default=None, help="Maximum dt growth factor during the post-retry recovery window.")
    parser.add_argument("--no-time-adapt-reject", action="store_true", help="Do not reject Newton-converged steps when the local time-error estimate is too large.")
    parser.add_argument("--snes-rtol", type=float, default=None)
    parser.add_argument("--snes-atol", type=float, default=None)
    parser.add_argument("--snes-stol", type=float, default=None)
    parser.add_argument("--snes-max-it", type=int, default=None)
    parser.add_argument("--no-snes-monitor", action="store_true")
    parser.add_argument("--snes-monitor", action="store_true")
    parser.add_argument(
        "--png-interval",
        type=float,
        default=None,
        help="Save preview PNG fields every N simulated seconds; 0 disables interval PNG snapshots.",
    )
    parser.add_argument(
        "--xdmf-interval",
        type=float,
        default=None,
        help="Save XDMF field snapshots every N simulated seconds; 0 disables interval XDMF snapshots.",
    )
    parser.add_argument(
        "--field-output-interval",
        type=float,
        default=None,
        help="Save software-plot NPZ snapshots every N simulated seconds; 0 disables field snapshots.",
    )
    parser.add_argument(
        "--field-output-variables",
        default=None,
        help="Comma-separated field names to save in NPZ snapshots, or 'all'. Examples: all or xi,ce,B,c,phil,ux,uy.",
    )
    parser.add_argument(
        "--diagnostics-interval",
        type=int,
        default=None,
        help="Write diagnostics every N accepted steps; 1 keeps every accepted step.",
    )
    parser.add_argument(
        "--progress-only",
        action="store_true",
        help="Only print compact accepted-step progress lines to the terminal.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable terminal output for diagnostics/debugging; default production runs are quiet.",
    )
    parser.add_argument(
        "--no-eta-clip",
        action="store_true",
        help="Disable post-solve clipping of eta_i to [0, 1].",
    )
    parser.add_argument(
        "--no-newton-profile",
        action="store_true",
        help="Disable per-Newton-iteration timing output in this profiled copy.",
    )
    parser.add_argument(
        "--newton-profile",
        action="store_true",
        help="Enable per-Newton-iteration timing output.",
    )
    parser.add_argument(
        "--newton-profile-file",
        default=None,
        help="CSV path for per-Newton-iteration timing output.",
    )
    parser.add_argument(
        "--linear-solver",
        choices=(
            "lu",
            "mumps",
            "gmres_ilu",
            "fgmres_ilu",
            "gmres_hypre",
            "fgmres_hypre",
            "gmres_gamg",
            "fgmres_gamg",
            "gmres_bjacobi",
            "fgmres_bjacobi",
            "gmres_jacobi",
            "fgmres_jacobi",
        ),
        default=None,
        help="Linear solver/preconditioner mode for Newton corrections; default preserves plain LU.",
    )
    parser.add_argument("--linear-rtol", type=float, default=None)
    parser.add_argument("--linear-atol", type=float, default=None)
    parser.add_argument("--linear-max-it", type=int, default=None)
    parser.add_argument(
        "--jacobian-lag",
        type=int,
        default=None,
        help="Reuse the assembled/reduced Jacobian for up to N Newton iterations; 1 rebuilds every iteration.",
    )
    parser.add_argument(
        "--jacobian-mode",
        choices=("full", "block"),
        default=None,
        help="Jacobian assembly mode. 'block' keeps F_total unchanged but drops off-diagonal block derivatives.",
    )
    parser.add_argument(
        "--reuse-jacobian-across-steps",
        action="store_true",
        help="Keep the cached Jacobian between accepted time steps so the next step can start by reusing it.",
    )
    parser.add_argument(
        "--fast-residual-bc",
        action="store_true",
        help="Skip residual apply_lifting and keep set_bc only; this is already the default in the optimized profile.",
    )
    parser.add_argument(
        "--safe-residual-bc",
        action="store_true",
        help="Use the standard residual apply_lifting path for accuracy/safety comparison.",
    )
    parser.add_argument(
        "--full-diagnostics",
        action="store_true",
        help="Write the full diagnostic row instead of the faster compact diagnostics.",
    )
    parser.add_argument(
        "--no-extrapolate-initial-guess",
        action="store_true",
        help="Disable linear extrapolation of the Newton initial guess from the last accepted step.",
    )
    parser.add_argument(
        "--extrapolate-max-factor",
        type=float,
        default=None,
        help="Cap for the extrapolated initial-guess factor dt/dt_prev; 1.0 gives w_n + (w_n-w_{n-1}).",
    )
    parser.add_argument("--no-lifecycle-stop", action="store_true", help="Disable stop when Li dendrite tip reaches Omega3 cathode region.")
    parser.add_argument("--lifecycle-xi-threshold", type=float, default=None, help="xi threshold used to locate the Li dendrite tip; default 0.5.")
    parser.add_argument("--lifecycle-target-margin", type=float, default=None, help="Normalized y/L margin below the mesh Omega1/Omega2 interface for lifecycle stop; default 1e-9.")
    args = parser.parse_args()
    main(
        msh_file=args.msh,
        gb_file=args.gb,
        charge_time=args.t_end,
        discharge_time=args.discharge_time,
        cycle_count=args.cycles,
        charge_cutoff_voltage=args.charge_cutoff_voltage,
        discharge_cutoff_voltage=args.discharge_cutoff_voltage,
        dt=args.dt,
        dt_min=args.dt_min,
        cathode_reaction_mode=args.cathode_reaction_mode,
        li_side_potential=args.li_side_potential,
        soc_init=args.soc_init,
        soc_min=args.soc_min,
        soc_max=args.soc_max,
        D_li=args.D_li,
        i0_ref_li=args.i0_ref_li,
        i0_c_ref=args.i0_c_ref,
        c_li_max=args.c_li_max,
        temperature=args.temperature,
        initial_cathode_overpotential=args.initial_cathode_overpotential,
        phis_init=args.phis_init,
        charge_current_sign=args.charge_current_sign,
        discharge_current_sign=args.discharge_current_sign,
        current_abs=args.current_density,
        E_li=args.E_li,
        nu_li=args.nu_li,
        E_se=args.E_se,
        nu_se=args.nu_se,
        E_cathode=args.E_cathode,
        nu_cathode=args.nu_cathode,
        anode_compressive_load=args.anode_compressive_load,
        cathode_swelling_scale=args.cathode_swelling_scale,
        gamma=args.gamma,
        k_xi=args.k_xi,
        W_xi=args.W_xi,
        L_gb=args.L_gb,
        W_gb=args.W_gb,
        k_gb=args.k_gb,
        li_layer_thickness=args.li_layer_thickness,
        enforce_top_xi_bc=False if args.no_top_xi_bc else None,
        mechanics_periodic_lr=False if args.no_mechanics_periodic else None,
        dt_max=args.dt_max,
        dt_max_charge=args.dt_max_charge,
        dt_max_discharge=args.dt_max_discharge,
        dt_growth=args.dt_growth,
        dt_shrink=args.dt_shrink,
        fixed_initial_steps=args.fixed_initial_steps,
        fixed_initial_dt=args.fixed_initial_dt,
        time_adapt_tol_max=args.time_adapt_tol_max,
        time_adapt_tol_min=args.time_adapt_tol_min,
        time_adapt_safety=args.time_adapt_safety,
        time_adapt_rho_abs=args.time_adapt_rho_abs,
        time_adapt_rho_rel=args.time_adapt_rho_rel,
        time_adapt_factor_min=args.time_adapt_factor_min,
        time_adapt_factor_max=args.time_adapt_factor_max,
        time_adapt_reject=False if args.no_time_adapt_reject else None,
        time_adapt_reject_factor=args.time_adapt_reject_factor,
        time_adapt_target_fraction=args.time_adapt_target_fraction,
        retry_recovery_steps=args.retry_recovery_steps,
        retry_recovery_growth=args.retry_recovery_growth,
        snes_rtol=args.snes_rtol,
        snes_atol=args.snes_atol,
        snes_stol=args.snes_stol,
        snes_max_it=args.snes_max_it,
        snes_monitor=True if args.snes_monitor else False if args.no_snes_monitor else None,
        png_interval=args.png_interval,
        xdmf_interval=args.xdmf_interval,
        field_output_interval_s=args.field_output_interval,
        field_output_variables=args.field_output_variables,
        diagnostics_interval=args.diagnostics_interval,
        progress_only=True if args.progress_only else None,
        quiet=False if args.verbose or args.progress_only or args.snes_monitor else None,
        clip_eta_after_solve=False if args.no_eta_clip else None,
        mechanics_residual_scale=args.mechanics_residual_scale,
        newton_profile=True if args.newton_profile else False if args.no_newton_profile else None,
        newton_profile_file=args.newton_profile_file,
        linear_solver=args.linear_solver,
        linear_rtol=args.linear_rtol,
        linear_atol=args.linear_atol,
        linear_max_it=args.linear_max_it,
        jacobian_lag=args.jacobian_lag,
        jacobian_mode=args.jacobian_mode,
        reuse_jacobian_across_steps=True if args.reuse_jacobian_across_steps else None,
        fast_residual_bc=False if args.safe_residual_bc else True if args.fast_residual_bc else None,
        light_diagnostics=False if args.full_diagnostics else None,
        extrapolate_initial_guess=False if args.no_extrapolate_initial_guess else None,
        extrapolate_max_factor=args.extrapolate_max_factor,
        lifecycle_stop_enabled=False if args.no_lifecycle_stop else None,
        lifecycle_xi_threshold=args.lifecycle_xi_threshold,
        lifecycle_target_margin=args.lifecycle_target_margin,
    )

















