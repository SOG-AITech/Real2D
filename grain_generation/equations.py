"""Grain-boundary indicators and COMSOL-style derived fields."""

import ufl
from dolfinx import fem


def build_allen_cahn_system(mesh_state, field_state, params):
    """Build the Allen-Cahn residual and Jacobian for the prepared fields."""
    msh = mesh_state.grain_mesh
    dx = mesh_state.dx
    eta = field_state.eta
    eta_n = field_state.eta_n
    etas = ufl.split(eta)
    etas_n = ufl.split(eta_n)
    tests = ufl.TestFunctions(field_state.ME)
    deta = ufl.TrialFunction(field_state.ME)
    dt_hat = fem.Constant(msh, params.dt / params.time_scale)
    ac_grad = params.time_scale * params.L_phi * params.k_phi / params.length_scale**2
    ac_chem = params.time_scale * params.L_phi * params.W_phi
    residual = 0
    for i in range(params.n_grains):
        cross = sum(etas[j] ** 2 for j in range(params.n_grains) if j != i)
        derivative = -etas[i] + etas[i] ** 3 + 2.0 * etas[i] * cross
        residual += (
            (etas[i] - etas_n[i]) / dt_hat * tests[i] * dx
            + ac_grad * ufl.dot(ufl.grad(etas[i]), ufl.grad(tests[i])) * dx
            + ac_chem * derivative * tests[i] * dx
        )
    from .state import EquationSystem
    return EquationSystem(residual, ufl.derivative(residual, eta, deta), dt_hat)



