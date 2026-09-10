"""Material laws, interpolation tables, and stress/strain helpers."""

import numpy as np
import ufl

from .config import E_EQ_TABLE


def h(z):
    return z**3 * (6.0 * z**2 - 15.0 * z + 10.0)


def hp(z):
    return 30.0 * z**2 * (1.0 - z) ** 2


def clip01_ufl(z):
    return ufl.max_value(0.0, ufl.min_value(1.0, z))


def xi_anisotropic_gradient_term(xi, v_xi, p):
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


def strain(u):
    return ufl.sym(ufl.grad(u))


def mechanical_energy_2d(E_expr, nu_expr, eps_eff_expr):
    """Plane-stress mechanical energy density used by the phase fields."""
    e11 = eps_eff_expr[0, 0]
    e22 = eps_eff_expr[1, 1]
    e12 = eps_eff_expr[0, 1]
    return 0.5 * (
        E_expr / (1.0 - nu_expr**2) * (e11**2 + nu_expr * e11 * e22)
        + E_expr / (1.0 - nu_expr**2) * (e22**2 + nu_expr * e11 * e22)
        + E_expr / (1.0 + 2.0 * nu_expr) * 2.0 * e12**2
    )


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


def piecewise_linear_ufl(x, table):
    x0, y0 = table[0]
    x1, y1 = table[1]
    result = y0 + (y1 - y0) / (x1 - x0) * (x - x0)
    for i in range(len(table) - 1, 0, -1):
        xa, ya = table[i - 1]
        xb, yb = table[i]
        value = ya + (yb - ya) / (xb - xa) * (x - xa)
        result = ufl.conditional(ufl.le(x, xb), value, result)
    return result


def pchip_slopes(table):
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


def smooth_table_ufl(x, table, slopes):
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


def smooth_table_numpy(theta, table, slopes):
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


def eeq_from_soc_numpy(theta):
    return smooth_table_numpy(theta, E_EQ_TABLE, E_EQ_SLOPES)


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


__all__ = [
    "h", "hp", "clip01_ufl", "xi_anisotropic_gradient_term", "strain",
    "mechanical_energy_2d",
    "plane_stress_tensor", "plane_stress_hydrostatic_stress", "smooth_li_profile",
    "sharp_li_profile", "piecewise_linear_ufl", "pchip_slopes", "E_EQ_SLOPES",
    "smooth_table_ufl", "smooth_table_numpy", "eeq_from_soc", "eeq_from_soc_numpy",
    "nearest_old_values",
]
