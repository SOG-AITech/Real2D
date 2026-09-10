"""Build shared UFL views over the mixed finite-element state."""

import ufl


def split_state(finite_element_state):
    """Return current/history components, tests, and vector helpers."""
    components = finite_element_state.components
    components_n = finite_element_state.previous_components
    components_nm1 = finite_element_state.previous_previous_components
    xi, phil, phis, c, ux, uy = components[:6]
    xi_n, phil_n, phis_n, c_n, ux_n, uy_n = components_n[:6]
    xi_nm1, phil_nm1, phis_nm1, c_nm1, ux_nm1, uy_nm1 = components_nm1[:6]
    tests = finite_element_state.tests
    return {
        "state": finite_element_state.state,
        "components": components, "components_n": components_n,
        "components_nm1": components_nm1, "tests": tests,
        "xi": xi, "phil": phil, "phis": phis, "c": c, "ux": ux, "uy": uy,
        "xi_n": xi_n, "phil_n": phil_n, "phis_n": phis_n, "c_n": c_n,
        "ux_n": ux_n, "uy_n": uy_n, "xi_nm1": xi_nm1, "phil_nm1": phil_nm1,
        "phis_nm1": phis_nm1, "c_nm1": c_nm1, "ux_nm1": ux_nm1, "uy_nm1": uy_nm1,
        "etas": components[6:], "etas_n": components_n[6:], "etas_nm1": components_nm1[6:],
        "v_xi": tests[0], "v_l": tests[1], "v_s": tests[2], "v_c": tests[3],
        "v_ux": tests[4], "v_uy": tests[5], "v_etas": tests[6:],
        "u_vec": ufl.as_vector((ux, uy)), "v_u": ufl.as_vector((tests[4], tests[5])),
        "dw": finite_element_state.trial,
    }


__all__ = ["split_state"]
