"""Derived electrochemical, phase, mechanical, and diagnostic fields."""

from .config import OMEGA1_REGIONS, OMEGA2, OMEGA3
from .fields import (
    copy_dg0_on_regions,
    mask_dg0_to_regions,
    recover_dg0_to_p1,
    update_interpolated,
)
from .output import mask_function_to_regions


def update_derived_fields(
    derived_outputs,
    derived_exprs,
    B,
    B_expr,
    domain_markers,
    hydro_omega1_dg0,
    hydro_cathode_dg0,
    hydro_omega2_dg0,
    hydro_omega12_dg0,
    hydro_omega23_dg0,
    hydro1,
    hydro2,
    hydro3,
    V_scalar,
    recovery_derived_indices,
    omega1_derived_indices,
    omega12_derived_indices,
    omega3_derived_indices,
    indices=None,
    include_b=True,
):
    """Refresh derived fields using the finite-element objects built by the workflow."""
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
            mask_function_to_regions(derived_outputs[idx], V_scalar, domain_markers, OMEGA1_REGIONS)
        elif idx in omega3_derived_indices:
            mask_function_to_regions(derived_outputs[idx], V_scalar, domain_markers, (OMEGA3,))
        elif idx in omega12_derived_indices:
            mask_function_to_regions(derived_outputs[idx], V_scalar, domain_markers, OMEGA1_REGIONS + (OMEGA2,))
        elif idx == 22:
            mask_function_to_regions(derived_outputs[idx], V_scalar, domain_markers, (OMEGA2, OMEGA3))


def update_diagnostic_derived_fields(update_derived, light_diagnostics,
                                     light_indices, full_indices):
    """Refresh only the derived fields needed by the cutoff diagnostics."""
    indices = light_indices if light_diagnostics else full_indices
    return update_derived(indices, include_b=False)


__all__ = ["update_derived_fields", "update_diagnostic_derived_fields"]
