"""Electrochemical constitutive and interface-reaction terms."""

from .constitutive import eeq_from_soc_numpy
from .equations import cathode_interface_flux_terms


def eeq_from_soc(theta):
    return eeq_from_soc_numpy(theta)


__all__ = ["cathode_interface_flux_terms", "eeq_from_soc"]
