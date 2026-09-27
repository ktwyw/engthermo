"""Electrochemical thermodynamics: standard cell potentials from Gibbs energies, the Nernst equation, the
temperature coefficient of the potential, and the limiting efficiencies of fuel cells and electrolysers.

>>> from engthermo import electrochem
>>> round(electrochem.standard_potential({"H2": -1, "O2": -0.5, "H2O(l)": 1}, n_electrons=2), 3)    # V
1.229
"""

from __future__ import annotations

import numpy as np

from . import reaction
from .constants import T_STD, F, R


def standard_potential(rxn: dict, n_electrons: float, T: float = T_STD) -> float:
    """E° = -Delta G°(T) / (n F) for the cell reaction as written (V)."""
    return float(-reaction.standard_gibbs(rxn, T) / (n_electrons * F))


def nernst(E0: float, n_electrons: float, reaction_quotient, T: float = T_STD) -> float:
    """E = E° - (RT / nF) ln Q."""
    return float(E0 - R * T / (n_electrons * F) * np.log(reaction_quotient))


def temperature_coefficient(rxn: dict, n_electrons: float, T: float = T_STD) -> float:
    """dE°/dT = Delta S°(T) / (n F) (V/K)."""
    return float(reaction.standard_entropy(rxn, T) / (n_electrons * F))


def fuel_cell_limits(rxn: dict, n_electrons: float, T: float = T_STD) -> dict:
    """Reversible potential, thermoneutral (enthalpy) potential, and the maximum (thermodynamic) efficiency
    Delta G / Delta H of a fuel cell running the given reaction."""
    dG, dH = reaction.standard_gibbs(rxn, T), reaction.standard_enthalpy(rxn, T)
    return {"E_rev": -dG / (n_electrons * F), "E_thermoneutral": -dH / (n_electrons * F), "efficiency_max": dG / dH,
            "dG": dG, "dH": dH, "T": T}


def efficiency_at_voltage(V: float, E_thermoneutral: float) -> float:
    """Voltage efficiency referred to the heating value: V / E_thermoneutral (fuel cell) or E_thermoneutral / V
    (electrolyser - use ``electrolyser_efficiency``)."""
    return V / E_thermoneutral


def electrolyser_efficiency(V: float, E_thermoneutral: float) -> float:
    """Energy efficiency of an electrolyser operating at cell voltage V, referred to the heating value of the
    hydrogen produced: E_thermoneutral / V."""
    return E_thermoneutral / V


def energy_per_mass(dG: float, molar_mass_kg: float) -> float:
    """Theoretical specific energy (Wh/kg) of a cell reaction: -Delta G per kg of reactants."""
    return -dG / molar_mass_kg / 3600.0
