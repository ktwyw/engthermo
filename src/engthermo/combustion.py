"""Combustion and high-temperature thermochemistry: heating values, stoichiometry, adiabatic flame temperatures
with temperature-dependent heat capacities, and Ellingham diagrams for the oxidation of metals.

>>> from engthermo import combustion
>>> hv = combustion.heating_values("CH4")
>>> round(hv["LHV"] / 1e3, 1), round(hv["HHV"] / 1e3, 1)                  # kJ/mol
(802.6, 890.6)
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from . import idealgas, reaction
from .components import get
from .constants import T_STD


def stoichiometry(fuel: str) -> dict:
    """Complete-combustion stoichiometry of a C/H/O fuel: moles of O2 per mole of fuel and the products."""
    f = reaction.parse_formula(get(fuel).formula)
    c, h, o = f.get("C", 0), f.get("H", 0), f.get("O", 0)
    n_O2 = c + h / 4 - o / 2
    return {"O2": n_O2, "CO2": c, "H2O": h / 2, "air": n_O2 / 0.21}


def heating_values(fuel: str) -> dict:
    """Lower (water vapour) and higher (liquid water) heating values at 25 degC, J per mole of fuel."""
    st = stoichiometry(fuel)
    rxn_v = {fuel: -1, "O2": -st["O2"], "CO2": st["CO2"], "H2O": st["H2O"]}
    rxn_l = {fuel: -1, "O2": -st["O2"], "CO2": st["CO2"], "H2O(l)": st["H2O"]}
    return {"LHV": -reaction.standard_enthalpy(rxn_v), "HHV": -reaction.standard_enthalpy(rxn_l), "stoichiometry": st}


def flue_gas(fuel: str, excess_air: float = 0.0) -> dict:
    """Moles of each flue-gas species per mole of fuel for complete combustion in air (79/21 N2/O2) with the
    given fractional excess air."""
    st = stoichiometry(fuel)
    n_O2_fed = st["O2"] * (1 + excess_air)
    return {"CO2": st["CO2"], "H2O": st["H2O"], "O2": n_O2_fed - st["O2"], "N2": n_O2_fed * 0.79 / 0.21}


def adiabatic_flame_temperature(fuel: str, excess_air: float = 0.0, T_in: float = T_STD) -> dict:
    """Adiabatic flame temperature for complete combustion in air (no dissociation): the products' enthalpy
    rise from 25 degC equals the heat released plus the sensible heat of the reactants above 25 degC."""
    prod = flue_gas(fuel, excess_air)
    st = stoichiometry(fuel)
    q = heating_values(fuel)["LHV"]
    n_air = st["O2"] * (1 + excess_air) / 0.21
    air = idealgas.Mixture({"N2": 0.79, "O2": 0.21})
    q += idealgas.enthalpy_change(fuel, T_STD, T_in) + n_air * air.enthalpy_change(T_STD, T_in)

    def f(T):
        return sum(n * idealgas.enthalpy_change(s, T_STD, T) for s, n in prod.items() if n > 0) - q

    T_ad = optimize.brentq(f, T_STD, 4000.0, xtol=1e-9)
    return {"T_ad": float(T_ad), "products": prod, "heat_released": float(q)}


# ------------------------------------------------------------------ Ellingham diagram
# Delta G° (kJ per mole of O2) = A + B T for the formation of oxides: linear approximations of the standard data as
# tabulated in Gaskell, "Introduction to the Thermodynamics of Materials" (each line applies below the metal's melting
# point). The gas-phase lines (C-CO, C-CO2, H2-H2O) are computed exactly from the component data instead.
ELLINGHAM_LINES = {
    "4/3 Al + O2 -> 2/3 Al2O3": (-1120.0, 0.208), "2 Mg + O2 -> 2 MgO": (-1203.0, 0.216),
    "Si + O2 -> SiO2": (-907.0, 0.175), "Ti + O2 -> TiO2": (-941.0, 0.177), "2 Zn + O2 -> 2 ZnO": (-696.0, 0.201),
    "2 Fe + O2 -> 2 FeO": (-529.0, 0.130), "2 Ni + O2 -> 2 NiO": (-478.0, 0.173),
    "4 Cu + O2 -> 2 Cu2O": (-336.0, 0.140),
}


def ellingham(T):
    """Delta G° (J per mole of O2) of oxide-formation reactions against T: the tabulated metal lines plus the
    carbon and hydrogen lines computed from the component data. Returns {reaction: array}."""
    T = np.atleast_1d(np.asarray(T, float))
    out = {k: (A + B * T) * 1e3 for k, (A, B) in ELLINGHAM_LINES.items()}
    out["2 C + O2 -> 2 CO"] = np.array([reaction.standard_gibbs({"C": -2, "O2": -1, "CO": 2}, t) for t in T])
    out["C + O2 -> CO2"] = np.array([reaction.standard_gibbs({"C": -1, "O2": -1, "CO2": 1}, t) for t in T])
    out["2 H2 + O2 -> 2 H2O"] = np.array([reaction.standard_gibbs({"H2": -2, "O2": -1, "H2O": 2}, t) for t in T])
    return out


def oxygen_partial_pressure(dG_per_mol_O2, T):
    """Equilibrium oxygen partial pressure (bar) of an oxide-formation line: p_O2 = exp(Delta G° / RT)."""
    return np.exp(np.asarray(dG_per_mol_O2, float) / (8.314462618 * np.asarray(T, float)))
