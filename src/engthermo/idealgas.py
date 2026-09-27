"""Ideal-gas properties with temperature-dependent heat capacities, for pure gases and mixtures.

Cp/R = A + B T + C T^2 + D T^-2 (Smith, Van Ness & Abbott form). Enthalpy and entropy are relative to the
ideal-gas reference state at T0 = 298.15 K and P0 = 1 bar; absolute values need the formation properties
(see ``reaction``). Units: J, mol, K, Pa.

>>> from engthermo import idealgas
>>> round(idealgas.cp("N2", 1000.0), 2)                        # J/(mol K)
32.67
>>> round(idealgas.enthalpy_change("air", 300.0, 800.0) / 1e3, 2)   # kJ/mol
15.12
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from .components import Component, get
from .constants import P_STD, T_STD, R


def _comp(c) -> Component:
    return c if isinstance(c, Component) else get(c)


def _scalar(x):
    return float(x) if np.ndim(x) == 0 else x


def cp(comp, T):
    """Molar isobaric heat capacity Cp (J/(mol K)) of an ideal gas at T (K)."""
    A, B, C, D = _comp(comp).cp
    T = np.asarray(T, float)
    return _scalar(R * (A + B * T + C * T**2 + D / T**2))


def enthalpy_change(comp, T1, T2):
    """Integral of Cp dT from T1 to T2 (J/mol)."""
    A, B, C, D = _comp(comp).cp
    T1, T2 = np.asarray(T1, float), np.asarray(T2, float)
    return _scalar(R * (A * (T2 - T1) + B / 2 * (T2**2 - T1**2) + C / 3 * (T2**3 - T1**3) - D * (1 / T2 - 1 / T1)))


def entropy_change(comp, T1, T2, P1=P_STD, P2=P_STD):
    """Entropy change of an ideal gas (J/(mol K)): integral of Cp/T dT minus R ln(P2/P1)."""
    A, B, C, D = _comp(comp).cp
    T1, T2 = np.asarray(T1, float), np.asarray(T2, float)
    return _scalar(R * (A * np.log(T2 / T1) + B * (T2 - T1) + C / 2 * (T2**2 - T1**2) - D / 2 * (1 / T2**2 - 1 / T1**2))
                   - R * np.log(P2 / P1))


def mean_cp_h(comp, T1, T2):
    """Mean heat capacity for enthalpy calculations, <Cp>_H = integral Cp dT / (T2 - T1)."""
    return _scalar(enthalpy_change(comp, T1, T2) / (np.asarray(T2, float) - np.asarray(T1, float)))


def enthalpy(comp, T):
    """Ideal-gas enthalpy relative to 298.15 K (J/mol): H(T) - H(T0)."""
    return enthalpy_change(comp, T_STD, T)


def entropy(comp, T, P=P_STD):
    """Ideal-gas entropy relative to the standard state (298.15 K, 1 bar) (J/(mol K))."""
    return entropy_change(comp, T_STD, T, P_STD, P)


T_HI = 4000.0                   # upper search limit


def _lowest_T(comp) -> float:
    """Lowest temperature at which the component's Cp polynomial is still physical (Cp > 2R): the fitted
    polynomials are only extrapolations below 298 K, and some turn negative near 100 K."""
    for T in np.arange(100.0, 300.0, 5.0):
        if cp(comp, T) > 2 * R:
            return float(T)
    return 300.0


def _bracket(comp, T1, P1, P2):
    """Compression heats, expansion cools: search on the right side of T1."""
    return (T1, T_HI) if P2 > P1 else (_lowest_T(comp), T1)


def isentropic_T(comp, T1, P1, P2) -> float:
    """Final temperature of a reversible adiabatic (isentropic) change of an ideal gas with variable Cp."""
    if P1 == P2:
        return float(T1)
    return float(optimize.brentq(lambda T: entropy_change(comp, T1, T, P1, P2), *_bracket(comp, T1, P1, P2), xtol=1e-9))


def isentropic_T_const_gamma(T1, P1, P2, gamma: float) -> float:
    """The textbook constant-gamma result T2 = T1 (P2/P1)^((gamma-1)/gamma), for comparison."""
    return float(T1 * (P2 / P1) ** ((gamma - 1) / gamma))


def final_T_from_enthalpy(comp, T1, dH) -> float:
    """Temperature reached when dH (J/mol) is added to an ideal gas at T1 (solves the Cp integral)."""
    lo, hi = (T1, T_HI) if dH > 0 else (_lowest_T(comp), T1)
    return float(optimize.brentq(lambda T: enthalpy_change(comp, T1, T) - dH, lo, hi, xtol=1e-9)) if dH else float(T1)


class Mixture:
    """Ideal-gas mixture with mole fractions. Properties are mole-fraction-weighted sums; the entropy adds the
    ideal mixing term -R sum y ln y (relative to the pure components at the same T and P).

    >>> m = Mixture({"N2": 0.79, "O2": 0.21})
    >>> round(m.cp(300.0), 2)
    29.29
    """

    def __init__(self, composition: dict):
        total = sum(composition.values())
        self.y = {get(k).name: v / total for k, v in composition.items()}
        self.M = sum(y * get(k).M for k, y in self.y.items())

    def cp(self, T):
        return sum(y * cp(k, T) for k, y in self.y.items())

    def enthalpy_change(self, T1, T2):
        return sum(y * enthalpy_change(k, T1, T2) for k, y in self.y.items())

    def entropy_change(self, T1, T2, P1=P_STD, P2=P_STD):
        return sum(y * entropy_change(k, T1, T2, P1, P2) for k, y in self.y.items())

    def mixing_entropy(self):
        """Entropy of mixing per mole of mixture, -R sum y ln y (J/(mol K))."""
        return -R * sum(y * np.log(y) for y in self.y.values() if y > 0)

    def isentropic_T(self, T1, P1, P2) -> float:
        if P1 == P2:
            return float(T1)
        lo = max(_lowest_T(k) for k in self.y)
        lo, hi = (T1, T_HI) if P2 > P1 else (lo, T1)
        return float(optimize.brentq(lambda T: self.entropy_change(T1, T, P1, P2), lo, hi, xtol=1e-9))
