"""Solid-liquid equilibrium: ideal and non-ideal solubility of a solid in a liquid, eutectic phase diagrams of
binaries with immiscible solids, and the lens (cigar) diagram of ideal solid and liquid solutions (isomorphous
alloys). Inputs per component: melting temperature Tm (K) and enthalpy of fusion dH_fus (J/mol).

>>> from engthermo import sle
>>> round(sle.ideal_solubility(T=300.0, Tm=353.4, dH_fus=19.1e3), 4)     # naphthalene in an ideal solvent
0.3144
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from .constants import R


def ideal_solubility(T, Tm: float, dH_fus: float, gamma: float = 1.0):
    """Mole fraction of a solid solute in a liquid at T (Schroeder-van Laar equation, heat-capacity terms
    neglected): ln(x gamma) = -(dH_fus / R)(1/T - 1/Tm). Returns 1 above the melting point."""
    T = np.asarray(T, float)
    x = np.exp(-dH_fus / R * (1 / T - 1 / Tm)) / gamma
    out = np.minimum(x, 1.0)
    return float(out) if np.ndim(out) == 0 else out


def liquidus_T(x, Tm: float, dH_fus: float, gamma=1.0) -> float:
    """Temperature at which a liquid of solute mole fraction x is saturated (inverse of ``ideal_solubility``)."""
    g = gamma if not callable(gamma) else None
    if g is None:
        f = lambda T: np.log(x * gamma(x, T)) + dH_fus / R * (1 / T - 1 / Tm)  # noqa: E731
        return float(optimize.brentq(f, 50.0, Tm, xtol=1e-10))
    return float(1 / (1 / Tm - R * np.log(x * g) / dH_fus))


def eutectic(Tm1: float, dH1: float, Tm2: float, dH2: float, gamma=None) -> dict:
    """Eutectic point of a binary whose solids are immiscible: the intersection of the two liquidus curves.
    ``gamma`` optionally gives activity coefficients gamma(x, T) -> (g1, g2) for a non-ideal liquid."""
    def T1(x1):
        return liquidus_T(x1, Tm1, dH1, (lambda x, T: gamma([x, 1 - x], T)[0]) if gamma else 1.0)

    def T2(x1):
        return liquidus_T(1 - x1, Tm2, dH2, (lambda x, T: gamma([1 - x, x], T)[1]) if gamma else 1.0)

    x_e = optimize.brentq(lambda x: T1(x) - T2(x), 1e-6, 1 - 1e-6, xtol=1e-10)
    return {"x1": float(x_e), "T": T1(x_e)}


def liquidus_curves(Tm1: float, dH1: float, Tm2: float, dH2: float, gamma=None, n: int = 101) -> dict:
    """Both liquidus branches of an eutectic system over x1 in [0, 1]."""
    e = eutectic(Tm1, dH1, Tm2, dH2, gamma)
    x_a = np.linspace(e["x1"], 1.0, n)
    x_b = np.linspace(0.0, e["x1"], n)
    g1 = (lambda xx, T: gamma([xx, 1 - xx], T)[0]) if gamma else 1.0
    g2 = (lambda xx, T: gamma([1 - xx, xx], T)[1]) if gamma else 1.0
    T_a = [liquidus_T(x, Tm1, dH1, g1) if x > 0 else np.nan for x in x_a]
    T_b = [liquidus_T(1 - x, Tm2, dH2, g2) if x < 1 else np.nan for x in x_b]
    return {"eutectic": e, "x1_branch1": x_a, "T_branch1": np.array(T_a), "x1_branch2": x_b, "T_branch2": np.array(T_b)}


def lens_diagram(Tm1: float, dH1: float, Tm2: float, dH2: float, n: int = 101) -> dict:
    """Liquidus and solidus of an ideal solid solution in equilibrium with an ideal liquid solution (e.g. Cu-Ni).
    With K_i = x_i^S / x_i^L = exp((dH_i / R)(1/T - 1/Tm_i)): x_1^L = (1 - K_2)/(K_1 - K_2), x_1^S = K_1 x_1^L."""
    T = np.linspace(min(Tm1, Tm2), max(Tm1, Tm2), n)
    K1, K2 = np.exp(dH1 / R * (1 / T - 1 / Tm1)), np.exp(dH2 / R * (1 / T - 1 / Tm2))
    with np.errstate(divide="ignore", invalid="ignore"):
        xL = (1 - K2) / (K1 - K2)
    xS = K1 * xL
    ok = (xL >= -1e-9) & (xL <= 1 + 1e-9)
    return {"T": T[ok], "x1_liquidus": np.clip(xL[ok], 0, 1), "x1_solidus": np.clip(xS[ok], 0, 1)}
