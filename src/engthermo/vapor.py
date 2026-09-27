"""Vapour pressure and phase change of pure substances: Clausius-Clapeyron, the Antoine equation, and
saturation from a cubic equation of state (equal fugacities of liquid and vapour).

>>> from engthermo import vapor
>>> round(vapor.antoine_psat("water", 373.124) / 1e3, 2)          # kPa at the normal boiling point
101.18
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from . import eos
from .components import Component, get
from .constants import R


def _comp(c) -> Component:
    return c if isinstance(c, Component) else get(c)


def clausius_clapeyron(T, p1: float, T1: float, dH_vap: float):
    """Vapour pressure at T from a known point (T1, p1) and a constant enthalpy of vaporisation (J/mol):
    ln(p/p1) = -(dH_vap/R)(1/T - 1/T1)."""
    out = p1 * np.exp(-dH_vap / R * (1 / np.asarray(T, float) - 1 / T1))
    return float(out) if np.ndim(out) == 0 else out


def antoine_psat(comp, T):
    """Antoine vapour pressure (Pa): ln p = A - B/(T + C) with the component's fitted constants."""
    c = _comp(comp)
    if c.antoine is None:
        raise ValueError(f"No Antoine constants for {c.name}.")
    A, B, C = c.antoine
    out = np.exp(A - B / (np.asarray(T, float) + C))
    return float(out) if np.ndim(out) == 0 else out


def antoine_Tsat(comp, p) -> float:
    """Saturation temperature (K) from the Antoine equation."""
    c = _comp(comp)
    A, B, C = c.antoine
    return float(B / (A - np.log(p)) - C)


def antoine_dH_vap(comp, T):
    """Enthalpy of vaporisation implied by the Antoine equation via Clausius-Clapeyron (ideal vapour, negligible
    liquid volume): dH = R T^2 d ln p / dT = R B (T/(T + C))^2 (J/mol)."""
    c = _comp(comp)
    A, B, C = c.antoine
    T = np.asarray(T, float)
    out = R * B * (T / (T + C)) ** 2
    return float(out) if np.ndim(out) == 0 else out


def eos_psat(name: str, comp, T: float, p_guess: float | None = None) -> dict:
    """Saturation pressure from a cubic equation of state: the pressure at which the liquid and vapour roots have
    equal fugacity. Returns p_sat, the two molar volumes and the enthalpy of vaporisation from the departure
    functions (H_R,vapour - H_R,liquid)."""
    c = _comp(comp)
    if T >= c.Tc:
        raise ValueError("T must be below the critical temperature.")
    p0 = p_guess or (float(antoine_psat(c, T)) if c.antoine else 0.1 * c.Pc)

    def f(lnp):
        p = np.exp(lnp)
        try:
            return eos.solve(name, c, T, p, "liquid")["lnphi"] - eos.solve(name, c, T, p, "vapor")["lnphi"]
        except ValueError:
            return np.nan

    # bracket: scan around the guess until the two roots exist and the difference changes sign
    lo, hi = np.log(p0) - 0.05, np.log(p0) + 0.05
    for _ in range(60):
        flo, fhi = f(lo), f(hi)
        if np.isfinite(flo) and np.isfinite(fhi) and flo * fhi < 0:
            break
        lo, hi = lo - 0.1, hi + 0.1
    else:
        # near the critical point the two-root window is narrow: scan it finely between 0.3 Pc and Pc
        grid = np.linspace(np.log(0.3 * c.Pc), np.log(c.Pc), 400)
        vals = np.array([f(g) for g in grid])
        ok = np.isfinite(vals)
        idx = [i for i in range(grid.size - 1) if ok[i] and ok[i + 1] and vals[i] * vals[i + 1] < 0]
        if not idx:
            raise RuntimeError("Could not bracket the saturation pressure.")
        lo, hi = grid[idx[0]], grid[idx[0] + 1]
    lnp = optimize.brentq(f, lo, hi, xtol=1e-12)
    p = float(np.exp(lnp))
    liq, vap = eos.solve(name, c, T, p, "liquid"), eos.solve(name, c, T, p, "vapor")
    return {"p_sat": p, "V_liquid": liq["V"], "V_vapor": vap["V"], "dH_vap": vap["H_R"] - liq["H_R"], "T": T}
