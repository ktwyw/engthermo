"""Chemical reaction equilibrium: standard enthalpy, entropy and Gibbs energy of reaction as functions of
temperature (from formation data and ideal-gas heat capacities), equilibrium constants, the equilibrium
extent of a single ideal-gas reaction, and Gibbs-energy minimisation for any number of simultaneous
reactions (ideal gas, 1 bar standard state).

A reaction is given as {species: stoichiometric coefficient}, negative for reactants:
    >>> from engthermo import reaction
    >>> K = reaction.equilibrium_constant({"N2": -1, "H2": -3, "NH3": 2}, T=298.15)
    >>> round(float(np.log10(K)), 2)                                       # ammonia synthesis at 25 degC
    5.76
"""

from __future__ import annotations

import re

import numpy as np
from scipy import optimize

from . import idealgas
from .components import get
from .constants import P_STD, T_STD, R

# formation properties not in the component table: (Hf, Gf, constant Cp) - liquid water (JANAF/TRC via Smith, Van Ness &
# Abbott Table C.4; Cp of liquid water 75.3 J/(mol K))
EXTRA_FORMATION = {"H2O(l)": (-285830.0, -237129.0, 75.3)}


def _formation(name: str):
    if name in EXTRA_FORMATION:
        return EXTRA_FORMATION[name][:2]
    c = get(name)
    if c.Hf is None:
        raise ValueError(f"No formation properties for {name}.")
    return c.Hf, c.Gf


def _extra_dh(name: str, T: float) -> float:
    """Sensible enthalpy of an EXTRA_FORMATION species (constant Cp) from 298.15 K to T."""
    return EXTRA_FORMATION[name][2] * (T - T_STD)


def _extra_ds(name: str, T: float) -> float:
    return EXTRA_FORMATION[name][2] * np.log(T / T_STD)


def standard_enthalpy(rxn: dict, T: float = T_STD) -> float:
    """Delta H° of reaction at T (J per mole of reaction): formation enthalpies at 298.15 K plus the ideal-gas
    Cp integrals of every species (a constant Cp for liquid water)."""
    dH = sum(nu * _formation(s)[0] for s, nu in rxn.items())
    if T != T_STD:
        dH += sum(nu * (_extra_dh(s, T) if s in EXTRA_FORMATION else idealgas.enthalpy_change(s, T_STD, T))
                  for s, nu in rxn.items())
    return float(dH)


def standard_entropy(rxn: dict, T: float = T_STD) -> float:
    """Delta S° of reaction at T (J/(mol K)), from (Delta H° - Delta G°)/T at 298.15 K plus the Cp/T integrals."""
    dH298 = sum(nu * _formation(s)[0] for s, nu in rxn.items())
    dG298 = sum(nu * _formation(s)[1] for s, nu in rxn.items())
    dS = (dH298 - dG298) / T_STD
    if T != T_STD:
        dS += sum(nu * (_extra_ds(s, T) if s in EXTRA_FORMATION else idealgas.entropy_change(s, T_STD, T))
                  for s, nu in rxn.items())
    return float(dS)


def standard_gibbs(rxn: dict, T: float = T_STD) -> float:
    """Delta G° of reaction at T (J/mol) = Delta H°(T) - T Delta S°(T) (1 bar ideal-gas standard state)."""
    return standard_enthalpy(rxn, T) - T * standard_entropy(rxn, T)


def equilibrium_constant(rxn: dict, T: float = T_STD) -> float:
    """K(T) = exp(-Delta G°(T) / RT), the thermodynamic equilibrium constant (dimensionless, p° = 1 bar)."""
    return float(np.exp(-standard_gibbs(rxn, T) / (R * T)))


def van_t_hoff(rxn: dict, T1: float, T2: float) -> float:
    """K(T2) from K(T1) assuming a constant heat of reaction (the van 't Hoff approximation), for comparison
    with the exact temperature dependence of ``equilibrium_constant``."""
    dH = standard_enthalpy(rxn, T1)
    return float(equilibrium_constant(rxn, T1) * np.exp(-dH / R * (1 / T2 - 1 / T1)))


def extent(rxn: dict, feed: dict, T: float, p: float = P_STD) -> dict:
    """Equilibrium extent of a single ideal-gas reaction from a feed {species: moles}: solves
    K = prod (y_i p/p°)^nu_i for the extent xi. Returns xi, the equilibrium moles and mole fractions."""
    species = list({*rxn, *feed})
    n0 = np.array([feed.get(s, 0.0) for s in species], float)
    nu = np.array([rxn.get(s, 0.0) for s in species], float)
    K = equilibrium_constant(rxn, T)
    # extent limits: no species negative
    lo = max((-n0[i] / nu[i] for i in range(len(species)) if nu[i] > 0), default=-np.inf)
    hi = min((-n0[i] / nu[i] for i in range(len(species)) if nu[i] < 0), default=np.inf)
    eps = 1e-12 * max(hi - lo, 1.0)

    def f(xi):
        n = n0 + nu * xi
        y = n / n.sum()
        with np.errstate(divide="ignore"):
            return float(np.sum(nu * np.log(np.maximum(y, 1e-300) * p / P_STD)) - np.log(K))

    xi = optimize.brentq(f, lo + eps, hi - eps, xtol=1e-14)
    n = n0 + nu * xi
    return {"extent": xi, "moles": dict(zip(species, n)), "y": dict(zip(species, n / n.sum())), "K": K, "T": T, "p": p}


def parse_formula(formula: str) -> dict:
    """Element counts of a formula such as 'C2H6O' or 'CH4'."""
    return {el: (int(n) if n else 1) for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula)}


def gibbs_minimization(species, feed: dict, T: float, p: float = P_STD) -> dict:
    """Equilibrium composition of an ideal-gas mixture of the given species at T and p by minimising the total
    Gibbs energy subject to element balances - the general method for many simultaneous reactions, solved with
    Lagrange multipliers (Smith, Van Ness & Abbott, section 13.9). Returns equilibrium moles and mole fractions."""
    species = list(species)
    comps = [get(s) for s in species]
    elements = sorted({el for c in comps for el in parse_formula(c.formula)})
    A = np.array([[parse_formula(c.formula).get(el, 0) for c in comps] for el in elements], float)   # element matrix
    n0 = np.array([feed.get(s, 0.0) for s in species], float)
    b = A @ n0
    # G°_i(T) = H_i(T) - T S_i(T) with H_i = Hf + int Cp dT and S_i = (Hf - Gf)/298.15 + int Cp/T dT (formation basis);
    # at 298.15 K this reduces to Gf, as it must
    g0 = np.array([_formation(s)[0] + idealgas.enthalpy_change(s, T_STD, T)
                   - T * ((_formation(s)[0] - _formation(s)[1]) / T_STD + idealgas.entropy_change(s, T_STD, T))
                   for s in species])                                                                 # J/mol

    # 1. the ideal-gas Gibbs energy is convex in the mole numbers and the element balances are linear, so a
    #    constrained minimiser finds the global minimum from any start
    def total_G(n):
        n = np.maximum(n, 1e-30)
        return float(np.sum(n * (g0 / (R * T) + np.log(n / n.sum() * p / P_STD))))

    def grad(n):
        n = np.maximum(n, 1e-30)
        return g0 / (R * T) + np.log(n / n.sum() * p / P_STD)

    scale = max(n0.sum(), 1e-12)
    mini = optimize.minimize(total_G, np.maximum(n0, 1e-3 * scale), jac=grad, method="SLSQP",
                             constraints=[{"type": "eq", "fun": lambda n: (A @ n - b) / scale,
                                           "jac": lambda n: A / scale}],
                             bounds=[(1e-15, None)] * len(species), options={"ftol": 1e-15, "maxiter": 2000})
    n = np.maximum(mini.x, 1e-15)

    # 2. polish with the Lagrange-multiplier stationarity equations (Smith, Van Ness & Abbott 13.9), from that start
    def eqs(v):
        nn, lam = np.exp(np.clip(v[: len(species)], -60, 60)), v[len(species):]
        stat = g0 / (R * T) + np.log(nn / nn.sum() * p / P_STD) + A.T @ lam
        return np.r_[stat, (A @ nn - b) / scale]

    lam0 = np.linalg.lstsq(A.T, -grad(n), rcond=None)[0]
    sol = optimize.root(eqs, np.r_[np.log(n), lam0], method="hybr", tol=1e-13)
    if sol.success and np.max(np.abs(sol.fun)) < 1e-9:
        n = np.exp(sol.x[: len(species)])
    elif not mini.success or np.max(np.abs(A @ n - b)) > 1e-8 * scale:
        raise RuntimeError("Gibbs minimisation did not converge.")
    res = mini
    return {"moles": dict(zip(species, n)), "y": dict(zip(species, n / n.sum())), "T": T, "p": p, "elements": elements,
            "success": bool(res.success)}
