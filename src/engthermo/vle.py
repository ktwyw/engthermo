"""Vapour-liquid equilibrium of mixtures at low pressure: Raoult's law and modified Raoult's law
(y_i p = x_i gamma_i p_sat_i), bubble and dew points, and isothermal flash by the Rachford-Rice equation.
``gamma`` is an optional function gamma(x, T) -> array of activity coefficients (default: ideal solution).

>>> from engthermo import vle
>>> r = vle.bubble_T(["benzene", "toluene"], [0.5, 0.5], p=101325.0)
>>> round(r["T"] - 273.15, 1), round(float(r["y"][0]), 3)
(92.1, 0.714)
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from .components import Component, get
from .vapor import antoine_psat, antoine_Tsat


def _comps(cs) -> list[Component]:
    return [c if isinstance(c, Component) else get(c) for c in cs]


def _psats(comps, T):
    return np.array([float(antoine_psat(c, T)) for c in comps])


def _gammas(gamma, x, T):
    return np.ones(len(x)) if gamma is None else np.asarray(gamma(np.asarray(x, float), T), float)


def bubble_P(comps, x, T: float, gamma=None) -> dict:
    """Bubble pressure and vapour composition at T for liquid composition x."""
    comps, x = _comps(comps), np.asarray(x, float)
    ps, g = _psats(comps, T), _gammas(gamma, x, T)
    p = float(np.sum(x * g * ps))
    return {"p": p, "y": x * g * ps / p, "T": T, "K": g * ps / p}


def dew_P(comps, y, T: float, gamma=None) -> dict:
    """Dew pressure and liquid composition at T for vapour composition y (iterates when gamma depends on x)."""
    comps, y = _comps(comps), np.asarray(y, float)
    ps = _psats(comps, T)
    x = y / ps / np.sum(y / ps)
    for _ in range(200):
        g = _gammas(gamma, x, T)
        p = 1 / np.sum(y / (g * ps))
        x_new = y * p / (g * ps)
        if np.max(np.abs(x_new - x)) < 1e-12:
            break
        x = x_new
    return {"p": float(p), "x": x_new, "T": T}


def bubble_T(comps, x, p: float, gamma=None) -> dict:
    """Bubble temperature and vapour composition at pressure p for liquid composition x."""
    comps, x = _comps(comps), np.asarray(x, float)
    Ts = [antoine_Tsat(c, p) for c in comps]
    T = float(optimize.brentq(lambda T: bubble_P(comps, x, T, gamma)["p"] - p, min(Ts) - 30, max(Ts) + 30, xtol=1e-10))
    return {"T": T, "y": bubble_P(comps, x, T, gamma)["y"], "p": p}


def dew_T(comps, y, p: float, gamma=None) -> dict:
    """Dew temperature and liquid composition at pressure p for vapour composition y."""
    comps, y = _comps(comps), np.asarray(y, float)
    Ts = [antoine_Tsat(c, p) for c in comps]
    T = float(optimize.brentq(lambda T: dew_P(comps, y, T, gamma)["p"] - p, min(Ts) - 30, max(Ts) + 30, xtol=1e-10))
    return {"T": T, "x": dew_P(comps, y, T, gamma)["x"], "p": p}


def rachford_rice(z, K) -> float:
    """Vapour fraction V/F from feed composition z and K-values (solves the Rachford-Rice equation)."""
    z, K = np.asarray(z, float), np.asarray(K, float)
    f = lambda V: np.sum(z * (K - 1) / (1 + V * (K - 1)))  # noqa: E731
    if f(0.0) <= 0:
        return 0.0
    if f(1.0) >= 0:
        return 1.0
    return float(optimize.brentq(f, 0.0, 1.0, xtol=1e-12))


def flash(comps, z, T: float, p: float, gamma=None) -> dict:
    """Isothermal flash at T and p: vapour fraction, liquid and vapour compositions (successive substitution
    on the activity coefficients when they depend on composition)."""
    comps, z = _comps(comps), np.asarray(z, float)
    ps = _psats(comps, T)
    x = z.copy()
    for _ in range(300):
        K = _gammas(gamma, x, T) * ps / p
        V = rachford_rice(z, K)
        x_new = z / (1 + V * (K - 1))
        x_new = x_new / x_new.sum()
        if np.max(np.abs(x_new - x)) < 1e-12:
            break
        x = x_new
    y = K * x_new
    return {"V": V, "x": x_new, "y": y / y.sum(), "K": K, "T": T, "p": p,
            "phase": "liquid" if V == 0 else "vapour" if V == 1 else "two-phase"}


def Txy(comps, p: float, gamma=None, n: int = 41) -> dict:
    """T-x-y diagram data of a binary at pressure p: x1, y1 and T along the bubble curve."""
    x1 = np.linspace(0, 1, n)
    T, y = [], []
    for v in x1:
        r = bubble_T(comps, [v, 1 - v], p, gamma)
        T.append(r["T"])
        y.append(r["y"][0])
    return {"x1": x1, "y1": np.array(y), "T": np.array(T)}


def Pxy(comps, T: float, gamma=None, n: int = 41) -> dict:
    """P-x-y diagram data of a binary at temperature T."""
    x1 = np.linspace(0, 1, n)
    p, y = [], []
    for v in x1:
        r = bubble_P(comps, [v, 1 - v], T, gamma)
        p.append(r["p"])
        y.append(r["y"][0])
    return {"x1": x1, "y1": np.array(y), "p": np.array(p)}


# ------------------------------------------------------------------ high pressure: equation of state for both phases
def _bubble_residual(name, comps, x, T, p, kij, y0, n_iter=200):
    """At fixed p: iterate the vapour composition to convergence and return (sum K_i x_i - 1, y, K)."""
    from . import eos

    y = y0.copy()
    for _ in range(n_iter):
        liq = eos.mixture(name, comps, x, T, p, kij, "liquid")
        vap = eos.mixture(name, comps, y, T, p, kij, "vapor")
        K = liq["phi_i"] / vap["phi_i"]
        y_new = _x_norm(K * x)
        if np.max(np.abs(y_new - y)) < 1e-12:
            y = y_new
            break
        y = y_new
    return float(np.sum(K * x) - 1), y, K


def eos_bubble_P(name: str, comps, x, T: float, kij=None, p0: float | None = None) -> dict:
    """Bubble pressure at T of a liquid of composition x with a cubic equation of state for both phases
    (phi-phi formulation): y_i phi_i^V = x_i phi_i^L. Successive substitution on the K-values from a Raoult's-law
    start; if that drifts to the trivial solution, the pressure is scanned for a sign change of the residual
    sum(K_i x_i) - 1 (vapour composition converged at each pressure) and the root is refined."""
    from . import eos
    from .vapor import antoine_psat

    comps, x = _comps(comps), _x_norm(x)
    ps = np.array([antoine_psat(c, min(T, 0.95 * c.Tc)) if c.antoine else 0.5 * c.Pc for c in comps])
    if p0 is None:
        p0 = float(np.sum(x * ps))                  # Raoult's law as the starting point
    p, y = p0, _x_norm(x * ps / p0)
    for _ in range(500):
        try:
            liq = eos.mixture(name, comps, x, T, p, kij, "liquid")
            vap = eos.mixture(name, comps, y, T, p, kij, "vapor")
        except ValueError:
            break
        K = liq["phi_i"] / vap["phi_i"]
        s = float(np.sum(K * x))
        y_new = K * x / s
        p_new = p * s
        if abs(p_new / p - 1) < 1e-11 and np.max(np.abs(y_new - y)) < 1e-11:
            p, y = p_new, y_new
            break
        p, y = p_new, y_new
    else:
        K = np.ones(x.size)
    if np.max(np.abs(K - 1)) > 1e-6 and np.isfinite(p):
        return {"p": float(p), "y": y, "K": K, "T": T}
    # fallback: bracket the bubble pressure by scanning
    residual = lambda pp, y0: _bubble_residual(name, comps, x, T, pp, kij, y0)  # noqa: E731
    result = _scan_pressure(residual, lambda pp: _x_norm(x * ps / pp), 3.0 * max(c.Pc for c in comps))
    if result is None:
        raise ValueError("Trivial solution (identical phases): no two-phase state at this T and composition - "
                         "probably above the mixture's critical region.")
    p_sol, y, K = result
    return {"p": float(p_sol), "y": y, "K": K, "T": T}


def _scan_pressure(residual, fresh_start, p_max, n: int = 80):
    """Bracket a bubble or dew pressure by scanning the residual over a log grid of pressures. ``residual(p, z0)``
    returns (residual, composition, K); a point whose iteration collapses to the trivial solution is retried from
    ``fresh_start(p)``, and an interval between a valid point and a trivial one is sub-scanned before giving up."""
    def evaluate(pp, z0):
        try:
            res, z, K = residual(pp, z0)
        except ValueError:
            return None
        if np.max(np.abs(K - 1)) < 1e-6:
            try:
                res, z, K = residual(pp, fresh_start(pp))
            except ValueError:
                return None
            if np.max(np.abs(K - 1)) < 1e-6:
                return None
        return res, z, K

    def search(grid, z_guess):
        prev = None
        for pg in grid:
            out = evaluate(pg, z_guess)
            if out is None:
                if prev is not None and pg / prev[0] > 1.02:            # valid -> trivial: look closer between them
                    inner = search(np.geomspace(prev[0], pg, 12)[1:-1], prev[2])
                    if inner is not None:
                        return inner
                prev = None
                continue
            res, z_guess, K = out
            if prev is not None and prev[1] * res < 0:
                f = lambda pp, z0=z_guess: residual(pp, z0)[0]  # noqa: E731
                p_sol = optimize.brentq(f, prev[0], pg, xtol=1e-6, rtol=1e-10)
                _, z, K = residual(p_sol, z_guess)
                return p_sol, z, K
            prev = (pg, res, z_guess)
        return None

    grid = np.geomspace(1e3, p_max, n)
    return search(grid, fresh_start(grid[0]))


def _dew_residual(name, comps, y, T, p, kij, x0, n_iter=200):
    """At fixed p: iterate the liquid composition to convergence and return (sum y_i/K_i - 1, x, K)."""
    from . import eos

    x = x0.copy()
    for _ in range(n_iter):
        liq = eos.mixture(name, comps, x, T, p, kij, "liquid")
        vap = eos.mixture(name, comps, y, T, p, kij, "vapor")
        K = liq["phi_i"] / vap["phi_i"]
        x_new = _x_norm(y / K)
        if np.max(np.abs(x_new - x)) < 1e-12:
            x = x_new
            break
        x = x_new
    return float(np.sum(y / K) - 1), x, K


def eos_dew_P(name: str, comps, y, T: float, kij=None, p_max: float | None = None) -> dict:
    """Dew pressure at T of a vapour of composition y (phi-phi formulation). The pressure is bracketed by scanning
    the residual sum(y_i/K_i) - 1 (liquid composition converged at each pressure), then refined with a root finder;
    for a retrograde gas this finds the lower dew point."""
    from .vapor import antoine_psat

    comps, y = _comps(comps), _x_norm(y)
    ps = np.array([antoine_psat(c, min(T, 0.95 * c.Tc)) if c.antoine else 0.5 * c.Pc for c in comps])
    p_hi = p_max or 3.0 * max(c.Pc for c in comps)          # mixture dew points can exceed the pure-component Pc
    residual = lambda pp, x0: _dew_residual(name, comps, y, T, pp, kij, x0)  # noqa: E731
    result = _scan_pressure(residual, lambda pp: _x_norm(y * pp / ps), p_hi)
    if result is not None:
        p_sol, x, K = result
        return {"p": float(p_sol), "x": x, "K": K, "T": T}
    raise ValueError("No dew point found: the vapour may be above its cricondentherm at this temperature.")


def _x_norm(x):
    x = np.asarray(x, float)
    return x / x.sum()


def eos_Pxy(name: str, comps, T: float, kij=None, n: int = 31, x_max: float = 1.0) -> dict:
    """P-x-y data of a binary at T from the equation of state (bubble curve), up to x1 = x_max."""
    x1 = np.linspace(0, x_max, n)
    p, y = [], []
    for v in x1:
        try:
            r = eos_bubble_P(name, comps, [v, 1 - v], T, kij)
            p.append(r["p"])
            y.append(r["y"][0])
        except (ValueError, RuntimeError):
            p.append(np.nan)
            y.append(np.nan)
    return {"x1": x1, "y1": np.array(y), "p": np.array(p)}
