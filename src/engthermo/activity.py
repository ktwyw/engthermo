"""Activity-coefficient models for liquid mixtures: Margules (one- and two-parameter), van Laar, Wilson, NRTL
and UNIQUAC (binary and multicomponent where the model allows), excess Gibbs energy, fitting of model
parameters to VLE data, azeotrope location and the Redlich-Kister area consistency test.

Every model is exposed as a function gamma(x, T) -> array, the form used by ``vle``:
    >>> from engthermo import activity
    >>> g = activity.margules(A12=1.2, A21=0.8)
    >>> [round(float(v), 4) for v in g([0.3, 0.7], 350.0)]
    [1.6006, 1.1302]
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize

from .constants import R


def _x(x):
    x = np.asarray(x, float)
    return x / x.sum()


def margules(A12: float, A21: float | None = None):
    """Two-parameter Margules (dimensionless A12, A21; A21 = A12 gives the one-parameter form):
    ln g1 = x2^2 [A12 + 2 (A21 - A12) x1], ln g2 = x1^2 [A21 + 2 (A12 - A21) x2]."""
    A21 = A12 if A21 is None else A21

    def gamma(x, T=None):
        x1, x2 = _x(x)
        return np.exp([x2**2 * (A12 + 2 * (A21 - A12) * x1), x1**2 * (A21 + 2 * (A12 - A21) * x2)])
    gamma.params, gamma.model = {"A12": A12, "A21": A21}, "margules"
    return gamma


def van_laar(A12: float, A21: float):
    """van Laar: ln g1 = A12 (A21 x2 / (A12 x1 + A21 x2))^2, ln g2 = A21 (A12 x1 / (A12 x1 + A21 x2))^2."""
    def gamma(x, T=None):
        x1, x2 = _x(x)
        d = A12 * x1 + A21 * x2
        return np.exp([A12 * (A21 * x2 / d) ** 2, A21 * (A12 * x1 / d) ** 2])
    gamma.params, gamma.model = {"A12": A12, "A21": A21}, "van_laar"
    return gamma


def wilson(Lambda):
    """Wilson equation with a matrix of parameters Lambda_ij (Lambda_ii = 1), any number of components:
    ln g_i = 1 - ln(sum_j x_j L_ij) - sum_k x_k L_ki / sum_j x_j L_kj. Cannot describe liquid-liquid splitting."""
    L = np.asarray(Lambda, float)

    def gamma(x, T=None):
        x = _x(x)
        s = L @ x
        return np.exp(1 - np.log(s) - (x / s) @ L)
    gamma.params, gamma.model = {"Lambda": L}, "wilson"
    return gamma


def wilson_from_energies(a, V, T):
    """Lambda_ij = (V_j / V_i) exp(-a_ij / (R T)) from interaction energies a_ij (J/mol) and molar volumes V (m3/mol),
    with Lambda_ii = 1."""
    a, V = np.asarray(a, float), np.asarray(V, float)
    L = np.outer(1 / V, V) * np.exp(-a / (R * T))
    np.fill_diagonal(L, 1.0)
    return L


def nrtl(tau, alpha):
    """NRTL with matrices tau_ij (dimensionless, tau_ii = 0) and alpha_ij (symmetric), any number of components."""
    tau, alpha = np.asarray(tau, float), np.asarray(alpha, float)
    G = np.exp(-alpha * tau)

    def gamma(x, T=None):
        x = _x(x)
        S = G.T @ x                                            # S_i = sum_j x_j G_ji
        C = (tau * G).T @ x                                    # C_i = sum_j x_j tau_ji G_ji
        term = C / S
        cross = np.array([np.sum(x * G[i, :] / S * (tau[i, :] - C / S)) for i in range(x.size)])
        return np.exp(term + cross)
    gamma.params, gamma.model = {"tau": tau, "alpha": alpha}, "nrtl"
    return gamma


def nrtl_from_energies(dg, alpha, T):
    """tau_ij = dg_ij / (R T) from interaction energies (J/mol), for use with ``nrtl``."""
    return np.asarray(dg, float) / (R * T)


def uniquac(r, q, tau):
    """UNIQUAC with pure-component size r_i, area q_i and interaction matrix tau_ij (tau_ii = 1), any number of
    components; z = 10."""
    r, q, tau = np.asarray(r, float), np.asarray(q, float), np.asarray(tau, float)
    z = 10.0

    def gamma(x, T=None):
        x = _x(x)
        phi, theta = r * x / (r @ x), q * x / (q @ x)
        lseg = z / 2 * (r - q) - (r - 1)
        ln_C = np.log(phi / x) + z / 2 * q * np.log(theta / phi) + lseg - phi / x * (x @ lseg)
        S = tau.T @ theta                                      # S_i = sum_j theta_j tau_ji
        ln_R = q * (1 - np.log(S) - (tau @ (theta / S)))
        return np.exp(ln_C + ln_R)
    gamma.params, gamma.model = {"r": r, "q": q, "tau": tau}, "uniquac"
    return gamma


def excess_gibbs(gamma, x, T: float) -> float:
    """G^E / RT = sum x_i ln gamma_i."""
    x = _x(x)
    return float(np.sum(x * np.log(gamma(x, T))))


def infinite_dilution(gamma, T: float, eps: float = 1e-9) -> tuple[float, float]:
    """Limiting activity coefficients gamma_1^inf and gamma_2^inf of a binary."""
    return float(gamma([eps, 1 - eps], T)[0]), float(gamma([1 - eps, eps], T)[1])


def redlich_kister_area(gamma, T: float) -> dict:
    """Area consistency test: for consistent binary data integral_0^1 ln(gamma1/gamma2) dx1 = 0 (Gibbs-Duhem).
    Returns the positive and negative areas and the index |A+ - A-| / (A+ + A-) (< 0.1 is the usual pass)."""
    f = lambda x1: np.log(gamma([x1, 1 - x1], T)[0] / gamma([x1, 1 - x1], T)[1])  # noqa: E731
    pos = integrate.quad(lambda x: max(f(x), 0.0), 1e-9, 1 - 1e-9, limit=200)[0]
    neg = -integrate.quad(lambda x: min(f(x), 0.0), 1e-9, 1 - 1e-9, limit=200)[0]
    return {"area_plus": pos, "area_minus": neg, "index": abs(pos - neg) / (pos + neg) if pos + neg > 0 else 0.0}


def fit(model: str, comps, x1, T, p, y1=None, p0=None, alpha: float = 0.3) -> dict:
    """Fit a binary model ('margules', 'van_laar', 'wilson' or 'nrtl') to isothermal or isobaric VLE data
    (x1, T, p, optionally y1) by least squares on ln(p_calc / p_exp) (and ln(y1_calc / y1_exp) when given),
    using modified Raoult's law with Antoine vapour pressures. Returns the fitted gamma function and parameters."""
    from .vle import bubble_P

    x1 = np.atleast_1d(np.asarray(x1, float))
    T, p = np.atleast_1d(np.asarray(T, float)), np.atleast_1d(np.asarray(p, float))
    T = np.broadcast_to(T, x1.shape)
    p = np.broadcast_to(p, x1.shape)

    def make(theta):
        if model == "margules":
            return margules(theta[0], theta[1])
        if model == "van_laar":
            return van_laar(theta[0], theta[1])
        if model == "wilson":
            return wilson([[1.0, theta[0]], [theta[1], 1.0]])
        if model == "nrtl":
            return nrtl([[0.0, theta[0]], [theta[1], 0.0]], [[0.0, alpha], [alpha, 0.0]])
        raise KeyError(f"Unknown model {model!r}.")

    def resid(theta):
        g = make(theta)
        out = []
        for xi, Ti, pi in zip(x1, T, p):
            r = bubble_P(comps, [xi, 1 - xi], Ti, g)
            out.append(np.log(r["p"] / pi))
            if y1 is not None:
                y_exp = float(np.asarray(y1)[list(x1).index(xi)])
                out.append(np.log(max(r["y"][0], 1e-12) / max(y_exp, 1e-12)) if xi > 0 else 0.0)
        return np.array(out)

    if model not in ("margules", "van_laar", "wilson", "nrtl"):
        raise KeyError(f"Unknown model {model!r}.")
    start = [0.5, 0.5] if p0 is None else p0
    bounds = ([0.01, 0.01], [10.0, 10.0]) if model == "wilson" else ([-5.0, -5.0], [5.0, 5.0])
    sol = optimize.least_squares(resid, start, bounds=bounds, x_scale="jac")
    g = make(sol.x)
    return {"gamma": g, "params": g.params, "model": model, "rms_p": float(np.sqrt(np.mean(sol.fun**2)))}


def azeotrope(comps, gamma, p: float | None = None, T: float | None = None) -> dict | None:
    """Locate a binary azeotrope (where y1 = x1, i.e. gamma1 p1sat = gamma2 p2sat) at pressure p (returns T) or
    temperature T (returns p). None if the mixture has no azeotrope."""
    from .vapor import antoine_psat
    from .vle import bubble_P, bubble_T

    def f(x1):
        Tloc = T if T is not None else bubble_T(comps, [x1, 1 - x1], p, gamma)["T"]
        g = gamma([x1, 1 - x1], Tloc)
        return np.log(g[0] * antoine_psat(comps[0], Tloc) / (g[1] * antoine_psat(comps[1], Tloc)))

    grid = np.linspace(0.005, 0.995, 60)
    vals = np.array([f(v) for v in grid])
    idx = np.nonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if idx.size == 0:
        return None
    x_az = optimize.brentq(f, grid[idx[0]], grid[idx[0] + 1], xtol=1e-10)
    if T is not None:
        return {"x1": x_az, "p": bubble_P(comps, [x_az, 1 - x_az], T, gamma)["p"], "T": T}
    return {"x1": x_az, "T": bubble_T(comps, [x_az, 1 - x_az], p, gamma)["T"], "p": p}
