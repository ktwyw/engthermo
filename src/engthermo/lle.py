"""Liquid-liquid equilibrium of binaries from an activity-coefficient model (equal activities in both phases),
and the Flory-Huggins theory of polymer solutions.

>>> from engthermo import activity, lle
>>> g = activity.margules(2.5)                         # symmetric, A = 2.5 > 2: two liquid phases
>>> r = lle.binary(g, T=300.0)
>>> round(float(r["x1_alpha"]), 4), round(float(r["x1_beta"]), 4)
(0.1448, 0.8552)
"""

from __future__ import annotations

import numpy as np
from scipy import optimize


def gibbs_of_mixing(gamma, x1, T: float):
    """Delta G_mix / RT = sum x_i ln(x_i gamma_i) of a binary (an array for an array of x1)."""
    x1 = np.atleast_1d(np.asarray(x1, float))
    def one(xi):
        g = gamma([xi, 1 - xi], T)
        return xi * np.log(xi * g[0]) + (1 - xi) * np.log((1 - xi) * g[1]) if 0 < xi < 1 else 0.0
    return np.array([one(xi) for xi in x1])


def binary(gamma, T: float) -> dict | None:
    """Compositions of the two coexisting liquid phases of a binary at T, from the isoactivity equations
    x_i^a gamma_i^a = x_i^b gamma_i^b, i = 1, 2. Starts from the common-tangent of the Gibbs energy of mixing
    curve; returns None if the liquid is completely miscible."""
    grid = np.linspace(1e-4, 1 - 1e-4, 400)
    g = gibbs_of_mixing(gamma, grid, T)
    # two-phase if the curve is not convex: check the lower convex hull
    d2 = np.gradient(np.gradient(g, grid), grid)
    if np.all(d2 > -1e-9):
        return None
    lo_idx = np.argmax(d2 < 0)
    hi_idx = len(grid) - 1 - np.argmax(d2[::-1] < 0)
    a0, b0 = grid[max(lo_idx // 2, 1)], grid[min((hi_idx + len(grid) - 1) // 2, len(grid) - 2)]

    def eqs(v):
        a, b = v
        ga, gb = gamma([a, 1 - a], T), gamma([b, 1 - b], T)
        return [np.log(a * ga[0]) - np.log(b * gb[0]), np.log((1 - a) * ga[1]) - np.log((1 - b) * gb[1])]

    sol = optimize.root(eqs, [a0, b0], method="hybr", tol=1e-12)
    a, b = sorted(sol.x)
    if not sol.success or b - a < 1e-6 or not (0 < a < 1 and 0 < b < 1):
        return None
    return {"x1_alpha": float(a), "x1_beta": float(b), "T": T}


def flory_huggins_activity(phi_p, chi: float, N: float) -> dict:
    """Flory-Huggins solvent (1) and polymer (2) activities for polymer volume fraction phi_p, interaction parameter
    chi and chain length N (segments per polymer molecule relative to the solvent):
    ln a_1 = ln(1 - phi) + (1 - 1/N) phi + chi phi^2."""
    phi = np.asarray(phi_p, float)
    ln_a1 = np.log(1 - phi) + (1 - 1 / N) * phi + chi * phi**2
    ln_a2 = np.log(phi) - (N - 1) * (1 - phi) + chi * N * (1 - phi) ** 2
    return {"ln_a_solvent": ln_a1, "ln_a_polymer": ln_a2}


def flory_huggins_gibbs(phi_p, chi: float, N: float):
    """Gibbs energy of mixing per lattice site:
    Delta G / (RT) = (1 - phi) ln(1 - phi) + (phi / N) ln phi + chi phi (1 - phi)."""
    phi = np.asarray(phi_p, float)
    return (1 - phi) * np.log(1 - phi) + phi / N * np.log(phi) + chi * phi * (1 - phi)


def flory_huggins_critical(N: float) -> dict:
    """Critical point of the Flory-Huggins solution: chi_c = (1 + 1/sqrt(N))^2 / 2, phi_c = 1 / (1 + sqrt(N))."""
    return {"chi_c": (1 + 1 / np.sqrt(N)) ** 2 / 2, "phi_c": 1 / (1 + np.sqrt(N))}


def flory_huggins_binodal(chi: float, N: float) -> dict | None:
    """Coexisting polymer volume fractions (dilute and concentrated phases) for chi above chi_c, from equal
    solvent and polymer activities. Solved in transformed variables (ln phi for the dilute phase, logit for the
    concentrated one) starting outside the spinodal points."""
    if chi <= flory_huggins_critical(N)["chi_c"]:
        return None
    sp = flory_huggins_spinodal(chi, N)

    def eqs(v):
        a, b = np.exp(v[0]), 1 / (1 + np.exp(-v[1]))
        A, B = flory_huggins_activity(a, chi, N), flory_huggins_activity(b, chi, N)
        return [A["ln_a_solvent"] - B["ln_a_solvent"], (A["ln_a_polymer"] - B["ln_a_polymer"]) / N]

    with np.errstate(all="ignore"):
        for f in (0.5, 0.8, 0.95, 0.3, 0.1):           # starts progressively closer to (then further from) the spinodal
            a0, b0 = f * sp[0], sp[1] + (1 - f) * (1 - sp[1])
            sol = optimize.root(eqs, [np.log(a0), np.log(b0 / (1 - b0))], tol=1e-12)
            a, b = np.exp(sol.x[0]), 1 / (1 + np.exp(-sol.x[1]))
            if sol.success and np.all(np.isfinite(sol.fun)) and 0 < a < sp[0] and sp[1] < b < 1:
                return {"phi_dilute": float(a), "phi_concentrated": float(b), "chi": chi, "N": N}
    # near the critical point the equations are nearly degenerate: continuation from a larger chi
    chi_c = flory_huggins_critical(N)["chi_c"]
    if chi < chi_c + 0.5 * (chi_c if chi_c > 1 else 1.0):
        start_chi = chi_c + 0.5 * (chi_c if chi_c > 1 else 1.0)
        guess = flory_huggins_binodal(start_chi, N)
        v = [np.log(guess["phi_dilute"]), np.log(guess["phi_concentrated"] / (1 - guess["phi_concentrated"]))]
        for c in np.linspace(start_chi, chi, 40)[1:]:
            def eqs_c(w, c=c):
                a, b = np.exp(w[0]), 1 / (1 + np.exp(-w[1]))
                A, B = flory_huggins_activity(a, c, N), flory_huggins_activity(b, c, N)
                return [A["ln_a_solvent"] - B["ln_a_solvent"], (A["ln_a_polymer"] - B["ln_a_polymer"]) / N]
            with np.errstate(all="ignore"):
                sol = optimize.root(eqs_c, v, tol=1e-12)
            if not sol.success:
                break
            v = sol.x
        else:
            a, b = np.exp(v[0]), 1 / (1 + np.exp(-v[1]))
            if 0 < a < sp[0] and sp[1] < b < 1:
                return {"phi_dilute": float(a), "phi_concentrated": float(b), "chi": chi, "N": N}
    raise RuntimeError("Binodal did not converge.")


def flory_huggins_spinodal(chi: float, N: float) -> np.ndarray:
    """Spinodal compositions: roots of d2(Delta G / RT)/d phi2 = 1/(1 - phi) + 1/(N phi) - 2 chi = 0, i.e. of
    2 chi N phi^2 + (N - 1 - 2 chi N) phi + 1 = 0 (empty below the critical chi)."""
    a, b, c = 2 * chi * N, N - 1 - 2 * chi * N, 1.0
    disc = b**2 - 4 * a * c
    if disc < 0:
        return np.array([])
    return np.sort(np.array([(-b - np.sqrt(disc)) / (2 * a), (-b + np.sqrt(disc)) / (2 * a)]))


def chi_from_temperature(T, A: float, B: float):
    """A common temperature dependence chi = A + B / T (B > 0 gives an upper critical solution temperature)."""
    return A + B / np.asarray(T, float)


def ucst(A: float, B: float, N: float) -> float:
    """Upper critical solution temperature (K) for chi = A + B/T: the T at which chi = chi_c."""
    return float(B / (flory_huggins_critical(N)["chi_c"] - A))


