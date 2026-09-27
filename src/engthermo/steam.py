"""Thermodynamic properties of water and steam: the IAPWS Industrial Formulation 1997 (IF97).

Regions 1 (liquid), 2 (vapour), 3 (near-critical), 4 (saturation line) and 5 (high-temperature steam) of
the IAPWS "Revised Release on the IAPWS Industrial Formulation 1997" (2007) are implemented from the release's
own equations, with the coefficient tables transcribed programmatically (see tools/transcribe_if97.py).
Units are SI: T in K, p in Pa, v in m3/kg, u/h in J/kg, s/cp/cv in J/(kg K), speed of sound w in m/s.

>>> from engthermo import steam
>>> st = steam.properties(T=300.0, p=3e6)                     # IAPWS verification point (Table 5)
>>> round(st["v"], 11), round(st["h"], 3)
(0.00100215168, 115331.273)
>>> round(steam.psat(500.0) / 1e6, 8)                          # Table 35: 2.63889776 MPa
2.63889776
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from . import _if97_coefficients as c

R_KJ = 0.461526          # kJ/(kg K), specific gas constant of water in the release
T_CRIT, P_CRIT, RHO_CRIT = 647.096, 22.064e6, 322.0
T_TRIPLE, P_TRIPLE = 273.16, 611.657


# ------------------------------------------------------------------ region boundaries (T in K, p in MPa)
def _p23(T):
    return c.B23_n[0] + c.B23_n[1] * T + c.B23_n[2] * T**2


def _psat_mpa(T):
    n = c.R4_n
    theta = T + n[8] / (T - n[9])
    A = theta**2 + n[0] * theta + n[1]
    B = n[2] * theta**2 + n[3] * theta + n[4]
    C = n[5] * theta**2 + n[6] * theta + n[7]
    return (2 * C / (-B + np.sqrt(B**2 - 4 * A * C))) ** 4


def _tsat_k(p):
    n = c.R4_n
    beta = p**0.25
    E = beta**2 + n[2] * beta + n[5]
    F = n[0] * beta**2 + n[3] * beta + n[6]
    G = n[1] * beta**2 + n[4] * beta + n[7]
    D = 2 * G / (-F - np.sqrt(F**2 - 4 * E * G))
    return (n[9] + D - np.sqrt((n[9] + D) ** 2 - 4 * (n[8] + n[9] * D))) / 2


def psat(T) -> float:
    """Saturation pressure (Pa) for 273.15 K <= T <= 647.096 K (IF97 Eq. 30)."""
    if not 273.15 <= T <= T_CRIT:
        raise ValueError("T must lie between 273.15 K and the critical temperature 647.096 K.")
    return float(_psat_mpa(T) * 1e6)


def Tsat(p) -> float:
    """Saturation temperature (K) for 611.213 Pa <= p <= 22.064 MPa (IF97 Eq. 31)."""
    if not 611.212 <= p <= P_CRIT:
        raise ValueError("p must lie between the triple-point pressure and the critical pressure 22.064 MPa.")
    return float(_tsat_k(p / 1e6))


def region(T, p) -> int:
    """IF97 region number for a single-phase state (T in K, p in Pa)."""
    pm = p / 1e6
    if p <= 0 or pm > 100 or T < 273.15 or T > 2273.15 or (T > 1073.15 and pm > 50):
        raise ValueError("State outside the range of IF97 (273.15-1073.15 K up to 100 MPa; "
                         "1073.15-2273.15 K up to 50 MPa).")
    if T > 1073.15:
        return 5
    if T <= 623.15:
        return 1 if pm >= _psat_mpa(T) else 2
    if T <= 863.15 and pm > _p23(T):
        return 3
    return 2


# ------------------------------------------------------------------ basic equations
def _gibbs_region1(T, pm):
    pi, tau = pm / 16.53, 1386.0 / T
    a, b = 7.1 - pi, tau - 1.222
    ii, jj, n = c.R1_I, c.R1_J, c.R1_n
    g = np.sum(n * a**ii * b**jj)
    g_p = -np.sum(n * ii * a ** (ii - 1) * b**jj)
    g_pp = np.sum(n * ii * (ii - 1) * a ** (ii - 2) * b**jj)
    g_t = np.sum(n * a**ii * jj * b ** (jj - 1))
    g_tt = np.sum(n * a**ii * jj * (jj - 1) * b ** (jj - 2))
    g_pt = -np.sum(n * ii * a ** (ii - 1) * jj * b ** (jj - 1))
    return _from_gibbs(T, pm, pi, tau, g, g_p, g_pp, g_t, g_tt, g_pt)


def _gibbs_ideal_residual(T, pm, T_star, J0, n0, ii, jj, n, tau_shift):
    pi, tau = pm, T_star / T
    g0 = np.log(pi) + np.sum(n0 * tau**J0)
    g0_t = np.sum(n0 * J0 * tau ** (J0 - 1))
    g0_tt = np.sum(n0 * J0 * (J0 - 1) * tau ** (J0 - 2))
    b = tau - tau_shift
    gr = np.sum(n * pi**ii * b**jj)
    gr_p = np.sum(n * ii * pi ** (ii - 1) * b**jj)
    gr_pp = np.sum(n * ii * (ii - 1) * pi ** (ii - 2) * b**jj)
    gr_t = np.sum(n * pi**ii * jj * b ** (jj - 1))
    gr_tt = np.sum(n * pi**ii * jj * (jj - 1) * b ** (jj - 2))
    gr_pt = np.sum(n * ii * pi ** (ii - 1) * jj * b ** (jj - 1))
    g_p, g_pp = 1 / pi + gr_p, -1 / pi**2 + gr_pp
    out = _from_gibbs(T, pm, pi, tau, g0 + gr, g_p, g_pp, g0_t + gr_t, g0_tt + gr_tt, gr_pt)
    # cv and w have their own forms for the ideal + residual split (IF97 Table 12)
    R = R_KJ
    out["cv"] = R * (-(tau**2) * (g0_tt + gr_tt) - (1 + pi * gr_p - tau * pi * gr_pt) ** 2 / (1 - pi**2 * gr_pp)) * 1e3
    w2 = R * 1e3 * T * (1 + 2 * pi * gr_p + pi**2 * gr_p**2) / (
        (1 - pi**2 * gr_pp) + (1 + pi * gr_p - tau * pi * gr_pt) ** 2 / (tau**2 * (g0_tt + gr_tt)))
    out["w"] = float(np.sqrt(w2))
    return out


def _from_gibbs(T, pm, pi, tau, g, g_p, g_pp, g_t, g_tt, g_pt):
    R = R_KJ
    v = pi * g_p * R * T / (pm * 1e3)                        # m3/kg (R T in kJ/kg, p in MPa -> factor 1e-3)
    out = {"v": float(v), "h": float(tau * g_t * R * T * 1e3), "u": float((tau * g_t - pi * g_p) * R * T * 1e3),
           "s": float((tau * g_t - g) * R * 1e3), "cp": float(-(tau**2) * g_tt * R * 1e3)}
    out["cv"] = float(R * (-(tau**2) * g_tt + (g_p - tau * g_pt) ** 2 / g_pp) * 1e3)
    out["w"] = float(np.sqrt(R * 1e3 * T * g_p**2 / ((g_p - tau * g_pt) ** 2 / (tau**2 * g_tt) - g_pp)))
    return out


def _gibbs_region2(T, pm):
    return _gibbs_ideal_residual(T, pm, 540.0, c.R2_J0, c.R2_n0, c.R2_I, c.R2_J, c.R2_n, 0.5)


def _gibbs_region5(T, pm):
    return _gibbs_ideal_residual(T, pm, 1000.0, c.R5_J0, c.R5_n0, c.R5_I, c.R5_J, c.R5_n, 0.0)


def _helmholtz_region3(T, rho):
    """Region 3 properties from density (kg/m3) and temperature (IF97 Eq. 28 and Table 31); p in Pa."""
    delta, tau = rho / RHO_CRIT, T_CRIT / T
    ii, jj, n, n1 = c.R3_I, c.R3_J, c.R3_n, c.R3_n1
    f = n1 * np.log(delta) + np.sum(n * delta**ii * tau**jj)
    f_d = n1 / delta + np.sum(n * ii * delta ** (ii - 1) * tau**jj)
    f_dd = -n1 / delta**2 + np.sum(n * ii * (ii - 1) * delta ** (ii - 2) * tau**jj)
    f_t = np.sum(n * delta**ii * jj * tau ** (jj - 1))
    f_tt = np.sum(n * delta**ii * jj * (jj - 1) * tau ** (jj - 2))
    f_dt = np.sum(n * ii * delta ** (ii - 1) * jj * tau ** (jj - 1))
    R = R_KJ
    p = delta * f_d * rho * R * T * 1e3
    cp = R * (-(tau**2) * f_tt + (delta * f_d - delta * tau * f_dt) ** 2 / (2 * delta * f_d + delta**2 * f_dd)) * 1e3
    w2 = R * 1e3 * T * (2 * delta * f_d + delta**2 * f_dd - (delta * f_d - delta * tau * f_dt) ** 2 / (tau**2 * f_tt))
    return {"p": float(p), "v": 1 / rho, "u": float(tau * f_t * R * T * 1e3),
            "h": float((tau * f_t + delta * f_d) * R * T * 1e3), "s": float((tau * f_t - f) * R * 1e3), "cp": float(cp),
            "cv": float(-(tau**2) * f_tt * R * 1e3), "w": float(np.sqrt(w2))}


def _region3_pressure(T, rho) -> float:
    """Pressure (Pa) in region 3 from T (K) and density (kg/m3): p = delta phi_delta rho R T."""
    delta, tau = rho / RHO_CRIT, T_CRIT / T
    f_d = c.R3_n1 / delta + np.sum(c.R3_n * c.R3_I * delta ** (c.R3_I - 1) * tau**c.R3_J)
    return float(delta * f_d * rho * R_KJ * T * 1e3)


def _region3_density(T, p, liquid_like: bool | None = None) -> float:
    """Density in region 3 for given T (K) and p (Pa), by bracketed root finding on p(rho, T) = p. Below the
    critical temperature the liquid-like and vapour-like branches are bracketed separately."""
    f = lambda rho: _region3_pressure(T, rho) - p  # noqa: E731
    if T >= T_CRIT:                                           # supercritical: p(rho) is monotonic
        return float(optimize.brentq(f, 0.5, 900.0, xtol=1e-12, rtol=1e-14))
    if liquid_like is None:
        liquid_like = p >= _psat_mpa(T) * 1e6
    # below Tc the equation has a van der Waals loop: bracket each branch by its spinodal (extremum of p)
    if liquid_like:
        rho_min = optimize.minimize_scalar(lambda r: _region3_pressure(T, r), bounds=(RHO_CRIT, 900.0),
                                           method="bounded").x
        lo, hi = rho_min, 900.0
    else:
        rho_max = optimize.minimize_scalar(lambda r: -_region3_pressure(T, r), bounds=(0.5, RHO_CRIT),
                                           method="bounded").x
        lo, hi = 0.5, rho_max
    return float(optimize.brentq(f, lo, hi, xtol=1e-12, rtol=1e-14))


# ------------------------------------------------------------------ public API
def properties(T: float, p: float) -> dict:
    """Properties of single-phase water or steam at T (K) and p (Pa): region, v, rho, u, h, s, cp, cv, w, phase."""
    reg = region(T, p)
    pm = p / 1e6
    if reg == 1:
        out = _gibbs_region1(T, pm)
    elif reg == 2:
        out = _gibbs_region2(T, pm)
    elif reg == 5:
        out = _gibbs_region5(T, pm)
    else:
        rho = _region3_density(T, p)
        out = _helmholtz_region3(T, rho)
        out.pop("p")
    out.update({"T": T, "p": p, "region": reg, "rho": 1 / out["v"],
                "phase": "liquid" if reg == 1 else "vapour" if reg in (2, 5) else "supercritical"})
    return out


def saturated(T: float | None = None, p: float | None = None) -> dict:
    """Saturated liquid ('liquid') and vapour ('vapour') properties at T (K) or p (Pa), plus T, p and the
    latent heat h_fg and s_fg."""
    if (T is None) == (p is None):
        raise ValueError("Give either T or p.")
    T = Tsat(p) if T is None else T
    p = psat(T) if p is None else p
    pm = p / 1e6
    if T <= 623.15:
        liq, vap = _gibbs_region1(T, pm), _gibbs_region2(T, pm)
    else:                                                    # near-critical: both phases from region 3
        liq = _helmholtz_region3(T, _region3_density(T, p, liquid_like=True))
        vap = _helmholtz_region3(T, _region3_density(T, p, liquid_like=False))
        liq.pop("p"), vap.pop("p")
    return {"T": T, "p": p, "liquid": liq, "vapour": vap, "h_fg": vap["h"] - liq["h"], "s_fg": vap["s"] - liq["s"]}


def two_phase(x: float, T: float | None = None, p: float | None = None) -> dict:
    """Properties of a liquid-vapour mixture of quality x (0..1) at T or p: v, u, h, s by lever rule."""
    if not 0 <= x <= 1:
        raise ValueError("Quality x must lie between 0 and 1.")
    sat = saturated(T, p)
    out = {k: (1 - x) * sat["liquid"][k] + x * sat["vapour"][k] for k in ("v", "u", "h", "s")}
    out.update({"T": sat["T"], "p": sat["p"], "x": x, "rho": 1 / out["v"], "phase": "two-phase", "region": 4})
    return out


def _state_from(p: float, key: str, value: float) -> dict:
    """State at pressure p (Pa) with given specific enthalpy ('h') or entropy ('s')."""
    if p < P_CRIT:
        sat = saturated(p=p)
        lo, hi = sat["liquid"][key], sat["vapour"][key]
        if lo <= value <= hi:
            return two_phase((value - lo) / (hi - lo), p=p)
        if value < lo:
            f = lambda T: properties(T, p)[key] - value  # noqa: E731
            T = optimize.brentq(f, 273.15, sat["T"], xtol=1e-9)
        else:
            f = lambda T: properties(T, p)[key] - value  # noqa: E731
            T_max = 1073.15 if p > 50e6 else 2273.15
            T = optimize.brentq(f, sat["T"], T_max, xtol=1e-9)
    else:
        f = lambda T: properties(T, p)[key] - value  # noqa: E731
        T = optimize.brentq(f, 273.15, 1073.15 if p > 50e6 else 2273.15, xtol=1e-9)
    return properties(T, p)


def state_ph(p: float, h: float) -> dict:
    """State from pressure (Pa) and specific enthalpy (J/kg) - liquid, two-phase (with quality x) or vapour."""
    return _state_from(p, "h", h)


def state_ps(p: float, s: float) -> dict:
    """State from pressure (Pa) and specific entropy (J/(kg K)) - e.g. the outlet of an isentropic turbine or pump."""
    return _state_from(p, "s", s)


def state(T: float | None = None, p: float | None = None, h: float | None = None, s: float | None = None,
          x: float | None = None) -> dict:
    """Any state from two of (T, p, h, s, x): (T, p), (p, h), (p, s), (p, x) or (T, x)."""
    if x is not None:
        return two_phase(x, T=T, p=p)
    if T is not None and p is not None:
        return properties(T, p)
    if p is not None and h is not None:
        return state_ph(p, h)
    if p is not None and s is not None:
        return state_ps(p, s)
    raise ValueError("Give (T, p), (p, h), (p, s), (p, x) or (T, x).")
