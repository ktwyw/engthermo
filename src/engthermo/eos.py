"""Equations of state for real gases and liquids: the second virial coefficient (Pitzer-Abbott correlation)
and the cubic equations van der Waals, Redlich-Kwong, Soave-Redlich-Kwong and Peng-Robinson, including
volume roots, compressibility factor, departure (residual) functions, fugacity coefficients, and van der
Waals mixing rules with binary interaction parameters for mixtures.

All cubics are written in the generic form (Smith, Van Ness & Abbott, chapter 3)
    p = RT/(V - b) - a(T) / ((V + eps b)(V + sigma b)),
with a(T) = Psi alpha(Tr) R^2 Tc^2 / Pc and b = Omega R Tc / Pc; departure functions follow chapter 6.
Units: K, Pa, m3/mol, J/mol.

>>> from engthermo import eos
>>> r = eos.solve("pr", "CO2", T=350.0, p=50e5)
>>> round(r["Z"], 4), round(r["lnphi"], 4)
(0.8292, -0.1677)
"""

from __future__ import annotations

import numpy as np

from .components import Component, get
from .constants import R

SQ2 = np.sqrt(2.0)
_CBRT2 = 2 ** (1 / 3)
OMEGA_RK, PSI_RK = (_CBRT2 - 1) / 3, 1 / (9 * (_CBRT2 - 1))      # 0.08664035, 0.42748023 (exact values)
OMEGA_PR, PSI_PR = 0.0777960739, 0.4572355289                    # exact Peng-Robinson values
# name: (eps, sigma, Omega, Psi) - textbook tables round these to 0.08664/0.42748 and 0.07780/0.45724
CUBICS = {
    "vdw": (0.0, 0.0, 1 / 8, 27 / 64),
    "rk": (0.0, 1.0, OMEGA_RK, PSI_RK),
    "srk": (0.0, 1.0, OMEGA_RK, PSI_RK),
    "pr": (1 - SQ2, 1 + SQ2, OMEGA_PR, PSI_PR),
}


def _comp(c) -> Component:
    return c if isinstance(c, Component) else get(c)


def alpha(name: str, Tr: float, omega: float) -> tuple[float, float]:
    """alpha(Tr) and d ln(alpha)/d ln(T) for the cubic equation ``name``."""
    if name == "vdw":
        return 1.0, 0.0
    if name == "rk":
        return Tr**-0.5, -0.5
    m = 0.480 + 1.574 * omega - 0.176 * omega**2 if name == "srk" else 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    a = (1 + m * (1 - np.sqrt(Tr))) ** 2
    return float(a), float(-m * np.sqrt(Tr) / np.sqrt(a))


def parameters(name: str, comp, T: float) -> dict:
    """a(T) (Pa m6/mol2), b (m3/mol) and d ln(alpha)/d ln(T) of a pure component."""
    if name not in CUBICS:
        raise KeyError(f"Unknown equation {name!r}; choose from {', '.join(CUBICS)}.")
    c = _comp(comp)
    eps, sig, Omega, Psi = CUBICS[name]
    al, dln = alpha(name, T / c.Tc, c.omega)
    return {"a": Psi * al * R**2 * c.Tc**2 / c.Pc, "b": Omega * R * c.Tc / c.Pc, "dlnalpha_dlnT": dln,
            "eps": eps, "sigma": sig}


def _z_roots(A: float, B: float, eps: float, sig: float) -> np.ndarray:
    """Real roots of the cubic in Z for A = a p/(RT)^2, B = b p/(RT)."""
    # (Z - B)(Z + eps B)(Z + sig B) ... expanded: Z^3 + c2 Z^2 + c1 Z + c0 = 0
    u, w = eps + sig, eps * sig
    c2 = -(1 + B - u * B)
    c1 = A + w * B**2 - u * B - u * B**2
    c0 = -(A * B + w * B**2 + w * B**3)
    roots = np.roots([1.0, c2, c1, c0])
    real = np.sort(roots[np.abs(roots.imag) < 1e-9].real)
    return real[real > B]


def _from_Z(Z: float, A: float, B: float, T: float, p: float, q: float, dln: float, eps: float, sig: float) -> dict:
    if sig != eps:
        integral = np.log((Z + sig * B) / (Z + eps * B)) / (sig - eps)
    else:
        integral = B / (Z + eps * B)
    lnphi = Z - 1 - np.log(Z - B) - q * integral
    H_R = R * T * (Z - 1 + (dln - 1) * q * integral)
    S_R = R * (np.log(Z - B) + dln * q * integral)
    return {"Z": float(Z), "V": float(Z * R * T / p), "lnphi": float(lnphi), "phi": float(np.exp(lnphi)),
            "H_R": float(H_R), "S_R": float(S_R), "G_R": float(R * T * lnphi)}


def solve(name: str, comp, T: float, p: float, phase: str = "vapor") -> dict:
    """Pure component at T and p: compressibility Z, molar volume V, fugacity coefficient (lnphi, phi) and the
    departure functions H_R, S_R, G_R (J/mol, J/(mol K)). ``phase`` chooses the largest ('vapor') or smallest
    ('liquid') real root when the cubic has three; 'stable' picks the root with the lower fugacity."""
    prm = parameters(name, comp, T)
    A, B = prm["a"] * p / (R * T) ** 2, prm["b"] * p / (R * T)
    q = prm["a"] / (prm["b"] * R * T)
    roots = _z_roots(A, B, prm["eps"], prm["sigma"])
    if roots.size == 0:
        raise ValueError("No physical root: state outside the range of the equation.")
    if phase == "stable" and roots.size > 1:
        cands = [_from_Z(z, A, B, T, p, q, prm["dlnalpha_dlnT"], prm["eps"], prm["sigma"])
                 for z in (roots[0], roots[-1])]
        out = min(cands, key=lambda d: d["lnphi"])
    else:
        Z = roots[0] if phase == "liquid" else roots[-1]
        out = _from_Z(Z, A, B, T, p, q, prm["dlnalpha_dlnT"], prm["eps"], prm["sigma"])
    out.update({"T": T, "p": p, "roots": roots, "phase": phase})
    return out


def pressure(name: str, comp, T: float, V: float) -> float:
    """p(T, V) of the cubic equation (Pa)."""
    prm = parameters(name, comp, T)
    a, b, eps, sig = prm["a"], prm["b"], prm["eps"], prm["sigma"]
    return float(R * T / (V - b) - a / ((V + eps * b) * (V + sig * b)))


# ------------------------------------------------------------------ virial (Pitzer-Abbott)
def virial_B(comp, T: float) -> dict:
    """Second virial coefficient (m3/mol) from the Pitzer correlation with Abbott's B0, B1 functions
    (Smith, Van Ness & Abbott, Eqs. 3.61-3.66), plus dB/dT and the departure functions at low pressure."""
    c = _comp(comp)
    Tr = T / c.Tc
    B0, B1 = 0.083 - 0.422 / Tr**1.6, 0.139 - 0.172 / Tr**4.2
    dB0, dB1 = 0.422 * 1.6 / Tr**2.6, 0.172 * 4.2 / Tr**5.2
    B = (B0 + c.omega * B1) * R * c.Tc / c.Pc
    dBdT = (dB0 + c.omega * dB1) * R / c.Pc
    return {"B": float(B), "dBdT": float(dBdT), "B0": float(B0), "B1": float(B1)}


def virial(comp, T: float, p: float) -> dict:
    """Z, V, fugacity coefficient and departure functions from the truncated virial equation Z = 1 + Bp/RT."""
    v = virial_B(comp, T)
    B, dBdT = v["B"], v["dBdT"]
    Z = 1 + B * p / (R * T)
    lnphi = B * p / (R * T)
    return {"Z": float(Z), "V": float(Z * R * T / p), "lnphi": float(lnphi), "phi": float(np.exp(lnphi)),
            "H_R": float(p * (B - T * dBdT)), "S_R": float(-p * dBdT), "B": B, "T": T, "p": p}


# ------------------------------------------------------------------ mixtures
def mixture(name: str, comps, x, T: float, p: float, kij=None, phase: str = "vapor") -> dict:
    """Mixture with van der Waals mixing rules: a = sum_i sum_j x_i x_j sqrt(a_i a_j)(1 - k_ij), b = sum_i x_i b_i.
    Returns Z, V, the mixture fugacity coefficient and departure functions, and the fugacity coefficients
    of each component in the mixture (``lnphi_i``)."""
    comps = [_comp(c) for c in comps]
    x = np.asarray(x, float)
    x = x / x.sum()
    n = len(comps)
    kij = np.zeros((n, n)) if kij is None else np.asarray(kij, float)
    prms = [parameters(name, c, T) for c in comps]
    a_i = np.array([q["a"] for q in prms])
    b_i = np.array([q["b"] for q in prms])
    a_ij = np.sqrt(np.outer(a_i, a_i)) * (1 - kij)
    a, b = float(x @ a_ij @ x), float(x @ b_i)
    eps, sig = prms[0]["eps"], prms[0]["sigma"]
    # temperature derivative of a: d(a_ij)/dT via d ln alpha / d ln T of each component
    dln_i = np.array([q["dlnalpha_dlnT"] for q in prms])
    da_ij_dlnT = 0.5 * a_ij * (dln_i[:, None] + dln_i[None, :])
    dln_mix = float(x @ da_ij_dlnT @ x) / a
    A, B = a * p / (R * T) ** 2, b * p / (R * T)
    q = a / (b * R * T)
    roots = _z_roots(A, B, eps, sig)
    if roots.size == 0:
        raise ValueError("No physical root.")
    Z = roots[0] if phase == "liquid" else roots[-1]
    out = _from_Z(Z, A, B, T, p, q, dln_mix, eps, sig)
    integral = np.log((Z + sig * B) / (Z + eps * B)) / (sig - eps) if sig != eps else B / (Z + eps * B)
    abar = 2 * a_ij @ x - a                       # partial-molar-type a: d(n^2 a)/dn_i / n - a
    qbar = q * (1 + abar / a - b_i / b)
    out["lnphi_i"] = b_i / b * (Z - 1) - np.log(Z - B) - qbar * integral
    out["phi_i"] = np.exp(out["lnphi_i"])
    out.update({"T": T, "p": p, "x": x, "a": a, "b": b, "roots": roots, "phase": phase})
    return out
