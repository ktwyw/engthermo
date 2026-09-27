"""Mixture properties: partial molar properties from a property-composition relation, excess properties,
and the Gibbs-Duhem check. Compositions are mole fractions; properties are per mole of mixture.

>>> import numpy as np
>>> from engthermo import mixtures
>>> V = lambda x1: 40.0 * x1 + 60.0 * (1 - x1) - 5.0 * x1 * (1 - x1)      # a made-up binary volume (cm3/mol)
>>> Vbar1, Vbar2 = mixtures.partial_molar_binary(V, 0.3)
>>> round(0.3 * Vbar1 + 0.7 * Vbar2, 10) == round(V(0.3), 10)             # partial molar properties reconstruct M
True
"""

from __future__ import annotations

import numpy as np


def partial_molar_binary(M, x1: float, h: float = 1e-6) -> tuple[float, float]:
    """Partial molar properties of both components of a binary from M(x1) by the tangent-intercept rule:
    M1_bar = M + (1 - x1) dM/dx1, M2_bar = M - x1 dM/dx1 (derivative by central differences)."""
    dM = (M(x1 + h) - M(x1 - h)) / (2 * h)
    m = M(x1)
    return float(m + (1 - x1) * dM), float(m - x1 * dM)


def partial_molar(M_total, n, i: int, h: float = 1e-6) -> float:
    """Partial molar property of component i from the total property nM(n1, n2, ...) (a function of mole
    numbers): the derivative with respect to n_i at constant other n_j."""
    n = np.asarray(n, float)
    up, dn = n.copy(), n.copy()
    up[i] += h
    dn[i] -= h
    return float((M_total(up) - M_total(dn)) / (2 * h))


def excess(M, M_pure, x):
    """Excess property M^E = M(x) - sum x_i M_i for an ideal-solution reference (linear mixing)."""
    x = np.asarray(x, float)
    return M(x) - float(np.dot(x, M_pure))


def gibbs_duhem_residual(lngamma1, lngamma2, x1, h: float = 1e-6) -> float:
    """x1 d ln(gamma1)/dx1 + x2 d ln(gamma2)/dx1 - zero for thermodynamically consistent activity coefficients."""
    d1 = (lngamma1(x1 + h) - lngamma1(x1 - h)) / (2 * h)
    d2 = (lngamma2(x1 + h) - lngamma2(x1 - h)) / (2 * h)
    return float(x1 * d1 + (1 - x1) * d2)
