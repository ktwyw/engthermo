"""UNIFAC: activity coefficients predicted from molecular groups (original UNIFAC, Fredenslund, Jones &
Prausnitz 1975, with the Hansen et al. 1991 parameter tables - see tools/transcribe_unifac.py).

Each molecule is described by its subgroups, e.g. ethanol = {"CH3": 1, "CH2": 1, "OH": 1}. The activity
coefficient is ln gamma = ln gamma_C (combinatorial, from sizes and shapes) + ln gamma_R (residual, from
group interactions).

>>> from engthermo import unifac
>>> g = unifac.gammas([{"CH3": 1, "CH2": 1, "OH": 1}, {"H2O": 1}], [0.4, 0.6], T=351.0)      # ethanol-water
>>> [round(float(v), 4) for v in g]
[1.3896, 1.3464]
"""

from __future__ import annotations

import numpy as np

from ._unifac_data import INTERACTIONS, SUBGROUPS

_BY_NAME = {v[0]: k for k, v in SUBGROUPS.items()}

# ready-made group assignments for the course's components
MOLECULES = {
    "water": {"H2O": 1}, "methanol": {"CH3OH": 1}, "ethanol": {"CH3": 1, "CH2": 1, "OH": 1},
    "acetone": {"CH3": 1, "CH3CO": 1}, "benzene": {"ACH": 6}, "toluene": {"ACH": 5, "ACCH3": 1},
    "n-hexane": {"CH3": 2, "CH2": 4}, "n-heptane": {"CH3": 2, "CH2": 5}, "n-octane": {"CH3": 2, "CH2": 6},
    "n-pentane": {"CH3": 2, "CH2": 3}, "n-butane": {"CH3": 2, "CH2": 2}, "propane": {"CH3": 2, "CH2": 1},
    "chloroform": {"CHCL3": 1}, "cyclohexane": {"CH2": 6}, "ethyl acetate": {"CH3": 1, "CH2": 1, "CH3COO": 1},
    "acetic acid": {"CH3": 1, "COOH": 1}, "1-propanol": {"CH3": 1, "CH2": 2, "OH": 1}, "1-butanol": {"CH3": 1, "CH2": 3, "OH": 1},
    "2-propanol": {"CH3": 2, "CH": 1, "OH": 1}, "diethyl ether": {"CH3": 2, "CH2": 1, "CH2O": 1},
    "methyl acetate": {"CH3": 1, "CH3COO": 1},
}


def _subgroup_id(name_or_id):
    if isinstance(name_or_id, int):
        return name_or_id
    try:
        return _BY_NAME[name_or_id]
    except KeyError:
        raise KeyError(f"Unknown UNIFAC subgroup {name_or_id!r}. Known: {', '.join(sorted(_BY_NAME))}") from None


def groups(molecule) -> dict:
    """Subgroup counts {subgroup id: count} for a dict of subgroup names/ids or a known molecule name."""
    if isinstance(molecule, str):
        if molecule not in MOLECULES:
            raise KeyError(f"No stored group assignment for {molecule!r}; give the groups as a dict. "
                           f"Known: {', '.join(MOLECULES)}")
        molecule = MOLECULES[molecule]
    return {_subgroup_id(k): v for k, v in molecule.items()}


def gammas(molecules, x, T: float) -> np.ndarray:
    """Activity coefficients of every component of a mixture at T (K) and mole fractions x."""
    mols = [groups(m) for m in molecules]
    x = np.maximum(np.asarray(x, float), 1e-12)          # a component absent from the mixture: its infinite-dilution limit
    x = x / x.sum()
    n = len(mols)
    sub_ids = sorted({sid for m in mols for sid in m})
    nu = np.array([[m.get(sid, 0) for sid in sub_ids] for m in mols], float)     # nu[i, k]: count of subgroup k in molecule i
    Rk = np.array([SUBGROUPS[sid][3] for sid in sub_ids])
    Qk = np.array([SUBGROUPS[sid][4] for sid in sub_ids])
    main = [SUBGROUPS[sid][1] for sid in sub_ids]
    # combinatorial part (Staverman-Guggenheim), z = 10
    r, q = nu @ Rk, nu @ Qk
    phi, theta = r * x / (r @ x), q * x / (q @ x)
    lseg = 5 * (r - q) - (r - 1)
    ln_gC = np.log(phi / x) + 5 * q * np.log(theta / phi) + lseg - phi / x * (x @ lseg)
    # residual part: group activity coefficients in the mixture and in each pure component
    psi = np.array([[np.exp(-INTERACTIONS[main[m]].get(main[k], 0.0) / T) if main[m] != main[k] else 1.0
                     for k in range(len(sub_ids))] for m in range(len(sub_ids))])

    def ln_Gamma(X):                                    # X: group mole fractions
        Theta = Qk * X / (Qk @ X)
        S = psi.T @ Theta                                # S[k] = sum_m Theta_m psi_mk
        return Qk * (1 - np.log(S) - (psi @ (Theta / S)))

    X_mix = (x @ nu) / (x @ nu).sum()
    lnG_mix = ln_Gamma(X_mix)
    ln_gR = np.empty(n)
    for i in range(n):
        Xi = nu[i] / nu[i].sum()
        ln_gR[i] = nu[i] @ (lnG_mix - ln_Gamma(Xi))
    return np.exp(ln_gC + ln_gR)


def gamma_function(molecules):
    """A gamma(x, T) function for the given molecules, in the form the ``vle`` module expects."""
    return lambda x, T: gammas(molecules, x, T)
