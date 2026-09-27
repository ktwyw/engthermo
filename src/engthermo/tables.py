"""Property tables for refrigerants (and any fluid given as saturation + superheated-vapour tables), with
the interpolation that textbook table look-ups do by hand. Units: K, Pa, m3/kg, J/kg, J/(kg K).

>>> from engthermo import tables
>>> r = tables.FluidTable("R134a")
>>> st = r.saturated(T=273.15)
>>> round(st["liquid"]["h"] / 1e3, 1), round(st["liquid"]["s"], 1)     # IIR reference state
(200.0, 1000.0)
"""

from __future__ import annotations

import numpy as np

from . import datasets


class FluidTable:
    """Saturation and superheated-vapour tables of a fluid ('R134a', 'ammonia' or 'propane' are bundled)."""

    def __init__(self, fluid: str):
        sat = np.loadtxt(datasets.path(f"{fluid}_saturation.csv"), delimiter=",", skiprows=3)
        self.fluid = fluid
        self.T_sat, self.p_sat = sat[:, 0], sat[:, 1]
        self._sat = {"v_f": sat[:, 2], "v_g": sat[:, 3], "h_f": sat[:, 4], "h_g": sat[:, 5], "s_f": sat[:, 6],
                     "s_g": sat[:, 7]}
        sup = np.loadtxt(datasets.path(f"{fluid}_superheated.csv"), delimiter=",", skiprows=3)
        self._isobars = {}
        for p in np.unique(sup[:, 0]):
            block = sup[sup[:, 0] == p]
            self._isobars[float(p)] = {"T": block[:, 1], "v": block[:, 2], "h": block[:, 3], "s": block[:, 4]}
        self._p_iso = np.array(sorted(self._isobars))

    # ---------------------------------------------------------------- saturation
    def psat(self, T: float) -> float:
        """Saturation pressure by interpolation of ln p against T."""
        self._check(T, self.T_sat, "T")
        return float(np.exp(np.interp(T, self.T_sat, np.log(self.p_sat))))

    def Tsat(self, p: float) -> float:
        self._check(p, self.p_sat, "p")
        return float(np.interp(np.log(p), np.log(self.p_sat), self.T_sat))

    def saturated(self, T: float | None = None, p: float | None = None) -> dict:
        """Saturated liquid and vapour properties at T or p (linear interpolation in T)."""
        if (T is None) == (p is None):
            raise ValueError("Give either T or p.")
        T = self.Tsat(p) if T is None else T
        p = self.psat(T) if p is None else p
        self._check(T, self.T_sat, "T")
        liq = {k: float(np.interp(T, self.T_sat, self._sat[k + "_f"])) for k in ("v", "h", "s")}
        vap = {k: float(np.interp(T, self.T_sat, self._sat[k + "_g"])) for k in ("v", "h", "s")}
        return {"T": T, "p": p, "liquid": liq, "vapour": vap, "h_fg": vap["h"] - liq["h"], "s_fg": vap["s"] - liq["s"]}

    def two_phase(self, x: float, T: float | None = None, p: float | None = None) -> dict:
        sat = self.saturated(T, p)
        out = {k: (1 - x) * sat["liquid"][k] + x * sat["vapour"][k] for k in ("v", "h", "s")}
        out.update({"T": sat["T"], "p": sat["p"], "x": x, "phase": "two-phase"})
        return out

    # ---------------------------------------------------------------- superheated vapour
    def _isobar_props(self, p: float, key: str, value: float) -> dict:
        """On the tabulated isobar p, the state with the given key ('T', 'h' or 's') = value."""
        iso = self._isobars[p]
        x = iso[key]
        if not x[0] - 1e-9 <= value <= x[-1] + 1e-9:
            raise ValueError(f"{key} = {value:g} is outside the superheated table at p = {p:.4g} Pa "
                             f"({x[0]:.6g}..{x[-1]:.6g}).")
        return {k: float(np.interp(value, x, iso[k])) for k in ("T", "v", "h", "s")}

    def superheated(self, p: float, T: float | None = None, h: float | None = None, s: float | None = None) -> dict:
        """Superheated vapour at pressure p and one of T, h or s: interpolation along the two neighbouring
        tabulated isobars, then linear interpolation in ln p between them."""
        key, value = next((k, v) for k, v in (("T", T), ("h", h), ("s", s)) if v is not None)
        self._check(p, self._p_iso, "p")
        i = int(np.searchsorted(self._p_iso, p))
        if i == 0 or np.isclose(p, self._p_iso[max(i - 1, 0)]):
            out = self._isobar_props(self._p_iso[max(i - 1, 0)] if i > 0 else self._p_iso[0], key, value)
        else:
            lo, hi = self._p_iso[i - 1], self._p_iso[i]
            a, b = self._isobar_props(lo, key, value), self._isobar_props(hi, key, value)
            f = (np.log(p) - np.log(lo)) / (np.log(hi) - np.log(lo))
            out = {k: (1 - f) * a[k] + f * b[k] for k in a}
        out.update({"p": p, "phase": "superheated vapour"})
        return out

    def state_ps(self, p: float, s: float) -> dict:
        """State at p with entropy s: two-phase (with quality) or superheated - e.g. an isentropic compressor outlet."""
        sat = self.saturated(p=p)
        if s <= sat["vapour"]["s"]:
            x = (s - sat["liquid"]["s"]) / sat["s_fg"]
            if x < 0:
                raise ValueError("State is a subcooled liquid: not covered by these tables.")
            return self.two_phase(x, p=p)
        return self.superheated(p, s=s)

    def state_ph(self, p: float, h: float) -> dict:
        """State at p with enthalpy h: two-phase (with quality) or superheated - e.g. after a throttling valve."""
        sat = self.saturated(p=p)
        if h <= sat["vapour"]["h"]:
            x = (h - sat["liquid"]["h"]) / sat["h_fg"]
            if x < 0:
                raise ValueError("State is a subcooled liquid: not covered by these tables.")
            return self.two_phase(x, p=p)
        return self.superheated(p, h=h)

    @staticmethod
    def _check(value, grid, name):
        if not grid[0] - 1e-9 <= value <= grid[-1] + 1e-9:
            raise ValueError(f"{name} = {value:g} is outside the table ({grid[0]:.6g}..{grid[-1]:.6g}).")
