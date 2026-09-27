"""Bundled data files (CSV) for the course, with sources.

>>> from engthermo import datasets
>>> "R134a_saturation.csv" in datasets.available()
True
"""

from __future__ import annotations

from importlib import resources

_CP = ("Generated from CoolProp (Bell et al., 2014; MIT licence) by tools/make_course_data.py, "
       "IIR reference state (h = 200 kJ/kg, s = 1 kJ/(kg K) for saturated liquid at 0 degC).")
INFO: dict[str, str] = {}
for _f, _n in (("R134a", "R134a"), ("ammonia", "ammonia (R717)"), ("propane", "propane (R290)")):
    INFO[f"{_f}_saturation.csv"] = f"Saturation table of {_n}: T, p, v_f, v_g, h_f, h_g, s_f, s_g. {_CP}"
    INFO[f"{_f}_superheated.csv"] = f"Superheated-vapour table of {_n}: 40 isobars, v, h, s vs T. {_CP}"
INFO["real_gas_reference_densities.csv"] = ("Reference densities and Z of CO2 (280-400 K, 1-200 bar) and hydrogen (200-350 K, 1-900 bar) "
                                             "from CoolProp's multiparameter reference equations (Span & Wagner 1996; Leachman et al. 2009).")
INFO["ethanol_water_vle_1atm.csv"] = ("Ethanol-water VLE at 1 atm (x, y, T): pseudo-experimental data from a Wilson model anchored "
                                       "to the experimental azeotrope and limiting activity coefficient, with realistic noise; independent of UNIFAC.")
INFO["methane_ethane_bubble_200K.csv"] = ("Bubble points of methane-ethane at 200 K from CoolProp's multi-fluid reference mixture model "
                                          "(HEOS backend), independent of the cubic equations.")
INFO["methanol_water_lab_vle.csv"] = ("Messy laboratory VLE export for methanol-water at 1 atm: UNIFAC bubble points with noise plus deliberate "
                                       "faults (mixed units, a duplicate, a decimal comma, a missing value, an inconsistent point).")
INFO["ethanol_water_volume.csv"] = ("Molar volume of ethanol-water mixtures at 25 degC: pure volumes from CoolProp, excess volume a "
                                    "representative Redlich-Kister fit (minimum about -1.1 cm3/mol near x_ethanol = 0.4).")


def available() -> list[str]:
    return sorted(INFO)


def path(name: str) -> str:
    """Filesystem path of a bundled data file."""
    if name not in INFO:
        raise KeyError(f"Unknown data file {name!r}. Available: {', '.join(available())}")
    return str(resources.files("engthermo.data").joinpath(name))


def info(name: str) -> str:
    if name not in INFO:
        raise KeyError(f"Unknown data file {name!r}.")
    return INFO[name]
