"""Pure-component data: molar mass, critical constants, acentric factor, ideal-gas heat-capacity
coefficients (Cp/R = A + B T + C T^2 + D T^-2) and standard enthalpy and Gibbs energy of formation.

Sources (see tools/make_components.py): for the 27 fluids available in CoolProp, molar mass and critical
constants come from CoolProp's reference equations of state the heat-capacity polynomial is fitted to
CoolProp's ideal-gas Cp over the stated range (within 1 %), and Antoine vapour-pressure constants are fitted to
CoolProp's saturation pressures between 1 kPa and 15 bar; SO3, NO, NO2 and HCl use the constants of Smith,
Van Ness & Abbott (Appendix B, Table C.1) and are marked ``verified=False``. Formation properties (ideal gas,
298.15 K, 1 bar) are the JANAF/TRC values tabulated by Smith, Van Ness & Abbott (Table C.4).

>>> from engthermo import components
>>> co2 = components.get("CO2")
>>> co2.Tc, co2.omega, co2.Hf
(304.1282, 0.22394, -393509.0)
"""

from __future__ import annotations

from dataclasses import dataclass

from ._component_data import DATA


@dataclass(frozen=True)
class Component:
    name: str
    formula: str
    M: float                    # kg/mol
    Tc: float                   # K
    Pc: float                   # Pa
    omega: float
    cp: tuple                   # (A, B, C, D): Cp/R = A + B T + C T^2 + D T^-2
    cp_range: tuple             # K
    Hf: float | None            # J/mol
    Gf: float | None            # J/mol
    aliases: tuple = ()
    coolprop: str | None = None
    verified: bool = True
    cp_fit_error: float | None = None
    antoine: tuple | None = None        # (A, B, C): ln(p/Pa) = A - B/(T + C), T in K
    antoine_range: tuple | None = None  # K
    antoine_fit_error: float | None = None


COMPONENTS = {d["name"]: Component(**d) for d in DATA}
_LOOKUP = {}
for comp in COMPONENTS.values():
    for key in (comp.name, comp.formula, *comp.aliases):
        _LOOKUP[key.lower()] = comp


def get(name: str) -> Component:
    """Look up a component by name, formula or alias (case-insensitive)."""
    try:
        return _LOOKUP[name.lower()]
    except KeyError:
        raise KeyError(f"Unknown component {name!r}. Available: {', '.join(available())}") from None


def available() -> list[str]:
    """Names of all components."""
    return sorted(COMPONENTS)
