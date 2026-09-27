"""Generate src/engthermo/_component_data.py.

For every fluid available in CoolProp (Bell et al., 2014; MIT licence), molar mass, critical constants and the
acentric factor are taken from CoolProp's reference equations of state (7 significant figures),
and the ideal-gas heat-capacity polynomial Cp/R = A + B T + C T^2 + D T^-2 is fitted (least squares) to
CoolProp's ideal-gas Cp over the fluid's valid temperature range; the largest fit error is recorded.
Standard enthalpies and Gibbs energies of formation (ideal gas, 298.15 K, 1 bar) are from the JANAF/TRC
values tabulated in Smith, Van Ness & Abbott (Table C.4). Species that CoolProp lacks keep the constants and
Cp coefficients of Smith, Van Ness & Abbott (Appendix B, Table C.1) and are marked verified=False.

    python tools/make_components.py
"""
from pathlib import Path

import CoolProp.CoolProp as CP
import numpy as np

R = 8.314462618
# name, formula, CoolProp name, aliases, Hf (J/mol), Gf (J/mol)
SPECIES = [
    ("nitrogen", "N2", "Nitrogen", ("N2",), 0.0, 0.0), ("oxygen", "O2", "Oxygen", ("O2",), 0.0, 0.0),
    ("hydrogen", "H2", "Hydrogen", ("H2",), 0.0, 0.0), ("argon", "Ar", "Argon", ("Ar",), 0.0, 0.0),
    ("air", "air", "Air", (), None, None),
    ("carbon dioxide", "CO2", "CarbonDioxide", ("CO2",), -393509.0, -394359.0),
    ("carbon monoxide", "CO", "CarbonMonoxide", ("CO",), -110525.0, -137169.0),
    ("water", "H2O", "Water", ("H2O", "steam"), -241818.0, -228572.0),
    ("methane", "CH4", "Methane", ("CH4",), -74520.0, -50460.0), ("ethane", "C2H6", "Ethane", ("C2H6",), -83820.0, -31855.0),
    ("propane", "C3H8", "n-Propane", ("C3H8",), -104680.0, -24290.0),
    ("n-butane", "C4H10", "n-Butane", ("butane", "nC4"), -125790.0, -16570.0),
    ("n-pentane", "C5H12", "n-Pentane", ("pentane", "nC5"), -146760.0, -8650.0),
    ("n-hexane", "C6H14", "n-Hexane", ("hexane", "nC6"), -166920.0, 150.0),
    ("n-heptane", "C7H16", "n-Heptane", ("heptane", "nC7"), -187780.0, 8260.0),
    ("n-octane", "C8H18", "n-Octane", ("octane", "nC8"), -208750.0, 16260.0),
    ("ethylene", "C2H4", "Ethylene", ("ethene",), 52510.0, 68460.0), ("propylene", "C3H6", "Propylene", ("propene",), 19710.0, 62205.0),
    ("benzene", "C6H6", "Benzene", (), 82930.0, 129665.0), ("toluene", "C7H8", "Toluene", (), 50170.0, 122050.0),
    ("methanol", "CH4O", "Methanol", ("CH3OH",), -200660.0, -161960.0), ("ethanol", "C2H6O", "Ethanol", ("C2H5OH",), -235100.0, -168490.0),
    ("acetone", "C3H6O", "Acetone", (), -215700.0, -152716.0), ("ammonia", "NH3", "Ammonia", ("NH3",), -46110.0, -16450.0),
    ("sulfur dioxide", "SO2", "SulfurDioxide", ("SO2",), -296830.0, -300194.0),
    ("hydrogen sulfide", "H2S", "HydrogenSulfide", ("H2S",), -20630.0, -33560.0),
    ("R134a", "C2H2F4", "R134a", ("1,1,1,2-tetrafluoroethane",), None, None),
]
# not in CoolProp: Smith, Van Ness & Abbott values (Tc K, Pc Pa, omega, M, cp A B C D, range)
# graphite (solid carbon, Table C.2) has no critical point: Tc, Pc, omega are NaN
TEXTBOOK = [
    ("carbon (graphite)", "C", ("graphite", "carbon"), 0.012011, float("nan"), float("nan"), float("nan"), (1.771, 0.771e-3, 0.0, -0.867e5), (298, 2000), 0.0, 0.0),
    ("sulfur trioxide", "SO3", ("SO3",), 0.080064, 490.9, 82.10e5, 0.424, (8.060, 1.056e-3, 0.0, -2.028e5), (298, 2000), -395720.0, -371060.0),
    ("nitric oxide", "NO", ("NO",), 0.030006, 180.2, 64.80e5, 0.583, (3.387, 0.629e-3, 0.0, 0.014e5), (298, 2000), 90250.0, 86550.0),
    ("nitrogen dioxide", "NO2", ("NO2",), 0.046006, 431.0, 101.00e5, 0.851, (4.982, 1.195e-3, 0.0, -0.792e5), (298, 2000), 33180.0, 51310.0),
    ("hydrogen chloride", "HCl", ("HCl",), 0.036461, 324.7, 83.10e5, 0.132, (3.156, 0.623e-3, 0.0, 0.151e5), (298, 2000), -92307.0, -95299.0),
]


def sig(x, n=5):
    return float(f"{x:.{n}g}")


rows = []
for name, formula, cpname, aliases, Hf, Gf in SPECIES:
    M, Tc, Pc, w = (CP.PropsSI(k, cpname) for k in ("molarmass", "Tcrit", "Pcrit", "acentric"))
    Tmax = min(1500.0, CP.PropsSI("Tmax", cpname))
    T = np.linspace(298.15, Tmax, 200)
    cp0 = np.array([CP.PropsSI("Cp0molar", "T", t, "P", 1e5, cpname) for t in T]) / R
    X = np.column_stack([np.ones_like(T), T, T**2, T**-2])
    coef, *_ = np.linalg.lstsq(X, cp0, rcond=None)
    err = float(np.max(np.abs(X @ coef / cp0 - 1)))
    if name == "argon":                     # monatomic: Cp = 5R/2 exactly
        coef, err = np.array([2.5, 0.0, 0.0, 0.0]), 0.0
    # Antoine constants ln(p/Pa) = A - B/(T + C) fitted to CoolProp's saturation pressure between 1 kPa and
    # 15 bar (within the triple point and 0.9 Tc); the range and the largest relative error are recorded
    antoine, ant_range, ant_err = None, None, None
    if name != "air":
        T_min = max(CP.PropsSI("Ttriple", cpname), CP.PropsSI("Tmin", cpname)) + 0.5
        T_lo = max(T_min, CP.PropsSI("T", "P", 1e3, "Q", 0, cpname)) if CP.PropsSI("P", "T", T_min, "Q", 0, cpname) < 1e3 else T_min
        T_hi = min(0.9 * Tc, CP.PropsSI("T", "P", 15e5, "Q", 0, cpname)) if CP.PropsSI("P", "T", 0.9 * Tc, "Q", 0, cpname) > 15e5 else 0.9 * Tc
        Ts = np.linspace(T_lo, T_hi, 80)
        ps = np.array([CP.PropsSI("P", "T", t, "Q", 0, cpname) for t in Ts])
        from scipy import optimize as _opt

        def resid(x, Ts=Ts, ps=ps):
            return x[0] - x[1] / (Ts + x[2]) - np.log(ps)
        sol = _opt.least_squares(resid, [21.0, 6.0 * Tc, -0.07 * Tc])       # starting guess scaled with Tc
        antoine = tuple(float(f"{v:.7g}") for v in sol.x)
        ant_range, ant_err = (round(float(T_lo), 2), round(float(T_hi), 2)), round(float(np.max(np.abs(np.exp(resid(sol.x)) - 1))), 4)
    rows.append(dict(name=name, formula=formula, aliases=aliases, M=sig(M, 7), Tc=sig(Tc, 7), Pc=sig(Pc, 7), omega=sig(w, 6),
                     cp=tuple(float(f"{v:.6g}") for v in coef), cp_range=(298.15, round(Tmax)), cp_fit_error=round(err, 4),
                     Hf=Hf, Gf=Gf, coolprop=cpname, verified=True, antoine=antoine, antoine_range=ant_range,
                     antoine_fit_error=ant_err))
for name, formula, aliases, M, Tc, Pc, w, cp, rng, Hf, Gf in TEXTBOOK:
    rows.append(dict(name=name, formula=formula, aliases=aliases, M=M, Tc=Tc, Pc=Pc, omega=w, cp=cp, cp_range=rng,
                     cp_fit_error=None, Hf=Hf, Gf=Gf, coolprop=None, verified=False, antoine=None, antoine_range=None,
                     antoine_fit_error=None))
out = ["# Generated by tools/make_components.py - do not edit by hand. See that script for sources.", "",
       'nan = float("nan")          # solids without a critical point', "", "DATA = ["] + [f"    {r!r}," for r in rows] + ["]", ""]
Path(__file__).resolve().parents[1].joinpath("src", "engthermo", "_component_data.py").write_text("\n".join(out))
print(f"wrote {len(rows)} components; largest Cp fit error {max(r['cp_fit_error'] for r in rows if r['cp_fit_error']):.2%}; "
      f"largest Antoine fit error {max(r['antoine_fit_error'] for r in rows if r['antoine_fit_error']):.2%}")
for r in rows:
    if r["cp_fit_error"] and r["cp_fit_error"] > 0.01:
        print(f"  {r['name']}: Cp fit error {r['cp_fit_error']:.2%} over {r['cp_range']}")
