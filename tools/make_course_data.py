"""Write the course data files into src/engthermo/data.

Refrigerant property tables are generated from CoolProp (Bell et al., Ind. Eng. Chem. Res. 53, 2014; MIT
licence) with the IIR reference state (h = 200 kJ/kg, s = 1 kJ/(kg K) for saturated liquid at 0 degC), the
convention of refrigeration handbooks. Run from the repository root:  python tools/make_course_data.py
"""
from pathlib import Path

import CoolProp.CoolProp as CP
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "src" / "engthermo" / "data"


def write(name, header, columns, names, fmt):
    lines = [f"# {h}" for h in header] + [",".join(names)]
    for row in zip(*columns):
        lines.append(",".join(f % v for f, v in zip(fmt, row)))
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", name)


REFRIGERANTS = {"R134a": ("R134a", -40, 100, 2), "ammonia": ("Ammonia", -50, 120, 2), "propane": ("n-Propane", -50, 90, 2)}
for key, (fluid, T_lo, T_hi, dT) in REFRIGERANTS.items():
    CP.set_reference_state(fluid, "IIR")
    T = np.arange(T_lo, T_hi + dT / 2, dT) + 273.15
    cols = {"p": [], "vf": [], "vg": [], "hf": [], "hg": [], "sf": [], "sg": []}
    for t in T:
        cols["p"].append(CP.PropsSI("P", "T", t, "Q", 0, fluid))
        for q, suffix in ((0, "f"), (1, "g")):
            cols["v" + suffix].append(1 / CP.PropsSI("D", "T", t, "Q", q, fluid))
            cols["h" + suffix].append(CP.PropsSI("H", "T", t, "Q", q, fluid))
            cols["s" + suffix].append(CP.PropsSI("S", "T", t, "Q", q, fluid))
    write(f"{key}_saturation.csv",
          [f"{fluid} saturation properties from CoolProp (IIR reference state). Columns: T (K), p (Pa), v_f, v_g (m3/kg),",
           "h_f, h_g (J/kg), s_f, s_g (J/(kg K))."],
          [T, cols["p"], cols["vf"], cols["vg"], cols["hf"], cols["hg"], cols["sf"], cols["sg"]],
          ["T_K", "p_Pa", "v_f", "v_g", "h_f", "h_g", "s_f", "s_g"], ["%.2f", "%.6g", "%.6g", "%.6g", "%.7g", "%.7g", "%.7g", "%.7g"])
    # superheated vapour: isobars from the saturation pressure at T_lo to that at T_hi, each from T_sat to T_sat + 120 K
    p_iso = np.geomspace(CP.PropsSI("P", "T", T[0], "Q", 1, fluid), CP.PropsSI("P", "T", T[-1], "Q", 1, fluid), 40)
    rows = []
    for p in p_iso:
        T_s = CP.PropsSI("T", "P", p, "Q", 1, fluid)
        for t in np.linspace(T_s + 0.01, T_s + 120, 31):        # just above saturation: unambiguously vapour
            rows.append((p, t, 1 / CP.PropsSI("D", "T", t, "P", p, fluid), CP.PropsSI("H", "T", t, "P", p, fluid),
                         CP.PropsSI("S", "T", t, "P", p, fluid)))
    rows = np.array(rows)
    write(f"{key}_superheated.csv",
          [f"{fluid} superheated vapour from CoolProp (IIR reference state): 40 isobars, each from saturation to +120 K.",
           "Columns: p (Pa), T (K), v (m3/kg), h (J/kg), s (J/(kg K))."],
          list(rows.T), ["p_Pa", "T_K", "v", "h", "s"], ["%.6g", "%.4f", "%.6g", "%.7g", "%.7g"])

# ------------------------------------------------------------------ reference densities of real gases (CoolProp reference EOS)
rows = []
for fluid, cpname, Ts, ps in (("CO2", "CarbonDioxide", (280.0, 300.0, 320.0, 350.0, 400.0), np.array([1, 5, 10, 20, 40, 60, 80, 100, 150, 200]) * 1e5),
                              ("hydrogen", "Hydrogen", (200.0, 250.0, 300.0, 350.0), np.array([1, 10, 50, 100, 200, 350, 500, 700, 900]) * 1e5)):
    for T in Ts:
        for p in ps:
            rows.append((fluid, T, p, CP.PropsSI("D", "T", T, "P", p, cpname), CP.PropsSI("Z", "T", T, "P", p, cpname)))
lines = ["# Reference densities of CO2 and hydrogen from CoolProp's multiparameter reference equations of state",
         "# (Span & Wagner 1996 for CO2; Leachman et al. 2009 for hydrogen). Columns: fluid, T (K), p (Pa), density (kg/m3), Z."]
lines += ["fluid,T_K,p_Pa,rho_kg_m3,Z"] + [f"{f},{t:.1f},{p:.6g},{d:.6g},{z:.6f}" for f, t, p, d, z in rows]
(OUT / "real_gas_reference_densities.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote real_gas_reference_densities.csv")

# ------------------------------------------------------------------ ethanol-water volume of mixing at 25 degC (representative)
# Molar volumes of the pure liquids at 25 degC from CoolProp; the excess volume is a Redlich-Kister representation with
# the well-known minimum of about -1.1 cm3/mol near x_ethanol = 0.4 (see e.g. Grolier & Wilhelm, Fluid Phase Equilib. 6, 1981).
V_eth, V_wat = 1 / CP.PropsSI("Dmolar", "T", 298.15, "P", 1e5, "Ethanol") * 1e6, 1 / CP.PropsSI("Dmolar", "T", 298.15, "P", 1e5, "Water") * 1e6
x = np.round(np.linspace(0, 1, 21), 3)
VE = x * (1 - x) * (-4.35 + 1.0 * (2 * x - 1) - 0.3 * (2 * x - 1) ** 2)
V = x * V_eth + (1 - x) * V_wat + VE
write("ethanol_water_volume.csv",
      ["Molar volume of ethanol (1) + water (2) mixtures at 25 degC and 1 bar. Pure-liquid volumes from CoolProp; the excess volume is a",
       "representative Redlich-Kister fit (minimum about -1.1 cm3/mol near x1 = 0.4). Columns: x_ethanol, molar volume (cm3/mol)."],
      [x, V], ["x_ethanol", "V_cm3_mol"], ["%.3f", "%.4f"])

# ------------------------------------------------------------------ pseudo-experimental VLE of ethanol-water at 1 atm
# Generated from a Wilson model whose two parameters are fitted to experimental facts of the system: the azeotrope at
# x_ethanol = 0.894 and 78.15 degC (Gmehling & Onken data collection) and the limiting activity coefficient of ethanol in
# water near 80 degC (about 4.2). Raoult's-law vapour pressures from the fitted Antoine constants. Noise: 0.15 K in T and
# 0.004 in y, seed 2026. NOT generated from UNIFAC, so that notebook 10 can test UNIFAC against it.
import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scipy import optimize as _opt  # noqa: E402

from engthermo import activity, vle  # noqa: E402

rng_v = np.random.default_rng(2026)


def wilson_misfit(theta):
    g = activity.wilson([[1.0, theta[0]], [theta[1], 1.0]])
    az = activity.azeotrope(["ethanol", "water"], g, p=101325.0)
    ginf = activity.infinite_dilution(g, 353.15)[0]
    if az is None:
        return [10.0, 10.0, 10.0]
    return [az["x1"] - 0.894, (az["T"] - (78.15 + 273.15)) / 10.0, np.log(ginf / 4.2)]


theta = _opt.least_squares(wilson_misfit, [0.2, 0.8], bounds=([0.01, 0.01], [5, 5])).x
g_ew = activity.wilson([[1.0, theta[0]], [theta[1], 1.0]])
xs = np.array([0.0, 0.02, 0.05, 0.08, 0.12, 0.17, 0.23, 0.30, 0.38, 0.47, 0.56, 0.65, 0.73, 0.80, 0.86, 0.91, 0.95, 1.0])
rows = []
for x in xs:
    r = vle.bubble_T(["ethanol", "water"], [x, 1 - x], 101325.0, g_ew)
    y = float(r["y"][0])
    rows.append((x, min(max(y + (rng_v.normal(0, 0.004) if 0 < x < 1 else 0.0), 0), 1), r["T"] - 273.15 + rng_v.normal(0, 0.15)))
rows = np.array(rows)
write("ethanol_water_vle_1atm.csv",
      ["Ethanol (1) + water (2) vapour-liquid equilibrium at 101.325 kPa: pseudo-experimental data generated from a Wilson model",
       f"anchored to the experimental azeotrope (x1 = 0.894, 78.15 degC) and limiting activity coefficient (Lambda12 = {theta[0]:.4f},",
       f"Lambda21 = {theta[1]:.4f}), with 0.15 K and 0.004 noise. Columns: x1 (liquid), y1 (vapour), T (degC)."],
      list(rows.T), ["x_ethanol", "y_ethanol", "T_C"], ["%.3f", "%.4f", "%.2f"])
print(f"   Wilson parameters used: {theta}")

# ------------------------------------------------------------------ reference bubble points of methane-ethane at 200 K
# from CoolProp's multi-fluid Helmholtz reference model for the mixture (GERG-2008 type), independent of the cubic equations
rows = []
AS = CP.AbstractState("HEOS", "Methane&Ethane")
for x1 in np.linspace(0.0, 0.9, 10):
    AS.set_mole_fractions([x1, 1 - x1])
    try:
        AS.update(CP.QT_INPUTS, 0.0, 200.0)
        rows.append((x1, AS.p(), AS.mole_fractions_vapor()[0]))
    except ValueError:
        pass
write("methane_ethane_bubble_200K.csv",
      ["Bubble points of methane (1) + ethane (2) at 200 K from CoolProp's multi-fluid reference mixture model (HEOS backend,",
       "GERG-2008 type). Columns: x1 (liquid), bubble pressure (Pa), y1 (vapour)."],
      list(np.array(rows).T), ["x_methane", "p_bubble_Pa", "y_methane"], ["%.3f", "%.6g", "%.5f"])

# ------------------------------------------------------------------ a messy laboratory VLE data file (notebook 17)
# Methanol (1) + water (2) at 101.3 kPa, "measured" in a student laboratory: generated from UNIFAC bubble points with
# 0.2 K and 0.005 noise, then given the faults of a real lab export: pressure column in mixed units (kPa and mmHg),
# temperatures in degC and one in K, a duplicated point, a decimal typo, a missing vapour analysis and one
# thermodynamically inconsistent point (a mis-sampled vapour). Faults are listed in the file header for the instructor.
from engthermo import unifac  # noqa: E402

rng_l = np.random.default_rng(17)
g_mw = unifac.gamma_function(["methanol", "water"])
xs = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
rows = []
for i, x in enumerate(xs, 1):
    r = vle.bubble_T(["methanol", "water"], [x, 1 - x], 101325.0, g_mw)
    T_C = r["T"] - 273.15 + rng_l.normal(0, 0.2)
    y = float(r["y"][0]) + rng_l.normal(0, 0.005)
    rows.append([f"run-{i:02d}", f"{x:.3f}", f"{y:.3f}", f"{T_C:.1f}", "101.3", "kPa"])
rows[3][4], rows[3][5] = "760", "mmHg"                       # pressure reported in mmHg
rows[7][4], rows[7][5] = "760", "mmHg"
rows[5][3] = f"{float(rows[5][3]) + 273.15:.1f}"                # temperature in kelvin
rows.insert(9, list(rows[8]))                                  # duplicated point
rows[11][2] = rows[11][2].replace("0.", "0,")                  # decimal comma typo
rows[2][2] = ""                                                # missing vapour analysis
rows[6][2] = f"{float(rows[6][2]) - 0.12:.3f}"                 # inconsistent point (mis-sampled vapour)
lines = ["# Methanol-water VLE at atmospheric pressure, teaching laboratory, spring term. Ebulliometer with recirculation;",
         "# compositions by refractive index. Generated for the course from UNIFAC bubble points (0.2 K, 0.005 noise) with",
         "# deliberate faults: mixed pressure units, one temperature in K, a duplicate, a decimal comma, a missing y,",
         "# and one inconsistent point (run-07: vapour sample mis-taken).",
         "run,x_methanol,y_methanol,T,p,p_unit"] + [",".join(r) for r in rows]
(OUT / "methanol_water_lab_vle.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote methanol_water_lab_vle.csv")
