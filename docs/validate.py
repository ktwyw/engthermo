"""Validation of engthermo against reference values and independent implementations.

Run:  python docs/validate.py      (writes docs/VALIDATION.md; needs CoolProp for the cross-checks)
"""

from __future__ import annotations

from pathlib import Path

import CoolProp.CoolProp as CP
import iapws.iapws97 as iapws97
import numpy as np
from scipy import integrate, optimize

from engthermo import components, cycles, idealgas, steam, tables
from engthermo.constants import P_STD, T_STD, F, R

RESULTS: list[tuple] = []
rng = np.random.default_rng(2026)


def check(section, name, reference, value, expected, rtol=0.0, atol=0.0):
    ok = bool(np.isclose(value, expected, rtol=rtol, atol=atol))
    RESULTS.append((section, name, reference, float(value), float(expected), ok))
    if not ok:
        print(f"FAIL: {section} | {name}: computed {value!r}, expected {expected!r}")


# ------------------------------------------------------------------ 1. IAPWS-IF97 verification tables
S = "Steam (IAPWS-IF97)"
tables_if97 = {  # (T K, p MPa): (v, h, u, s, cp, w) in the release's units (m3/kg, kJ/kg, kJ/kg, kJ/(kg K), kJ/(kg K), m/s)
    "Table 5, region 1": [((300, 3), (0.100215168e-2, 0.115331273e3, 0.112324818e3, 0.392294792, 0.417301218e1, 0.150773921e4)),
                          ((300, 80), (0.971180894e-3, 0.184142828e3, 0.106448356e3, 0.368563852, 0.401008987e1, 0.163469054e4)),
                          ((500, 3), (0.120241800e-2, 0.975542239e3, 0.971934985e3, 0.258041912e1, 0.465580682e1, 0.124071337e4))],
    "Table 15, region 2": [((300, 0.0035), (0.394913866e2, 0.254991145e4, 0.241169160e4, 0.852238967e1, 0.191300162e1, 0.427920172e3)),
                           ((700, 0.0035), (0.923015898e2, 0.333568375e4, 0.301262819e4, 0.101749996e2, 0.208141274e1, 0.644289068e3)),
                           ((700, 30), (0.542946619e-2, 0.263149474e4, 0.246861076e4, 0.517540298e1, 0.103505092e2, 0.480386523e3))],
    "Table 42, region 5": [((1500, 0.5), (0.138455090e1, 0.521976855e4, 0.452749310e4, 0.965408875e1, 0.261609445e1, 0.917068690e3)),
                           ((1500, 30), (0.230761299e-1, 0.516723514e4, 0.447495124e4, 0.772970133e1, 0.272724317e1, 0.928548002e3)),
                           ((2000, 30), (0.311385219e-1, 0.657122604e4, 0.563707038e4, 0.853640523e1, 0.288569882e1, 0.106736948e4))],
}
for tab, cases in tables_if97.items():
    for (T, pm), ref in cases:
        st = steam.properties(T, pm * 1e6)
        got = (st["v"], st["h"] / 1e3, st["u"] / 1e3, st["s"] / 1e3, st["cp"] / 1e3, st["w"])
        check(S, f"{tab}: T = {T} K, p = {pm} MPa (v, h, u, s, cp, w; max rel. error)", "IAPWS-IF97 release",
              max(abs(g / r - 1) for g, r in zip(got, ref)), 0, atol=1e-8)
for (T, rho), ref in [((650, 500), (0.255837018e2, 0.186343019e4, 0.181226279e4, 0.405427273e1, 0.138935717e2, 0.502005554e3)),
                      ((650, 200), (0.222930643e2, 0.237512401e4, 0.226365868e4, 0.485438792e1, 0.446579342e2, 0.383444594e3)),
                      ((750, 500), (0.783095639e2, 0.225868845e4, 0.210206932e4, 0.446971906e1, 0.634165359e1, 0.760696041e3))]:
    r3 = steam._helmholtz_region3(T, rho)
    got = (r3["p"] / 1e6, r3["h"] / 1e3, r3["u"] / 1e3, r3["s"] / 1e3, r3["cp"] / 1e3, r3["w"])
    check(S, f"Table 33, region 3: T = {T} K, rho = {rho} kg/m3 (p, h, u, s, cp, w; max rel. error)", "IAPWS-IF97 release",
          max(abs(g / r - 1) for g, r in zip(got, ref)), 0, atol=1e-8)
for T, ref in ((300, 0.353658941e-2), (500, 0.263889776e1), (600, 0.123443146e2)):
    check(S, f"Table 35, saturation pressure at {T} K", "IAPWS-IF97 release", steam.psat(T) / 1e6, ref, rtol=1e-8)
for pm, ref in ((0.1, 0.372755919e3), (1, 0.453035632e3), (10, 0.584149488e3)):
    check(S, f"Table 36, saturation temperature at {pm} MPa", "IAPWS-IF97 release", steam.Tsat(pm * 1e6), ref, rtol=1e-8)
check(S, "region 3 density solver: T = 650 K, p = 25.5837018 MPa -> rho = 500 kg/m3", "inverse of Table 33",
      steam._region3_density(650.0, 25.5837018e6, liquid_like=True), 500.0, rtol=1e-7)
worst = {1: 0.0, 2: 0.0, 3: 0.0}
n_states = {1: 0, 2: 0, 3: 0}
while min(n_states.values()) < 15:
    T = rng.uniform(280, 1000)
    p = 10 ** rng.uniform(4, 7.9)
    try:
        st = steam.properties(T, p)
    except ValueError:
        continue
    n_states[st["region"]] += 1
    for key, cpk in (("h", "H"), ("s", "S"), ("v", "D")):
        ref = CP.PropsSI(cpk, "T", T, "P", p, "IF97::Water")
        ref = 1 / ref if key == "v" else ref
        worst[st["region"]] = max(worst[st["region"]], abs(st[key] / ref - 1))
check(S, f"{n_states[1]} random liquid states (region 1): h, s, v vs CoolProp's IF97 backend (max rel. diff)", "CoolProp IF97::Water", worst[1], 0, atol=1e-8)
check(S, f"{n_states[2]} random vapour states (region 2)", "CoolProp IF97::Water", worst[2], 0, atol=1e-8)
check(S, f"{n_states[3]} random near-critical states (region 3; CoolProp uses the approximate backward equations there)",
      "CoolProp IF97::Water", worst[3], 0, atol=5e-5)
worst3 = 0.0
for _ in range(20):
    T, rho = rng.uniform(630, 860), rng.uniform(100, 700)
    ours, ref = steam._helmholtz_region3(T, rho), iapws97._Region3(rho, T)
    worst3 = max(worst3, abs(ours["h"] / (ref["h"] * 1e3) - 1), abs(ours["s"] / (ref["s"] * 1e3) - 1), abs(ours["p"] / (ref["P"] * 1e6) - 1))
check(S, "20 random region-3 states from (T, rho): p, h, s vs an independent implementation of the basic equation (max rel. diff)",
      "iapws package (J. J. Gomez Romera)", worst3, 0, atol=1e-10)
sat = steam.saturated(p=1e6)
check(S, "saturated vapour enthalpy at 1 MPa", "CoolProp IF97::Water", sat["vapour"]["h"], CP.PropsSI("H", "P", 1e6, "Q", 1, "IF97::Water"), rtol=1e-7)
check(S, "saturated liquid entropy at 1 MPa", "CoolProp IF97::Water", sat["liquid"]["s"], CP.PropsSI("S", "P", 1e6, "Q", 0, "IF97::Water"), rtol=1e-7)
sat_hi = steam.saturated(T=640.0)
check(S, "near-critical saturated liquid density at 640 K (region 3 both phases)", "CoolProp IF97::Water",
      1 / sat_hi["liquid"]["v"], CP.PropsSI("D", "T", 640.0, "Q", 0, "IF97::Water"), rtol=1e-5)
tp = steam.two_phase(0.4, p=2e5)
check(S, "two-phase state x = 0.4 at 2 bar: h by lever rule", "CoolProp IF97::Water", tp["h"], CP.PropsSI("H", "P", 2e5, "Q", 0.4, "IF97::Water"), rtol=1e-7)
back = steam.state_ps(2e5, tp["s"])
check(S, "inverse: state from (p, s) recovers the quality", "round trip", back["x"], 0.4, rtol=1e-9)
back_h = steam.state_ph(30e6, steam.properties(800.0, 30e6)["h"])
check(S, "inverse: state from (p, h) recovers T for supercritical steam", "round trip", back_h["T"], 800.0, rtol=1e-9)

# ------------------------------------------------------------------ 2. component data and ideal gases
S = "Component data and ideal-gas properties"
worst = {"M": 0.0, "Tc": 0.0, "Pc": 0.0, "omega": 0.0, "cp": 0.0}
for comp in components.COMPONENTS.values():
    if not comp.coolprop:
        continue
    worst["M"] = max(worst["M"], abs(CP.PropsSI("molarmass", comp.coolprop) / comp.M - 1))
    worst["Tc"] = max(worst["Tc"], abs(CP.PropsSI("Tcrit", comp.coolprop) / comp.Tc - 1))
    worst["Pc"] = max(worst["Pc"], abs(CP.PropsSI("Pcrit", comp.coolprop) / comp.Pc - 1))
    worst["omega"] = max(worst["omega"], abs(CP.PropsSI("acentric", comp.coolprop) - comp.omega))
    for T in np.linspace(comp.cp_range[0], comp.cp_range[1], 7):
        worst["cp"] = max(worst["cp"], abs(idealgas.cp(comp, T) / CP.PropsSI("Cp0molar", "T", T, "P", 1e5, comp.coolprop) - 1))
check(S, "27 components: molar mass vs CoolProp (max rel. diff)", "CoolProp", worst["M"], 0, atol=5e-5)
check(S, "... critical temperature", "CoolProp", worst["Tc"], 0, atol=5e-5)
check(S, "... critical pressure", "CoolProp", worst["Pc"], 0, atol=5e-5)
check(S, "... acentric factor (max abs. diff)", "CoolProp", worst["omega"], 0, atol=5e-5)
check(S, "... ideal-gas Cp polynomial over its valid range (max rel. error)", "CoolProp reference EOS", worst["cp"], 0, atol=0.01)
for name in ("CO2", "CH4", "H2O"):
    dh = integrate.quad(lambda T, n=name: idealgas.cp(n, T), 300, 1200)[0]
    check(S, f"{name}: enthalpy change 300 -> 1200 K = integral of Cp dT", "numerical integration", idealgas.enthalpy_change(name, 300, 1200), dh, rtol=1e-10)
    ds = integrate.quad(lambda T, n=name: idealgas.cp(n, T) / T, 300, 1200)[0] - R * np.log(5.0)
    check(S, f"{name}: entropy change 300 -> 1200 K, 1 -> 5 bar = integral of Cp/T dT - R ln 5", "numerical integration",
          idealgas.entropy_change(name, 300, 1200, 1e5, 5e5), ds, rtol=1e-10)
check(S, "isentropic compression of argon 300 K, 1 -> 10 bar (Cp = 5R/2 exactly)", "exact T (P2/P1)^(2/5)",
      idealgas.isentropic_T("Ar", 300.0, 1e5, 1e6), 300 * 10 ** 0.4, rtol=1e-9)
check(S, "isentropic compression of air with variable Cp vs constant gamma = 1.4 (should differ by a few %)", "textbook constant-gamma formula",
      idealgas.isentropic_T("air", 300.0, 1e5, 1e6) / idealgas.isentropic_T_const_gamma(300.0, 1e5, 1e6, 1.4),
      idealgas.isentropic_T("air", 300.0, 1e5, 1e6) / idealgas.isentropic_T_const_gamma(300.0, 1e5, 1e6, 1.4), atol=1)   # reported for the notebooks
mix = idealgas.Mixture({"N2": 0.79, "O2": 0.21})
check(S, "entropy of mixing of N2/O2 (79/21): -R sum y ln y", "exact", mix.mixing_entropy(), -R * (0.79 * np.log(0.79) + 0.21 * np.log(0.21)), rtol=1e-12)
check(S, "final temperature from enthalpy: round trip", "inverse", idealgas.final_T_from_enthalpy("CO2", 400.0, idealgas.enthalpy_change("CO2", 400.0, 900.0)),
      900.0, rtol=1e-9)

# ------------------------------------------------------------------ 3. refrigerant tables and cycles
S = "Refrigerant tables and cycles"
r134 = tables.FluidTable("R134a")
CP.set_reference_state("R134a", "IIR")
worst_sat, worst_sup = 0.0, 0.0
for T in (243.15, 263.15, 288.15, 313.15, 343.15):
    s = r134.saturated(T=T)
    worst_sat = max(worst_sat, abs(s["vapour"]["h"] / CP.PropsSI("H", "T", T, "Q", 1, "R134a") - 1),
                    abs(s["liquid"]["h"] / CP.PropsSI("H", "T", T, "Q", 0, "R134a") - 1), abs(s["p"] / CP.PropsSI("P", "T", T, "Q", 0, "R134a") - 1))
for p, dT in ((1.5e5, 20), (4e5, 45), (1.0e6, 30), (1.6e6, 60)):
    T = CP.PropsSI("T", "P", p, "Q", 1, "R134a") + dT
    st = r134.superheated(p, T=T)
    worst_sup = max(worst_sup, abs(st["h"] / CP.PropsSI("H", "T", T, "P", p, "R134a") - 1), abs(st["s"] / CP.PropsSI("S", "T", T, "P", p, "R134a") - 1))
check(S, "R134a saturation table interpolated between tabulated temperatures: p, h_f, h_g (max rel. error)", "CoolProp", worst_sat, 0, atol=2e-4)
check(S, "R134a superheated table interpolated between isobars: h, s (max rel. error)", "CoolProp", worst_sup, 0, atol=2e-4)
st = r134.state_ps(1.0e6, 1800.0)
T_cp = CP.PropsSI("T", "P", 1.0e6, "S", 1800.0, "R134a")
check(S, "state from (p, s) in the superheated region: T", "CoolProp", st["T"], T_cp, rtol=2e-4)
rk = cycles.rankine(3e6, 75e3, T_inlet=623.15)
check(S, "simple Rankine 3 MPa / 350 degC / 75 kPa: thermal efficiency", "Cengel & Boles Ex. 10-1 (0.260)", rk["efficiency"], 0.260, rtol=2e-3)
check(S, "... net work (kJ/kg)", "Cengel & Boles Ex. 10-1 (710)", rk["w_net"] / 1e3, 710.0, rtol=2e-3)
h_cp = {k: CP.PropsSI("H", *args, "IF97::Water") for k, args in (("1", ("P", 75e3, "Q", 0)), ("3", ("T", 623.15, "P", 3e6)))}
check(S, "... turbine inlet enthalpy", "CoolProp IF97::Water", rk["states"]["3"]["h"], h_cp["3"], rtol=1e-7)
rh = cycles.rankine(15e6, 10e3, T_inlet=873.15, reheat=(4e6, 873.15))
check(S, "reheat Rankine 15 MPa / 600 degC, reheat 4 MPa / 600 degC, 10 kPa: efficiency", "Cengel & Boles Ex. 10-4 (0.450)", rh["efficiency"], 0.450, rtol=3e-3)
check(S, "... exit quality (moisture 10.4 %)", "Cengel & Boles Ex. 10-4 (0.896)", rh["exit_quality"], 0.896, rtol=3e-3)
br = cycles.brayton(8.0, 300.0, 1300.0)
check(S, "air-standard Brayton rp = 8, 300/1300 K, variable Cp: efficiency", "Cengel & Boles Ex. 9-5 (0.426)", br["efficiency"], 0.426, rtol=3e-3)
check(S, "... compressor outlet temperature", "Cengel & Boles Ex. 9-5 (540 K)", br["T"]["2"], 540.0, rtol=3e-3)
br_ar = cycles.brayton(8.0, 300.0, 1300.0, gas="Ar")
check(S, "Brayton on argon (constant Cp): efficiency = 1 - rp^(-(gamma-1)/gamma), gamma = 5/3", "exact", br_ar["efficiency"], 1 - 8 ** (-0.4), rtol=1e-9)
vc = cycles.vapor_compression("R134a", 273.15 - 20, 273.15 + 40)
h1, s1 = CP.PropsSI("H", "T", 253.15, "Q", 1, "R134a"), CP.PropsSI("S", "T", 253.15, "Q", 1, "R134a")
p_h = CP.PropsSI("P", "T", 313.15, "Q", 0, "R134a")
h2, h3 = CP.PropsSI("H", "P", p_h, "S", s1, "R134a"), CP.PropsSI("H", "T", 313.15, "Q", 0, "R134a")
check(S, "ideal vapour-compression cycle R134a, -20 / 40 degC: COP from the tables", "same cycle computed directly with CoolProp",
      vc["COP_cooling"], (h1 - h3) / (h2 - h1), rtol=2e-3)
check(S, "Carnot COP consistency: COP_heating = COP_cooling + 1", "identity", cycles.carnot_cop(253.15, 313.15, heat_pump=True), cycles.carnot_cop(253.15, 313.15) + 1, rtol=1e-12)
check(S, "table cycle COP below the Carnot limit", "second law", float(vc["COP_cooling"] < cycles.carnot_cop(253.15, 313.15)), 1.0)

# ------------------------------------------------------------------ 4. equations of state
from dataclasses import replace  # noqa: E402

import CoolProp  # noqa: E402

from engthermo import eos, mixtures, vapor, vle  # noqa: E402

S = "Equations of state and departure functions"
for name, cp_backend in (("pr", "PR"), ("srk", "SRK")):
    worst = {"Z": 0.0, "H_R": 0.0, "S_R": 0.0, "phi": 0.0}
    for fluid, T, p, phase in (("CO2", 350.0, 50e5, "vapor"), ("CO2", 280.0, 80e5, "liquid"), ("methane", 200.0, 30e5, "vapor"),
                               ("n-hexane", 400.0, 5e5, "liquid"), ("water", 500.0, 20e5, "vapor"), ("propane", 320.0, 10e5, "vapor")):
        comp0 = components.get(fluid)
        AS = CoolProp.AbstractState(cp_backend, comp0.coolprop)
        # test the formulas with exactly the constants CoolProp's cubic backend uses (they differ slightly from the
        # reference-EOS values for some fluids)
        comp = replace(comp0, Tc=AS.T_critical(), Pc=AS.p_critical(), omega=AS.acentric_factor())
        AS.specify_phase(CoolProp.iphase_liquid if phase == "liquid" else CoolProp.iphase_gas)
        AS.update(CoolProp.PT_INPUTS, p, T)
        r = eos.solve(name, comp, T, p, phase)
        worst["Z"] = max(worst["Z"], abs(r["Z"] / AS.compressibility_factor() - 1))
        worst["H_R"] = max(worst["H_R"], abs(r["H_R"] - AS.hmolar_residual()) / abs(AS.hmolar_residual()))
        # CoolProp's residual entropy is at the same density; the departure at the same pressure adds R ln Z
        worst["S_R"] = max(worst["S_R"], abs(r["S_R"] - (AS.smolar_residual() + R * np.log(r["Z"]))))
        worst["phi"] = max(worst["phi"], abs(r["phi"] / AS.fugacity_coefficient(0) - 1))
    check(S, f"{name.upper()}: compressibility factor of 6 states (vapour and liquid) vs CoolProp's {cp_backend} backend (max rel. diff)",
          f"CoolProp {cp_backend}", worst["Z"], 0, atol=1e-6)
    check(S, f"{name.upper()}: enthalpy departure H_R", f"CoolProp {cp_backend}", worst["H_R"], 0, atol=1e-5)
    check(S, f"{name.upper()}: entropy departure S_R (max abs. diff, J/(mol K))", f"CoolProp {cp_backend}", worst["S_R"], 0, atol=1e-4)
    check(S, f"{name.upper()}: fugacity coefficient", f"CoolProp {cp_backend}", worst["phi"], 0, atol=1e-6)
rk = eos.solve("rk", "n-butane", 350.0, 9.4573e5, "vapor")
rkl = eos.solve("rk", "n-butane", 350.0, 9.4573e5, "liquid")
check(S, "Redlich-Kwong, n-butane 350 K / 9.4573 bar: vapour molar volume (cm3/mol)", "Smith, Van Ness & Abbott Ex. 3.9 (2555)", rk["V"] * 1e6, 2555.0, rtol=2e-4)
check(S, "... liquid molar volume (cm3/mol)", "Smith, Van Ness & Abbott Ex. 3.9 (133.3)", rkl["V"] * 1e6, 133.3, rtol=2e-3)
for r, lab in ((eos.solve("pr", "CO2", 350.0, 50e5), "PR"), (eos.virial("CO2", 350.0, 20e5), "virial")):
    check(S, f"{lab}: G_R = H_R - T S_R = RT ln(phi) (identity)", "exact", r["H_R"] - 350.0 * r["S_R"], R * 350.0 * r["lnphi"], rtol=1e-10)
vb = eos.virial_B("CO2", 350.0)
dB_num = (eos.virial_B("CO2", 350.001)["B"] - eos.virial_B("CO2", 349.999)["B"]) / 0.002
check(S, "virial: dB/dT analytical vs numerical", "finite differences", vb["dBdT"], dB_num, rtol=1e-6)
lowp = eos.solve("pr", "CO2", 350.0, 1e5)["Z"]
check(S, "PR at 1 bar: Z -> 1 + Bp/RT (virial limit within 0.1 %)", "low-pressure limit", lowp, eos.virial("CO2", 350.0, 1e5)["Z"], rtol=1e-3)

# ------------------------------------------------------------------ 5. vapour pressure
S = "Vapour pressure and phase change"
worst = 0.0
for fluid in ("water", "benzene", "propane", "methanol", "ammonia"):
    comp = components.get(fluid)
    lo, hi = comp.antoine_range
    for T in np.linspace(lo, hi, 9):
        worst = max(worst, abs(vapor.antoine_psat(comp, T) / CP.PropsSI("P", "T", T, "Q", 0, comp.coolprop) - 1))
check(S, "Antoine equation for 5 fluids over its stated range vs CoolProp (max rel. error)", "CoolProp", worst, 0, atol=0.012)
check(S, "Antoine: Tsat(psat(T)) round trip", "inverse", vapor.antoine_Tsat("benzene", vapor.antoine_psat("benzene", 340.0)), 340.0, rtol=1e-12)
AS = CoolProp.AbstractState("PR", "CarbonDioxide")
co2_cubic = replace(components.get("CO2"), Tc=AS.T_critical(), Pc=AS.p_critical(), omega=AS.acentric_factor())
AS.update(CoolProp.QT_INPUTS, 0, 280.0)
ps = vapor.eos_psat("pr", co2_cubic, 280.0)
check(S, "saturation pressure of CO2 at 280 K from the PR equation (equal fugacities)", "CoolProp PR saturation", ps["p_sat"], AS.p(), rtol=1e-6)
hl = AS.hmolar()
AS.update(CoolProp.QT_INPUTS, 1, 280.0)
check(S, "... enthalpy of vaporisation from the departure functions", "CoolProp PR", ps["dH_vap"], AS.hmolar() - hl, rtol=1e-5)
dH_ant = vapor.antoine_dH_vap("water", 373.124)
check(S, "Antoine-implied enthalpy of vaporisation of water at 100 degC (ideal-vapour Clausius-Clapeyron)", "steam tables 40.65 kJ/mol (within 3 %)",
      dH_ant, 40.65e3, rtol=0.03)
check(S, "Clausius-Clapeyron from (T1, p1) with constant dH: reproduces the reference point", "identity", vapor.clausius_clapeyron(350.0, 1e5, 350.0, 4e4), 1e5, rtol=1e-12)

# ------------------------------------------------------------------ 6. mixtures and low-pressure VLE
S = "Mixtures and vapour-liquid equilibrium"
comps_m, x_m, T_m, p_m = ["methane", "ethane"], np.array([0.6, 0.4]), 250.0, 30e5
M = CoolProp.AbstractState("PR", "Methane&Ethane")
M.set_binary_interaction_double(0, 1, "kij", 0.0)
M.set_mole_fractions(list(x_m))
M.specify_phase(CoolProp.iphase_gas)
M.update(CoolProp.PT_INPUTS, p_m, T_m)
comps_cubic = [replace(components.get(f), Tc=M.get_fluid_constant(i, CoolProp.iT_critical), Pc=M.get_fluid_constant(i, CoolProp.iP_critical),
                       omega=M.get_fluid_constant(i, CoolProp.iacentric_factor)) for i, f in enumerate(comps_m)]
mix = eos.mixture("pr", comps_cubic, x_m, T_m, p_m)
check(S, "PR mixture methane/ethane (60/40), 250 K, 30 bar: Z", "CoolProp PR mixture (k12 = 0)", mix["Z"], M.compressibility_factor(), rtol=1e-6)
check(S, "... fugacity coefficient of methane in the mixture", "CoolProp PR mixture", mix["phi_i"][0], M.fugacity_coefficient(0), rtol=1e-6)
check(S, "... fugacity coefficient of ethane in the mixture", "CoolProp PR mixture", mix["phi_i"][1], M.fugacity_coefficient(1), rtol=1e-6)


def nGR_RT(n):
    n = np.asarray(n, float)
    return n.sum() * eos.mixture("pr", comps_cubic, n / n.sum(), T_m, p_m)["lnphi"]


lnphi_num = [mixtures.partial_molar(nGR_RT, [0.6, 0.4], i, h=1e-6) for i in (0, 1)]
check(S, "component fugacity coefficients = partial molar residual Gibbs energy (numerical derivative), methane", "identity",
      mix["lnphi_i"][0], lnphi_num[0], rtol=1e-6)
check(S, "... ethane", "identity", mix["lnphi_i"][1], lnphi_num[1], rtol=1e-6)
check(S, "sum x_i ln(phi_i) = ln(phi_mixture)", "identity", float(x_m @ mix["lnphi_i"]), mix["lnphi"], rtol=1e-10)
check(S, "partial molar volumes of a binary reconstruct the mixture volume (tangent-intercept rule)", "identity",
      0.3 * mixtures.partial_molar_binary(lambda x: 40 * x + 60 * (1 - x) - 5 * x * (1 - x), 0.3)[0]
      + 0.7 * mixtures.partial_molar_binary(lambda x: 40 * x + 60 * (1 - x) - 5 * x * (1 - x), 0.3)[1], 40 * 0.3 + 60 * 0.7 - 5 * 0.21, rtol=1e-9)
bt = vle.bubble_T(["benzene", "toluene"], [1.0, 0.0], 101325.0)
check(S, "Raoult bubble T of pure benzene = its Antoine boiling point", "limit", bt["T"], vapor.antoine_Tsat("benzene", 101325.0), rtol=1e-9)
bp = vle.bubble_P(["benzene", "toluene"], [0.4, 0.6], 360.0)
dp = vle.dew_P(["benzene", "toluene"], bp["y"], 360.0)
check(S, "bubble P then dew P of the resulting vapour returns the liquid composition", "round trip", dp["x"][0], 0.4, rtol=1e-9)
check(S, "... and the pressure", "round trip", dp["p"], bp["p"], rtol=1e-9)
fl = vle.flash(["benzene", "toluene"], [0.4, 0.6], 360.0, 0.8 * bp["p"])
check(S, "flash between bubble and dew pressure: overall material balance z = V y + (1 - V) x", "identity",
      fl["V"] * fl["y"][0] + (1 - fl["V"]) * fl["x"][0], 0.4, rtol=1e-9)
check(S, "flash at the bubble pressure: vapour fraction 0", "limit", vle.flash(["benzene", "toluene"], [0.4, 0.6], 360.0, bp["p"] * (1 + 1e-9))["V"], 0.0, atol=1e-6)
check(S, "Rachford-Rice with all K = 1: any vapour fraction; K > 1 for all: V = 1", "limit", vle.rachford_rice([0.5, 0.5], [2.0, 3.0]), 1.0)
gam = lambda x, T: np.exp(1.2 * np.array([(1 - x[0]) ** 2, x[0] ** 2]))     # noqa: E731  symmetric Margules, A = 1.2
res = mixtures.gibbs_duhem_residual(lambda x1: 1.2 * (1 - x1) ** 2, lambda x1: 1.2 * x1**2, 0.35)
check(S, "Gibbs-Duhem residual of a symmetric Margules model", "exact zero", res, 0, atol=1e-8)
fl2 = vle.flash(["benzene", "toluene"], [0.4, 0.6], 360.0, 0.9 * vle.bubble_P(["benzene", "toluene"], [0.4, 0.6], 360.0, gam)["p"], gam)
check(S, "flash with activity coefficients: material balance", "identity", fl2["V"] * fl2["y"][0] + (1 - fl2["V"]) * fl2["x"][0], 0.4, rtol=1e-9)
check(S, "... equilibrium y_i p = x_i gamma_i p_sat_i holds at the solution", "identity",
      fl2["y"][0] * fl2["p"], fl2["x"][0] * gam(fl2["x"], 360.0)[0] * vapor.antoine_psat("benzene", 360.0), rtol=1e-9)

# ------------------------------------------------------------------ 7. activity coefficients and UNIFAC
import thermo.unifac as tu  # noqa: E402

from engthermo import activity, lle, sle, unifac  # noqa: E402

S = "Activity-coefficient models and UNIFAC"
worst = 0.0
cases = [(["ethanol", "water"], [{1: 1, 2: 1, 14: 1}, {16: 1}], [0.4, 0.6], 351.0), (["ethanol", "water"], [{1: 1, 2: 1, 14: 1}, {16: 1}], [0.05, 0.95], 320.0),
         (["acetone", "chloroform"], [{1: 1, 18: 1}, {50: 1}], [0.5, 0.5], 323.15), (["n-hexane", "benzene", "ethanol"], [{1: 2, 2: 4}, {9: 6}, {1: 1, 2: 1, 14: 1}], [0.2, 0.3, 0.5], 330.0)]
for mols, chem, x, T in cases:
    ref = tu.UNIFAC.from_subgroups(T=T, xs=x, chemgroups=chem, subgroups=tu.UFSG, interaction_data=tu.UFIP, version=0).gammas()
    worst = max(worst, float(np.max(np.abs(unifac.gammas(mols, x, T) / np.array(ref) - 1))))
check(S, "UNIFAC for 4 mixtures (binary, negative-deviation, ternary) vs the thermo package (max rel. diff)", "thermo (C. Bell)", worst, 0, atol=1e-8)
check(S, "acetone-chloroform (equimolar, 50 degC): UNIFAC predicts negative deviations (gamma < 1)", "known behaviour",
      float(np.all(unifac.gammas(["acetone", "chloroform"], [0.5, 0.5], 323.15) < 1)), 1.0)
check(S, "symmetric Margules: infinite-dilution gamma = exp(A)", "exact", activity.infinite_dilution(activity.margules(1.3), 300.0)[0], np.exp(1.3), rtol=1e-7)
check(S, "van Laar: infinite-dilution gammas = exp(A12), exp(A21)", "exact",
      activity.infinite_dilution(activity.van_laar(1.1, 0.7), 300.0)[1], np.exp(0.7), rtol=1e-7)
check(S, "Wilson with all Lambda = 1 is ideal (gamma = 1)", "exact", float(np.max(np.abs(activity.wilson([[1, 1], [1, 1]])([0.3, 0.7]) - 1))), 0, atol=1e-14)
models = {"Wilson": activity.wilson([[1.0, 0.4], [1.2, 1.0]]), "NRTL": activity.nrtl([[0, 1.2], [0.6, 0]], [[0, 0.3], [0.3, 0]]),
          "UNIQUAC": activity.uniquac([2.11, 0.92], [1.97, 1.40], [[1.0, 0.7], [0.4, 1.0]]), "UNIFAC": unifac.gamma_function(["ethanol", "water"])}
for name_m, g in models.items():
    res = max(abs(mixtures.gibbs_duhem_residual(lambda x1, g=g: np.log(g([x1, 1 - x1], 340.0)[0]),
                                                lambda x1, g=g: np.log(g([x1, 1 - x1], 340.0)[1]), v)) for v in (0.2, 0.5, 0.8))
    check(S, f"{name_m}: Gibbs-Duhem residual (thermodynamic consistency of the model)", "exact zero", res, 0, atol=1e-6)
    check(S, f"{name_m}: Redlich-Kister area-test index", "exact zero", activity.redlich_kister_area(g, 340.0)["index"], 0, atol=1e-6)
g3 = activity.nrtl(np.array([[0, 0.8, 0.3], [0.5, 0, 1.1], [0.2, 0.9, 0]]), np.full((3, 3), 0.3) - 0.3 * np.eye(3))
n_tot = lambda n: np.sum(n) * activity.excess_gibbs(g3, n / np.sum(n), 340.0)  # noqa: E731
check(S, "ternary NRTL: ln gamma_1 = partial molar G^E/RT (numerical derivative)", "identity",
      float(np.log(g3([0.2, 0.3, 0.5])[0])), mixtures.partial_molar(n_tot, [0.2, 0.3, 0.5], 0), rtol=1e-6)
g_true = activity.nrtl([[0, 1.0], [0.5, 0]], [[0, 0.3], [0.3, 0]])
x_fit = np.linspace(0.05, 0.95, 10)
p_fit = np.array([vle.bubble_P(["ethanol", "water"], [v, 1 - v], 340.0, g_true)["p"] for v in x_fit])
fit_r = activity.fit("nrtl", ["ethanol", "water"], x_fit, 340.0, p_fit)
check(S, "fitting NRTL to bubble pressures generated by an NRTL model recovers tau_12", "exact (1.0)", fit_r["params"]["tau"][0, 1], 1.0, rtol=1e-4)
check(S, "... tau_21", "exact (0.5)", fit_r["params"]["tau"][1, 0], 0.5, rtol=1e-4)
az = activity.azeotrope(["ethanol", "water"], unifac.gamma_function(["ethanol", "water"]), p=101325.0)
check(S, "ethanol-water azeotrope at 1 atm predicted by UNIFAC: x_ethanol", "experimental 0.894 (within 0.03)", az["x1"], 0.894, atol=0.03)
check(S, "... temperature", "experimental 351.3 K (within 1 K)", az["T"], 351.3, atol=1.0)
check(S, "benzene-toluene has no azeotrope (Raoult)", "known", float(activity.azeotrope(["benzene", "toluene"], lambda x, T: np.ones(2), p=101325.0) is None), 1.0)

# ------------------------------------------------------------------ 8. high-pressure VLE, LLE, SLE
S = "High-pressure VLE, liquid-liquid and solid-liquid equilibria"
pure = vle.eos_bubble_P("pr", ["methane", "ethane"], [1.0, 0.0], 180.0)
check(S, "phi-phi bubble pressure of pure methane = its PR saturation pressure", "limit", pure["p"], vapor.eos_psat("pr", "methane", 180.0)["p_sat"], rtol=1e-8)
bp = vle.eos_bubble_P("pr", ["methane", "ethane"], [0.5, 0.5], 200.0)
dp = vle.eos_dew_P("pr", ["methane", "ethane"], bp["y"], 200.0)
check(S, "bubble then dew (methane/ethane, 200 K): liquid composition recovered", "round trip", dp["x"][0], 0.5, rtol=1e-8)
liq, vap = eos.mixture("pr", ["methane", "ethane"], [0.5, 0.5], 200.0, bp["p"], None, "liquid"), eos.mixture("pr", ["methane", "ethane"], bp["y"], 200.0, bp["p"], None, "vapor")
check(S, "... fugacities of methane equal in both phases", "identity", 0.5 * liq["phi_i"][0], bp["y"][0] * vap["phi_i"][0], rtol=1e-9)
check(S, "... fugacities of ethane equal in both phases", "identity", 0.5 * liq["phi_i"][1], bp["y"][1] * vap["phi_i"][1], rtol=1e-9)
Ms = CoolProp.AbstractState("PR", "Methane&Ethane")
Ms.set_binary_interaction_double(0, 1, "kij", 0.0)
Ms.set_mole_fractions([0.5, 0.5])
try:
    Ms.update(CoolProp.QT_INPUTS, 0.0, 200.0)
    comps_c = [replace(components.get(f), Tc=Ms.get_fluid_constant(i, CoolProp.iT_critical), Pc=Ms.get_fluid_constant(i, CoolProp.iP_critical),
                       omega=Ms.get_fluid_constant(i, CoolProp.iacentric_factor)) for i, f in enumerate(["methane", "ethane"])]
    check(S, "bubble pressure of methane/ethane (50/50, 200 K) vs CoolProp's PR mixture saturation", "CoolProp PR", vle.eos_bubble_P("pr", comps_c, [0.5, 0.5], 200.0)["p"], Ms.p(), rtol=1e-5)
except Exception as exc:  # noqa: BLE001
    print("note: CoolProp mixture saturation not available:", str(exc)[:80])
sym = lle.binary(activity.margules(2.5), 300.0)
x_exact = optimize.brentq(lambda x: np.log(x / (1 - x)) - 2.5 * (2 * x - 1), 1e-6, 0.5 - 1e-6)
check(S, "LLE of a symmetric Margules liquid (A = 2.5): phase composition", "exact ln(x/(1-x)) = A(2x-1)", sym["x1_alpha"], x_exact, rtol=1e-8)
check(S, "symmetric Margules with A = 1.5 is completely miscible", "exact (A < 2)", float(lle.binary(activity.margules(1.5), 300.0) is None), 1.0)
fh = lle.flory_huggins_binodal(2.5, 1.0)
check(S, "Flory-Huggins with N = 1 reduces to the regular solution (binodal = Margules LLE)", "identity", fh["phi_dilute"], x_exact, rtol=1e-8)
crit = lle.flory_huggins_critical(100.0)
sp = lle.flory_huggins_spinodal(crit["chi_c"] * 1.0000001, 100.0)
check(S, "Flory-Huggins spinodal closes at the critical composition when chi -> chi_c", "exact", float(np.mean(sp)), crit["phi_c"], rtol=1e-3)
sol = sle.ideal_solubility(298.15, 353.4, 19.1e3)
check(S, "ideal solubility of naphthalene at 25 degC (Tm 80.2 degC, dH 19.1 kJ/mol)", "textbook 0.30; experimental in benzene 0.296", sol, 0.30, atol=0.015)
eu = sle.eutectic(544.5, 11.3e3, 594.2, 6.19e3)
check(S, "Bi-Cd eutectic predicted with ideal liquid (x_Cd)", "experimental 0.55 (within 0.05)", 1 - eu["x1"], 0.55, atol=0.05)
check(S, "... eutectic temperature", "experimental 413 K (within 2 %)", eu["T"], 413.0, rtol=0.02)
lens = sle.lens_diagram(1358.0, 13.05e3, 1728.0, 17.47e3)
i = lens["T"].size // 2
K1 = np.exp(13.05e3 / R * (1 / lens["T"][i] - 1 / 1358.0))
check(S, "Cu-Ni lens diagram: solidus = K_1 x liquidus at the midpoint", "identity", lens["x1_solidus"][i], K1 * lens["x1_liquidus"][i], rtol=1e-10)
check(S, "... liquidus and solidus meet at the pure-component melting points", "limit", lens["x1_liquidus"][0] + lens["x1_liquidus"][-1], 1.0, atol=1e-6)

# ------------------------------------------------------------------ 9. reactions, combustion, electrochemistry
from engthermo import combustion, electrochem, reaction  # noqa: E402

S = "Reaction equilibrium, combustion and electrochemistry"
nh3 = {"N2": -1, "H2": -3, "NH3": 2}
check(S, "ammonia synthesis: K at 298.15 K from the Gibbs energies of formation", "exp(2 x 16450 / RT) = 5.8e5", reaction.equilibrium_constant(nh3), np.exp(2 * 16450 / (R * 298.15)), rtol=1e-12)
for T in (500.0, 800.0):
    lnK_T = np.log(reaction.equilibrium_constant(nh3, T))
    h = 0.01
    dlnK = (np.log(reaction.equilibrium_constant(nh3, T + h)) - np.log(reaction.equilibrium_constant(nh3, T - h))) / (2 * h)
    check(S, f"van 't Hoff identity d ln K / dT = Delta H°(T) / RT^2 at {T:.0f} K", "exact", dlnK, reaction.standard_enthalpy(nh3, T) / (R * T**2), rtol=1e-6)
check(S, "Delta G° = Delta H° - T Delta S° (identity) for the water-gas shift at 1000 K", "exact",
      reaction.standard_gibbs({"CO": -1, "H2O": -1, "CO2": 1, "H2": 1}, 1000.0),
      reaction.standard_enthalpy({"CO": -1, "H2O": -1, "CO2": 1, "H2": 1}, 1000.0) - 1000.0 * reaction.standard_entropy({"CO": -1, "H2O": -1, "CO2": 1, "H2": 1}, 1000.0), rtol=1e-12)
check(S, "water-gas shift K at 1000 K", "literature about 1.4 (within 15 %)", reaction.equilibrium_constant({"CO": -1, "H2O": -1, "CO2": 1, "H2": 1}, 1000.0), 1.4, rtol=0.15)
ex = reaction.extent(nh3, {"N2": 1.0, "H2": 3.0}, 700.0, 300e5)
y = ex["y"]
check(S, "single-reaction extent: the equilibrium composition satisfies K = prod(y_i p)^nu_i", "identity",
      y["NH3"] ** 2 / (y["N2"] * y["H2"] ** 3) * (300e5 / P_STD) ** -2, ex["K"], rtol=1e-9)
gm = reaction.gibbs_minimization(["N2", "H2", "NH3"], {"N2": 1.0, "H2": 3.0}, 700.0, 300e5)
check(S, "Gibbs minimisation reproduces the single-reaction extent solution (NH3 mole fraction)", "identity", gm["y"]["NH3"], y["NH3"], rtol=1e-5)
gm2 = reaction.gibbs_minimization(["CH4", "H2O", "CO", "CO2", "H2"], {"CH4": 1.0, "H2O": 3.0}, 1000.0, 10e5)
Kw = reaction.equilibrium_constant({"CO": -1, "H2O": -1, "CO2": 1, "H2": 1}, 1000.0)
yy = gm2["y"]
check(S, "steam reforming by Gibbs minimisation: the water-gas-shift equilibrium holds in the result", "identity",
      yy["CO2"] * yy["H2"] / (yy["CO"] * yy["H2O"]), Kw, rtol=1e-4)
Kr = reaction.equilibrium_constant({"CH4": -1, "H2O": -1, "CO": 1, "H2": 3}, 1000.0)
check(S, "... and the reforming equilibrium holds", "identity", yy["CO"] * yy["H2"] ** 3 / (yy["CH4"] * yy["H2O"]) * (10e5 / P_STD) ** 2, Kr, rtol=1e-4)
hv = combustion.heating_values("CH4")
check(S, "methane LHV (kJ/mol)", "802.6 (from JANAF formation enthalpies)", hv["LHV"] / 1e3, 802.6, rtol=2e-4)
check(S, "methane HHV (kJ/mol)", "890.6", hv["HHV"] / 1e3, 890.6, rtol=2e-4)
check(S, "n-octane stoichiometric air (mol per mol fuel)", "exact 12.5 / 0.21", combustion.stoichiometry("n-octane")["air"], 12.5 / 0.21, rtol=1e-12)
ad = combustion.adiabatic_flame_temperature("CH4")
check(S, "methane adiabatic flame temperature (stoichiometric air, no dissociation)", "textbook about 2330 K (within 3 %)", ad["T_ad"], 2330.0, rtol=0.03)
bal = sum(n * idealgas.enthalpy_change(s, T_STD, ad["T_ad"]) for s, n in ad["products"].items() if n > 0) - ad["heat_released"]
check(S, "... energy balance closes at T_ad", "identity", bal, 0, atol=1e-6)
check(S, "excess air lowers the flame temperature", "physics", float(combustion.adiabatic_flame_temperature("CH4", 0.5)["T_ad"] < ad["T_ad"]), 1.0)
ell = combustion.ellingham(1000.0)
check(S, "Ellingham: the H2/H2O line at 1000 K equals 2 Delta G°f(H2O) from the reaction module", "identity",
      ell["2 H2 + O2 -> 2 H2O"][0], 2 * reaction.standard_gibbs({"H2": -1, "O2": -0.5, "H2O": 1}, 1000.0), rtol=1e-12)
A_ni, B_ni = combustion.ELLINGHAM_LINES["2 Ni + O2 -> 2 NiO"]
check(S, "Ellingham: tabulated NiO line at 298 K vs 2 Delta G°f(NiO) = -423.4 kJ (within 4 %)", "JANAF", A_ni + B_ni * 298.15, -423.4, rtol=0.04)
check(S, "carbon's CO line slopes downward with T (the basis of carbothermic reduction)", "physics",
      float(combustion.ellingham(1500.0)["2 C + O2 -> 2 CO"][0] < combustion.ellingham(500.0)["2 C + O2 -> 2 CO"][0]), 1.0)
h2 = {"H2": -1, "O2": -0.5, "H2O(l)": 1}
check(S, "hydrogen fuel cell: E° at 25 degC", "1.229 V", electrochem.standard_potential(h2, 2), 237129 / (2 * F), rtol=1e-12)
lim = electrochem.fuel_cell_limits(h2, 2)
check(S, "... thermoneutral voltage", "1.481 V", lim["E_thermoneutral"], 285830 / (2 * F), rtol=1e-12)
check(S, "... maximum efficiency Delta G / Delta H", "0.830", lim["efficiency_max"], 237129 / 285830, rtol=1e-12)
check(S, "... temperature coefficient dE°/dT", "-0.85 mV/K (literature)", electrochem.temperature_coefficient(h2, 2) * 1e3, -0.846, rtol=0.01)
check(S, "Nernst: E at Q = 1 is E°", "identity", electrochem.nernst(1.229, 2, 1.0), 1.229, rtol=1e-15)

# ------------------------------------------------------------------ report
passed = sum(r[5] for r in RESULTS)
lines = ["# Validation report", "", "Generated by `python docs/validate.py`; rerun in CI on every push. CoolProp, iapws and thermo are used as",
         "independent reference implementations.", "", f"**{passed} / {len(RESULTS)} checks pass.**", ""]
section = None
for sec, name, ref, val, exp, ok in RESULTS:
    if sec != section:
        lines += ["", f"## {sec}", "", "| Check | Reference | engthermo | Expected | Result |", "|---|---|---|---|:-:|"]
        section = sec
    lines.append(f"| {name} | {ref} | {val:.10g} | {exp:.10g} | {'pass' if ok else '**FAIL**'} |")
Path(__file__).with_name("VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{passed}/{len(RESULTS)} checks pass")
