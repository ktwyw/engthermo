"""Generate the course notebooks (single source of truth) and execute them.

    python notebooks/build_notebooks.py            # write and execute all notebooks
    python notebooks/build_notebooks.py 03 15      # only notebooks whose names start with 03 or 15
    python notebooks/build_notebooks.py --no-run   # write without executing
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
NOTEBOOKS: dict[str, list] = {}
EXTRAS: dict[str, dict] = {}

COLAB = '''try:                      # on Google Colab (or anywhere engthermo is missing): install it from GitHub
    import engthermo
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/ktwyw/engthermo"], check=True)'''

STYLE = '''plt.rcParams.update({"figure.figsize": (7, 4), "figure.dpi": 90, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

import inspect
from IPython.display import Code

def show_source(obj):
    """Display the source code of a library function or class."""
    return Code(inspect.getsource(obj), language="python")

def heading(text):
    print(f"\\n{text}\\n" + "=" * len(text))'''


def md(text):
    return new_markdown_cell(text.strip("\n"))


def code(text):
    return new_code_cell(text.strip("\n"))


def setup_cell(extra_imports=()):
    lines = ["import matplotlib.pyplot as plt", "import numpy as np", "import pandas as pd"]
    for imp in extra_imports:
        if imp not in lines:
            lines.append(imp)
    return code(COLAB + "\n" + "\n".join(lines) + "\n" + STYLE)


def read_data(name):
    """Code snippet reading a bundled CSV file (comment lines start with #)."""
    return f'pd.read_csv(datasets.path("{name}"), comment="#")'



# =====================================================================================================
NOTEBOOKS["00_python_for_thermodynamics"] = [
    md(r"""
# 00 · Python for thermodynamics

**Goal:** the Python tools used throughout the course - unit conversions, property tables and how to read
them in code, root finding for inverse problems, and plotting property diagrams - on the substances of the
first notebooks: water and the refrigerant R134a.

Thermodynamics is a subject of tables and diagrams. Doing it in Python means the tables are looked up by a
function instead of by finger, every calculation is reproducible, and a whole diagram costs one loop.
"""),
    setup_cell(["from scipy import optimize", "from engthermo import datasets, steam, tables"]),
    md(r"""
## Units
`engthermo` uses SI units everywhere: K, Pa, J, kg, mol. Textbooks and data sheets use degC, bar, kPa and kJ; convert
at the edges of a calculation, never in the middle:
"""),
    code(r"""def C_to_K(t):
    return t + 273.15
bar, kPa, kJ = 1e5, 1e3, 1e3
st = steam.properties(T=C_to_K(350), p=3 * 1e6)
print(f"steam at 350 degC and 3 MPa: h = {st['h']/kJ:.1f} kJ/kg, s = {st['s']/kJ:.4f} kJ/(kg K), v = {st['v']:.5f} m3/kg")"""),
    md(r"""
## Property tables in a file
The refrigerant tables are plain CSV files. Look at one, then read it with pandas:
"""),
    code(r"""path = datasets.path("R134a_saturation.csv")
with open(path, encoding="utf-8") as fh:
    print("".join(fh.readlines()[:5]))
sat = pd.read_csv(path, comment="#")
print(sat.shape)
print(sat.iloc[[0, 20, 40, 60, 70]].to_string(index=False))"""),
    md(r"""
The header lines say where the numbers come from (CoolProp, with the IIR reference state that refrigeration
handbooks use: h = 200 kJ/kg and s = 1 kJ/(kg K) for saturated liquid at 0 degC). Always read the header:
enthalpies from tables with different reference states cannot be mixed.

## Interpolating in a table
The table lists every 2 K. At -7 degC (266.15 K), a table user interpolates between the rows at 265.15 K and
267.15 K; `np.interp` does the same:
"""),
    code(r"""T = 266.15
h_g = np.interp(T, sat.T_K, sat.h_g)
p = np.exp(np.interp(T, sat.T_K, np.log(sat.p_Pa)))                 # pressure: interpolate ln p, which is nearly linear in T
r134 = tables.FluidTable("R134a")
lib = r134.saturated(T=T)
print(f"by hand: h_g = {h_g/1e3:.2f} kJ/kg, p_sat = {p/1e3:.2f} kPa;  library: {lib['vapour']['h']/1e3:.2f} kJ/kg, {lib['p']/1e3:.2f} kPa")"""),
    md(r"""
Interpolating the logarithm of the pressure is the trick to remember: vapour pressure rises exponentially
with temperature, so a straight line through ln p is far more accurate than one through p itself.

## Inverse problems: root finding
Tables give properties from (T, p). Often the question is the other way round - which temperature gives
this enthalpy? A root finder answers it:
"""),
    code(r"""p = 1e6
target_h = 3.0e6                                                   # J/kg
f = lambda T: steam.properties(T, p)["h"] - target_h
T_found = optimize.brentq(f, steam.Tsat(p) + 0.01, 1073.15)
print(f"steam at 1 MPa with h = 3000 kJ/kg: T = {T_found - 273.15:.2f} degC   (library: {steam.state_ph(p, target_h)['T'] - 273.15:.2f} degC)")"""),
    md(r"""
`brentq` needs a bracket [a, b] on which the function changes sign - here from just above the saturation
temperature to the top of the steam tables. Most of the "inverse" functions of this course (turbine outlets,
flash temperatures, equilibrium conversions) are exactly this pattern.

## Diagrams in one loop
"""),
    code(r"""Ts = np.linspace(280, 647, 150)
sats = [steam.saturated(T=T) for T in Ts]
plt.plot([s["liquid"]["s"] / 1e3 for s in sats], Ts - 273.15, "b", label="saturated liquid")
plt.plot([s["vapour"]["s"] / 1e3 for s in sats], Ts - 273.15, "r", label="saturated vapour")
for p in (1e5, 1e6, 1e7):
    T_line = np.linspace(280, 900, 100)
    s_line = [steam.state(T=T, p=p)["s"] / 1e3 if T > steam.Tsat(p) else steam.state(T=T, p=p)["s"] / 1e3 for T in T_line]
    plt.plot(s_line, T_line - 273.15, "k--", lw=0.8)
    plt.text(s_line[-1] + 0.1, T_line[-1] - 273.15, f"{p/1e5:g} bar", fontsize=8)
plt.xlabel("s (kJ/(kg K))"); plt.ylabel("T (degC)"); plt.title("T-s diagram of water"); plt.legend(); plt.show()"""),
    md(r"""
The saturation dome and three isobars - the diagram every steam-cycle discussion is drawn on (notebook 02).

## Exercises
1. Draw the p-v diagram of water (log axes) with the saturation dome and the isotherms 100, 300, 500 and 700 degC.
2. Interpolate the ammonia saturation table by hand at -12 degC (h_f, h_g, p) and compare with `FluidTable`.
"""),
]

NOTEBOOKS["01_first_law_and_energy_balances"] = [
    md(r"""
# 01 · The first law and energy balances

Compressors, turbines, heat exchangers, nozzles and valves: the equipment of a chemical plant or a power
station is analysed with one equation - the steady-flow energy balance - and property data. This notebook
applies it to real gases and steam, with heat capacities that depend on temperature.
"""),
    setup_cell(["from scipy import integrate", "from engthermo import idealgas, steam", "from engthermo.constants import R"]),
    md(r"""
## The steady-flow energy balance
For a device with one inlet and one outlet, per unit mass or mole of fluid,
$$q - w_s = \Delta h + \Delta(\tfrac12 v^2) + g\,\Delta z,$$
with $q$ the heat added, $w_s$ the shaft work delivered and $\Delta h$ the change in enthalpy. Kinetic and potential
terms are usually negligible in chemical process equipment. Everything then reduces to enthalpy differences,
and enthalpy comes from heat capacities, tables or equations of state.

## Heat capacities depend on temperature
"""),
    code(r"""T = np.linspace(300, 1500, 200)
for gas in ("Ar", "N2", "CO2", "CH4", "H2O"):
    plt.plot(T, idealgas.cp(gas, T) / R, label=gas)
plt.xlabel("T (K)"); plt.ylabel("Cp / R"); plt.legend(); plt.title("Ideal-gas heat capacities"); plt.show()
for gas in ("N2", "CO2", "CH4"):
    print(f"{gas}: Cp at 300 K {idealgas.cp(gas, 300):.1f}, at 1000 K {idealgas.cp(gas, 1000):.1f} J/(mol K)")"""),
    md(r"""
Argon's heat capacity is exactly $5R/2$: a monatomic gas stores energy only in translation. Molecules also
rotate and, increasingly at high temperature, vibrate - so $C_p$ of methane almost doubles between room
temperature and 1000 K. "Constant $C_p$" is a convenient fiction that must be checked.

## Heating a gas stream
A furnace heats 100 mol/s of CO$_2$ from 400 to 1200 K. The heat duty is the enthalpy change:
"""),
    code(r"""dH = idealgas.enthalpy_change("CO2", 400.0, 1200.0)
Q = 100.0 * dH
print(f"enthalpy change {dH/1e3:.2f} kJ/mol -> heat duty {Q/1e6:.2f} MW")
print(f"with Cp fixed at its 400 K value: {100 * idealgas.cp('CO2', 400.0) * 800 / 1e6:.2f} MW ({idealgas.cp('CO2', 400.0) * 800 / dH - 1:+.0%})")
print(f"mean heat capacity for this interval: {idealgas.mean_cp_h('CO2', 400.0, 1200.0):.1f} J/(mol K)")"""),
    md(r"""
Using the room-temperature heat capacity would undersize the furnace by a fifth. The mean heat capacity
$\langle C_p\rangle_H$ (the integral divided by the temperature interval) is the number that makes the simple
formula $Q = n\langle C_p\rangle_H\Delta T$ exact.

## An air compressor: the work of a real gas
Compressed-air energy storage compresses air from 1 bar and 300 K to 10 bar. For a reversible adiabatic
compressor the outlet temperature follows from constant entropy:
"""),
    code(r"""T1, p1, p2 = 300.0, 1e5, 1e6
T2 = idealgas.isentropic_T("air", T1, p1, p2)
T2_gamma = idealgas.isentropic_T_const_gamma(T1, p1, p2, gamma=1.4)
w = idealgas.enthalpy_change("air", T1, T2)
print(f"outlet temperature: variable Cp {T2:.1f} K, constant gamma = 1.4 {T2_gamma:.1f} K")
print(f"compressor work {w/1e3:.3f} kJ/mol = {w / 0.028851 / 1e3:.1f} kJ/kg")
for eta in (0.85, 0.75):
    w_real = w / eta
    T2_real = idealgas.final_T_from_enthalpy("air", T1, w_real)
    print(f"isentropic efficiency {eta:.0%}: work {w_real/1e3:.3f} kJ/mol, outlet {T2_real:.1f} K")"""),
    md(r"""
The constant-$\gamma$ formula overestimates the outlet temperature, because it ignores the rise of $C_p$ with
temperature. Real compressors are less than ideal: the same pressure rise needs more work, and the extra work
ends up as heat in the gas - hotter outlet, and the reason compressed-air storage loses energy unless that
heat is stored too.

## Two-stage compression with intercooling
Compressing in two stages with cooling back to 300 K between them saves work:
"""),
    code(r"""def two_stage(p_mid):
    Ta = idealgas.isentropic_T("air", T1, p1, p_mid)
    Tb = idealgas.isentropic_T("air", T1, p_mid, p2)
    return idealgas.enthalpy_change("air", T1, Ta) + idealgas.enthalpy_change("air", T1, Tb)
p_mid = np.linspace(1.5e5, 8e5, 60)
w2 = np.array([two_stage(p) for p in p_mid])
plt.plot(p_mid / 1e5, w2 / 1e3); plt.axhline(w / 1e3, color="k", ls=":", label="single stage")
plt.xlabel("intermediate pressure (bar)"); plt.ylabel("work (kJ/mol)"); plt.legend(); plt.show()
best = p_mid[np.argmin(w2)]
print(f"optimum intermediate pressure {best/1e5:.2f} bar (theory: sqrt(p1 p2) = {np.sqrt(p1*p2)/1e5:.2f} bar); saving {1 - w2.min()/w:.1%}")"""),
    md(r"""
The optimum lies at the geometric mean of the pressures, where both stages do equal work, and it saves about
a sixth of the work. With more stages the process approaches isothermal compression, the minimum possible.

## Steam: a turbine and a throttling valve
"""),
    code(r"""inlet = steam.properties(T=773.15, p=8e6)
outlet_s = steam.state_ps(10e3, inlet["s"])                      # isentropic expansion to 10 kPa
w_s = inlet["h"] - outlet_s["h"]
print(f"turbine, 8 MPa / 500 degC -> 10 kPa: isentropic work {w_s/1e3:.1f} kJ/kg, exit quality {outlet_s['x']:.3f}")
valve_in = steam.properties(T=573.15, p=5e6)
valve_out = steam.state_ph(1e5, valve_in["h"])                   # throttling: h constant
print(f"throttling valve, 5 MPa / 300 degC -> 1 bar: outlet {valve_out['T'] - 273.15:.1f} degC, {valve_out['phase']}")"""),
    md(r"""
A throttling valve does no work and exchanges no heat, so the enthalpy is unchanged - but the temperature
falls (the Joule-Thomson effect; an ideal gas would show none). The expansion in the turbine, by contrast,
delivers work equal to the enthalpy drop.
"""),
]

NOTEBOOKS["02_steam_and_power_cycles"] = [
    md(r"""
# 02 · Steam and power cycles

Most of the world's electricity comes from steam turbines - in coal, gas, nuclear, biomass, geothermal and
solar-thermal plants alike. The Rankine cycle that runs them is a compact case study in applied
thermodynamics: property tables, energy balances, isentropic processes and the second law all appear in one
calculation. This notebook builds the cycle up from its simplest form and asks what limits its efficiency.
"""),
    setup_cell(["from engthermo import cycles, steam"]),
    md(r"""
## The steam tables as a function
`engthermo.steam` implements the IAPWS-IF97 industrial formulation - the same equations behind commercial
steam tables and power-plant software. Locate a few states:
"""),
    code(r"""for T_C, p_bar in ((150, 1), (150, 10), (350, 30), (700, 250)):
    st = steam.state(T=T_C + 273.15, p=p_bar * 1e5)
    print(f"{T_C:4d} degC, {p_bar:5.0f} bar: {st['phase']:<14} h = {st['h']/1e3:8.1f} kJ/kg  s = {st['s']/1e3:.4f} kJ/(kg K)")
wet = steam.state(p=1e5, x=0.5)
print(f"wet steam, 1 bar, x = 0.5: T = {wet['T'] - 273.15:.1f} degC, h = {wet['h']/1e3:.1f} kJ/kg")"""),
    md(r"""
## The simple ideal Rankine cycle
Pump (1 -> 2), boiler (2 -> 3), turbine (3 -> 4), condenser (4 -> 1). A classic textbook case: boiler at 3 MPa,
turbine inlet 350 degC, condenser at 75 kPa (Cengel & Boles, Example 10-1):
"""),
    code(r"""rk = cycles.rankine(p_boiler=3e6, p_condenser=75e3, T_inlet=350 + 273.15)
for k, st in rk["states"].items():
    print(f"state {k}: T = {st['T'] - 273.15:6.1f} degC, p = {st['p']/1e5:6.2f} bar, h = {st['h']/1e3:7.1f} kJ/kg, s = {st['s']/1e3:.4f}"
          + (f", x = {st['x']:.3f}" if "x" in st and st["x"] < 1 else ""))
print(f"turbine work {rk['w_turbine']/1e3:.1f}, pump work {rk['w_pump']/1e3:.2f}, heat in {rk['q_in']/1e3:.1f} kJ/kg")
print(f"thermal efficiency {rk['efficiency']:.1%}   (Carnot between the extreme temperatures: "
      f"{cycles.carnot_efficiency(623.15, rk['states']['1']['T']):.1%})")"""),
    code(r"""def plot_cycle(rk, label):
    keys = list(rk["states"])
    s = [rk["states"][k]["s"] / 1e3 for k in keys] + [rk["states"][keys[0]]["s"] / 1e3]
    T = [rk["states"][k]["T"] - 273.15 for k in keys] + [rk["states"][keys[0]]["T"] - 273.15]
    plt.plot(s, T, "o-", label=label)
Ts = np.linspace(280, 647, 120)
sats = [steam.saturated(T=T) for T in Ts]
plt.plot([x["liquid"]["s"] / 1e3 for x in sats], Ts - 273.15, "k", lw=0.8); plt.plot([x["vapour"]["s"] / 1e3 for x in sats], Ts - 273.15, "k", lw=0.8)
plot_cycle(rk, "simple Rankine")
plt.xlabel("s (kJ/(kg K))"); plt.ylabel("T (degC)"); plt.legend(); plt.show()"""),
    md(r"""
(The straight lines between states are schematic; the boiler path really follows the 3 MPa isobar.) The
efficiency is far below the Carnot limit because heat is added over a range of temperatures - most of it
during boiling at 234 degC, not at 350 degC.

## What raises the efficiency?
"""),
    code(r"""base = dict(p_boiler=3e6, p_condenser=75e3, T_inlet=623.15)
variants = {"base case": base,
            "condenser at 10 kPa": {**base, "p_condenser": 10e3},
            "superheat to 600 degC": {**base, "T_inlet": 873.15},
            "boiler at 15 MPa": {**base, "p_boiler": 15e6},
            "all three": {"p_boiler": 15e6, "p_condenser": 10e3, "T_inlet": 873.15}}
rows = [(name, cycles.rankine(**kw)["efficiency"], cycles.rankine(**kw)["exit_quality"], cycles.rankine(**kw)["w_net"] / 1e3) for name, kw in variants.items()]
print(pd.DataFrame(rows, columns=["variant", "efficiency", "turbine exit quality", "w_net (kJ/kg)"]).round(3).to_string(index=False))"""),
    md(r"""
Every change that raises the *average* temperature of heat addition or lowers the temperature of heat
rejection helps. But raising the boiler pressure lowers the quality at the turbine exit: liquid droplets
erode turbine blades, and a quality below about 0.88-0.90 is not acceptable. Reheating solves that:

## Reheat
"""),
    code(r"""rh = cycles.rankine(p_boiler=15e6, p_condenser=10e3, T_inlet=873.15, reheat=(4e6, 873.15))
no_rh = cycles.rankine(p_boiler=15e6, p_condenser=10e3, T_inlet=873.15)
print(f"without reheat: efficiency {no_rh['efficiency']:.1%}, exit quality {no_rh['exit_quality']:.3f}")
print(f"with reheat at 4 MPa to 600 degC: efficiency {rh['efficiency']:.1%}, exit quality {rh['exit_quality']:.3f}")
plot_cycle(rh, "reheat")
plt.plot([x["liquid"]["s"] / 1e3 for x in sats], Ts - 273.15, "k", lw=0.8); plt.plot([x["vapour"]["s"] / 1e3 for x in sats], Ts - 273.15, "k", lw=0.8)
plt.xlabel("s (kJ/(kg K))"); plt.ylabel("T (degC)"); plt.legend(); plt.show()"""),
    md(r"""
Reheat raises the exit quality to a safe value and, because the reheated steam is added at high temperature,
slightly improves the efficiency as well. (Cengel & Boles, Example 10-4: 45.0 % and 89.6 %.)

## Real turbines and pumps
"""),
    code(r"""for eta in (1.0, 0.9, 0.8):
    r = cycles.rankine(p_boiler=15e6, p_condenser=10e3, T_inlet=873.15, reheat=(4e6, 873.15), eta_turbine=eta, eta_pump=0.85)
    print(f"turbine isentropic efficiency {eta:.0%}: cycle efficiency {r['efficiency']:.1%}, exit quality {r['exit_quality']:.3f}")"""),
    md(r"""
Turbine losses cost efficiency but, as a side effect, raise the exit quality: the lost work reappears as
enthalpy in the exhaust steam.

## Sizing a plant
A 600 MW plant on the reheat cycle with 90 % turbine efficiency:
"""),
    code(r"""r = cycles.rankine(p_boiler=15e6, p_condenser=10e3, T_inlet=873.15, reheat=(4e6, 873.15), eta_turbine=0.9, eta_pump=0.85)
P = 600e6
m = P / r["w_net"]
print(f"steam mass flow {m:.0f} kg/s; boiler duty {m * r['q_in']/1e6:.0f} MW; condenser duty {m * r['q_out']/1e6:.0f} MW")
print(f"cooling water (10 K rise, cp 4.18 kJ/(kg K)): {m * r['q_out'] / (4180 * 10):.0f} kg/s")"""),
    md(r"""
More than half of the fuel's heat leaves through the condenser: cooling towers and the rivers and coasts
that supply cooling water are as much a part of a power station as its boiler.
"""),
]

NOTEBOOKS["03_entropy_refrigeration_and_heat_pumps"] = [
    md(r"""
# 03 · Entropy, refrigeration and heat pumps

The second law sets limits: no heat engine beats Carnot, no refrigerator moves heat uphill for free, and
every real process creates entropy - and destroys the ability to do work. This notebook quantifies those
limits and applies them to the machine that heats and cools most buildings today: the vapour-compression
cycle, run as a refrigerator or as a heat pump.
"""),
    setup_cell(["from engthermo import cycles, steam, tables"]),
    md(r"""
## Entropy generation in everyday processes
Throttling and heat exchange across a temperature difference are the workhorses of process plants - and
both generate entropy:
"""),
    code(r"""inlet = steam.properties(T=573.15, p=5e6)
outlet = steam.state_ph(1e5, inlet["h"])
print(f"throttling steam 5 MPa -> 1 bar: entropy generated {(outlet['s'] - inlet['s']):.1f} J/(kg K)")
T0 = 298.15
print(f"   exergy destroyed T0 * s_gen = {T0 * (outlet['s'] - inlet['s'])/1e3:.1f} kJ/kg of steam (out of a work potential of "
      f"{(inlet['h'] - steam.properties(T0, 1e5)['h'] - T0 * (inlet['s'] - steam.properties(T0, 1e5)['s']))/1e3:.0f} kJ/kg)")
Q, T_hot, T_cold = 1e6, 800.0, 400.0
print(f"1 MW of heat flowing from {T_hot:.0f} K to {T_cold:.0f} K: entropy generated {Q/T_cold - Q/T_hot:.0f} W/K, "
      f"i.e. {T0 * (Q/T_cold - Q/T_hot)/1e3:.0f} kW of work potential lost")"""),
    md(r"""
The exergy (work potential) destroyed is $T_0 s_{gen}$: the pressure let down in a valve could have driven a turbine;
heat transferred across a large temperature difference could have driven a heat engine. Exergy accounting
finds where a process wastes its potential.

## The limits: Carnot
"""),
    code(r"""for T_cold_C in (-20, 0, 5):
    for T_hot_C in (30, 45):
        print(f"cold {T_cold_C:4d} degC, hot {T_hot_C:3d} degC: max COP refrigeration {cycles.carnot_cop(T_cold_C + 273.15, T_hot_C + 273.15):.2f}, "
              f"heat pump {cycles.carnot_cop(T_cold_C + 273.15, T_hot_C + 273.15, heat_pump=True):.2f}")"""),
    md(r"""
The closer the two temperatures, the more heat can be moved per unit of work. A heat pump warming a house from
5 degC to 45 degC could deliver almost 8 kW of heat per kW of electricity - in the ideal limit.

## The vapour-compression cycle
Evaporator (4 -> 1, heat absorbed at low temperature), compressor (1 -> 2), condenser (2 -> 3, heat rejected),
expansion valve (3 -> 4). A domestic freezer: R134a evaporating at -20 degC, condensing at 40 degC.
"""),
    code(r"""vc = cycles.vapor_compression("R134a", T_evaporator=253.15, T_condenser=313.15)
for k, st in vc["states"].items():
    print(f"state {k}: {st['phase']:<20} T = {st['T'] - 273.15:6.1f} degC, p = {st['p']/1e5:5.2f} bar, h = {st['h']/1e3:6.1f} kJ/kg" + (f", x = {st['x']:.3f}" if "x" in st else ""))
print(f"compressor work {vc['w_compressor']/1e3:.1f} kJ/kg, cooling effect {vc['q_evaporator']/1e3:.1f} kJ/kg")
print(f"COP cooling {vc['COP_cooling']:.2f}, heating {vc['COP_heating']:.2f}   (Carnot: {cycles.carnot_cop(253.15, 313.15):.2f})")"""),
    code(r"""r134 = tables.FluidTable("R134a")
Ts = r134.T_sat
plt.semilogy([r134.saturated(T=T)["liquid"]["h"] / 1e3 for T in Ts], r134.p_sat / 1e5, "k", lw=0.8)
plt.semilogy([r134.saturated(T=T)["vapour"]["h"] / 1e3 for T in Ts], r134.p_sat / 1e5, "k", lw=0.8)
keys = ["1", "2", "3", "4", "1"]
plt.semilogy([vc["states"][k]["h"] / 1e3 for k in keys], [vc["states"][k]["p"] / 1e5 for k in keys], "o-")
plt.xlabel("h (kJ/kg)"); plt.ylabel("p (bar)"); plt.title("R134a, p-h diagram"); plt.show()"""),
    md(r"""
On the p-h diagram the cycle is a rectangle-like loop: the horizontal legs are the heat exchanged in the
evaporator and condenser, the compressor is the sloping leg, the valve a vertical drop. Even this ideal
cycle reaches only about three quarters of the Carnot COP: the throttling valve and the superheated
compressor discharge are irreversibilities built into the cycle itself.

## What changes the COP?
"""),
    code(r"""rows = []
for T_ev in (-30, -20, -10, 0):
    for T_cd in (30, 40, 50):
        v = cycles.vapor_compression("R134a", T_ev + 273.15, T_cd + 273.15)
        rows.append((T_ev, T_cd, v["COP_cooling"], v["p_high"] / v["p_low"], v["states"]["2"]["T"] - 273.15))
print(pd.DataFrame(rows, columns=["T_evap (degC)", "T_cond (degC)", "COP", "pressure ratio", "discharge T (degC)"]).round(2).to_string(index=False))"""),
    md(r"""
Every degree the evaporator can be warmer, or the condenser colder, pays off - which is why heat exchangers
are oversized and condensers are placed where cooling is coldest. Real compressors are not isentropic:
"""),
    code(r"""for eta in (1.0, 0.8, 0.65):
    v = cycles.vapor_compression("R134a", 253.15, 313.15, eta_compressor=eta, superheat=5.0, subcooling=5.0)
    print(f"compressor efficiency {eta:.0%} (5 K superheat and subcooling): COP {v['COP_cooling']:.2f}, discharge {v['states']['2']['T'] - 273.15:.0f} degC")"""),
    md(r"""
## Choosing a refrigerant
The same cycle with three refrigerants: R134a (a synthetic HFC being phased down for its global-warming
potential), ammonia (industrial refrigeration) and propane (R290, increasingly used in heat pumps):
"""),
    code(r"""rows = []
for fluid in ("R134a", "ammonia", "propane"):
    v = cycles.vapor_compression(fluid, 263.15, 313.15)
    rows.append((fluid, v["COP_cooling"], v["p_low"] / 1e5, v["p_high"] / 1e5, v["states"]["2"]["T"] - 273.15, v["q_evaporator"] / 1e3))
print(pd.DataFrame(rows, columns=["refrigerant", "COP", "p_evap (bar)", "p_cond (bar)", "discharge T (degC)", "cooling effect (kJ/kg)"]).round(2).to_string(index=False))"""),
    md(r"""
The thermodynamic COPs are similar - the cycle, not the fluid, sets most of the efficiency. The fluids differ
in practical ways: ammonia's large cooling effect per kilogram means small compressors, but its discharge
temperature is high and it is toxic; propane is flammable but efficient and benign to the atmosphere;
R134a is safe to handle but a potent greenhouse gas. Refrigerant selection is engineering trade-off, with
thermodynamics only one voice.

## Heat pumps for buildings
A heat pump moves heat from the outdoor air into the house. As the outdoor temperature falls, the
evaporator must run colder and the COP drops - just when the house needs most heat:
"""),
    code(r"""T_out = np.arange(-15, 16, 2.5)
cop = [cycles.vapor_compression("propane", T + 273.15 - 8, 273.15 + 45, eta_compressor=0.7)["COP_heating"] for T in T_out]
plt.plot(T_out, cop, "o-"); plt.xlabel("outdoor temperature (degC)"); plt.ylabel("heating COP")
plt.title("Air-source heat pump (propane, condenser 45 degC, 8 K approach, 70 % compressor)"); plt.show()
print(f"COP at +10 degC: {cop[-3]:.2f}; at -15 degC: {cop[0]:.2f}")"""),
    md(r"""
Even at -15 degC the heat pump delivers more than twice the heat of an electric resistance heater. Lowering
the condenser temperature - underfloor heating at 35 degC instead of radiators at 55 degC - raises the COP
further; that is why heat-pump buildings are designed around low-temperature heat emitters.
"""),
]

NOTEBOOKS["04_real_gases_and_equations_of_state"] = [
    md(r"""
# 04 · Real gases and equations of state

How much CO$_2$ fits into a pipeline at 100 bar? How much hydrogen into a 700-bar tank? The ideal-gas law is
wrong by tens of percent at such conditions. Equations of state (EOS) - the virial equation at moderate
pressure, and the cubic equations of van der Waals, Redlich-Kwong, Soave and Peng-Robinson - describe real
gases and, remarkably, liquids too, from just three numbers per substance: $T_c$, $p_c$ and $\omega$.
"""),
    setup_cell(["from engthermo import components, datasets, eos", "from engthermo.constants import R"]),
    md(r"""
## Compressibility: how far from ideal?
The compressibility factor $Z = pV/(RT)$ is 1 for an ideal gas. Reference data for CO$_2$ (from the
high-accuracy Span-Wagner equation) show what real gases do:
"""),
    code(r"""ref = pd.read_csv(datasets.path("real_gas_reference_densities.csv"), comment="#")
co2 = ref[ref.fluid == "CO2"]
for T, grp in co2.groupby("T_K"):
    plt.plot(grp.p_Pa / 1e5, grp.Z, "o-", ms=3, label=f"{T:.0f} K")
plt.axhline(1, color="k", lw=0.8); plt.xlabel("p (bar)"); plt.ylabel("Z"); plt.title("CO2: reference data"); plt.legend(); plt.show()"""),
    md(r"""
Below the critical temperature (304 K) the curves drop steeply where the gas condenses; near the critical
point $Z$ falls to about 0.3; at high temperature the gas stays closer to ideal. Attractive forces lower $Z$,
repulsion (molecular volume) raises it at very high pressure.

## The virial equation: the first correction
$Z = 1 + Bp/RT$, with the second virial coefficient $B(T)$ from the Pitzer-Abbott corresponding-states
correlation, needs only $T_c$, $p_c$ and $\omega$:
"""),
    code(r"""T = 320.0
sel = co2[co2.T_K == T]
pp = np.linspace(1e5, 60e5, 50)
Zv = [eos.virial("CO2", T, p)["Z"] for p in pp]
plt.plot(sel.p_Pa / 1e5, sel.Z, "ko", label="reference"); plt.plot(pp / 1e5, Zv, label="virial (truncated at B)")
plt.xlabel("p (bar)"); plt.ylabel("Z"); plt.title("CO2 at 320 K"); plt.legend(); plt.show()
for p in (10e5, 30e5, 60e5):
    z_ref = float(np.interp(p, sel.p_Pa, sel.Z))
    print(f"{p/1e5:3.0f} bar: reference Z = {z_ref:.3f}, virial {eos.virial('CO2', T, p)['Z']:.3f}, ideal 1.000")"""),
    md(r"""
The virial equation captures the first departure from ideality and fails as the gas approaches condensation.
Its rule of thumb: adequate up to about half the critical pressure for gases (worse near $T_c$).

## Cubic equations of state
Van der Waals (1873) added two corrections to the ideal-gas law - molecular volume $b$ and attraction $a$ -
giving an equation cubic in the volume. Its successors keep the form and refine $a(T)$:

| Equation | year | temperature dependence of $a$ |
|---|---|---|
| van der Waals | 1873 | none |
| Redlich-Kwong | 1949 | $T^{-1/2}$ |
| Soave-Redlich-Kwong | 1972 | $[1 + m(\omega)(1 - \sqrt{T_r})]^2$ |
| Peng-Robinson | 1976 | as Soave, with a different volume dependence (better liquid densities) |
"""),
    code(r"""pp = np.linspace(1e5, 200e5, 80)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
for T in (300.0, 400.0):
    sel = co2[co2.T_K == T]
    a1.plot(sel.p_Pa / 1e5, sel.Z, "ko")
    for name in ("vdw", "rk", "srk", "pr"):
        a1.plot(pp / 1e5, [eos.solve(name, "CO2", T, p, "stable")["Z"] for p in pp], label=f"{name.upper()} {T:.0f} K" if T == 300 else None, lw=1)
a1.set(xlabel="p (bar)", ylabel="Z", title="CO2: cubic equations vs reference"); a1.legend(fontsize=7)
rows = []
for name in ("vdw", "rk", "srk", "pr"):
    err = [abs(eos.solve(name, "CO2", T, p, "stable")["Z"] / z - 1) for T, p, z in zip(co2.T_K, co2.p_Pa, co2.Z)]
    rows.append((name.upper(), 100 * np.mean(err), 100 * np.max(err)))
a2.bar([r[0] for r in rows], [r[1] for r in rows]); a2.set(ylabel="mean |error| in Z (%)", title="over 50 reference states")
plt.show()
print(pd.DataFrame(rows, columns=["equation", "mean error %", "max error %"]).round(1).to_string(index=False))"""),
    md(r"""
The refinement of $a(T)$ matters. Peng-Robinson averages under 2 % error over the whole range, with the largest
errors (up to 9 %) in the dense, liquid-like states; Redlich-Kwong and SRK average 4-5 % and err by 20 % in
the liquid; van der Waals, historic and instructive, is not accurate enough for design. Three handbook
parameters take Peng-Robinson a long way - which is why it is the workhorse of process simulators.

## Three roots: the liquid, the vapour and the one in between
Below $T_c$ the cubic can have three real roots in $V$. The smallest is the liquid, the largest the vapour, the
middle one is unstable. The p-V isotherm shows why:
"""),
    code(r"""T = 280.0
V = np.logspace(np.log10(4.5e-5), -2.5, 400)
p_iso = np.array([eos.pressure("pr", "CO2", T, v) for v in V])
psat = 41.6e5                                        # saturation pressure at 280 K (notebook 06 computes it)
plt.semilogx(V * 1e6, p_iso / 1e5); plt.axhline(psat / 1e5, color="r", ls="--", label="saturation pressure")
plt.ylim(0, 100); plt.xlabel("V (cm3/mol)"); plt.ylabel("p (bar)"); plt.title("PR isotherm of CO2 at 280 K"); plt.legend(); plt.show()
r = eos.solve("pr", "CO2", T, psat)
print("roots at the saturation pressure, Z =", np.round(r["roots"], 4), " -> V =", np.round(r["roots"] * R * T / psat * 1e6, 1), "cm3/mol")"""),
    md(r"""
The isotherm has a loop (a "van der Waals loop"): its middle part, where pressure rises with volume, is
mechanically unstable and never observed. At the saturation pressure the outer roots are the coexisting liquid
and vapour; the horizontal line replacing the loop is the phase change.

## Application: how much fits in the tank?
"""),
    code(r"""h2 = ref[ref.fluid == "hydrogen"]
sel = h2[h2.T_K == 300.0]
for p in (350e5, 700e5):
    z_ref = float(np.interp(p, sel.p_Pa, sel.Z))
    z_pr = eos.solve("pr", "H2", 300.0, p)["Z"]
    M = components.get("H2").M
    rho_ideal, rho_ref = p * M / (R * 300.0), p * M / (z_ref * R * 300.0)
    print(f"hydrogen at {p/1e5:.0f} bar, 300 K: Z = {z_ref:.3f} (PR: {z_pr:.3f}); density {rho_ref:.1f} kg/m3, ideal gas would predict "
          f"{rho_ideal:.1f} kg/m3 ({rho_ideal/rho_ref - 1:+.0%})")
sel = co2[co2.T_K == 300.0]
z = float(np.interp(100e5, sel.p_Pa, sel.Z))
print(f"CO2 at 100 bar, 300 K: Z = {z:.3f}; density {100e5 * 0.04401 / (z * R * 300):.0f} kg/m3 - a dense liquid, "
      f"{1/z:.0f} times the ideal-gas value")"""),
    md(r"""
For hydrogen the deviation goes the other way: at 700 bar $Z$ is about 1.5, so a tank holds a third *less*
hydrogen than the ideal-gas law promises - a central number in the design of fuel-cell vehicles. For CO$_2$ at
100 bar, the "gas" is a liquid of about 800 kg/m$^3$: pipelines and storage for carbon capture run in this dense
phase, and their design starts with an equation of state.
"""),
]

NOTEBOOKS["05_departure_functions_and_fugacity"] = [
    md(r"""
# 05 · Departure functions and fugacity

Ideal-gas enthalpies and entropies (notebook 01) are easy; real-gas ones follow by adding a **departure
function** - the difference between the real fluid and the ideal gas at the same $T$ and $p$ - which an equation of
state supplies. The same equation supplies the **fugacity**, the "effective pressure" that replaces $p$ in
equilibrium calculations. Both are needed for compressors, expanders and heat exchangers handling real gases.
"""),
    setup_cell(["from scipy import optimize", "from engthermo import eos, idealgas", "from engthermo.constants import R"]),
    md(r"""
## Departure functions from the Peng-Robinson equation
$H = H^{ig}(T) + H^R(T, p)$ and $S = S^{ig}(T, p) + S^R(T, p)$. The departures vanish as $p \to 0$:
"""),
    code(r"""T = 320.0
pp = np.linspace(1e5, 100e5, 60)
HR = [eos.solve("pr", "CO2", T, p, "stable")["H_R"] for p in pp]
SR = [eos.solve("pr", "CO2", T, p, "stable")["S_R"] for p in pp]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
a1.plot(pp / 1e5, np.array(HR) / 1e3); a1.set(xlabel="p (bar)", ylabel="H_R (kJ/mol)", title="CO2 at 320 K: enthalpy departure")
a2.plot(pp / 1e5, SR); a2.set(xlabel="p (bar)", ylabel="S_R (J/(mol K))", title="entropy departure"); plt.show()
r = eos.solve("pr", "CO2", T, 50e5)
print(f"at 50 bar: H_R = {r['H_R']/1e3:.2f} kJ/mol (compare with the ideal-gas enthalpy change for 50 K: "
      f"{idealgas.enthalpy_change('CO2', T, T + 50)/1e3:.2f} kJ/mol), S_R = {r['S_R']:.2f} J/(mol K)")"""),
    md(r"""
Both departures are negative: attractive forces lower the enthalpy (energy is needed to pull the molecules
apart) and the entropy (the molecules are more constrained). At 50 bar the enthalpy departure of CO$_2$ is
comparable to the effect of a 50 K temperature change - not a small correction.

## A real-gas compressor
The last stage of a CO$_2$ pipeline compressor takes the gas from 40 bar and 300 K to 80 bar - close to the
critical point (304 K, 74 bar), where real-gas effects are largest. An isentropic compressor keeps
$S = S^{ig} + S^R$ constant:
"""),
    code(r"""T1, p1, p2 = 300.0, 40e5, 80e5
s1 = idealgas.entropy("CO2", T1, p1) + eos.solve("pr", "CO2", T1, p1)["S_R"]
f = lambda T: idealgas.entropy("CO2", T, p2) + eos.solve("pr", "CO2", T, p2, "stable")["S_R"] - s1
T2 = optimize.brentq(f, T1, 900.0)
h1 = idealgas.enthalpy("CO2", T1) + eos.solve("pr", "CO2", T1, p1)["H_R"]
h2 = idealgas.enthalpy("CO2", T2) + eos.solve("pr", "CO2", T2, p2, "stable")["H_R"]
T2_ig = idealgas.isentropic_T("CO2", T1, p1, p2)
w_ig = idealgas.enthalpy_change("CO2", T1, T2_ig)
print(f"real gas (PR): outlet {T2:.1f} K, work {(h2 - h1)/1e3:.3f} kJ/mol")
print(f"ideal gas:     outlet {T2_ig:.1f} K, work {w_ig/1e3:.3f} kJ/mol  ({w_ig/(h2 - h1) - 1:+.1%})")"""),
    md(r"""
The ideal-gas model overestimates the work by a third: the attractive forces that lower $Z$ also lower the
enthalpy of the compressed gas, so less shaft work is needed. Yet the real gas leaves *hotter*: the entropy
departure is more negative at 80 bar than at 40 bar, so the ideal-gas part of the entropy - and with it the
temperature - must rise more to keep the total entropy constant. The lower work comes from the enthalpy
departure at the outlet, not from a lower temperature. For a capture plant compressing thousands of tonnes a
day, a third of the last-stage motor is a very real number.

## Fugacity: the effective pressure
The fugacity $f = \phi p$ replaces the pressure in every equilibrium criterion for real fluids. It equals $p$
for an ideal gas; the fugacity coefficient $\phi$ comes from the equation of state:
"""),
    code(r"""pp = np.linspace(1e5, 80e5, 80)
phi = [eos.solve("pr", "CO2", T1, p, "stable")["phi"] for p in pp]
plt.plot(pp / 1e5, pp / 1e5, "k--", label="ideal gas (f = p)"); plt.plot(pp / 1e5, np.array(phi) * pp / 1e5, label="CO2 at 320 K (PR)")
plt.xlabel("p (bar)"); plt.ylabel("fugacity (bar)"); plt.legend(); plt.show()
for p in (10e5, 40e5, 80e5):
    print(f"{p/1e5:3.0f} bar: phi = {eos.solve('pr', 'CO2', T1, p, 'stable')['phi']:.3f}, f = {eos.solve('pr', 'CO2', T1, p, 'stable')['phi'] * p/1e5:.1f} bar")"""),
    md(r"""
At 80 bar the fugacity is only 54 bar: the molecules "feel" two thirds of the pressure, because their mutual
attraction does part of the confining. Fugacity has a direct physical meaning too - the work of isothermal,
reversible compression is $RT\ln(f_2/f_1)$:
"""),
    code(r"""from scipy import integrate
f1, f2 = eos.solve("pr", "CO2", T1, 1e5)["phi"] * 1e5, eos.solve("pr", "CO2", T1, 60e5)["phi"] * 60e5
w_iso = R * T1 * np.log(f2 / f1)
w_num = integrate.quad(lambda p: eos.solve("pr", "CO2", T1, p, "stable")["V"], 1e5, 60e5, limit=200)[0]
print(f"isothermal compression work 1 -> 60 bar: RT ln(f2/f1) = {w_iso/1e3:.3f} kJ/mol; integral V dp = {w_num/1e3:.3f} kJ/mol; "
      f"ideal gas RT ln(p2/p1) = {R*T1*np.log(60)/1e3:.3f} kJ/mol")"""),
    md(r"""
The three routes agree where they should: fugacity is defined precisely so that $RT\,d\ln f = V\,dp$ at
constant temperature. Isothermal compression needs far less work than adiabatic - which is why multistage
compressors with intercoolers (notebook 01) approach it.
"""),
]

NOTEBOOKS["06_vapour_pressure_and_phase_change"] = [
    md(r"""
# 06 · Vapour pressure and phase change

The vapour pressure of a liquid decides whether a storage tank is at atmospheric pressure or at 15 bar, how
much of a solvent evaporates, and at what temperature a mixture boils. This notebook describes it three
ways - Clausius-Clapeyron, the Antoine equation, and an equation of state - and shows what each can do.
"""),
    setup_cell(["from engthermo import components, eos, vapor", "from engthermo.constants import R"]),
    md(r"""
## Clausius-Clapeyron: vapour pressure from the heat of vaporisation
Along the saturation line $dp/dT = \Delta H_{vap}/(T\Delta V)$. With an ideal vapour and a negligible liquid volume,
$d\ln p/dT = \Delta H_{vap}/(RT^2)$, and for constant $\Delta H_{vap}$, $\ln p$ is a straight line in $1/T$:
"""),
    code(r"""T_ref, p_ref = 373.124, 101325.0                          # water: normal boiling point
dH = 40.65e3                                              # J/mol at 100 degC
Ts = np.linspace(280, 470, 60)
p_cc = vapor.clausius_clapeyron(Ts, p_ref, T_ref, dH)
p_ant = vapor.antoine_psat("water", Ts)
plt.semilogy(1000 / Ts, p_cc / 1e5, label="Clausius-Clapeyron, constant dH"); plt.semilogy(1000 / Ts, p_ant / 1e5, "--", label="Antoine (fitted to steam tables)")
plt.xlabel("1000 / T (1/K)"); plt.ylabel("p_sat (bar)"); plt.title("Water"); plt.legend(); plt.show()
for T in (300.0, 373.124, 450.0):
    print(f"{T:7.2f} K: Clausius-Clapeyron {vapor.clausius_clapeyron(T, p_ref, T_ref, dH)/1e3:8.2f} kPa, Antoine {vapor.antoine_psat('water', T)/1e3:8.2f} kPa")"""),
    md(r"""
Above the boiling point Clausius-Clapeyron with one reference point and one heat of vaporisation stays
within 2 %; at 27 degC it is 18 % too high, because the heat of vaporisation of water is larger at low
temperature (44 kJ/mol at 25 degC against 40.7 at 100 degC), so the true line is steeper there. The slope of
$\ln p$ against $1/T$ *is* the heat of vaporisation: a way to measure it from vapour-pressure data alone.

## The Antoine equation
$\ln p = A - B/(T + C)$: three constants per substance, fitted to data. `engthermo`'s constants are fitted to
CoolProp's reference vapour pressures between 1 kPa and 15 bar, with the range and the fit error stored:
"""),
    code(r"""for name in ("water", "ethanol", "benzene", "propane", "ammonia"):
    c = components.get(name)
    print(f"{name:<8}: A = {c.antoine[0]:.4f}, B = {c.antoine[1]:8.2f}, C = {c.antoine[2]:8.3f}; valid {c.antoine_range[0]:.0f}-{c.antoine_range[1]:.0f} K, "
          f"max error {c.antoine_fit_error:.1%}; normal boiling point {vapor.antoine_Tsat(name, 101325.0) - 273.15:6.1f} degC")"""),
    md(r"""
Antoine is the everyday tool for vapour pressures - but only inside its fitted range. Extrapolating an Antoine
equation towards the critical point or far below 1 kPa can be wrong by large factors.

## Saturation from an equation of state
A cubic equation contains the phase change: at the saturation pressure the liquid and vapour roots have
equal fugacity. Solving that condition gives the vapour pressure, the coexisting volumes and, from the
departure functions, the heat of vaporisation:
"""),
    code(r"""rows = []
for T in (230.0, 260.0, 280.0, 295.0):
    s = vapor.eos_psat("pr", "CO2", T)
    rows.append((T, s["p_sat"] / 1e5, s["V_liquid"] * 1e6, s["V_vapor"] * 1e6, s["dH_vap"] / 1e3, vapor.antoine_psat("CO2", T) / 1e5))
print(pd.DataFrame(rows, columns=["T (K)", "p_sat PR (bar)", "V_liq (cm3/mol)", "V_vap (cm3/mol)", "dH_vap (kJ/mol)", "p_sat Antoine (bar)"]).round(2).to_string(index=False))"""),
    md(r"""
The Peng-Robinson vapour pressures agree with the Antoine values within a few percent - remarkable for an
equation whose only inputs are $T_c$, $p_c$ and $\omega$ (the acentric factor was in fact *defined* from the vapour
pressure at $T_r$ = 0.7, which is why cubic equations reproduce it). The heat of vaporisation falls towards the
critical point, where liquid and vapour become identical and it vanishes.

## Application: storage pressure of LPG
Propane is stored as a liquid under its own vapour pressure. The tank must withstand the pressure at the
hottest temperature the tank can reach:
"""),
    code(r"""for T_C in (-42, 0, 20, 40, 55):
    T = T_C + 273.15
    print(f"{T_C:4d} degC: vapour pressure of propane {vapor.antoine_psat('propane', T)/1e5:5.2f} bar (PR: {vapor.eos_psat('pr', 'propane', T)['p_sat']/1e5:5.2f} bar)")
print(f"molar enthalpy of vaporisation at 20 degC: {vapor.eos_psat('pr', 'propane', 293.15)['dH_vap']/1e3:.1f} kJ/mol")"""),
    md(r"""
At -42 degC propane boils at atmospheric pressure - refrigerated storage. At ambient temperature the tank sees
8-10 bar, and in the sun up to 20 bar: that is why LPG cylinders are thick-walled and why their design
temperature is set well above the ambient. Every substance's storage strategy - refrigerated, pressurised, or
both - is read off its vapour-pressure curve.
"""),
]

NOTEBOOKS["07_mixture_fundamentals"] = [
    md(r"""
# 07 · Mixture fundamentals

Mix 50 mL of ethanol with 50 mL of water and you get about 96 mL. Properties of mixtures are not simply the
sum of the parts - and the concepts that describe how they differ, **partial molar properties** and
**excess properties**, are the foundation of everything that follows: phase equilibria, activity coefficients,
separation processes.
"""),
    setup_cell(["from engthermo import datasets, mixtures", "from engthermo.constants import R"]),
    md(r"""
## The volume of ethanol-water mixtures
"""),
    code(r"""d = pd.read_csv(datasets.path("ethanol_water_volume.csv"), comment="#")
x1, V = d.x_ethanol.to_numpy(), d.V_cm3_mol.to_numpy()
V1, V2 = V[-1], V[0]                                                 # pure ethanol and pure water
V_ideal = x1 * V1 + (1 - x1) * V2
plt.plot(x1, V, "o-", label="measured (representative)"); plt.plot(x1, V_ideal, "--", label="ideal mixing")
plt.xlabel("x ethanol"); plt.ylabel("molar volume (cm3/mol)"); plt.legend(); plt.show()
i = np.argmin(V - V_ideal)
print(f"largest contraction at x = {x1[i]:.2f}: {V[i] - V_ideal[i]:.2f} cm3/mol ({(V[i] - V_ideal[i])/V_ideal[i]:.1%})")"""),
    md(r"""
The mixture is smaller than the sum of its parts: water and ethanol molecules pack more tightly together
than apart (hydrogen bonding). The difference from ideal mixing is the **excess volume** $V^E$.

## Partial molar volumes
The volume of a mixture is $V = x_1\bar V_1 + x_2\bar V_2$, where $\bar V_i$ - the partial molar volume - is the volume
one mole of $i$ effectively occupies *in the mixture*. It is found from the slope of $V(x_1)$ by the
tangent-intercept construction:
"""),
    code(r"""poly = np.polyfit(x1, V, 6)                                         # smooth representation of the data
Vf = lambda x: np.polyval(poly, x)
xx = np.linspace(0, 1, 100)
Vbar1 = np.array([mixtures.partial_molar_binary(Vf, x)[0] for x in xx])
Vbar2 = np.array([mixtures.partial_molar_binary(Vf, x)[1] for x in xx])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
a1.plot(x1, V, "o", ms=4); a1.plot(xx, Vf(xx), "k", lw=1)
x0 = 0.3
b1, b2 = mixtures.partial_molar_binary(Vf, x0)
a1.plot([0, 1], [b2, b1], "r--", label=f"tangent at x = {x0}"); a1.plot([0, 1], [b2, b1], "ro")
a1.set(xlabel="x ethanol", ylabel="V (cm3/mol)", title="tangent-intercept construction"); a1.legend()
a2.plot(xx, Vbar1, label="ethanol"); a2.plot(xx, Vbar2, label="water")
a2.axhline(V1, color="C0", ls=":", lw=1); a2.axhline(V2, color="C1", ls=":", lw=1)
a2.set(xlabel="x ethanol", ylabel="partial molar volume (cm3/mol)"); a2.legend(); plt.show()
print(f"at x = 0.3: V1_bar = {b1:.2f} (pure {V1:.2f}), V2_bar = {b2:.2f} (pure {V2:.2f}) cm3/mol; check x1 V1_bar + x2 V2_bar = {x0*b1 + (1-x0)*b2:.4f} vs V = {Vf(x0):.4f}")"""),
    md(r"""
The tangent's intercepts at $x_1$ = 0 and 1 are the partial molar volumes. At infinite dilution a molecule of
ethanol in water occupies markedly less volume than in pure ethanol - it slips into the water structure.
Partial molar quantities are what mixture calculations actually use: the volume of a tank, the enthalpy of
a stream, and above all the chemical potential (the partial molar Gibbs energy).

## Excess properties and ideal mixing
For an ideal solution $V^E = 0$ and $H^E = 0$, yet mixing still changes the Gibbs energy, through the entropy of
mixing $-R\sum x_i\ln x_i$:
"""),
    code(r"""xx = np.linspace(1e-6, 1 - 1e-6, 200)
S_mix = -R * (xx * np.log(xx) + (1 - xx) * np.log(1 - xx))
G_mix = -298.15 * S_mix
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
a1.plot(xx, S_mix); a1.set(xlabel="x1", ylabel="ideal entropy of mixing (J/(mol K))")
a2.plot(xx, G_mix / 1e3); a2.set(xlabel="x1", ylabel="ideal Gibbs energy of mixing at 25 degC (kJ/mol)"); plt.show()
print(f"maximum entropy of mixing R ln 2 = {R*np.log(2):.3f} J/(mol K) at x = 0.5; Gibbs energy of mixing {-298.15*R*np.log(2)/1e3:.3f} kJ/mol")"""),
    md(r"""
Mixing is spontaneous even without any energetic effect because the Gibbs energy of mixing is negative;
separating a mixture back into its pure components costs at least that much work - the thermodynamic minimum
behind every distillation column and membrane (notebook 09 makes this quantitative).

## The Gibbs-Duhem equation
The partial molar properties of a mixture are not independent: $\sum x_i\,d\bar M_i = 0$ at constant $T$ and $p$. A
consequence used throughout the course: if one component's partial molar property is known over the whole
composition range, the other's follows.
"""),
    code(r"""h = 1e-4
gd = [x * (mixtures.partial_molar_binary(Vf, x + h)[0] - mixtures.partial_molar_binary(Vf, x - h)[0]) / (2 * h)
      + (1 - x) * (mixtures.partial_molar_binary(Vf, x + h)[1] - mixtures.partial_molar_binary(Vf, x - h)[1]) / (2 * h) for x in (0.2, 0.5, 0.8)]
print("Gibbs-Duhem residual x1 dV1_bar/dx1 + x2 dV2_bar/dx1 at x = 0.2, 0.5, 0.8:", np.round(gd, 6))"""),
    md(r"""
The residual is zero (to numerical precision) because both partial molar volumes were derived consistently from
one function $V(x)$. For activity coefficients *measured* separately, the Gibbs-Duhem equation becomes a test of
the data's consistency (notebook 10).
"""),
]

NOTEBOOKS["08_raoults_law_and_flash"] = [
    md(r"""
# 08 · Raoult's law and flash calculations

The simplest model of a vapour-liquid mixture - Raoult's law, $y_i p = x_i p_i^{sat}$ - already answers the
central questions of distillation and flash separation: at what temperature does a mixture boil, what is
the composition of the first vapour, and how does a partly vaporised feed split between the phases?
"""),
    setup_cell(["from engthermo import vapor, vle"]),
    md(r"""
## Bubble and dew points
A benzene-toluene mixture (the textbook near-ideal pair) at atmospheric pressure:
"""),
    code(r"""p = 101325.0
comps = ["benzene", "toluene"]
for x_b in (0.2, 0.5, 0.8):
    bub = vle.bubble_T(comps, [x_b, 1 - x_b], p)
    dew = vle.dew_T(comps, [x_b, 1 - x_b], p)
    print(f"x_benzene = {x_b}: bubble point {bub['T']-273.15:.1f} degC (first vapour y = {bub['y'][0]:.3f}); "
          f"dew point of a vapour of the same composition {dew['T']-273.15:.1f} degC (first liquid x = {dew['x'][0]:.3f})")
print(f"pure components: benzene boils at {vapor.antoine_Tsat('benzene', p)-273.15:.1f} degC, toluene at {vapor.antoine_Tsat('toluene', p)-273.15:.1f} degC")"""),
    md(r"""
The first vapour is richer in benzene, the more volatile component - the effect distillation exploits. A
liquid starts boiling at its bubble point and, if kept at that pressure, is completely vaporised at the dew
point of the same composition; between the two it is a two-phase mixture.

## The T-x-y diagram
"""),
    code(r"""d = vle.Txy(comps, p, n=41)
plt.plot(d["x1"], d["T"] - 273.15, label="bubble curve (liquid)"); plt.plot(d["y1"], d["T"] - 273.15, label="dew curve (vapour)")
plt.fill_betweenx(d["T"] - 273.15, d["x1"], d["y1"], alpha=0.1)
plt.xlabel("mole fraction benzene"); plt.ylabel("T (degC)"); plt.title("benzene-toluene at 1 atm (Raoult's law)"); plt.legend(); plt.show()
Tb = vle.bubble_T(comps, [0.5, 0.5], p)["T"]
print(f"tie line at {Tb-273.15:.1f} degC: liquid x = 0.500, vapour y = {vle.bubble_T(comps, [0.5, 0.5], p)['y'][0]:.3f}")"""),
    md(r"""
Every horizontal line inside the lens connects a liquid (on the bubble curve) with the vapour in equilibrium
with it (on the dew curve): a **tie line**. The width of the lens is the driving force for separation.

## Relative volatility
"""),
    code(r"""alpha = vapor.antoine_psat("benzene", Tb) / vapor.antoine_psat("toluene", Tb)
print(f"relative volatility benzene/toluene at {Tb-273.15:.0f} degC: {alpha:.2f}")
x = np.linspace(0, 1, 100)
plt.plot(x, alpha * x / (1 + (alpha - 1) * x), label=f"y = a x / (1 + (a - 1) x), a = {alpha:.2f}"); plt.plot(x, x, "k--", lw=0.8)
plt.plot(d["x1"], d["y1"], ".", ms=4, label="from the bubble-point calculation")
plt.xlabel("x benzene"); plt.ylabel("y benzene"); plt.legend(); plt.show()"""),
    md(r"""
For an ideal mixture the relative volatility $\alpha = p_1^{sat}/p_2^{sat}$ is nearly constant, and the x-y curve
follows from it alone - the basis of the McCabe-Thiele method for distillation columns. A relative volatility
near 1 means an expensive separation; an azeotrope (notebook 09) means an impossible one.

## The flash drum
A feed is heated and let down to a lower pressure; part of it vaporises. The Rachford-Rice equation gives the
vapour fraction and both compositions:
"""),
    code(r"""feed = ["propane", "n-butane", "n-pentane"]
z = [0.3, 0.4, 0.3]
T, p = 320.0, 5e5
fl = vle.flash(feed, z, T, p)
print(f"feed {z} at {T-273.15:.0f} degC and {p/1e5:.0f} bar: vapour fraction {fl['V']:.3f}")
print(pd.DataFrame({"component": feed, "z": z, "x (liquid)": fl["x"], "y (vapour)": fl["y"], "K = y/x": fl["K"]}).round(3).to_string(index=False))
print(f"material balance check z1 = V y1 + (1 - V) x1 = {fl['V']*fl['y'][0] + (1 - fl['V'])*fl['x'][0]:.3f}")"""),
    md(r"""
The vapour concentrates the light propane, the liquid the heavy pentane; the K-values $y_i/x_i$ show each
component's preference. A single flash is a crude separation - a distillation column is a stack of them. How
the split moves with temperature:
"""),
    code(r"""Ts = np.linspace(295, 345, 51)
V = [vle.flash(feed, z, T, p)["V"] for T in Ts]
plt.plot(Ts - 273.15, V); plt.xlabel("T (degC)"); plt.ylabel("vapour fraction"); plt.title("flash of the C3/C4/C5 feed at 5 bar"); plt.show()
Tb = vle.bubble_T(feed, z, p)["T"]; Td = vle.dew_T(feed, z, p)["T"]
print(f"bubble point {Tb-273.15:.1f} degC, dew point {Td-273.15:.1f} degC: the feed is two-phase between them")"""),
    md(r"""
Raoult's law works for mixtures of similar molecules (hydrocarbons of the same family). For unlike molecules -
ethanol and water, acetone and chloroform - the liquid is not ideal, the vapour pressure curves bend, and
azeotropes appear. Notebook 09 adds the activity coefficients that describe them.
"""),
]

NOTEBOOKS["09_activity_coefficient_models"] = [
    md(r"""
# 09 · Activity-coefficient models

Ethanol and water do not obey Raoult's law: the mixture boils at a lower temperature than either predicts,
and no distillation column can take ethanol beyond 89 mol % - the azeotrope. Activity coefficients describe
such non-ideal liquids; this notebook fits the classical models (Margules, van Laar, Wilson, NRTL) to data,
checks the data for thermodynamic consistency, and locates the azeotrope that shapes the bioethanol industry.
"""),
    setup_cell(["from engthermo import activity, datasets, mixtures, vapor, vle"]),
    md("## The data, and Raoult's law's failure"),
    code(r"""d = pd.read_csv(datasets.path("ethanol_water_vle_1atm.csv"), comment="#")
comps, p = ["ethanol", "water"], 101325.0
raoult = vle.Txy(comps, p, n=41)
plt.plot(d.x_ethanol, d.T_C, "o", label="data (liquid)"); plt.plot(d.y_ethanol, d.T_C, "s", label="data (vapour)")
plt.plot(raoult["x1"], raoult["T"] - 273.15, "k--", lw=1, label="Raoult's law"); plt.plot(raoult["y1"], raoult["T"] - 273.15, "k--", lw=1)
plt.xlabel("mole fraction ethanol"); plt.ylabel("T (degC)"); plt.title("ethanol-water at 1 atm"); plt.legend(); plt.show()"""),
    md(r"""
Raoult's law predicts a smooth lens between 100 and 78.3 degC; the real mixture boils *below* both pure
components near x = 0.9 - a **minimum-boiling azeotrope**, where liquid and vapour have the same composition.
Positive deviations from Raoult's law (activity coefficients above 1) cause it: water and ethanol molecules
prefer their own kind.

## Activity coefficients from the data
Modified Raoult's law, $y_i p = x_i\gamma_i p_i^{sat}$, turns each data point into activity coefficients:
"""),
    code(r"""inner = d[(d.x_ethanol > 0) & (d.x_ethanol < 1)]
T = inner.T_C.to_numpy() + 273.15
g1 = inner.y_ethanol * p / (inner.x_ethanol * vapor.antoine_psat("ethanol", T))
g2 = (1 - inner.y_ethanol) * p / ((1 - inner.x_ethanol) * vapor.antoine_psat("water", T))
plt.semilogy(inner.x_ethanol, g1, "o", label="ethanol"); plt.semilogy(inner.x_ethanol, g2, "s", label="water")
plt.axhline(1, color="k", lw=0.8); plt.xlabel("x ethanol"); plt.ylabel("activity coefficient"); plt.legend(); plt.show()
print(f"ethanol at infinite dilution in water: gamma about {g1.iloc[0]:.1f}; water in ethanol: about {g2.iloc[-1]:.1f}")"""),
    md(r"""
Both coefficients exceed 1 and grow towards infinite dilution: a molecule surrounded by the other species is
"uncomfortable" and escapes more readily than Raoult's law expects. The excess Gibbs energy
$G^E/RT = \sum x_i\ln\gamma_i$ summarises the non-ideality in one curve.

## Fitting the models
"""),
    code(r"""fits = {m: activity.fit(m, comps, inner.x_ethanol, T, p, y1=inner.y_ethanol) for m in ("margules", "van_laar", "wilson", "nrtl")}
xx = np.linspace(0.001, 0.999, 200)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
a1.plot(inner.x_ethanol, inner.x_ethanol * np.log(g1) + (1 - inner.x_ethanol) * np.log(g2), "ko", label="from data")
for m, f in fits.items():
    a1.plot(xx, [activity.excess_gibbs(f["gamma"], [x, 1 - x], 351.0) for x in xx], label=m)
    txy = vle.Txy(comps, p, f["gamma"], n=41)
    a2.plot(txy["x1"], txy["T"] - 273.15, lw=1, label=m); a2.plot(txy["y1"], txy["T"] - 273.15, lw=1, color=a2.lines[-1].get_color())
a1.set(xlabel="x ethanol", ylabel="G^E / RT", title="excess Gibbs energy"); a1.legend(fontsize=8)
a2.plot(d.x_ethanol, d.T_C, "ko", ms=4); a2.plot(d.y_ethanol, d.T_C, "ks", ms=4); a2.set(xlabel="x, y ethanol", ylabel="T (degC)", title="fitted T-x-y"); a2.legend(fontsize=8)
plt.show()
for m, f in fits.items():
    az = activity.azeotrope(comps, f["gamma"], p=p)
    print(f"{m:<9}: rms error in p {100*f['rms_p']:.2f} %; azeotrope at x = {az['x1']:.3f}, {az['T']-273.15:.2f} degC")"""),
    md(r"""
All four models fit the pressures within about 1 %, but they disagree on the azeotrope: Wilson lands on
x = 0.890 (experimentally 0.894 at 78.15 degC), van Laar and NRTL about 0.905, and the symmetric-looking
Margules model 0.924. The azeotrope sits where the two curves cross at a shallow angle, so it magnifies small
differences in the models' shapes. Wilson and NRTL, which describe asymmetric systems best and carry a
temperature dependence, are the industrial favourites; Margules and van Laar are simple but extrapolate poorly.

## Are the data consistent? The area test
The Gibbs-Duhem equation links the two activity coefficients: for a consistent isobaric data set,
$\int_0^1\ln(\gamma_1/\gamma_2)\,dx_1 \approx 0$ (the areas above and below zero cancel):
"""),
    code(r"""ratio = np.log(g1 / g2)
x_d = inner.x_ethanol.to_numpy()
plt.plot(x_d, ratio, "o-"); plt.axhline(0, color="k", lw=0.8); plt.fill_between(x_d, ratio, 0, alpha=0.2)
plt.xlabel("x ethanol"); plt.ylabel("ln(gamma1 / gamma2)"); plt.show()
pos = np.trapezoid(np.maximum(ratio, 0), x_d); neg = -np.trapezoid(np.minimum(ratio, 0), x_d)
print(f"area above zero {pos:.3f}, below {neg:.3f}: index |A+ - A-|/(A+ + A-) = {abs(pos - neg)/(pos + neg):.3f} (pass if < 0.10)")
print(f"the fitted Wilson model itself: index {activity.redlich_kister_area(fits['wilson']['gamma'], 351.0)['index']:.4f}")"""),
    md(r"""
The data pass. Real published data sets do not always: the area test is the first filter applied before any
parameters are fitted, because inconsistent data cannot be described by any model that obeys the
Gibbs-Duhem equation.

## The azeotrope and bioethanol
Fermentation gives about 10 % ethanol; distillation concentrates it - but only to the azeotrope:
"""),
    code(r"""az = activity.azeotrope(comps, fits["wilson"]["gamma"], p=p)
print(f"azeotrope at 1 atm: {az['x1']:.3f} mole fraction = {az['x1']*46.07/(az['x1']*46.07 + (1-az['x1'])*18.02):.1%} by mass, {az['T']-273.15:.1f} degC")
for p_low in (0.3e5, 0.1e5):
    az_low = activity.azeotrope(comps, fits["wilson"]["gamma"], p=p_low)
    print(f"at {p_low/1e5:.1f} bar: " + (f"azeotrope at x = {az_low['x1']:.3f}, {az_low['T']-273.15:.1f} degC" if az_low else "no azeotrope"))"""),
    md(r"""
Ordinary distillation stops at about 95-96 % by mass. Fuel-grade ethanol needs more: pressure-swing
distillation (the azeotrope moves, or vanishes, at low pressure), an entrainer that breaks it (azeotropic
distillation), or molecular sieves. The whole design question turns on the activity coefficients of this
notebook.
"""),
]

NOTEBOOKS["10_unifac_prediction"] = [
    md(r"""
# 10 · UNIFAC: predicting activity coefficients

Fitted models need data for every pair of substances - and industry works with thousands. UNIFAC sidesteps
the problem: a molecule is a collection of *groups* (CH$_3$, CH$_2$, OH, H$_2$O, ...), and the interactions between
groups, fitted once to a large database, predict the activity coefficients of any mixture built from them.
It is the standard first estimate when no data exist - for solvent selection, screening and preliminary design.
"""),
    setup_cell(["from engthermo import activity, datasets, unifac, vle"]),
    md("## Molecules as groups"),
    code(r"""for name in ("ethanol", "water", "acetone", "benzene", "n-hexane", "chloroform"):
    print(f"{name:<11}: {unifac.MOLECULES[name]}")
g = unifac.gammas(["ethanol", "water"], [0.4, 0.6], 351.0)
print(f"\nethanol-water at x = 0.4, 351 K: gamma = {g.round(3)}")"""),
    md(r"""
## A genuine test: ethanol-water
The data file of notebook 09 was built from the experimental azeotrope and limiting activity coefficient,
*not* from UNIFAC - so this is a real prediction test:
"""),
    code(r"""d = pd.read_csv(datasets.path("ethanol_water_vle_1atm.csv"), comment="#")
comps, p = ["ethanol", "water"], 101325.0
g_uni = unifac.gamma_function(comps)
txy = vle.Txy(comps, p, g_uni, n=41)
plt.plot(d.x_ethanol, d.T_C, "ko", ms=4, label="data"); plt.plot(d.y_ethanol, d.T_C, "ks", ms=4)
plt.plot(txy["x1"], txy["T"] - 273.15, "C0", label="UNIFAC (no fitted parameters)"); plt.plot(txy["y1"], txy["T"] - 273.15, "C0")
plt.xlabel("x, y ethanol"); plt.ylabel("T (degC)"); plt.legend(); plt.show()
az = activity.azeotrope(comps, g_uni, p=p)
inner = d[(d.x_ethanol > 0) & (d.x_ethanol < 1)]
T_pred = np.array([vle.bubble_T(comps, [x, 1 - x], p, g_uni)["T"] - 273.15 for x in inner.x_ethanol])
y_pred = np.array([vle.bubble_T(comps, [x, 1 - x], p, g_uni)["y"][0] for x in inner.x_ethanol])
print(f"UNIFAC azeotrope: x = {az['x1']:.3f} at {az['T']-273.15:.2f} degC (experimental 0.894, 78.15 degC)")
print(f"errors against the data: T within {np.max(np.abs(T_pred - inner.T_C)):.2f} K, y within {np.max(np.abs(y_pred - inner.y_ethanol)):.3f}")"""),
    md(r"""
With no information about ethanol-water except the two molecules' groups, UNIFAC predicts the shape of the
diagram and the azeotrope to within 0.003 in composition and 0.1 K. Elsewhere it is less exact: bubble
temperatures deviate by up to 2.8 K and vapour compositions by up to 0.06, mostly at low ethanol content.
That is typical of UNIFAC - the right shape and the key features, with errors of a few percent in the
details. It is not a substitute for data in final design, but it is a remarkably good start.

## Positive and negative deviations
"""),
    code(r"""x = np.linspace(0.001, 0.999, 100)
fig, ax = plt.subplots()
for pair in (["n-hexane", "ethanol"], ["ethanol", "water"], ["benzene", "toluene"], ["acetone", "chloroform"]):
    ge = [activity.excess_gibbs(unifac.gamma_function(pair), [v, 1 - v], 323.15) for v in x]
    ax.plot(x, ge, label=f"{pair[0]} - {pair[1]}")
ax.axhline(0, color="k", lw=0.8); ax.set(xlabel="x1", ylabel="G^E / RT at 50 degC"); ax.legend(); plt.show()
for pair in (["n-hexane", "ethanol"], ["acetone", "chloroform"]):
    ginf = activity.infinite_dilution(unifac.gamma_function(pair), 323.15)
    print(f"{pair[0]} in {pair[1]}: gamma_inf = {ginf[0]:.2f};  {pair[1]} in {pair[0]}: {ginf[1]:.2f}")"""),
    md(r"""
Hexane-ethanol: strongly positive (the alcohol's hydrogen bonds exclude the alkane; a large limiting activity
coefficient hints at limited miscibility, notebook 12). Benzene-toluene: nearly zero, as Raoult's law assumed.
Acetone-chloroform: **negative** deviations - the C-H of chloroform hydrogen-bonds to the ketone oxygen, the
molecules attract each other more than themselves, and the mixture has a *maximum*-boiling azeotrope.

## Solvent screening
Which solvent extracts acetone from water best? A low activity coefficient of acetone in the solvent means a
high affinity:
"""),
    code(r"""rows = []
for solvent in ("water", "benzene", "toluene", "n-hexane", "chloroform", "ethyl acetate", "1-butanol"):
    ginf = activity.infinite_dilution(unifac.gamma_function(["acetone", solvent]), 298.15)[0]
    rows.append((solvent, ginf))
print(pd.DataFrame(rows, columns=["solvent", "gamma_inf of acetone at 25 degC"]).sort_values("gamma_inf of acetone at 25 degC").round(2).to_string(index=False))"""),
    md(r"""
Chloroform's affinity for acetone (activity coefficient below 1) is the hydrogen bond above; the alkane's is
poor. Screening hundreds of candidate solvents this way, before any experiment, is UNIFAC's everyday job in
process development - followed by measurements on the shortlist.
"""),
]

NOTEBOOKS["11_high_pressure_phase_equilibria"] = [
    md(r"""
# 11 · High-pressure phase equilibria

Natural gas processing, CO$_2$ capture and liquefied gases live at tens of bar, where the vapour is far from
ideal and Raoult's law is useless. An equation of state for both phases - the $\phi$-$\phi$ approach - handles
it: the condition is equal fugacities, $x_i\hat\phi_i^L = y_i\hat\phi_i^V$, with the fugacity coefficients of each
component in each mixture phase from the Peng-Robinson equation and mixing rules.
"""),
    setup_cell(["from engthermo import datasets, eos, vle"]),
    md("## Methane-ethane at 200 K"),
    code(r"""ref = pd.read_csv(datasets.path("methane_ethane_bubble_200K.csv"), comment="#")
comps = ["methane", "ethane"]
pr = vle.eos_Pxy("pr", comps, 200.0, n=19, x_max=0.9)
plt.plot(ref.x_methane, ref.p_bubble_Pa / 1e5, "ko", label="reference (multi-fluid model)"); plt.plot(ref.y_methane, ref.p_bubble_Pa / 1e5, "ks")
plt.plot(pr["x1"], pr["p"] / 1e5, "C0", label="Peng-Robinson, k12 = 0"); plt.plot(pr["y1"], pr["p"] / 1e5, "C0--")
plt.xlabel("x, y methane"); plt.ylabel("p (bar)"); plt.title("methane-ethane, 200 K"); plt.legend(); plt.show()
p_pr = np.array([vle.eos_bubble_P("pr", comps, [x, 1 - x], 200.0)["p"] for x in ref.x_methane])
print(f"bubble pressures: PR within {np.max(np.abs(p_pr / ref.p_bubble_Pa - 1)):.1%} of the reference model")"""),
    md(r"""
With no adjustable parameter the Peng-Robinson equation reproduces the bubble curve of this simple mixture
within a few percent. The vapour is strongly enriched in methane; at high pressure the liquid and vapour
curves converge towards the mixture's critical point.

## Interaction parameters
For unlike molecules the geometric-mean combining rule $a_{12} = \sqrt{a_1a_2}$ needs a correction, the binary
interaction parameter $k_{ij}$:
"""),
    code(r"""fig, ax = plt.subplots()
for kij in (-0.05, 0.0, 0.05, 0.1):
    k = [[0, kij], [kij, 0]]
    d = vle.eos_Pxy("pr", comps, 200.0, kij=k, n=15, x_max=0.85)
    ax.plot(d["x1"], d["p"] / 1e5, label=f"k12 = {kij}")
ax.plot(ref.x_methane, ref.p_bubble_Pa / 1e5, "ko", label="reference"); ax.set(xlabel="x methane", ylabel="p (bar)"); ax.legend(); plt.show()"""),
    md(r"""
A positive $k_{ij}$ weakens the cross-attraction and raises the bubble pressure. For hydrocarbon pairs
$k_{ij}$ is near zero; for CO$_2$-hydrocarbon, N$_2$-hydrocarbon or water-anything it is essential and is
fitted to data. Process simulators ship large tables of them.

## Hydrocarbon dew point of a natural gas
Pipeline gas must not condense at the coldest point of the line. The dew pressure of a lean gas:
"""),
    code(r"""gas = ["methane", "ethane", "propane", "n-butane"]
y = [0.90, 0.06, 0.03, 0.01]
for T in (200.0, 220.0, 240.0, 245.0, 250.0):
    try:
        dew = vle.eos_dew_P("pr", gas, y, T)
        print(f"{T:.0f} K: dew pressure {dew['p']/1e5:6.2f} bar; first liquid drop: " + ", ".join(f"{c} {x:.2f}" for c, x in zip(gas, dew["x"])))
    except ValueError:
        print(f"{T:.0f} K: no dew point at any pressure - above the cricondentherm")"""),
    md(r"""
The first droplets are rich in the heaviest components: a trace of butane controls the dew point of the whole
gas. Above about 246 K (-27 degC) this gas does not condense at any pressure - its **cricondentherm**; below
it, the dew pressure falls steeply with temperature, so a pipeline at 70 bar and -20 degC would be well inside
the two-phase region. This is why gas is "conditioned" - the heavy ends removed - before it enters a pipeline,
and why the hydrocarbon dew point is a contractual specification.

## CO$_2$ in natural gas
Sour and CO$_2$-rich gases: how much CO$_2$ dissolves in liquid methane at low temperature, and what does a flash
of a CO$_2$-rich gas give? (With the recommended $k_{ij}$ of about 0.1 for CO$_2$-methane.)
"""),
    code(r"""k = [[0, 0.1], [0.1, 0]]
d0 = vle.eos_Pxy("pr", ["methane", "CO2"], 230.0, n=13, x_max=0.6)
d1 = vle.eos_Pxy("pr", ["methane", "CO2"], 230.0, kij=k, n=13, x_max=0.6)
plt.plot(d0["x1"], d0["p"] / 1e5, "--", label="k12 = 0"); plt.plot(d1["x1"], d1["p"] / 1e5, label="k12 = 0.1 (recommended)")
plt.xlabel("x methane"); plt.ylabel("bubble pressure (bar)"); plt.title("methane-CO2 at 230 K"); plt.legend(); plt.show()
mix = eos.mixture("pr", ["methane", "CO2"], [0.5, 0.5], 230.0, 30e5, kij=k, phase="vapor")
print(f"fugacity coefficients in a 50/50 vapour at 230 K, 30 bar: methane {mix['phi_i'][0]:.3f}, CO2 {mix['phi_i'][1]:.3f}")"""),
    md(r"""
The interaction parameter changes the bubble pressures by tens of percent - a warning that $k_{ij} = 0$ is a
guess, not a default. Fugacity coefficients well below 1 for CO$_2$ in the vapour show how far from ideal the
gas is; every equilibrium and every compressor duty in a CO$_2$-rich process depends on such numbers.
"""),
]

NOTEBOOKS["12_liquid_liquid_equilibria_and_polymers"] = [
    md(r"""
# 12 · Liquid-liquid equilibria and polymer solutions

Oil and water separate; so do butanol and water beyond a few percent - and so, under the right conditions,
do polymer solutions and blends. Two liquid phases form when mixing lowers the Gibbs energy less than
staying apart would: the same activity coefficients that bend a T-x-y diagram, if large enough, split the
liquid. This notebook computes miscibility gaps from activity models and Flory-Huggins theory, the
foundation of polymer solution thermodynamics.
"""),
    setup_cell(["from engthermo import activity, lle, unifac"]),
    md(r"""
## When does a liquid split? The Gibbs energy of mixing
"""),
    code(r"""x = np.linspace(0.001, 0.999, 300)
fig, ax = plt.subplots()
for A in (1.0, 2.0, 3.0):
    ax.plot(x, lle.gibbs_of_mixing(activity.margules(A), x, 300.0), label=f"symmetric Margules, A = {A}")
ax.axhline(0, color="k", lw=0.8); ax.set(xlabel="x1", ylabel="Delta G_mix / RT"); ax.legend(); plt.show()
for A in (1.5, 2.5, 3.0):
    r = lle.binary(activity.margules(A), 300.0)
    print(f"A = {A}: " + ("completely miscible" if r is None else f"two phases with x1 = {r['x1_alpha']:.3f} and {r['x1_beta']:.3f}"))"""),
    md(r"""
For weak non-ideality the curve is convex everywhere: any mixture has a lower Gibbs energy than any two
phases it could split into. Above a critical value ($A$ = 2 for the symmetric model) the curve develops a
hump, and a common tangent touches two compositions: the mixture splits into those two phases, with the
same chemical potential (activity) of each component in both.

## Butanol-water predicted by UNIFAC
"""),
    code(r"""g_bw = unifac.gamma_function(["1-butanol", "water"])
for T_C in (25, 60, 90):
    r = lle.binary(g_bw, T_C + 273.15)
    print(f"{T_C} degC: butanol-rich phase x_butanol = {r['x1_beta']:.3f}, water-rich phase x_butanol = {r['x1_alpha']:.4f}")
print("experimental at 25 degC: about 0.50 and 0.019 (7.8 wt % butanol in the aqueous phase)")"""),
    md(r"""
UNIFAC, parameterised on vapour-liquid data, predicts both phases of this system within about 5 % - and the
widening of the miscibility gap's water-rich branch with temperature. Such agreement is not guaranteed:
the water-rich branch is controlled by the very large activity coefficient of butanol at infinite dilution,
which VLE-based parameters often get wrong by tens of percent. For design, liquid-liquid equilibria are fitted
with NRTL or UNIQUAC parameters from LLE data themselves; UNIFAC is the screening tool.

## Extraction: the distribution coefficient
"""),
    code(r"""ginf_w = activity.infinite_dilution(unifac.gamma_function(["acetone", "water"]), 298.15)[0]
for solvent in ("1-butanol", "ethyl acetate", "toluene"):
    ginf_s = activity.infinite_dilution(unifac.gamma_function(["acetone", solvent]), 298.15)[0]
    print(f"acetone from water into {solvent:<14}: distribution coefficient (mole-fraction basis) about {ginf_w/ginf_s:.1f}")"""),
    md(r"""
At low solute concentration the distribution coefficient between two immiscible solvents is the ratio of the
solute's infinite-dilution activity coefficients, $K = \gamma^\infty_{water}/\gamma^\infty_{solvent}$. A large $K$ means a
small solvent flow in the extractor - the first number an extraction design needs.

## Polymer solutions: Flory-Huggins theory
A polymer chain of $N$ segments on a lattice mixes with solvent with a tiny entropy (the segments cannot move
independently) and an interaction energy $\chi$ per contact:
$\Delta G/RT = (1-\phi)\ln(1-\phi) + (\phi/N)\ln\phi + \chi\phi(1-\phi)$. Because the entropy is so small,
even a mild unfavourable $\chi$ splits the solution:
"""),
    code(r"""phi = np.linspace(0.001, 0.999, 400)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
for N in (1, 10, 100):
    crit = lle.flory_huggins_critical(N)
    a1.plot(phi, lle.flory_huggins_gibbs(phi, 0.6, N), label=f"N = {N}")
    print(f"N = {N:4d}: critical chi = {crit['chi_c']:.3f} at polymer volume fraction {crit['phi_c']:.3f}")
a1.set(xlabel="polymer volume fraction", ylabel="Delta G / RT per site, chi = 0.6"); a1.legend()
chis = np.linspace(0.55, 1.2, 40)
N = 100
bin_lo, bin_hi, sp = [], [], []
for c in chis:
    b = lle.flory_huggins_binodal(c, N)
    bin_lo.append(b["phi_dilute"] if b else np.nan); bin_hi.append(b["phi_concentrated"] if b else np.nan)
    s = lle.flory_huggins_spinodal(c, N); sp.append(s if s.size else [np.nan, np.nan])
sp = np.array(sp)
a2.plot(bin_lo, chis, "C0", label="binodal"); a2.plot(bin_hi, chis, "C0"); a2.plot(sp[:, 0], chis, "C1--", label="spinodal"); a2.plot(sp[:, 1], chis, "C1--")
a2.invert_yaxis(); a2.set(xlabel="polymer volume fraction", ylabel="chi (larger = worse solvent)", title=f"phase diagram, N = {N}"); a2.legend(); plt.show()"""),
    md(r"""
Two features distinguish polymers from small molecules: the critical $\chi$ falls towards 0.5 as the chains
lengthen (a solvent that is marginal for the monomer is a non-solvent for the polymer), and the phase diagram
is strongly asymmetric - the dilute phase contains almost no polymer while the concentrated phase is a gel.
Between the binodal and the spinodal the solution is metastable; inside the spinodal it decomposes
spontaneously - the mechanism behind many porous membranes and polymer foams.

## Temperature: the upper critical solution temperature
With $\chi = A + B/T$, the polymer dissolves on heating and precipitates on cooling below the UCST:
"""),
    code(r"""A, B = 0.20, 90.0                                     # an illustrative polystyrene-cyclohexane-like system (theta temperature ~ 300 K)
for N in (100, 1000, 10000):
    print(f"N = {N:5d}: UCST = {lle.ucst(A, B, N):.1f} K; theta temperature (chi = 0.5, infinite N) = {B/(0.5 - A):.1f} K")"""),
    md(r"""
The UCST rises with molar mass towards the theta temperature - which is why fractionating a polymer by
molar mass works (cooling precipitates the longest chains first), and why polymer blends of high molar mass
almost never mix: with two long chains, even the small entropy of mixing of a polymer solution is gone.
"""),
]

NOTEBOOKS["13_solid_liquid_equilibria_and_alloys"] = [
    md(r"""
# 13 · Solid-liquid equilibria and alloy phase diagrams

How much sugar dissolves in water, at what temperature antifreeze freezes, and why an alloy of two metals
melts over a range of temperatures - and sometimes far below either metal - are all questions of
solid-liquid equilibrium. The thermodynamics is the same as for vapour-liquid equilibrium: equal chemical
potentials in both phases; the difference is that the solid's reference state involves the enthalpy of fusion.
"""),
    setup_cell(["from engthermo import activity, sle, unifac"]),
    md(r"""
## Ideal solubility
For a solid that forms an ideal solution with the liquid, the Schroeder-van Laar equation gives its mole
fraction: $\ln x = -\frac{\Delta H_{fus}}{R}\left(\frac1T - \frac1{T_m}\right)$. It depends only on the *solute's* melting point and
enthalpy of fusion - the solvent does not appear. Naphthalene ($T_m$ = 80.2 degC, $\Delta H_{fus}$ = 19.1 kJ/mol):
"""),
    code(r"""T = np.linspace(273.15, 353.4, 100)
plt.plot(T - 273.15, sle.ideal_solubility(T, 353.4, 19.1e3)); plt.xlabel("T (degC)"); plt.ylabel("x naphthalene (ideal)"); plt.show()
x25 = sle.ideal_solubility(298.15, 353.4, 19.1e3)
print(f"ideal solubility at 25 degC: {x25:.3f}   (experimental: in benzene 0.296, in toluene 0.286, in hexane 0.12, in water ~1e-5)")"""),
    md(r"""
In benzene and toluene - chemically similar to naphthalene - the ideal value is nearly exact. In hexane the
solubility is lower (positive deviations, $\gamma > 1$) and in water it is negligible: "like dissolves like" is
the activity coefficient at work. UNIFAC estimates it:
"""),
    code(r"""from scipy import optimize
for solvent in ("benzene", "n-hexane", "ethanol"):
    g = unifac.gamma_function([{"ACH": 8, "AC": 2}, solvent])            # naphthalene: 8 aromatic CH + 2 fused carbons
    f = lambda x: np.log(x * g([x, 1 - x], 298.15)[0]) + 19.1e3 / 8.314462618 * (1 / 298.15 - 1 / 353.4)
    x_sol = optimize.brentq(f, 1e-6, 0.999)
    print(f"naphthalene in {solvent:<9} at 25 degC: x = {x_sol:.3f} (gamma = {g([x_sol, 1 - x_sol], 298.15)[0]:.2f})")"""),
    md(r"""
## Freezing-point depression: antifreeze
Dissolving anything in water lowers its freezing point, because the solid that forms is pure ice while the
liquid's water activity is reduced. For an ideal solution the same equation, applied to *water* as the solute
crystallising as ice ($T_m$ = 273.15 K, $\Delta H_{fus}$ = 6.01 kJ/mol):
"""),
    code(r"""x_glycol = np.linspace(0, 0.5, 51)
T_freeze = [sle.liquidus_T(1 - x, 273.15, 6.01e3) for x in x_glycol]         # water mole fraction 1 - x
M_g, M_w = 62.07, 18.02
w_glycol = x_glycol * M_g / (x_glycol * M_g + (1 - x_glycol) * M_w)
plt.plot(w_glycol * 100, np.array(T_freeze) - 273.15, label="ideal solution")
plt.plot([20, 30, 40, 50], [-7.8, -14.1, -23.0, -36.4], "ko", label="experimental (ethylene glycol)")
plt.xlabel("ethylene glycol (mass %)"); plt.ylabel("freezing point (degC)"); plt.legend(); plt.show()"""),
    md(r"""
The ideal-solution curve captures the trend; the real antifreeze depresses the freezing point somewhat more
at high concentration (glycol-water shows moderate negative deviations). A 50 % mixture protects to about
-36 degC - the basis of every engine coolant and of the low-temperature heat-transfer fluids of solar and
geothermal systems.

## Eutectic alloys: Bi-Cd
Two metals that do not dissolve in each other's solid but mix ideally as liquids: each liquidus branch is the
ideal solubility of one metal in the melt, and they meet at the **eutectic** - the lowest melting point of the
system, where the liquid freezes into a fine mixture of both solids:
"""),
    code(r"""Tm_Bi, dH_Bi, Tm_Cd, dH_Cd = 544.5, 11.3e3, 594.2, 6.19e3
cur = sle.liquidus_curves(Tm_Bi, dH_Bi, Tm_Cd, dH_Cd)
plt.plot(1 - cur["x1_branch1"], cur["T_branch1"] - 273.15, "C0", label="liquidus (Bi crystallises)")
plt.plot(1 - cur["x1_branch2"], cur["T_branch2"] - 273.15, "C1", label="liquidus (Cd crystallises)")
Te = cur["eutectic"]["T"] - 273.15
plt.hlines(Te, 0, 1, colors="k", linestyles=":", label="eutectic temperature")
plt.plot(1 - cur["eutectic"]["x1"], Te, "ko"); plt.plot([0, 1], [Tm_Bi - 273.15, Tm_Cd - 273.15], "ks")
plt.xlabel("x cadmium"); plt.ylabel("T (degC)"); plt.title("Bi-Cd (ideal liquid, immiscible solids)"); plt.legend(fontsize=8); plt.show()
print(f"predicted eutectic: {1 - cur['eutectic']['x1']:.2f} mole fraction Cd at {Te:.0f} degC; experimental: 0.55 at 140 degC")"""),
    md(r"""
With only the melting points and enthalpies of fusion the eutectic is predicted within a few degrees. Low
melting eutectics are the basis of solders (Sn-Pb, Sn-Ag-Cu), of fusible alloys for safety devices, and of the
salt mixtures (e.g. NaNO$_3$-KNO$_3$) used as heat-transfer and storage media in solar thermal power plants.

## Isomorphous alloys: Cu-Ni
Copper and nickel mix completely in both liquid and solid. With ideal solutions in both phases, the
liquidus and solidus follow from the two equilibrium constants $K_i = x_i^S/x_i^L$:
"""),
    code(r"""lens = sle.lens_diagram(1358.0, 13.05e3, 1728.0, 17.47e3)
plt.plot(1 - lens["x1_liquidus"], lens["T"] - 273.15, label="liquidus"); plt.plot(1 - lens["x1_solidus"], lens["T"] - 273.15, label="solidus")
plt.fill_betweenx(lens["T"] - 273.15, 1 - lens["x1_liquidus"], 1 - lens["x1_solidus"], alpha=0.15)
plt.xlabel("x nickel"); plt.ylabel("T (degC)"); plt.title("Cu-Ni: ideal liquid and solid solutions"); plt.legend(); plt.show()
i = np.argmin(np.abs(lens["x1_liquidus"] - 0.5))
print(f"a 50/50 melt starts to freeze at {lens['T'][i]-273.15:.0f} degC, and the first solid contains {1 - lens['x1_solidus'][i]:.2f} mole fraction Ni")"""),
    md(r"""
The lens (or cigar) diagram: an alloy freezes over a temperature range, and the first solid is richer in the
higher-melting metal than the liquid it came from. Cast slowly, the composition of the solid changes as
freezing proceeds - the origin of *coring* and segregation in castings, which heat treatments must remove.
The same two-phase lens, with the roles of vapour and liquid, was the T-x-y diagram of notebook 08: the
thermodynamics of separation is one subject.
"""),
]

NOTEBOOKS["14_chemical_reaction_equilibrium"] = [
    md(r"""
# 14 · Chemical reaction equilibrium

How much ammonia can a reactor make at 450 degC and 200 bar? Why is hydrogen made by steam reforming at
900 degC and methanol synthesised at 250 degC? Thermodynamics sets the ceiling on every reaction's conversion
before any catalyst is chosen. This notebook computes equilibrium constants from formation data, follows them
with temperature, and finds equilibrium compositions - for one reaction and for a whole reacting mixture.
"""),
    setup_cell(["from engthermo import reaction", "from engthermo.constants import R"]),
    md(r"""
## From formation data to the equilibrium constant
$\Delta G^\circ(T) = \Delta H^\circ(T) - T\Delta S^\circ(T)$, built from the enthalpies and Gibbs energies of formation at 25 degC and the
heat capacities; then $K = \exp(-\Delta G^\circ/RT)$:
"""),
    code(r"""rxns = {"ammonia synthesis: N2 + 3 H2 -> 2 NH3": {"N2": -1, "H2": -3, "NH3": 2},
        "methanol synthesis: CO + 2 H2 -> CH3OH": {"CO": -1, "H2": -2, "methanol": 1},
        "steam reforming: CH4 + H2O -> CO + 3 H2": {"CH4": -1, "H2O": -1, "CO": 1, "H2": 3},
        "water-gas shift: CO + H2O -> CO2 + H2": {"CO": -1, "H2O": -1, "CO2": 1, "H2": 1}}
Ts = np.linspace(298.15, 1300, 200)
fig, ax = plt.subplots()
for name, rxn in rxns.items():
    ax.plot(1000 / Ts, [np.log10(reaction.equilibrium_constant(rxn, T)) for T in Ts], label=name.split(":")[0])
    print(f"{name:<42}: dH298 = {reaction.standard_enthalpy(rxn)/1e3:7.1f} kJ/mol, K(298) = {reaction.equilibrium_constant(rxn):9.3g}, K(1000 K) = {reaction.equilibrium_constant(rxn, 1000.0):9.3g}")
ax.axhline(0, color="k", lw=0.8); ax.set(xlabel="1000 / T (1/K)", ylabel="log10 K"); ax.legend(); plt.show()"""),
    md(r"""
The van 't Hoff plot: exothermic reactions (ammonia, methanol, shift) have $K$ falling with temperature -
they want low temperature but need high temperature for the catalyst to work, the central compromise of
their processes. Endothermic reforming goes the other way: only above about 900 K is $K$ large.

## The van 't Hoff approximation
"""),
    code(r"""rxn = rxns["ammonia synthesis: N2 + 3 H2 -> 2 NH3"]
for T in (500.0, 700.0, 900.0):
    print(f"{T:.0f} K: exact K = {reaction.equilibrium_constant(rxn, T):.3e}, van 't Hoff from 298 K with constant dH = {reaction.van_t_hoff(rxn, 298.15, T):.3e}")"""),
    md(r"""
Assuming a constant heat of reaction is adequate over a hundred kelvin, not over six hundred: the heat
capacities change $\Delta H^\circ$ enough to shift $K$ by a factor of two or more.

## Ammonia: the Haber-Bosch compromise
For a single reaction the equilibrium composition follows from $K = \prod(y_ip/p^\circ)^{\nu_i}$ and the reaction extent:
"""),
    code(r"""feed = {"N2": 1.0, "H2": 3.0}
rows = []
for T_C in (300, 400, 450, 500):
    for p_bar in (50, 100, 200, 300):
        ex = reaction.extent(rxn, feed, T_C + 273.15, p_bar * 1e5)
        rows.append((T_C, p_bar, ex["y"]["NH3"]))
t = pd.DataFrame(rows, columns=["T (degC)", "p (bar)", "y_NH3"]).pivot(index="T (degC)", columns="p (bar)", values="y_NH3")
print("equilibrium mole fraction of ammonia (stoichiometric feed, ideal gas):"); print(t.round(3))"""),
    md(r"""
Low temperature and high pressure favour ammonia, as Le Chatelier's principle says - the reaction is
exothermic and reduces the number of moles. But below about 400 degC the iron catalyst is too slow, so
industry runs at 400-500 degC and 150-300 bar, accepts 15-25 % conversion per pass, and recycles the rest.
(At 300 bar the gas is not ideal; fugacity coefficients, notebook 05, raise the equilibrium conversion somewhat.)

Inerts (argon, methane) that accumulate in the recycle dilute the reactants and lower the conversion:
"""),
    code(r"""for inert in (0.0, 0.5, 1.0):
    ex = reaction.extent(rxn, {"N2": 1.0, "H2": 3.0, "Ar": inert}, 723.15, 200e5)
    print(f"{inert:.1f} mol Ar per mol N2: y_NH3 = {ex['y']['NH3']:.3f}, extent {ex['extent']:.3f}")"""),
    md(r"""
## Hydrogen by steam reforming: many reactions at once
Reforming, water-gas shift and methanation happen together. Rather than juggling three coupled
equilibria, minimise the Gibbs energy of the whole mixture subject to the element balances:
"""),
    code(r"""species = ["CH4", "H2O", "CO", "CO2", "H2"]
rows = []
for T_C in (500, 600, 700, 800, 900, 1000):
    gm = reaction.gibbs_minimization(species, {"CH4": 1.0, "H2O": 3.0}, T_C + 273.15, 20e5)
    rows.append((T_C, *[gm["y"][s] for s in species], 1 - gm["moles"]["CH4"]))
print(pd.DataFrame(rows, columns=["T (degC)"] + [f"y {s}" for s in species] + ["CH4 conversion"]).round(3).to_string(index=False))"""),
    md(r"""
At 20 bar and a steam-to-carbon ratio of 3, methane conversion needs about 900 degC - the reason reformer
tubes glow. The CO/CO$_2$ split is governed by the shift equilibrium and the hydrogen yield by both reactions
together. The reformate then passes shift reactors at lower temperature to convert CO into more H$_2$ and CO$_2$:
"""),
    code(r"""gm_ref = reaction.gibbs_minimization(species, {"CH4": 1.0, "H2O": 3.0}, 900 + 273.15, 20e5)
n = gm_ref["moles"]
shift = {"CO": -1, "H2O": -1, "CO2": 1, "H2": 1}
for T_C in (450, 350, 200):
    sh = reaction.extent(shift, n, T_C + 273.15, 20e5)                     # only the shift reaction proceeds on a shift catalyst
    print(f"shift reactor at {T_C} degC: H2 {sh['moles']['H2']:.3f} mol, CO {sh['moles']['CO']:.4f} mol, CO2 {sh['moles']['CO2']:.3f} mol per mol CH4 fed")
print(f"\noverall: about {sh['moles']['H2']:.2f} mol H2 and {sh['moles']['CO2'] + sh['moles']['CO']:.2f} mol CO2 per mol of methane - "
      f"{(sh['moles']['CO2'] + sh['moles']['CO']) * 44.01 / (sh['moles']['H2'] * 2.016):.1f} kg CO2 per kg H2 before any fuel is burnt for the reformer")
full = reaction.gibbs_minimization(species, n, 450 + 273.15, 20e5)
print(f"\nbut the FULL equilibrium of the same gas at 450 degC: CH4 {full['moles']['CH4']:.2f} mol, H2 {full['moles']['H2']:.2f} mol - methanation!")"""),
    md(r"""
Two lessons. First, the shift reactors recover most of the hydrogen bound in CO: about 3.7 mol H$_2$ per mol of
methane against the stoichiometric maximum of 4. Second - and easy to miss - the *complete* equilibrium of
the reformer gas at 450 degC is not shifted gas at all but methane: at that temperature and pressure,
methanation (CO + 3 H$_2$ -> CH$_4$ + H$_2$O) is thermodynamically overwhelming. Shift reactors work only because
their catalysts are selective. Thermodynamics tells you what *can* happen; the catalyst decides what *does*.

"Grey" hydrogen from natural gas thus carries about 5.5 kg of CO$_2$ per kg of hydrogen from the chemistry
alone and roughly 9-10 kg once the reformer's fuel is counted - the numbers that motivate carbon capture on
reformers ("blue" hydrogen) and electrolysis from renewable power ("green" hydrogen, notebook 16).
"""),
]

NOTEBOOKS["15_high_temperature_thermochemistry"] = [
    md(r"""
# 15 · High-temperature thermochemistry: combustion and metals

Burning fuels and reducing ores are the two oldest high-temperature industries, and both are governed by
the same quantities: reaction enthalpies for the heat released and Gibbs energies for what can react at all.
This notebook computes heating values and flame temperatures, compares fuels by their CO$_2$ emissions, and
uses the Ellingham diagram to explain how metals are won from their oxides - and why hydrogen may replace
carbon in making steel.
"""),
    setup_cell(["from engthermo import combustion, components, reaction", "from engthermo.constants import R"]),
    md("## Heating values and CO$_2$ intensity of fuels"),
    code(r"""rows = []
for fuel in ("H2", "CH4", "propane", "n-octane", "methanol", "ethanol"):
    hv = combustion.heating_values(fuel)
    M = components.get(fuel).M
    co2 = combustion.stoichiometry(fuel)["CO2"] * 44.01e-3
    rows.append((fuel, hv["LHV"] / 1e3, hv["HHV"] / 1e3, hv["LHV"] / M / 1e6, co2 / (hv["LHV"] / 1e6) * 1e3))
print(pd.DataFrame(rows, columns=["fuel", "LHV (kJ/mol)", "HHV (kJ/mol)", "LHV (MJ/kg)", "g CO2 per MJ (LHV)"]).round(1).to_string(index=False))"""),
    md(r"""
Per kilogram, hydrogen carries almost three times the energy of hydrocarbons and emits no CO$_2$ at the
point of use; methane emits a quarter less CO$_2$ per megajoule than octane, the basis of coal-to-gas and
oil-to-gas switching; alcohols sit in between. The difference between HHV and LHV is the latent heat of
the water formed - recoverable only if the flue gas is cooled below its dew point (condensing boilers).

## The adiabatic flame temperature
If no heat is lost, all of the heat released goes into the products:
"""),
    code(r"""for fuel in ("CH4", "H2", "propane"):
    print(f"{fuel:<8}: adiabatic flame temperature in stoichiometric air {combustion.adiabatic_flame_temperature(fuel)['T_ad']:.0f} K")
excess = np.linspace(0, 2.0, 21)
T_ad = [combustion.adiabatic_flame_temperature("CH4", e)["T_ad"] for e in excess]
T_pre = [combustion.adiabatic_flame_temperature("CH4", e, T_in=600.0)["T_ad"] for e in excess]
plt.plot(excess * 100, T_ad, label="reactants at 25 degC"); plt.plot(excess * 100, T_pre, label="reactants preheated to 600 K")
plt.xlabel("excess air (%)"); plt.ylabel("adiabatic flame temperature (K)"); plt.title("methane in air"); plt.legend(); plt.show()"""),
    md(r"""
Excess air dilutes the flame - a furnace at 100 % excess air is 500 K cooler - while preheating the
combustion air (with heat recovered from the flue gas) raises it. Both are levers in furnace design: the
first keeps NO$_x$ formation and materials within limits, the second saves fuel. These temperatures ignore
dissociation of CO$_2$ and H$_2$O, which caps real flames about 100-200 K lower.

## The Ellingham diagram
For the oxidation of metals and of carbon, the standard Gibbs energy per mole of O$_2$ against temperature:
"""),
    code(r"""T = np.linspace(300, 2000, 200)
lines = combustion.ellingham(T)
fig, ax = plt.subplots(figsize=(8, 6))
for name, dG in lines.items():
    style = "k" if name.startswith(("2 C", "C ", "2 H2")) else None
    ax.plot(T, dG / 1e3, color=style, lw=2 if style else 1)
    ax.text(T[-1] + 10, dG[-1] / 1e3, name, fontsize=7, va="center")
ax.set(xlabel="T (K)", ylabel="Delta G° per mol O2 (kJ)", title="Ellingham diagram", xlim=(300, 2600)); plt.show()"""),
    md(r"""
Every line slopes upward - oxidation consumes gas, losing entropy - except the C + O$_2$ line for CO, which
*produces* gas and slopes downward. That single fact made the industrial revolution: at high enough
temperature carbon's line crosses below any metal oxide's, so carbon (coke) can reduce it. The lower a
metal's line, the more stable its oxide and the harder the metal is to win:
"""),
    code(r"""from scipy import optimize
def crossing(metal_line):
    f = lambda T_: combustion.ellingham(T_)["2 C + O2 -> 2 CO"][0] - combustion.ellingham(T_)[metal_line][0]
    try:
        return optimize.brentq(f, 300, 3000)
    except ValueError:
        return np.nan
for metal in ("4 Cu + O2 -> 2 Cu2O", "2 Ni + O2 -> 2 NiO", "2 Fe + O2 -> 2 FeO", "2 Zn + O2 -> 2 ZnO", "Si + O2 -> SiO2", "2 Mg + O2 -> 2 MgO", "4/3 Al + O2 -> 2/3 Al2O3"):
    Tc = crossing(metal)
    print(f"{metal:<26}: carbon reduces the oxide above {Tc:5.0f} K" if np.isfinite(Tc) else f"{metal:<26}: not reducible by carbon below 3000 K")"""),
    md(r"""
Copper and nickel are easy, iron needs a blast furnace at over 1000 K, zinc and silicon need more; magnesium
and aluminium cannot be won with carbon at any practical temperature - which is why aluminium is made by
electrolysis (notebook 16) and why it was a precious metal until 1886. The equilibrium oxygen partial pressure
of each line, $p_{O_2} = \exp(\Delta G^\circ/RT)$, shows how oxygen-free a furnace atmosphere must be:
"""),
    code(r"""for name in ("2 Fe + O2 -> 2 FeO", "2 Mg + O2 -> 2 MgO"):
    pO2 = combustion.oxygen_partial_pressure(combustion.ellingham(1000.0)[name][0], 1000.0)
    print(f"{name}: equilibrium p_O2 at 1000 K = {pO2:.1e} bar")"""),
    md(r"""
## Hydrogen as a reductant: green steel
Hydrogen's line also rises with temperature, like a metal's. Where it crosses the iron line, hydrogen can
reduce iron oxide:
"""),
    code(r"""f = lambda T_: combustion.ellingham(T_)["2 H2 + O2 -> 2 H2O"][0] - combustion.ellingham(T_)["2 Fe + O2 -> 2 FeO"][0]
T_cross = optimize.brentq(f, 300, 2500)
print(f"hydrogen reduces FeO above about {T_cross:.0f} K; at 1200 K the equilibrium H2O/H2 ratio is "
      f"{np.exp(-f(1200.0) / (2 * R * 1200.0)):.2f}")"""),
    md(r"""
Direct reduction of iron ore with hydrogen is thermodynamically possible at furnace temperatures, with
water as the only by-product - the route to "green" steel if the hydrogen itself is made without CO$_2$.
The equilibrium H$_2$O/H$_2$ ratio limits how much of the hydrogen is used per pass, so the gas is recycled
after drying: the same thermodynamic ceiling and recycle logic as the ammonia loop of notebook 14.
"""),
]

NOTEBOOKS["16_electrochemical_thermodynamics"] = [
    md(r"""
# 16 · Electrochemical thermodynamics

A fuel cell turns the Gibbs energy of a reaction directly into electrical work; an electrolyser runs the
same reaction backwards; a battery stores it. The link is Faraday's law: $\Delta G = -nFE$. This notebook
computes the reversible voltages, efficiencies and energy densities that bound every electrochemical energy
technology - fuel cells, hydrogen electrolysis and batteries.
"""),
    setup_cell(["from engthermo import electrochem, reaction", "from engthermo.constants import F, R"]),
    md(r"""
## The hydrogen fuel cell
H$_2$ + ½O$_2$ -> H$_2$O(l) transfers two electrons per molecule of hydrogen:
"""),
    code(r"""h2 = {"H2": -1, "O2": -0.5, "H2O(l)": 1}
lim = electrochem.fuel_cell_limits(h2, 2)
print(f"Delta G° = {lim['dG']/1e3:.1f} kJ/mol, Delta H° = {lim['dH']/1e3:.1f} kJ/mol")
print(f"reversible cell voltage E° = {lim['E_rev']:.3f} V; thermoneutral voltage Delta H/(nF) = {lim['E_thermoneutral']:.3f} V")
print(f"maximum efficiency Delta G / Delta H = {lim['efficiency_max']:.1%}  (a heat engine at the same 25 degC would deliver nothing)")"""),
    md(r"""
The reversible voltage of 1.23 V is the fuel cell's ceiling; the 83 % "thermodynamic efficiency" is the
fraction of the heating value available as work - a limit not set by Carnot, because no heat engine is
involved. Real cells run at 0.6-0.8 V under load, and the difference appears as heat:
"""),
    code(r"""for V in (1.0, 0.8, 0.7, 0.6):
    print(f"cell voltage {V:.1f} V: efficiency (HHV basis) {electrochem.efficiency_at_voltage(V, lim['E_thermoneutral']):.1%}, "
          f"heat released {(lim['E_thermoneutral'] - V) * 2 * F / 1e3:.0f} kJ per mol H2")"""),
    md(r"""
## Temperature and pressure
"""),
    code(r"""for T in (298.15, 353.15, 1073.15):
    rxn = h2 if T < 373 else {"H2": -1, "O2": -0.5, "H2O": 1}          # above 100 degC the product is steam
    print(f"{T-273.15:5.0f} degC: E° = {electrochem.standard_potential(rxn, 2, T):.3f} V, max efficiency {electrochem.fuel_cell_limits(rxn, 2, T)['efficiency_max']:.1%}")
print(f"temperature coefficient at 25 degC: {electrochem.temperature_coefficient(h2, 2)*1e3:.2f} mV/K")
E0 = lim["E_rev"]
for p_H2, p_O2 in ((1, 1), (3, 3), (1, 0.21)):
    Q = 1 / (p_H2 * p_O2**0.5)
    print(f"p_H2 = {p_H2} bar, p_O2 = {p_O2} bar: E = {electrochem.nernst(E0, 2, Q):.3f} V")"""),
    md(r"""
The reversible voltage falls with temperature (the reaction's entropy is negative), so high-temperature
fuel cells give up a little voltage but gain fast kinetics and tolerance to carbon monoxide. Pressurising
the reactants raises it slightly (the Nernst equation); running on air instead of oxygen costs about 10 mV.

## Water electrolysis: making green hydrogen
"""),
    code(r"""print(f"minimum (reversible) voltage {lim['E_rev']:.3f} V; thermoneutral voltage {lim['E_thermoneutral']:.3f} V")
for V in (1.48, 1.7, 1.9, 2.1):
    eta = electrochem.electrolyser_efficiency(V, lim["E_thermoneutral"])
    kwh = V * 2 * F / 2.016e-3 / 3.6e6
    print(f"cell voltage {V:.2f} V: efficiency (HHV) {eta:.1%}, electricity {kwh:.1f} kWh per kg H2")"""),
    md(r"""
Between the reversible and the thermoneutral voltage an electrolyser absorbs heat from its surroundings
(the entropy term); above it, it releases heat. Practical cells run at 1.7-2.1 V, needing 45-55 kWh per kg
of hydrogen against a heating value of 39.4 kWh/kg - a 70-85 % efficiency. Combined with the fuel cell's
50-60 %, a hydrogen "round trip" returns about 40 % of the electricity: thermodynamics, not engineering
immaturity, is the largest part of that loss.

## Batteries: theoretical energy densities
The maximum energy a cell can store per kilogram of active materials follows from $\Delta G$ and the masses:
"""),
    code(r"""reaction.EXTRA_FORMATION["ZnO"] = (-350.5e3, -320.5e3, 40.3)      # JANAF (solid), Cp 40.3 J/(mol K)
cells = {"zinc-air: 2 Zn + O2 -> 2 ZnO": ({"O2": -1, "ZnO": 2}, 4, 2 * 65.38e-3),
         "hydrogen-oxygen (fuel cell): H2 + 1/2 O2 -> H2O": (h2, 2, 2.016e-3 + 16.0e-3)}
for name, (rxn, n_e, mass) in cells.items():
    dG = reaction.standard_gibbs(rxn) if "Zn" not in name else 2 * (-320.5e3)
    print(f"{name:<52}: E° = {-dG/(n_e*F):.2f} V, theoretical specific energy {electrochem.energy_per_mass(dG, mass):.0f} Wh/kg")
print("for comparison (from tabulated cell reactions): lead-acid E° 2.04 V, about 170 Wh/kg; lithium-ion about 400-500 Wh/kg theoretical")"""),
    md(r"""
Theoretical specific energies are upper bounds: practical cells reach a quarter to a third of them once
electrolyte, current collectors and casing are added. The ranking survives - metal-air and hydrogen systems
have the highest thermodynamic ceilings because oxygen from the air is a "free" reactant - and it explains
why lithium (the lightest metal with a high potential) dominates portable storage.
"""),
]

NOTEBOOKS["17_from_messy_vle_data_to_a_model"] = [
    md(r"""
# 17 · From messy laboratory data to a thermodynamic model

**Problem.** A student laboratory measured the vapour-liquid equilibrium of methanol and water at
atmospheric pressure and exported the results. The file has the faults such files always have. The task:
turn it into a clean data set, test it, fit an activity-coefficient model with uncertainties, and compare
with a prediction - the complete workflow of a thermodynamic property measurement.
"""),
    setup_cell(["from engthermo import activity, datasets, unifac, vapor, vle"]),
    md("## 1. Look at the file before reading it"),
    code(r"""path = datasets.path("methanol_water_lab_vle.csv")
with open(path, encoding="utf-8") as fh:
    raw = fh.read().splitlines()
print("\n".join(raw[:6])); print("..."); print("\n".join(raw[12:20]))"""),
    md(r"""
Already visible: a pressure column in two different units, a temperature that is clearly in kelvin, a
duplicated run, a missing vapour composition, and a decimal comma that breaks the column count. A blind
`pd.read_csv` would fail or, worse, silently misread.

## 2. Read defensively, then repair
"""),
    code(r"""import io
fixed, log = [], []
for line in raw:
    if line.startswith("#"):
        continue
    f = line.split(",")
    if len(f) == 7:                                   # one field too many: a decimal comma split a number in two
        f = f[:2] + [f[2] + "." + f[3]] + f[4:]
        log.append(f"decimal comma repaired in {f[0]}")
    fixed.append(",".join(f))
d = pd.read_csv(io.StringIO("\n".join(fixed)))
d["p_kPa"] = np.where(d.p_unit == "mmHg", d.p * 101.325 / 760, d.p)                  # unit conversion
log.append(f"{(d.p_unit == 'mmHg').sum()} pressures converted from mmHg to kPa")
kelvin = d["T"] > 200
d.loc[kelvin, "T"] = d.loc[kelvin, "T"] - 273.15
log.append(f"{kelvin.sum()} temperature converted from K to degC")
dup = d.duplicated(subset=["x_methanol", "y_methanol", "T"])
d = d[~dup]; log.append(f"{dup.sum()} duplicated run removed")
missing = d.y_methanol.isna()
d = d[~missing]; log.append(f"{missing.sum()} run without vapour analysis dropped")
d = d.sort_values("x_methanol").reset_index(drop=True)
print("\n".join(log)); print(d[["run", "x_methanol", "y_methanol", "T", "p_kPa"]].to_string(index=False))"""),
    md(r"""
Every repair is logged: a reader of the final report must be able to see what was done to the raw data.

## 3. Test before fitting: point-to-point consistency
Activity coefficients from each point, and the Redlich-Kister area test:
"""),
    code(r"""T = d["T"].to_numpy() + 273.15
p = d.p_kPa.to_numpy() * 1e3
x, y = d.x_methanol.to_numpy(), d.y_methanol.to_numpy()
g1 = y * p / (x * vapor.antoine_psat("methanol", T)); g2 = (1 - y) * p / ((1 - x) * vapor.antoine_psat("water", T))
ratio = np.log(g1 / g2)
plt.plot(x, ratio, "o-"); plt.axhline(0, color="k", lw=0.8); plt.xlabel("x methanol"); plt.ylabel("ln(gamma1/gamma2)"); plt.show()
def area_index(sel):
    pos, neg = np.trapezoid(np.maximum(ratio[sel], 0), x[sel]), -np.trapezoid(np.minimum(ratio[sel], 0), x[sel])
    return abs(pos - neg) / (pos + neg)
print(f"area index {area_index(np.ones(x.size, bool)):.3f} (pass if < 0.10)")
print("activity coefficients:", ", ".join(f"{r}: {a:.2f}/{b:.2f}" for r, a, b in zip(d.run, g1, g2)))"""),
    md(r"""
The data set fails the area test - and the curve shows why: one point breaks an otherwise smooth trend, with
activity coefficients that jump out of line. The area test says *something* is wrong; it cannot say what.
The point must be found point by point.

## 4. Find the outlier objectively
Fit the model to all points, look at the residuals, and refit without the largest one:
"""),
    code(r"""comps = ["methanol", "water"]
def residuals(sel):
    fit = activity.fit("wilson", comps, x[sel], T[sel], p[sel], y1=y[sel])
    res = [vle.bubble_T(comps, [xi, 1 - xi], pi, fit["gamma"]) for xi, pi in zip(x, p)]
    return fit, np.array([r["T"] for r in res]) - T, np.array([r["y"][0] for r in res]) - y
fit_all, dT_all, dy_all = residuals(np.ones(x.size, bool))
worst = int(np.argmax(np.abs(dy_all)))
print(f"largest residual in y: run {d.run[worst]} ({dy_all[worst]:+.3f}; the others are within {np.max(np.delete(np.abs(dy_all), worst)):.3f})")
keep = np.ones(x.size, bool); keep[worst] = False
fit_clean, dT, dy = residuals(keep)
print(f"rms residuals: all points dT {np.sqrt(np.mean(dT_all**2)):.2f} K, dy {np.sqrt(np.mean(dy_all**2)):.3f}; "
      f"without {d.run[worst]}: dT {np.sqrt(np.mean(dT[keep]**2)):.2f} K, dy {np.sqrt(np.mean(dy[keep]**2)):.3f}")"""),
    md(r"""
The residual of one run is ten times the others', and removing it cuts the remaining scatter tenfold. With
that point excluded the area test is passed comfortably:
"""),
    code(r"""print(f"area index without {d.run[worst]}: {area_index(keep):.3f} (pass if < 0.10)")"""),
    md(r"""
A note in the laboratory log turns out to explain the point (a mis-taken vapour sample). Removing a point
needs such a reason *and* the documentation of it - never a silent deletion.

## 5. Parameters with uncertainties
How well are the two Wilson parameters determined? A bootstrap: refit to resampled data sets and look at
the spread:
"""),
    code(r"""rng = np.random.default_rng(1)
idx = np.nonzero(keep)[0]
samples = []
for _ in range(60):
    pick = rng.choice(idx, size=idx.size, replace=True)
    f = activity.fit("wilson", comps, x[pick], T[pick], p[pick], y1=y[pick], p0=[fit_clean["params"]["Lambda"][0, 1], fit_clean["params"]["Lambda"][1, 0]])
    samples.append([f["params"]["Lambda"][0, 1], f["params"]["Lambda"][1, 0]])
samples = np.array(samples)
L12, L21 = fit_clean["params"]["Lambda"][0, 1], fit_clean["params"]["Lambda"][1, 0]
print(f"Wilson Lambda12 = {L12:.3f} +/- {samples[:, 0].std():.3f}, Lambda21 = {L21:.3f} +/- {samples[:, 1].std():.3f} (bootstrap, 60 resamples)")
ginf = activity.infinite_dilution(fit_clean["gamma"], 340.0)
print(f"limiting activity coefficients at 340 K: methanol in water {ginf[0]:.2f}, water in methanol {ginf[1]:.2f}")"""),
    md(r"""
## 6. Compare with a prediction, and report
"""),
    code(r"""g_uni = unifac.gamma_function(comps)
txy_fit, txy_uni = vle.Txy(comps, 101325.0, fit_clean["gamma"], n=41), vle.Txy(comps, 101325.0, g_uni, n=41)
plt.plot(x[keep], T[keep] - 273.15, "ko", label="data (kept)"); plt.plot(y[keep], T[keep] - 273.15, "ks")
plt.plot(x[~keep], T[~keep] - 273.15, "rx", ms=9, label="excluded run")
plt.plot(txy_fit["x1"], txy_fit["T"] - 273.15, "C0", label="Wilson fit"); plt.plot(txy_fit["y1"], txy_fit["T"] - 273.15, "C0")
plt.plot(txy_uni["x1"], txy_uni["T"] - 273.15, "C1--", lw=1, label="UNIFAC prediction"); plt.plot(txy_uni["y1"], txy_uni["T"] - 273.15, "C1--", lw=1)
plt.xlabel("x, y methanol"); plt.ylabel("T (degC)"); plt.legend(); plt.show()
report = f'''# VLE of methanol (1) + water (2) at 101.3 kPa - laboratory data reduction

Data: {x.size} runs; {"; ".join(log)}; run {d.run[worst]} excluded (vapour sample mis-taken; residual in y {dy_all[worst]:+.3f}).
Consistency: Redlich-Kister area index {area_index(keep):.3f} after the exclusion (pass; {area_index(np.ones(x.size, bool)):.3f} with it).
Model: Wilson, Lambda12 = {L12:.3f} +/- {samples[:, 0].std():.3f}, Lambda21 = {L21:.3f} +/- {samples[:, 1].std():.3f};
rms deviations {np.sqrt(np.mean(dT[keep]**2)):.2f} K in T and {np.sqrt(np.mean(dy[keep]**2)):.3f} in y. No azeotrope.
Comparison: UNIFAC predicts the data within {np.max(np.abs(np.array([vle.bubble_T(comps, [xi, 1 - xi], 101325.0, g_uni)["T"] for xi in x[keep]]) - T[keep])):.1f} K.
'''
print(report)"""),
    md(r"""
A report that states what was measured, what was excluded and why, how consistent the data are, and how
uncertain the parameters are - so that a process designer can use the numbers with confidence, and a
reviewer can check every step from the raw file. (The file was generated for the course from UNIFAC with added
noise and faults; that is why the prediction agrees so well, and the outlier is exactly the one the file's
header confesses to.)
"""),
]

# =====================================================================================================
# Teaching layer
def _x(objectives, prereq, time, inside, exercises=(), implement=""):
    return dict(objectives=objectives, prereq=prereq, time=time, inside=inside, exercises=list(exercises),
                implement=implement)


EXTRAS["00_python_for_thermodynamics"] = _x(
    ["convert units safely and read property tables from files", "interpolate in tables (and why ln p)",
     "solve inverse problems with a root finder", "draw property diagrams"],
    "basic Python; no thermodynamics needed", "45 min",
    [md(r"""
## Inside the algorithm: linear interpolation by hand
`np.interp` finds the two table rows around the target and draws a straight line between them:
"""), code(r"""i = np.searchsorted(sat.T_K.to_numpy(), T)
T_lo, T_hi = sat.T_K[i - 1], sat.T_K[i]
f = (T - T_lo) / (T_hi - T_lo)
print(f"rows {i-1} and {i}: {T_lo:.2f} and {T_hi:.2f} K, fraction {f:.2f} -> h_g = {((1 - f) * sat.h_g[i - 1] + f * sat.h_g[i])/1e3:.2f} kJ/kg")""")],
    implement="**Implement it yourself:** write `Tsat_from_p(p, table)` that inverts the saturation table by interpolating "
              "T against ln p, and compare with `FluidTable.Tsat` for R134a at 2, 5 and 10 bar.")

EXTRAS["01_first_law_and_energy_balances"] = _x(
    ["apply the steady-flow energy balance to compressors, turbines, heat exchangers and valves",
     "use temperature-dependent heat capacities and mean heat capacities", "compute isentropic processes with variable Cp",
     "recognise irreversibility through isentropic efficiencies"],
    "notebook 00; the first law from an introductory course", "60 min",
    [md(r"""
## Inside the algorithm: the enthalpy integral
`enthalpy_change` integrates the heat-capacity polynomial analytically; check it against numerical integration:
"""), code(r"""num = integrate.quad(lambda T: idealgas.cp("CO2", T), 400.0, 1200.0)[0]
print(f"numerical {num/1e3:.4f} kJ/mol, analytical {idealgas.enthalpy_change('CO2', 400.0, 1200.0)/1e3:.4f} kJ/mol")""")],
    ["A gas turbine's exhaust (mostly air, 1.5 kg/s at 750 K) heats water from 20 to 90 degC in a heat-recovery exchanger while "
     "cooling to 400 K. How much water can be heated per second? (Liquid water: cp = 4.18 kJ/(kg K).)",
     "Three-stage compression of air from 1 to 27 bar with intercooling to 300 K: choose the intermediate pressures, compute the "
     "work, and compare with single-stage compression and with isothermal compression (RT ln(p2/p1))."],
    "**Implement it yourself:** write `isentropic_T(gas, T1, p1, p2)` yourself using `entropy_change` and `brentq`, "
    "and reproduce the compressor outlet temperature of the notebook.")

EXTRAS["02_steam_and_power_cycles"] = _x(
    ["locate states in the steam tables and on the T-s diagram", "analyse the Rankine cycle and its variants",
     "explain what limits the efficiency and how reheat and superheat help", "size the streams of a power plant"],
    "notebook 01", "60-75 min",
    [md(r"""
## Inside the algorithm: the isentropic turbine outlet
At the condenser pressure, the isentropic exit state has the inlet entropy. If that lies between the saturated
liquid and vapour entropies, the state is wet and the quality follows from the lever rule:
"""), code(r"""inlet = steam.properties(T=623.15, p=3e6)
sat = steam.saturated(p=75e3)
x = (inlet["s"] - sat["liquid"]["s"]) / sat["s_fg"]
h4 = sat["liquid"]["h"] + x * sat["h_fg"]
print(f"by hand: x = {x:.4f}, h4 = {h4/1e3:.1f} kJ/kg;  library: x = {rk['states']['4']['x']:.4f}, h4 = {rk['states']['4']['h']/1e3:.1f} kJ/kg")""")],
    ["Plot the thermal efficiency and the exit quality of the simple cycle (3 MPa, 350 degC) against the condenser pressure from "
     "5 to 100 kPa. Why do power plants go to such lengths to keep the condenser cold?",
     "For the 15 MPa / 600 degC cycle, vary the reheat pressure from 1 to 8 MPa. Where is the efficiency highest, and what "
     "happens to the exit quality?"],
    "**Implement it yourself:** compute the simple Rankine cycle's efficiency from the four state enthalpies using only "
    "`steam.two_phase`, `steam.properties`, `steam.state_ps` and `steam.state_ph`, and compare with `cycles.rankine`.")

EXTRAS["03_entropy_refrigeration_and_heat_pumps"] = _x(
    ["compute entropy generation and exergy destruction of processes", "use the Carnot limits for refrigerators and heat pumps",
     "analyse the vapour-compression cycle with property tables", "compare refrigerants and understand heat-pump performance"],
    "notebooks 01-02", "60-75 min",
    [md(r"""
## Inside the algorithm: the COP from table look-ups
The four enthalpies come straight from the tables: h1 saturated vapour at T_evap, h2 from (p_cond, s1), h3 saturated
liquid at T_cond, h4 = h3 (throttling):
"""), code(r"""h1 = r134.saturated(T=253.15)["vapour"]["h"]; s1 = r134.saturated(T=253.15)["vapour"]["s"]
p_c = r134.psat(313.15)
h2 = r134.superheated(p_c, s=s1)["h"]
h3 = r134.saturated(T=313.15)["liquid"]["h"]
print(f"by hand: COP = (h1 - h4)/(h2 - h1) = {(h1 - h3)/(h2 - h1):.3f};  library: {vc['COP_cooling']:.3f}")""")],
    ["A cold store must hold -25 degC with ammonia condensing at 35 degC. Compute the COP, the compressor discharge temperature and the "
     "pressure ratio, and compare with a two-stage arrangement (ideal intercooling at the geometric-mean pressure - treat it as two "
     "cycles in series with an intermediate temperature of 0 degC).",
     "For the propane heat pump, compare the seasonal energy use of radiators (condenser 55 degC) with underfloor heating "
     "(condenser 35 degC) for a house needing 12 000 kWh of heat per winter at an average outdoor temperature of 4 degC."],
    "**Implement it yourself:** compute the entropy generated in the real compressor of the R134a cycle (80 % efficiency) "
    "from the table states, and the exergy destroyed per kilogram of refrigerant.")

EXTRAS["04_real_gases_and_equations_of_state"] = _x(
    ["quantify real-gas behaviour with the compressibility factor", "use the virial and cubic equations of state",
     "interpret the three roots of a cubic equation", "compute gas storage densities at high pressure"],
    "notebook 01; the ideal-gas law", "60 min",
    [md(r"""
## Inside the algorithm: solving the cubic for Z
For the Peng-Robinson equation, with $A = ap/(RT)^2$ and $B = bp/RT$, the compressibility factor satisfies
$Z^3 - (1 - B)Z^2 + (A - 3B^2 - 2B)Z - (AB - B^2 - B^3) = 0$. `np.roots` finds all three:
"""), code(r"""prm = eos.parameters("pr", "CO2", 280.0)
A_, B_ = prm["a"] * 41.6e5 / (R * 280.0) ** 2, prm["b"] * 41.6e5 / (R * 280.0)
roots = np.roots([1, -(1 - B_), A_ - 3 * B_**2 - 2 * B_, -(A_ * B_ - B_**2 - B_**3)])
print("by hand:", np.sort(roots.real).round(4), " library:", np.round(eos.solve("pr", "CO2", 280.0, 41.6e5)["roots"], 4))""")],
    ["A pipeline carries CO2 at 300 K and 100 bar. Using the Peng-Robinson equation, how many tonnes are in 10 km of 0.5 m inside "
     "diameter? How wrong is the ideal-gas estimate? Compare the PR density with the reference data.",
     "Plot Z of methane at 200 K from 1 to 300 bar with the SRK and PR equations. At which pressure does Z pass through a minimum, "
     "and what does it mean physically that Z rises above 1 at very high pressure?"],
    "**Implement it yourself:** write the van der Waals equation from scratch ($a = 27R^2T_c^2/(64p_c)$, $b = RT_c/(8p_c)$), "
    "solve it for the molar volume of CO2 at 320 K and 50 bar with `brentq`, and compare with `eos.solve('vdw', ...)`.")

EXTRAS["05_departure_functions_and_fugacity"] = _x(
    ["compute real-gas enthalpy and entropy with departure functions", "analyse compressors and expanders handling real gases",
     "explain fugacity and the fugacity coefficient", "relate fugacity to compression work"],
    "notebooks 01 and 04", "60 min",
    [md(r"""
## Inside the algorithm: the enthalpy departure of the Peng-Robinson equation
$H^R/RT = Z - 1 + \left(\frac{d\ln\alpha}{d\ln T} - 1\right) q\,I$ with $q = a/(bRT)$ and
$I = \frac{1}{\sigma - \epsilon}\ln\frac{Z + \sigma B}{Z + \epsilon B}$ (Smith, Van Ness & Abbott, Eq. 6.66):
"""), code(r"""prm = eos.parameters("pr", "CO2", 320.0)
r = eos.solve("pr", "CO2", 320.0, 50e5)
Bq = prm["b"] * 50e5 / (R * 320.0); q = prm["a"] / (prm["b"] * R * 320.0)
I = np.log((r["Z"] + prm["sigma"] * Bq) / (r["Z"] + prm["eps"] * Bq)) / (prm["sigma"] - prm["eps"])
print(f"by hand: H_R = {R*320.0*(r['Z'] - 1 + (prm['dlnalpha_dlnT'] - 1)*q*I)/1e3:.4f} kJ/mol;  library: {r['H_R']/1e3:.4f} kJ/mol")""")],
    ["Methane is expanded in a turbo-expander from 60 bar and 250 K to 5 bar (isentropic efficiency 85 %). Find the outlet "
     "temperature and the work with the PR equation, and with the ideal-gas model. Which one predicts liquid formation? "
     "(Compare the outlet temperature with methane's saturation temperature at 5 bar from `vapor.antoine_Tsat`.)",
     "Compute the Joule-Thomson temperature drop of CO2 throttled from 60 bar and 320 K to 1 bar (constant enthalpy, PR "
     "departures plus ideal-gas enthalpy). Why is this used in some CO2 capture and liquefaction processes?"],
    "**Implement it yourself:** compute $\\ln\\phi$ of CO2 at 320 K and 50 bar by numerical integration of "
    "$\\int_0^p (Z - 1)\\,dp/p$ and compare with the PR fugacity coefficient.")

EXTRAS["06_vapour_pressure_and_phase_change"] = _x(
    ["derive and use the Clausius-Clapeyron equation", "apply the Antoine equation within its range",
     "compute saturation properties from an equation of state", "choose storage conditions from the vapour-pressure curve"],
    "notebooks 04-05", "45-60 min",
    [md(r"""
## Inside the algorithm: equal fugacities
At the saturation pressure the liquid and vapour roots of the cubic have the same fugacity. Scan the pressure
and find where $\ln\phi_L - \ln\phi_V$ changes sign:
"""), code(r"""T = 280.0
pp = np.linspace(35e5, 50e5, 301)
diff = [eos.solve("pr", "CO2", T, p, "liquid")["lnphi"] - eos.solve("pr", "CO2", T, p, "vapor")["lnphi"] for p in pp]
i = np.nonzero(np.diff(np.sign(diff)))[0][0]
print(f"sign change between {pp[i]/1e5:.2f} and {pp[i+1]/1e5:.2f} bar;  library: {vapor.eos_psat('pr', 'CO2', T)['p_sat']/1e5:.3f} bar")""")],
    ["Trouton's rule states that the molar entropy of vaporisation at the normal boiling point is about 85-88 J/(mol K) "
     "for most liquids. Test it with the Antoine-implied heats of vaporisation of benzene, toluene, n-hexane, acetone, "
     "ethanol and water. Which substances break the rule, and why?",
     "Ammonia is stored as a liquid. Compare the wall pressure for a refrigerated tank at -33 degC with a pressurised tank at "
     "40 degC. Using the enthalpy of vaporisation, estimate the boil-off rate (kg/h) of the refrigerated tank for a heat leak of 5 kW."],
    "**Implement it yourself:** fit your own Antoine constants to the vapour pressure of benzene computed with the PR equation "
    "between 300 and 500 K (use `scipy.optimize.least_squares` on ln p), and compare with the stored constants.")

EXTRAS["07_mixture_fundamentals"] = _x(
    ["distinguish excess properties from ideal mixing", "compute partial molar properties by the tangent-intercept method",
     "explain the entropy and Gibbs energy of ideal mixing", "apply the Gibbs-Duhem equation"],
    "notebooks 01 and 06; calculus", "45-60 min",
    [md(r"""
## Inside the algorithm: partial molar properties by differentiating with respect to mole numbers
The definition is $\bar V_i = (\partial(nV)/\partial n_i)_{T,p,n_j}$. Numerically: add a tiny amount of component $i$ to the mixture
and watch the total volume change.
"""), code(r"""nV = lambda n: (n[0] + n[1]) * Vf(n[0] / (n[0] + n[1]))
print(f"by mole-number derivative: V1_bar = {mixtures.partial_molar(nV, [0.3, 0.7], 0):.4f}, V2_bar = {mixtures.partial_molar(nV, [0.3, 0.7], 1):.4f}")
print(f"by tangent intercepts:     V1_bar = {b1:.4f}, V2_bar = {b2:.4f}")""")],
    ["How much volume is 'lost' when 1 L of ethanol is mixed with 1 L of water at 25 degC? (Densities: ethanol 0.785 g/mL, water "
     "0.997 g/mL; use the data file.)",
     "The minimum work to separate 1 mol of an ideal equimolar mixture at 25 degC equals $-\\Delta G_{mix}$. Compare it with the "
     "heat of vaporisation of benzene (about 31 kJ/mol) to see why distillation uses far more energy than the thermodynamic minimum."],
    "**Implement it yourself:** fit the excess volume $V^E = x_1 x_2 (A + B(2x_1 - 1) + C(2x_1 - 1)^2)$ (Redlich-Kister) to the "
    "data with `np.linalg.lstsq`, and use it to compute the partial molar volumes at infinite dilution "
    "($\\bar V_1^\\infty = V_1 + A - B + C$).")

EXTRAS["08_raoults_law_and_flash"] = _x(
    ["compute bubble and dew points with Raoult's law", "read and construct T-x-y and x-y diagrams",
     "perform isothermal flash calculations with the Rachford-Rice equation", "use relative volatility to judge separations"],
    "notebooks 06-07", "60 min",
    [md(r"""
## Inside the algorithm: the Rachford-Rice equation
With K-values $K_i = p_i^{sat}/p$, the vapour fraction $V$ solves $\sum_i z_i(K_i - 1)/(1 + V(K_i - 1)) = 0$ - a
monotonic function of $V$, so `brentq` on [0, 1] cannot miss it:
"""), code(r"""from scipy import optimize
K = np.array([vapor.antoine_psat(c, T) for c in feed]) / p
f = lambda V: np.sum(np.array(z) * (K - 1) / (1 + V * (K - 1)))
V_hand = optimize.brentq(f, 0, 1)
print(f"K-values {np.round(K, 3)}; V by hand {V_hand:.4f}, library {fl['V']:.4f}")""")],
    ["Draw the P-x-y diagram of benzene-toluene at 90 degC. Read off the bubble and dew pressures of an equimolar liquid.",
     "A liquefied petroleum gas is 40 % propane and 60 % n-butane (mole). At 20 degC, what is the pressure in a storage vessel, "
     "and what is the composition of the vapour above the liquid? How does the vapour composition change as the vessel empties "
     "(assume the liquid becomes progressively richer in butane)?"],
    "**Implement it yourself:** write a bubble-temperature solver from scratch (Raoult's law, Antoine vapour pressures, "
    "`brentq` on the sum of $y_i$ minus 1) and reproduce the T-x-y diagram.")

EXTRAS["09_activity_coefficient_models"] = _x(
    ["extract activity coefficients from VLE data", "fit Margules, van Laar, Wilson and NRTL models",
     "test data for thermodynamic consistency", "locate azeotropes and understand their consequences for distillation"],
    "notebooks 07-08", "75 min",
    [md(r"""
## Inside the algorithm: the two-parameter Margules equations
$\ln\gamma_1 = x_2^2[A_{12} + 2(A_{21} - A_{12})x_1]$, $\ln\gamma_2 = x_1^2[A_{21} + 2(A_{12} - A_{21})x_2]$:
"""), code(r"""A12, A21 = fits["margules"]["params"]["A12"], fits["margules"]["params"]["A21"]
x1_ = 0.3
by_hand = np.exp([0.7**2 * (A12 + 2 * (A21 - A12) * x1_), x1_**2 * (A21 + 2 * (A12 - A21) * 0.7)])
print("by hand:", by_hand.round(5), " library:", fits["margules"]["gamma"]([x1_, 0.7]).round(5))""")],
    ["Fit the Wilson model to only the data with x_ethanol < 0.5. How well does it predict the azeotrope? What does this say about "
     "extrapolating activity models beyond the data range?",
     "Using the fitted NRTL model, compute the minimum reflux-related quantity that matters most for a designer: the relative "
     "volatility $\\alpha = y_1 x_2/(x_1 y_2)$ as a function of x_ethanol at 1 atm. Where does it fall below 1.1?"],
    "**Implement it yourself:** write a bubble-temperature solver with modified Raoult's law (activity coefficients from "
    "`activity.margules`) and reproduce the fitted T-x-y curve.")

EXTRAS["10_unifac_prediction"] = _x(
    ["describe molecules as UNIFAC groups", "predict activity coefficients without binary data",
     "judge UNIFAC's accuracy against data", "use infinite-dilution activity coefficients for solvent screening"],
    "notebook 09", "60 min",
    [md(r"""
## Inside the algorithm: the combinatorial part
UNIFAC's combinatorial term depends only on sizes and shapes: $\ln\gamma_i^C = \ln(\Phi_i/x_i) + 5q_i\ln(\theta_i/\Phi_i) + l_i - (\Phi_i/x_i)\sum_j x_jl_j$,
with $r_i = \sum\nu_kR_k$ and $q_i = \sum\nu_kQ_k$ from the group tables:
"""), code(r"""from engthermo._unifac_data import SUBGROUPS
def r_q(mol):
    ids = unifac.groups(mol)
    return sum(n * SUBGROUPS[k][3] for k, n in ids.items()), sum(n * SUBGROUPS[k][4] for k, n in ids.items())
r1, q1 = r_q("ethanol"); r2, q2 = r_q("water")
x1_, x2_ = 0.4, 0.6
r_, q_ = np.array([r1, r2]), np.array([q1, q2]); x_ = np.array([x1_, x2_])
Phi, Th = r_ * x_ / (r_ @ x_), q_ * x_ / (q_ @ x_)
l_ = 5 * (r_ - q_) - (r_ - 1)
lnC = np.log(Phi / x_) + 5 * q_ * np.log(Th / Phi) + l_ - Phi / x_ * (x_ @ l_)
print(f"ethanol: r = {r1:.4f}, q = {q1:.3f}; water: r = {r2:.4f}, q = {q2:.3f}; combinatorial gammas at x = 0.4: {np.exp(lnC).round(4)}")""")],
    ["Predict the T-x-y diagram of acetone-methanol at 1 atm with UNIFAC and locate its azeotrope. (Experimental: about 0.80 mole "
     "fraction acetone at 55.7 degC.)",
     "Use UNIFAC to rank benzene, toluene, n-hexane and 1-butanol as solvents for extracting ethanol from water at 25 degC, and "
     "comment on the trade-off between affinity and the solvent's own solubility in water."],
    "**Implement it yourself:** compute the infinite-dilution activity coefficient of ethanol in water at 351 K by evaluating "
    "UNIFAC at x = 1e-6, and compare with the value extracted from the data file's first points.")

EXTRAS["11_high_pressure_phase_equilibria"] = _x(
    ["apply the phi-phi formulation with an equation of state for both phases", "compute bubble and dew points of gas mixtures",
     "understand and use binary interaction parameters", "solve gas-conditioning and CO2-mixture problems"],
    "notebooks 04-05 and 08", "60-75 min",
    [md(r"""
## Inside the algorithm: successive substitution
Guess p and y; compute the fugacity coefficients of both phases; update $K_i = \hat\phi_i^L/\hat\phi_i^V$, $y_i = K_ix_i/\sum K_ix_i$ and
$p \leftarrow p\sum K_ix_i$; repeat until $\sum K_ix_i = 1$:
"""), code(r"""x_ = np.array([0.5, 0.5]); T_ = 200.0
p_, y_ = 20e5, x_.copy()
for it in range(60):
    liq = eos.mixture("pr", comps, x_, T_, p_, phase="liquid"); vap = eos.mixture("pr", comps, y_, T_, p_, phase="vapor")
    K = liq["phi_i"] / vap["phi_i"]; s = np.sum(K * x_)
    y_, p_ = K * x_ / s, p_ * s
    if abs(s - 1) < 1e-10:
        break
print(f"by hand: p = {p_/1e5:.4f} bar, y1 = {y_[0]:.5f} after {it+1} iterations;  library: {vle.eos_bubble_P('pr', comps, x_, T_)['p']/1e5:.4f} bar")""")],
    ["A CO2 capture unit compresses a gas of 90 % CO2 and 10 % N2 at 30 degC. Find the bubble and dew pressures of this mixture "
     "with the PR equation (try k12 = 0 and k12 = -0.02) - can it be liquefied at 30 degC at all? Compare with pure CO2.",
     "For the natural gas of the notebook, plot the dew-point pressure against temperature from 230 to 300 K (the 'dew-point curve'). "
     "A pipeline operates at 70 bar: down to what temperature does the gas stay dry?"],
    "**Implement it yourself:** write the dew-pressure counterpart of the successive-substitution loop above and check it "
    "against `vle.eos_dew_P` for the natural gas at 270 K.")

EXTRAS["12_liquid_liquid_equilibria_and_polymers"] = _x(
    ["explain liquid-liquid splitting from the Gibbs energy of mixing", "compute binary LLE from activity coefficients",
     "estimate extraction distribution coefficients", "use Flory-Huggins theory for polymer solutions and their phase diagrams"],
    "notebooks 09-10", "60-75 min",
    [md(r"""
## Inside the algorithm: the isoactivity equations
The two phases have equal activities of both components: $x_1^\alpha\gamma_1^\alpha = x_1^\beta\gamma_1^\beta$ and likewise for component 2.
Solve the two equations with `scipy.optimize.root`:
"""), code(r"""from scipy import optimize
g = activity.margules(2.5)
def eqs(v):
    a, b = v
    ga, gb = g([a, 1 - a]), g([b, 1 - b])
    return [np.log(a * ga[0]) - np.log(b * gb[0]), np.log((1 - a) * ga[1]) - np.log((1 - b) * gb[1])]
sol = optimize.root(eqs, [0.1, 0.9])
print("by hand:", np.round(sorted(sol.x), 5), " library:", [round(v, 5) for v in lle.binary(g, 300.0).values() if isinstance(v, float)][:2])""")],
    ["Hexane-ethanol: with UNIFAC, find the temperature below which the liquid splits (the upper critical solution temperature). "
     "Scan `lle.binary` from 60 degC downwards.",
     "For the illustrative polymer solution (chi = 0.20 + 90/T), plot the cloud-point (UCST) temperature against molar mass for "
     "N = 50 to 50 000 on a log axis. What does this imply for polymer fractionation by cooling?"],
    "**Implement it yourself:** derive the Flory-Huggins spinodal condition $\\partial^2(\\Delta G/RT)/\\partial\\phi^2 = 0$ by "
    "differentiating the Gibbs energy numerically, and reproduce `lle.flory_huggins_spinodal(0.7, 100)`.")

EXTRAS["13_solid_liquid_equilibria_and_alloys"] = _x(
    ["compute ideal and non-ideal solubilities of solids", "explain freezing-point depression and eutectics",
     "construct eutectic and lens (isomorphous) phase diagrams from fusion data", "connect phase diagrams to alloys and heat-transfer fluids"],
    "notebooks 09-10", "60 min",
    [md(r"""
## Inside the algorithm: the eutectic as the intersection of two liquidus curves
Each liquidus is the temperature at which the melt is saturated with one solid; the eutectic is where both are:
"""), code(r"""from scipy import optimize
R_ = 8.314462618
T_liq = lambda x, Tm, dH: 1 / (1 / Tm - R_ * np.log(x) / dH)              # ideal liquidus (x = mole fraction of the crystallising metal)
x_e = optimize.brentq(lambda xBi: T_liq(xBi, Tm_Bi, dH_Bi) - T_liq(1 - xBi, Tm_Cd, dH_Cd), 1e-6, 1 - 1e-6)
print(f"by hand: x_Bi = {x_e:.4f}, T = {T_liq(x_e, Tm_Bi, dH_Bi) - 273.15:.1f} degC;  library: x_Bi = {cur['eutectic']['x1']:.4f}, {Te:.1f} degC")""")],
    ["Tin (Tm 505 K, 7.03 kJ/mol) and lead (Tm 600.6 K, 4.77 kJ/mol) form the classic solder eutectic. Predict it with ideal liquids "
     "and compare with the experimental 183 degC at 0.74 mole fraction Sn. What does the discrepancy tell you about the Sn-Pb liquid?",
     "Solar salt is 60/40 by mass NaNO3 (Tm 307 degC, 15.0 kJ/mol) / KNO3 (Tm 334 degC, 10.0 kJ/mol). Predict the eutectic of the ideal "
     "melt and compare with the observed 222 degC. Why does a low melting point matter for a thermal-storage medium?"],
    "**Implement it yourself:** compute the Cu-Ni liquidus and solidus at 1500 K from the two equilibrium constants by hand and "
    "compare with `sle.lens_diagram`.")

EXTRAS["14_chemical_reaction_equilibrium"] = _x(
    ["compute reaction enthalpies, Gibbs energies and equilibrium constants as functions of temperature",
     "find equilibrium conversions of single reactions and the effects of T, p and inerts",
     "solve multi-reaction equilibria by Gibbs minimisation", "connect equilibrium limits to process conditions"],
    "notebooks 01 and 07", "75 min",
    [md(r"""
## Inside the algorithm: Delta G°(T) from formation data
$\Delta H^\circ(T) = \Delta H^\circ_{298} + \int C_p\,dT$, $\Delta S^\circ(T) = (\Delta H^\circ_{298} - \Delta G^\circ_{298})/298.15 + \int C_p/T\,dT$, then $\Delta G^\circ = \Delta H^\circ - T\Delta S^\circ$:
"""), code(r"""from engthermo import components, idealgas
T_ = 700.0
dH298 = sum(nu * components.get(s).Hf for s, nu in rxn.items()); dG298 = sum(nu * components.get(s).Gf for s, nu in rxn.items())
dH = dH298 + sum(nu * idealgas.enthalpy_change(s, 298.15, T_) for s, nu in rxn.items())
dS = (dH298 - dG298) / 298.15 + sum(nu * idealgas.entropy_change(s, 298.15, T_) for s, nu in rxn.items())
print(f"by hand: K(700 K) = {np.exp(-(dH - T_ * dS) / (R * T_)):.4e};  library: {reaction.equilibrium_constant(rxn, T_):.4e}")""")],
    ["Methanol synthesis (CO + 2 H2 -> CH3OH) is run at 250 degC and 50-100 bar. Compute the equilibrium methanol mole fraction for a "
     "stoichiometric feed at 50 and 100 bar, and at 300 degC. Why is the process run with a large recycle?",
     "Use Gibbs minimisation to find the equilibrium of the reformer at a steam-to-carbon ratio of 2 and of 4 (900 degC, 20 bar). "
     "How does the ratio affect methane conversion and the CO/CO2 split? (Carbon formation is ignored here.)"],
    "**Implement it yourself:** solve the ammonia equilibrium extent at 450 degC and 200 bar by writing the "
    "equation $K = y_{NH_3}^2/(y_{N_2}y_{H_2}^3)\\,(p/p^\\circ)^{-2}$ in terms of the extent and using `brentq`; compare with `reaction.extent`.")

EXTRAS["15_high_temperature_thermochemistry"] = _x(
    ["compute heating values and CO2 intensities of fuels", "calculate adiabatic flame temperatures with variable heat capacities",
     "read and use an Ellingham diagram", "explain carbothermic and hydrogen reduction of metal oxides"],
    "notebooks 01 and 14", "60-75 min",
    [md(r"""
## Inside the algorithm: the flame-temperature energy balance
The products' sensible enthalpy above 25 degC must equal the lower heating value; solve for T:
"""), code(r"""from scipy import optimize
from engthermo import idealgas
prod = combustion.flue_gas("CH4"); q = combustion.heating_values("CH4")["LHV"]
f = lambda T_: sum(n * idealgas.enthalpy_change(s, 298.15, T_) for s, n in prod.items() if n > 0) - q
print(f"by hand: T_ad = {optimize.brentq(f, 298.15, 4000):.1f} K;  library: {combustion.adiabatic_flame_temperature('CH4')['T_ad']:.1f} K")""")],
    ["A gas turbine burns methane with 200 % excess air and reactants preheated to 700 K. Find the flame temperature, and the "
     "excess air needed to keep it at 1800 K (a typical turbine-inlet limit).",
     "From the Ellingham diagram, at what temperature does the C/CO line cross the Si/SiO2 line? Compare with the operating "
     "temperature of a silicon smelting furnace (about 2000 K). Why is aluminium produced electrolytically instead?"],
    "**Implement it yourself:** compute the CO2 emitted per MJ of heat for a coal approximated as pure carbon (LHV from the "
    "formation enthalpy of CO2), and compare with methane. What is the ratio?")

EXTRAS["16_electrochemical_thermodynamics"] = _x(
    ["relate Gibbs energies to cell voltages", "apply the Nernst equation and the temperature coefficient",
     "compute fuel-cell and electrolyser efficiencies and hydrogen electricity consumption", "estimate theoretical battery energy densities"],
    "notebooks 14-15", "60 min",
    [md(r"""
## Inside the algorithm: Faraday's law
Two electrons per hydrogen molecule; the electrical work per mole is $nFE$, and it cannot exceed $-\Delta G$:
"""), code(r"""dG = reaction.standard_gibbs(h2)
print(f"E° = -Delta G / (n F) = {-dG / (2 * F):.4f} V;  library: {electrochem.standard_potential(h2, 2):.4f} V")
print(f"electrical work per mol H2 at E°: {2 * F * (-dG / (2 * F)) / 1e3:.1f} kJ = -Delta G")""")],
    ["A solid-oxide fuel cell at 800 degC runs on hydrogen and produces steam. Compute its reversible voltage and maximum "
     "efficiency, and compare with a 25 degC cell. Which loses more of the heating value as reversible heat?",
     "Round trip: an electrolyser at 1.85 V feeds hydrogen to a fuel cell at 0.72 V. What fraction of the electricity returns? "
     "How does this compare with a pumped-hydro or battery storage round trip of 75-90 %?"],
    "**Implement it yourself:** write the Nernst equation for the hydrogen cell as a function of p_H2, p_O2 and T, and plot E "
    "against temperature from 25 to 200 degC at 1 bar (use the steam reaction above 100 degC).")

EXTRAS["17_from_messy_vle_data_to_a_model"] = _x(
    ["repair a real laboratory export (units, duplicates, typos, missing values) and log every step",
     "test VLE data for consistency and detect outliers", "fit an activity model with uncertainties",
     "compare with a prediction and write a reproducible report"],
    "notebooks 09-10", "75-90 min",
    [md(r"""
## Inside the algorithm: activity coefficients from one data point
Modified Raoult's law solved for the activity coefficients, $\gamma_i = y_i p/(x_i p_i^{sat})$, with the Antoine vapour pressures at
the measured temperature:
"""), code(r"""i = 5
Ti = T[i]
print(f"run {d.run[i]}: x = {x[i]:.3f}, y = {y[i]:.3f}, T = {Ti-273.15:.1f} degC: "
      f"gamma_methanol = {y[i]*p[i]/(x[i]*vapor.antoine_psat('methanol', Ti)):.3f}, gamma_water = {(1-y[i])*p[i]/((1-x[i])*vapor.antoine_psat('water', Ti)):.3f}")""")],
    ["Repeat the fit with the NRTL model. Do both models give the same limiting activity coefficients within the bootstrap "
     "uncertainty? Which parameter is least well determined, and which data would improve it?",
     "Suppose the barometer had been misread and the true pressure was 99.5 kPa throughout. Redo the analysis: how much do the "
     "fitted parameters change, and does the area test still pass? What does this say about recording the pressure?"],
    "**Implement it yourself:** turn steps 1-2 into a function `load_lab_vle(path)` returning a clean DataFrame in SI units and a "
    "list of the repairs made; test it on the file.")


def apply_extras(name, cells):
    ex = EXTRAS[name]
    header = md("**What you will learn**\n" + "\n".join(f"- {o}" for o in ex["objectives"])
                + f"\n\n**Before you start:** {ex['prereq']}  ·  **Time:** about {ex['time']}")
    cells = [cells[0], header] + list(cells[1:])
    idx = [i for i, c in enumerate(cells) if c.cell_type == "markdown" and "## Exercises" in c.source]
    if idx:
        i = idx[0]
        n_items = len(re.findall(r"^\d+\. ", cells[i].source, flags=re.M))
        cells[i] = md(cells[i].source.rstrip() + f"\n{n_items + 1}. {ex['implement']}")
        return cells[:i] + list(ex["inside"]) + cells[i:]
    items = ex["exercises"] + [ex["implement"]]
    exercises = md("## Exercises\n" + "\n".join(f"{k}. {t}" for k, t in enumerate(items, 1)))
    return cells + list(ex["inside"]) + [exercises]


def write_all(run: bool = True, only=()):
    import nbclient

    for name in sorted(NOTEBOOKS):
        if only and not name.startswith(tuple(only)):
            continue
        cells = apply_extras(name, NOTEBOOKS[name])
        nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                                "language": "python"},
                                                 "language_info": {"name": "python"}})
        if run:
            nbclient.NotebookClient(nb, timeout=900, kernel_name="python3",
                                    resources={"metadata": {"path": str(HERE)}}).execute()
        nbformat.write(nb, HERE / f"{name}.ipynb")
        print("wrote", name)


if __name__ == "__main__":
    write_all(run="--no-run" not in sys.argv, only=[a for a in sys.argv[1:] if not a.startswith("--")])
