"""Build the worked solutions to every exercise of the course, as executed notebooks.

The exercise texts are taken from ../notebooks/build_notebooks.py, so questions and solutions cannot
drift apart; the build fails if a notebook's number of solutions differs from its number of exercises.

    python solutions/build_solutions.py            # build and execute all solution notebooks
    python solutions/build_solutions.py 05 12      # only those whose names start with 05 or 12
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("course", HERE.parent / "notebooks" / "build_notebooks.py")
course = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(course)
md, code = course.md, course.code
SOLUTIONS: dict[str, dict] = {}


def solutions(name, setup="", imports=()):
    """Register the solutions of one course notebook: a function returning one list of cells per exercise."""
    def deco(fn):
        SOLUTIONS[name] = {"setup": setup, "imports": list(imports), "answers": fn()}
        return fn
    return deco


def exercises(name) -> list[str]:
    cells = course.apply_extras(name, list(course.NOTEBOOKS[name]))
    text = [c.source for c in cells if c.cell_type == "markdown" and "## Exercises" in c.source][0]
    items = re.split(r"\n(?=\d+\. )", text.split("## Exercises")[1].strip())
    return [re.sub(r"^\d+\.\s*", "", it).strip() for it in items]


def notebook_cells(name) -> list:
    title = re.sub(r"^#\s*", "", course.NOTEBOOKS[name][0].source.splitlines()[0])
    entry = SOLUTIONS[name]
    ex = exercises(name)
    if len(ex) != len(entry["answers"]):
        raise ValueError(f"{name}: {len(ex)} exercises but {len(entry['answers'])} solutions")
    cells = [md(f"""
# Solutions · {title}

Worked solutions to the exercises of [notebook {name[:2]}](../notebooks/{name}.ipynb). Try each exercise
yourself before reading its solution - the learning happens in the attempt. Solutions are one way to
answer each question; other correct approaches exist.
"""), course.setup_cell(entry["imports"])] + ([code(entry["setup"])] if entry["setup"] else [])
    for i, (text, answer) in enumerate(zip(ex, entry["answers"]), 1):
        cells.append(md(f"## Exercise {i}\n\n{text}"))
        cells.extend(answer)
    return cells


def write_all(only=()):
    import nbclient

    for name in SOLUTIONS:
        if only and not name.startswith(tuple(only)):
            continue
        nb = new_notebook(cells=notebook_cells(name),
                          metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                    "language_info": {"name": "python"}})
        nbclient.NotebookClient(nb, timeout=900, kernel_name="python3",
                                resources={"metadata": {"path": str(HERE)}}).execute()
        path = HERE / f"{name[:2]}_solutions.ipynb"
        nbformat.write(nb, path)
        print("wrote", path.name)




# =====================================================================================================
@solutions("00_python_for_thermodynamics", imports=["from engthermo import datasets, steam, tables"])
def _():
    return [
        [code(r"""Ts = np.linspace(280, 647, 150)
sats = [steam.saturated(T=T) for T in Ts]
plt.loglog([s["liquid"]["v"] for s in sats], [s["p"] / 1e5 for s in sats], "b", label="saturated liquid")
plt.loglog([s["vapour"]["v"] for s in sats], [s["p"] / 1e5 for s in sats], "r", label="saturated vapour")
for T_C in (100, 300, 500, 700):
    pp = np.logspace(3, np.log10(100e6), 200)
    vv = [steam.state(T=T_C + 273.15, p=p)["v"] if not (T_C + 273.15 <= 647 and abs(p - steam.psat(T_C + 273.15)) < 1) else np.nan for p in pp]
    plt.loglog(vv, pp / 1e5, "k--", lw=0.8); plt.text(vv[-1], pp[-1] / 1e5 * 1.3, f"{T_C} degC", fontsize=8)
plt.xlabel("v (m3/kg)"); plt.ylabel("p (bar)"); plt.title("p-v diagram of water"); plt.legend(); plt.show()"""),
         md(r"""
On log axes the dome and the isotherms are all visible over six decades of volume. Isotherms below the
critical temperature cross the dome horizontally (the phase change at constant pressure); the 700 degC
isotherm passes smoothly above it, never condensing however high the pressure.
""")],
        [code(r"""sat = pd.read_csv(datasets.path("ammonia_saturation.csv"), comment="#")
T = -12 + 273.15
h_f, h_g = np.interp(T, sat.T_K, sat.h_f), np.interp(T, sat.T_K, sat.h_g)
p = np.exp(np.interp(T, sat.T_K, np.log(sat.p_Pa)))
lib = tables.FluidTable("ammonia").saturated(T=T)
print(f"by hand: h_f = {h_f/1e3:.1f}, h_g = {h_g/1e3:.1f} kJ/kg, p = {p/1e5:.3f} bar")
print(f"library: h_f = {lib['liquid']['h']/1e3:.1f}, h_g = {lib['vapour']['h']/1e3:.1f} kJ/kg, p = {lib['p']/1e5:.3f} bar")"""),
         md(r"""
Identical: `FluidTable` does the same linear interpolation in temperature (and in the logarithm of the
pressure) that a table user does by hand. Ammonia's enthalpy of vaporisation is about 1.3 MJ/kg - six times
that of R134a - which is why ammonia plants need so little refrigerant mass.
""")],
        [code(r"""def Tsat_from_p(p, table):
    return float(np.interp(np.log(p), np.log(table.p_Pa), table.T_K))
r134 = pd.read_csv(datasets.path("R134a_saturation.csv"), comment="#")
ft = tables.FluidTable("R134a")
for p_bar in (2, 5, 10):
    print(f"{p_bar:2d} bar: by hand {Tsat_from_p(p_bar * 1e5, r134) - 273.15:.2f} degC, library {ft.Tsat(p_bar * 1e5) - 273.15:.2f} degC")"""),
         md(r"""
Inverting the table means interpolating temperature against ln p - the same straight line read the other
way. The two agree exactly because the library uses the same rule.
""")],
    ]


# =====================================================================================================
@solutions("01_first_law_and_energy_balances", imports=["from scipy import optimize", "from engthermo import idealgas"])
def _():
    return [
        [code(r"""m_gas = 1.5
air = idealgas.Mixture({"N2": 0.79, "O2": 0.21})
Q = m_gas / air.M * air.enthalpy_change(400.0, 750.0)                 # J/s given up by the exhaust
m_water = Q / (4180 * 70)
print(f"heat recovered {Q/1e3:.1f} kW -> {m_water:.2f} kg/s of water heated from 20 to 90 degC")"""),
         md(r"""
The exhaust's enthalpy change between 750 and 400 K (mean heat capacity about 30 J/(mol K)) heats nearly two
kilograms of water per second by 70 K. Heat-recovery steam generators and district-heating exchangers on gas
turbines run exactly this balance.
""")],
        [code(r"""T1, p1, p2 = 300.0, 1e5, 27e5
r = (p2 / p1) ** (1 / 3)                                              # equal pressure ratios per stage
stages = [(p1, p1 * r), (p1 * r, p1 * r**2), (p1 * r**2, p2)]
w3 = sum(idealgas.enthalpy_change("air", T1, idealgas.isentropic_T("air", T1, a, b)) for a, b in stages)
w1 = idealgas.enthalpy_change("air", T1, idealgas.isentropic_T("air", T1, p1, p2))
w_iso = 8.314462618 * T1 * np.log(p2 / p1)
print(f"intermediate pressures {p1*r/1e5:.2f} and {p1*r**2/1e5:.2f} bar")
print(f"work: single stage {w1/1e3:.2f}, three stages {w3/1e3:.2f}, isothermal {w_iso/1e3:.2f} kJ/mol")"""),
         md(r"""
Three stages with intercooling save nearly 30 % of the single-stage work and come within about 17 % of the
isothermal minimum. Each intercooler also lowers the outlet temperature, which matters for the mechanical
design of the compressor and for compressed-air storage.
""")],
        [code(r"""def isentropic_T(gas, T1, p1, p2):
    return optimize.brentq(lambda T: idealgas.entropy_change(gas, T1, T, p1, p2), T1 if p2 > p1 else 150.0, 4000.0 if p2 > p1 else T1)
print(f"by hand: {isentropic_T('air', 300.0, 1e5, 1e6):.2f} K;  library: {idealgas.isentropic_T('air', 300.0, 1e5, 1e6):.2f} K")"""),
         md(r"""
The isentropic condition is one equation in one unknown - the entropy change from the inlet state must be
zero - and a bracketed root finder solves it for any heat-capacity function.
""")],
    ]


# =====================================================================================================
@solutions("02_steam_and_power_cycles", imports=["from engthermo import cycles, steam"])
def _():
    return [
        [code(r"""p_cond = np.linspace(5e3, 100e3, 40)
res = [cycles.rankine(3e6, p, T_inlet=623.15) for p in p_cond]
fig, a1 = plt.subplots()
a1.plot(p_cond / 1e3, [r["efficiency"] for r in res], "C0"); a1.set(xlabel="condenser pressure (kPa)", ylabel="efficiency", title="simple Rankine, 3 MPa / 350 degC")
a2 = a1.twinx(); a2.plot(p_cond / 1e3, [r["exit_quality"] for r in res], "C1"); a2.set_ylabel("exit quality", color="C1"); plt.show()
print(f"5 kPa: efficiency {res[0]['efficiency']:.1%}, quality {res[0]['exit_quality']:.3f}; 100 kPa: {res[-1]['efficiency']:.1%}, {res[-1]['exit_quality']:.3f}")"""),
         md(r"""
Every kilopascal of condenser pressure counts: from 100 to 5 kPa the efficiency rises by about eleven points -
the biggest single lever in the cycle - because the saturation temperature of the condensing steam falls from
100 to 33 degC. The cost is a wetter turbine exhaust (and the cooling water to reach such a low temperature),
which is why plants add superheat and reheat.
""")],
        [code(r"""p_rh = np.linspace(1e6, 8e6, 29)
res = [cycles.rankine(15e6, 10e3, T_inlet=873.15, reheat=(p, 873.15)) for p in p_rh]
fig, a1 = plt.subplots()
a1.plot(p_rh / 1e6, [r["efficiency"] for r in res], "C0"); a1.set(xlabel="reheat pressure (MPa)", ylabel="efficiency")
a2 = a1.twinx(); a2.plot(p_rh / 1e6, [r["exit_quality"] for r in res], "C1"); a2.set_ylabel("exit quality", color="C1"); plt.show()
i = int(np.argmax([r["efficiency"] for r in res]))
print(f"highest efficiency {res[i]['efficiency']:.2%} at a reheat pressure of {p_rh[i]/1e6:.1f} MPa (exit quality {res[i]['exit_quality']:.3f})")"""),
         md(r"""
The efficiency is a flat maximum near a fifth of the boiler pressure (the usual rule of thumb says a quarter;
the maximum is so flat that either is right), while the exit quality keeps improving as the reheat pressure
rises. Designers pick the reheat pressure to satisfy
the moisture limit, and the efficiency barely notices.
""")],
        [code(r"""s1 = steam.two_phase(0.0, p=75e3)
s2 = steam.state_ph(3e6, s1["h"] + (steam.state_ps(3e6, s1["s"])["h"] - s1["h"]))
s3 = steam.properties(623.15, 3e6)
s4 = steam.state_ps(75e3, s3["s"])
eta = ((s3["h"] - s4["h"]) - (s2["h"] - s1["h"])) / (s3["h"] - s2["h"])
print(f"by hand: efficiency {eta:.4f};  library: {cycles.rankine(3e6, 75e3, T_inlet=623.15)['efficiency']:.4f}")"""),
         md(r"""
Four state functions and one formula reproduce the library exactly - the cycle calculation is nothing more
than the steam tables applied at four points.
""")],
    ]


# =====================================================================================================
@solutions("03_entropy_refrigeration_and_heat_pumps", imports=["from engthermo import cycles, tables"])
def _():
    return [
        [code(r"""one = cycles.vapor_compression("ammonia", 248.15, 308.15)
lo = cycles.vapor_compression("ammonia", 248.15, 273.15)              # low stage: -25 -> 0 degC
hi = cycles.vapor_compression("ammonia", 273.15, 308.15)              # high stage: 0 -> 35 degC
# two cycles in series: the high stage must absorb the low stage's condenser heat
w_two = lo["w_compressor"] + hi["w_compressor"] * lo["q_condenser"] / hi["q_evaporator"]
print(f"single stage: COP {one['COP_cooling']:.2f}, discharge {one['states']['2']['T']-273.15:.0f} degC, pressure ratio {one['p_high']/one['p_low']:.1f}")
print(f"two stages:   COP {lo['q_evaporator']/w_two:.2f}, discharge temperatures {lo['states']['2']['T']-273.15:.0f} and {hi['states']['2']['T']-273.15:.0f} degC, "
      f"pressure ratios {lo['p_high']/lo['p_low']:.1f} and {hi['p_high']/hi['p_low']:.1f}")"""),
         md(r"""
Splitting the compression roughly halves each stage's pressure ratio, cuts the discharge temperature from
well above 100 degC to values the compressor oil tolerates, and improves the COP. Large industrial ammonia
plants for cold stores are two-stage for exactly these reasons.
""")],
        [code(r"""for name, T_cond in (("radiators, 55 degC", 55), ("underfloor heating, 35 degC", 35)):
    cop = cycles.vapor_compression("propane", 4 + 273.15 - 8, T_cond + 273.15, eta_compressor=0.7)["COP_heating"]
    print(f"{name:<28}: seasonal COP {cop:.2f} -> {12000 / cop:.0f} kWh of electricity for 12 000 kWh of heat")"""),
         md(r"""
Lowering the condenser temperature by 20 K saves about a third of the electricity over the winter. The
building's heat emitters - not the heat pump - decide much of a heat-pump system's efficiency, which is why
new heat-pump installations come with large radiators or floor heating.
""")],
        [code(r"""vc = cycles.vapor_compression("R134a", 253.15, 313.15, eta_compressor=0.8)
s1, s2 = vc["states"]["1"], vc["states"]["2"]
s_gen = s2["s"] - s1["s"]
print(f"real compressor (80 %): entropy generated {s_gen:.1f} J/(kg K) per kg; exergy destroyed at T0 = 298 K: {298.15 * s_gen / 1e3:.1f} kJ/kg "
      f"(of {vc['w_compressor']/1e3:.1f} kJ/kg of work)")"""),
         md(r"""
An adiabatic compressor's entropy generation is simply the entropy rise across it; a fifth of the work input
is destroyed as exergy - the price of the 80 % isentropic efficiency, and the largest loss in the cycle.
""")],
    ]


# =====================================================================================================
@solutions("04_real_gases_and_equations_of_state", imports=["from scipy import optimize", "from engthermo import components, datasets, eos",
                                                            "from engthermo.constants import R"])
def _():
    return [
        [code(r"""T, p = 300.0, 100e5
r = eos.solve("pr", "CO2", T, p, "stable")
M = components.get("CO2").M
V_pipe = np.pi * 0.25**2 * 10e3
rho_pr, rho_ideal = p * M / (r["Z"] * R * T), p * M / (R * T)
ref = pd.read_csv(datasets.path("real_gas_reference_densities.csv"), comment="#")
rho_ref = float(ref[(ref.fluid == "CO2") & (ref.T_K == 300.0) & (ref.p_Pa == 100e5)].rho_kg_m3.iloc[0])
print(f"PR density {rho_pr:.0f} kg/m3 (reference {rho_ref:.0f}, ideal gas {rho_ideal:.0f}): {rho_pr * V_pipe / 1e3:.0f} t of CO2 in 10 km "
      f"(ideal gas: {rho_ideal * V_pipe / 1e3:.0f} t)")"""),
         md(r"""
The pipeline holds about 1500 tonnes of dense-phase CO$_2$; the ideal-gas law would underestimate the inventory
by a factor of four. The PR density itself is 5 % low in this dense region (see notebook 04's error table),
which is why CO$_2$ pipeline design uses reference equations rather than cubics.
""")],
        [code(r"""pp = np.linspace(1e5, 1500e5, 300)
for name in ("srk", "pr"):
    Z = np.array([eos.solve(name, "methane", 200.0, p, "stable")["Z"] for p in pp])
    plt.plot(pp / 1e5, Z, label=name.upper())
    print(f"{name.upper()}: minimum Z = {Z.min():.3f} at {pp[Z.argmin()]/1e5:.0f} bar; Z > 1 above {pp[np.nonzero(Z > 1)[0][0]]/1e5:.0f} bar")
plt.axhline(1, color="k", lw=0.8); plt.xlabel("p (bar)"); plt.ylabel("Z"); plt.title("methane at 200 K"); plt.legend(); plt.show()"""),
         md(r"""
$Z$ falls steeply to about a third at 70 bar - the vapour has condensed to a dense, liquid-like phase - and
then rises slowly; only above roughly 400 bar does it exceed 1, when the molecules' own volume makes the fluid
*less* compressible than an ideal gas. (The problem asked up to 300 bar, where $Z$ is still below 1: the
scan had to be extended to see the crossing.) The same competition between $a$ and $b$ appears in every
cubic equation.
""")],
        [code(r"""c = components.get("CO2")
a, b = 27 * R**2 * c.Tc**2 / (64 * c.Pc), R * c.Tc / (8 * c.Pc)
T, p = 320.0, 50e5
f = lambda V: R * T / (V - b) - a / V**2 - p
V = optimize.brentq(f, 1.05 * b, 1.0)                                   # the vapour root: search above the liquid region
print(f"van der Waals by hand: V = {V*1e6:.1f} cm3/mol, Z = {p*V/(R*T):.4f};  library: {eos.solve('vdw', 'CO2', T, p)['V']*1e6:.1f} cm3/mol")"""),
         md(r"""
Solving the van der Waals equation for the volume by root finding is equivalent to solving the cubic for Z.
The bracket matters: starting just above $b$ and searching upward finds the vapour root; a liquid root, if
present, would lie in the narrow interval just above $b$.
""")],
    ]


# =====================================================================================================
@solutions("05_departure_functions_and_fugacity", imports=["from scipy import optimize, integrate", "from engthermo import eos, idealgas, vapor",
                                                          "from engthermo.constants import R"])
def _():
    return [
        [code(r"""T1, p1, p2, eta = 250.0, 60e5, 5e5, 0.85
h = lambda T, p, ph: idealgas.enthalpy("methane", T) + eos.solve("pr", "methane", T, p, ph)["H_R"]
s = lambda T, p, ph: idealgas.entropy("methane", T, p) + eos.solve("pr", "methane", T, p, ph)["S_R"]
T_sat = optimize.brentq(lambda T: vapor.eos_psat("pr", "methane", T)["p_sat"] - p2, 100.0, 180.0)      # dew point at 5 bar (PR)
T2s = optimize.brentq(lambda T: s(T, p2, "vapor") - s(T1, p1, "stable"), T_sat, T1)                    # vapour branch
w = eta * (h(T1, p1, "stable") - h(T2s, p2, "vapor"))
T2 = optimize.brentq(lambda T: h(T1, p1, "stable") - h(T, p2, "vapor") - w, T_sat, T1)
T2_ig = idealgas.isentropic_T("methane", T1, p1, p2)
print(f"PR: isentropic outlet {T2s:.1f} K, actual (85 %) outlet {T2:.1f} K, work {w/1e3:.2f} kJ/mol; dew point at 5 bar {T_sat:.1f} K")
print(f"ideal gas: isentropic outlet {T2_ig:.1f} K, {T2_ig - T_sat:.0f} K above the dew point")"""),
         md(r"""
Neither model predicts liquid for this pressure ratio, but they differ by nine kelvin, and that is the
whole story: the real-gas outlet is within a fraction of a kelvin of the dew point, the ideal-gas outlet
comfortably above it. A slightly larger pressure ratio, or a colder inlet, and the real expander produces
liquid where the ideal-gas model would still see dry gas - which is exactly how turbo-expanders are used in
gas plants, to condense the heavier components. The difference comes from the enthalpy and entropy
departures at the cold, dense inlet, where methane is far from ideal.
""")],
        [code(r"""h_in = idealgas.enthalpy("CO2", 320.0) + eos.solve("pr", "CO2", 320.0, 60e5, "stable")["H_R"]
T_out = optimize.brentq(lambda T: idealgas.enthalpy("CO2", T) + eos.solve("pr", "CO2", T, 1e5)["H_R"] - h_in, 150.0, 320.0)
print(f"Joule-Thomson expansion 60 bar -> 1 bar from 320 K: outlet {T_out:.1f} K, a drop of {320 - T_out:.1f} K")"""),
         md(r"""
Throttling a dense gas cools it by tens of kelvin - the Joule-Thomson effect, absent in an ideal gas. Cascaded
with heat exchangers it is the basis of Linde-type liquefaction and of some CO$_2$ liquefaction and capture
schemes, where the pressure energy of the compressed gas is spent on cooling itself.
""")],
        [code(r"""T, p = 320.0, 50e5
integral = integrate.quad(lambda pp: (eos.solve("pr", "CO2", T, pp, "stable")["Z"] - 1) / pp, 1.0, p, limit=200)[0]
print(f"ln phi by integration of (Z - 1)/p: {integral:.5f};  PR closed form: {eos.solve('pr', 'CO2', T, p)['lnphi']:.5f}")"""),
         md(r"""
$\ln\phi = \int_0^p (Z - 1)\,dp/p$ is the definition of the fugacity coefficient made numerical; the closed-form
expression of the cubic equation is this integral evaluated analytically.
""")],
    ]

# =====================================================================================================
@solutions("06_vapour_pressure_and_phase_change", imports=["from scipy import optimize", "from engthermo import components, eos, vapor",
                                                          "from engthermo.constants import R"])
def _():
    return [
        [code(r"""rows = []
for name in ("benzene", "toluene", "n-hexane", "acetone", "ethanol", "water"):
    Tb = vapor.antoine_Tsat(name, 101325.0)
    dS = vapor.antoine_dH_vap(name, Tb) / Tb
    rows.append((name, Tb - 273.15, vapor.antoine_dH_vap(name, Tb) / 1e3, dS))
print(pd.DataFrame(rows, columns=["substance", "Tb (degC)", "dH_vap at Tb (kJ/mol)", "dS_vap (J/(mol K))"]).round(1).to_string(index=False))"""),
         md(r"""
The hydrocarbons and acetone sit near Trouton's 85-88 J/(mol K): boiling releases a similar amount of
"disorder" for any ordinary liquid. Ethanol and water break the rule with about 110 J/(mol K): hydrogen
bonding orders the liquid, so more entropy is gained on vaporising it. (The Antoine-implied values carry the
ideal-vapour approximation; the true entropies of vaporisation are a few percent lower.)
""")],
        [code(r"""for T_C in (-33.3, 40.0):
    T = T_C + 273.15
    print(f"ammonia at {T_C:6.1f} degC: vapour pressure {vapor.antoine_psat('ammonia', T)/1e5:6.2f} bar")
dH = vapor.antoine_dH_vap("ammonia", 239.85)
M = components.get("ammonia").M
print(f"boil-off for a 5 kW heat leak: {5e3 / (dH / M) * 3600:.1f} kg/h  (dH_vap {dH/M/1e3:.0f} kJ/kg)")"""),
         md(r"""
The refrigerated tank is at atmospheric pressure - a thin-walled, insulated vessel - but must vent or
re-liquefy about 13 kg/h per 5 kW of heat leak; the pressurised tank needs no refrigeration but walls rated
for 15 bar and more in the sun. Large ammonia storage is refrigerated, small storage pressurised.
""")],
        [code(r"""Ts = np.linspace(300, 500, 40)
p_pr = np.array([vapor.eos_psat("pr", "benzene", T)["p_sat"] for T in Ts])
resid = lambda x: x[0] - x[1] / (Ts + x[2]) - np.log(p_pr)
fit = optimize.least_squares(resid, [21.0, 3000.0, -50.0]).x
stored = components.get("benzene").antoine
print(f"fitted to PR: A = {fit[0]:.3f}, B = {fit[1]:.1f}, C = {fit[2]:.2f};  stored: A = {stored[0]:.3f}, B = {stored[1]:.1f}, C = {stored[2]:.2f}")
print(f"max deviation of the fit from the PR pressures: {np.max(np.abs(np.exp(resid(fit)) - 1)):.2%}")"""),
         md(r"""
Three Antoine constants reproduce two hundred kelvin of vapour-pressure curve within a fraction of a percent.
The fitted constants differ visibly from the stored ones, yet both sets describe nearly the same curve: the
three constants are strongly correlated (a change in C can be compensated by A and B), so different data
ranges give different sets. Compare Antoine equations by their curves, never by their constants - and always
with their temperature range.
""")],
    ]


# =====================================================================================================
@solutions("07_mixture_fundamentals", imports=["from engthermo import datasets, mixtures", "from engthermo.constants import R"],
           setup=r"""d = pd.read_csv(datasets.path("ethanol_water_volume.csv"), comment="#")
x1, V = d.x_ethanol.to_numpy(), d.V_cm3_mol.to_numpy()""")
def _():
    return [
        [code(r"""M_e, M_w = 46.07, 18.02
n_e, n_w = 1000 * 0.785 / M_e, 1000 * 0.997 / M_w                   # moles in 1 L of each
x = n_e / (n_e + n_w)
V_mix = (n_e + n_w) * np.interp(x, x1, V)
V_pure = n_e * V[-1] + n_w * V[0]
print(f"x_ethanol = {x:.3f}; volume of the mixture {V_mix/1e3:.2f} L against {V_pure/1e3:.2f} L for the two liquids: {V_pure - V_mix:.0f} mL lost")"""),
         md(r"""
About 3 % of the volume vanishes on mixing - a bartender's rule of thumb and a nuisance for anyone who
meters alcohol by volume. The molar volumes of the file give the answer through the same partial-molar
reasoning as the notebook.
""")],
        [code(r"""W_min = R * 298.15 * np.log(2)                                     # -Delta G_mix per mole of equimolar mixture
print(f"minimum separation work {W_min/1e3:.2f} kJ/mol of mixture; heat of vaporisation of benzene about 31 kJ/mol -> ratio {31e3/W_min:.0f}")"""),
         md(r"""
Distillation boils the whole feed at least once - and often several times, in the reflux - so it spends
twenty or more times the thermodynamic minimum. That gap is the reason for heat integration, heat pumps on
columns, and the search for membrane and adsorption alternatives in the chemical industry.
""")],
        [code(r"""V_ideal = x1 * V[-1] + (1 - x1) * V[0]
VE = V - V_ideal
inner = (x1 > 0) & (x1 < 1)
X = np.column_stack([np.ones(inner.sum()), 2 * x1[inner] - 1, (2 * x1[inner] - 1) ** 2])
coef, *_ = np.linalg.lstsq(X, VE[inner] / (x1[inner] * (1 - x1[inner])), rcond=None)
A, B, C = coef
print(f"Redlich-Kister: A = {A:.3f}, B = {B:.3f}, C = {C:.3f} cm3/mol")
print(f"partial molar volume at infinite dilution: ethanol in water {V[-1] + A - B + C:.2f} cm3/mol (pure {V[-1]:.2f}), water in ethanol {V[0] + A + B + C:.2f} cm3/mol (pure {V[0]:.2f})")"""),
         md(r"""
The Redlich-Kister expansion turns the excess volume into three numbers, and its end-points give the
infinite-dilution partial molar volumes directly: an ethanol molecule alone among water molecules takes about
6 cm$^3$/mol less space than in pure ethanol, a water molecule alone among ethanol molecules about 4 cm$^3$/mol less.
""")],
    ]


# =====================================================================================================
@solutions("08_raoults_law_and_flash", imports=["from scipy import optimize", "from engthermo import vapor, vle"])
def _():
    return [
        [code(r"""T = 90 + 273.15
d = vle.Pxy(["benzene", "toluene"], T, n=41)
plt.plot(d["x1"], d["p"] / 1e3, label="bubble curve (liquid)"); plt.plot(d["y1"], d["p"] / 1e3, label="dew curve (vapour)")
plt.xlabel("mole fraction benzene"); plt.ylabel("p (kPa)"); plt.title("benzene-toluene at 90 degC"); plt.legend(); plt.show()
bub, dew = vle.bubble_P(["benzene", "toluene"], [0.5, 0.5], T), vle.dew_P(["benzene", "toluene"], [0.5, 0.5], T)
print(f"equimolar liquid: bubble pressure {bub['p']/1e3:.1f} kPa (vapour y = {bub['y'][0]:.3f}); equimolar vapour: dew pressure {dew['p']/1e3:.1f} kPa (liquid x = {dew['x'][0]:.3f})")"""),
         md(r"""
On the P-x-y diagram of an ideal mixture the bubble curve is a straight line (Raoult's law is linear in x),
the dew curve a hyperbola. Reducing the pressure at constant temperature boils the liquid at the bubble
pressure and vaporises it completely at the dew pressure - a flash by pressure reduction, the counterpart of
the T-x-y reading.
""")],
        [code(r"""T = 293.15
for x_p in (0.4, 0.3, 0.2, 0.1):
    r = vle.bubble_P(["propane", "n-butane"], [x_p, 1 - x_p], T)
    print(f"liquid {x_p:.0%} propane: vessel pressure {r['p']/1e5:.2f} bar, vapour {r['y'][0]:.0%} propane")"""),
         md(r"""
The vapour drawn off is much richer in propane than the liquid, so as the vessel empties the remaining liquid
becomes butane-rich and the pressure falls - from over 4 bar to under 3. LPG appliances see a gas whose
composition and pressure drift over the life of a cylinder; regulators and burner design allow for it.
""")],
        [code(r"""def bubble_T(comps, x, p):
    f = lambda T: sum(xi * vapor.antoine_psat(c, T) for c, xi in zip(comps, x)) / p - 1
    return optimize.brentq(f, 250.0, 450.0)
xs = np.linspace(0, 1, 21)
T_hand = [bubble_T(["benzene", "toluene"], [x, 1 - x], 101325.0) for x in xs]
lib = vle.Txy(["benzene", "toluene"], 101325.0, n=21)
print("agrees with the library to within 1e-9 K:", bool(np.max(np.abs(np.array(T_hand) - lib["T"])) < 1e-9))"""),
         md(r"""
The bubble-temperature condition is simply $\sum x_ip_i^{sat}(T)/p = 1$, monotonic in T, and a bracketed root
finder solves it in a few lines - the same code the library runs.
""")],
    ]


# =====================================================================================================
@solutions("09_activity_coefficient_models", imports=["from scipy import optimize", "from engthermo import activity, datasets, vapor, vle"],
           setup=r"""d = pd.read_csv(datasets.path("ethanol_water_vle_1atm.csv"), comment="#")
inner = d[(d.x_ethanol > 0) & (d.x_ethanol < 1)]
comps, p = ["ethanol", "water"], 101325.0
T = inner.T_C.to_numpy() + 273.15
fit_all = activity.fit("wilson", comps, inner.x_ethanol, T, p, y1=inner.y_ethanol)""")
def _():
    return [
        [code(r"""low = inner.x_ethanol < 0.5
fit_low = activity.fit("wilson", comps, inner.x_ethanol[low], T[low], p, y1=inner.y_ethanol[low])
for name, f in (("all data", fit_all), ("x < 0.5 only", fit_low)):
    az = activity.azeotrope(comps, f["gamma"], p=p)
    print(f"{name:<13}: Lambda = {f['params']['Lambda'][0, 1]:.3f}, {f['params']['Lambda'][1, 0]:.3f}; azeotrope at x = {az['x1']:.3f}, {az['T']-273.15:.2f} degC")"""),
         md(r"""
Fitted to the water-rich half only, the model recovers almost the same parameters and the same azeotrope.
That is suspiciously good, and there is a reason: this data file was generated from a Wilson model, so any
subset of it contains the whole model. With real data the two halves usually disagree, because the
parameters describing dilute ethanol in water are only loosely tied to those describing dilute water in
ethanol, and the azeotrope at x = 0.9 lies far from the water-rich data. The exercise's lesson survives the
synthetic data: extrapolation is where a fitted model is least trustworthy - and a perfect result should
make you ask why.
""")],
        [code(r"""fit_nrtl = activity.fit("nrtl", comps, inner.x_ethanol, T, p, y1=inner.y_ethanol)
xs = np.linspace(0.02, 0.98, 49)
alpha = []
for x in xs:
    r = vle.bubble_T(comps, [x, 1 - x], p, fit_nrtl["gamma"])
    y = r["y"][0]
    alpha.append(y * (1 - x) / (x * (1 - y)))
alpha = np.array(alpha)
plt.plot(xs, alpha); plt.axhline(1, color="k", lw=0.8); plt.xlabel("x ethanol"); plt.ylabel("relative volatility"); plt.show()
print(f"relative volatility falls below 1.1 above x = {xs[np.nonzero(alpha < 1.1)[0][0]]:.2f}; it equals 1 at the azeotrope")"""),
         md(r"""
The relative volatility of ethanol over water drops from about 10 in dilute solution to 1 at the azeotrope.
Above about 0.85 it is below 1.1, and each further tray of a column gains almost nothing - the practical
reason why distillation stops at 95 % ethanol even before the azeotrope is reached.
""")],
        [code(r"""g = fit_all["gamma"]
def bubble_T(x1):
    f = lambda T: (x1 * g([x1, 1 - x1], T)[0] * vapor.antoine_psat("ethanol", T) + (1 - x1) * g([x1, 1 - x1], T)[1] * vapor.antoine_psat("water", T)) / p - 1
    return optimize.brentq(f, 300.0, 400.0)
xs = np.linspace(0.05, 0.95, 10)
lib = [vle.bubble_T(comps, [x, 1 - x], p, g)["T"] for x in xs]
print("agrees with the library to within 1e-9 K:", bool(np.max(np.abs(np.array([bubble_T(x) for x in xs]) - lib)) < 1e-9))"""),
         md(r"""
Modified Raoult's law adds the activity coefficients inside the same sum; because they depend on temperature
(weakly, for Wilson) the function is evaluated inside the root finder, which handles it without trouble.
""")],
    ]


# =====================================================================================================
@solutions("10_unifac_prediction", imports=["from engthermo import activity, datasets, unifac, vle"])
def _():
    return [
        [code(r"""comps = ["acetone", "methanol"]
g = unifac.gamma_function(comps)
d = vle.Txy(comps, 101325.0, g, n=41)
plt.plot(d["x1"], d["T"] - 273.15, label="bubble"); plt.plot(d["y1"], d["T"] - 273.15, label="dew")
plt.xlabel("x, y acetone"); plt.ylabel("T (degC)"); plt.title("acetone-methanol at 1 atm, UNIFAC"); plt.legend(); plt.show()
az = activity.azeotrope(comps, g, p=101325.0)
print(f"UNIFAC azeotrope: x_acetone = {az['x1']:.3f} at {az['T']-273.15:.1f} degC  (experimental about 0.80 at 55.7 degC)")"""),
         md(r"""
UNIFAC predicts the minimum-boiling azeotrope of acetone-methanol within a few hundredths in composition
and a degree in temperature, again from group parameters alone.
""")],
        [code(r"""rows = []
for solvent in ("benzene", "toluene", "n-hexane", "1-butanol"):
    g_e = activity.infinite_dilution(unifac.gamma_function(["ethanol", solvent]), 298.15)[0]
    g_w = activity.infinite_dilution(unifac.gamma_function(["water", solvent]), 298.15)[0]
    rows.append((solvent, g_e, activity.infinite_dilution(unifac.gamma_function(["ethanol", "water"]), 298.15)[0] / g_e, g_w))
print(pd.DataFrame(rows, columns=["solvent", "gamma_inf ethanol in solvent", "distribution coefficient", "gamma_inf water in solvent"]).round(2).to_string(index=False))"""),
         md(r"""
Butanol has by far the best affinity for ethanol (its OH group), but water's activity coefficient in butanol
is small too: butanol dissolves a lot of water and is itself soluble in water (notebook 12), so the extract is
wet and solvent is lost. The hydrocarbons extract poorly but stay cleanly separated. Extraction solvent
selection is a trade-off between affinity for the solute and rejection of the carrier.
""")],
        [code(r"""ginf = unifac.gammas(["ethanol", "water"], [1e-6, 1 - 1e-6], 351.0)[0]
d = pd.read_csv(datasets.path("ethanol_water_vle_1atm.csv"), comment="#")
from engthermo import vapor
first = d.iloc[1]
g_data = first.y_ethanol * 101325.0 / (first.x_ethanol * vapor.antoine_psat("ethanol", first.T_C + 273.15))
print(f"UNIFAC gamma_inf of ethanol in water at 351 K: {ginf:.2f}; from the data point at x = {first.x_ethanol}: {g_data:.2f}")"""),
         md(r"""
Evaluating UNIFAC at a vanishing mole fraction gives the limiting activity coefficient directly - and it is
about 70 % higher than the value implied by the data (which were anchored to an experimental limiting
coefficient of about 4.2). UNIFAC reproduced the azeotrope well but not the dilute region: limiting activity
coefficients are where group-contribution methods are weakest, and also where they matter most for
environmental and separation calculations. They are best measured directly (by gas chromatography or
ebulliometry) rather than predicted.
""")],
    ]


# =====================================================================================================
@solutions("11_high_pressure_phase_equilibria", imports=["from engthermo import eos, vapor, vle"])
def _():
    return [
        [code(r"""for T_C in (30, 20, 0, -20):
    T = T_C + 273.15
    line = f"{T_C:4d} degC: pure CO2 saturates at {vapor.eos_psat('pr', 'CO2', T)['p_sat']/1e5:5.1f} bar; 90/10 CO2/N2 -> "
    for kij in (0.0, -0.02):
        k = [[0, kij], [kij, 0]]
        try:
            line += f"k12 = {kij}: bubble {vle.eos_bubble_P('pr', ['CO2', 'N2'], [0.9, 0.1], T, kij=k)['p']/1e5:5.1f} bar"
        except ValueError:
            line += f"k12 = {kij}: no bubble point"
        try:
            line += f", dew {vle.eos_dew_P('pr', ['CO2', 'N2'], [0.9, 0.1], T, kij=k)['p']/1e5:5.1f} bar;  "
        except ValueError:
            line += ", no dew point;  "
    print(line)"""),
         md(r"""
At 30 degC the equation finds neither a bubble nor a dew point: with 10 % nitrogen - a gas far above its own
critical temperature that does not want to be in a liquid - the mixture cannot be liquefied at that
temperature at all, whereas pure CO$_2$ liquefies at 72 bar. From 20 degC downwards the mixture has a bubble
point, but at 70-80 bar - far above pure CO$_2$'s saturation pressure at the same temperature. The interaction parameter
shifts the pressures by a few percent and hardly changes the picture. Liquefying a CO$_2$ stream for transport
therefore means removing the nitrogen or cooling well below ambient - a cost that CO$_2$ purity specifications
in capture projects are written around. (Near the critical region the PR equation with k12 = 0 is only
qualitative; the reference mixture model puts the bubble point at 82 bar at 0 degC and still finds one at 20 degC.)
""")],
        [code(r"""gas = ["methane", "ethane", "propane", "n-butane"]
y = [0.90, 0.06, 0.03, 0.01]
Ts = np.arange(200, 251, 5.0)
p_dew = []
for T in Ts:
    try:
        p_dew.append(vle.eos_dew_P("pr", gas, y, T)["p"] / 1e5)
    except ValueError:
        p_dew.append(np.nan)
plt.plot(Ts, p_dew, "o-"); plt.axhline(70, color="r", ls="--", label="pipeline at 70 bar")
plt.xlabel("T (K)"); plt.ylabel("dew pressure (bar)"); plt.title("hydrocarbon dew-point curve"); plt.legend(); plt.show()
ok = np.isfinite(p_dew)
print(f"dew pressures {np.nanmin(p_dew):.1f}-{np.nanmax(p_dew):.1f} bar between {Ts[ok].min():.0f} and {Ts[ok].max():.0f} K; no dew point above the cricondentherm")"""),
         md(r"""
Above the cricondentherm (about 246 K) the gas cannot condense at any pressure. Below it, condensation starts
when the pipeline pressure drops to the dew pressure - 32 bar at 245 K, 22 bar at 240 K - so at 70 bar this lean
gas stays single-phase down to at least 200 K on the branch computed here. (A lean gas also has a second,
high-pressure dew branch - the retrograde region - which this scan does not trace.) A richer gas would have a
higher cricondentherm and dew pressures, and would need conditioning before the pipeline.
""")],
        [code(r"""comps, T_ = gas, 270.0
y_ = np.array(y); p_ = 20e5; x_ = y_ * p_ / np.array([vapor.antoine_psat(c, min(T_, 0.95 * 190.56 if c == "methane" else T_)) for c in comps]); x_ /= x_.sum()
for it in range(200):
    liq = eos.mixture("pr", comps, x_, T_, p_, phase="liquid"); vap = eos.mixture("pr", comps, y_, T_, p_, phase="vapor")
    K = liq["phi_i"] / vap["phi_i"]; s = np.sum(y_ / K)
    x_, p_ = y_ / K / s, p_ / s
    if abs(s - 1) < 1e-10:
        break
print(f"by hand at 270 K: " + ("trivial solution (identical phases)" if np.max(np.abs(K - 1)) < 1e-3 else f"dew pressure {p_/1e5:.3f} bar"))
try:
    print(f"library: {vle.eos_dew_P('pr', comps, y, T_)['p']/1e5:.3f} bar")
except ValueError as e:
    print("library:", str(e)[:70])"""),
         md(r"""
At 270 K this gas is above its cricondentherm, so there is no dew point to find: successive substitution
drifts to the trivial solution where liquid and vapour become identical, and the library reports the
absence of a dew point explicitly. That is the correct answer - a lesson in checking that an iteration has
found a physical solution rather than a fixed point of the algebra.
""")],
    ]



# =====================================================================================================
@solutions("12_liquid_liquid_equilibria_and_polymers", imports=["from engthermo import activity, lle, unifac"])
def _():
    return [
        [code(r"""g = unifac.gamma_function(["n-hexane", "ethanol"])
Ts = np.arange(60, -30, -5.0)
gap = []
for T_C in Ts:
    r = lle.binary(g, T_C + 273.15)
    gap.append(np.nan if r is None else r["x1_beta"] - r["x1_alpha"])
plt.plot(Ts, gap, "o-"); plt.xlabel("T (degC)"); plt.ylabel("width of the miscibility gap (x1 beta - x1 alpha)"); plt.show()
first = Ts[np.nonzero(np.isfinite(gap))[0][0]] if np.any(np.isfinite(gap)) else None
print("UNIFAC predicts liquid-liquid splitting below about", f"{first:.0f} degC" if first is not None else "no splitting down to -25 degC")"""),
         md(r"""
UNIFAC's large positive deviations for hexane-ethanol (limiting activity coefficients of 7 and 20, notebook 10)
imply a miscibility gap at low temperature; the predicted upper critical solution temperature is a rough
number, since UNIFAC's parameters were not fitted to liquid-liquid data, but the tendency is real -
alcohol-hydrocarbon mixtures do phase-separate in the cold, a known problem for alcohol-blended fuels.
""")],
        [code(r"""Ns = np.logspace(np.log10(50), np.log10(50000), 30)
T_c = [lle.ucst(0.20, 90.0, N) for N in Ns]
plt.semilogx(Ns, T_c); plt.axhline(90.0 / 0.3, color="k", ls=":", label="theta temperature"); plt.xlabel("chain length N"); plt.ylabel("UCST (K)"); plt.legend(); plt.show()
print(f"UCST: N = 50 -> {T_c[0]:.0f} K, N = 50 000 -> {T_c[-1]:.0f} K; theta temperature {90/0.3:.0f} K")"""),
         md(r"""
The cloud point rises steeply with chain length at first and saturates towards the theta temperature. Cooling
a polydisperse solution slowly therefore precipitates the longest chains first, then progressively shorter
ones: fractional precipitation, a classical way to narrow a polymer's molar-mass distribution.
""")],
        [code(r"""chi, N = 0.7, 100
phi = np.linspace(0.002, 0.95, 4000)
G = lle.flory_huggins_gibbs(phi, chi, N)
d2 = np.gradient(np.gradient(G, phi), phi)
sign_change = np.nonzero(np.diff(np.sign(d2)))[0]
print("numerical spinodal:", np.round(phi[sign_change], 4), " library:", np.round(lle.flory_huggins_spinodal(chi, N), 4))"""),
         md(r"""
The spinodal is where the curvature of the Gibbs energy changes sign; a numerical second derivative locates
both roots within the grid spacing, matching the analytical quadratic of the library.
""")],
    ]


# =====================================================================================================
@solutions("13_solid_liquid_equilibria_and_alloys", imports=["from engthermo import sle", "from engthermo.constants import R"])
def _():
    return [
        [code(r"""e = sle.eutectic(505.0, 7.03e3, 600.6, 4.77e3)                        # component 1 = Sn, 2 = Pb
print(f"ideal-liquid prediction: eutectic at x_Sn = {e['x1']:.2f}, {e['T']-273.15:.0f} degC; experimental 0.74, 183 degC")"""),
         md(r"""
Here the ideal model fails badly: it puts the eutectic a hundred kelvin too low and at the wrong composition.
Unlike Bi-Cd (notebook 13), the Sn-Pb liquid shows strong positive deviations from ideality (activity
coefficients above 1 raise both liquidus curves), and lead dissolves up to about 19 % tin in the solid, which
the immiscible-solids model cannot represent. The lesson cuts both ways: the same two numbers per metal
that predicted Bi-Cd within five kelvin are not enough for Sn-Pb. A real phase diagram needs the activity
coefficients of the melt - and a check against measurement.
""")],
        [code(r"""e = sle.eutectic(307.0 + 273.15, 15.0e3, 334.0 + 273.15, 10.0e3)         # 1 = NaNO3, 2 = KNO3
M1, M2 = 84.99, 101.10
w1 = e["x1"] * M1 / (e["x1"] * M1 + (1 - e["x1"]) * M2)
print(f"ideal eutectic: x_NaNO3 = {e['x1']:.2f} ({w1:.0%} by mass), {e['T']-273.15:.0f} degC; observed about 222 degC (solar salt 60/40 by mass is near it)")"""),
         md(r"""
The ideal-solution eutectic is about thirty degrees below the observed one - the two nitrates form solid
solutions and their melt is not ideal - but it lands in the right region and at nearly the right composition.
A low melting point matters because the salt must stay liquid in the tanks and pipes of a solar plant: every
degree of freezing point is a degree of heat tracing needed overnight, and the eutectic composition is the
safest choice against freezing.
""")],
        [code(r"""T = 1500.0
K1, K2 = np.exp(13.05e3 / R * (1 / T - 1 / 1358.0)), np.exp(17.47e3 / R * (1 / T - 1 / 1728.0))
xL = (1 - K2) / (K1 - K2); xS = K1 * xL
lens = sle.lens_diagram(1358.0, 13.05e3, 1728.0, 17.47e3, n=401)
xL_lib, xS_lib = np.interp(T, lens["T"], lens["x1_liquidus"]), np.interp(T, lens["T"], lens["x1_solidus"])
print(f"by hand at 1500 K: liquidus x_Cu = {xL:.4f}, solidus x_Cu = {xS:.4f};  library (interpolated to 1500 K): {xL_lib:.4f}, {xS_lib:.4f}")"""),
         md(r"""
Two equilibrium constants and two mole-fraction sums give both curves in closed form; the library does the
same at every temperature.
""")],
    ]


# =====================================================================================================
@solutions("14_chemical_reaction_equilibrium", imports=["from scipy import optimize", "from engthermo import reaction"])
def _():
    return [
        [code(r"""meoh = {"CO": -1, "H2": -2, "methanol": 1}
for T_C, p_bar in ((250, 50), (250, 100), (300, 100)):
    ex = reaction.extent(meoh, {"CO": 1.0, "H2": 2.0}, T_C + 273.15, p_bar * 1e5)
    print(f"{T_C} degC, {p_bar:3d} bar: y_methanol = {ex['y']['methanol']:.3f}, CO conversion {ex['extent']:.1%}")"""),
         md(r"""
The equilibrium conversion is 50-70 % per pass at 250 degC and falls to about 40 % at 300 degC (the reaction
is exothermic), and real reactors stay well below equilibrium because the catalyst is slow at low
temperature. Methanol plants therefore condense the product and recycle the unconverted gas; the per-pass
equilibrium sets the recycle ratio and with it the compressor and reactor sizes.
""")],
        [code(r"""species = ["CH4", "H2O", "CO", "CO2", "H2"]
for sc in (2.0, 3.0, 4.0):
    gm = reaction.gibbs_minimization(species, {"CH4": 1.0, "H2O": sc}, 900 + 273.15, 20e5)
    n = gm["moles"]
    print(f"steam/carbon {sc:.0f}: CH4 conversion {1 - n['CH4']:.3f}, H2 {n['H2']:.2f} mol, CO/CO2 = {n['CO']/n['CO2']:.2f}")"""),
         md(r"""
More steam pushes both the reforming and the shift reactions to the right: higher methane conversion, more
hydrogen and less CO. It also keeps carbon (coke) from forming on the catalyst - the real reason for
steam-to-carbon ratios of about 3 - at the cost of heating and later condensing the extra steam.
""")],
        [code(r"""rxn = {"N2": -1, "H2": -3, "NH3": 2}
T, p = 723.15, 200e5
K = reaction.equilibrium_constant(rxn, T)
def f(xi):                                  # feed 1 N2 + 3 H2; extent xi
    n_N2, n_H2, n_NH3 = 1 - xi, 3 - 3 * xi, 2 * xi
    n = n_N2 + n_H2 + n_NH3
    y = np.array([n_N2, n_H2, n_NH3]) / n
    return y[2]**2 / (y[0] * y[1]**3) * (p / 1e5)**-2 - K
xi = optimize.brentq(f, 1e-9, 1 - 1e-9)
print(f"by hand: extent {xi:.4f}, y_NH3 = {2*xi/(4 - 2*xi):.4f};  library: {reaction.extent(rxn, {'N2': 1.0, 'H2': 3.0}, T, p)['extent']:.4f}")"""),
         md(r"""
Writing the mole fractions in terms of the extent turns the equilibrium condition into one equation in one
unknown on the interval (0, 1). The library solves the same equation in logarithmic form, which is better
conditioned when K is very large or very small.
""")],
    ]


# =====================================================================================================
@solutions("15_high_temperature_thermochemistry", imports=["from scipy import optimize", "from engthermo import combustion, reaction"])
def _():
    return [
        [code(r"""T_ad = combustion.adiabatic_flame_temperature("CH4", excess_air=2.0, T_in=700.0)["T_ad"]
e_needed = optimize.brentq(lambda e: combustion.adiabatic_flame_temperature("CH4", e, T_in=700.0)["T_ad"] - 1800.0, 0.0, 6.0)
print(f"200 % excess air, reactants at 700 K: flame temperature {T_ad:.0f} K; to hold 1800 K the excess air must be {e_needed:.0%}")"""),
         md(r"""
Gas turbines burn lean - two to three times the stoichiometric air - precisely to hold the turbine-inlet
temperature within what the blade alloys and cooling can bear. The excess air is not wasted: it becomes
the working fluid that drives the turbine.
""")],
        [code(r"""f = lambda T: combustion.ellingham(T)["2 C + O2 -> 2 CO"][0] - combustion.ellingham(T)["Si + O2 -> SiO2"][0]
T_cross = optimize.brentq(f, 500, 3000)
print(f"carbon reduces silica above {T_cross:.0f} K ({T_cross-273.15:.0f} degC)")
f_al = lambda T: combustion.ellingham(T)["2 C + O2 -> 2 CO"][0] - combustion.ellingham(T)["4/3 Al + O2 -> 2/3 Al2O3"][0]
print(f"carbon would reduce alumina above {optimize.brentq(f_al, 500, 3500):.0f} K")"""),
         md(r"""
Silicon smelting needs about 2000 K - exactly where the lines cross, which is why it is done in submerged-arc
furnaces at that temperature. Alumina would need well over 2000 K, where aluminium boils and carbides form;
the Hall-Heroult electrolytic process, which supplies the Gibbs energy electrically at 960 degC, is cheaper
than any furnace. Electrochemistry (notebook 16) is the alternative when the Ellingham diagram says no.
""")],
        [code(r"""lhv_C = -reaction.standard_enthalpy({"C": -1, "O2": -1, "CO2": 1})
co2_per_MJ_coal = 44.01e-3 / (lhv_C / 1e6) * 1e3
hv = combustion.heating_values("CH4")
co2_per_MJ_gas = 44.01e-3 / (hv["LHV"] / 1e6) * 1e3
print(f"carbon: LHV {lhv_C/1e3:.1f} kJ/mol -> {co2_per_MJ_coal:.0f} g CO2/MJ; methane: {co2_per_MJ_gas:.0f} g CO2/MJ; ratio {co2_per_MJ_coal/co2_per_MJ_gas:.2f}")"""),
         md(r"""
Per unit of heat, burning carbon emits about twice the CO$_2$ of burning methane - the hydrogen in methane
carries almost half its heating value without any CO$_2$. Real coals contain some hydrogen and lie a little
below pure carbon, but the factor of nearly two is the thermodynamic core of coal-to-gas switching.
""")],
    ]


# =====================================================================================================
@solutions("16_electrochemical_thermodynamics", imports=["from engthermo import electrochem, reaction", "from engthermo.constants import F, R"])
def _():
    return [
        [code(r"""steam_rxn = {"H2": -1, "O2": -0.5, "H2O": 1}
liquid_rxn = {"H2": -1, "O2": -0.5, "H2O(l)": 1}
sofc, pem = electrochem.fuel_cell_limits(steam_rxn, 2, 1073.15), electrochem.fuel_cell_limits(liquid_rxn, 2, 298.15)
for name, lim in (("SOFC at 800 degC (steam product)", sofc), ("PEM cell at 25 degC (liquid water)", pem)):
    print(f"{name:<36}: E_rev = {lim['E_rev']:.3f} V, max efficiency {lim['efficiency_max']:.1%}, reversible heat {(lim['dH'] - lim['dG'])/1e3:.1f} kJ/mol")"""),
         md(r"""
The high-temperature cell has a lower reversible voltage and a lower ceiling on efficiency: more of the
reaction enthalpy must leave as heat because the entropy change is larger at 800 degC with steam as the
product. In exchange it needs no platinum, tolerates carbon monoxide, and its waste heat is hot enough to
run a turbine - which is why SOFC systems reach the highest *system* efficiencies despite the lower cell limit.
""")],
        [code(r"""lim = electrochem.fuel_cell_limits({"H2": -1, "O2": -0.5, "H2O(l)": 1}, 2)
eta_el = electrochem.electrolyser_efficiency(1.85, lim["E_thermoneutral"])
eta_fc = electrochem.efficiency_at_voltage(0.72, lim["E_thermoneutral"])
print(f"electrolyser at 1.85 V: {eta_el:.1%}; fuel cell at 0.72 V: {eta_fc:.1%}; round trip {eta_el * eta_fc:.1%} (voltage ratio 0.72/1.85 = {0.72/1.85:.1%})")"""),
         md(r"""
Less than 40 % of the electricity returns - the ratio of the two cell voltages, plus compression, storage and
auxiliary losses not counted here. Batteries and pumped hydro return 75-90 %. Hydrogen storage makes sense
where its other properties matter: seasonal storage, energy density for transport, or a chemical feedstock.
""")],
        [code(r"""def E_cell(T, p_H2=1.0, p_O2=1.0):
    rxn = {"H2": -1, "O2": -0.5, "H2O(l)": 1} if T < 373.15 else {"H2": -1, "O2": -0.5, "H2O": 1}
    E0 = electrochem.standard_potential(rxn, 2, T)
    return electrochem.nernst(E0, 2, 1 / (p_H2 * p_O2**0.5), T)
Ts = np.linspace(298.15, 473.15, 100)
plt.plot(Ts - 273.15, [E_cell(T) for T in Ts]); plt.axvline(100, color="k", ls=":", lw=1)
plt.xlabel("T (degC)"); plt.ylabel("reversible cell voltage (V)"); plt.title("hydrogen cell at 1 bar"); plt.show()
print(f"E at 25 degC {E_cell(298.15):.3f} V, at 90 degC {E_cell(363.15):.3f} V, at 150 degC {E_cell(423.15):.3f} V (steam product)")"""),
         md(r"""
The reversible voltage falls by about 0.85 mV per kelvin below 100 degC. At the boiling point the two branches
meet - liquid water and steam have the same Gibbs energy there at 1 bar, so the curve is continuous - and above
it the slope is gentler, because the entropy change of the reaction is smaller in magnitude when the product
is a gas.
""")],
    ]


# =====================================================================================================
@solutions("17_from_messy_vle_data_to_a_model", imports=["import io", "from engthermo import activity, datasets, vapor, vle"],
           setup=r"""raw = open(datasets.path("methanol_water_lab_vle.csv"), encoding="utf-8").read().splitlines()

def load_lab_vle(lines):
    fixed, log = [], []
    for line in lines:
        if line.startswith("#"):
            continue
        f = line.split(",")
        if len(f) == 7:
            f = f[:2] + [f[2] + "." + f[3]] + f[4:]
            log.append(f"decimal comma repaired in {f[0]}")
        fixed.append(",".join(f))
    d = pd.read_csv(io.StringIO("\n".join(fixed)))
    d["p_Pa"] = np.where(d.p_unit == "mmHg", d.p * 101325 / 760, d.p * 1e3)
    log.append(f"{(d.p_unit == 'mmHg').sum()} pressures converted from mmHg")
    kelvin = d["T"] > 200
    d["T_K"] = np.where(kelvin, d["T"], d["T"] + 273.15)
    log.append(f"{kelvin.sum()} temperature converted from K")
    dup = d.duplicated(subset=["x_methanol", "y_methanol", "T"])
    log.append(f"{dup.sum()} duplicate removed")
    d = d[~dup]
    missing = d.y_methanol.isna()
    log.append(f"{missing.sum()} run without vapour analysis dropped")
    d = d[~missing].sort_values("x_methanol").reset_index(drop=True)
    return d[["run", "x_methanol", "y_methanol", "T_K", "p_Pa"]], log

d, log = load_lab_vle(raw)
keep = d.run != "run-07"
comps = ["methanol", "water"]
x, y, T, p = d.x_methanol.to_numpy(), d.y_methanol.to_numpy(), d.T_K.to_numpy(), d.p_Pa.to_numpy()""")
def _():
    return [
        [code(r"""rng = np.random.default_rng(1)
idx = np.nonzero(keep.to_numpy())[0]
out = {}
for model in ("wilson", "nrtl"):
    base = activity.fit(model, comps, x[idx], T[idx], p[idx], y1=y[idx])
    samples = []
    for _ in range(40):
        pick = rng.choice(idx, size=idx.size, replace=True)
        f = activity.fit(model, comps, x[pick], T[pick], p[pick], y1=y[pick])
        samples.append(activity.infinite_dilution(f["gamma"], 340.0))
    samples = np.array(samples)
    ginf = activity.infinite_dilution(base["gamma"], 340.0)
    print(f"{model:<7}: gamma_inf methanol in water {ginf[0]:.2f} +/- {samples[:, 0].std():.2f}, water in methanol {ginf[1]:.2f} +/- {samples[:, 1].std():.2f}")"""),
         md(r"""
The two models agree on both limiting activity coefficients within their bootstrap uncertainties. The
methanol-in-water limit is the less certain one: the data start at x = 0.02, and the dilute region is where
the models' shapes differ most. Two or three points below x = 0.02 would tighten it more than any
additional point in the middle of the range.
""")],
        [code(r"""for p_label, p_used in (("101.3 kPa as recorded", p), ("99.5 kPa (barometer corrected)", np.full_like(p, 99.5e3))):
    g1 = y * p_used / (x * vapor.antoine_psat("methanol", T)); g2 = (1 - y) * p_used / ((1 - x) * vapor.antoine_psat("water", T))
    ratio = np.log(g1 / g2)[keep]
    pos, neg = np.trapezoid(np.maximum(ratio, 0), x[keep]), -np.trapezoid(np.minimum(ratio, 0), x[keep])
    f = activity.fit("wilson", comps, x[keep], T[keep], p_used[keep], y1=y[keep])
    print(f"{p_label:<32}: Lambda12 = {f['params']['Lambda'][0, 1]:.3f}, Lambda21 = {f['params']['Lambda'][1, 0]:.3f}, area index {abs(pos - neg)/(pos + neg):.3f}")"""),
         md(r"""
A 2 % error in the pressure shifts both activity coefficients by 2 % - the area test, which depends on their
ratio, barely notices - but the fitted parameters move by several percent, and the change looks like a real
difference in the mixture. The barometer reading is as much part of a VLE measurement as the compositions,
and it belongs in the file.
""")],
        [code(r"""print(d.to_string(index=False)); print("\n".join(log))"""),
         md(r"""
The function reproduces every repair of the notebook and returns the clean table in SI units together with the
log - the reusable version of the workflow, ready for the next term's file.
""")],
    ]


if __name__ == "__main__":
    write_all(only=[a for a in sys.argv[1:] if not a.startswith("--")])
