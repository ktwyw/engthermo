"""Generate the README figures from the library (docs/figures/*.png). Run: python tools/make_readme_figures.py"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from engthermo import combustion, cycles, eos, reaction, sle, steam, unifac, vle  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False})

fig, ax = plt.subplots(2, 3, figsize=(13, 7.2))

# 1. T-s diagram of water with a reheat Rankine cycle
Ts = np.linspace(274.0, 647.0, 200)
sats = [steam.saturated(T=T) for T in Ts]
ax[0, 0].plot([s["liquid"]["s"] / 1e3 for s in sats], Ts - 273.15, "k", lw=1)
ax[0, 0].plot([s["vapour"]["s"] / 1e3 for s in sats], Ts - 273.15, "k", lw=1)
cyc = cycles.rankine(15e6, 10e3, T_inlet=873.15, reheat=(4e6, 873.15))
st = cyc["states"]
order = ["1", "2", "3", "4", "5", "6", "1"]
ax[0, 0].plot([st[k]["s"] / 1e3 for k in order], [st[k]["T"] - 273.15 for k in order], "C3o-", ms=4, label=f"reheat Rankine, efficiency {cyc['efficiency']:.1%}")
ax[0, 0].set(xlabel="entropy (kJ/(kg K))", ylabel="T (degC)", title="Steam (IAPWS-IF97) with a reheat Rankine cycle")
ax[0, 0].legend(frameon=False, fontsize=8)

# 2. Peng-Robinson isotherms of CO2 through the critical region
V = np.logspace(np.log10(4.5e-5), -2, 400)
for T in (280.0, 304.13, 330.0, 380.0):
    p = np.array([eos.pressure("pr", "CO2", T, v) for v in V]) / 1e5
    ax[0, 1].semilogx(V * 1e6, p, label=f"{T:.0f} K" + (" (critical)" if abs(T - 304.13) < 0.1 else ""))
ax[0, 1].set(ylim=(0, 150), xlabel="molar volume (cm3/mol)", ylabel="p (bar)", title="Peng-Robinson isotherms of CO2")
ax[0, 1].legend(frameon=False, fontsize=8)

# 3. ethanol-water T-x-y: UNIFAC vs Raoult's law
comps = ["ethanol", "water"]
raoult = vle.Txy(comps, 101325.0, n=41)
real = vle.Txy(comps, 101325.0, unifac.gamma_function(comps), n=41)
ax[0, 2].plot(raoult["x1"], raoult["T"] - 273.15, "k--", lw=1, label="Raoult's law")
ax[0, 2].plot(raoult["y1"], raoult["T"] - 273.15, "k--", lw=1)
ax[0, 2].plot(real["x1"], real["T"] - 273.15, "C0", label="UNIFAC (azeotrope at x = 0.89)")
ax[0, 2].plot(real["y1"], real["T"] - 273.15, "C0")
ax[0, 2].set(xlabel="mole fraction ethanol", ylabel="T (degC)", title="Ethanol-water T-x-y at 1 atm")
ax[0, 2].legend(frameon=False, fontsize=8)

# 4. ammonia synthesis equilibrium
rxn = {"N2": -1, "H2": -3, "NH3": 2}
T_C = np.linspace(250, 600, 40)
for p_bar in (50, 100, 200, 300):
    y = [reaction.extent(rxn, {"N2": 1.0, "H2": 3.0}, T + 273.15, p_bar * 1e5)["y"]["NH3"] for T in T_C]
    ax[1, 0].plot(T_C, y, label=f"{p_bar} bar")
ax[1, 0].set(xlabel="T (degC)", ylabel="equilibrium mole fraction NH3", title="Ammonia synthesis equilibrium")
ax[1, 0].legend(frameon=False, fontsize=8)

# 5. Ellingham diagram
T_e = np.linspace(300, 2000, 200)
lines = combustion.ellingham(T_e)
ends = sorted((dG[-1] / 1e3, name) for name, dG in lines.items())
label_y = []
for y_end, _ in ends:                                    # push labels apart by at least 40 kJ
    label_y.append(y_end if not label_y else max(y_end, label_y[-1] + 40))
for (y_end, name), y_lab in zip(ends, label_y):
    dG = lines[name]
    gas = name.startswith(("2 C", "C ", "2 H2"))
    ax[1, 1].plot(T_e, dG / 1e3, color="k" if gas else None, lw=1.6 if gas else 1)
    ax[1, 1].annotate(name.replace(" -> ", " > "), (T_e[-1], y_end), (T_e[-1] + 60, y_lab), fontsize=6, va="center",
                      arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "0.5"})
ax[1, 1].set(xlim=(300, 2700), xlabel="T (K)", ylabel="Delta G per mol O2 (kJ)", title="Ellingham diagram (carbon and hydrogen lines in black)")

# 6. Cu-Ni lens diagram
lens = sle.lens_diagram(1358.0, 13.05e3, 1728.0, 17.47e3)
ax[1, 2].plot(1 - lens["x1_liquidus"], lens["T"] - 273.15, label="liquidus")
ax[1, 2].plot(1 - lens["x1_solidus"], lens["T"] - 273.15, label="solidus")
ax[1, 2].fill_betweenx(lens["T"] - 273.15, 1 - lens["x1_liquidus"], 1 - lens["x1_solidus"], alpha=0.15)
ax[1, 2].set(xlabel="mole fraction Ni", ylabel="T (degC)", title="Cu-Ni lens diagram (ideal solutions)")
ax[1, 2].legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig(OUT / "gallery.png", dpi=130)
print("wrote", OUT / "gallery.png")
