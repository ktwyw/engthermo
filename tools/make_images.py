"""Generate the README images (docs/images/*.png and *.gif) from the library. Run: python tools/make_images.py"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from engthermo import combustion, cycles, eos, reaction, sle, steam, unifac, vapor, vle  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=110)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ static figures
fig, ax = plt.subplots(figsize=(5.6, 4))
Ts = np.linspace(274.0, 647.0, 200)
sats = [steam.saturated(T=T) for T in Ts]
ax.plot([s["liquid"]["s"] / 1e3 for s in sats], Ts - 273.15, "k", lw=1)
ax.plot([s["vapour"]["s"] / 1e3 for s in sats], Ts - 273.15, "k", lw=1)
cyc = cycles.rankine(15e6, 10e3, T_inlet=873.15, reheat=(4e6, 873.15))
st = cyc["states"]
order = ["1", "2", "3", "4", "5", "6", "1"]
ax.plot([st[k]["s"] / 1e3 for k in order], [st[k]["T"] - 273.15 for k in order], "C3o-", ms=4, label=f"reheat Rankine cycle, efficiency {cyc['efficiency']:.1%}")
ax.set(xlabel="entropy (kJ/(kg K))", ylabel="T (degC)", title="Steam (IAPWS-IF97) with a reheat Rankine cycle")
ax.legend(frameon=False, fontsize=8)
save(fig, "ts_rankine.png")

fig, ax = plt.subplots(figsize=(5.6, 4))
V = np.logspace(np.log10(4.5e-5), -2, 400)
for T in (280.0, 304.13, 330.0, 380.0):
    p = np.array([eos.pressure("pr", "CO2", T, v) for v in V]) / 1e5
    ax.semilogx(V * 1e6, p, label=f"{T:.0f} K" + (" (critical)" if abs(T - 304.13) < 0.1 else ""))
ax.set(ylim=(0, 150), xlabel="molar volume (cm3/mol)", ylabel="p (bar)", title="Peng-Robinson isotherms of CO2")
ax.legend(frameon=False, fontsize=8)
save(fig, "pr_isotherms.png")

fig, ax = plt.subplots(figsize=(5.6, 4))
comps = ["ethanol", "water"]
raoult = vle.Txy(comps, 101325.0, n=41)
real = vle.Txy(comps, 101325.0, unifac.gamma_function(comps), n=41)
ax.plot(raoult["x1"], raoult["T"] - 273.15, "k--", lw=1, label="Raoult's law")
ax.plot(raoult["y1"], raoult["T"] - 273.15, "k--", lw=1)
ax.plot(real["x1"], real["T"] - 273.15, "C0", label="UNIFAC: azeotrope at x = 0.89")
ax.plot(real["y1"], real["T"] - 273.15, "C0")
ax.set(xlabel="mole fraction ethanol", ylabel="T (degC)", title="Ethanol-water T-x-y at 1 atm")
ax.legend(frameon=False, fontsize=8)
save(fig, "ethanol_water_txy.png")

fig, ax = plt.subplots(figsize=(5.6, 4))
rxn = {"N2": -1, "H2": -3, "NH3": 2}
T_C = np.linspace(250, 600, 40)
for p_bar in (50, 100, 200, 300):
    y = [reaction.extent(rxn, {"N2": 1.0, "H2": 3.0}, T + 273.15, p_bar * 1e5)["y"]["NH3"] for T in T_C]
    ax.plot(T_C, y, label=f"{p_bar} bar")
ax.set(xlabel="T (degC)", ylabel="equilibrium mole fraction NH3", title="Ammonia synthesis: the Haber-Bosch compromise")
ax.legend(frameon=False, fontsize=8)
save(fig, "ammonia_equilibrium.png")

fig, ax = plt.subplots(figsize=(5.6, 4))
T_e = np.linspace(300, 2000, 200)
lines = combustion.ellingham(T_e)
ends = sorted((dG[-1] / 1e3, name) for name, dG in lines.items())
label_y = []
for y_end, _ in ends:
    label_y.append(y_end if not label_y else max(y_end, label_y[-1] + 45))
for (y_end, name), y_lab in zip(ends, label_y):
    dG = lines[name]
    gas = name.startswith(("2 C", "C ", "2 H2"))
    ax.plot(T_e, dG / 1e3, color="k" if gas else None, lw=1.6 if gas else 1)
    ax.annotate(name.replace(" -> ", " > "), (T_e[-1], y_end), (T_e[-1] + 60, y_lab), fontsize=6, va="center",
                arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "0.5"})
ax.set(xlim=(300, 2750), xlabel="T (K)", ylabel="Delta G per mol O2 (kJ)", title="Ellingham diagram (C and H2 lines in black)")
save(fig, "ellingham.png")

fig, ax = plt.subplots(figsize=(5.6, 4))
lens = sle.lens_diagram(1358.0, 13.05e3, 1728.0, 17.47e3)
ax.plot(1 - lens["x1_liquidus"], lens["T"] - 273.15, label="liquidus")
ax.plot(1 - lens["x1_solidus"], lens["T"] - 273.15, label="solidus")
ax.fill_betweenx(lens["T"] - 273.15, 1 - lens["x1_liquidus"], 1 - lens["x1_solidus"], alpha=0.15)
ax.set(xlabel="mole fraction Ni", ylabel="T (degC)", title="Cu-Ni lens diagram (ideal solutions)")
ax.legend(frameon=False, fontsize=8)
save(fig, "cu_ni_lens.png")

# ------------------------------------------------------------------ hero animation: CO2 isotherms through the critical point
T_anim = np.linspace(255.0, 345.0, 46)
frames = []
for T in T_anim:
    p = np.array([eos.pressure("pr", "CO2", T, v) for v in V]) / 1e5
    tie = None
    if T < 303.0:
        s = vapor.eos_psat("pr", "CO2", T)
        tie = (s["p_sat"] / 1e5, s["V_liquid"] * 1e6, s["V_vapor"] * 1e6)
    frames.append((T, p, tie))
sat_T = np.linspace(255.0, 303.0, 60)
dome_L, dome_V, dome_p = [], [], []
for T in sat_T:
    s = vapor.eos_psat("pr", "CO2", T)
    dome_L.append(s["V_liquid"] * 1e6)
    dome_V.append(s["V_vapor"] * 1e6)
    dome_p.append(s["p_sat"] / 1e5)

fig, ax = plt.subplots(figsize=(6.4, 4))
ax.semilogx(dome_L + dome_V[::-1], dome_p + dome_p[::-1], color="0.6", lw=1, label="two-phase dome (equal fugacities)")
(iso,) = ax.semilogx([], [], "C3", lw=2, label="isotherm")
(tie_line,) = ax.semilogx([], [], "C0o-", ms=4, lw=1.5, label="saturation tie line")
ax.set(xlim=(40, 1e4), ylim=(0, 120), xlabel="molar volume (cm3/mol)", ylabel="p (bar)")
ax.legend(frameon=False, fontsize=8, loc="upper right")
title = ax.set_title("")
fig.tight_layout()


def update(i):
    T, p, tie = frames[i]
    iso.set_data(V * 1e6, p)
    if tie:
        tie_line.set_data([tie[1], tie[2]], [tie[0], tie[0]])
        state = f"liquid and vapour coexist at {tie[0]:.1f} bar"
    else:
        tie_line.set_data([], [])
        state = "supercritical: no phase boundary" if T > 304.2 else "near the critical point (304.1 K)"
    title.set_text(f"Peng-Robinson CO2 at {T:.0f} K: {state}")
    return iso, tie_line, title


anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
anim.save(OUT / "co2_isotherms.gif", writer=PillowWriter(fps=6), dpi=80)
plt.close(fig)
print("wrote co2_isotherms.gif")
