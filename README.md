<div align="center">

# engthermo

**Chemical engineering thermodynamics - readable, validated Python, with an 18-notebook course.**

[![tests](https://github.com/ktwyw/engthermo/actions/workflows/tests.yml/badge.svg)](https://github.com/ktwyw/engthermo/actions/workflows/tests.yml)
[![validation](https://img.shields.io/badge/validation-150%2F150%20checks-brightgreen)](docs/VALIDATION.md)
[![notebooks](https://img.shields.io/badge/notebooks-18%20%2B%2018%20solutions-orange)](notebooks)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--8488--9833-a6ce39)](https://orcid.org/0000-0002-8488-9833)

</div>

`engthermo` covers an undergraduate course in chemical engineering thermodynamics, from the first law and
steam power cycles through real fluids, mixtures and phase equilibria to reaction equilibrium, combustion and
electrochemistry - with examples from chemical and materials engineering and energy technologies. Every method
is checked against reference formulations (IAPWS-IF97), independent implementations (CoolProp, thermo) and
textbook examples, and a course of executed notebooks with worked solutions teaches the ideas.

```python
from engthermo import cycles, reaction, steam, vle

steam.state(p=3e6, s=6.7e3)                                 # IAPWS-IF97: any state from two properties
cycles.rankine(p_boiler=15e6, p_condenser=10e3, T_inlet=873.15, reheat=(4e6, 873.15))["efficiency"]
vle.bubble_T(["ethanol", "water"], [0.5, 0.5], p=101325.0)  # Raoult's law (add gamma=... for real mixtures)
reaction.equilibrium_constant({"N2": -1, "H2": -3, "NH3": 2}, T=723.15)
```

## Why engthermo?

- **Data you can trust.** The IAPWS-IF97 coefficient tables and the UNIFAC group tables are transcribed
  programmatically from their published sources, never typed; critical constants, heat capacities and Antoine
  vapour pressures are taken or fitted from CoolProp's reference equations, with fit errors recorded; every
  source is named in the code.
- **Validated, not just tested.** 150 checks in [`docs/VALIDATION.md`](docs/VALIDATION.md), rerun in CI: the
  IAPWS verification tables to nine figures, cubic equations and mixture fugacities against CoolProp to 1e-6,
  UNIFAC against the thermo package to 1e-8, textbook cycles (Cengel & Boles) and mixtures (Smith, Van Ness &
  Abbott), exact identities (Gibbs-Duhem, van 't Hoff, element balances), and known azeotropes, eutectics and
  cell voltages.
- **Robust where textbooks are silent.** Trivial-solution detection and pressure scanning in equation-of-state
  VLE, spinodal-bracketed liquid-liquid solvers, continuation near critical points, and a Gibbs minimiser that
  raises rather than returning a composition that violates element balances.
- **Lightweight.** Only NumPy and SciPy are required; CoolProp, iapws and thermo are used for validation only.

## What's inside

| Module | Methods |
|---|---|
| `steam` | IAPWS-IF97 regions 1-5: any state from (T, p), (p, h), (p, s), (p, x); saturation; two-phase states |
| `components`, `idealgas` | 32 species with sourced data; ideal-gas Cp, H, S, isentropic processes, mixtures |
| `tables`, `cycles` | refrigerant tables (R134a, ammonia, propane); Carnot, Rankine (reheat, efficiencies), Brayton, vapour-compression |
| `eos` | virial (Pitzer-Abbott); van der Waals, RK, SRK, Peng-Robinson: roots, departure functions, fugacity; mixtures with k_ij |
| `vapor` | Clausius-Clapeyron, Antoine (fitted, with ranges), saturation from an equation of state |
| `mixtures`, `activity` | partial molar and excess properties; Margules, van Laar, Wilson, NRTL, UNIQUAC; fitting, azeotropes, consistency test |
| `unifac` | original UNIFAC (Hansen et al. 1991 tables), group assignments for common molecules |
| `vle` | Raoult and modified Raoult: bubble/dew points, Rachford-Rice flash, diagrams; phi-phi bubble and dew points with an EOS |
| `lle`, `sle` | binary liquid-liquid equilibria; Flory-Huggins theory; solubility, eutectics, lens diagrams |
| `reaction` | Delta H, Delta S, Delta G and K(T); single-reaction extent; Gibbs minimisation for many reactions |
| `combustion`, `electrochem` | heating values, flue gas, adiabatic flame temperature, Ellingham diagram; cell potentials, Nernst, fuel-cell and electrolyser limits |
| `datasets` | 11 bundled data files (property tables and course data) with sources |

## Learn: the course

Eighteen executed notebooks, each built around engineering problems, with learning objectives, an "Inside the
algorithm" section and exercises (including an "Implement it yourself" task); every exercise has a worked
solution in [`solutions/`](solutions). They open in Google Colab and install `engthermo` automatically.

| # | Notebook | Applications |
|---|---|---|
| 00 | [Python for thermodynamics](notebooks/00_python_for_thermodynamics.ipynb) | units, tables, root finding, property diagrams |
| 01 | [The first law and energy balances](notebooks/01_first_law_and_energy_balances.ipynb) | furnaces, compressors with intercooling, turbines, valves |
| 02 | [Steam and power cycles](notebooks/02_steam_and_power_cycles.ipynb) | Rankine cycles, reheat, plant sizing |
| 03 | [Entropy, refrigeration and heat pumps](notebooks/03_entropy_refrigeration_and_heat_pumps.ipynb) | exergy, vapour-compression cycles, refrigerants, heat pumps |
| 04 | [Real gases and equations of state](notebooks/04_real_gases_and_equations_of_state.ipynb) | CO2 pipelines, hydrogen tanks |
| 05 | [Departure functions and fugacity](notebooks/05_departure_functions_and_fugacity.ipynb) | real-gas compressors, fugacity |
| 06 | [Vapour pressure and phase change](notebooks/06_vapour_pressure_and_phase_change.ipynb) | Antoine, saturation from an EOS, LPG storage |
| 07 | [Mixture fundamentals](notebooks/07_mixture_fundamentals.ipynb) | partial molar volumes, ideal mixing, Gibbs-Duhem |
| 08 | [Raoult's law and flash calculations](notebooks/08_raoults_law_and_flash.ipynb) | T-x-y diagrams, flash drums, relative volatility |
| 09 | [Activity-coefficient models](notebooks/09_activity_coefficient_models.ipynb) | ethanol-water, fitting, consistency, the bioethanol azeotrope |
| 10 | [UNIFAC prediction](notebooks/10_unifac_prediction.ipynb) | prediction without data, solvent screening |
| 11 | [High-pressure phase equilibria](notebooks/11_high_pressure_phase_equilibria.ipynb) | natural-gas dew points, CO2 mixtures, interaction parameters |
| 12 | [Liquid-liquid equilibria and polymer solutions](notebooks/12_liquid_liquid_equilibria_and_polymers.ipynb) | extraction, Flory-Huggins, UCST |
| 13 | [Solid-liquid equilibria and alloy phase diagrams](notebooks/13_solid_liquid_equilibria_and_alloys.ipynb) | solubility, antifreeze, Bi-Cd and Cu-Ni diagrams |
| 14 | [Chemical reaction equilibrium](notebooks/14_chemical_reaction_equilibrium.ipynb) | ammonia, methanol, hydrogen by steam reforming |
| 15 | [High-temperature thermochemistry](notebooks/15_high_temperature_thermochemistry.ipynb) | fuels and CO2, flame temperatures, Ellingham diagram, green steel |
| 16 | [Electrochemical thermodynamics](notebooks/16_electrochemical_thermodynamics.ipynb) | fuel cells, electrolysers, batteries |
| 17 | [From messy laboratory data to a model](notebooks/17_from_messy_vle_data_to_a_model.ipynb) | data repair, consistency, fitting with uncertainties, reporting |

## Install

```bash
pip install "engthermo @ git+https://github.com/ktwyw/engthermo"
# or, for development (adds CoolProp, iapws and thermo for the validation suite):
git clone https://github.com/ktwyw/engthermo && cd engthermo && pip install -e ".[dev]"
```

## Validation at a glance

| Reference | What is checked | Agreement |
|---|---|---|
| IAPWS-IF97 verification tables; CoolProp; iapws | steam properties in all five regions, saturation line, two-phase and inverse states | 1e-8 |
| CoolProp reference equations | critical constants, ideal-gas Cp (fitted), Antoine vapour pressures, refrigerant tables | ≤ 1 %; tables 2e-4 |
| CoolProp cubic backends | Z, departure functions, fugacity, saturation (PR, SRK); mixture fugacities and a mixture bubble point (PR) | ≤ 1e-6 |
| thermo package | UNIFAC activity coefficients (binary, ternary, negative deviations) | 1e-8 |
| Cengel & Boles; Smith, Van Ness & Abbott | Rankine, reheat and Brayton cycles; Redlich-Kwong volumes | ≤ 0.3 % |
| Exact identities | van 't Hoff, Gibbs-Duhem, G = H - TS, element balances, Rachford-Rice, energy balances | ≤ 1e-6 |
| Experiment / literature | ethanol-water azeotrope, Bi-Cd eutectic, naphthalene solubility, fuel-cell voltage and its temperature coefficient | within stated tolerances |

## Citing

If `engthermo` helps your work, please cite it using [`CITATION.cff`](CITATION.cff) (GitHub shows a
"Cite this repository" button).

## Author

**Yanwei Wang** - personal open-source project.
[GitHub @ktwyw](https://github.com/ktwyw) · [ORCID 0000-0002-8488-9833](https://orcid.org/0000-0002-8488-9833) ·
wangyanwei@gmail.com

Also by the author: [engmath](https://github.com/ktwyw/engmath) (numerical methods),
[engstat](https://github.com/ktwyw/engstat) (engineering statistics), [fluidmech](https://github.com/ktwyw/fluidmech)
(fluid mechanics) and [engrheo](https://github.com/ktwyw/engrheo) (rheology).

## License

MIT - see [LICENSE](LICENSE). Contributions welcome: see [CONTRIBUTING.md](CONTRIBUTING.md).
