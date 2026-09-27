"""engthermo - chemical engineering thermodynamics: readable, validated Python with a notebook course.

Modules
-------
constants   physical constants and standard-state conventions
components  pure-component data (critical constants, ideal-gas Cp, formation properties) with sources
idealgas    ideal-gas Cp, enthalpy and entropy changes, isentropic processes, mixtures
steam       IAPWS-IF97 properties of water and steam (regions 1-5)
tables      refrigerant property tables (saturation, superheated) with interpolation
cycles      Carnot limits, Rankine, Brayton and vapour-compression cycles
datasets    bundled data files with sources
eos         virial and cubic equations of state (vdW, RK, SRK, PR): roots, departure functions, fugacity, mixtures
vapor       Clausius-Clapeyron, Antoine, saturation from an equation of state
mixtures    partial molar and excess properties, Gibbs-Duhem check
vle         Raoult's and modified Raoult's law: bubble/dew points, flash, diagrams; phi-phi bubble/dew with an EOS
activity    Margules, van Laar, Wilson, NRTL, UNIQUAC; fitting, azeotropes, consistency test
unifac      UNIFAC group-contribution prediction of activity coefficients
lle         liquid-liquid equilibrium of binaries; Flory-Huggins polymer solutions
sle         solid-liquid equilibrium: solubility, eutectics, lens diagrams of alloys
reaction    reaction enthalpy, entropy, Gibbs energy and K(T); equilibrium extent; Gibbs minimisation
combustion  heating values, flue gas, adiabatic flame temperature, Ellingham diagram
electrochem cell potentials, Nernst equation, fuel-cell and electrolyser limits
"""

__version__ = "0.1.0"

from . import (
    activity,
    combustion,
    components,
    constants,
    cycles,
    datasets,
    electrochem,
    eos,
    idealgas,
    lle,
    mixtures,
    reaction,
    sle,
    steam,
    tables,
    unifac,
    vapor,
    vle,
)

__all__ = ["activity", "combustion", "components", "constants", "cycles", "datasets", "electrochem", "eos", "idealgas",
           "lle", "mixtures", "reaction", "sle", "steam", "tables", "unifac", "vapor", "vle", "__version__"]
