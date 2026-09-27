# Methods, data and references

SI units throughout (K, Pa, J, mol). Standard state: ideal gas at 1 bar; formation properties at 298.15 K.

## Data (`components`, `_if97_coefficients`, `_unifac_data`)

- **Steam:** IAPWS, *Revised Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties
  of Water and Steam* (2007). Coefficient tables transcribed programmatically from the `iapws` package
  (J. J. Gómez Romera) by `tools/transcribe_if97.py`; verified against the release's Tables 5, 15, 33, 35, 36, 42.
- **Components:** molar mass, T_c, p_c and ω from CoolProp's reference equations of state (Bell et al., *Ind.
  Eng. Chem. Res.* 53, 2014); Cp/R = A + BT + CT² + DT⁻² fitted to CoolProp's ideal-gas Cp over each species'
  range (≤ 1 %); Antoine ln p = A − B/(T + C) fitted between 1 kPa and 15 bar (≤ 1.6 %); formation enthalpies
  and Gibbs energies (JANAF/TRC) as tabulated by Smith, Van Ness & Abbott (Table C.4). SO₃, NO, NO₂, HCl and
  graphite: Smith, Van Ness & Abbott Appendix B / Table C.1-C.2 (marked `verified=False`).
- **UNIFAC:** Fredenslund, Jones & Prausnitz, *AIChE J.* 21 (1975); parameter tables of Hansen et al., *Ind. Eng.
  Chem. Res.* 30 (1991), transcribed from the `thermo` package (C. Bell) by `tools/transcribe_unifac.py`.
- **Refrigerant tables:** generated from CoolProp with the IIR reference state (`tools/make_course_data.py`).

## Energy and cycles (`idealgas`, `steam`, `cycles`)

Steady-flow energy balance; isentropic processes solved from ∫Cp/T dT − R ln(p₂/p₁) = 0 with variable Cp.
Rankine, reheat and vapour-compression cycles as in Cengel & Boles, *Thermodynamics: An Engineering Approach*.

## Equations of state (`eos`, `vapor`)

Generic cubic p = RT/(V − b) − a(T)/[(V + εb)(V + σb)] with Ω, Ψ, ε, σ per equation (Smith, Van Ness & Abbott,
Table 3.1, exact constants): van der Waals; Redlich & Kwong, *Chem. Rev.* 44 (1949); Soave, *Chem. Eng. Sci.* 27
(1972); Peng & Robinson, *Ind. Eng. Chem. Fundam.* 15 (1976). Departure functions H^R/RT = Z − 1 + (d ln α/d ln T − 1)qI,
S^R/R = ln(Z − B) + (d ln α/d ln T)qI, ln φ = Z − 1 − ln(Z − B) − qI (SVNA Eqs. 6.66-6.67). Mixtures: van der
Waals mixing rules with k_ij; ln φ̂_i = (b_i/b)(Z − 1) − ln(Z − B) − q̄_i I (SVNA 13.9-13.10). Virial: Pitzer
correlation with Abbott's B⁰, B¹ (SVNA Eqs. 3.61-3.66). Saturation from an EOS: equal fugacities of the liquid
and vapour roots.

## Mixtures and phase equilibria (`mixtures`, `activity`, `unifac`, `vle`, `lle`, `sle`)

Partial molar properties by the tangent-intercept rule and mole-number derivatives; Gibbs-Duhem
Σx_i d ln γ_i = 0; Redlich-Kister area test (Redlich & Kister, *Ind. Eng. Chem.* 40, 1948). Activity models:
Margules, van Laar; Wilson, *J. Am. Chem. Soc.* 86 (1964); NRTL, Renon & Prausnitz, *AIChE J.* 14 (1968);
UNIQUAC, Abrams & Prausnitz, *AIChE J.* 21 (1975). Modified Raoult's law y_i p = x_iγ_ip_i^sat; Rachford-Rice
flash; φ-φ bubble and dew points by successive substitution with trivial-solution detection and pressure
scanning. Liquid-liquid: isoactivity equations from the common tangent of ΔG_mix; Flory-Huggins theory
(Flory, *J. Chem. Phys.* 10, 1942; Huggins, *J. Phys. Chem.* 46, 1942) with χ_c = (1 + 1/√N)²/2. Solid-liquid:
Schroeder-van Laar equation ln x = −(ΔH_fus/R)(1/T − 1/T_m); eutectics as intersecting liquidus curves; lens
diagrams from K_i = exp[(ΔH_i/R)(1/T − 1/T_m,i)] (Gaskell, *Introduction to the Thermodynamics of Materials*).

## Reactions, combustion, electrochemistry (`reaction`, `combustion`, `electrochem`)

ΔH°(T), ΔS°(T) from formation data and Cp integrals; K = exp(−ΔG°/RT); equilibrium extent from
K = Π(y_ip/p°)^ν_i; Gibbs minimisation with Lagrange multipliers (SVNA section 13.9). Heating values from
formation enthalpies; adiabatic flame temperature from the products' enthalpy balance; Ellingham lines for
oxides as linear approximations of Gaskell's tables (gas-phase lines computed exactly). ΔG = −nFE; Nernst
equation; dE/dT = ΔS/(nF).

## General references

Smith, Van Ness & Abbott, *Introduction to Chemical Engineering Thermodynamics*, 7th ed. (2005) · Cengel &
Boles, *Thermodynamics: An Engineering Approach* · Sandler, *Chemical, Biochemical, and Engineering
Thermodynamics* · Elliott & Lira, *Introductory Chemical Engineering Thermodynamics* · Poling, Prausnitz &
O'Connell, *The Properties of Gases and Liquids* · Gaskell, *Introduction to the Thermodynamics of Materials*.
