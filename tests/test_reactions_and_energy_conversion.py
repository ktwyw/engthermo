import pytest

from engthermo import combustion, electrochem, reaction


def test_reaction_interfaces():
    assert reaction.parse_formula("C2H6O") == {"C": 2, "H": 6, "O": 1} and reaction.parse_formula("CH4") == {"C": 1, "H": 4}
    with pytest.raises(ValueError):
        reaction.standard_enthalpy({"air": -1})                                # no formation data for a mixture
    rxn = {"CO": -1, "H2O": -1, "CO2": 1, "H2": 1}
    assert reaction.standard_enthalpy(rxn) == pytest.approx(-41166.0, rel=1e-6)   # exothermic shift at 25 degC
    assert reaction.van_t_hoff(rxn, 298.15, 1000.0) == pytest.approx(reaction.equilibrium_constant(rxn, 1000.0), rel=0.5)
    ex = reaction.extent(rxn, {"CO": 1.0, "H2O": 1.0}, 1000.0)
    assert 0 < ex["extent"] < 1 and ex["y"]["CO2"] == pytest.approx(ex["y"]["H2"])
    inert = reaction.extent({"N2": -1, "H2": -3, "NH3": 2}, {"N2": 1.0, "H2": 3.0, "Ar": 1.0}, 700.0, 300e5)
    assert inert["moles"]["Ar"] == 1.0 and inert["y"]["NH3"] < reaction.extent({"N2": -1, "H2": -3, "NH3": 2}, {"N2": 1.0, "H2": 3.0}, 700.0, 300e5)["y"]["NH3"]


def test_combustion_quantities():
    st = combustion.stoichiometry("ethanol")
    assert st["O2"] == 3.0 and st["CO2"] == 2 and st["H2O"] == 3.0
    fg = combustion.flue_gas("CH4", excess_air=0.2)
    assert fg["O2"] == pytest.approx(0.4) and fg["N2"] == pytest.approx(2.4 * 0.79 / 0.21)
    assert combustion.heating_values("H2")["HHV"] == pytest.approx(285830.0)
    hot_air = combustion.adiabatic_flame_temperature("CH4", T_in=600.0)
    assert hot_air["T_ad"] > combustion.adiabatic_flame_temperature("CH4")["T_ad"]
    e = combustion.ellingham([800.0, 1200.0])
    assert e["2 Mg + O2 -> 2 MgO"][0] < e["2 Fe + O2 -> 2 FeO"][0]            # Mg is the stronger reducer
    assert combustion.oxygen_partial_pressure(-4e5, 1000.0) < 1e-20


def test_electrochem_relations():
    h2 = {"H2": -1, "O2": -0.5, "H2O(l)": 1}
    E0 = electrochem.standard_potential(h2, 2)
    assert electrochem.nernst(E0, 2, 10.0) < E0
    lim = electrochem.fuel_cell_limits(h2, 2)
    assert electrochem.electrolyser_efficiency(1.8, lim["E_thermoneutral"]) == pytest.approx(lim["E_thermoneutral"] / 1.8)
    assert electrochem.energy_per_mass(lim["dG"], 0.018015) == pytest.approx(237129 / 0.018015 / 3600)
    assert electrochem.efficiency_at_voltage(0.7, lim["E_thermoneutral"]) < 0.5
