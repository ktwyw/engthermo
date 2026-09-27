import numpy as np
import pytest

from engthermo import components, cycles, datasets, idealgas, steam, tables


def test_components_lookup_and_flags():
    assert components.get("co2").name == "carbon dioxide" and components.get("CH3OH").formula == "CH4O"
    assert components.get("SO3").verified is False and components.get("N2").verified is True
    with pytest.raises(KeyError):
        components.get("unobtainium")
    assert len(components.available()) == 32


def test_idealgas_basic_relations():
    assert idealgas.cp("Ar", 500.0) == pytest.approx(2.5 * 8.314462618)
    assert idealgas.enthalpy_change("N2", 400.0, 400.0) == 0.0
    assert idealgas.entropy_change("N2", 300.0, 300.0, 1e5, 1e6) == pytest.approx(-8.314462618 * np.log(10))
    assert idealgas.mean_cp_h("CO2", 300.0, 1000.0) > idealgas.cp("CO2", 300.0)      # Cp rises with T
    mix = idealgas.Mixture({"N2": 3, "O2": 1})                                        # unnormalised composition
    assert mix.y["nitrogen"] == pytest.approx(0.75) and mix.M == pytest.approx(0.75 * 0.028014 + 0.25 * 0.031999, rel=1e-3)
    assert mix.isentropic_T(300.0, 1e5, 5e5) > 300.0


def test_steam_regions_and_errors():
    assert steam.region(300.0, 3e6) == 1 and steam.region(700.0, 1e6) == 2 and steam.region(650.0, 25e6) == 3
    assert steam.region(1500.0, 1e6) == 5 and steam.region(900.0, 50e6) == 2
    with pytest.raises(ValueError):
        steam.region(1500.0, 60e6)
    with pytest.raises(ValueError):
        steam.psat(700.0)
    with pytest.raises(ValueError):
        steam.two_phase(1.5, p=1e5)
    with pytest.raises(ValueError):
        steam.state(T=400.0)


def test_steam_state_dispatch_and_phases():
    assert steam.state(T=400.0, p=1e5)["phase"] == "vapour"
    assert steam.state(T=400.0, p=1e6)["phase"] == "liquid"
    assert steam.Tsat(101325.0) == pytest.approx(373.124, abs=0.005)          # water boils at 99.974 degC (ITS-90)
    sat = steam.saturated(p=101325.0)
    assert sat["h_fg"] == pytest.approx(2256e3, rel=2e-3)
    wet = steam.state(p=1e5, h=1.5e6)
    assert wet["phase"] == "two-phase" and 0 < wet["x"] < 1
    hot = steam.state(p=1e5, s=8000.0)
    assert hot["phase"] == "vapour" and hot["T"] > steam.Tsat(1e5)


def test_tables_interfaces():
    t = tables.FluidTable("ammonia")
    with pytest.raises(ValueError):
        t.saturated(T=500.0)
    with pytest.raises(ValueError):
        t.state_ps(5e5, 10.0)                                   # subcooled liquid
    st = t.state_ph(3e5, t.saturated(T=300.0)["liquid"]["h"])  # throttling: ends two-phase
    assert st["phase"] == "two-phase" and 0 < st["x"] < 1
    assert t.Tsat(t.psat(280.0)) == pytest.approx(280.0, rel=1e-6)


def test_cycles_consistency():
    rk = cycles.rankine(5e6, 10e3, T_inlet=723.15, eta_turbine=0.85, eta_pump=0.8)
    ideal = cycles.rankine(5e6, 10e3, T_inlet=723.15)
    assert rk["efficiency"] < ideal["efficiency"] and rk["exit_quality"] > ideal["exit_quality"]
    assert rk["q_in"] - rk["q_out"] == pytest.approx(rk["w_net"], rel=1e-9)             # first law around the cycle
    sat_cycle = cycles.rankine(2e6, 20e3)
    assert sat_cycle["states"]["3"]["x"] == 1.0
    vc = cycles.vapor_compression("propane", 263.15, 308.15, eta_compressor=0.7, superheat=5.0, subcooling=3.0)
    assert vc["COP_heating"] == pytest.approx(vc["COP_cooling"] + 1, rel=1e-9)
    assert vc["states"]["2"]["T"] > vc["states"]["1"]["T"] and vc["states"]["1"]["phase"] == "superheated vapour"
    with pytest.raises(KeyError):
        datasets.path("R22_saturation.csv")


def test_package_exposes_every_module():
    import pkgutil

    import engthermo

    modules = sorted(m.name for m in pkgutil.iter_modules(engthermo.__path__) if not m.name.startswith("_") and m.name != "data")
    assert all(hasattr(engthermo, m) for m in modules), [m for m in modules if not hasattr(engthermo, m)]
    assert set(modules) <= set(engthermo.__all__)
