import numpy as np
import pytest

from engthermo import components, eos, mixtures, vapor, vle


def test_cubic_roots_and_phases():
    with pytest.raises(KeyError):
        eos.solve("vdw2", "CO2", 300.0, 1e5)
    r = eos.solve("pr", "n-hexane", 400.0, 5e5, "liquid")
    v = eos.solve("pr", "n-hexane", 400.0, 5e5, "vapor")
    assert r["roots"].size == 3 and r["Z"] < v["Z"]
    stable = eos.solve("pr", "n-hexane", 400.0, 5e5, "stable")
    assert stable["lnphi"] == pytest.approx(min(r["lnphi"], v["lnphi"]))
    assert eos.pressure("pr", "n-hexane", 400.0, r["V"]) == pytest.approx(5e5, rel=1e-8)   # p(T, V) inverts the root
    assert eos.solve("vdw", "N2", 300.0, 1e5)["Z"] == pytest.approx(1.0, abs=2e-3)


def test_virial_and_departure_signs():
    v = eos.virial("CO2", 350.0, 10e5)
    assert v["Z"] < 1 and v["H_R"] < 0 and v["S_R"] < 0 and v["phi"] < 1
    assert eos.virial_B("Ar", 500.0)["B"] > eos.virial_B("Ar", 200.0)["B"]                 # B rises with T


def test_mixture_reduces_to_pure():
    pure = eos.solve("srk", "methane", 250.0, 30e5)
    mix = eos.mixture("srk", ["methane", "ethane"], [1.0, 0.0], 250.0, 30e5)
    assert mix["Z"] == pytest.approx(pure["Z"], rel=1e-12) and mix["lnphi_i"][0] == pytest.approx(pure["lnphi"], rel=1e-10)
    kij = [[0.0, 0.05], [0.05, 0.0]]
    with_k = eos.mixture("srk", ["methane", "ethane"], [0.5, 0.5], 250.0, 30e5, kij=kij)
    assert with_k["a"] < eos.mixture("srk", ["methane", "ethane"], [0.5, 0.5], 250.0, 30e5)["a"]


def test_vapor_pressure_interfaces():
    with pytest.raises(ValueError):
        vapor.antoine_psat("SO3", 300.0)
    with pytest.raises(ValueError):
        vapor.eos_psat("pr", "CO2", 400.0)                     # above Tc
    assert vapor.antoine_dH_vap("water", 373.15) > 0
    assert vapor.eos_psat("srk", "propane", 300.0)["V_vapor"] > vapor.eos_psat("srk", "propane", 300.0)["V_liquid"]


def test_vle_diagrams_and_flash_regimes():
    txy = vle.Txy(["benzene", "toluene"], 101325.0, n=11)
    assert txy["T"][0] > txy["T"][-1] and np.all(txy["y1"][1:-1] > txy["x1"][1:-1])        # benzene is more volatile
    pxy = vle.Pxy(["benzene", "toluene"], 360.0, n=5)
    assert pxy["p"][-1] > pxy["p"][0]
    dew = vle.dew_T(["benzene", "toluene"], [0.5, 0.5], 101325.0)
    bub = vle.bubble_T(["benzene", "toluene"], [0.5, 0.5], 101325.0)
    assert dew["T"] > bub["T"]
    assert vle.flash(["benzene", "toluene"], [0.5, 0.5], 300.0, 101325.0)["phase"] == "liquid"
    assert vle.flash(["benzene", "toluene"], [0.5, 0.5], 400.0, 101325.0)["phase"] == "vapour"


def test_mixtures_helpers():
    M_total = lambda n: (n[0] + n[1]) * (40 * n[0] / (n[0] + n[1]) + 60 * n[1] / (n[0] + n[1]))   # ideal: partial molar = pure
    assert mixtures.partial_molar(M_total, [0.3, 0.7], 0) == pytest.approx(40.0, abs=1e-6)
    assert mixtures.excess(lambda x: 40 * x[0] + 60 * x[1] - 5 * x[0] * x[1], [40, 60], [0.5, 0.5]) == pytest.approx(-1.25)
    assert components.get("methane").antoine_range[0] < 120
