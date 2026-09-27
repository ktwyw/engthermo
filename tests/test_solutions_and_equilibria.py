import numpy as np
import pytest

from engthermo import activity, lle, sle, unifac, vle


def test_unifac_interfaces():
    with pytest.raises(KeyError):
        unifac.groups({"XYZ": 1})
    with pytest.raises(KeyError):
        unifac.groups("unobtainium")
    g = unifac.gammas(["benzene", "toluene"], [0.5, 0.5], 360.0)
    assert np.all(np.abs(g - 1) < 0.02)                                    # nearly ideal pair
    assert unifac.gammas(["ethanol", "water"], [1.0, 0.0], 350.0)[0] == pytest.approx(1.0, abs=1e-9)   # pure limit


def test_activity_models_and_fit_interface():
    g = activity.margules(1.0)
    assert g([0.5, 0.5])[0] == pytest.approx(g([0.5, 0.5])[1])            # symmetric
    assert g.model == "margules" and g.params["A21"] == 1.0
    with pytest.raises(KeyError):
        activity.fit("magic", ["benzene", "toluene"], [0.5], 350.0, 1e5)
    L = activity.wilson_from_energies([[0, 500.0], [800.0, 0]], [58.7e-6, 18.1e-6], 350.0)
    assert L.shape == (2, 2) and L[0, 0] == 1.0 and L[1, 1] == 1.0
    assert activity.excess_gibbs(activity.margules(0.0), [0.4, 0.6], 300.0) == 0.0
    r = activity.redlich_kister_area(activity.margules(1.5, 0.5), 300.0)
    assert r["index"] < 1e-6


def test_lle_regimes():
    assert lle.binary(activity.margules(2.0 * 0.99), 300.0) is None
    r = lle.binary(activity.margules(3.0), 300.0)
    assert r["x1_alpha"] < 0.1 and r["x1_beta"] > 0.9 and r["x1_alpha"] == pytest.approx(1 - r["x1_beta"], rel=1e-8)
    assert lle.flory_huggins_binodal(0.4, 100.0) is None
    b = lle.flory_huggins_binodal(0.7, 100.0)
    assert 0 < b["phi_dilute"] < lle.flory_huggins_critical(100.0)["phi_c"] < b["phi_concentrated"] < 1
    assert lle.ucst(0.2, 300.0, 100.0) == pytest.approx(300.0 / (lle.flory_huggins_critical(100.0)["chi_c"] - 0.2))
    with pytest.raises(RuntimeError):
        lle.flory_huggins_binodal(50.0, 1e6)                                # absurd parameters: no convergence


def test_sle_shapes_and_limits():
    assert sle.ideal_solubility(400.0, 353.4, 19.1e3) == 1.0                  # above the melting point
    T = sle.liquidus_T(0.5, 353.4, 19.1e3)
    assert sle.ideal_solubility(T, 353.4, 19.1e3) == pytest.approx(0.5, rel=1e-9)
    curves = sle.liquidus_curves(544.5, 11.3e3, 594.2, 6.19e3, n=21)
    assert curves["T_branch1"][-1] == pytest.approx(544.5) and curves["T_branch2"][0] == pytest.approx(594.2)
    lens = sle.lens_diagram(1358.0, 13.05e3, 1728.0, 17.47e3)
    assert np.all(lens["x1_solidus"] <= lens["x1_liquidus"] + 1e-12)          # solid richer in the high-melting metal (2)


def test_eos_vle_trivial_solution_is_rejected():
    with pytest.raises(ValueError):
        vle.eos_bubble_P("pr", ["methane", "ethane"], [0.98, 0.02], 250.0)    # nearly pure methane, far above its Tc
    r = vle.eos_bubble_P("pr", ["methane", "ethane"], [0.5, 0.5], 250.0)     # a bubble point does exist here (ref. 63 bar)
    assert 55e5 < r["p"] < 70e5 and r["y"][0] > 0.5
    d = vle.eos_Pxy("pr", ["methane", "ethane"], 200.0, n=5, x_max=0.6)
    assert np.all(np.isfinite(d["p"])) and np.all(np.diff(d["p"]) > 0)
