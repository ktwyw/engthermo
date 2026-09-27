"""Power and refrigeration cycles: Carnot limits, Rankine cycles (steam), Brayton cycles (ideal gas with
variable Cp) and vapour-compression refrigeration cycles (refrigerant tables). Units: K, Pa, J/kg (per unit
mass of working fluid) or J/mol for gas cycles.

>>> from engthermo import cycles
>>> round(cycles.carnot_efficiency(T_hot=773.15, T_cold=303.15), 4)
0.6079
"""

from __future__ import annotations

from . import idealgas, steam
from .tables import FluidTable


def carnot_efficiency(T_hot: float, T_cold: float) -> float:
    """1 - T_cold/T_hot: the upper limit of any heat engine between two reservoirs."""
    return 1 - T_cold / T_hot


def carnot_cop(T_cold: float, T_hot: float, heat_pump: bool = False) -> float:
    """Maximum coefficient of performance of a refrigerator (T_cold/(T_hot - T_cold)) or heat pump (T_hot/(...))."""
    return (T_hot if heat_pump else T_cold) / (T_hot - T_cold)


def rankine(p_boiler: float, p_condenser: float, T_inlet: float | None = None, eta_turbine: float = 1.0,
            eta_pump: float = 1.0, reheat: tuple | None = None) -> dict:
    """Rankine cycle on steam. States: 1 saturated liquid at p_condenser -> 2 pump outlet -> 3 turbine inlet
    (T_inlet, or saturated vapour if None) -> 4 turbine outlet; optional reheat = (p_reheat, T_reheat) inserts
    a high-pressure turbine, reheating and a low-pressure turbine. Isentropic efficiencies default to 1.
    Returns the states, the specific work and heat terms (J/kg), the thermal efficiency and the exit quality."""
    s1 = steam.two_phase(0.0, p=p_condenser)
    s2s = steam.state_ps(p_boiler, s1["s"])
    w_pump = (s2s["h"] - s1["h"]) / eta_pump
    s2 = steam.state_ph(p_boiler, s1["h"] + w_pump)
    s3 = steam.properties(T_inlet, p_boiler) if T_inlet else steam.two_phase(1.0, p=p_boiler)
    states, w_turb, q_in = {"1": s1, "2": s2, "3": s3}, 0.0, s3["h"] - s2["h"]
    inlet = s3
    if reheat:
        p_rh, T_rh = reheat
        out_s = steam.state_ps(p_rh, inlet["s"])
        w = eta_turbine * (inlet["h"] - out_s["h"])
        states["4"] = steam.state_ph(p_rh, inlet["h"] - w)
        w_turb += w
        states["5"] = steam.properties(T_rh, p_rh)
        q_in += states["5"]["h"] - states["4"]["h"]
        inlet, last = states["5"], "6"
    else:
        last = "4"
    out_s = steam.state_ps(p_condenser, inlet["s"])
    w = eta_turbine * (inlet["h"] - out_s["h"])
    states[last] = steam.state_ph(p_condenser, inlet["h"] - w)
    w_turb += w
    q_out = states[last]["h"] - s1["h"]
    return {"states": states, "w_turbine": w_turb, "w_pump": w_pump, "w_net": w_turb - w_pump, "q_in": q_in,
            "q_out": q_out, "efficiency": (w_turb - w_pump) / q_in, "exit_quality": states[last].get("x", 1.0),
            "back_work_ratio": w_pump / w_turb}


def brayton(pressure_ratio: float, T1: float, T3: float, eta_compressor: float = 1.0, eta_turbine: float = 1.0,
            gas="air", p1: float = 1.0e5) -> dict:
    """Air-standard Brayton (gas-turbine) cycle with temperature-dependent Cp: 1 -> 2 compression, 2 -> 3 heating
    to T3, 3 -> 4 expansion. Returns temperatures, molar work and heat (J/mol) and the efficiency."""
    p2 = p1 * pressure_ratio
    T2s = idealgas.isentropic_T(gas, T1, p1, p2)
    w_c = idealgas.enthalpy_change(gas, T1, T2s) / eta_compressor
    T2 = idealgas.final_T_from_enthalpy(gas, T1, w_c)
    T4s = idealgas.isentropic_T(gas, T3, p2, p1)
    w_t = eta_turbine * idealgas.enthalpy_change(gas, T4s, T3)
    T4 = idealgas.final_T_from_enthalpy(gas, T3, -w_t)
    q_in = idealgas.enthalpy_change(gas, T2, T3)
    return {"T": {"1": T1, "2": T2, "3": T3, "4": T4}, "w_compressor": w_c, "w_turbine": w_t, "w_net": w_t - w_c,
            "q_in": q_in, "efficiency": (w_t - w_c) / q_in, "back_work_ratio": w_c / w_t}


def vapor_compression(fluid, T_evaporator: float, T_condenser: float, eta_compressor: float = 1.0,
                      superheat: float = 0.0, subcooling: float = 0.0) -> dict:
    """Vapour-compression refrigeration / heat-pump cycle from property tables (fluid name or FluidTable):
    1 compressor inlet (saturated or superheated vapour at the evaporator pressure) -> 2 compressor outlet at the
    condenser pressure -> 3 condenser outlet (saturated or subcooled liquid) -> 4 after the throttling valve.
    Returns the states, specific work and heat (J/kg), and the COPs for cooling and heating."""
    tab = fluid if isinstance(fluid, FluidTable) else FluidTable(fluid)
    p_low, p_high = tab.psat(T_evaporator), tab.psat(T_condenser)
    if superheat > 0:
        s1 = tab.superheated(p_low, T=T_evaporator + superheat)
    else:
        s1 = {**tab.saturated(T=T_evaporator)["vapour"], "T": T_evaporator, "p": p_low, "phase": "saturated vapour"}
    s2s = tab.state_ps(p_high, s1["s"])
    w = (s2s["h"] - s1["h"]) / eta_compressor
    s2 = tab.state_ph(p_high, s1["h"] + w)
    sat_h = tab.saturated(T=T_condenser)
    if subcooling > 0:
        # subcooled liquid: enthalpy taken as that of saturated liquid at the same temperature (textbook approximation)
        liq = tab.saturated(T=T_condenser - subcooling)["liquid"]
        s3 = {**liq, "T": T_condenser - subcooling, "p": p_high, "phase": "subcooled liquid"}
    else:
        s3 = {**sat_h["liquid"], "T": T_condenser, "p": p_high, "phase": "saturated liquid"}
    s4 = tab.state_ph(p_low, s3["h"])
    q_L, q_H = s1["h"] - s4["h"], s2["h"] - s3["h"]
    return {"states": {"1": s1, "2": s2, "3": s3, "4": s4}, "w_compressor": w, "q_evaporator": q_L, "q_condenser": q_H,
            "COP_cooling": q_L / w, "COP_heating": q_H / w, "p_low": p_low, "p_high": p_high}
