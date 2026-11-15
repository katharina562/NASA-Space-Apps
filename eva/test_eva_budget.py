"""Hand-calculation checks for the EVA budget model. Run: python eva/test_eva_budget.py"""
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eva_budget as eb  # noqa: E402

A = eb.load_assumptions()


def close(a, b, tol):
    assert abs(a - b) <= tol, f"{a} != {b} (tol {tol})"


def test_speed():
    close(eb.walk_speed_mps(0, 1.0, A), 0.75, 1e-9)
    close(eb.walk_speed_mps(10, 1.0, A), 0.65, 1e-9)
    close(eb.walk_speed_mps(-10, 1.0, A), 0.65, 1e-9)        # downhill uses |theta|
    close(eb.walk_speed_mps(20, 1.0, A), 0.55, 1e-9)
    close(1000 / eb.walk_speed_mps(0, 1.0, A) / 60, 22.2, 0.1)  # ~22 min/km


def test_pandolf_by_hand():
    W, L, G, eta, v, S = 75, 65, 0.38, 1.2, 0.75, 0.0
    rest = max(1.5 * W * G, 100)
    hand = rest + 2.0 * G * (W + L) * (L / W) ** 2 + eta * G * (W + L) * (1.5 * v ** 2 + 0.35 * v * S)
    close(eb.metabolic_walk_W(v, S, eta, W, L, A), hand, 1e-9)
    # downhill grade is treated as level
    close(eb.metabolic_walk_W(v, -15, eta, W, L, A), hand, 1e-9)
    assert eb.metabolic_walk_W(0.65, 17.6, eta, W, L, A) > hand


def test_consumable_examples():
    # 400 W for 4 h = 5.76 MJ -> ~0.27 kg O2 ; 300 W for 4 h -> ~1.8 kg water
    e = 400 * 4 * 3600 / 1e6
    close(e, 5.76, 1e-9)
    close(e * A["consumables"]["o2_kg_per_MJ"], 0.274, 0.002)
    close(300 * 4 * 3600 / (A["consumables"]["h_fg_MJ_per_kg"] * 1e6), 1.78, 0.02)


def test_heat_balance_closes():
    env = {"t_air_K": 240, "t_ground_K": 250, "t_sky_K": 190, "g_horiz_Wm2": 300}
    th = eb.suit_heat_balance(350, env, A)
    s = A["suit_thermal"]
    ts = th["t_surface_K"]
    sink4 = 0.5 * 190 ** 4 + 0.5 * 250 ** 4
    lhs = s["u_insulation_Wm2K"] * s["area_m2"] * (s["t_inner_K"] - ts) + th["q_sun_W"]
    rhs = s["emissivity"] * eb.SIGMA * s["area_m2"] * (ts ** 4 - sink4) + s["h_conv_Wm2K"] * s["area_m2"] * (ts - 240)
    close(lhs, rhs, 1e-6)
    close(th["q_evap_W"] + th["q_leak_W"], 350, 1e-6)


def test_cold_night_needs_less_water_than_noon():
    clim = eb.Climate(A)
    night = eb.suit_heat_balance(300, clim.at(5.5), A)
    noon = eb.suit_heat_balance(300, clim.at(13.0), A)
    assert night["q_leak_W"] > noon["q_leak_W"]
    assert night["q_evap_W"] < noon["q_evap_W"]


def test_steep_leg_is_no_go():
    route = pd.DataFrame([{"kind": "walk", "name": "wall", "length_m": 10, "dz_m": 5,
                           "terrain": "plain", "duration_min": 0}])
    _, s = eb.run_eva(route, A, eb.Climate(A), 9.0)
    assert "no_steep_legs" in s["failed"]


def test_pacing_holds_peak_limit():
    peak_W = A["limits"]["peak_btuh"] / eb.W_TO_BTUH
    v0 = eb.walk_speed_mps(14, 1.0, A)
    assert eb.metabolic_walk_W(v0, 25, 1.5, 75, 80, A) > peak_W        # too hard at the Apollo speed
    v = eb.pace_speed(peak_W, v0, 25, 1.5, 75, 80, A)
    assert v < v0
    close(eb.metabolic_walk_W(v, 25, 1.5, 75, 80, A), peak_W, 0.5)
    route = pd.DataFrame([{"kind": "walk", "name": "pitch", "length_m": 20, "dz_m": 5,
                           "terrain": "ridge", "duration_min": 0}])
    legs, s = eb.run_eva(route, A, eb.Climate(A), 9.0, "worst")
    assert bool(legs["paced"].iloc[0]) and s["peak_btuh"] <= A["limits"]["peak_btuh"] + 1e-6


def _trip(kind, length=1000.0, dz=0.0, stop_min=30):
    leg = {"kind": kind, "name": "out", "length_m": length, "dz_m": dz, "terrain": "plain", "duration_min": 0.0}
    back = dict(leg, name="back", dz_m=-dz)
    stop = {"kind": "stop", "name": "work", "length_m": 0.0, "dz_m": 0.0, "terrain": "plain", "duration_min": stop_min}
    return pd.DataFrame([leg, stop, back])


def test_rover_speed_and_energy_by_hand():
    rv = A["rover"]
    close(eb.drive_speed_mps(0, A), rv["v_max_mps"], 1e-9)
    close(eb.drive_speed_mps(rv["max_slope_deg"], A), rv["v_max_mps"] * (1 - rv["v_slope_loss"]), 1e-9)
    # 1 km flat: rolling force * distance / efficiency + hotel load
    t = 1000 / rv["v_max_mps"]
    hand = rv["mass_kg"] * 3.71 * rv["rolling_coeff"] * 1000 / rv["drive_efficiency"] / 3600 + rv["hotel_W"] * t / 3600
    close(eb.drive_energy_Wh(1000, 0, t, A), hand, 1e-6)
    assert eb.drive_energy_Wh(1000, 50, t, A) > hand          # climbing costs more
    assert eb.drive_energy_Wh(1000, -50, t, A) < hand         # downhill costs less, never negative
    assert eb.drive_energy_Wh(1000, -500, t, A) >= rv["hotel_W"] * t / 3600 - 1e-9


def test_driving_is_faster_and_easier_than_walking():
    clim = eb.Climate(A)
    _, walk = eb.run_eva(_trip("walk"), A, clim, 9.0)
    _, drive = eb.run_eva(_trip("drive"), A, clim, 9.0)
    assert drive["duration_h"] < walk["duration_h"]
    assert drive["o2_kg"] < walk["o2_kg"] and drive["water_kg"] < walk["water_kg"]
    assert drive["rover_battery_end_pct"] is not None and 0 < drive["rover_battery_end_pct"] < 100
    assert walk["rover_battery_end_pct"] is None and "rover_battery" not in walk["checks"]
    # suit battery only drains on foot: driving 2 km + 30 min work leaves more than walking it
    assert drive["suit_battery_end_pct"] > walk["suit_battery_end_pct"]


def test_suit_charges_in_rover_and_batteries_can_fail():
    clim = eb.Climate(A)
    legs, s = eb.run_eva(_trip("drive", stop_min=120), A, clim, 9.0)
    pct = legs["suit_battery_pct"].tolist()
    assert pct[1] < pct[0] and pct[2] > pct[1]                 # drains at the stop, recharges on the drive back
    _, far = eb.run_eva(_trip("drive", length=40000.0, stop_min=0), A, clim, 7.0)
    assert "rover_battery" in far["failed"]                    # 80 km round trip is beyond the battery
    _, steep = eb.run_eva(_trip("drive", length=100.0, dz=40.0), A, clim, 9.0)
    assert "no_steep_legs" in steep["failed"]                  # 22 deg is beyond the rover's 15 deg limit


def test_mars_season_and_daylight():
    from datetime import datetime, timezone
    # Mars24 worked example (Allison & McEwen 2000): 2000-01-06 00:00 UTC -> Ls 277.19 deg
    close(eb.mars_solar_longitude_deg(datetime(2000, 1, 6, tzinfo=timezone.utc)), 277.19, 0.02)
    # equinox (Ls 0): declination 0 -> 12 h day, sun up 06:00-18:00 at any latitude
    eq = {**A, "climate": {**A["climate"], "solar_longitude_deg": 0.0}}
    rise, sets = eb.solar_window_hours(eq)
    close(rise, 6.0, 1e-9)
    close(sets, 18.0, 1e-9)
    close(eb.sun_elevation_deg(rise, eq), 0.0, 1e-6)
    lo, hi = eb.solar_window_hours(eq, 5.0)                   # sun >= 5 deg: 20 min later / earlier
    close(lo, 6.0 + 5.0 / 15.0 / math.cos(math.radians(eq["climate"]["latitude_deg"])), 0.01)


def test_placeholder_route_runs():
    route = eb.load_route(Path(__file__).resolve().parent / "data" / "route_placeholder.csv")
    legs, s = eb.run_eva(route, A, eb.Climate(A), 9.0)
    close(s["walk_distance_m"], 2230, 1e-6)
    close(s["energy_MJ"], legs["energy_MJ"].sum(), 1e-9)
    assert 0 < s["duration_h"] < 6


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
