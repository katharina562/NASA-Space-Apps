"""ARES-X EVA budget model.

Takes a route (walking segments + science stops) and an EVA start time and
returns walking time, metabolic energy, O2, cooling water and heater demand,
checked against the Apollo-derived limits. Runs three cases (best / nominal /
worst) as an uncertainty band and sweeps the start time over the sol to answer
"when should the astronaut go?".

All constants live in assumptions.json.

Usage:
    python eva/eva_budget.py                       # placeholder route, 08:00 start
    python eva/eva_budget.py --route my_route.csv --start 9.5 --out out/eva
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIGMA = 5.670374e-8          # Stefan-Boltzmann, W/m2K4
W_TO_BTUH = 3.412
CASES = ("best", "nominal", "worst")


# ---------------------------------------------------------------- inputs ----

def load_assumptions(path: Path = HERE / "assumptions.json") -> dict:
    """Load the assumptions sheet and flatten {"value": x} entries to x."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    def flat(d):
        if isinstance(d, dict):
            if "value" in d:
                return d["value"]
            return {k: flat(v) for k, v in d.items() if not k.startswith("_")}
        return d

    return flat(raw)


def load_route(path: Path) -> pd.DataFrame:
    """Route CSV, one row per leg, in walking order.

    kind=walk: length_m (horizontal), dz_m (elevation change, + = uphill), terrain
    kind=stop: duration_min
    """
    df = pd.read_csv(path, comment="#")
    df["kind"] = df["kind"].str.strip().str.lower()
    for col in ("length_m", "dz_m", "duration_min"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)
    df["terrain"] = df.get("terrain", "plain")
    df["terrain"] = df["terrain"].fillna("plain").str.strip().str.lower()
    bad = set(df.loc[~df["kind"].isin(["walk", "stop", "drive"]), "kind"])
    if bad:
        raise ValueError(f"unknown kind(s) in route: {bad}")
    return df


# --------------------------------------------------------------- climate ----

def _diurnal(h: float, t_min: float, t_max: float, h_min: float, h_max: float) -> float:
    """Half-cosine rise from h_min to h_max, half-cosine fall back to h_min + 24."""
    h = h % 24.0
    if h_min <= h <= h_max:
        f = (1 - math.cos(math.pi * (h - h_min) / (h_max - h_min))) / 2
    else:
        span = 24.0 - (h_max - h_min)
        f = (1 + math.cos(math.pi * ((h - h_max) % 24.0) / span)) / 2
    return t_min + (t_max - t_min) * f


def mars_solar_longitude_deg(when: datetime | None = None) -> float:
    """Return Mars areocentric solar longitude (Ls) for an Earth UTC time.

    Implements the NASA GISS Mars24/Allison-McEwen short-series algorithm.
    The present-day TT-UTC offset is sufficient here because a leap second
    would change Ls by far less than the precision shown in the interface.
    (Added by Cilli.)
    """
    if when is None:
        when = datetime.now(timezone.utc)
    elif when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    else:
        when = when.astimezone(timezone.utc)

    jd_ut = 2440587.5 + when.timestamp() / 86400.0
    jd_tt = jd_ut + 69.184 / 86400.0
    days_j2000 = jd_tt - 2451545.0

    mean_anomaly = 19.3871 + 0.52402073 * days_j2000
    fictitious_mean_sun = 270.3871 + 0.524038496 * days_j2000
    perturbers = (
        (0.0071, 2.2353, 49.409),
        (0.0057, 2.7543, 168.173),
        (0.0039, 1.1177, 191.837),
        (0.0037, 15.7866, 21.736),
        (0.0021, 2.1354, 15.704),
        (0.0020, 2.4694, 95.528),
        (0.0018, 32.8493, 49.095),
    )
    pbs = sum(
        amplitude * math.cos(math.radians(0.985626 * days_j2000 / period + phase))
        for amplitude, period, phase in perturbers
    )
    m = math.radians(mean_anomaly)
    equation_of_center = (
        (10.691 + 3.0e-7 * days_j2000) * math.sin(m)
        + 0.623 * math.sin(2 * m)
        + 0.050 * math.sin(3 * m)
        + 0.005 * math.sin(4 * m)
        + 0.0005 * math.sin(5 * m)
        + pbs
    )
    return (fictitious_mean_sun + equation_of_center) % 360.0


def solar_declination_deg(A: dict) -> float:
    """Solar declination for the selected Mars solar longitude (Ls)."""
    ls = math.radians(A["climate"]["solar_longitude_deg"])
    return math.degrees(math.asin(math.sin(math.radians(25.19)) * math.sin(ls)))


def solar_window_hours(A: dict, min_elevation_deg: float = 0.0) -> tuple[float, float] | None:
    """Return morning/evening crossings of a Sun-elevation threshold in local solar time.

    ``None`` means the threshold is never reached. At the Gale site this function
    always returns a pair, but the general handling keeps the calculation safe for
    other latitudes. (Added by Cilli.)
    """
    lat = math.radians(A["climate"]["latitude_deg"])
    dec = math.radians(solar_declination_deg(A))
    altitude = math.radians(min_elevation_deg)
    denominator = math.cos(lat) * math.cos(dec)
    if abs(denominator) < 1e-12:
        return None
    cos_hour_angle = (
        math.sin(altitude) - math.sin(lat) * math.sin(dec)
    ) / denominator
    if cos_hour_angle >= 1.0:
        return None
    if cos_hour_angle <= -1.0:
        return (0.0, 24.0)
    half_day = math.degrees(math.acos(cos_hour_angle)) / 15.0
    return (12.0 - half_day, 12.0 + half_day)


def sun_elevation_deg(hour: float, A: dict) -> float:
    c = A["climate"]
    lat = math.radians(c["latitude_deg"])
    dec = math.radians(solar_declination_deg(A))
    omega = math.radians(15.0 * (hour - 12.0))          # Local Mean Solar Time, 24 Mars hours
    s = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(omega)
    return math.degrees(math.asin(max(-1.0, min(1.0, s))))


class Climate:
    """Hourly environment at the site: parametric default or a CSV table.

    CSV columns: hour, t_air_K, t_ground_K, t_sky_K, g_horiz_Wm2 (interpolated, periodic).
    """

    def __init__(self, A: dict, table: Path | None = None):
        self.A = A
        self.table = None
        if table is not None:
            t = pd.read_csv(table).sort_values("hour")
            self.table = t

    def at(self, hour: float) -> dict:
        if self.table is not None:
            t = self.table
            h = hour % 24.0
            xs = np.concatenate([t["hour"].to_numpy() - 24, t["hour"].to_numpy(), t["hour"].to_numpy() + 24])
            def interp(col):
                ys = np.tile(t[col].to_numpy(), 3)
                return float(np.interp(h, xs, ys))
            return {k: interp(k) for k in ("t_air_K", "t_ground_K", "t_sky_K", "g_horiz_Wm2")}

        c = self.A["climate"]
        t_air = _diurnal(hour, c["t_air_min_K"], c["t_air_max_K"], c["t_air_min_hour"], c["t_air_max_hour"])
        t_gnd = _diurnal(hour, c["t_ground_min_K"], c["t_ground_max_K"], c["t_air_min_hour"], c["t_ground_max_hour"])
        elev = sun_elevation_deg(hour, self.A)
        g = c["solar_const_Wm2"] * c["transmittance"] * max(0.0, math.sin(math.radians(elev)))
        return {"t_air_K": t_air, "t_ground_K": t_gnd, "t_sky_K": c["t_sky_factor"] * t_air, "g_horiz_Wm2": g}

    def hourly_table(self, step: float = 0.5) -> pd.DataFrame:
        rows = [{"hour": h, **self.at(h), "sun_elev_deg": sun_elevation_deg(h, self.A)}
                for h in np.arange(0, 24, step)]
        return pd.DataFrame(rows)


# ---------------------------------------------------------------- models ----

def walk_speed_mps(theta_deg: float, speed_factor: float, A: dict) -> float:
    """Apollo-derived V = 0.75 - 0.01*|theta|, capped at 0.75 m/s, times terrain factor."""
    w = A["walking"]
    v = min(w["v_flat_mps"], w["v_flat_mps"] - w["v_slope_coeff"] * abs(theta_deg))
    return max(v, 0.05) * speed_factor


def metabolic_walk_W(v: float, grade_pct: float, eta: float, W: float, L: float, A: dict) -> float:
    """Gravity-corrected Pandolf (NASA JSC CSRM, ICES-2025-129).

    M = 1.5*W*G + 2.0*G*(W+L)*(L/W)^2 + eta*G*(W+L)*(1.5*V^2 + 0.35*V*S)
    The resting term gets a floor (rest_floor_W); downhill is treated as level.
    """
    G = A["astronaut"]["gravity_ratio"]
    S = max(grade_pct, 0.0)
    rest = max(1.5 * W * G, A["astronaut"]["rest_floor_W"])
    load = 2.0 * G * (W + L) * (L / W) ** 2
    move = eta * G * (W + L) * (1.5 * v ** 2 + 0.35 * v * S)
    return rest + load + move


def drive_speed_mps(theta_deg: float, A: dict) -> float:
    """Rover speed: v_max on the flat, falling linearly to (1 - v_slope_loss) * v_max at max slope."""
    rv = A["rover"]
    f = min(abs(theta_deg) / rv["max_slope_deg"], 1.0)
    return rv["v_max_mps"] * (1.0 - rv["v_slope_loss"] * f)


def drive_energy_Wh(length_m: float, dz_m: float, dur_s: float, A: dict) -> float:
    """Battery energy to drive one leg: rolling resistance + climbing, no regen, plus hotel load."""
    rv = A["rover"]
    g = 3.71
    theta = math.atan2(dz_m, max(length_m, 1e-6))
    d = math.hypot(length_m, dz_m)
    force = rv["mass_kg"] * g * (rv["rolling_coeff"] * math.cos(theta) + math.sin(theta))
    wheel_J = max(force, 0.0) * d
    return wheel_J / rv["drive_efficiency"] / 3600 + rv["hotel_W"] * dur_s / 3600


def pace_speed(target_W: float, v_max: float, grade_pct: float, eta: float, W: float, L: float, A: dict) -> float:
    """Highest speed <= v_max whose metabolic rate stays at target_W (M rises with V)."""
    lo, hi = 0.05, v_max
    if metabolic_walk_W(lo, grade_pct, eta, W, L, A) >= target_W:
        return lo
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if metabolic_walk_W(mid, grade_pct, eta, W, L, A) <= target_W else (lo, mid)
    return lo


def suit_heat_balance(q_internal_W: float, env: dict, A: dict) -> dict:
    """Steady suit heat balance.

    Outer surface (temperature Ts):
        U*A*(T_in - Ts) + alpha*A_proj*G = eps*sigma*A*(Ts^4 - T_sink^4) + h*A*(Ts - T_air)
    Inside the suit:
        Q_met + Q_elec = U*A*(T_in - Ts) + Q_evap   (Q_evap < 0 -> heater needed)
    """
    s = A["suit_thermal"]
    U, area, eps, h = s["u_insulation_Wm2K"], s["area_m2"], s["emissivity"], s["h_conv_Wm2K"]
    Fs = s["view_factor_sky"]
    t_sink4 = Fs * env["t_sky_K"] ** 4 + (1 - Fs) * env["t_ground_K"] ** 4
    q_sun = s["solar_absorptivity"] * s["area_proj_m2"] * env["g_horiz_Wm2"]
    t_in, t_air = s["t_inner_K"], env["t_air_K"]

    def residual(ts):
        return U * area * (t_in - ts) + q_sun - eps * SIGMA * area * (ts ** 4 - t_sink4) - h * area * (ts - t_air)

    lo, hi = 100.0, 400.0                      # residual is monotonic decreasing in Ts
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if residual(mid) > 0 else (lo, mid)
    ts = (lo + hi) / 2
    leak = U * area * (t_in - ts)              # heat lost from the suit (negative = gained)
    q_evap = q_internal_W - leak
    return {"t_surface_K": ts, "q_leak_W": leak, "q_sun_W": q_sun,
            "q_evap_W": max(q_evap, 0.0), "q_heater_W": max(-q_evap, 0.0)}


# ------------------------------------------------------------ simulation ----

def run_eva(route: pd.DataFrame, A: dict, climate: Climate, start_hour: float, case: str = "nominal"):
    """Go through the route leg by leg (walk / drive / stop) from start_hour.

    Returns (legs DataFrame, summary dict)."""
    cs = A["cases"][case]
    W, L, margin = A["astronaut"]["body_mass_kg"], cs["suit_load_kg"], cs["margin"]
    q_elec = A["suit_thermal"]["q_elec_W"]
    cons = A["consumables"]
    max_slope = A["walking"]["max_slope_deg"]
    peak_W = A["limits"]["peak_btuh"] / W_TO_BTUH
    rv = A.get("rover", {})
    suit_cap = A.get("suit_power", {}).get("battery_Wh", float("inf"))
    rover_cap = rv.get("battery_Wh", float("inf"))
    suit_Wh, rover_Wh, suit_min_Wh = suit_cap, rover_cap, suit_cap
    has_rover = bool((route["kind"] == "drive").any())

    rows, t_h = [], start_hour
    for i, leg in route.iterrows():
        r = {"leg": i + 1, "name": leg.get("name", ""), "kind": leg["kind"], "terrain": leg["terrain"],
             "length_m": leg["length_m"], "dz_m": leg["dz_m"]}
        if leg["kind"] == "walk":
            terr = A["terrain"].get(leg["terrain"])
            if terr is None:
                raise ValueError(f"leg {i + 1}: unknown terrain '{leg['terrain']}'")
            length = max(leg["length_m"], 1e-6)
            theta = math.degrees(math.atan2(leg["dz_m"], length))
            grade = 100.0 * leg["dz_m"] / length
            eta = terr["eta"][cs["terrain_index"]]
            v = walk_speed_mps(theta, terr["speed_factor"], A)
            m_W = metabolic_walk_W(v, grade, eta, W, L, A)
            paced = False
            if A["walking"].get("pace_to_peak_limit") and m_W > peak_W:
                v = pace_speed(peak_W, v, grade, eta, W, L, A)   # slow down on steep pitches
                m_W = metabolic_walk_W(v, grade, eta, W, L, A)
                paced = True
            dur_s = math.hypot(length, leg["dz_m"]) / v * margin
            r.update(slope_deg=theta, grade_pct=grade, speed_mps=v, paced=paced, no_go=abs(theta) > max_slope)
        elif leg["kind"] == "drive":
            length = max(leg["length_m"], 1e-6)
            theta = math.degrees(math.atan2(leg["dz_m"], length))
            v = drive_speed_mps(theta, A)
            dur_s = math.hypot(length, leg["dz_m"]) / v * rv["margin"]
            m_W = rv["seated_metabolic_W"]
            r.update(slope_deg=theta, grade_pct=100.0 * leg["dz_m"] / length, speed_mps=v, paced=False,
                     no_go=abs(theta) > rv["max_slope_deg"])
        else:
            dur_s = leg["duration_min"] * 60.0
            m_W = cs["stop_work_W"]
            r.update(slope_deg=0.0, grade_pct=0.0, speed_mps=0.0, paced=False, no_go=False)

        mid_h = t_h + dur_s / 7200.0
        env = climate.at(mid_h)
        th = suit_heat_balance(m_W + q_elec, env, A)
        e_MJ = m_W * dur_s / 1e6
        r.update(start_h=t_h, duration_min=dur_s / 60, metabolic_W=m_W, metabolic_btuh=m_W * W_TO_BTUH,
                 energy_MJ=e_MJ, o2_kg=e_MJ * cons["o2_kg_per_MJ"],
                 water_kg=th["q_evap_W"] * dur_s / (cons["h_fg_MJ_per_kg"] * 1e6),
                 heater_Wh=th["q_heater_W"] * dur_s / 3600, t_air_C=env["t_air_K"] - 273.15,
                 q_leak_W=th["q_leak_W"], q_evap_W=th["q_evap_W"])

        # batteries: docked in the rover the suit runs on rover power and charges;
        # walking or working it runs on its own battery
        dur_h = dur_s / 3600
        suit_draw_W = q_elec + th["q_heater_W"]
        if leg["kind"] == "drive":
            rover_used = drive_energy_Wh(leg["length_m"], leg["dz_m"], dur_s, A) \
                + (suit_draw_W + rv["suit_charge_W"]) * dur_h
            rover_Wh -= rover_used
            suit_Wh = min(suit_cap, suit_Wh + rv["suit_charge_W"] * dur_h)
        else:
            rover_used = rv.get("hotel_W", 0) * dur_h if has_rover else 0.0   # parked, relay radio on
            rover_Wh -= rover_used
            suit_Wh -= suit_draw_W * dur_h
        suit_min_Wh = min(suit_min_Wh, suit_Wh)
        r.update(rover_Wh=rover_used, suit_battery_pct=100 * suit_Wh / suit_cap)
        rows.append(r)
        t_h += dur_s / 3600.0

    legs = pd.DataFrame(rows)
    s = summarize(legs, A, start_hour, t_h, case)
    drives = bool((legs["kind"] == "drive").any())
    s.update(drive_distance_m=float(legs.loc[legs["kind"] == "drive", "length_m"].sum()),
             drive_time_h=float(legs.loc[legs["kind"] == "drive", "duration_min"].sum() / 60),
             suit_battery_end_pct=100 * suit_Wh / suit_cap, suit_battery_min_pct=100 * suit_min_Wh / suit_cap,
             rover_battery_end_pct=100 * rover_Wh / rover_cap if drives else None,
             rover_used_kWh=(rover_cap - rover_Wh) / 1000 if drives else 0.0)
    s["checks"]["suit_battery"] = suit_min_Wh >= A.get("suit_power", {}).get("reserve_fraction", 0) * suit_cap
    if drives:
        s["checks"]["rover_battery"] = rover_Wh >= rv["reserve_fraction"] * rover_cap
    s["failed"] = [k for k, ok in s["checks"].items() if not ok]
    s["pass"] = not s["failed"]
    return legs, s


def summarize(legs: pd.DataFrame, A: dict, start_h: float, end_h: float, case: str) -> dict:
    lim, cons = A["limits"], A["consumables"]
    dur_h = end_h - start_h
    energy = legs["energy_MJ"].sum()
    avg_btuh = energy * 1e6 / (dur_h * 3600) * W_TO_BTUH
    usable = 1 - cons["reserve_fraction"]
    s = {
        "case": case, "start_h": start_h, "end_h": end_h, "duration_h": dur_h,
        "walk_distance_m": float(legs.loc[legs["kind"] == "walk", "length_m"].sum()),
        "walk_time_h": float(legs.loc[legs["kind"] == "walk", "duration_min"].sum() / 60),
        "energy_MJ": energy, "energy_kcal": energy * 239.006,
        "avg_btuh": avg_btuh, "peak_btuh": float(legs["metabolic_btuh"].max()),
        "o2_kg": float(legs["o2_kg"].sum()), "water_kg": float(legs["water_kg"].sum()),
        "heater_Wh": float(legs["heater_Wh"].sum()),
        "max_slope_deg": float(legs["slope_deg"].abs().max()),
    }
    s["o2_used_pct"] = 100 * s["o2_kg"] / cons["o2_capacity_kg"]
    s["water_used_pct"] = 100 * s["water_kg"] / cons["water_capacity_kg"]
    checks = {
        "no_steep_legs": not bool(legs["no_go"].any()),
        "duration": dur_h <= lim["max_eva_h"],
        "peak_metabolic": s["peak_btuh"] <= lim["peak_btuh"],
        "avg_metabolic": avg_btuh <= lim["avg_btuh"],
        "o2_reserve": s["o2_kg"] <= usable * cons["o2_capacity_kg"],
        "water_reserve": s["water_kg"] <= usable * cons["water_capacity_kg"],
        "daylight": min(sun_elevation_deg(start_h, A), sun_elevation_deg(end_h, A)) >= lim["min_sun_elev_deg"],
    }
    s["checks"] = checks
    s["failed"] = [k for k, ok in checks.items() if not ok]
    s["pass"] = not s["failed"]
    return s


def verdict(results: dict) -> str:
    """GO if even the worst case passes, MARGINAL if only nominal passes, else NO-GO."""
    if results["worst"]["pass"]:
        return "GO"
    if results["nominal"]["pass"]:
        return "MARGINAL"
    return "NO-GO"


def sweep_start_times(route, A, climate, starts) -> pd.DataFrame:
    rows = []
    for h in starts:
        res = {c: run_eva(route, A, climate, h, c)[1] for c in CASES}
        row = {"start_h": h, "verdict": verdict(res), "end_h_nominal": res["nominal"]["end_h"]}
        for c in CASES:
            for k in ("duration_h", "water_kg", "o2_kg", "heater_Wh", "avg_btuh"):
                row[f"{k}_{c}"] = res[c][k]
            row[f"failed_{c}"] = ";".join(res[c]["failed"])
        rows.append(row)
    return pd.DataFrame(rows)


def best_start(sweep: pd.DataFrame):
    """Earliest-lowest-water start among GO starts (falls back to MARGINAL)."""
    for level in ("GO", "MARGINAL"):
        ok = sweep[sweep["verdict"] == level]
        if len(ok):
            r = ok.sort_values(["water_kg_nominal", "heater_Wh_nominal", "start_h"]).iloc[0]
            return float(r["start_h"]), level
    return None, "NO-GO"


# --------------------------------------------------------------- outputs ----

def hhmm(h: float) -> str:
    m = int(round(h * 60)) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def plot_sweep(sweep: pd.DataFrame, best_h, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, ink, muted, grid = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1"
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    x = sweep["start_h"]

    half = (x.iloc[1] - x.iloc[0]) / 2 if len(x) > 1 else 0.125
    go = sweep["verdict"] == "GO"
    for h, g in zip(x, go):
        if not g:
            ax.axvspan(h - half, h + half, color="#ecebe7", lw=0, zorder=0)
    if go.any():
        g0, g1 = x[go].min(), x[go].max()
        ax.annotate("", (g0 - half, 1.0), (g1 + half, 1.0), xycoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="<->", color=muted, lw=1))
        ax.text((g0 + g1) / 2, 0.97, f"GO window {hhmm(g0)}–{hhmm(g1)}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", color=muted, fontsize=8.5)
    ax.fill_between(x, sweep["water_kg_best"], sweep["water_kg_worst"], color=blue, alpha=0.18, lw=0,
                    label="Best–worst case")
    ax.plot(x, sweep["water_kg_nominal"], color=blue, lw=2, label="Nominal")
    if best_h is not None:
        yb = float(sweep.loc[sweep["start_h"] == best_h, "water_kg_nominal"].iloc[0])
        ax.plot([best_h], [yb], "o", ms=8, color=blue, mec="#fcfcfb", mew=2, zorder=5)
        ax.annotate(f"Best start {hhmm(best_h)}\n{yb:.2f} kg", (best_h, yb), xytext=(10, -28),
                    textcoords="offset points", color=ink, fontsize=9)

    ax.set_title("Cooling water used vs. EVA start time", loc="left", color=ink, fontsize=12, pad=18)
    ax.text(0, 1.03, "Gray: not GO in the worst case (limits, 25% reserve, return in daylight)",
            transform=ax.transAxes, color=muted, fontsize=8.5)
    ax.set_xlabel("EVA start time (local mean solar time)", color=muted, fontsize=9)
    ax.set_ylabel("Cooling water (kg)", color=muted, fontsize=9)
    ticks = np.arange(math.floor(x.min()), math.ceil(x.max()) + 1, 1)
    ax.set_xticks(ticks, [f"{int(t):02d}:00" for t in ticks])
    ax.set_ylim(0, sweep["water_kg_worst"].max() * 1.35)
    ax.set_xlim(x.min() - half, x.max() + half)
    ax.tick_params(colors=muted, labelsize=8.5, length=0)
    ax.grid(axis="y", color=grid, lw=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_assumptions_md(path_json: Path, path_md: Path) -> None:
    raw = json.loads(path_json.read_text(encoding="utf-8"))
    lines = ["# EVA budget assumptions", "", "| Group | Parameter | Value | Unit | Source |", "|---|---|---|---|---|"]
    for group, items in raw.items():
        if group.startswith("_") or not isinstance(items, dict):
            continue
        for key, v in items.items():
            if isinstance(v, dict) and "value" in v:
                lines.append(f"| {group} | {key} | {v['value']} | {v['unit']} | {v['source']} |")
        if "_source" in items:
            lines.append(f"| {group} | (table) | see assumptions.json | | {items['_source']} |")
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description="ARES-X EVA budget")
    real = HERE.parent / "out" / "route" / "route_B.csv"     # from prep/route_dem.py
    p.add_argument("--route", type=Path, default=real if real.exists() else HERE / "data" / "route_placeholder.csv")
    p.add_argument("--assumptions", type=Path, default=HERE / "assumptions.json")
    p.add_argument("--climate", type=Path, default=None, help="optional hourly climate CSV")
    p.add_argument("--start", type=float, default=None, help="EVA start hour (LMST); default = best start")
    p.add_argument("--sweep-from", type=float, default=6.0)
    p.add_argument("--sweep-to", type=float, default=16.0)
    p.add_argument("--sweep-step", type=float, default=0.25)
    p.add_argument("--out", type=Path, default=HERE.parent / "out" / "eva")
    a = p.parse_args(argv)

    A = load_assumptions(a.assumptions)
    route = load_route(a.route)
    climate = Climate(A, a.climate)
    a.out.mkdir(parents=True, exist_ok=True)

    sweep = sweep_start_times(route, A, climate, np.arange(a.sweep_from, a.sweep_to + 1e-9, a.sweep_step))
    best_h, best_level = best_start(sweep)
    start = a.start if a.start is not None else (best_h if best_h is not None else 8.0)

    results, legs_nom = {}, None
    for c in CASES:
        legs, s = run_eva(route, A, climate, start, c)
        results[c] = s
        if c == "nominal":
            legs_nom = legs

    legs_nom.round(4).to_csv(a.out / "legs_nominal.csv", index=False)
    sweep.round(4).to_csv(a.out / "start_time_sweep.csv", index=False)
    climate.hourly_table().round(2).to_csv(a.out / "climate_hourly.csv", index=False)
    plot_sweep(sweep, best_h, a.out / "water_vs_start_time.png")
    write_assumptions_md(a.assumptions, a.out / "assumptions.md")

    payload = {
        "route_file": a.route.name, "start_h": start, "start_hhmm": hhmm(start),
        "verdict": verdict(results), "best_start_h": best_h,
        "best_start_hhmm": hhmm(best_h) if best_h is not None else None, "best_start_level": best_level,
        "cases": results,
        "legs": json.loads(legs_nom.round(4).to_json(orient="records")),
        "sweep": json.loads(sweep.round(4).to_json(orient="records")),
    }
    (a.out / "eva_budget.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    n = results["nominal"]
    print(f"Route {a.route.name}: {n['walk_distance_m']:.0f} m walking, start {hhmm(start)} LMST")
    print(f"{'':10s}{'best':>10s}{'nominal':>10s}{'worst':>10s}")
    for key, fmt in [("duration_h", "{:.2f}"), ("walk_time_h", "{:.2f}"), ("avg_btuh", "{:.0f}"),
                     ("peak_btuh", "{:.0f}"), ("energy_MJ", "{:.2f}"), ("o2_kg", "{:.3f}"),
                     ("water_kg", "{:.2f}"), ("heater_Wh", "{:.0f}")]:
        print(f"{key:10s}" + "".join(f"{fmt.format(results[c][key]):>10s}" for c in CASES))
    for c in CASES:
        print(f"{c:8s} {'PASS' if results[c]['pass'] else 'FAIL: ' + ', '.join(results[c]['failed'])}")
    print(f"Verdict at {hhmm(start)}: {verdict(results)}")
    go = sweep[sweep["verdict"] == "GO"]["start_h"]
    if len(go):
        print(f"GO start window: {hhmm(go.min())}-{hhmm(go.max())}; best start {hhmm(best_h)} ({best_level})")
    else:
        print(f"No GO start time; best available: {best_level}")
    print(f"Outputs in {a.out}")


if __name__ == "__main__":
    main()
