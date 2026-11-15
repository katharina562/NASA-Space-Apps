"""ARES-X two-mode EVA simulation: astronaut + rover + base.

EVERYTHING HERE IS SIMULATED. Positions, link states and position uncertainty are
model outputs for planning and demonstration, not measurements.

Grid convention: a position is (row, col) as floats on the app's routing grid;
integer values are cell centres. `dx`, `dy` are map units per cell, `kx`, `ky`
are map units per ground metre (the HiRISE/DEM grid is equirectangular with
lat_ts = 0, so one ground metre east-west is 1/cos(lat) map units).

Pure numpy / scipy: no Streamlit, so it can be unit tested (test_sim.py).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import map_coordinates

MARS_R = 3396190.0
OK, DOWN, UNKNOWN = "connected", "disconnected", "unknown"


# ------------------------------------------------------------------ terrain --

class Terrain:
    """Elevation grid + scale. NaN elevation = no data (link state 'unknown')."""

    def __init__(self, elevation, dx, dy, kx=1.0, ky=1.0):
        self.z = np.asarray(elevation, dtype=np.float64)
        self.rows, self.cols = self.z.shape
        self.nodata = (~np.isfinite(self.z)).astype(np.float64)
        fill = np.nanmin(self.z) if np.isfinite(self.z).any() else 0.0
        self.z_filled = np.where(np.isfinite(self.z), self.z, fill)
        self.dx, self.dy, self.kx, self.ky = float(dx), float(dy), float(kx), float(ky)

    def ground_m(self, P, Q):
        """Ground distance in metres between grid positions (arrays of shape (N, 2) or (2,))."""
        P, Q = np.atleast_2d(P).astype(float), np.atleast_2d(Q).astype(float)
        e = (Q[:, 1] - P[:, 1]) * self.dx / self.kx
        n = (Q[:, 0] - P[:, 0]) * self.dy / self.ky
        return np.hypot(e, n)

    def offset_rc(self, east_m, north_m):
        """Ground offset (metres east, north) -> grid offset (d_row, d_col)."""
        return -north_m * self.ky / self.dy, east_m * self.kx / self.dx

    def _inside(self, P):
        return (P[:, 0] >= -0.5) & (P[:, 0] <= self.rows - 0.5) & (P[:, 1] >= -0.5) & (P[:, 1] <= self.cols - 0.5)

    def links(self, P, h_p, Q, h_q, max_range_m, colocated_m=1.0):
        """Predicted radio-link state for each pair P[i] <-> Q[i].

        Simplified coverage model, NOT a radio propagation calculation:
        straight line between the antennas (heights h_p, h_q above ground) with Mars
        curvature, blocked if terrain rises above it, plus a maximum range.
          * beyond max range                      -> disconnected
          * an endpoint outside the elevation data -> unknown
          * terrain above the line (known data)    -> disconnected
          * line crosses missing data              -> unknown
          * otherwise                              -> connected
        Pairs closer than `colocated_m` (e.g. astronaut aboard the rover) are connected.
        """
        P, Q = np.atleast_2d(P).astype(float), np.atleast_2d(Q).astype(float)
        N = len(P)
        dist = self.ground_m(P, Q)
        state = np.full(N, OK, dtype=object)
        if N == 0:
            return state

        ends_ok = self._inside(P) & self._inside(Q)
        nd_p = map_coordinates(self.nodata, P.T, order=0, mode="constant", cval=1.0) > 0.5
        nd_q = map_coordinates(self.nodata, Q.T, order=0, mode="constant", cval=1.0) > 0.5
        ends_ok &= ~nd_p & ~nd_q

        z0 = map_coordinates(self.z_filled, P.T, order=1, mode="nearest") + h_p
        z1 = map_coordinates(self.z_filled, Q.T, order=1, mode="nearest") + h_q
        span = np.max(np.hypot(Q[:, 0] - P[:, 0], Q[:, 1] - P[:, 1]))
        n = max(2, int(math.ceil(span * 2)))                  # ~2 samples per grid cell
        ts = np.linspace(0.0, 1.0, n + 1)[1:-1]
        blocked = np.zeros(N, dtype=bool)
        missing = np.zeros(N, dtype=bool)
        for t in ts:
            r = P[:, 0] + t * (Q[:, 0] - P[:, 0])
            c = P[:, 1] + t * (Q[:, 1] - P[:, 1])
            nd = map_coordinates(self.nodata, [r, c], order=0, mode="constant", cval=1.0) > 0.5
            ground = map_coordinates(self.z_filled, [r, c], order=1, mode="nearest")
            beam = z0 + t * (z1 - z0) - t * (1 - t) * dist ** 2 / (2 * MARS_R)
            blocked |= (ground > beam) & ~nd
            missing |= nd

        state[~ends_ok] = UNKNOWN
        state[ends_ok & missing] = UNKNOWN
        state[ends_ok & blocked] = DOWN            # known terrain block wins over missing data
        state[dist > max_range_m] = DOWN           # out of range regardless of terrain
        state[dist < colocated_m] = OK
        return state


# ------------------------------------------------------------------- legs ----

def abort_to_rover(legs, t_abort_s):
    """'Return to the rover now' at time t_abort_s (mode B, while on foot).

    `legs`: list of dicts with kind (walk/drive/stop), name, a, b (grid positions),
    length_m, dz_m, terrain, duration_min and dur_s (from the EVA model).
    Returns (new_legs, ok). The astronaut stops where they are, walks the path they
    came on back to the parked rover (same legs reversed), skips the rest of the
    on-foot work, then the planned drive back follows.
    """
    t, i = 0.0, 0
    while i < len(legs) and t + legs[i]["dur_s"] <= t_abort_s:
        t += legs[i]["dur_s"]
        i += 1
    if i >= len(legs):
        return legs, False
    cur = legs[i]
    if cur["kind"] == "drive" or cur["name"] == "Walk back":
        return legs, False                          # aboard already, or already returning
    later_drive = [j for j in range(i, len(legs)) if legs[j]["kind"] == "drive"]
    if not later_drive:
        return legs, False                          # no rover to return to

    out = [dict(l) for l in legs[:i]]
    f = (t_abort_s - t) / max(cur["dur_s"], 1e-9)
    if cur["kind"] == "walk" and f > 0:
        a, b = np.asarray(cur["a"], float), np.asarray(cur["b"], float)
        out.append(dict(cur, b=tuple(a + f * (b - a)), length_m=cur["length_m"] * f, dz_m=cur["dz_m"] * f,
                        duration_min=cur["duration_min"] * f, dur_s=cur["dur_s"] * f))
    elif cur["kind"] == "stop" and f > 0:
        out.append(dict(cur, name="Work (aborted)", duration_min=(t_abort_s - t) / 60, dur_s=t_abort_s - t))

    last_drive = max([j for j, l in enumerate(out) if l["kind"] == "drive"], default=-1)
    walked = [l for l in out[last_drive + 1:] if l["kind"] == "walk"]
    back = [dict(l, name="Return to rover", a=l["b"], b=l["a"], dz_m=-l["dz_m"])
            for l in reversed(walked)]
    out += back + [dict(l) for l in legs[later_drive[0]:]]
    return out, True


# ------------------------------------------------------------ simulation ----

def network_links(terrain: Terrain, astro, rover, base_rc, comms, nav, relay_rc=None):
    """Link states for every sample and whether the base gets the astronaut's data.

    Nodes: astronaut (A), crew rover (C), relay rover (Q, optional, parked all mission), base (B).
    The suit radio talks only to a rover (access point), never straight to the base.
    Base is live if  A-C up and (C-B up or C-Q-B up)   or   A-Q up and Q-B up.
    """
    astro = np.atleast_2d(np.asarray(astro, float))
    rover = np.atleast_2d(np.asarray(rover, float))
    n = len(astro)
    base = np.repeat(np.asarray(base_rc, float)[None], n, 0)
    sh, rh, bh = comms["suit_antenna_m"], comms["rover_antenna_m"], comms["base_antenna_m"]
    ar = terrain.links(astro, sh, rover, rh, comms["suit_rover_range_m"], colocated_m=nav["dock_m"])
    rb = terrain.links(rover, rh, base, bh, comms["rover_base_range_m"], colocated_m=nav["dock_m"])
    none = np.full(n, None, dtype=object)
    if relay_rc is None:
        full = (ar == OK) & (rb == OK)
        path = np.where(full, "astronaut → rover → base", "none")
        return {"ar": ar, "rb": rb, "aq": none, "rq": none, "qb": none, "full": full, "path": path}
    relay = np.repeat(np.asarray(relay_rc, float)[None], n, 0)
    aq = terrain.links(astro, sh, relay, rh, comms["suit_rover_range_m"], colocated_m=nav["dock_m"])
    rq = terrain.links(rover, rh, relay, rh, comms["rover_base_range_m"], colocated_m=nav["dock_m"])
    qb = terrain.links(relay, rh, base, bh, comms["rover_base_range_m"], colocated_m=nav["dock_m"])
    a_c, c_b, a_q, c_q, q_b = (ar == OK), (rb == OK), (aq == OK), (rq == OK), (qb == OK)
    via_c = a_c & c_b
    via_cq = a_c & c_q & q_b & ~via_c
    via_q = a_q & q_b & ~via_c & ~via_cq
    full = via_c | via_cq | via_q
    path = np.full(n, "none", dtype=object)
    path[via_c] = "astronaut → rover → base"
    path[via_cq] = "astronaut → rover → relay rover → base"
    path[via_q] = "astronaut → relay rover → base"
    return {"ar": ar, "rb": rb, "aq": aq, "rq": rq, "qb": qb, "full": full, "path": path}


def build_timeline(legs, base_rc, terrain: Terrain, nav: dict, comms: dict, dt_s: float = 5.0, seed: int = 7,
                   relay_rc=None):
    """Sample the EVA every dt_s seconds of simulated time.

    `legs` carry timing and consumables from the EVA model (dur_s, o2_kg, water_kg,
    suit_battery_pct). Rover and astronaut move together on drive legs, the rover
    stays parked on walk legs, both stand still on stops.

    Navigation (illustrative, linear growth, no statistical confidence level):
      rover radius  += rover_drift_frac * distance driven
      on foot       += walk_drift_frac * distance walked + time_drift_m_per_min * minutes
      at the base (surveyed point)        -> radius = base_fix_m       (valid correction)
      astronaut docked with / aboard rover -> astronaut estimate = rover estimate,
                                             radius = rover radius + dock_add_m (valid correction)
      Radio contact alone never corrects the position.
    """
    base_rc = np.asarray(base_rc, float)
    ends = np.cumsum([l["dur_s"] for l in legs]) if legs else np.array([0.0])
    T = float(ends[-1]) if legs else 0.0
    t = np.arange(0.0, T + dt_s, dt_s)
    t[-1] = T
    N = len(t)

    astro = np.zeros((N, 2))
    rover = np.zeros((N, 2))
    leg_i = np.zeros(N, dtype=int)
    starts = np.concatenate([[0.0], ends[:-1]]) if legs else np.array([0.0])

    # rover position at the start of each leg (where it was last driven to)
    rover_at = []
    pos = base_rc.copy()
    for l in legs:
        rover_at.append(pos.copy())
        if l["kind"] == "drive":
            pos = np.asarray(l["b"], float)

    for k, tk in enumerate(t):
        i = int(min(np.searchsorted(ends, tk, side="right"), len(legs) - 1)) if legs else 0
        leg_i[k] = i
        if not legs:
            astro[k] = rover[k] = base_rc
            continue
        l = legs[i]
        f = 0.0 if l["dur_s"] <= 0 else min(max((tk - starts[i]) / l["dur_s"], 0.0), 1.0)
        a, b = np.asarray(l["a"], float), np.asarray(l["b"], float)
        p = a + f * (b - a)
        astro[k] = p
        rover[k] = p if l["kind"] == "drive" else rover_at[i]

    # ---- navigation estimate (illustrative)
    rng = np.random.default_rng(seed)
    sig_r = np.zeros(N)
    sig_a = np.zeros(N)
    est_a = np.zeros((N, 2))
    est_r = np.zeros((N, 2))
    fix_src = np.empty(N, dtype=object)
    last_fix_t = np.zeros(N)
    phi_r, phi_a = rng.uniform(0, 2 * math.pi, 2)
    sr = sa = nav["base_fix_m"]
    lf = 0.0
    work_end = min([e for l, e in zip(legs, ends) if l["name"] == "Work at destination"], default=math.inf)
    sample = np.empty(N, dtype=object)
    in_rover = False
    for k in range(N):
        if k:
            d_r = float(terrain.ground_m(rover[k - 1], rover[k])[0])
            d_a = float(terrain.ground_m(astro[k - 1], astro[k])[0])
            sr += nav["rover_drift_frac"] * d_r
            phi_r += rng.normal(0, 0.15)
            phi_a += rng.normal(0, 0.15)
        at_base_r = float(terrain.ground_m(rover[k], base_rc)[0]) <= nav["dock_m"]
        docked = float(terrain.ground_m(astro[k], rover[k])[0]) <= nav["dock_m"]
        at_base_a = float(terrain.ground_m(astro[k], base_rc)[0]) <= nav["dock_m"]
        src = None
        if at_base_r:
            sr = nav["base_fix_m"]
        if at_base_a:
            sa, src = nav["base_fix_m"], "base survey point"
        elif docked:
            sa, src = sr + nav["dock_add_m"], "docked with rover"
        elif k:
            sa += nav["walk_drift_frac"] * d_a + nav["time_drift_m_per_min"] * (t[k] - t[k - 1]) / 60.0
        if src is not None:
            lf = t[k]
        sig_r[k], sig_a[k], fix_src[k], last_fix_t[k] = sr, sa, src, lf
        if t[k] >= work_end:
            in_rover = in_rover or docked
            sample[k] = "in rover" if in_rover else "carried by astronaut"
        else:
            sample[k] = "not collected"

        er = terrain.offset_rc(0.6 * sr * math.cos(phi_r), 0.6 * sr * math.sin(phi_r))
        est_r[k] = rover[k] + er
        if docked or at_base_a:
            est_a[k] = est_r[k] if not at_base_a else astro[k] + terrain.offset_rc(
                0.6 * sa * math.cos(phi_a), 0.6 * sa * math.sin(phi_a))
        else:
            est_a[k] = astro[k] + terrain.offset_rc(0.6 * sa * math.cos(phi_a), 0.6 * sa * math.sin(phi_a))

    # ---- predicted links (true positions)
    L = network_links(terrain, astro, rover, base_rc, comms, nav, relay_rc)
    ar, rb, full = L["ar"], L["rb"], L["full"]
    last_rx = np.full(N, -1, dtype=int)
    cur = -1
    for k in range(N):
        if full[k]:
            cur = k
        last_rx[k] = cur

    # ---- EVA model consumables, interpolated inside each leg
    o2 = np.zeros(N)
    water = np.zeros(N)
    suit = np.zeros(N)
    if legs:
        cum_o2 = np.concatenate([[0.0], np.cumsum([l.get("o2_kg", 0.0) for l in legs])])
        cum_w = np.concatenate([[0.0], np.cumsum([l.get("water_kg", 0.0) for l in legs])])
        batt = [100.0] + [l.get("suit_battery_pct", 100.0) for l in legs]
        for k in range(N):
            i = leg_i[k]
            f = 0.0 if legs[i]["dur_s"] <= 0 else min(max((t[k] - starts[i]) / legs[i]["dur_s"], 0.0), 1.0)
            o2[k] = cum_o2[i] + f * (cum_o2[i + 1] - cum_o2[i])
            water[k] = cum_w[i] + f * (cum_w[i + 1] - cum_w[i])
            suit[k] = batt[i] + f * (batt[i + 1] - batt[i])

    return {
        "t": t, "T": T, "astro": astro, "rover": rover, "leg": leg_i,
        "est_astro": est_a, "est_rover": est_r, "sigma_astro": sig_a, "sigma_rover": sig_r,
        "fix_source": fix_src, "last_fix_t": last_fix_t, "sample": sample, "work_end": work_end,
        "ar": ar, "rb": rb, "full": full, "last_rx": last_rx,
        "aq": L["aq"], "rq": L["rq"], "qb": L["qb"], "path": L["path"],
        "o2_used_kg": o2, "water_used_kg": water, "suit_battery_pct": suit,
    }


def index_at(tl, t_s):
    """Timeline sample at or just before simulated time t_s."""
    return int(np.clip(np.searchsorted(tl["t"], t_s, side="right") - 1, 0, len(tl["t"]) - 1))


def phase(legs, leg_i, t_s, T):
    """(astronaut status, rover status) in plain words."""
    if t_s >= T:
        return "back at base · EVA complete", "at base"
    if t_s <= 0:
        return "at base · aboard, ready", "at base · ready"
    l = legs[leg_i]
    name, kind = l["name"], l["kind"]
    if kind == "drive":
        return "aboard rover (riding)", "driving with astronaut aboard"
    rover_word = "parked · access point + relay" if any(x["kind"] == "drive" for x in legs[:leg_i]) else "at base"
    if kind == "stop":
        together = l.get("together", False)
        word = "collecting sample next to rover" if together else (
            "work aborted" if "aborted" in name else "collecting sample")
        return word, ("parked at work site" if together else rover_word)
    return {"Walk in": "walking to the sample site", "Walk back": "walking back to the rover",
            "Return to rover": "returning to rover early", "Walk out": "walking out",
            }.get(name, "walking"), rover_word
