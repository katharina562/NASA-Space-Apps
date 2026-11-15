"""Checks for the two-mode simulation on a synthetic terrain. Run: python test_sim.py"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim  # noqa: E402

NAV = {"base_fix_m": 1.0, "rover_drift_frac": 0.02, "walk_drift_frac": 0.03,
       "time_drift_m_per_min": 0.2, "dock_m": 3.0, "dock_add_m": 0.5}
COMMS = {"base_antenna_m": 5.0, "rover_antenna_m": 2.5, "suit_antenna_m": 2.0,
         "suit_rover_range_m": 1500.0, "rover_base_range_m": 10000.0}


def flat(n=60, cell=10.0):
    return sim.Terrain(np.zeros((n, n)), cell, cell)


def leg(kind, a, b, dur_s, name, **kw):
    d = dict(kind=kind, name=name, a=a, b=b, dur_s=dur_s, duration_min=dur_s / 60,
             length_m=math.dist(a, b) * 10, dz_m=0.0, terrain="plain")
    d.update(kw)
    return d


def mode_b_legs():
    """Base (10,10) -> drive to park (10,30) -> walk to (40,30) -> work -> walk back -> drive home."""
    return [
        leg("drive", (10, 10), (10, 30), 200, "Drive out"),
        leg("walk", (10, 30), (40, 30), 600, "Walk in"),
        leg("stop", (40, 30), (40, 30), 300, "Work at destination"),
        leg("walk", (40, 30), (10, 30), 600, "Walk back"),
        leg("drive", (10, 30), (10, 10), 200, "Drive back"),
    ]


def test_links_flat_obstructed_unknown_and_range():
    T = flat()
    P, Q = np.array([[10.0, 10.0]]), np.array([[10.0, 50.0]])
    assert T.links(P, 2, Q, 2, 1e4)[0] == sim.OK
    z = np.zeros((60, 60))
    z[:, 30] = 50.0                                      # a 50 m wall between them
    assert sim.Terrain(z, 10, 10).links(P, 2, Q, 2, 1e4)[0] == sim.DOWN
    z = np.zeros((60, 60))
    z[:, 30] = np.nan                                    # a gap in the elevation data
    assert sim.Terrain(z, 10, 10).links(P, 2, Q, 2, 1e4)[0] == sim.UNKNOWN
    assert T.links(P, 2, np.array([[10.0, 80.0]]), 2, 1e4)[0] == sim.UNKNOWN   # outside the data
    assert T.links(P, 2, Q, 2, 100.0)[0] == sim.DOWN     # 400 m apart, 100 m range
    # a tall base antenna does not see through a mountain
    z = np.zeros((60, 60))
    z[:, 30] = 500.0
    assert sim.Terrain(z, 10, 10).links(P, 100, Q, 2, 1e4)[0] == sim.DOWN


def test_metres_to_map_units():
    T = sim.Terrain(np.zeros((10, 10)), 15.0, 15.0, kx=1 / math.cos(math.radians(4.8)), ky=1.0)
    dr, dc = T.offset_rc(east_m=100.0, north_m=100.0)
    assert abs(dc * 15.0 - 100.0 / math.cos(math.radians(4.8))) < 1e-9   # map x stretched by 1/cos(lat)
    assert abs(dr * 15.0 + 100.0) < 1e-9                                 # north = up = smaller row
    assert abs(T.ground_m((0, 0), (0, dc))[0] - 100.0) < 1e-9


def test_mode_a_colocated():
    legs = [leg("drive", (10, 10), (40, 40), 600, "Drive out"),
            leg("stop", (40, 40), (40, 40), 300, "Work at destination", together=True),
            leg("drive", (40, 40), (10, 10), 600, "Drive back")]
    tl = sim.build_timeline(legs, (10, 10), flat(), NAV, COMMS)
    assert np.allclose(tl["astro"], tl["rover"])
    assert (tl["ar"] == sim.OK).all()
    assert tl["sample"][-1] == "in rover"


def test_mode_b_rover_parked_while_walking():
    legs = mode_b_legs()
    tl = sim.build_timeline(legs, (10, 10), flat(), NAV, COMMS)
    on_foot = np.isin(tl["leg"], [1, 2, 3])
    assert np.allclose(tl["rover"][on_foot], [10, 30])
    assert not np.allclose(tl["astro"][on_foot], [10, 30])
    k = sim.index_at(tl, 1000)                            # at the work site
    assert tl["sample"][k] == "not collected"
    k = sim.index_at(tl, 1200)                            # walking back with the sample
    assert tl["sample"][k] == "carried by astronaut"
    assert tl["sample"][-1] == "in rover"


def test_obstruction_stops_live_updates_and_data_goes_stale():
    z = np.zeros((60, 60))
    z[25:28, 20:40] = 60.0                                # ridge between park (row 10) and work site (row 40)
    tl = sim.build_timeline(mode_b_legs(), (10, 10), sim.Terrain(z, 10, 10), NAV, COMMS)
    k = sim.index_at(tl, 1000)
    assert tl["ar"][k] == sim.DOWN and tl["rb"][k] == sim.OK and not tl["full"][k]
    assert tl["last_rx"][k] < k                          # base holds an older sample
    assert tl["t"][k] - tl["t"][tl["last_rx"][k]] > 300   # ... and it is stale
    # local estimate keeps updating while the base view is frozen
    assert not np.allclose(tl["est_astro"][k], tl["est_astro"][tl["last_rx"][k]])
    assert tl["full"][-1] and tl["last_rx"][-1] == len(tl["t"]) - 1


def test_uncertainty_grows_and_only_valid_fixes_reduce_it():
    z = np.zeros((60, 60))
    z[25:28, 20:40] = 60.0
    tl = sim.build_timeline(mode_b_legs(), (10, 10), sim.Terrain(z, 10, 10), NAV, COMMS)
    s, full = tl["sigma_astro"], tl["full"]
    walk = np.isin(tl["leg"], [1, 2, 3]) & (tl["fix_source"] == None)   # noqa: E711  away from the rover
    ks = np.where(walk)[0]
    # on foot the radius never shrinks, even when the radio comes back
    assert (np.diff(s[ks]) >= -1e-9).all()
    reconnect = [k for k in ks[1:] if full[k] and not full[k - 1]]
    assert reconnect, "test terrain should produce a reconnection on the way back"
    for k in reconnect:
        assert s[k] >= s[k - 1]
    # docking with the rover is a valid correction: radius drops to the rover's + dock_add
    k = sim.index_at(tl, 1700 + 10)
    assert tl["fix_source"][k] == "docked with rover"
    assert abs(s[k] - (tl["sigma_rover"][k] + NAV["dock_add_m"])) < 1e-9
    assert s[k] < s[ks[-1]]
    # back at the surveyed base point
    assert abs(s[-1] - NAV["base_fix_m"]) < 1e-9


def test_relay_rover_keeps_base_live_behind_a_ridge():
    z = np.zeros((60, 60))
    z[25:28, 20:40] = 60.0                                # ridge between parked crew rover and work site
    T = sim.Terrain(z, 10, 10)
    no_relay = sim.build_timeline(mode_b_legs(), (10, 10), T, NAV, COMMS)
    relay = sim.build_timeline(mode_b_legs(), (10, 10), T, NAV, COMMS, relay_rc=(26, 50))   # east of the ridge
    k = sim.index_at(no_relay, 1000)                      # astronaut at the work site
    assert not no_relay["full"][k]
    assert relay["ar"][k] == sim.DOWN                     # crew rover still blocked ...
    assert relay["aq"][k] == sim.OK and relay["qb"][k] == sim.OK
    assert relay["full"][k] and relay["path"][k] == "astronaut → relay rover → base"
    assert relay["full"].sum() > no_relay["full"].sum()
    # a relay rover with no line to the base adds nothing
    z2 = z.copy()
    z2[:, 45] = 80.0                                      # wall between relay and base
    blocked = sim.build_timeline(mode_b_legs(), (10, 10), sim.Terrain(z2, 10, 10), NAV, COMMS, relay_rc=(26, 50))
    assert blocked["qb"][k] == sim.DOWN and not blocked["full"][k]


def test_abort_returns_to_rover_and_switches_mode():
    legs = mode_b_legs()
    new, ok = sim.abort_to_rover(legs, 200 + 300)        # halfway through "Walk in"
    assert ok
    names = [l["name"] for l in new]
    assert names == ["Drive out", "Walk in", "Return to rover", "Drive back"]
    assert np.allclose(new[1]["b"], (25, 30)) and np.allclose(new[2]["a"], (25, 30))
    assert np.allclose(new[2]["b"], (10, 30))
    tl = sim.build_timeline(new, (10, 10), flat(), NAV, COMMS)
    assert tl["sample"][-1] == "not collected"
    assert np.allclose(tl["astro"][-1], (10, 10))
    assert not sim.abort_to_rover(legs, 100)[1]          # aboard: nothing to abort
    assert not sim.abort_to_rover(legs, 1300)[1]         # already walking back


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
