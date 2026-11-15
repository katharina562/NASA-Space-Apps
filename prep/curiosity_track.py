"""Curiosity's whole drive inside the app's map window, from the NASA MMGIS waypoint file.

Keeps every end-of-drive waypoint that falls inside the map (bounds of data/dem_wide.tif), in sol order,
and numbers the stretches: a new segment starts whenever the rover leaves the map and comes back.

Run:  python prep/curiosity_track.py      (waypoint file: out/waypoints/MSL_waypoints.json)
Out:  share/team_app/data/curiosity_track.csv
"""
import csv
import json
import math
from pathlib import Path

import rasterio

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "share" / "team_app"
MARS_R = 3396190.0

with rasterio.open(APP / "data" / "dem_wide.tif") as s:
    b = s.bounds
feats = json.loads((ROOT / "out" / "waypoints" / "MSL_waypoints.json").read_text())["features"]
pts = sorted((f["properties"] for f in feats), key=lambda p: (p["sol"], p["site"], p["drive"]))

rows, seg, prev_in = [], 0, False
for p in pts:
    x, y = MARS_R * math.radians(p["lon"]), MARS_R * math.radians(p["lat"])
    inside = b.left <= x <= b.right and b.bottom <= y <= b.top
    if inside and not prev_in:
        seg += 1
    if inside:
        rows.append({"segment": seg, "sol": p["sol"], "site": p["site"], "drive": p["drive"],
                     "lat": p["lat"], "lon": p["lon"], "elev_geoid_m": p["elev_geoid"]})
    prev_in = inside

with open(APP / "data" / "curiosity_track.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} waypoints inside the map, sols {rows[0]['sol']}-{rows[-1]['sol']}, {seg} segment(s)")
print("first:", rows[0]["lat"], rows[0]["lon"], " last:", rows[-1]["lat"], rows[-1]["lon"])
