# Mars Navigator — ARES-X EVA planner

Streamlit EVA planner for a human Marswalk in the Mount Sharp boxwork (Gale Crater, Mars).
Choose **mode A (travel with rover)** or **mode B (explore on foot, rover parked)**, click a destination: the app
plans the round trip, checks it against the
astronaut's EVA budget (O2, cooling water, suit and rover battery, effort, daylight) and says
**GO / MARGINAL / NO-GO**, shows where the base has a **live radio link** to the astronaut, and the best time to leave.

Uses real NASA/USGS data (not for actual mission planning).

## Run

**Easiest:** double-click `run_windows.bat` (Windows) or run `bash run_mac_linux.sh` (macOS/Linux).
The first start installs the libraries (a few minutes); then the app opens at http://localhost:8501.

Manual:

```bash
cd "NASA Space Apps"
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Data (~50 MB; included in this repository)

The app uses the wide 16:9 window if it is there, otherwise the 3×3 km crop:

- `data/dem_wide.tif` + `data/hirise_wide.tif` — 6.4 × 3.6 km (16:9, fills a wide screen), made by
  `python prep/make_wide_crop.py` (streams only the window from the same USGS / HiRISE sources)
- `data/dem_crop.tif` + `data/hirise_crop.tif` — 3 × 3 km fallback
- Sources: USGS MSL Gale Merged DEM 1 m v3; HiRISE colour mosaic (NASA/JPL/UArizona)

The routing grid always uses ~15 m cells.

They come from `prep/cilli_prep.py` (folder `out/`) or from the `rasters/` folder of `ARES-X_dataset_full.zip`.

## How it works

- **Route:** DEM resampled to a ~15 m grid; slope per cell; A* shortest path through cells ≤ the slope limit (default 20°).
- **Hazard overlay:** yellow 50–75% of the limit, orange 75–100%, red above the limit (no-go).
- **Season and daylight:** the default orbital position (`Ls`) follows the current real Mars position using
  NASA's Mars24 algorithm. A manual override is in Mission controls → Environment. The app calculates
  geometric sunrise, sunset and Sun elevation at departure and return for Gale Crater; dust and terrain
  shadows are not included.
- **EVA budget** (`eva/eva_budget.py`, constants in `eva/assumptions.json`): out + work time + back.
  Apollo walking speed vs. slope, gravity-corrected Pandolf metabolic rate, O2 and cooling water,
  suit heat balance by time of sol, checks against 4 h, 2000/1600 BTU/h, 25% reserve and daylight.
  Three cases (best / nominal / worst): GO = worst case passes, MARGINAL = only nominal passes.
- **Boxwork ground** (rough at 1 m scale) walks slower and costs more energy in the model.
- **Mode A — travel with rover:** the astronaut rides all the way to the destination; rover and astronaut stay
  together. Only works if the rover can drive there.
- **Mode B — explore on foot:** ride to a parking spot, the rover stays parked as access point and radio relay
  (astronaut ↔ rover ↔ base), the astronaut walks the rest. Pick the spot with "Set rover parking", or let the
  app pick the quickest spot that can relay. The rover powers and charges the suit while riding.
- **Rover limits** (Mission controls → Terrain) are separate from the walking limit: max driving slope (default 15°)
  and "avoid boxwork". Battery and antenna values are placeholders in `eva/assumptions.json`.
- **Lead antenna (relay rover):** an uncrewed second rover drives to a high point hours before the EVA and stays
  parked all mission. Auto spot: drivable, sees the base and as many science stops as possible (move it with
  "Set relay rover", switch it off in Mission controls → Comms). Drill pair, crew rover parked near the base:
  live link 35% without it → 100% with it.
- **Live link:** the suit talks only to a rover; the base is live through astronaut → crew rover → base,
  → crew rover → relay rover → base, or → relay rover → base. Each is a straight line between the
  antennas over the elevation grid with Mars curvature, plus a maximum range (Mission controls → Comms).
  Simplified coverage model, not a radio propagation calculation.
- **Science stops** (yellow, click = go there) and Curiosity's real drive (white dashed) from the NASA waypoint file.
- Base camp starts at habitat B (lon 137.370908°E, lat −4.811979°); switch "Map click action" to move it.

Coordinates: planetocentric latitude, east-positive longitude, Mars sphere R = 3,396,190 m (not WGS84).

## Simulation (all simulated data)

Mission monitor (right): **Start / Pause / Resume / Reset**, speed 1–300×, and in mode B **Return to rover now**.
The astronaut and rover move along the planned routes with the EVA model's timing (simulated seconds, advanced by
wall time × speed; the map layer updates once per second without reloading the base map). Code: `sim.py`,
tests: `python test_sim.py`.

- **Markers:** base, destination, sample site (filled when collected), rover, astronaut estimate with its
  uncertainty circle, the true position as a small white dot (simulation only), and the base's last received
  position (grey, dashed) when the data is stale.
- **Links:** green = connected, red dashed = disconnected, grey dotted = unknown (outside the elevation data).
  The base gets live updates only when both links are up; otherwise it keeps the last received position and
  readings and shows their age as STALE. The astronaut's own estimate keeps updating.
- **Position uncertainty (Fehlerkreis):** radius in ground metres, **illustrative** (not a confidence level).
  Grows without a valid correction: rover 2% of distance driven; astronaut on foot 3% of distance walked +
  0.2 m/min. Valid corrections only: the surveyed base point (1 m) and docking with the rover (within 3 m: the
  astronaut takes the rover's estimate + 0.5 m). A radio reconnection never corrects the position. Circles are
  converted to map units with the grid's real scale (east–west × 1/cos(latitude) for the equirectangular grid).
  Constants: `navigation` in `eva/assumptions.json`.
- **Readings:** O2, water and suit battery come from the existing EVA model (planned, nominal), not from sensors.

**Limits:** positions follow the plan exactly (no navigation errors acted on, no detours); the uncertainty model is
linear and illustrative; radio = line of sight + range only (no Fresnel zone, diffraction, multipath, link budget);
the 15 m grid smooths the 1–2 m boxwork ridges, so small obstructions are missed; cameras and motion sensors are
concept hardware and not simulated; no beacons or satellites; the relay rover's own battery is not budgeted. "Return to rover" re-runs the EVA model
for the shortened trip, but the dialog numbers stay those of the planned EVA.

## Curiosity data on the map (real NASA data)

- **Whole drive inside the map** (white dashed, sols 4080 → today): `python prep/curiosity_track.py` reads the MMGIS
  waypoint file (`out/waypoints/MSL_waypoints.json`, NASA/JPL) → `data/curiosity_track.csv`.
- **81 Navcam photos** at the spot where they were taken (sols 4466–4955): `python prep/rover_photos.py`.
  Rover position = waypoint with the same site/drive; camera direction and field of view from each frame's CAHVOR
  camera model + rover attitude (NASA raw-image API). The blue area is the ground the photo shows: inside the
  vertical field of view and in line of sight over the 1 m DEM. Click it to open the photo (thumbnails in
  `static/photos/`, served by Streamlit static serving, see `.streamlit/config.toml`). Credit NASA/JPL-Caltech.
  Toggle both in Mission controls → Terrain.
- **Image/elevation alignment:** the wide HiRISE window is warped onto the DEM grid and shifted by the measured
  6 m E / 2 m N offset (phase correlation, now 0–2 m everywhere). Grey strips of the HiRISE mosaic (~8%) are
  colourised for display: their brightness is real, the tint is borrowed from the coloured parts.

## Science mission and thermal data

- **SCIENCE MISSION · GROUNDWATER** (button, top centre): shows our main mission (how long groundwater lasted under
  Mount Sharp, from NASA's 2025–26 boxwork results) and the three sample sites (yellow), which are only shown while
  the mission is on.
- **Map layer "Thermal inertia (sand vs rock)"**: THEMIS night-time thermal inertia, 100 m pixels (USGS Astrogeology
  global mosaic, tile 30S120E, ~20 % accuracy), cut with `python prep/themis_ti.py` → `data/themis_ti.tif`.
  Low (blue) = loose sand that cools fast at night, high (red) = rock that stays warm. A hint only.
- Seam lines along the colourised strips are faded (`remove_seams` in `prep/make_wide_crop.py`).
