# ARES-X update for the team repo (15 Nov 2026)

Copy the whole `NASA Space Apps/` folder over the repo folder (same layout as Cilli's version). Your
`data/dem_crop.tif` and `data/hirise_crop.tif` can stay: they are the fallback if the wide files are missing.
Run as before: `run_windows.bat` / `bash run_mac_linux.sh` (or `streamlit run app.py` in this folder; the
`.streamlit/config.toml` turns on static serving for the photo thumbnails).

## What is new (all on top of Cilli's layout)
- **Merge:** Cilli's full-screen map, menu, dock, dialog and Mars24 season + our EVA model, rover, batteries,
  radio, stops, track, reserves, timeline (`CHANGES.md`).
- **Modes A/B:** A = ride the rover all the way, B = rover parked as relay, walk the rest; rover limits separate
  from walking; "Set rover parking"; clear "ROVER CAN'T REACH" message + "Switch to mode B".
- **Lead antenna:** uncrewed relay rover deployed before the EVA. It picks its spot automatically from the
  route to the chosen destination; `Set relay rover` on the map overrides it, "back to auto spot" (Comms tab) resets.
- **Simulation** (`sim.py`, tests `python test_sim.py`): Start/Pause/Reset/speed, Return to rover, uncertainty
  circle, stale base data, link states; mission monitor panel.
- **Wide 16:9 map** 6.1 × 3.6 km (`data/dem_wide.tif`, `data/hirise_wide.tif`), image aligned to the DEM
  (0–2 m), grey strips colourised, seam lines faded; map always fills the screen.
- **Curiosity data:** whole drive inside the map (`data/curiosity_track.csv`, sols 4080–5073) and 81 Navcam
  photos with view wedges (`data/rover_photos.json`, `static/photos/`).
- **Science mission button** (groundwater) with the sample sites; **thermal inertia layer** (`data/themis_ti.tif`).
- **3D-look icons** for base, crew rover, relay rover, astronaut.

`prep/` holds the scripts that made the data (they expect Zakariae's folder layout; the outputs are already
included, so you don't need to run them). Credits: NASA/JPL-Caltech, USGS Astrogeology, HiRISE (UArizona),
THEMIS (ASU).

## Uploading to GitHub
The two map files `data/dem_wide.tif` (34 MB) and `data/hirise_wide.tif` (40 MB) are over GitHub's 25 MB
browser-upload limit, so push with git or GitHub Desktop (limit 100 MB per file), not drag-and-drop.
