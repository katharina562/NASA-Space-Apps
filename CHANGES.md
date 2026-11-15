# Changes (15 Nov): lead antenna + clearer modes

- **Lead antenna / relay rover**: second, uncrewed rover deployed before the EVA, parked all mission on a drivable
  spot that sees the base and the science stops (`relay_spot`, `sim.network_links`). Own icon (tall mast, radio
  waves, RELAY plate); new habitat icon for the base (BASE plate). Signal map shows where the astronaut has a link
  via the relay. Test: `test_relay_rover_keeps_base_live_behind_a_ridge`.
- Mode A that the rover can't reach now says "ROVER CAN'T REACH · WALKING WORKS", names the reason (boxwork /
  slope) and offers a SWITCH TO MODE B button. "Set rover parking" in mode A switches to B.
- Moved parking / relay spots: your click stays as a grey pin with a dashed line to the real spot, reason in the
  bottom bar. Fixed the mode radio showing A while the app ran B.

# Changes (14 Nov, evening): merge + two-mode simulation

Base = Cilli's version (full-screen map, Mission controls menu, status chip, bottom dock, details pop-up,
Mars24 season + sunrise/sunset). Added on top:

- **From the rover version:** science stops + Curiosity's track, rover routing (Dijkstra), suit/rover batteries,
  radio line of sight, reserve gauges, EVA timeline, mode comparison, scale bar, zoom-out fix. Shown in the
  details pop-up (tabs) and the dock (live link %).
- **Two operating modes:** A = travel with rover (together all the way), B = explore on foot with the rover
  parked as access point and relay. Rover parking can be set on the map. Rover slope limit and "avoid boxwork"
  are separate settings from the walking limit.
- **Comms model:** astronaut ↔ rover and rover ↔ base checked separately (line of sight + max range, antenna
  heights and ranges in Mission controls → Comms). The planned live link % uses the same model.
- **Simulation** (`sim.py`, tests `test_sim.py`): Start / Pause / Resume / Reset, speed, Return to rover;
  uncertainty circle, stale base data, link states connected / disconnected / unknown. See README.
- `eva_budget.py`: Cilli's Mars24 Ls, solar declination and sunrise/sunset functions merged with our
  rover/battery code; new test. `assumptions.json`: comms ranges + `navigation` block.
- Satellite image sent as WebP (the map is re-sent on every simulation tick).
- `requirements.txt`: Streamlit >= 1.42 (pop-up, keyed containers).

# Changes (Zakariae, 14 Nov)

Replace the files in `NASA Space Apps/` with these. The `data/` .tif files are not included (put `dem_crop.tif` and `hirise_crop.tif` in `data/`, see README).

## Bug fixes
1. **Hazard overlay was always invisible**: the yellow/orange/red slope classes were computed but never drawn. Now painted (yellow 50–75% of the limit, orange 75–100%, red above = no-go), with a legend in the sidebar.
2. **`requirements.txt` missing `rasterio`** (crash on fresh install); also added `scipy` and `pandas`.
3. **Data paths** were relative to the folder you start Streamlit from; now relative to `app.py`.
4. **Satellite image was grey** (band 1 only); now HiRISE colour.
5. **Default base camp** was the map centre (inside the boxwork); now habitat B.
6. **"No safe route" message** now says why: destination too steep, base camp too steep, outside the data, or cut off by steep terrain.
7. `stauts.py` pointed to `data/mars_ortho.tif`; now `data/hirise_crop.tif`.
8. README described synthetic data and a wrong folder; rewritten.
9. **Map could not zoom out**: folium only passes `min_zoom`/`max_zoom` to a tile layer, and with `tiles=None` they were dropped, so Leaflet stopped at zoom 0 (1 px = 1 m) and the 3 km area never fit on screen. Now set on the map itself (−4 to 3), and panning stays on the Mars image.

## Map improvements
- Metric scale bar (map units are metres).
- Curiosity's real drive (dashed white) and the three science stops (yellow) as layers; layer box (top right) toggles them and the overlays.
- Clicking a science stop routes exactly to it (snaps to the stop).
- Needs `data/boxwork_waypoints.csv` (included) for Curiosity's track.

## New: rover, batteries and live link to base
- **Travel mode** in the sidebar: 🚶 on foot or 🚙 with rover.
- **Rover plan:** the rover drives (≤ 15° slopes, no boxwork ridges), parks, and the astronaut walks the rest. Fastest
  plan from a Dijkstra travel-time search from the base (driving) and from the destination (walking).
- **Relay parking:** the rover parks on a spot whose mast sees both the base and the work site, so the base gets the
  astronaut's voice, position and suit data live. It takes the fastest such spot; if none exists the app says so.
- **Batteries:** suit battery (drains on foot, charges while riding) and rover battery (driving energy from rolling
  resistance and climbing, relay radio, suit charging). Both checked against the 25% reserve.
- **Live link to base:** radio line of sight on the elevation grid (with Mars curvature). The route is solid where the
  base has a live link and dotted where it doesn't; "Radio shadow" layer shows the dead zones.
- **On foot vs. with rover** table for the same destination (time, O2, water, live link, status).
- Rover, battery and antenna numbers are placeholders in `eva/assumptions.json` (LRV-class: 8 kWh, ~5 km/h) for Marwane.

## New: mission-tool layout
- Status row: GO/NO-GO, time outside, distance, live link %, best start time.
- Plan panel: verdict with the reason, reserve gauges (O2, water, suit and rover battery) with the 25% mark,
  EVA timeline (each phase with time, distance and link status), walk-vs-rover comparison, best-start chart.
- Dark theme, short sidebar (advanced settings folded away), map legend under the map.
- The rover is drawn as a six-wheeled Mars rover (mast camera, antenna dish, solar panel, rocker-bogie) on the map,
  in the relay banner, legend and comparison table; it waits at the base until a rover trip is planned.

## New: EVA budget for the round trip
- New folder `eva/` (`eva_budget.py` + `assumptions.json`). Both files are needed.
- After a route is found the app runs base → destination → work time → base through the EVA model and shows **GO / MARGINAL / NO-GO**, round-trip time, cooling water, O2 and peak metabolic rate (best–worst range on hover), and the reason if it fails.
- Sidebar "EVA Plan": start time slider and work time at the destination.
- "When to go": GO window and best start time, with a cooling-water vs. start-time chart.
- Boxwork ground (rough at 1 m scale) is detected from the full DEM and walks slower in the model.

The A* routing itself is unchanged.
