
import copy
import os
import json
import math
import heapq
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import folium
import streamlit as st

from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform as warp_transform
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter, uniform_filter, map_coordinates
from branca.element import MacroElement
from jinja2 import Template
from streamlit_folium import st_folium

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "eva"))
sys.path.insert(0, str(HERE))
import eva_budget as eb  # noqa: E402  EVA budget model (Zakariae)
import sim  # noqa: E402  two-mode astronaut + rover simulation (simulated data only)
import time  # noqa: E402

st.set_page_config(
    page_title="Mars Navigator",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    [data-testid="stHeader"], [data-testid="stToolbar"], footer {
        display: none !important;
    }
    [data-testid="stAppViewContainer"], .stApp {
        overflow: hidden !important;
    }
    .block-container {
        max-width: 100% !important;
        padding: 0.35rem 0.5rem 0 !important;
    }
    [data-testid="stVerticalBlock"] {
        gap: 0.35rem !important;
    }
    [data-testid="stLayoutWrapper"]:has(> [data-testid="stPopover"]) {
        position: fixed !important;
        top: 12px;
        left: 12px;
        z-index: 10000;
        width: auto !important;
    }
    [data-testid="stPopover"] button {
        background: rgba(10, 16, 22, .94) !important;
        color: #e8f0f2 !important;
        border: 1px solid #34424d !important;
        border-radius: 2px !important;
        font-family: monospace !important;
        letter-spacing: .04em;
    }
    [data-testid="stPopoverBody"] {
        background: rgba(10, 16, 22, .98) !important;
        color: #e8f0f2 !important;
        border: 1px solid #34424d !important;
        border-radius: 2px !important;
        box-shadow: 0 12px 32px rgba(0, 0, 0, .48) !important;
    }
    [data-testid="stPopoverBody"] p,
    [data-testid="stPopoverBody"] label,
    [data-testid="stPopoverBody"] span,
    [data-testid="stPopoverBody"] h1,
    [data-testid="stPopoverBody"] h2,
    [data-testid="stPopoverBody"] h3 {
        color: #dce6e9 !important;
    }
    [data-testid="stPopoverBody"] [role="tab"] {
        color: #91a0ad !important;
        font-family: monospace !important;
    }
    [data-testid="stPopoverBody"] [role="tab"][aria-selected="true"] {
        color: #54d6e8 !important;
    }
    [data-testid="stPopoverBody"] [data-baseweb="tab-highlight"] {
        background-color: #54d6e8 !important;
    }
    [data-testid="stPopoverBody"] [data-baseweb="select"] > div,
    [data-testid="stPopoverBody"] [data-baseweb="input"] > div {
        background: #111a22 !important;
        color: #e8f0f2 !important;
        border-color: #34424d !important;
    }
    [data-baseweb="popover"] ul,
    [data-baseweb="menu"] {
        background: #111a22 !important;
        color: #e8f0f2 !important;
    }
    [data-baseweb="popover"] li {
        color: #e8f0f2 !important;
    }
    [data-testid="stDialog"] {
        background: rgba(2, 6, 10, .68) !important;
    }
    [data-testid="stDialog"] [role="dialog"] {
        background: rgba(10, 16, 22, .98) !important;
        color: #e8f0f2 !important;
        border: 1px solid #34424d !important;
        border-top: 3px solid #54d6e8 !important;
        border-radius: 2px !important;
        box-shadow: 0 18px 48px rgba(0, 0, 0, .58) !important;
    }
    [data-testid="stDialog"] h2,
    [data-testid="stDialog"] p,
    [data-testid="stDialog"] label,
    [data-testid="stDialog"] span {
        color: #dce6e9 !important;
    }
    [data-testid="stDialog"] [data-testid="stMetricLabel"] p {
        color: #91a0ad !important;
        font: 600 10px monospace !important;
        letter-spacing: .08em;
    }
    [data-testid="stDialog"] [data-testid="stMetricValue"] {
        color: #e8f0f2 !important;
        font-family: monospace !important;
    }
    [data-testid="stDialog"] [data-testid="stAlert"] {
        background: #111a22 !important;
        border: 1px solid #34424d !important;
        border-radius: 2px !important;
    }
    [data-testid="stDialog"] [data-testid="stAlert"] > div,
    [data-testid="stDialog"] [data-baseweb="notification"] {
        background: #111a22 !important;
        border: 0 !important;
        color: #dce6e9 !important;
    }
    [data-testid="stDialog"] [data-testid="stAlertContentSuccess"] p {
        color: #48c78e !important;
    }
    [data-testid="stDialog"] [data-testid="stAlertContentInfo"] p {
        color: #54d6e8 !important;
    }
    [data-testid="stDialog"] button {
        background: #111a22 !important;
        color: #e8f0f2 !important;
        border-color: #34424d !important;
        border-radius: 2px !important;
    }
    iframe[title="streamlit_folium.st_folium"] {
        height: calc(100vh - 0.7rem) !important;
    }
    [data-testid="stElementContainer"]:has(iframe[title="streamlit_folium.st_folium"]) {
        height: calc(100vh - 0.7rem) !important;
        overflow: hidden !important;
    }
    .st-key-mission_btn {
        position: fixed !important;
        top: 12px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 10000;
        width: auto !important;
    }
    .st-key-mission_btn button {
        background: rgba(10, 16, 22, .94) !important;
        color: #f2b400 !important;
        border: 1px solid #f2b400 !important;
        border-radius: 2px !important;
        font-family: monospace !important;
        letter-spacing: .06em;
        box-shadow: 0 6px 18px rgba(0, 0, 0, .35);
    }
    .st-key-route_dock {
        position: fixed !important;
        left: 50%;
        bottom: 14px;
        transform: translateX(-50%);
        z-index: 10000;
        width: min(820px, calc(100vw - 32px));
        padding: 0.45rem 0.7rem 0.35rem;
        background: rgba(10, 16, 22, .94);
        border: 1px solid #34424d;
        border-top: 3px solid #54d6e8;
        box-shadow: 0 8px 24px rgba(0, 0, 0, .38);
    }
    .st-key-route_dock [data-testid="stMetric"] {
        padding: 0 !important;
    }
    .st-key-route_dock [data-testid="stMetricLabel"] p {
        color: #91a0ad !important;
        font: 600 10px monospace !important;
        letter-spacing: .08em;
    }
    .st-key-route_dock [data-testid="stMetricValue"] {
        color: #e8f0f2 !important;
        font: 600 19px monospace !important;
    }
    .st-key-route_dock .route-dock-status {
        padding-top: 0.15rem;
        font: 600 12px monospace;
        letter-spacing: .08em;
    }
    .st-key-route_dock button {
        margin-top: 0.25rem;
        min-height: 2.15rem;
        border-radius: 2px !important;
    }
    .st-key-route_dock .route-dock-why {
        color: #9eacb6;
        font: 400 12px monospace;
        padding: 0.1rem 0 0.25rem;
    }
    [data-testid="stDialog"] [role="tab"] {
        color: #91a0ad !important;
        font-family: monospace !important;
    }
    [data-testid="stDialog"] [role="tab"][aria-selected="true"] {
        color: #54d6e8 !important;
    }
    [data-testid="stDialog"] [data-baseweb="tab-highlight"] {
        background-color: #54d6e8 !important;
    }
    /* reserve gauges, timeline and comparison tables (merged from the rover version) */
    .sec { font: 600 10px monospace; letter-spacing: .08em; text-transform: uppercase;
           color: #91a0ad; margin: 12px 0 6px; }
    .note { font: 400 12px monospace; color: #9eacb6; margin: 2px 0 8px; }
    .bar { margin-bottom: 10px; color: #dce6e9; }
    .bar .bh { display: flex; justify-content: space-between; font: 13px monospace; }
    .bar .bt { position: relative; height: 10px; background: #1b2731; margin: 4px 0 2px; overflow: hidden; }
    .bar .bf { height: 100%; }
    .bf.good { background: #48c78e; } .bf.warn { background: #f0b44d; } .bf.bad { background: #e25b5b; }
    .bar .br { position: absolute; top: 0; bottom: 0; width: 2px; background: #e8f0f2; opacity: .85; }
    .bar .bd { font: 11px monospace; color: #91a0ad; }
    table.tl { width: 100%; border-collapse: collapse; font: 13px monospace; color: #dce6e9;
               font-variant-numeric: tabular-nums; }
    table.tl th { text-align: left; font-weight: 600; color: #91a0ad; font-size: 10px; text-transform: uppercase;
                  letter-spacing: .08em; padding: 4px 6px; border-bottom: 1px solid #34424d; }
    table.tl td { padding: 6px; border-bottom: 1px solid #1f2c36; }
    table.tl td.num, table.tl th.num { text-align: right; }
    .on { color: #48c78e; } .off { color: #e25b5b; }
    .unk { color: #91a0ad; }
    .st-key-monitor {
        position: fixed !important;
        right: 12px;
        top: 84px;
        z-index: 10000;
        width: min(360px, calc(100vw - 32px)) !important;
        max-height: calc(100vh - 190px);
        overflow-y: auto;
        background: rgba(10, 16, 22, .94);
        border: 1px solid #34424d;
        border-top: 3px solid #e879f9;
        box-shadow: 0 8px 24px rgba(0, 0, 0, .38);
    }
    .st-key-monitor details, .st-key-monitor summary { background: transparent !important; border: 0 !important; }
    .st-key-monitor summary p { color: #e8f0f2 !important; font: 600 11px monospace !important; letter-spacing: .08em; }
    .st-key-monitor p, .st-key-monitor li, .st-key-monitor span { color: #dce6e9; }
    .st-key-monitor button { border-radius: 2px !important; min-height: 2rem; background: #111a22 !important;
                             color: #e8f0f2 !important; border-color: #34424d !important; }
    .st-key-monitor [data-baseweb="select"] > div { background: #111a22 !important; color: #e8f0f2 !important;
                                                    border-color: #34424d !important; }
    .mon-clock { font: 600 11px monospace; color: #e879f9; letter-spacing: .06em; margin-bottom: 4px; }
    table.mon { width: 100%; border-collapse: collapse; font: 11px/1.35 monospace; color: #dce6e9; }
    table.mon th { text-align: left; color: #91a0ad; font-weight: 400; padding: 2px 6px 2px 0;
                   vertical-align: top; white-space: nowrap; }
    table.mon td { padding: 2px 0; border-bottom: 1px solid #1f2c36; }
    </style>
    """,
    unsafe_allow_html=True
)

# Paths relative to this file, so the app runs from any working directory.
# Prefer the wide 16:9 window (6.4 x 3.6 km, prep/make_wide_crop.py); fall back to the 3 x 3 km crop.
_WIDE = (HERE / "data" / "hirise_wide.tif", HERE / "data" / "dem_wide.tif")
ORTHO, DTM = _WIDE if all(p.exists() for p in _WIDE) else (HERE / "data" / "hirise_crop.tif",
                                                           HERE / "data" / "dem_crop.tif")
ASSUMPTIONS = HERE / "eva" / "assumptions.json"

# Mars lon/lat (planetocentric, east-positive, sphere R = 3,396,190 m)
MARS_LONLAT = CRS.from_proj4("+proj=longlat +R=3396190 +no_defs")
HABITAT_B = (137.370908, -4.811979)   # chosen habitat (lon, lat)

# Science stops (Curiosity waypoints) and Curiosity's drive through the boxwork
STOPS = [
    ("Ridge (sol 4711)", 137.3810041, -4.80800755),
    ("Hollow (sol 4709)", 137.3810495, -4.80799931),
    ("Nodules, Bococo (sol 4625)", 137.379759371, -4.805862088),
]
# whole drive inside the map (prep/curiosity_track.py); older file = boxwork sols only
TRACK = next(p for p in (HERE / "data" / "curiosity_track.csv", HERE / "data" / "boxwork_waypoints.csv")
             if p.exists() or p.name == "boxwork_waypoints.csv")
PHOTOS = HERE / "data" / "rover_photos.json"          # prep/rover_photos.py
THERMAL_TIF = HERE / "data" / "themis_ti.tif"          # prep/themis_ti.py
THERMAL_LAYER = "Thermal inertia (sand vs rock)"

# Routing grid: ~15 m cells whatever the window size (3 km -> 200 cells, 6.4 km -> ~430).
# Map image: longest side in pixels (~2.5 m per pixel on the wide window).
GRID_CELL_M = 15.0
IMAGE_SIZE = 2600

# RMS height of the 1 m DEM above its 3 m-smoothed version; above this
# the ground counts as boxwork (slower, costlier walking in the EVA model)
ROUGH_RIDGE_M = 0.25


def webp_data_url(rgba, quality=82):
    """RGBA uint8 array -> compact data URL for an image overlay."""
    import base64
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="WEBP", quality=quality, method=4)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def hhmm(h):
    m = int(round(h * 60)) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def southern_season(ls):
    if ls < 90:
        return "Southern autumn"
    if ls < 180:
        return "Southern winter"
    if ls < 270:
        return "Southern spring"
    return "Southern summer"


def assumptions_for_season(solar_longitude_deg, rover_max=None):
    A = eb.load_assumptions(ASSUMPTIONS)
    A["climate"]["solar_longitude_deg"] = float(solar_longitude_deg)
    if rover_max is not None:
        A["rover"]["max_slope_deg"] = float(rover_max)
    return A


# ==========================================================
# LOAD AND ALIGN DATASETS
# ==========================================================

@st.cache_data(show_spinner="Processing Mars terrain...")
def load_data(ortho_path, dtm_path, mtime):
    """Cached per file and modification time, so new data files are picked up without a restart."""

    with rasterio.open(ortho_path) as ortho, \
         rasterio.open(dtm_path) as dtm:

        if ortho.crs is None or dtm.crs is None:
            raise ValueError(
                "Missing georeferencing. Check the DTM file."
            )

        if not ortho.crs.is_projected:
            raise ValueError(
                "The orthoimage must use a projected CRS."
            )

        if not math.isclose(
            ortho.crs.linear_units_factor[1], 1.0
        ):
            raise ValueError(
                "The orthoimage coordinates must be in meters."
            )

        bounds = ortho.bounds

        def shape_for(max_size):
            scale = min(
                1.0,
                max_size / max(ortho.width, ortho.height)
            )
            return (
                max(2, round(ortho.height * scale)),
                max(2, round(ortho.width * scale))
            )

        ih, iw = shape_for(IMAGE_SIZE)
        rows = max(2, round((bounds.top - bounds.bottom) / GRID_CELL_M))
        cols = max(2, round((bounds.right - bounds.left) / GRID_CELL_M))

        # FIX: show HiRISE in colour (was band 1 only, grey)
        n_bands = min(ortho.count, 3)

        image = ortho.read(
            list(range(1, n_bands + 1)),
            out_shape=(n_bands, ih, iw),
            resampling=Resampling.bilinear
        ).astype(np.float32)

        if n_bands < 3:
            image = np.repeat(image[:1], 3, axis=0)

        image_mask = ortho.read_masks(
            1,
            out_shape=(ih, iw),
            resampling=Resampling.nearest
        ) > 0

        valid_image = image_mask & np.all(
            np.isfinite(image), axis=0
        )

        # one stretch for all bands keeps the colour balance
        low, high = np.percentile(
            image[:, valid_image], [2, 98]
        )

        rgb = np.clip(
            (image - low) / max(high - low, 1e-9),
            0, 1
        )

        rgba = np.dstack([
            (np.moveaxis(rgb, 0, -1) * 255).astype(np.uint8),
            np.where(valid_image, 255, 0).astype(np.uint8)
        ])

        # Resample DTM to routing grid
        transform = from_bounds(
            *bounds, cols, rows
        )

        elevation = np.full(
            (rows, cols), np.nan, dtype=np.float32
        )

        reproject(
            source=rasterio.band(dtm, 1),
            destination=elevation,
            src_transform=dtm.transform,
            src_crs=dtm.crs,
            src_nodata=dtm.nodata,
            dst_transform=transform,
            dst_crs=ortho.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear
        )

        # DTM validity mask
        coverage = np.zeros(
            (rows, cols), dtype=np.uint8
        )

        reproject(
            source=dtm.read_masks(1),
            destination=coverage,
            src_transform=dtm.transform,
            src_crs=dtm.crs,
            src_nodata=0,
            dst_transform=transform,
            dst_crs=ortho.crs,
            dst_nodata=0,
            resampling=Resampling.nearest
        )

        elevation[coverage == 0] = np.nan

        # NEW: ground roughness from the full 1 m DEM -> boxwork mask
        z = dtm.read(1, masked=True).astype("float64")
        z_filled = z.filled(float(np.nanmean(z.compressed())))
        resid = z_filled - gaussian_filter(z_filled, 3.0)
        rough_1m = np.sqrt(uniform_filter(resid ** 2, 7))
        rough_1m[np.ma.getmaskarray(z)] = np.nan

        rough = np.full((rows, cols), np.nan, dtype=np.float32)

        reproject(
            source=rough_1m.astype(np.float32),
            destination=rough,
            src_transform=dtm.transform,
            src_crs=dtm.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs=ortho.crs,
            dst_nodata=np.nan,
            resampling=Resampling.average
        )

        # NEW: habitat B in map coordinates (default base camp)
        hx, hy = warp_transform(
            MARS_LONLAT, ortho.crs,
            [HABITAT_B[0]], [HABITAT_B[1]]
        )
        habitat = (hx[0], hy[0])

        # map units per ground metre at the site (equirectangular lat_ts = 0:
        # east-west is stretched by 1/cos(lat)); used to draw circles in metres
        d_deg = 1e-4
        sx_, sy_ = warp_transform(
            MARS_LONLAT, ortho.crs,
            [HABITAT_B[0], HABITAT_B[0] + d_deg, HABITAT_B[0]],
            [HABITAT_B[1], HABITAT_B[1], HABITAT_B[1] + d_deg]
        )
        ground = 3396190.0 * math.radians(d_deg)
        kx = abs(sx_[1] - sx_[0]) / (ground * math.cos(math.radians(HABITAT_B[1])))
        ky = abs(sy_[2] - sy_[0]) / ground

        # science stops and Curiosity's drive in map coordinates
        sx, sy = warp_transform(
            MARS_LONLAT, ortho.crs,
            [s[1] for s in STOPS], [s[2] for s in STOPS]
        )
        stops = [(s[0], x, y) for s, x, y in zip(STOPS, sx, sy)]

        track = []
        if TRACK.exists():
            wp = pd.read_csv(TRACK)
            if "segment" not in wp:
                wp = wp.sort_values("sol").assign(segment=1)
            tx, ty = warp_transform(
                MARS_LONLAT, ortho.crs,
                wp["lon"].tolist(), wp["lat"].tolist()
            )
            track = [(int(g), int(s), x, y) for g, s, x, y in zip(wp["segment"], wp["sol"], tx, ty)]

        dx =(bounds.right - bounds.left) / cols
        dy = (bounds.top - bounds.bottom) / rows

        # Calculate slope from elevation
        valid = np.isfinite(elevation)

        # Require a complete 3x3 neighborhood
        padded = np.pad(
            valid, 1, constant_values=False
        )

        slope_valid = np.ones_like(valid)

        for dr in range(3):
            for dc in range(3):
                slope_valid &= padded[
                    dr:dr + rows,
                    dc:dc + cols
                ]

        filled = np.where(valid, elevation, 0)

        grad_y, grad_x = np.gradient(
            filled, dy, dx
        )

        slope = np.degrees(
            np.arctan(
                np.hypot(grad_x, grad_y)
            )
        )

        slope[~slope_valid] = np.nan

        return {
            "image": rgba,
            # compressed once (WebP keeps transparency): the map is re-sent on every
            # simulation tick, a raw PNG would be ~7 MB each time
            "image_url": webp_data_url(rgba),
            "elevation": elevation,
            "slope": slope,
            "is_ridge": np.nan_to_num(rough, nan=0) > ROUGH_RIDGE_M,
            "habitat": habitat,
            "stops": stops,
            "track": track,
            "kx": kx,
            "ky": ky,
            "crs": ortho.crs.to_wkt(),
            "bounds": bounds,
            "rows": rows,
            "cols": cols,
            "dx": dx,
            "dy": dy
        }


try:
    DATA_KEY = (str(ORTHO), str(DTM), max(p.stat().st_mtime for p in (ORTHO, DTM, TRACK) if p.exists()))
    data = load_data(*DATA_KEY)
except Exception as e:
    st.error(f"Could not load datasets: {e}")
    st.stop()


@st.cache_resource
def _active_data():
    return {"key": None}


# Route, viewshed and relay caches are keyed by grid cells, not by dataset: drop them when the data
# changes (e.g. after prep/make_wide_crop.py), otherwise old-grid results would be reused.
if _active_data()["key"] != DATA_KEY:
    if _active_data()["key"] is not None:
        st.cache_data.clear()
        data = load_data(*DATA_KEY)
    _active_data()["key"] = DATA_KEY


bounds = data["bounds"]
elevation = data["elevation"]
slope = data["slope"]
is_ridge = data["is_ridge"]
rows = data["rows"]
cols = data["cols"]
dx = data["dx"]
dy = data["dy"]
GRID_CRS_WKT = CRS.from_wkt(data["crs"])


@st.cache_data(show_spinner=False)
def load_photos(path, mtime):
    return json.loads(Path(path).read_text(encoding="utf-8")).get("photos", [])


def _inside(lon, lat):
    x, y = warp_transform(MARS_LONLAT, GRID_CRS_WKT, [lon], [lat])
    return bounds.left <= x[0] <= bounds.right and bounds.bottom <= y[0] <= bounds.top


# Curiosity photos inside the map, grouped by the rover position they were taken from
PHOTO_LIST = ([p for p in load_photos(str(PHOTOS), PHOTOS.stat().st_mtime) if _inside(p["rover_lon"], p["rover_lat"])]
              if PHOTOS.exists() else [])
@st.cache_data(show_spinner=False)
def load_thermal(path, mtime):
    with rasterio.open(path) as s:
        b = s.bounds
        return s.read(1).astype("float32"), [[b.bottom, b.left], [b.top, b.right]]


THERMAL = load_thermal(str(THERMAL_TIF), THERMAL_TIF.stat().st_mtime) if THERMAL_TIF.exists() else None

PHOTO_SITES = {}
for _p in PHOTO_LIST:
    PHOTO_SITES.setdefault((round(_p["rover_lat"], 6), round(_p["rover_lon"], 6)), []).append(_p)


# ==========================================================
# COORDINATE CONVERSION
# ==========================================================

def map_to_grid(x, y):

    col = int(
        (x - bounds.left) /
        (bounds.right - bounds.left) * cols
    )

    row = int(
        (bounds.top - y) /
        (bounds.top - bounds.bottom) * rows
    )

    return (
        max(0, min(rows - 1, row)),
        max(0, min(cols - 1, col))
    )


def grid_to_map(cell):

    row, col = cell

    x = bounds.left + (col + 0.5) * dx
    y = bounds.top - (row + 0.5) * dy

    return x, y


# ==========================================================
# A* SHORTEST SAFE ROUTE
# ==========================================================

def astar(start, goal, max_slope):

    safe = (
        np.isfinite(slope) &
        (slope <= max_slope)
    )

    if not safe[start] or not safe[goal]:
        return None

    directions = [
        (-1, 0), (1, 0),
        (0, -1), (0, 1),
        (-1, -1), (-1, 1),
        (1, -1), (1, 1)
    ]

    def heuristic(cell):
        return math.hypot(
            (cell[1] - goal[1]) * dx,
            (cell[0] - goal[0]) * dy
        )

    queue = [(heuristic(start), 0, start)]

    distances = {start: 0}
    parents = {}

    while queue:

        _, cost, current = heapq.heappop(queue)

        if cost > distances.get(
            current, float("inf")
        ):
            continue

        if current == goal:

            path = [current]

            while current in parents:
                current = parents[current]
                path.append(current)

            return path[::-1]

        r, c = current

        for dr, dc in directions:

            nr = r + dr
            nc = c + dc

            if not (
                0 <= nr < rows and
                0 <= nc < cols
            ):
                continue

            if not safe[nr, nc]:
                continue

            # No diagonal movement through
            # corners of unsafe terrain
            if dr != 0 and dc != 0:
                if not (
                    safe[r, nc] and safe[nr, c]
                ):
                    continue

            step = math.hypot(
                dc * dx, dr * dy
            )

            # Check slope of actual movement
            dz = (
                elevation[nr, nc] -
                elevation[r, c]
            )

            movement_slope = math.degrees(
                math.atan2(abs(dz), step)
            )

            if movement_slope > max_slope:
                continue

            neighbor = (nr, nc)
            new_cost = cost + step

            if new_cost < distances.get(
                neighbor, float("inf")
            ):

                distances[neighbor] = new_cost
                parents[neighbor] = current

                heapq.heappush(
                    queue,
                    (
                        new_cost + heuristic(neighbor),
                        new_cost,
                        neighbor
                    )
                )

    return None


# ==========================================================
# MODEL CONSTANTS (all values live in eva/assumptions.json)
# ==========================================================

A = eb.load_assumptions(ASSUMPTIONS)
RV = A["rover"]
COMMS = A["comms"]
RESERVE_PCT = 100 * A["consumables"]["reserve_fraction"]
MARS_R = 3396190.0

DIRECTIONS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1)
]

EL_FILLED = np.where(np.isfinite(elevation), elevation, np.nanmin(elevation)).astype(np.float64)
NAV = A["navigation"]
TERRAIN = sim.Terrain(elevation, dx, dy, data["kx"], data["ky"])


# ==========================================================
# TRAVEL TIME FROM A POINT (Dijkstra) - used for the rover plan
# ==========================================================

@st.cache_data(show_spinner=False)
def reach(start, travel, max_slope, rover_max=None, avoid_box=True):
    """Fastest travel time (s) from `start` to every cell, on foot or by rover.

    Walking: slope <= max_slope (astronaut limit). Rover: slope <= rover_max (rover limit,
    set separately) and, if avoid_box, no boxwork ridges. Returns (time, parent_row, parent_col).
    """
    rover_max = RV["max_slope_deg"] if rover_max is None else rover_max
    if travel == "drive":
        limit = rover_max
        ok = np.isfinite(slope) & (slope <= limit)
        if avoid_box:
            ok &= ~is_ridge
    else:
        limit = max_slope
        ok = np.isfinite(slope) & (slope <= limit)

    ok_l, el, ridge = ok.tolist(), elevation.tolist(), is_ridge.tolist()
    v_flat, k_slope = A["walking"]["v_flat_mps"], A["walking"]["v_slope_coeff"]
    sf_plain, sf_ridge = A["terrain"]["plain"]["speed_factor"], A["terrain"]["ridge"]["speed_factor"]
    v_max, loss, r_max = RV["v_max_mps"], RV["v_slope_loss"], rover_max
    steps = [(dr, dc, math.hypot(dc * dx, dr * dy)) for dr, dc in DIRECTIONS]

    inf = float("inf")
    time_s = [[inf] * cols for _ in range(rows)]
    par_r = np.full((rows, cols), -1, dtype=np.int16)
    par_c = np.full((rows, cols), -1, dtype=np.int16)
    time_s[start[0]][start[1]] = 0.0
    queue = [(0.0, start[0], start[1])]

    while queue:
        cost, r, c = heapq.heappop(queue)
        if cost > time_s[r][c]:
            continue
        for dr, dc, step in steps:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or not ok_l[nr][nc]:
                continue
            if dr and dc and not (ok_l[r][nc] and ok_l[nr][c]):
                continue                      # no cutting corners of unsafe terrain
            dz = el[nr][nc] - el[r][c]
            move_slope = math.degrees(math.atan2(abs(dz), step))
            if not move_slope <= limit:
                continue
            if travel == "drive":
                v = v_max * (1.0 - loss * min(move_slope / r_max, 1.0))
            else:
                v = min(v_flat, v_flat - k_slope * move_slope) * (sf_ridge if ridge[nr][nc] else sf_plain)
            new_cost = cost + math.hypot(step, dz) / v
            if new_cost < time_s[nr][nc]:
                time_s[nr][nc] = new_cost
                par_r[nr, nc], par_c[nr, nc] = r, c
                heapq.heappush(queue, (new_cost, nr, nc))

    return np.array(time_s), par_r, par_c


def backtrack(par_r, par_c, cell):
    """Cells from `cell` back to the point the search started from."""
    path = [cell]
    while par_r[cell] >= 0:
        cell = (int(par_r[cell]), int(par_c[cell]))
        path.append(cell)
    return path


# ==========================================================
# RADIO LINE OF SIGHT (live link to the base)
# ==========================================================

def line_of_sight(src, h_src, targets, h_tgt):
    """True where a straight radio path from `src` (antenna h_src above ground) reaches
    each target cell (antenna h_tgt) without hitting terrain. Includes Mars curvature."""
    targets = np.asarray(targets).reshape(-1, 2)
    r0, c0 = src
    tr, tc = targets[:, 0].astype(float), targets[:, 1].astype(float)
    dist = np.hypot((tc - c0) * dx, (tr - r0) * dy)
    z0 = EL_FILLED[r0, c0] + h_src
    zt = EL_FILLED[targets[:, 0], targets[:, 1]] + h_tgt
    n = max(2, int(dist.max() / min(dx, dy)))
    clear = np.ones(len(targets), dtype=bool)
    for t in np.linspace(0, 1, n + 1)[1:-1]:
        ground = map_coordinates(
            EL_FILLED, [r0 + t * (tr - r0), c0 + t * (tc - c0)], order=1, mode="nearest"
        )
        beam = z0 + t * (zt - z0) - t * (1 - t) * dist ** 2 / (2 * MARS_R)
        clear &= ground <= beam
    return clear


@st.cache_data(show_spinner="Computing radio coverage...")
def viewshed(src, h_src, h_target):
    """Cells with a radio line to an antenna h_src above `src`, for antennas h_target above ground."""
    rr, cc = np.indices((rows, cols))
    cells = np.column_stack([rr.ravel(), cc.ravel()])
    seen = line_of_sight(src, h_src, cells, h_target)
    return seen.reshape(rows, cols) & np.isfinite(elevation)


def coverage(base_cell, h_target, h_base):
    """Cells with a direct radio line to the base antenna, for a suit or rover antenna height."""
    return viewshed(base_cell, h_base, h_target)


def ground_dist_grid(cell):
    """Ground distance (m) from `cell` to every grid cell."""
    rr, cc = np.indices((rows, cols))
    return np.hypot((cc - cell[1]) * dx / data["kx"], (rr - cell[0]) * dy / data["ky"])


# ==========================================================
# EVA PLAN: route + legs for one operating mode
#   "ride"  mode A: astronaut aboard the rover all the way to the destination
#   "drive" mode B: ride to a parking spot, rover stays parked, astronaut walks
#   "walk"  reference: on foot from the base, no rover (the team's A* route)
# ==========================================================

def path_legs(path, kind, name):
    return [{
        "kind": kind,
        "name": name,
        "length_m": math.hypot((b[1] - a[1]) * dx, (b[0] - a[0]) * dy),
        "dz_m": float(elevation[b] - elevation[a]),
        "terrain": "ridge" if is_ridge[b] else "plain",
        "duration_min": 0.0,
        "a": a,
        "b": b,
    } for a, b in zip(path, path[1:])]


def reverse_legs(legs, name):
    return [dict(l, name=name, dz_m=-l["dz_m"], a=l["b"], b=l["a"]) for l in reversed(legs)]


@st.cache_data(show_spinner="Planning the route...")
def make_plan(travel, start, goal, max_slope, stop_min, rover_max, avoid_box, comms, park_cell=None,
              relay_cell=None):
    """Route and leg table for a round trip base -> destination -> base."""
    ok_cell = lambda cell: bool(np.isfinite(slope[cell]) and slope[cell] <= max_slope)

    if not np.isfinite(slope[goal]):
        return {"error": "Destination is outside the elevation data."}
    if not ok_cell(goal):
        return {"error": f"Destination itself is too steep ({slope[goal]:.0f}° > {max_slope}°)."}
    if not ok_cell(start) and travel == "walk":
        return {"error": f"Base camp is on terrain steeper than {max_slope}°. Move the base camp."}

    def stop(together=False):
        return [{"kind": "stop", "name": "Work at destination", "length_m": 0.0, "dz_m": 0.0,
                 "terrain": "plain", "duration_min": float(stop_min), "a": goal, "b": goal,
                 "together": together}]

    if travel == "walk":
        path = astar(start, goal, max_slope)
        if path is None:
            return {"error": f"No walking route below {max_slope}°: the destination is cut off by steep terrain."}
        out = path_legs(path, "walk", "Walk out")
        legs = out + stop() + reverse_legs(out, "Walk back")
        cells = path[1:] + [goal] + path[-2::-1]
        return {"travel": travel, "walk_path": path, "drive_path": [], "park": None,
                "legs": legs, "cells": cells}

    t_drive, dpr, dpc = reach(start, "drive", max_slope, rover_max, avoid_box)
    if not np.isfinite(t_drive[start]):
        return {"error": f"The rover cannot start here: base camp is steeper than the rover limit ({rover_max}°)"
                         + (" or on boxwork." if avoid_box else ".")}

    if travel == "ride":
        if not np.isfinite(t_drive[goal]):
            # name the real reason: would it work if the rover could cross boxwork?
            if avoid_box and is_ridge[goal]:
                why = "the destination is on boxwork ridges, which the rover avoids"
            elif avoid_box and np.isfinite(reach(start, "drive", max_slope, rover_max, False)[0][goal]):
                why = "boxwork ridges block every drivable way there"
            else:
                why = f"the ground on the way is steeper than the rover limit ({rover_max}°)"
            return {"error": f"Rover can't drive there: {why}.", "rover_unreachable": True}
        drive_path = backtrack(dpr, dpc, goal)[::-1]
        d_out = path_legs(drive_path, "drive", "Drive out")
        legs = d_out + stop(together=True) + reverse_legs(d_out, "Drive back")
        cells = drive_path[1:] + [goal] + drive_path[-2::-1]
        return {"travel": travel, "walk_path": [], "drive_path": drive_path, "park": goal,
                "legs": legs, "cells": cells, "relay": None, "park_note": ""}

    # mode B: drive to a parking spot, walk the rest
    t_walk, wpr, wpc = reach(goal, "walk", max_slope)
    total = t_drive + t_walk
    if not np.isfinite(total).any():
        return {"error": "The rover cannot get close enough: no drivable ground connects to a walking route."}

    # relay spot: the rover mast sees the astronaut at the work site, and reaches the base
    # directly or through the pre-deployed relay rover, within radio range
    to_base = (coverage(start, comms["rover_antenna_m"], comms["base_antenna_m"])
               & (ground_dist_grid(start) <= comms["rover_base_range_m"]))
    if relay_cell is not None and relay_sees_base(relay_cell, start, comms):
        to_base |= (viewshed(relay_cell, comms["rover_antenna_m"], comms["rover_antenna_m"])
                    & (ground_dist_grid(relay_cell) <= comms["rover_base_range_m"]))
    relay_ok = (to_base
                & viewshed(goal, comms["suit_antenna_m"], comms["rover_antenna_m"])
                & (ground_dist_grid(goal) <= comms["suit_rover_range_m"]))
    park_note = ""
    if park_cell is not None:
        if np.isfinite(total[park_cell]):
            park = tuple(park_cell)
        else:
            d = np.where(np.isfinite(total), ground_dist_grid(park_cell), np.inf)
            park = tuple(int(i) for i in np.unravel_index(np.argmin(d), d.shape))
            why = ("the rover can't drive there" if not np.isfinite(t_drive[park_cell])
                   else "no walking route from there to the destination")
            park_note = f"Parking moved {d[park]:.0f} m: {why}."
        relay = bool(relay_ok[park])
    else:
        # safety first: fastest spot that can relay, even if a spot without relay would be quicker
        relay_total = np.where(relay_ok & np.isfinite(total), total, np.inf)
        if np.isfinite(relay_total).any():
            pick, relay = relay_total, True
        else:
            pick, relay = np.where(np.isfinite(total), total, np.inf), False
        park = tuple(int(i) for i in np.unravel_index(np.argmin(pick), pick.shape))
    drive_path = backtrack(dpr, dpc, park)[::-1]          # base -> parking spot
    walk_path = backtrack(wpr, wpc, park)                 # parking spot -> destination
    d_out = path_legs(drive_path, "drive", "Drive out")
    w_in = path_legs(walk_path, "walk", "Walk in")
    legs = d_out + w_in + stop() + reverse_legs(w_in, "Walk back") + reverse_legs(d_out, "Drive back")
    cells = (drive_path[1:] + walk_path[1:] + [goal] + walk_path[-2::-1] + drive_path[-2::-1])
    return {"travel": travel, "walk_path": walk_path, "drive_path": drive_path, "park": park,
            "legs": legs, "cells": cells, "relay": relay, "park_note": park_note,
            "relay_cost_min": float((total[park] - np.nanmin(np.where(np.isfinite(total), total, np.nan))) / 60)}


def relay_sees_base(relay_cell, base_cell, comms):
    st_ = TERRAIN.links(np.array([relay_cell], float), comms["rover_antenna_m"], np.array([base_cell], float),
                        comms["base_antenna_m"], comms["rover_base_range_m"], colocated_m=NAV["dock_m"])
    return bool(st_[0] == sim.OK)


@st.cache_data(show_spinner="Placing the relay rover...")
def relay_spot(base_cell, targets, rover_max, avoid_box, comms, wanted=None):
    """Where the uncrewed relay rover parks (driven there hours before the EVA, parked all mission).

    Auto: a drivable spot (reachable from the base within the rover limits) whose mast sees the base and
    as many science stops as possible; among those the one closest to the stops. `wanted` = a clicked
    spot, moved to the nearest drivable cell if needed.
    """
    t_drive, pr, pc = reach(base_cell, "drive", 90, rover_max, avoid_box)
    drivable = np.isfinite(t_drive)
    if not drivable.any():
        return {"error": "The relay rover cannot leave the base (rover limits)."}
    note = ""
    if wanted is not None:
        if drivable[wanted]:
            cell = tuple(wanted)
        else:
            d = np.where(drivable, ground_dist_grid(wanted), np.inf)
            cell = tuple(int(i) for i in np.unravel_index(np.argmin(d), d.shape))
            note = (f"Relay rover moved {d[cell]:.0f} m: it can't drive to the chosen spot "
                    f"(slope > {rover_max}°{' or boxwork' if avoid_box else ''}).")
    else:
        sees_base = (coverage(base_cell, comms["rover_antenna_m"], comms["base_antenna_m"])
                     & (ground_dist_grid(base_cell) <= comms["rover_base_range_m"]))
        seen = np.zeros(drivable.shape, dtype=int)
        for t in targets:
            seen += (viewshed(t, comms["suit_antenna_m"], comms["rover_antenna_m"])
                     & (ground_dist_grid(t) <= comms["suit_rover_range_m"]))
        rr, cc = np.indices(drivable.shape)
        mr, mc = np.mean([t[0] for t in targets]), np.mean([t[1] for t in targets])
        dist = np.hypot((cc - mc) * dx / data["kx"], (rr - mr) * dy / data["ky"])
        score = seen * 1e6 - dist
        ok = drivable & sees_base
        score = np.where(ok if ok.any() else drivable, score, -np.inf)
        cell = tuple(int(i) for i in np.unravel_index(np.argmax(score), score.shape))
    path = backtrack(pr, pc, cell)[::-1]
    drive_m = sum(math.hypot((b[1] - a[1]) * dx, (b[0] - a[0]) * dy) for a, b in zip(path, path[1:]))
    seen_stops = sum(bool(TERRAIN.links(np.array([t], float), comms["suit_antenna_m"], np.array([cell], float),
                                        comms["rover_antenna_m"], comms["suit_rover_range_m"])[0] == sim.OK)
                     for t in targets)
    return {"cell": cell, "path": path, "note": note, "drive_m": drive_m,
            "drive_min": float(t_drive[cell] * RV["margin"] / 60),
            "sees_base": relay_sees_base(cell, base_cell, comms), "stops_seen": seen_stops}


def link_per_leg(plan, base_cell, comms, relay_cell=None):
    """True per leg where the base has a live link to the astronaut at the end of that leg.

    Same rule as the simulation (sim.network_links): the suit talks to a rover (crew rover or the
    pre-deployed relay rover), the rover to the base. Reference walk without the crew rover: direct
    suit <-> base, or via the relay rover.
    """
    cells = np.array(plan["cells"], dtype=float).reshape(-1, 2)
    if plan["travel"] == "walk":
        base = np.repeat(np.array([base_cell], float), len(cells), 0)
        up = TERRAIN.links(cells, comms["suit_antenna_m"], base, comms["base_antenna_m"],
                           comms["suit_rover_range_m"]) == sim.OK
        if relay_cell is not None:
            L = sim.network_links(TERRAIN, cells, np.repeat(np.array([relay_cell], float), len(cells), 0),
                                  base_cell, comms, NAV, relay_cell)
            up |= L["full"]
        return up
    kinds = [l["kind"] for l in plan["legs"]]
    rover = np.array([cells[i] if k == "drive" else plan["park"] for i, k in enumerate(kinds)], float)
    return sim.network_links(TERRAIN, cells, rover, base_cell, comms, NAV, relay_cell)["full"]


@st.cache_data(show_spinner="Computing EVA budget...")
def eva_cases(legs, start_h, solar_longitude_deg, rover_max):
    A_s = assumptions_for_season(solar_longitude_deg, rover_max)
    route = pd.DataFrame(legs)
    climate = eb.Climate(A_s)
    out, nominal_legs = {}, None
    for case in eb.CASES:
        legs_df, summary = eb.run_eva(route, A_s, climate, start_h, case)
        out[case] = summary
        if case == "nominal":
            nominal_legs = legs_df
    return out, nominal_legs


@st.cache_data(show_spinner="Checking start times...")
def eva_sweep(legs, solar_longitude_deg, rover_max):
    A_s = assumptions_for_season(solar_longitude_deg, rover_max)
    return eb.sweep_start_times(pd.DataFrame(legs), A_s, eb.Climate(A_s), np.arange(6.0, 16.01, 0.5))


def evaluate(plan, base_cell, start_h, solar_longitude_deg, rover_max, comms, relay_cell=None):
    """EVA budget + live-link share for a plan."""
    res, legs_df = eva_cases(plan["legs"], start_h, solar_longitude_deg, rover_max)
    link = link_per_leg(plan, base_cell, comms, relay_cell)
    minutes = legs_df["duration_min"].to_numpy()
    legs_df = legs_df.assign(link=link)
    return {"res": res, "legs": legs_df, "verdict": eb.verdict(res),
            "link_pct": 100 * float((minutes * link).sum() / max(minutes.sum(), 1e-9)), "link": link}


CHECK_TEXT = {
    "no_steep_legs": "a leg is steeper than the walking or rover slope limit",
    "duration": "more than 4 h outside",
    "peak_metabolic": "peak effort above 2000 BTU/h",
    "avg_metabolic": "average effort above 1600 BTU/h",
    "o2_reserve": "oxygen below the 25% reserve",
    "water_reserve": "cooling water below the 25% reserve",
    "daylight": "outside daylight (sun below 5° at start or return)",
    "suit_battery": "suit battery below the 25% reserve",
    "rover_battery": "rover battery below the 25% reserve",
}

# ---- map icons: realistic 3/4-view drawings with light, shading and ground shadow (SVG, not real 3D)

def _svg(view_w, view_h, width, body):
    h = round(width * view_h / view_w)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {view_w} {view_h}" width="{width}" '
            f'height="{h}" style="vertical-align:middle;overflow:visible">{body}</svg>')


_METAL = (
    '<linearGradient id="{p}top" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffffff"/>'
    '<stop offset="1" stop-color="#cfd5db"/></linearGradient>'
    '<linearGradient id="{p}side" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#dfe4e8"/>'
    '<stop offset=".55" stop-color="#aab3bb"/><stop offset="1" stop-color="#6f7983"/></linearGradient>'
    '<linearGradient id="{p}dark" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#9aa3ab"/>'
    '<stop offset="1" stop-color="#4e5760"/></linearGradient>'
    '<linearGradient id="{p}glass" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#3b5873"/>'
    '<stop offset=".5" stop-color="#132232"/><stop offset="1" stop-color="#05090d"/></linearGradient>'
    '<linearGradient id="{p}solar" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#3a6aa8"/>'
    '<stop offset=".6" stop-color="#173560"/><stop offset="1" stop-color="#0b1a30"/></linearGradient>'
    '<radialGradient id="{p}tire" cx=".4" cy=".35" r=".7"><stop offset="0" stop-color="#4a5057"/>'
    '<stop offset="1" stop-color="#121417"/></radialGradient>'
    '<radialGradient id="{p}shadow" cx=".5" cy=".5" r=".5"><stop offset="0" stop-color="#000" stop-opacity=".6"/>'
    '<stop offset="1" stop-color="#000" stop-opacity="0"/></radialGradient>'
)


def _wheel(p, cx, cy, rx, ry, near=True):
    hub = (f'<ellipse cx="{cx}" cy="{cy}" rx="{rx * .42:.1f}" ry="{ry * .42:.1f}" fill="url(#{p}dark)"/>'
           f'<ellipse cx="{cx - rx * .12:.1f}" cy="{cy - ry * .14:.1f}" rx="{rx * .16:.1f}" ry="{ry * .16:.1f}" '
           'fill="#e6eaee" opacity=".8"/>') if near else ""
    return (f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="url(#{p}tire)" stroke="#07090b" '
            f'stroke-width=".6"/>{hub}')


def rover_svg(width=40, glow=False, glow_color="#54d6e8"):
    """Crew rover (pressurised, six wheels) in 3/4 view: cabin with wrap-around windscreen, side window,
    suitport hatch, roof solar array, mast camera, navigation light and ground shadow."""
    p = "rv"
    halo = (f'<ellipse cx="50" cy="61" rx="46" ry="8" fill="{glow_color}" opacity=".18"/>' if glow else "")
    body = (
        f'<defs>{_METAL.format(p=p)}</defs>'
        + halo +
        f'<ellipse cx="50" cy="61" rx="44" ry="7" fill="url(#{p}shadow)"/>'
        # far-side wheels
        + _wheel(p, 33, 47, 6, 7, False) + _wheel(p, 56, 47, 6, 7, False) + _wheel(p, 79, 47, 6, 7, False) +
        # lower chassis
        f'<path d="M12 49 L88 49 L91 41 L11 41 Z" fill="url(#{p}dark)"/>'
        # cabin: side, roof, nose
        f'<path d="M11 42 L11 27 Q11 20 18 20 L70 20 Q83 20 90 31 L91 42 Z" fill="url(#{p}side)" '
        'stroke="#3d454d" stroke-width=".5"/>'
        f'<path d="M18 20 L26 12 L74 12 Q86 12 93 22 L90 31 Q83 20 70 20 Z" fill="url(#{p}top)" '
        'stroke="#8d969f" stroke-width=".5"/>'
        f'<path d="M90 31 L93 22 Q97 29 96 37 L91 42 Z" fill="url(#{p}dark)"/>'
        # windscreen + side window + reflections
        f'<path d="M71 21.5 Q83 21.5 89 31 L89.6 36 L71 36 Z" fill="url(#{p}glass)"/>'
        '<path d="M75 22.5 L80 22.5 L74 35 L71.5 35 Z" fill="#fff" opacity=".18"/>'
        f'<rect x="54" y="24" width="13" height="8" rx="2" fill="url(#{p}glass)"/>'
        '<path d="M56 24.5 L59 24.5 L57 31.5 L55 31.5 Z" fill="#fff" opacity=".15"/>'
        # suitport hatch + panel seams + accent stripe
        '<rect x="16" y="24" width="11" height="12" rx="2.5" fill="none" stroke="#5c656e" stroke-width=".8"/>'
        '<circle cx="21.5" cy="30" r="2.2" fill="#7b858f"/>'
        '<line x1="38" y1="21" x2="38" y2="41" stroke="#7b858f" stroke-width=".4"/>'
        '<path d="M11 38.5 L91 38.5" stroke="#54d6e8" stroke-width=".9" opacity=".85"/>'
        # roof solar array
        f'<path d="M27 18.6 L33 13.6 L60 13.6 L54 18.6 Z" fill="url(#{p}solar)" stroke="#8fb0d6" stroke-width=".4"/>'
        '<path d="M36 18.6 L42 13.6 M45 18.6 L51 13.6 M30 16.1 L57 16.1" stroke="#5f86b8" stroke-width=".35"/>'
        # mast + camera head + nav light
        f'<rect x="66" y="4" width="1.6" height="10" fill="url(#{p}dark)"/>'
        f'<rect x="62.5" y="1" width="8.5" height="4" rx="1" fill="url(#{p}side)" stroke="#3d454d" stroke-width=".4"/>'
        '<circle cx="65" cy="3" r=".9" fill="#0b1218"/><circle cx="68.5" cy="3" r=".9" fill="#0b1218"/>'
        '<circle cx="93.5" cy="25" r="1.2" fill="#9ff0ff"/>'
        # near-side wheels
        + _wheel(p, 24, 51, 8, 9) + _wheel(p, 50, 51, 8, 9) + _wheel(p, 76, 51, 8, 9)
    )
    return _svg(100, 70, width, body)


def relay_svg(width=40):
    """Lead antenna / relay rover in 3/4 view: compact four-wheel chassis, telescopic mast, parabolic
    high-gain dish with feed struts, beacon light and ground shadow."""
    p = "rl"
    body = (
        f'<defs>{_METAL.format(p=p)}'
        f'<radialGradient id="{p}dish" cx=".35" cy=".35" r=".75"><stop offset="0" stop-color="#ffffff"/>'
        '<stop offset=".7" stop-color="#cfd6dc"/><stop offset="1" stop-color="#8b959e"/></radialGradient>'
        '<radialGradient id="rlbeacon" cx=".5" cy=".5" r=".5"><stop offset="0" stop-color="#fff2c4"/>'
        '<stop offset=".4" stop-color="#f0b44d"/><stop offset="1" stop-color="#f0b44d" stop-opacity="0"/>'
        '</radialGradient></defs>'
        f'<ellipse cx="40" cy="84" rx="30" ry="5.5" fill="url(#{p}shadow)"/>'
        + _wheel(p, 30, 73, 4.5, 5.5, False) + _wheel(p, 56, 73, 4.5, 5.5, False) +
        # chassis
        f'<path d="M14 76 L64 76 L66 69 L13 69 Z" fill="url(#{p}dark)"/>'
        f'<path d="M13 70 L13 60 Q13 56 17 56 L60 56 Q66 56 67 62 L67 70 Z" fill="url(#{p}side)" '
        'stroke="#3d454d" stroke-width=".5"/>'
        f'<path d="M17 56 L23 50 L64 50 Q70 50 72 55 L67 62 Q66 56 60 56 Z" fill="url(#{p}top)" '
        'stroke="#8d969f" stroke-width=".5"/>'
        f'<path d="M24 54.8 L28.5 51 L44 51 L39.5 54.8 Z" fill="url(#{p}solar)"/>'
        '<path d="M13 66.5 L67 66.5" stroke="#f0b44d" stroke-width="1"/>'
        # telescopic mast (two sections)
        f'<rect x="51" y="34" width="4" height="19" fill="url(#{p}dark)"/>'
        f'<rect x="51.8" y="20" width="2.4" height="15" fill="url(#{p}side)"/>'
        # dish (back shading + face), feed struts and horn
        '<ellipse cx="50" cy="19" rx="19" ry="10" transform="rotate(-24 50 19)" fill="#6f7983"/>'
        f'<ellipse cx="49" cy="18" rx="18" ry="9" transform="rotate(-24 49 18)" fill="url(#{p}dish)" '
        'stroke="#5c656e" stroke-width=".6"/>'
        '<ellipse cx="49" cy="18" rx="12" ry="6" transform="rotate(-24 49 18)" fill="none" stroke="#a7b0b8" '
        'stroke-width=".5"/>'
        '<path d="M36 23 L44 8 M62 11 L44 8 M47 26 L44 8" stroke="#5c656e" stroke-width=".7"/>'
        f'<rect x="42.3" y="5.5" width="3.4" height="4" rx=".8" fill="url(#{p}dark)"/>'
        # beacon
        '<circle cx="62" cy="47" r="5" fill="url(#rlbeacon)"/><circle cx="62" cy="47" r="1.3" fill="#fff2c4"/>'
        + _wheel(p, 23, 75, 6, 7) + _wheel(p, 58, 75, 6, 7)
    )
    return _svg(80, 90, width, body)


def habitat_svg(width=56):
    """Base camp in 3/4 view: central dome on a sintered landing pad, two cylindrical habitat modules,
    airlock, two tilted solar arrays, comms dish and lights, with ground shadow."""
    p = "hb"
    body = (
        f'<defs>{_METAL.format(p=p)}'
        f'<radialGradient id="{p}dome" cx=".35" cy=".25" r=".8"><stop offset="0" stop-color="#ffffff"/>'
        '<stop offset=".55" stop-color="#d3d9de"/><stop offset="1" stop-color="#7d8790"/></radialGradient>'
        f'<linearGradient id="{p}cyl" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#f4f6f8"/>'
        '<stop offset=".45" stop-color="#d6dce1"/><stop offset="1" stop-color="#717b85"/></linearGradient>'
        f'<radialGradient id="{p}pad" cx=".5" cy=".45" r=".55"><stop offset="0" stop-color="#8a6a52"/>'
        '<stop offset=".85" stop-color="#5e4433"/><stop offset="1" stop-color="#5e4433" stop-opacity="0"/>'
        '</radialGradient></defs>'
        f'<ellipse cx="60" cy="62" rx="58" ry="15" fill="url(#{p}pad)"/>'
        '<ellipse cx="60" cy="61" rx="50" ry="11.5" fill="none" stroke="#c9a888" stroke-opacity=".35" '
        'stroke-width=".7" stroke-dasharray="3 2"/>'
        f'<ellipse cx="60" cy="63" rx="46" ry="8" fill="url(#{p}shadow)"/>'
        # solar arrays (behind)
        + "".join(
            f'<path d="M{x} 39 L{x + 3} 41" stroke="#4e5760" stroke-width="1"/>'
            f'<path d="M{x - 8} 22 L{x + 12} 22 L{x + 16} 36 L{x - 4} 36 Z" fill="url(#{p}solar)" '
            'stroke="#8fb0d6" stroke-width=".5"/>'
            f'<path d="M{x - 1} 22 L{x + 3} 36 M{x + 6} 22 L{x + 10} 36 M{x - 6} 29 L{x + 14} 29" '
            'stroke="#5f86b8" stroke-width=".4"/>'
            for x in (16, 92)) +
        # cylindrical modules (left / right) with end caps
        f'<rect x="12" y="42" width="34" height="14" rx="3" fill="url(#{p}cyl)" stroke="#5c656e" stroke-width=".5"/>'
        f'<ellipse cx="12.5" cy="49" rx="3.2" ry="7" fill="url(#{p}dark)"/>'
        f'<rect x="74" y="42" width="34" height="14" rx="3" fill="url(#{p}cyl)" stroke="#5c656e" stroke-width=".5"/>'
        f'<ellipse cx="107.5" cy="49" rx="3.2" ry="7" fill="url(#{p}side)" stroke="#5c656e" stroke-width=".4"/>'
        '<path d="M20 42.5 L20 55.5 M29 42.5 L29 55.5 M38 42.5 L38 55.5 M82 42.5 L82 55.5 M91 42.5 L91 55.5 '
        'M100 42.5 L100 55.5" stroke="#9aa3ab" stroke-width=".4"/>'
        f'<rect x="24" y="46" width="5" height="3" rx="1" fill="url(#{p}glass)"/>'
        f'<rect x="91" y="46" width="5" height="3" rx="1" fill="url(#{p}glass)"/>'
        # central dome
        f'<path d="M40 56 Q40 26 60 26 Q80 26 80 56 Z" fill="url(#{p}dome)" stroke="#5c656e" stroke-width=".6"/>'
        '<path d="M46 56 Q46 31 60 29 M74 56 Q74 31 60 29 M41 46 Q60 41 79 46" fill="none" stroke="#a7b0b8" '
        'stroke-width=".5"/>'
        f'<ellipse cx="60" cy="56" rx="20.5" ry="3.2" fill="url(#{p}dark)"/>'
        f'<path d="M52 36 Q55 33 58 33 L58 38 L52 39 Z" fill="url(#{p}glass)"/>'
        # airlock (front)
        f'<rect x="54" y="53" width="12" height="9" rx="2" fill="url(#{p}cyl)" stroke="#5c656e" stroke-width=".5"/>'
        f'<rect x="57.5" y="55" width="5" height="7" rx="1" fill="url(#{p}glass)"/>'
        # comms dish on the right module + lights
        f'<line x1="98" y1="42" x2="98" y2="35" stroke="#4e5760" stroke-width="1"/>'
        f'<ellipse cx="98" cy="33" rx="6" ry="3.2" transform="rotate(-20 98 33)" fill="url(#{p}top)" '
        'stroke="#5c656e" stroke-width=".5"/>'
        '<circle cx="60" cy="26.5" r="1.1" fill="#9ff0ff"/><circle cx="12" cy="44" r=".9" fill="#9ff0ff"/>'
    )
    return _svg(120, 80, width, body)


def camera_pin(n):
    """Curiosity camera position: dark disc with a camera body/lens and the number of photos."""
    return (
        '<div style="position:relative;width:28px;height:28px">'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 28" width="28" height="28" '
        'style="filter:drop-shadow(0 1px 2px rgba(0,0,0,.9))">'
        '<defs><radialGradient id="cplens" cx=".35" cy=".35" r=".7"><stop offset="0" stop-color="#bfe9ff"/>'
        '<stop offset=".5" stop-color="#1e3a5f"/><stop offset="1" stop-color="#05090d"/></radialGradient></defs>'
        '<circle cx="14" cy="14" r="13" fill="rgba(10,16,22,.88)" stroke="#7dd3fc" stroke-width="1.4"/>'
        '<rect x="6.5" y="10" width="15" height="10" rx="2" fill="#d9dee3" stroke="#0b1218" stroke-width=".6"/>'
        '<rect x="9.5" y="8" width="5" height="3" rx=".8" fill="#9aa3ab"/>'
        '<circle cx="14" cy="15" r="3.6" fill="url(#cplens)" stroke="#0b1218" stroke-width=".6"/>'
        '</svg>'
        f'<div style="position:absolute;right:-6px;top:-6px;font:700 9px/14px monospace;min-width:14px;'
        f'height:14px;text-align:center;border-radius:7px;background:#7dd3fc;color:#0a1016">{n}</div></div>'
    )


def plate(svg, text, color):
    """Icon with a small label underneath (for map markers)."""
    return (f'<div style="display:flex;flex-direction:column;align-items:center;gap:1px">{svg}'
            f'<div style="font:600 8px/1.3 monospace;letter-spacing:.14em;color:{color};'
            f'background:rgba(10,16,22,.85);border:1px solid {color}66;padding:0 4px;white-space:nowrap">'
            f'{text}</div></div>')


TONE = {"GO": "good", "MARGINAL": "warn", "NO-GO": "bad"}
ICON = {"GO": "✓", "MARGINAL": "!", "NO-GO": "✕"}



# ==========================================================
# MISSION CONTROLS — compact overlay menu
# ==========================================================

current_solar_longitude_deg = round(eb.mars_solar_longitude_deg(), 2)

MODE_LABEL = {
    "ride": "A · Travel with rover",
    "drive": "B · Explore on foot (rover parked)",
    "walk": "Reference · on foot from base, no rover",
}

with st.popover("☰  MARS NAVIGATOR / MISSION CONTROLS"):
    plan_tab, terrain_tab, comms_tab, environment_tab, data_tab = st.tabs(
        ["EVA", "Terrain", "Comms", "Environment", "Data"]
    )

    with plan_tab:
        # a button or map click may ask to switch to mode B. Pre-setting the widget's session state left the
        # radio showing the old choice, so a switch creates a fresh widget (new key) with B as its default.
        if st.session_state.pop("_switch_to_b", False):
            st.session_state["_mode_gen"] = st.session_state.get("_mode_gen", 0) + 1
            st.session_state["_mode_default"] = 1
        travel = st.radio(
            "Operating mode",
            ["ride", "drive"],
            index=st.session_state.get("_mode_default", 1),
            key=f"travel_mode_{st.session_state.get('_mode_gen', 0)}",
            format_func=MODE_LABEL.get,
            help="A: the astronaut rides the rover all the way; both stay together. "
                 "B: ride to a parking spot, the rover stays there as access point and radio relay "
                 "(astronaut ↔ rover ↔ base) while the astronaut walks to the sample site and back."
        )
        mode = st.radio(
            "Map click action",
            ["Select destination", "Set base camp", "Set rover parking", "Set relay rover"],
            horizontal=True,
            help="Rover parking is used in mode B (a click switches to B). Without a chosen spot the app picks "
                 "the quickest spot that can relay to the base. 'Set relay rover' moves the pre-deployed "
                 "relay rover."
        )
        if st.session_state.get("park") is not None:
            st.caption("Rover parking: chosen on the map (mode B).")
            if st.button("Clear rover parking (auto relay spot)"):
                st.session_state.pop("park", None)
                st.rerun()
        start_h = st.select_slider(
            "EVA start time (local solar time)",
            options=[6.0 + 0.25 * i for i in range(41)],
            value=6.5,
            format_func=hhmm
        )
        stop_min = st.slider(
            "Work time at destination (min)",
            min_value=0,
            max_value=120,
            value=30,
            step=10
        )

    with terrain_tab:
        layer = st.selectbox(
            "Map layer",
            ["Satellite", "Elevation", "Slope"] + ([THERMAL_LAYER] if THERMAL is not None else [])
        )
        if layer == THERMAL_LAYER:
            ti_v = THERMAL[0][np.isfinite(THERMAL[0])]
            st.caption(
                "THEMIS night-time thermal inertia, 100 m pixels (USGS / ASU, ~20 % accuracy). "
                f"🟦 low ≈ {np.percentile(ti_v, 2):.0f}: loose sand, cools fast at night · "
                f"🟥 high ≈ {np.percentile(ti_v, 98):.0f}: solid rock, stays warm (J m⁻² K⁻¹ s⁻½). "
                "Coarse: only a hint for the sand question at the butte tip."
            )
        show_track = st.checkbox("Curiosity's drive (NASA waypoints)", value=True)
        show_photos = st.checkbox(
            f"Curiosity photos + view wedges ({len(PHOTO_LIST)})",
            value=True,
            help="Real Navcam photos (NASA/JPL-Caltech) at the spot where Curiosity took them. The blue area is "
                 "the ground each photo shows (field of view, cut by the terrain). Click an area to open the photo."
        )
        max_slope = st.slider(
            "Astronaut: max walking slope (°)",
            min_value=5,
            max_value=40,
            value=20,
            step=1
        )
        rover_max = st.slider(
            "Rover: max driving slope (°)",
            min_value=5,
            max_value=30,
            value=int(RV["max_slope_deg"]),
            step=1,
            help="Separate from the walking limit. Default from eva/assumptions.json."
        )
        avoid_box = st.checkbox(
            "Rover avoids boxwork ridges",
            value=bool(RV.get("avoid_boxwork", True)),
            help="1–2 m boxwork ridges are not drivable: the rover parks at the edge and the astronaut walks in."
        )
        show_hazards = st.checkbox(
            "Highlight steep terrain (walking limit)",
            value=True
        )
        hazard_opacity = st.slider(
            "Hazard overlay opacity",
            min_value=0.1,
            max_value=1.0,
            value=0.6,
            step=0.1
        )
        if show_hazards:
            st.caption(
                f"🟨 {0.5 * max_slope:.0f}–{0.75 * max_slope:.0f}°  "
                f"🟧 {0.75 * max_slope:.0f}–{max_slope}°  "
                f"🟥 > {max_slope}° (no-go on foot)"
            )

    with comms_tab:
        st.caption(
            "Predicted links use a simplified coverage model: straight line between the antennas over the "
            "elevation grid (with Mars curvature) plus a maximum range. It is not a radio propagation "
            "calculation (no Fresnel zone, diffraction, reflections or link budget). A taller antenna "
            "does not see through a ridge."
        )
        base_h = st.slider("Base antenna height (m)", 1.0, 20.0, float(COMMS["base_antenna_m"]), 0.5)
        rover_h = st.slider("Rover antenna height (m)", 0.5, 6.0, float(COMMS["rover_antenna_m"]), 0.5)
        suit_h = st.slider("Suit antenna height (m)", 1.0, 3.0, float(COMMS["suit_antenna_m"]), 0.1)
        ar_range = st.slider("Astronaut ↔ rover max range (m)", 100, 3000,
                             int(COMMS["suit_rover_range_m"]), 50)
        rb_range = st.slider("Rover ↔ base max range (m)", 500, 20000,
                             int(COMMS["rover_base_range_m"]), 500)
        use_relay = st.checkbox(
            "Relay rover (uncrewed, deployed before the EVA)",
            value=True,
            help="A second rover drives to a high point hours before the EVA and stays parked all mission as "
                 "a fixed relay: astronaut → (crew rover or relay rover) → base. Auto: the drivable spot that "
                 "sees the base and the most of your planned route to the destination. 'Set relay rover' on the "
                 "map overrides it; 'back to auto spot' returns to automatic."
        )
        if st.session_state.get("relay_point") is not None and st.button("Relay rover: back to auto spot"):
            st.session_state.pop("relay_point", None)
            st.rerun()
        show_shadow = st.checkbox(
            "Signal map: where the astronaut has no link (dark)",
            value=False,
            help="With the relay rover: dark = an astronaut there has no line to the relay rover (or the relay "
                 "has no line to the base). Without it: dark = a rover parked there cannot see the base."
        )

    with environment_tab:
        season_source = st.selectbox(
            "Mars orbit position",
            ["Current position (automatic)", "Manual override"]
        )
        if season_source == "Current position (automatic)":
            solar_longitude_deg = current_solar_longitude_deg
            position_label = "MARS NOW"
            st.write(f"NASA Mars24 calculation: **Ls {solar_longitude_deg:.2f}°**")
        else:
            position_label = "MARS SCENARIO"
            solar_longitude_deg = st.slider(
                "Manual solar longitude (Ls °)",
                min_value=0.0,
                max_value=359.0,
                value=float(round(current_solar_longitude_deg)),
                step=1.0
            )
        st.caption(f"{southern_season(solar_longitude_deg)} at Gale Crater (Ls 0° = northern spring equinox).")
        if st.button("Reset navigation"):
            for k in ("base", "destination", "last_click", "last_obj", "show_route_dialog", "park", "sim",
                      "relay_point"):
                st.session_state.pop(k, None)
            st.rerun()
        st.caption(
            "Slope and daylight limits are planning assumptions, not validated "
            "astronaut safety thresholds."
        )

    with data_tab:
        st.write("Routing grid", f"{cols} × {rows}")
        st.write("Grid cells", f"{dx:.2f} × {dy:.2f} m")
        st.write("Valid slope coverage", f"{100 * np.isfinite(slope).mean():.1f}%")
        st.write("Boxwork / rough ground", f"{100 * is_ridge.mean():.1f}%")

comms = {
    "base_antenna_m": base_h, "rover_antenna_m": rover_h, "suit_antenna_m": suit_h,
    "suit_rover_range_m": float(ar_range), "rover_base_range_m": float(rb_range),
}

season_A = assumptions_for_season(solar_longitude_deg)
sunrise, sunset = eb.solar_window_hours(season_A)
safe_window = eb.solar_window_hours(
    season_A,
    season_A["limits"]["min_sun_elev_deg"]
)


# ==========================================================
# SESSION STATE
# ==========================================================

if "base" not in st.session_state:
    # default base camp = habitat B (was the map centre, inside the boxwork)
    hx, hy = data["habitat"]
    inside = bounds.left <= hx <= bounds.right and bounds.bottom <= hy <= bounds.top
    st.session_state.base = (hx, hy) if inside else (
        (bounds.left + bounds.right) / 2,
        (bounds.bottom + bounds.top) / 2
    )

for _k in ("destination", "last_click", "park", "relay_point"):
    if _k not in st.session_state:
        st.session_state[_k] = None

base_x, base_y = st.session_state.base
destination = st.session_state.destination
base_cell = map_to_grid(base_x, base_y)
park_point = st.session_state.park
park_cell = map_to_grid(*park_point) if park_point is not None else None

# ---- relay rover. Auto: the spot that sees the base and the most of the planned route to the current
# destination (the part where the astronaut is: walk in mode B, drive in mode A, plus the destination).
# Without a destination: the science stops. Manual ("Set relay rover") always wins.
STOP_CELLS = tuple(map_to_grid(x, y) for _, x, y in data["stops"])
RELAY_SAMPLES = 8


def route_targets(p):
    """Up to RELAY_SAMPLES evenly spaced cells of the astronaut's part of a plan, ending at the destination."""
    path = p["walk_path"] if len(p["walk_path"]) > 1 else p["drive_path"]
    if len(path) < 2:
        return ()
    idx = np.unique(np.linspace(0, len(path) - 1, RELAY_SAMPLES).round().astype(int))
    return tuple(tuple(int(v) for v in path[i]) for i in idx)


relay_point = st.session_state.relay_point
relay_info, relay_cell = None, None
relay_targets, relay_target_kind = STOP_CELLS, "science stops"
if use_relay:
    if destination is not None:
        p0 = make_plan(travel, base_cell, map_to_grid(*destination), max_slope, stop_min, rover_max, avoid_box,
                       comms, park_cell if travel == "drive" else None, None)
        if "error" not in p0 and route_targets(p0):
            relay_targets, relay_target_kind = route_targets(p0), "points of the planned route"
    relay_info = relay_spot(base_cell, relay_targets, rover_max, avoid_box, comms,
                            map_to_grid(*relay_point) if relay_point is not None else None)
    relay_cell = relay_info.get("cell")


# ==========================================================
# PLAN THE EVA (selected mode + the other options for comparison)
# ==========================================================

plan, result = None, None
refs = {}                     # travel -> (plan, result) or error text
route_status = "Select a destination."

if destination is not None:

    goal = map_to_grid(*destination)
    for tr in ("ride", "drive", "walk"):
        p_ = make_plan(tr, base_cell, goal, max_slope, stop_min, rover_max, avoid_box, comms,
                       park_cell if tr == "drive" else None, relay_cell)
        if "error" in p_:
            refs[tr] = p_["error"]
        else:
            refs[tr] = (p_, evaluate(p_, base_cell, start_h, solar_longitude_deg, rover_max, comms, relay_cell))

    if isinstance(refs[travel], str):
        route_status = refs[travel]
    else:
        plan, result = refs[travel]

verdict = None
if result is not None:
    n, lo, hi = (result["res"][c] for c in ("nominal", "best", "worst"))
    verdict = result["verdict"]
    one_way = (plan["drive_path"] + plan["walk_path"][1:]) if plan["drive_path"] else plan["walk_path"]
    route_distance = (n["walk_distance_m"] + n["drive_distance_m"]) / 2
    heights = [float(elevation[c]) for c in one_way]
    elevation_gain = sum(max(0.0, b - a) for a, b in zip(heights, heights[1:]))
    route_max_slope = max(float(slope[c]) for c in one_way)

if verdict == "GO":
    status_color, status_text = "#48c78e", "EVA GO"
elif verdict == "MARGINAL":
    status_color, status_text = "#f0b44d", "EVA MARGINAL"
elif destination is not None and plan is None and travel == "ride" and not isinstance(refs.get("drive"), str):
    # mode A fails only because the rover can't get there; walking (mode B) works
    status_color, status_text = "#f0b44d", "ROVER CAN'T REACH · WALKING WORKS"
elif destination is not None:
    status_color = "#e25b5b"
    status_text = "EVA NO-GO" if plan is not None else "NO SAFE ROUTE"
else:
    status_color, status_text = "#91a0ad", "SELECT DESTINATION"


# ==========================================================
# SIMULATION STATE (simulated data only; elapsed time in simulated seconds)
# ==========================================================

SPEEDS = [1, 10, 30, 60, 120, 300]


@st.cache_data(show_spinner="Building the simulation...")
def sim_timeline(plan_legs, base_cell, start_h, solar_longitude_deg, rover_max, comms, abort_t, relay_cell=None):
    """EVA-model timing + consumables per leg, optional 'return to rover', then the sampled timeline."""
    def with_eva(legs):
        _, df = eva_cases(legs, start_h, solar_longitude_deg, rover_max)
        out = []
        for l, (_, r) in zip(legs, df.iterrows()):
            out.append(dict(l, dur_s=float(r["duration_min"]) * 60, o2_kg=float(r["o2_kg"]),
                            water_kg=float(r["water_kg"]), suit_battery_pct=float(r["suit_battery_pct"])))
        return out

    legs = with_eva([dict(l) for l in plan_legs])
    aborted = False
    if abort_t is not None:
        new, aborted = sim.abort_to_rover(legs, abort_t)
        if aborted:
            keep = ("kind", "name", "length_m", "dz_m", "terrain", "duration_min", "a", "b", "together")
            legs = with_eva([{k: l[k] for k in keep if k in l} for l in new])
    tl = sim.build_timeline(legs, base_cell, TERRAIN, NAV, comms, relay_rc=relay_cell)
    return legs, tl, aborted


sim_key = (travel, st.session_state.base, destination, park_point if travel == "drive" else None,
           start_h, stop_min, max_slope, rover_max, avoid_box, tuple(sorted(comms.items())), relay_cell,
           solar_longitude_deg if season_source == "Manual override" else "auto")
if st.session_state.get("sim", {}).get("key") != sim_key:
    old_speed = st.session_state.get("sim", {}).get("speed", 60)
    st.session_state.sim = {"key": sim_key, "t": 0.0, "running": False, "wall": time.monotonic(),
                            "speed": old_speed, "abort": None, "ls": solar_longitude_deg}
S = st.session_state.sim

sim_legs, TL, sim_aborted = (None, None, False)
if plan is not None:
    sim_legs, TL, sim_aborted = sim_timeline(plan["legs"], base_cell, start_h, S["ls"], rover_max, comms,
                                             S["abort"], relay_cell)

# Optional data export for the explainer video (video/): ARES_EXPORT=<file.json> streamlit run ...
if os.environ.get("ARES_EXPORT") and TL is not None:
    def _xy(rc):
        return [bounds.left + (float(rc[1]) + 0.5) * dx, bounds.top - (float(rc[0]) + 0.5) * dy]
    _k = slice(None, None, 4)                                  # every 4th sample = 20 s of simulated time
    Path(os.environ["ARES_EXPORT"]).write_text(json.dumps({
        "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top], "kx": data["kx"], "ky": data["ky"],
        "base": _xy(base_cell), "relay": _xy(relay_cell) if relay_cell else None, "park": _xy(plan["park"]),
        "relay_path": [_xy(c) for c in relay_info["path"]] if relay_info else [],
        "drive_path": [_xy(c) for c in plan["drive_path"]], "walk_path": [_xy(c) for c in plan["walk_path"]],
        "destination": list(destination), "stops": [[n_, x_, y_] for n_, x_, y_ in data["stops"]],
        "track": [[s_, x_, y_] for _, s_, x_, y_ in data["track"]],
        "photos": PHOTO_LIST, "start_h": start_h, "T": TL["T"], "verdict": verdict,
        "metrics": {"one_way_m": route_distance, "round_trip_h": n["duration_h"], "link_pct": result["link_pct"],
                    "o2_kg": n["o2_kg"], "water_kg": n["water_kg"], "peak_btuh": n["peak_btuh"],
                    "max_slope": route_max_slope, "drive_m": n["drive_distance_m"] / 2,
                    "walk_m": n["walk_distance_m"] / 2},
        "tl": {"t": TL["t"][_k].tolist(), "astro": [_xy(p) for p in TL["astro"][_k]],
               "rover": [_xy(p) for p in TL["rover"][_k]], "est": [_xy(p) for p in TL["est_astro"][_k]],
               "sigma": TL["sigma_astro"][_k].round(2).tolist(),
               "ar": TL["ar"][_k].tolist(), "rb": TL["rb"][_k].tolist(), "aq": TL["aq"][_k].tolist(),
               "qb": TL["qb"][_k].tolist(), "full": TL["full"][_k].tolist(), "last_rx": TL["last_rx"][_k].tolist(),
               "sample": TL["sample"][_k].tolist(), "o2": TL["o2_used_kg"][_k].round(4).tolist(),
               "water": TL["water_used_kg"][_k].round(3).tolist(),
               "batt": TL["suit_battery_pct"][_k].round(1).tolist(),
               "phase": [sim.phase(sim_legs, int(TL["leg"][i]), float(TL["t"][i]), TL["T"])[0]
                         for i in range(0, len(TL["t"]), 4)]},
    }, default=float))


def advance_clock():
    """Move simulated time forward by wall time x speed. Returns True when the EVA just finished."""
    now = time.monotonic()
    done = False
    if S["running"] and TL is not None:
        S["t"] = min(TL["T"], S["t"] + (now - S["wall"]) * S["speed"])
        if S["t"] >= TL["T"]:
            S["running"], done = False, True
    S["wall"] = now
    return done


# ==========================================================
# MAP (static part: imagery, overlays, planned routes)
# ==========================================================

center_x = (bounds.left + bounds.right) / 2
center_y = (bounds.bottom + bounds.top) / 2

image_bounds = [
    [bounds.bottom, bounds.left],
    [bounds.top, bounds.right]
]

m = folium.Map(
    location=[center_y, center_x],
    crs="Simple",
    tiles=None,
    zoom_control=True,
    zoom_snap=0.25,
    zoom_delta=0.5
)
m.get_root().html.add_child(folium.Element(
    "<style>.leaflet-top.leaflet-left { top: 48px; }"
    # the map fills the whole window, whatever the screen height (was fixed at 900 px)
    " html, body, #root, #parent, .float-child { height: 100vh !important; margin: 0; }"
    " #map_div { height: 100vh !important; }"
    # no browser focus box around clicked map shapes (drew a big rectangle around photo wedges)
    " .leaflet-interactive:focus, .leaflet-container :focus { outline: none !important; }</style>"
))


class MetricScale(MacroElement):
    """Metric scale bar (map units are metres, so Leaflet's scale is correct)."""
    _template = Template("""{% macro script(this, kwargs) %}
        L.control.scale({imperial: false, maxWidth: 160}).addTo({{ this._parent.get_name() }});
    {% endmacro %}""")


m.add_child(MetricScale())

# folium hands min_zoom/max_zoom only to a tile layer. With tiles=None they are
# dropped and Leaflet stops at zoom 0 (1 px = 1 m), so the 3 km map could not be
# zoomed out. Set them on the map itself: -4 shows the whole area, 3 = 8 px/m.
m.options.update(min_zoom=-5, max_zoom=3)


class CoverView(MacroElement):
    """The image always fills the screen: the furthest zoom-out is the one where the image just covers
    the view (recomputed when the window is resized), and panning stops at the image edge. The bottom
    edge gets a small extra margin so the strip behind the bottom dock can still be dragged into view."""
    _template = Template("""{% macro script(this, kwargs) %}
        (function(map) {
            var img = L.latLngBounds([[{{ this.s }}, {{ this.w }}], [{{ this.n }}, {{ this.e }}]]);
            var pan = L.latLngBounds([[{{ this.s_pad }}, {{ this.w }}], [{{ this.n }}, {{ this.e }}]]);
            function cover() {
                var z = map.getBoundsZoom(img, true);
                map.setMinZoom(z);
                if (map.getZoom() < z) { map.setZoom(z); }
            }
            map.setMaxBounds(pan);
            map.options.maxBoundsViscosity = 1.0;
            cover();
            map.setView(img.getCenter(), map.getMinZoom(), {animate: false});
            map.on('resize', cover);
        })({{ this._parent.get_name() }});
    {% endmacro %}""")

    def __init__(self, b, bottom_pad):
        super().__init__()
        self.s, self.w, self.n, self.e = b.bottom, b.left, b.top, b.right
        self.s_pad = b.bottom - bottom_pad

folium.raster_layers.ImageOverlay(
    image=data["image_url"],
    bounds=image_bounds,
    name="Satellite"
).add_to(m)

# ---- steep terrain hazard overlay (walking limit)
if show_hazards:

    hazard_overlay = np.zeros((rows, cols, 4), dtype=np.uint8)
    valid = np.isfinite(slope)

    moderate = valid & (slope >= 0.5 * max_slope) & (slope < 0.75 * max_slope)
    steep = valid & (slope >= 0.75 * max_slope) & (slope <= max_slope)
    unsafe = valid & (slope > max_slope)

    hazard_overlay[moderate] = (242, 200, 0, 110)
    hazard_overlay[steep] = (245, 120, 20, 170)
    hazard_overlay[unsafe] = (225, 35, 35, 230)

    folium.raster_layers.ImageOverlay(
        image=hazard_overlay,
        bounds=image_bounds,
        opacity=hazard_opacity,
        name="Steep terrain"
    ).add_to(m)

# ---- signal map. With the relay rover: where an astronaut has no line to it (or it has none to the
# base). Without it: where a parked rover has no line of sight to the base antenna.
if show_shadow:

    if relay_cell is not None:
        seen = (viewshed(relay_cell, rover_h, suit_h) & (ground_dist_grid(relay_cell) <= ar_range)
                if relay_info["sees_base"] else np.zeros((rows, cols), dtype=bool))
    else:
        seen = coverage(base_cell, rover_h, base_h) & (ground_dist_grid(base_cell) <= rb_range)
    shadow = np.zeros((rows, cols, 4), dtype=np.uint8)
    shadow[~seen & np.isfinite(elevation)] = (10, 20, 60, 165)

    folium.raster_layers.ImageOverlay(
        image=shadow,
        bounds=image_bounds,
        name="Signal map"
    ).add_to(m)

# ---- THEMIS thermal inertia (100 m pixels): low = loose sand (cold at night), high = rock (warm at night)
if layer == THERMAL_LAYER and THERMAL is not None:
    ti, ti_bounds = THERMAL
    v = ti[np.isfinite(ti)]
    lo_, hi_ = np.percentile(v, 2), np.percentile(v, 98)
    t_ = np.clip((ti - lo_) / max(hi_ - lo_, 1e-9), 0, 1)
    # blue (sand) -> pale -> orange-red (rock)
    stops_rgb = np.array([[40, 90, 200], [120, 170, 230], [240, 230, 200], [245, 160, 70], [200, 50, 30]], float)
    pos = t_ * (len(stops_rgb) - 1)
    i0 = np.clip(np.floor(pos).astype(int), 0, len(stops_rgb) - 2)
    fr = (pos - i0)[..., None]
    rgb = stops_rgb[i0] * (1 - fr) + stops_rgb[i0 + 1] * fr
    ov = np.zeros(ti.shape + (4,), dtype=np.uint8)
    ov[..., :3] = rgb.astype(np.uint8)
    ov[..., 3] = np.where(np.isfinite(ti), 165, 0)
    folium.raster_layers.ImageOverlay(image=ov, bounds=ti_bounds, name="Thermal inertia",
                                      mercator_project=False).add_to(m)

# ---- elevation or slope on the routing grid
if layer in ("Elevation", "Slope"):

    values = elevation if layer == "Elevation" else slope
    valid = np.isfinite(values)

    if np.any(valid):

        low = float(np.nanmin(values)) if layer == "Elevation" else 0.0
        high = float(np.nanmax(values)) if layer == "Elevation" else max(30.0, float(max_slope))

        normalized = np.clip((values - low) / max(high - low, 1e-9), 0, 1)
        normalized = np.where(valid, normalized, 0)

        overlay = np.zeros((rows, cols, 4), dtype=np.uint8)
        overlay[:, :, 0] = (255 * normalized).astype(np.uint8)
        overlay[:, :, 1] = (255 * (1 - np.abs(2 * normalized - 1))).astype(np.uint8)
        overlay[:, :, 2] = (255 * (1 - normalized)).astype(np.uint8)
        overlay[:, :, 3] = np.where(valid, 220, 0).astype(np.uint8)

        folium.raster_layers.ImageOverlay(
            image=overlay,
            bounds=image_bounds,
            name=layer
        ).add_to(m)

# ---- Curiosity's real drive (every waypoint inside the map), one line per stretch inside the map
if data["track"] and show_track:
    track_layer = folium.FeatureGroup(name="Curiosity's drive")
    for seg in sorted({t[0] for t in data["track"]}):
        pts = [t for t in data["track"] if t[0] == seg]
        folium.PolyLine(
            [[y, x] for _, _, x, y in pts],
            color="#ffffff", weight=2, opacity=0.9, dash_array="5 5",
            tooltip=f"Curiosity's drive, sols {pts[0][1]}–{pts[-1][1]} (NASA waypoints)"
        ).add_to(track_layer)
    for _, sol_, x, y in (data["track"][0], data["track"][-1]):
        folium.Marker([y, x], icon=folium.DivIcon(
            html=f'<div style="font:600 9px monospace;color:#fff;background:rgba(10,16,22,.8);padding:0 3px;'
                 f'white-space:nowrap">sol {sol_}</div>', icon_size=(60, 12), icon_anchor=(-4, 6))).add_to(track_layer)
    track_layer.add_to(m)

# ---- Curiosity photos: camera position + the ground each frame shows (view footprint)
PHOTO_COLOR = "#7dd3fc"
if show_photos and PHOTO_LIST:
    ph_layer = folium.FeatureGroup(name="Curiosity photos")
    for ph in PHOTO_LIST:
        if len(ph["footprint"]) < 3:
            continue
        fx, fy = warp_transform(MARS_LONLAT, GRID_CRS_WKT, [p[0] for p in ph["footprint"]],
                                [p[1] for p in ph["footprint"]])
        cx, cy = warp_transform(MARS_LONLAT, GRID_CRS_WKT, [ph["rover_lon"]], [ph["rover_lat"]])
        n_half = len(fx) // 2
        wedge = [[cy[0], cx[0]], [fy[0], fx[0]]] + [[y, x] for x, y in zip(fx[n_half:], fy[n_half:])] \
            + [[fy[n_half - 1], fx[n_half - 1]], [cy[0], cx[0]]]
        tip = (f"📷 sol {ph['sol']} · {ph['camera'].replace('_', ' ').title()} · looking {ph['az']:.0f}° "
               f"(el {ph['el']:+.0f}°) · sees ~{ph['far_m'] or 0:.0f} m")
        pop = (f'<div style="width:330px;font:11px monospace">'
               f'<img src="/{ph["thumb"]}" style="width:330px;display:block;margin-bottom:4px">'
               f'<b>Curiosity {ph["camera"].replace("_", " ").title()}</b> · {ph["lmst"]}<br>'
               f'looking {ph["az"]:.0f}° from north, {ph["el"]:+.0f}° · field of view {ph["hfov"]:.0f}°×{ph["vfov"]:.0f}°'
               + ('<br><i>blue line = our EVA route drawn into the photo</i>' if ph.get("route_drawn") else "")
               + f'<br><a href="{ph["page"]}" target="_blank">NASA raw image {ph["id"]}</a> · NASA/JPL-Caltech</div>')
        # light styling: 81 overlapping wedges must not hide the terrain or the routes
        folium.Polygon([[y, x] for x, y in zip(fx, fy)], color=PHOTO_COLOR, weight=0.8, opacity=0.45,
                       fill=True, fill_color=PHOTO_COLOR, fill_opacity=0.045, tooltip=tip,
                       popup=folium.Popup(pop, max_width=350)).add_to(ph_layer)
        folium.PolyLine(wedge[:2] + [wedge[-2], wedge[-1]], color=PHOTO_COLOR, weight=1, opacity=0.5,
                        dash_array="2 3").add_to(ph_layer)
    for (lat_, lon_), group in PHOTO_SITES.items():
        cx, cy = warp_transform(MARS_LONLAT, GRID_CRS_WKT, [lon_], [lat_])
        sols = sorted({g["sol"] for g in group})
        folium.Marker([cy[0], cx[0]], tooltip=f"📷 Curiosity camera position · sol {', '.join(map(str, sols))} · "
                                               f"{len(group)} photos (click a blue wedge to open one)",
                      icon=folium.DivIcon(html=camera_pin(len(group)), icon_size=(28, 28),
                                          icon_anchor=(14, 14))).add_to(ph_layer)
    ph_layer.add_to(m)

# science stops of the groundwater mission: only shown while the mission is switched on
if st.session_state.get("mission_on"):
    stops_layer = folium.FeatureGroup(name="Science stops (click = go there)")
    for name, x, y in data["stops"]:
        folium.CircleMarker(
            [y, x], radius=7, color="#ffffff", weight=2,
            fill=True, fill_color="#f2b400", fill_opacity=1,
            tooltip=f"{name} · groundwater mission sample site (click = plan the EVA)"
        ).add_to(stops_layer)
    stops_layer.add_to(m)

m.add_child(CoverView(bounds, 0.08 * (bounds.top - bounds.bottom)))   # replaces fit_bounds

# ---- base, destination, planned parking, planned routes
folium.Marker(
    [base_y, base_x],
    tooltip=f"Base camp (habitat) · antenna mast {base_h} m",
    icon=folium.DivIcon(html=plate(habitat_svg(80), "BASE", "#54d6e8"), icon_size=(80, 66),
                        icon_anchor=(40, 41))
).add_to(m)

if destination is not None:
    folium.Marker(
        [destination[1], destination[0]],
        tooltip="Destination (sample site)",
        icon=folium.Icon(color="red", icon="flag")
    ).add_to(m)

WALK_COLOR, DRIVE_COLOR = "#54d6e8", "#e879f9"
LINK_COLOR = {sim.OK: "#48c78e", sim.DOWN: "#e25b5b", sim.UNKNOWN: "#91a0ad"}
LINK_DASH = {sim.OK: None, sim.DOWN: "8 6", sim.UNKNOWN: "2 6"}
LINK_ICON = {sim.OK: "● connected", sim.DOWN: "✕ disconnected", sim.UNKNOWN: "? unknown"}


def draw_route(path, link, color, label):
    """Solid where the base has a live link, dotted where it does not."""
    if len(path) < 2:
        return
    start = 0
    for i in range(1, len(link) + 1):
        if i == len(link) or link[i] != link[start]:
            pts = [grid_to_map(c) for c in path[start:i + 1]]
            folium.PolyLine(
                [[y, x] for x, y in pts],
                color=color, weight=5, opacity=0.9,
                dash_array=None if link[start] else "2 9",
                tooltip=f"{label} (planned) · {'live link to base' if link[start] else 'NO live link to base'}"
            ).add_to(m)
            start = i


def pin_html(text, bg, fg="#0a1016", size=22):
    return (f'<div style="display:inline-flex;vertical-align:middle;width:{size}px;height:{size}px;'
            f'border-radius:50%;background:{bg};color:{fg};'
            f'border:2px solid #0a1016;box-shadow:0 0 0 2px {bg}55;font:700 {size // 2}px monospace;'
            f'align-items:center;justify-content:center">{text}</div>')


if plan is not None:
    link = result["link"]
    n_drive = max(len(plan["drive_path"]) - 1, 0)
    n_walk = max(len(plan["walk_path"]) - 1, 0)
    draw_route(plan["drive_path"], link[:n_drive], DRIVE_COLOR, "Rover route")
    draw_route(plan["walk_path"], link[n_drive:n_drive + n_walk], WALK_COLOR, "Walking route")
    if travel == "drive" and plan["park"] != base_cell:
        px, py = grid_to_map(plan["park"])
        folium.Marker(
            [py, px],
            tooltip="Rover parking spot (mode B)" + (" · chosen on the map" if park_point else " · auto relay spot"),
            icon=folium.DivIcon(html=pin_html("P", "#e879f9"), icon_size=(22, 22), icon_anchor=(11, 11))
        ).add_to(m)


def moved_marker(clicked, actual, label, why):
    """Grey pin where the user clicked + dashed line to where the vehicle really goes."""
    ax_, ay_ = grid_to_map(actual)
    folium.PolyLine([[clicked[1], clicked[0]], [ay_, ax_]], color="#91a0ad", weight=2, dash_array="4 4",
                    tooltip=why).add_to(m)
    folium.Marker([clicked[1], clicked[0]], tooltip=f"You clicked here. {why}",
                  icon=folium.DivIcon(html=pin_html(label, "#91a0ad"), icon_size=(22, 22),
                                      icon_anchor=(11, 11))).add_to(m)


if park_point is not None and travel == "drive":
    if plan is None:
        folium.Marker(
            [park_point[1], park_point[0]],
            tooltip="Rover parking spot (chosen)",
            icon=folium.DivIcon(html=pin_html("P", "#e879f9"), icon_size=(22, 22), icon_anchor=(11, 11))
        ).add_to(m)
    elif plan.get("park_note"):
        moved_marker(park_point, plan["park"], "P", plan["park_note"])

# ---- relay rover: driven out hours before the EVA, parked all mission
RELAY_COLOR = "#f0b44d"
if relay_cell is not None:
    if len(relay_info["path"]) > 1:
        folium.PolyLine([[y, x] for x, y in (grid_to_map(c) for c in relay_info["path"])],
                        color=RELAY_COLOR, weight=2, opacity=0.7, dash_array="3 6",
                        tooltip=f"Relay rover deployment drive ({relay_info['drive_m']:.0f} m, "
                                f"~{relay_info['drive_min']:.0f} min, done before the EVA)").add_to(m)
    qx, qy = grid_to_map(relay_cell)
    folium.Marker(
        [qy, qx],
        tooltip=(f"Lead antenna / relay rover (uncrewed, simulated) · "
                 f"{'placed by you' if relay_point is not None else 'auto spot for this EVA'} · deployed before "
                 f"the EVA, parked all mission · sees the base: {'yes' if relay_info['sees_base'] else 'NO'} · "
                 f"sees {relay_info['stops_seen']}/{len(relay_targets)} {relay_target_kind}"),
        icon=folium.DivIcon(html=plate(relay_svg(46), "RELAY", RELAY_COLOR), icon_size=(50, 66),
                            icon_anchor=(25, 48))
    ).add_to(m)
    if relay_point is not None and relay_info.get("note"):
        moved_marker(relay_point, relay_cell, "R", relay_info["note"])

# ---- status chip (top right) and legend (bottom left), drawn inside the map
map_status = f"""
<div style="position:fixed;top:12px;right:12px;z-index:9999;
background:rgba(10,16,22,.92);color:#e8f0f2;border:1px solid #34424d;
border-left:4px solid {status_color};padding:10px 14px;font:600 12px monospace;
letter-spacing:.08em;box-shadow:0 6px 18px rgba(0,0,0,.28)">
<span style="color:{status_color}">{status_text}</span>
<span style="color:#9eacb6;font-weight:400"> · MODE {'A' if travel == 'ride' else 'B'}</span><br>
<span style="color:#9eacb6;font-weight:400">{position_label} LS {solar_longitude_deg:.2f}° · SUN {hhmm(sunrise)}–{hhmm(sunset)}</span>
</div>
"""
m.get_root().html.add_child(folium.Element(map_status))


def _line(color, style="solid", width=4):
    return (f'<span style="display:inline-block;width:20px;border-top:{width}px {style} {color};'
            'vertical-align:middle;margin-right:6px"></span>')


ASTRO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 50" width="{w}" '
    'style="vertical-align:middle;overflow:visible">'
    '<defs><linearGradient id="asuit" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#ffffff"/>'
    '<stop offset=".6" stop-color="#d9dee3"/><stop offset="1" stop-color="#8f99a3"/></linearGradient>'
    '<radialGradient id="avisor" cx=".35" cy=".3" r=".8"><stop offset="0" stop-color="#ffe7a3"/>'
    '<stop offset=".5" stop-color="#d79a2b"/><stop offset="1" stop-color="#6b4210"/></radialGradient>'
    '<radialGradient id="ashadow" cx=".5" cy=".5" r=".5"><stop offset="0" stop-color="#000" stop-opacity=".6"/>'
    '<stop offset="1" stop-color="#000" stop-opacity="0"/></radialGradient></defs>'
    '<ellipse cx="20" cy="47" rx="12" ry="2.8" fill="url(#ashadow)"/>'
    '<rect x="23" y="17" width="9" height="15" rx="2" fill="#9aa3ab" stroke="#4e5760" stroke-width=".5"/>'
    '<rect x="13" y="33" width="6" height="13" rx="2.5" fill="url(#asuit)" stroke="#5c656e" stroke-width=".5"/>'
    '<rect x="20.5" y="33" width="6" height="13" rx="2.5" fill="url(#asuit)" stroke="#5c656e" stroke-width=".5"/>'
    '<path d="M10 20 Q10 15 16 15 L25 15 Q29 15 29 20 L28 35 L11 35 Z" fill="url(#asuit)" stroke="#5c656e" '
    'stroke-width=".5"/>'
    '<rect x="6.5" y="17" width="5" height="13" rx="2.5" fill="url(#asuit)" stroke="#5c656e" stroke-width=".5"/>'
    '<rect x="27.5" y="17" width="5" height="13" rx="2.5" fill="url(#asuit)" stroke="#5c656e" stroke-width=".5"/>'
    '<rect x="15" y="21" width="9" height="5" rx="1" fill="#c7cfd6" stroke="#5c656e" stroke-width=".4"/>'
    '<circle cx="17" cy="23.5" r=".8" fill="#54d6e8"/>'
    '<circle cx="19.5" cy="9.5" r="8" fill="url(#asuit)" stroke="#5c656e" stroke-width=".6"/>'
    '<ellipse cx="18.5" cy="10" rx="5.6" ry="4.6" fill="url(#avisor)"/>'
    '<path d="M15 8 Q17 6 20 6.4" stroke="#fff" stroke-width=".8" fill="none" opacity=".8"/>'
    '</svg>'
)

legend = f"""
<div style="position:fixed;left:12px;bottom:44px;z-index:9999;background:rgba(10,16,22,.88);
color:#dce6e9;border:1px solid #34424d;padding:6px 10px;font:11px monospace;line-height:1.7">
<b style="color:#91a0ad">PLANNED</b><br>
{_line(WALK_COLOR)}walking route<br>
{_line(DRIVE_COLOR)}rover route<br>
{_line('#cbd5e1', 'dotted')}no live link to base<br>
{pin_html('P', '#e879f9', size=14)} rover parking (B) · {pin_html('P', '#91a0ad', size=14)} your click (moved)<br>
{habitat_svg(30)} base camp<br>
{relay_svg(16)} lead antenna (relay rover) · {_line('#f0b44d', 'dashed', 2)}its drive out<br>
<span style="color:#e25b5b">⚑</span> destination · <span style="color:#22c55e">◆</span> sample site<br>
<span style="color:#f2b400">●</span> science stop · {_line('#ffffff', 'dashed', 2)}Curiosity<br>
<b style="color:#91a0ad">SIMULATED</b><br>
{rover_svg(26)} crew rover · {ASTRO_SVG.format(w=12)} astronaut (estimate)<br>
<span style="color:#54d6e8">◯</span> position uncertainty (illustrative)<br>
<span style="color:#fff">•</span> true position (sim only) · <span style="color:#91a0ad">◌</span> base view<br>
{_line(LINK_COLOR[sim.OK], 'solid', 3)}link up {_line(LINK_COLOR[sim.DOWN], 'dashed', 3)}down {_line(LINK_COLOR[sim.UNKNOWN], 'dotted', 3)}unknown
</div>
"""
m.get_root().html.add_child(folium.Element(legend))


# ==========================================================
# ROUTE ASSESSMENT (dialog) + DOCK
# ==========================================================

def reserve_bar(label, left_pct, worst_left_pct, detail):
    """Gauge of what is left at return; the white tick is the 25% reserve."""
    tone = "good" if worst_left_pct >= RESERVE_PCT else ("warn" if left_pct >= RESERVE_PCT else "bad")
    width = max(0.0, min(100.0, left_pct))
    return (f'<div class="bar"><div class="bh"><span>{label}</span><span><b>{left_pct:.0f}%</b> left at return</span></div>'
            f'<div class="bt"><div class="bf {tone}" style="width:{width:.0f}%"></div>'
            f'<div class="br" style="left:{RESERVE_PCT:.0f}%"></div></div><div class="bd">{detail}</div></div>')


def rng(key, fmt):
    """Best–worst range for a metric's hover help."""
    return f"best–worst case {lo[key]:{fmt}}–{hi[key]:{fmt}}"


@st.dialog("EVA route assessment", width="large")
def show_route_assessment():
    text = (f"**{verdict}** · {MODE_LABEL[travel]} · "
            f"depart {hhmm(start_h)} · return {hhmm(n['end_h'])}")
    if verdict == "GO":
        st.success("✓ " + text)
    elif verdict == "MARGINAL":
        st.warning("! " + text + " · worst case fails")
    else:
        st.error("✕ " + text)

    fails = n["failed"] if verdict == "NO-GO" else hi["failed"]
    if fails:
        st.caption("LIMITING CONDITIONS · " + " · ".join(CHECK_TEXT.get(f, f) for f in fails))
    else:
        st.caption("ALL LIMITS MET WITH A 25% RESERVE, EVEN IN THE WORST CASE")

    # ---- where the rover parks
    if travel == "drive":
        walk_in = sum(l["length_m"] for l in plan["legs"] if l["name"] == "Walk in")
        if plan.get("park_note"):
            st.markdown(f'<div class="note">{plan["park_note"]}</div>', unsafe_allow_html=True)
        if plan["relay"]:
            extra = plan.get("relay_cost_min", 0.0)
            st.markdown(
                f'<div class="note">{rover_svg(30)} ROVER RELAY · parks {walk_in:.0f} m from the destination, '
                'on a spot that sees both the base and your work site: voice, position and suit data reach '
                'the base live.' + (f' Costs {extra:.0f} min more than the quickest parking spot.'
                                    if extra >= 1 and not park_point else '') + '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="note">{rover_svg(30)} <span class="off">NO RELAY FROM THIS SPOT</span> · the parked '
                f'rover ({walk_in:.0f} m away) cannot see both the base and the work site; the base loses '
                'contact while you work.</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(f'<div class="note">{rover_svg(30)} Astronaut rides aboard all the way; rover and astronaut '
                    'stay together, the rover talks to the base.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ONE-WAY RANGE", f"{route_distance:.0f} m",
              help=f"drive {n['drive_distance_m'] / 2:.0f} m + walk {n['walk_distance_m'] / 2:.0f} m")
    c2.metric("ROUND TRIP", f"{n['duration_h']:.2f} h", help=rng("duration_h", ".2f") + " h · limit 4 h")
    c3.metric("LIVE LINK", f"{result['link_pct']:.0f}%",
              help="Share of the EVA time (planned, end of each route step) with the full path "
                   "astronaut ↔ rover ↔ base up")
    c4.metric("MAX SLOPE", f"{route_max_slope:.1f}°")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("O₂", f"{n['o2_kg']:.3f} kg", help=rng("o2_kg", ".3f") + " kg")
    c2.metric("COOLING WATER", f"{n['water_kg']:.2f} kg", help=rng("water_kg", ".2f") + " kg")
    c3.metric("PEAK LOAD", f"{n['peak_btuh']:.0f} BTU/h", help=rng("peak_btuh", ".0f") + " BTU/h · limit 2000")
    c4.metric("ELEVATION GAIN", f"{elevation_gain:.1f} m", help="Climb on the way out")

    sweep = eva_sweep(plan["legs"], solar_longitude_deg, rover_max)
    go = sweep.loc[sweep["verdict"] == "GO", "start_h"]
    best_h, _ = eb.best_start(sweep)
    return_sun = eb.sun_elevation_deg(n["end_h"], season_A)
    if len(go):
        st.info(
            f"GO window {hhmm(go.min())}–{hhmm(go.max())} · "
            f"recommended departure {hhmm(best_h)} · sun at return {return_sun:.0f}°"
        )
    else:
        st.warning(f"No departure time passes the worst-case limits. Sun at return {return_sun:.0f}°.")

    res_tab, time_tab, cmp_tab, when_tab, more_tab = st.tabs(
        ["Reserves", "Timeline", "Compare modes", "When to go", "Details"]
    )

    with res_tab:
        cons = A["consumables"]
        bars = [
            reserve_bar("Oxygen", 100 - n["o2_used_pct"], 100 - hi["o2_used_pct"],
                        f"uses {n['o2_kg']:.3f} kg of {cons['o2_capacity_kg']} kg "
                        f"(range {lo['o2_kg']:.3f}–{hi['o2_kg']:.3f})"),
            reserve_bar("Cooling water", 100 - n["water_used_pct"], 100 - hi["water_used_pct"],
                        f"uses {n['water_kg']:.2f} kg of {cons['water_capacity_kg']} kg "
                        f"(range {lo['water_kg']:.2f}–{hi['water_kg']:.2f})"),
            reserve_bar("Suit battery", n["suit_battery_end_pct"], hi["suit_battery_min_pct"],
                        f"lowest point {n['suit_battery_min_pct']:.0f}% · recharged while riding the rover"),
            reserve_bar("Rover battery", n["rover_battery_end_pct"], hi["rover_battery_end_pct"],
                        f"uses {n['rover_used_kWh']:.2f} kWh of {RV['battery_Wh'] / 1000:.0f} kWh "
                        "(driving, relay radio, suit charging)"),
        ]
        st.markdown('<div class="sec">Reserves at return · white mark = 25% reserve</div>' + "".join(bars),
                    unsafe_allow_html=True)

    with time_tab:
        legs_df = result["legs"]
        phases = legs_df.groupby((legs_df["name"] != legs_df["name"].shift()).cumsum(), sort=False).agg(
            name=("name", "first"), start=("start_h", "first"), minutes=("duration_min", "sum"),
            metres=("length_m", "sum"),
            linked=("link", lambda s: float((s * legs_df.loc[s.index, "duration_min"]).sum())))
        trs = ""
        for _, p in phases.iterrows():
            share = 100 * p["linked"] / max(p["minutes"], 1e-9)
            cls, txt = ("on", "● live") if share >= 99.5 else (("off", "○ none") if share < 0.5 else
                                                              ("off", f"◐ {share:.0f}%"))
            trs += (f'<tr><td>{hhmm(p["start"])}</td><td>{p["name"]}</td>'
                    f'<td class="num">{p["metres"]:.0f} m</td><td class="num">{p["minutes"]:.0f} min</td>'
                    f'<td class="{cls}">{txt}</td></tr>')
        st.markdown(
            '<table class="tl"><tr><th>Time</th><th>Phase</th>'
            '<th class="num">Distance</th><th class="num">Duration</th><th>Link to base</th></tr>'
            f'{trs}<tr><td>{hhmm(n["end_h"])}</td><td colspan="4">Back at base camp</td></tr></table>',
            unsafe_allow_html=True
        )

    with cmp_tab:
        rows_cmp = ""
        for tr in ("ride", "drive", "walk"):
            r_ = refs.get(tr)
            you = " ◀" if tr == travel else ""
            if isinstance(r_, str) or r_ is None:
                rows_cmp += f'<tr><td>{MODE_LABEL[tr]}{you}</td><td colspan="5" class="off">no route</td></tr>'
                continue
            nn = r_[1]["res"]["nominal"]
            rows_cmp += (f'<tr><td>{MODE_LABEL[tr]}{you}</td>'
                         f'<td class="num">{nn["duration_h"]:.2f} h</td><td class="num">{nn["o2_kg"]:.3f} kg</td>'
                         f'<td class="num">{nn["water_kg"]:.2f} kg</td><td class="num">{r_[1]["link_pct"]:.0f}%</td>'
                         f'<td class="{"on" if r_[1]["verdict"] == "GO" else "off"}">{r_[1]["verdict"]}</td></tr>')
        st.markdown(
            '<div class="sec">Same destination, same start time</div><table class="tl"><tr><th></th>'
            '<th class="num">Time</th><th class="num">Oxygen</th><th class="num">Water</th>'
            f'<th class="num">Live link</th><th>Status</th></tr>{rows_cmp}</table>',
            unsafe_allow_html=True
        )
        st.caption("Reference row: walking from the base without the rover (direct suit ↔ base radio).")

    with when_tab:
        st.markdown('<div class="sec">Cooling water (kg) by start time</div>', unsafe_allow_html=True)
        chart = sweep.set_index("start_h")[["water_kg_best", "water_kg_nominal", "water_kg_worst"]]
        chart.columns = ["Best", "Nominal", "Worst"]
        chart.index = [hhmm(h) for h in chart.index]
        st.line_chart(chart, height=200)
        if safe_window:
            st.caption(f"Daylight with sun ≥ {season_A['limits']['min_sun_elev_deg']}°: "
                       f"{hhmm(safe_window[0])}–{hhmm(safe_window[1])} · sunrise {hhmm(sunrise)}, "
                       f"sunset {hhmm(sunset)} ({southern_season(solar_longitude_deg).lower()}, "
                       f"Ls {solar_longitude_deg:.1f}°).")

    with more_tab:
        walk_legs = [l for l in plan["legs"] if l["kind"] == "walk"]
        st.write(f"Peak effort: {n['peak_btuh']:.0f} BTU/h (worst case {hi['peak_btuh']:.0f}, limit 2000)")
        st.write(f"Average effort: {n['avg_btuh']:.0f} BTU/h (limit 1600)")
        if walk_legs:
            st.write("Boxwork ground on the walk: "
                     f"{100 * np.mean([l['terrain'] == 'ridge' for l in walk_legs]):.0f}%")
        st.caption(
            "Nominal values; ranges are best–worst case. Rover, battery and radio numbers are "
            f"placeholders in eva/assumptions.json (LRV-class rover, 8 kWh, {rover_max}° limit; antennas: base "
            f"{base_h} m, rover {rover_h} m, suit {suit_h} m; ranges {ar_range} m / {rb_range} m). "
            "Live link = line of sight on the 15 m elevation grid plus max range (simplified coverage "
            "model). Distances follow the routing grid; small-scale hazards are not included."
        )

    if st.button("RETURN TO MAP", use_container_width=True):
        st.rerun()


# A persistent compact summary remains available after the dialog is closed.
if destination is not None:
    with st.container(key="route_dock"):
        if result is not None:
            status_col, distance_col, time_col, link_col, slope_col, details_col = st.columns(
                [1.25, 1, 1, 1, 1, 0.9],
                vertical_alignment="center"
            )
            status_col.markdown(
                f'<div class="route-dock-status" style="color:{status_color}">'
                f'{status_text}</div>',
                unsafe_allow_html=True
            )
            distance_col.metric("ONE-WAY", f"{route_distance:.0f} m")
            time_col.metric("ROUND TRIP", f"{n['duration_h']:.2f} h")
            link_col.metric("LIVE LINK", f"{result['link_pct']:.0f}%")
            slope_col.metric("MAX SLOPE", f"{route_max_slope:.1f}°")
            if details_col.button("DETAILS", key="open_route_assessment", use_container_width=True):
                st.session_state.show_route_dialog = True
                st.rerun()
            notes = [x for x in (plan.get("park_note") if travel == "drive" else "",
                                 relay_info.get("note") if relay_info else "") if x]
            if notes:
                st.markdown(f'<div class="route-dock-why">{" · ".join(notes)}</div>', unsafe_allow_html=True)
        elif travel == "ride" and not isinstance(refs.get("drive"), str):
            why_col, btn_col = st.columns([4, 1.3], vertical_alignment="center")
            nb = refs["drive"][1]
            why_col.markdown(
                f'<div class="route-dock-status" style="color:{status_color}">{status_text}</div>'
                f'<div class="route-dock-why">{route_status} Mode B works: the rover parks nearby and you walk '
                f'the rest ({nb["verdict"]}, {nb["res"]["nominal"]["duration_h"]:.2f} h).</div>',
                unsafe_allow_html=True
            )
            if btn_col.button("SWITCH TO MODE B", key="switch_to_b", use_container_width=True):
                st.session_state["_switch_to_b"] = True
                st.rerun()
        else:
            st.markdown(
                f'<div class="route-dock-status" style="color:{status_color}">{status_text}</div>'
                f'<div class="route-dock-why">{route_status}</div>',
                unsafe_allow_html=True
            )

# ==========================================================
# SCIENCE MISSION: how long did groundwater last? (button top centre)
# ==========================================================

@st.dialog("Science mission · How long did groundwater last on Mars?", width="large")
def show_mission():
    st.markdown(
        """
**The question.** Mars once had liquid water, but for how long did *groundwater* keep flowing under
Mount Sharp? The boxwork is one of the last places where it left a clear record.

**The place.** The boxwork region on Mount Sharp (Aeolis Mons, Gale Crater): a network of ridges about
1–2 m high with sandy hollows between them, seen from orbit as "spiderwebs". Groundwater flowed through
cracks in the rock and left minerals that cemented the crack walls; the softer rock around them later
eroded away, leaving the hard ridges standing.

**What Curiosity found (2025–2026).** Clay minerals in the ridges and carbonate minerals in the hollows,
pea-sized nodules left behind as the groundwater dried out, and a water table that was higher and stable
for longer than scientists expected.

**What our astronaut does.** One Marswalk from the habitat to three sample sites (the yellow points on the
map), then back. The samples go to Earth, where labs can **date** the minerals, which a rover cannot do on
Mars. The ages tell how long the groundwater lasted.

| Stop | Target | Why |
|---|---|---|
| 1 | Ridge, sol 4711 | minerals that cemented the cracks (when water flowed) |
| 2 | Hollow, sol 4709 | rock between the ridges (carbonates) |
| 3 | Nodules "Bococo", sol 4625 | minerals left as the water dried out (when it ended) |

**Why a human.** Hours instead of months, on-the-spot choice of the best samples, hands to collect them,
and a way to bring them home.

*Click a yellow point to plan the EVA to it. Sources: NASA/JPL "Curiosity sees Martian spiderwebs up close";
[Space.com](https://www.space.com/space-exploration/mars-rovers/curiosity-rover-finds-clues-to-mars-watery-past-in-rocky-spiderwebs);
[Astrobiology.com, Feb 2026](https://astrobiology.com/2026/02/23/mars-curiosity-rover-observes-evidence-of-past-groundwater/).*
        """
    )
    c1, c2 = st.columns(2)
    if c1.button("SHOW SAMPLE SITES ON THE MAP", use_container_width=True, key="mission_go"):
        st.rerun()
    if c2.button("HIDE MISSION", use_container_width=True, key="mission_off"):
        st.session_state.mission_on = False
        st.rerun()


with st.container(key="mission_btn"):
    on = st.session_state.get("mission_on", False)
    if st.button(("◉ " if on else "○ ") + "SCIENCE MISSION · GROUNDWATER", key="mission_toggle",
                 help="Our main mission: sample the boxwork to find out how long groundwater lasted on Mars. "
                      "Shows the sample sites (yellow) on the map."):
        st.session_state.mission_on = not on
        st.session_state.show_mission_card = not on
        st.rerun()

# only one dialog per run (Streamlit raises if two are opened)
if st.session_state.pop("show_mission_card", False):
    st.session_state.show_route_dialog = False
    show_mission()
elif plan is not None and st.session_state.get("show_route_dialog", False):
    st.session_state.show_route_dialog = False
    show_route_assessment()


# ==========================================================
# LIVE VIEW: simulated astronaut + rover + links + monitor (fragment, reruns
# every second while the simulation runs; the base map is not rebuilt)
# ==========================================================

GRID_CRS = CRS.from_wkt(data["crs"])


def rc_to_map(rc):
    r, c = float(rc[0]), float(rc[1])
    return bounds.left + (c + 0.5) * dx, bounds.top - (r + 0.5) * dy


def rc_to_lonlat(rc):
    x, y = rc_to_map(rc)
    lon, lat = warp_transform(GRID_CRS, MARS_LONLAT, [x], [y])
    return lon[0], lat[0]


def circle_m(rc, radius_m, n_pts=48):
    """Polygon (lat/lng = map y/x) for a circle of radius_m GROUND metres (x stretched by kx)."""
    x, y = rc_to_map(rc)
    a = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
    return [[y + radius_m * data["ky"] * math.sin(t), x + radius_m * data["kx"] * math.cos(t)] for t in a]


def fmt_age(s):
    s = int(round(s))
    return f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600 else (f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s")


def clock(t_s):
    h = start_h + t_s / 3600.0
    total = int(round(h * 3600)) % 86400
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def sim_layer(k):
    """Dynamic map layer for timeline sample k."""
    fg = folium.FeatureGroup(name="Simulation")
    astro, rover = TL["astro"][k], TL["rover"][k]
    ax, ay = rc_to_map(astro)
    rx, ry = rc_to_map(rover)
    bx, by = rc_to_map(base_cell)
    together = float(TERRAIN.ground_m(astro, rover)[0]) <= NAV["dock_m"]

    # links (predicted, from simulated true positions)
    if not together:
        s_ar = TL["ar"][k]
        folium.PolyLine([[ay, ax], [ry, rx]], color=LINK_COLOR[s_ar], weight=3, dash_array=LINK_DASH[s_ar],
                        tooltip=f"Astronaut ↔ rover: {s_ar} (predicted)").add_to(fg)
    s_rb = TL["rb"][k]
    if float(TERRAIN.ground_m(rover, base_cell)[0]) > NAV["dock_m"]:
        folium.PolyLine([[ry, rx], [by, bx]], color=LINK_COLOR[s_rb], weight=3, dash_array=LINK_DASH[s_rb],
                        tooltip=f"Rover ↔ base: {s_rb} (predicted)").add_to(fg)

    # relay rover links (thinner): astronaut ↔ relay, relay ↔ base, and crew rover ↔ relay when it is used
    if relay_cell is not None:
        qx, qy = rc_to_map(relay_cell)
        for (p_x, p_y), state, label in (((ax, ay), TL["aq"][k], "Astronaut ↔ relay rover"),
                                         ((bx, by), TL["qb"][k], "Relay rover ↔ base")):
            folium.PolyLine([[p_y, p_x], [qy, qx]], color=LINK_COLOR[state], weight=2, opacity=0.8,
                            dash_array=LINK_DASH[state], tooltip=f"{label}: {state} (predicted)").add_to(fg)
        if "rover → relay" in TL["path"][k]:
            folium.PolyLine([[ry, rx], [qy, qx]], color=LINK_COLOR[TL["rq"][k]], weight=2,
                            tooltip="Crew rover ↔ relay rover: connected (in use)").add_to(fg)

    # sample site
    if destination is not None:
        collected = TL["sample"][k] != "not collected"
        folium.Marker(
            [destination[1], destination[0]],
            tooltip="Sample collected" if collected else "Sample site (planned)",
            icon=folium.DivIcon(
                html=f'<div style="width:14px;height:14px;transform:rotate(45deg);border:2px solid #22c55e;'
                     f'background:{"#22c55e" if collected else "rgba(10,16,22,.6)"}"></div>',
                icon_size=(14, 14), icon_anchor=(-8, 7))
        ).add_to(fg)

    # what the base last received (frozen when the path is down)
    j = TL["last_rx"][k]
    if j >= 0 and j != k:
        ex, ey = rc_to_map(TL["est_astro"][j])
        folium.Polygon(circle_m(TL["est_astro"][j], TL["sigma_astro"][j]), color="#91a0ad", weight=1.5,
                       dash_array="4 4", fill=False,
                       tooltip=f"Base view: last received {clock(TL['t'][j])}, STALE").add_to(fg)
        folium.CircleMarker([ey, ex], radius=5, color="#91a0ad", weight=2, fill=False,
                            tooltip=f"Base view: astronaut position received {clock(TL['t'][j])} "
                                    f"(age {fmt_age(TL['t'][k] - TL['t'][j])}, stale)").add_to(fg)

    # rover (true simulated position; the rover's own estimate is not drawn)
    folium.Marker(
        [ry, rx],
        tooltip="Rover (simulated)" + (" · astronaut aboard" if together else " · parked"),
        icon=folium.DivIcon(html=rover_svg(60, glow=True), icon_size=(60, 42), icon_anchor=(30, 36))
    ).add_to(fg)

    # astronaut: estimated position + uncertainty circle, and the simulation truth as a small dot
    est = TL["est_astro"][k]
    sx, sy = rc_to_map(est)
    folium.Polygon(circle_m(est, TL["sigma_astro"][k]), color="#54d6e8", weight=2, fill=True,
                   fill_opacity=0.12, tooltip=f"Position uncertainty radius {TL['sigma_astro'][k]:.1f} m "
                                              "(illustrative, not a confidence level)").add_to(fg)
    folium.Marker(
        [sy, sx],
        tooltip="Astronaut: estimated position (local, simulated)",
        icon=folium.DivIcon(html=ASTRO_SVG.format(w=22), icon_size=(22, 28),
                            icon_anchor=(11, 46) if together else (11, 14))   # body centred on the circle
    ).add_to(fg)
    folium.CircleMarker([ay, ax], radius=3, color="#ffffff", weight=1, fill=True, fill_color="#ffffff",
                        fill_opacity=1, tooltip="Astronaut TRUE position (simulation only; "
                                                "nobody on Mars knows this)").add_to(fg)
    return fg


def link_badge(state):
    cls = {sim.OK: "on", sim.DOWN: "off", sim.UNKNOWN: "unk"}[state]
    return f'<span class="{cls}">{LINK_ICON[state]}</span>'


def monitor(k):
    """Compact monitoring panel for timeline sample k."""
    t_now = TL["t"][k]
    leg = sim_legs[TL["leg"][k]]
    astro_status, rover_status = sim.phase(sim_legs, TL["leg"][k], t_now, TL["T"])
    on_foot = t_now < TL["T"] and (leg["kind"] == "walk" or (leg["kind"] == "stop" and not leg.get("together")))
    now_mode = "B · on foot, rover parked" if on_foot else "A · together with the rover"
    d_ar = float(TERRAIN.ground_m(TL["astro"][k], TL["rover"][k])[0])
    lon, lat = rc_to_lonlat(TL["est_astro"][k])
    sig = TL["sigma_astro"][k]
    fix_t = TL["last_fix_t"][k]
    src = next((TL["fix_source"][i] for i in range(k, -1, -1) if TL["fix_source"][i]), "base survey point")
    j = TL["last_rx"][k]
    live = bool(TL["full"][k])
    cons = A["consumables"]

    def readings(i):
        return (f"O₂ {100 * (1 - TL['o2_used_kg'][i] / cons['o2_capacity_kg']):.0f}% · "
                f"water {100 * (1 - TL['water_used_kg'][i] / cons['water_capacity_kg']):.0f}% · "
                f"suit batt {TL['suit_battery_pct'][i]:.0f}%")

    if j >= 0:
        blon, blat = rc_to_lonlat(TL["est_astro"][j])
        age = t_now - TL["t"][j]
        base_line = (f"{clock(TL['t'][j])} · age {fmt_age(age)} "
                     + ('<span class="on">LIVE</span>' if live else '<span class="off">STALE</span>'))
        base_view = f"{blat:.5f}°, {blon:.5f}°E ± {TL['sigma_astro'][j]:.1f} m<br>{readings(j)}"
    else:
        base_line, base_view = '<span class="off">no data yet</span>', "–"

    rows_ = [
        ("Mode (plan)", MODE_LABEL[travel]),
        ("Now", now_mode),
        ("Astronaut", astro_status),
        ("Rover", rover_status + (" · carries tool kit" + (" + sample" if TL["sample"][k] == "in rover" else ""))),
        ("Sample", TL["sample"][k]),
        ("Astro ↔ rover", f"{d_ar:.0f} m"),
        ("Link astro ↔ rover", link_badge(TL["ar"][k])),
        ("Link rover ↔ base", link_badge(TL["rb"][k])),
    ] + ([
        ("Link astro ↔ relay", link_badge(TL["aq"][k])),
        ("Link relay ↔ base", link_badge(TL["qb"][k])),
    ] if relay_cell is not None else []) + [
        ("Base receives", f'<span class="on">live</span> via {TL["path"][k].replace("astronaut → ", "")}'
                          if live else '<span class="off">no (path down)</span>'),
        ("Estimate (local)", f"{lat:.5f}°, {lon:.5f}°E<br>at {clock(t_now)}"),
        ("Uncertainty", f"{sig:.1f} m radius <i>(illustrative)</i><br>last fix {clock(fix_t)} "
                        f"({src}), {fmt_age(t_now - fix_t)} ago"),
        ("Base last update", base_line),
        ("Base has", base_view),
        ("EVA model now", readings(k) + " <i>(planned, nominal)</i>"),
    ]
    body = "".join(f"<tr><th>{a}</th><td>{b}</td></tr>" for a, b in rows_)
    return (f'<div class="mon-clock">SIM {clock(t_now)} LMST · T+{fmt_age(t_now)} / {fmt_age(TL["T"])}'
            f'{" · RETURNED EARLY" if sim_aborted else ""}</div>'
            f'<table class="mon">{body}</table>')


ASSUMPTIONS_TEXT = f"""
**Everything in this panel is simulated** (positions, link states, uncertainty). Real data in the app:
HiRISE image, 1 m DEM, Curiosity waypoints.

**Movement.** The astronaut and rover follow the planned routes with the EVA model's timing
(walking speed vs slope, rover speed, work time). Mode A: together all the time. Mode B: the rover
drives to its parking spot and stays there while the astronaut walks; "Return to rover" sends the
astronaut back along the path walked so far, then both drive home.

**Position uncertainty (Fehlerkreis).** Radius in metres, *illustrative*, not a statistical confidence
level. It grows linearly without a valid correction: rover {100 * NAV['rover_drift_frac']:.0f}% of
distance driven; astronaut on foot {100 * NAV['walk_drift_frac']:.0f}% of distance walked +
{NAV['time_drift_m_per_min']} m per minute. Valid corrections: being at the surveyed base point
(radius {NAV['base_fix_m']} m) or docking with the rover (within {NAV['dock_m']:.0f} m: the astronaut takes
the rover's estimate + {NAV['dock_add_m']} m). A radio reconnection does **not** correct the position.
The estimated point is offset from the true point by a slowly turning error of 0.6 × the radius.
Circles are drawn in ground metres (map x stretched by 1/cos(latitude) for the equirectangular grid).

**Predicted links.** Every link (astronaut ↔ crew rover, crew rover ↔ base, and with the relay rover:
astronaut ↔ relay, crew rover ↔ relay, relay ↔ base) is checked separately: straight line between
the antennas over the elevation grid (Mars curvature) plus maximum range. Outside the elevation data =
unknown. Simplified coverage model, not a radio propagation calculation. The suit radio talks only to a
rover, never straight to the base. The base gets live updates only when a complete path is up
(astronaut → crew rover → base, → crew rover → relay → base, or → relay → base); otherwise it keeps
the last received position and readings and shows their age.

**Relay rover.** An uncrewed second rover drives to its spot hours before the EVA (same slope and boxwork
limits as the crew rover) and stays parked all mission. Auto spot (for the chosen destination): drivable,
sees the base, sees as many points of the planned route (walk in mode B, drive in mode A, and the
destination) as possible, closest to them; without a destination it uses the science stops. A spot you
set with "Set relay rover" always wins. It does not correct the astronaut's position.

**Not simulated:** cameras and motion sensors (concept hardware), beacons, satellites.
EVA numbers (O₂, water, battery) come from the existing EVA model, not from sensors.
"""


def live_view():
    finished = advance_clock()
    k = sim.index_at(TL, S["t"]) if TL is not None else None

    # Render a fresh copy each time: rendering the same folium map twice (as a fragment
    # rerun would) breaks its script. The copy keeps the element ids, so the base map
    # is not reloaded; only the simulation layer changes.
    map_data = st_folium(
        copy.deepcopy(m),
        height=900,
        use_container_width=True,
        returned_objects=["last_clicked", "last_object_clicked", "last_object_clicked_tooltip"],
        feature_group_to_add=sim_layer(k) if TL is not None else None,
        key="mars_map"
    )

    with st.container(key="monitor"):
        with st.expander("MISSION MONITOR · SIMULATION", expanded=True):
            if TL is None:
                st.caption("Select a destination with a valid route to run the simulation.")
            else:
                st.markdown(monitor(k), unsafe_allow_html=True)
                b1, b2, b3 = st.columns(3)
                t_done = S["t"] >= TL["T"]
                if S["running"]:
                    if b1.button("⏸ Pause", key="sim_pause", use_container_width=True):
                        S["running"] = False
                        st.rerun()
                else:
                    label = "▶ Start" if S["t"] == 0 or t_done else "▶ Resume"
                    if b1.button(label, key="sim_start", use_container_width=True):
                        if t_done:
                            S["t"], S["abort"] = 0.0, None
                        S["running"], S["wall"] = True, time.monotonic()
                        st.rerun()
                if b2.button("↺ Reset", key="sim_reset", use_container_width=True):
                    S.update(t=0.0, running=False, abort=None, wall=time.monotonic())
                    st.rerun()
                speed = b3.selectbox("Speed", SPEEDS, index=SPEEDS.index(S["speed"]),
                                     format_func=lambda v: f"{v}×", key="sim_speed",
                                     label_visibility="collapsed")
                if speed != S["speed"]:
                    S["speed"] = speed
                can_return = travel == "drive" and S["abort"] is None and sim.abort_to_rover(sim_legs, S["t"])[1]
                if can_return and st.button("↩ Return to rover now (switch to mode A)", key="sim_abort",
                                            use_container_width=True):
                    S["abort"] = S["t"]
                    st.rerun()
                with st.expander("Model & assumptions"):
                    st.markdown(ASSUMPTIONS_TEXT)

    # ---- map clicks
    def as_point(click):
        return (click["lng"], click["lat"]) if click else None

    raw_point = as_point(map_data.get("last_clicked")) if map_data else None
    obj_point = as_point(map_data.get("last_object_clicked")) if map_data else None
    point = None

    if obj_point is not None and obj_point != st.session_state.get("last_obj"):
        # a click on a science stop / route. Remember the map click that
        # came with it so it can't override this point on the next rerun.
        st.session_state.last_obj = obj_point
        st.session_state.last_click = raw_point
        tip = (map_data.get("last_object_clicked_tooltip") or "") if map_data else ""
        if not tip.startswith("📷"):                     # a click on a photo opens its popup, nothing else
            point = obj_point
            # snap to the stop itself when the click landed on its marker (within 40 m)
            near = min(data["stops"], key=lambda s: math.dist((s[1], s[2]), obj_point))
            if math.dist((near[1], near[2]), obj_point) < 40:
                point = (near[1], near[2])
    elif raw_point is not None and raw_point != st.session_state.last_click:
        st.session_state.last_click = raw_point
        point = raw_point

    if point is not None:
        x, y = point
        if bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top:
            if mode == "Set base camp":
                st.session_state.base = point
                st.session_state.destination = None
            elif mode == "Set rover parking":
                st.session_state.park = point
                if travel != "drive":
                    st.session_state["_switch_to_b"] = True      # parking only exists in mode B
            elif mode == "Set relay rover":
                st.session_state.relay_point = point
            else:
                st.session_state.destination = point
                st.session_state.show_route_dialog = False
            st.rerun()

    if finished:
        st.rerun()


st.fragment(run_every=1.0 if S["running"] else None)(live_view)()
