"""
ARES-X - wider 16:9 data window for the team app (6.4 x 3.6 km ground)
---------------------------------------------------------------------
Streams only the needed window from the same sources as cilli_prep.py:
  * USGS MSL Gale Merged DEM 1 m v3          -> share/team_app/data/dem_wide.tif
  * HiRISE LRGB colour mosaic (COG), ~1 m    -> share/team_app/data/hirise_wide.tif

Box centred between habitat B and the science stops, so the base, the stops, the relay spots
and Curiosity's drive all fit, with a 16:9 shape that fills a wide screen.

    python prep/make_wide_crop.py

Coordinates: planetocentric latitude, east-positive longitude, Mars sphere R = 3,396,190 m.
"""
import math
import os
import sys

import numpy as np
import rasterio
import rasterio.transform
import rasterio.warp
import rasterio.windows
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

OUT = os.path.join(os.path.dirname(__file__), "..", "share", "team_app", "data")
DEM_URL = os.environ.get("DEM_URL") or "/vsicurl/https://planetarymaps.usgs.gov/mosaic/Mars/MSL/MSL_Gale_DEM_Mosaic_1m_v3.tif"
HIRISE_URL = os.environ.get("HIRISE_URL") or ("/vsicurl/https://asc-pds-services.s3.us-west-2.amazonaws.com/"
                                              "mosaic/Mars/MSL/MSL_Gale_HiRISE-LRGB_78quads_sharp_cog.tif")

HABITAT_B = (-4.811979, 137.370908)                      # lat, lon
STOPS = [(-4.805862088, 137.379759371), (-4.807999778, 137.381049648)]
WIDTH_M, HEIGHT_M = 6400.0, 3600.0                       # 16:9, ground metres

MARS_R = 3396190.0
M_PER_DEG = MARS_R * math.pi / 180.0
c_lat = (HABITAT_B[0] + sum(s[0] for s in STOPS) / len(STOPS)) / 2
c_lon = (HABITAT_B[1] + sum(s[1] for s in STOPS) / len(STOPS)) / 2
d_lat = HEIGHT_M / 2 / M_PER_DEG
d_lon = WIDTH_M / 2 / (M_PER_DEG * math.cos(math.radians(c_lat)))
LAT_MIN, LAT_MAX = c_lat - d_lat, c_lat + d_lat
LON_MIN, LON_MAX = c_lon - d_lon, c_lon + d_lon
print(f"Box: lat {LAT_MIN:.5f} .. {LAT_MAX:.5f}   lon {LON_MIN:.5f} .. {LON_MAX:.5f}", flush=True)


def window_for_box(src):
    pcrs = CRS.from_wkt(src.crs.to_wkt())
    fwd = Transformer.from_crs(pcrs.geodetic_crs, pcrs, always_xy=True)
    xs, ys = fwd.transform([LON_MIN, LON_MAX, LON_MIN, LON_MAX], [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX])
    return from_bounds(min(xs), min(ys), max(xs), max(ys), src.transform).round_offsets().round_lengths()


def save_tif(path, arr, transform, crs, nodata=None):
    arr = arr if arr.ndim == 3 else arr[np.newaxis]
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype=arr.dtype, crs=crs, transform=transform, nodata=nodata, compress="deflate",
                       predictor=3 if arr.dtype.kind == "f" else 2, tiled=True) as dst:
        dst.write(arr)
    print(f"  wrote {path}: {arr.shape[2]} x {arr.shape[1]} px, {os.path.getsize(path) / 1e6:.0f} MB", flush=True)


# Image-to-DEM co-registration, measured by phase correlation of DEM hillshade vs. image in all four
# quarters (same offset everywhere): the image content sits 6 m west and 2 m south of the DEM.
SHIFT_E, SHIFT_N = 6.0, 2.0


def colorize(img):
    """Colour the greyscale strips of the HiRISE LRGB mosaic (approximate, for display only).

    Grey = R, G and B nearly equal over a ~40 m neighbourhood. For every brightness level the mean colour
    of the truly coloured pixels of this image is learned and applied to the grey pixels, so brightness and
    texture stay real and only the tint is borrowed. Soft blend at the strip edges.
    """
    from scipy.ndimage import uniform_filter
    f = img[:3].astype(np.float32)
    lum = f.mean(0)
    chroma = uniform_filter(f.max(0) - f.min(0), 41)
    w = np.clip((8.0 - chroma) / 4.0, 0.0, 1.0)            # 1 = clearly grey (<4), 0 = coloured (>8)
    w[lum < 3] = 0.0                                        # keep no-data black
    colour = (chroma > 10) & (lum > 3)
    if colour.sum() < 1000 or w.max() == 0:
        return img
    b = np.clip(lum, 0, 255).astype(int)
    n = np.bincount(b[colour], minlength=256).astype(float)
    lut = np.zeros((3, 256))
    for c in range(3):
        s = np.bincount(b[colour], weights=f[c][colour], minlength=256)
        good = n > 20
        lut[c] = np.interp(np.arange(256), np.flatnonzero(good), s[good] / n[good])
    tinted = lut[:, b]
    out = f * (1 - w) + tinted * w
    print(f"  colourised {100 * (w > 0.5).mean():.1f}% of the image (grey strips)", flush=True)
    return remove_seams(np.clip(out, 0, 255).astype(np.uint8), w > 0)


def remove_seams(img, grey, open_px=21):
    """Fade the thin bluish seam lines left along the edges of the grey strips.

    Seam = bluish pixels (B close to or above R; Mars terrain is red) in a thin structure (removed by a
    21 px morphological opening, so wide blue areas such as dark dunes stay) and within 60 m of a former
    grey strip. Seam pixels are filled from their non-seam neighbourhood (normalised Gaussian blur).
    """
    from scipy.ndimage import binary_dilation, binary_erosion, binary_opening, gaussian_filter
    f = img.astype(np.float32)
    bluish = (f[2] - f[0]) > -12
    seam = bluish & ~binary_opening(bluish, np.ones((open_px, open_px)))
    edge = binary_dilation(grey, iterations=25) & ~binary_erosion(grey, iterations=25)   # band along strip edges
    seam &= edge
    seam = binary_dilation(seam, np.ones((5, 5)))
    keep = (~seam).astype(np.float32)
    w = gaussian_filter(keep, 6)
    fill = np.stack([gaussian_filter(c * keep, 6) for c in f]) / np.maximum(w, 1e-3)
    w2 = gaussian_filter(keep, 25)                          # wider fill where the seam is wide
    fill2 = np.stack([gaussian_filter(c * keep, 25) for c in f]) / np.maximum(w2, 1e-3)
    fill = np.where((w < 0.15)[None], fill2, fill)
    print(f"  faded seam lines: {100 * seam.mean():.2f}% of the pixels", flush=True)
    return np.clip(np.where(seam[None], fill, f), 0, 255).astype(np.uint8)


def shift_onto(img, tr, crs, shape):
    """Move the image by the measured co-registration offset onto the DEM grid."""
    out = np.zeros((img.shape[0],) + shape, dtype=img.dtype)
    rasterio.warp.reproject(img, out, src_transform=Affine.translation(SHIFT_E, SHIFT_N) * tr, src_crs=crs,
                            dst_transform=tr, dst_crs=crs, resampling=Resampling.bilinear)
    return out


if "--fix-image" in sys.argv:
    # post-process an uncorrected (raw) cut of the image, path given after the flag (no download)
    p = os.path.join(OUT, "hirise_wide.tif")
    with rasterio.open(sys.argv[sys.argv.index("--fix-image") + 1]) as s:
        img, tr, crs = s.read(), s.transform, s.crs
    save_tif(p, colorize(shift_onto(img, tr, crs, img.shape[1:])), tr, crs)
    sys.exit(0)

os.makedirs(OUT, exist_ok=True)
with rasterio.Env(GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2", CPL_VSIL_CURL_CHUNK_SIZE="1048576"):
    # The HiRISE mosaic covers less than the DEM: clip the box to it (a window partly outside the
    # image was silently clipped by the old read and shifted the whole picture ~160 m).
    with rasterio.open(HIRISE_URL) as hs:
        hb = hs.bounds
    with rasterio.open(DEM_URL) as src:
        win = window_for_box(src)
        l, b, r, t = rasterio.windows.bounds(win, src.transform)
        l, b, r, t = max(l, hb.left), max(b, hb.bottom), min(r, hb.right), min(t, hb.top)
        win = from_bounds(l, b, r, t, src.transform).round_offsets(op="ceil").round_lengths(op="floor")
        print(f"DEM: streaming {win.width} x {win.height} m (clipped to the HiRISE coverage) ...", flush=True)
        z = src.read(1, window=win, masked=True).astype("float32")
        z = np.ma.masked_where(z < -1e20, z)
        print(f"  DEM missing pixels: {100 * np.ma.getmaskarray(z).mean():.2f}%", flush=True)
        grid_tr, grid_crs, grid_shape = src.window_transform(win), src.crs, z.shape
        save_tif(os.path.join(OUT, "dem_wide.tif"), z.filled(np.nan), grid_tr, grid_crs, nodata=np.nan)

    print("HiRISE: reading the 1 m level and warping it onto the DEM grid ...", flush=True)
    with rasterio.open(HIRISE_URL, overview_level=1) as src:             # 0.25 m native -> 1 m level
        gl, gb, gr, gt = rasterio.transform.array_bounds(*grid_shape, grid_tr)
        hw = from_bounds(gl - 4, gb - 4, gr + 4, gt + 4, src.transform).round_offsets().round_lengths()
        img = src.read(window=hw, boundless=True, fill_value=0)
        out = np.zeros((src.count,) + grid_shape, dtype=img.dtype)
        rasterio.warp.reproject(img, out, src_transform=Affine.translation(SHIFT_E, SHIFT_N) * src.window_transform(hw),
                                src_crs=src.crs, dst_transform=grid_tr, dst_crs=grid_crs,
                                resampling=Resampling.bilinear)
        save_tif(os.path.join(OUT, "hirise_wide.tif"), colorize(out), grid_tr, grid_crs)
print("done", flush=True)
