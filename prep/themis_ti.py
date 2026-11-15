"""THEMIS night-time thermal inertia (100 m/pixel) for the app's map window: soft sand vs. solid rock.

Thermal inertia = how well the ground holds heat. Loose sand cools fast at night (low inertia), solid rock
stays warm (high inertia). Source: USGS Astrogeology, THEMIS Global Thermal Inertia Mosaic, quantitative
32-bit, tile 30S120E (Fergason et al.; ~20 % absolute accuracy). Same Mars equirectangular projection as
our DEM, so only the window is streamed (a few MB).

Run:  python prep/themis_ti.py
Out:  share/team_app/data/themis_ti.tif   (J m^-2 K^-1 s^-1/2, NaN = no data)
"""
import os

import numpy as np
import rasterio
from rasterio.windows import from_bounds

ROOT = os.path.join(os.path.dirname(__file__), "..")
APP_DATA = os.path.join(ROOT, "share", "team_app", "data")
TI_URL = ("/vsicurl/https://asc-astropedia.s3.us-west-2.amazonaws.com/Mars/Odyssey/"
          "THEMIS-Global-Thermal-Inertia-Mosaic/Quantitative-32-Bit/THEMIS_TI_Mosaic_Quant_30S120E_100mpp.cub")

with rasterio.open(os.path.join(APP_DATA, "dem_wide.tif")) as d:
    b = d.bounds
with rasterio.Env(GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2"):
    with rasterio.open(TI_URL) as s:
        win = from_bounds(b.left - 200, b.bottom - 200, b.right + 200, b.top + 200, s.transform)
        win = win.round_offsets(op="floor").round_lengths(op="ceil")
        ti = s.read(1, window=win).astype("float32")
        ti[(ti < 0) | (ti > 1e5)] = np.nan
        tr, crs = s.window_transform(win), s.crs
out = os.path.join(APP_DATA, "themis_ti.tif")
with rasterio.open(out, "w", driver="GTiff", height=ti.shape[0], width=ti.shape[1], count=1, dtype="float32",
                   crs=crs, transform=tr, nodata=np.nan) as dst:
    dst.write(ti, 1)
v = ti[np.isfinite(ti)]
print(f"wrote {out}: {ti.shape[1]} x {ti.shape[0]} px (100 m), thermal inertia {np.percentile(v, 2):.0f}"
      f"-{np.percentile(v, 98):.0f} (median {np.median(v):.0f}), no-data {100 * np.isnan(ti).mean():.1f}%")
