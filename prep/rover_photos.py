"""Curiosity photos on the map: where each Navcam frame was taken and which ground it shows.

For every raw image (NASA raw-image API) we take:
  * the rover position at that moment: waypoint with the same site/drive (MMGIS waypoint file),
  * the camera model (CAHVOR, in the rover frame, in the delivered image's pixel grid) and the rover
    attitude quaternion (rover -> local north/east/down).
For a fan of image columns, the top and bottom edge rays give the viewing direction and the vertical
field of view. Walking out along each direction over the 1 m DEM, the ground that is both inside the
vertical field of view and in line of sight forms the view footprint ("range of view") on the map.

Run:  python prep/rover_photos.py
Out:  share/team_app/data/rover_photos.json, share/team_app/static/photos/<id>.jpg (thumbnails)
Credit: NASA/JPL-Caltech.
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np
import rasterio
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "share" / "team_app"
API = "https://mars.nasa.gov/api/v1/raw_image_items/"
UA = {"User-Agent": "Mozilla/5.0 (ARES-X Space Apps project)"}
MARS_R = 3396190.0
M_PER_DEG = MARS_R * math.pi / 180

# raw image ids picked by the team (sol 4837) + the frames already used in out/photos (sols 4625, 4711)
IDS = [1565099, 1565108, 1565250, 1565256, 1565350, 1565367, 1564408, 1564417, 1564419, 1564422, 1564426,
       1564430, 1564434, 1564437, 1564440, 1564445, 1564448, 1564452, 1564460, 1565096, 1565097, 1565098,
       1565037, 1564370, 1564373, 1564496, 1565040,
       1529130, 1529138, 1529139, 1499001, 1499114, 1499003,
       # second batch picked by the team (along the drive)
       1438771, 1438772, 1457274, 1457279, 1463402, 1463406, 1469023, 1469019, 1474383, 1474384, 1478617,
       1478653, 1482879, 1482877, 1487735, 1491632, 1491625, 1497247, 1497244, 1501269, 1501267, 1506903,
       1506902, 1514266, 1514262, 1514259, 1549645, 1549641, 1557447, 1557444, 1561090, 1561087, 1569455,
       1569101, 1574458, 1574454, 1592246, 1592243, 1597645, 1597643, 1601024, 1601023, 1606230, 1606226,
       1610692, 1610688, 1613888, 1613892]
ANNOTATED = {p.name.split("_", 2)[2].replace(".jpg", ""): p for p in (ROOT / "out" / "photos").glob("sol*_az*_*.jpg")}


def vec(s):
    return np.array([float(v) for v in s.strip("()").split(",")])


def quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def fetch(i):
    d = requests.get(API, params={"condition_1": f"{i}:id:eq", "per_page": 1}, headers=UA, timeout=60).json()
    return d["items"][0]


class Dem:
    def __init__(self, path):
        self.ds = rasterio.open(path)
        self.z = self.ds.read(1)

    def at_lonlat(self, lon, lat):
        x, y = MARS_R * np.radians(lon), MARS_R * np.radians(lat)        # eqc, lat_ts = 0
        r, c = rasterio.transform.rowcol(self.ds.transform, x, y)
        r, c = np.asarray(r), np.asarray(c)
        ok = (r >= 0) & (r < self.z.shape[0]) & (c >= 0) & (c < self.z.shape[1])
        out = np.full(r.shape, np.nan)
        out[ok] = self.z[r[ok], c[ok]]
        return out


def footprint(it, wp, dem: Dem, w, h):
    comps = [vec(c) for c in it["camera_model_component_list"].split(";")]
    C, A, H, V = comps[:4]
    R = quat_to_R(vec(it["attitude"]))                     # rover -> site (north, east, down)

    def ray(x, y):
        d = np.cross(V - y * A, H - x * A)
        d = d / np.linalg.norm(d)
        d = d if d @ A > 0 else -d
        return R @ d                                        # site frame

    lat0, lon0, z0 = wp["lat"], wp["lon"], wp["elev_geoid"]
    coslat = math.cos(math.radians(lat0))
    cam = R @ C                                             # camera offset from the rover origin (NED)
    cam_n, cam_e, cam_z = cam[0], cam[1], z0 - cam[2]       # rover origin ~ on the ground
    centre = ray(w / 2, h / 2)
    az_c = math.degrees(math.atan2(centre[1], centre[0])) % 360
    el_c = math.degrees(math.asin(-centre[2]))
    left, right = ray(0, h / 2), ray(w - 1, h / 2)
    hfov = math.degrees(math.acos(np.clip(left @ right, -1, 1)))
    vfov = math.degrees(math.acos(np.clip(ray(w / 2, 0) @ ray(w / 2, h - 1), -1, 1)))

    d = np.concatenate([np.arange(2, 50, 1.0), np.arange(50, 400, 3.0), np.arange(400, 3000, 8.0)])
    near, far = [], []
    for x in np.linspace(0, w - 1, 21):
        top, bot = ray(x, 0), ray(x, h - 1)
        mid = ray(x, h / 2)
        az = math.atan2(mid[1], mid[0])
        el_top, el_bot = math.asin(-top[2]), math.asin(-bot[2])
        n, e = cam_n + d * math.cos(az), cam_e + d * math.sin(az)
        lat, lon = lat0 + n / M_PER_DEG, lon0 + e / (M_PER_DEG * coslat)
        hgt = dem.at_lonlat(lon, lat)
        ang = np.arctan2(hgt - cam_z, d)
        valid = np.isfinite(ang)
        if not valid.any():
            continue
        ang_f = np.where(valid, ang, -np.inf)
        seen = ang_f >= np.maximum.accumulate(np.concatenate([[-np.inf], ang_f[:-1]])) - 1e-6
        ok = valid & seen & (ang_f >= el_bot) & (ang_f <= el_top)
        if not ok.any():
            continue
        i0, i1 = np.argmax(ok), len(ok) - 1 - np.argmax(ok[::-1])
        near.append((lon[i0], lat[i0]))
        far.append((lon[i1], lat[i1]))
    poly = near + far[::-1]
    far_m = [math.hypot((p[1] - lat0) * M_PER_DEG, (p[0] - lon0) * M_PER_DEG * coslat) for p in far]
    return {"az": round(az_c, 1), "el": round(el_c, 1), "hfov": round(hfov, 1), "vfov": round(vfov, 1),
            "footprint": [[round(a, 7), round(b, 7)] for a, b in poly],
            "far_m": round(float(np.median(far_m)), 0) if far_m else None}


def main():
    wps = json.loads((ROOT / "out" / "waypoints" / "MSL_waypoints.json").read_text())["features"]
    by_rmc = {(f["properties"]["site"], f["properties"]["drive"]): f["properties"] for f in wps}
    dem = Dem(APP / "data" / "dem_wide.tif")
    thumbs = APP / "static" / "photos"
    thumbs.mkdir(parents=True, exist_ok=True)
    out = []
    old = APP / "data" / "rover_photos.json"
    done = ({p["id"]: p for p in json.loads(old.read_text())["photos"]} if old.exists() else {})
    for i in IDS:
        if i in done:                                      # keep photos already processed exactly as they are
            out.append(done[i])
            continue
        it = fetch(i)
        wp = by_rmc.get((it["site"], it["drive"]))
        if wp is None:                                     # latest waypoint of the same site, earlier drive
            same = [v for (s, dr), v in by_rmc.items() if s == it["site"] and dr <= it["drive"]]
            wp = max(same, key=lambda v: v["drive"]) if same else None
        if wp is None or not it.get("camera_model_component_list"):
            print(f"skip {i}: no position or no camera model")
            continue
        img = Image.open(io.BytesIO(requests.get(it["https_url"], headers=UA, timeout=120).content)).convert("RGB")
        w, h = img.size
        fp = footprint(it, wp, dem, w, h)
        key = it["imageid"][:13]
        src = Image.open(ANNOTATED[key]).convert("RGB") if key in ANNOTATED else img
        src.thumbnail((640, 640))
        src.save(thumbs / f"{i}.jpg", quality=78)
        out.append({"id": i, "imageid": it["imageid"], "sol": it["sol"], "site": it["site"], "drive": it["drive"],
                    "camera": it["instrument"], "lmst": (it["extended"].get("lmst") or "").replace("Sol-", "sol "),
                    "rover_lat": wp["lat"], "rover_lon": wp["lon"], "rover_yaw": wp["yaw"],
                    "image_px": [w, h], "route_drawn": key in ANNOTATED,
                    "thumb": f"app/static/photos/{i}.jpg",
                    "page": f"https://mars.nasa.gov/raw_images/{i}/?site=msl", "image_url": it["https_url"], **fp})
        print(f"{i}: sol {it['sol']} {it['instrument']} az {fp['az']:5.1f} el {fp['el']:5.1f} "
              f"fov {fp['hfov']:.0f}x{fp['vfov']:.0f} far {fp['far_m']} m, {len(fp['footprint'])} pts")
    (APP / "data" / "rover_photos.json").write_text(json.dumps(
        {"credit": "NASA/JPL-Caltech (Curiosity Navcam raw images)", "photos": out}, indent=1))
    print(f"{len(out)} photos written")


if __name__ == "__main__":
    main()
