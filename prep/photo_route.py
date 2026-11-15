"""Draw the EVA route into real Curiosity Navcam photos.

Each raw Navcam frame from NASA's raw-image API carries its CAHVOR camera model
(in the rover frame) and the rover attitude quaternion (rover -> site frame,
site = north/east/down). With the rover's lat/lon from the waypoint file and
the 1 m DEM we can project any point on the map into the photo:

    map point (lon, lat, elev) -> local north/east/down from the rover
    -> rover frame (inverse attitude) -> CAHVOR projection -> pixel (x, y)

Checks drawn on every photo:
  * DEM skyline: the horizon computed from the 1 m DEM, projected into the photo.
    If it sits on the real skyline, the geometry is right.
  * Route points hidden behind terrain (DEM line of sight) are drawn dashed.

Run:  python prep/photo_route.py            (after cilli_prep.py and route_dem.py)
Out:  out/photos/*.jpg (annotated), out/photos/photos.json (for the web page)
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np
import rasterio
import requests
from PIL import Image, ImageDraw
from pyproj import CRS, Transformer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "photos"
API = "https://mars.nasa.gov/api/v1/raw_image_items/"
UA = {"User-Agent": "Mozilla/5.0 (ARES-X Space Apps project)"}
MARS_R = 3396190.0
M_PER_DEG = MARS_R * math.pi / 180


def vec(s: str) -> np.ndarray:
    return np.array([float(v) for v in s.strip("()").split(",")])


def quat_to_R(q):
    """Rotation matrix for quaternion (s, x, y, z): rover frame -> site frame."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class Camera:
    def __init__(self, item: dict):
        comps = [vec(c) for c in item["camera_model_component_list"].split(";")]
        self.C, self.A, self.H, self.V = comps[:4]
        self.O = comps[4] if len(comps) > 4 else self.A
        self.Rr = comps[5] if len(comps) > 5 else np.zeros(3)
        self.R = quat_to_R(vec(item["attitude"]))        # rover -> site
        x0, y0, x1, y1 = [int(v) for v in vec(item["subframe_rect"])]
        self.w, self.h = x1, y1

    def site_to_pixel(self, p_site: np.ndarray):
        """p_site: (n, 3) north/east/down metres from the rover origin -> (n, 2) pixels, in_front mask."""
        p = p_site @ self.R - self.C                        # R^T p_site - C  (rover frame, from camera)
        zeta = p @ self.O
        lam = p - zeta[:, None] * self.O
        tau = np.einsum("ij,ij->i", lam, lam) / np.maximum(zeta, 1e-9) ** 2
        mu = self.Rr[0] + self.Rr[1] * tau + self.Rr[2] * tau ** 2
        pp = p + mu[:, None] * lam
        a = pp @ self.A
        xy = np.stack([(pp @ self.H) / a, (pp @ self.V) / a], 1)
        return xy, a > 0.1

    def cam_site(self):
        return self.R @ self.C                              # camera position, site frame

    def look_az_el(self):
        d = self.R @ self.A
        return math.degrees(math.atan2(d[1], d[0])) % 360, math.degrees(math.asin(-d[2]))


class Terrain:
    def __init__(self, lon0, lat0, z0):
        ds = rasterio.open(ROOT / "out" / "dem_crop.tif")
        self.ds, self.z = ds, ds.read(1)
        pc = CRS.from_wkt(ds.crs.to_wkt())
        self.fwd = Transformer.from_crs(pc.geodetic_crs, pc, always_xy=True)
        self.lon0, self.lat0, self.z0 = lon0, lat0, z0
        self.coslat = math.cos(math.radians(lat0))

    def ned(self, lon, lat, z):
        lon, lat, z = map(np.asarray, (lon, lat, z))
        return np.stack([(lat - self.lat0) * M_PER_DEG,
                         (lon - self.lon0) * M_PER_DEG * self.coslat,
                         -(z - self.z0)], -1)

    def height_at_ne(self, n, e):
        lat = self.lat0 + n / M_PER_DEG
        lon = self.lon0 + e / (M_PER_DEG * self.coslat)
        x, y = self.fwd.transform(lon, lat)
        rows, cols = rasterio.transform.rowcol(self.ds.transform, x, y)
        rows, cols = np.asarray(rows), np.asarray(cols)
        ok = (rows >= 0) & (rows < self.z.shape[0]) & (cols >= 0) & (cols < self.z.shape[1])
        out = np.full(rows.shape, np.nan)
        out[ok] = self.z[rows[ok], cols[ok]]
        return out

    def skyline(self, cam, az_list, max_d=2200):
        """Max terrain elevation angle seen from the camera for each azimuth -> site vectors."""
        c = cam                                            # camera position NED
        d = np.concatenate([np.arange(3, 60, 1.0), np.arange(60, max_d, 4.0)])
        pts = []
        for az in az_list:
            a = math.radians(az)
            n, e = c[0] + d * math.cos(a), c[1] + d * math.sin(a)
            h = self.height_at_ne(n, e)
            up = (h - self.z0) - (-c[2])                   # terrain height above the camera
            ang = np.arctan2(up, d)
            if np.all(np.isnan(ang)):
                pts.append(None)
                continue
            i = int(np.nanargmax(ang))
            pts.append(np.array([n[i] - 0, e[i] - 0, -(h[i] - self.z0)]))
        return pts

    def visible(self, cam, p):
        """Line of sight from the camera (NED) to point p (NED) over the DEM."""
        v = p - cam
        dist = math.hypot(v[0], v[1])
        if dist < 2:
            return True
        t = np.linspace(0.02, 0.97, max(10, int(dist / 1.5)))
        q = cam[None, :] + t[:, None] * v[None, :]
        h = self.height_at_ne(q[:, 0], q[:, 1]) - self.z0     # terrain height above rover ground
        ray = -q[:, 2]                                        # ray height above rover ground
        return not np.any(h > ray + 0.3)


def fetch_items(sol_lo, sol_hi, instruments="NAV_LEFT_B"):
    items, page = [], 0
    while True:
        p = {"order": "sol asc", "per_page": 100, "page": page, "condition_1": "msl:mission",
             "condition_2": f"{sol_lo}:sol:gte", "condition_3": f"{sol_hi}:sol:lte",
             "condition_4": f"{instruments}:instrument:in", "extended": "thumbnail::sample_type::noteq"}
        d = requests.get(API, params=p, timeout=60, headers=UA).json()
        items += d["items"]
        if not d.get("more"):
            return items
        page += 1


def route_points(T: Terrain):
    gj = json.loads((ROOT / "out" / "route" / "route_B.geojson").read_text())
    lines, stops = [], []
    for f in gj["features"]:
        if f["geometry"]["type"] == "LineString":
            c = np.array(f["geometry"]["coordinates"])
            lines.append(c)
        else:
            stops.append(f)
    return gj, lines, stops


def dem_z(T: Terrain, lon, lat):
    x, y = T.fwd.transform(lon, lat)
    r, c = rasterio.transform.rowcol(T.ds.transform, x, y)
    return T.z[np.asarray(r), np.asarray(c)]


def annotate(item, cam: Camera, T: Terrain, lines, stops, path: Path):
    img = Image.open(io.BytesIO(requests.get(item["https_url"], timeout=60, headers=UA).content)).convert("RGB")
    sx, sy = img.width / cam.w, img.height / cam.h
    draw = ImageDraw.Draw(img, "RGBA")
    cpos = cam.cam_site()
    az0, el0 = cam.look_az_el()

    # DEM skyline across the field of view
    sky = T.skyline(cpos, np.arange(az0 - 35, az0 + 35, 0.5))
    sky = [p for p in sky if p is not None]
    if sky:
        xy, front = cam.site_to_pixel(np.array(sky))
        pts = [(x * sx, y * sy) for (x, y), f in zip(xy, front) if f]
        draw.line(pts, fill=(255, 220, 0, 230), width=2)

    # route: one polyline per hop, solid where visible, faint where hidden
    n_vis = n_in = 0
    for c in lines:
        z = dem_z(T, c[:, 0], c[:, 1]) + 0.3                # a little above ground
        p = T.ned(c[:, 0], c[:, 1], z)
        xy, front = cam.site_to_pixel(p)
        vis = np.array([T.visible(cpos, q) for q in p])
        for i in range(len(p) - 1):
            if not (front[i] and front[i + 1]):
                continue
            a, b = (xy[i, 0] * sx, xy[i, 1] * sy), (xy[i + 1, 0] * sx, xy[i + 1, 1] * sy)
            inside = 0 <= a[0] <= img.width and 0 <= a[1] <= img.height
            n_in += inside
            if vis[i] and vis[i + 1]:
                n_vis += inside
                draw.line([a, b], fill=(0, 200, 255, 255), width=5)
            else:
                draw.line([a, b], fill=(0, 200, 255, 90), width=3)

    for f in stops:
        lon, lat = f["geometry"]["coordinates"]
        p = T.ned(lon, lat, float(dem_z(T, lon, lat)) + 1.0)[None, :]
        xy, front = cam.site_to_pixel(p)
        if front[0]:
            x, y = xy[0, 0] * sx, xy[0, 1] * sy
            if 0 <= x <= img.width and 0 <= y <= img.height:
                col = (255, 60, 200, 255) if f["properties"]["role"] == "habitat" else (255, 210, 0, 255)
                draw.ellipse([x - 9, y - 9, x + 9, y + 9], outline=col, width=3)
                draw.text((x + 12, y - 8), f["properties"]["name"], fill=col)

    img.save(path, quality=90)
    return n_in, n_vis


SITES = [  # waypoint sol where Curiosity parked, label
    (4711, "Drill pair (ridge / hollow)"),
    (4625, "Bococo nodules"),
]
MAX_PER_SITE = 3


def frames_at(sol, wp_sols):
    """Full-size Navcam frames taken while the rover stood at the end-of-drive position of `sol`."""
    later = [s for s in wp_sols if s > sol]
    end = min(later[0] - 1 if later else sol + 20, sol + 20)
    items = [i for i in fetch_items(sol, end)
             if i.get("camera_model_component_list") and i["extended"].get("sample_type") == "full"
             and i["subframe_rect"] == "(1,1,1024,1024)"]
    on_sol = [i for i in items if i["sol"] == sol]
    if not on_sol:
        return []
    key = max(((i["site"], i["drive"]) for i in on_sol), key=lambda k: (k[0], k[1]))  # last drive of that sol
    return [i for i in items if (i["site"], i["drive"]) == key]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wp = {int(r.split(",")[0]): r.split(",") for r in (ROOT / "out" / "boxwork_waypoints.csv").read_text().splitlines()[1:]}
    sites = []
    for sol, label in SITES:
        _, lat0, lon0, z0 = (float(v) for v in wp[sol][:4])
        T = Terrain(lon0, lat0, z0)                      # waypoint elevations equal the DEM
        _, lines, stops = route_points(T)

        # route points in local coordinates + line of sight from the mast, once per site
        cam_h = np.array([0, 0, -1.9])                   # ~Navcam height above ground (only for ranking)
        route = []
        for c in lines:
            z = dem_z(T, c[:, 0], c[:, 1]) + 0.3
            p = T.ned(c[:, 0], c[:, 1], z)
            route.append((p, np.array([T.visible(cam_h, q) for q in p])))

        items = frames_at(sol, sorted(wp))
        print(f"sol {sol} ({label}): {len(items)} full Navcam frames")
        scored = []
        for it in items:
            cam = Camera(it)
            az, el = cam.look_az_el()
            if el < -30 or el > 10:
                continue
            n_vis = 0
            for p, vis in route:
                xy, front = cam.site_to_pixel(p)
                inside = front & (xy[:, 0] > 0) & (xy[:, 0] < cam.w) & (xy[:, 1] > 0) & (xy[:, 1] < cam.h)
                n_vis += int(np.sum(inside & vis))
            scored.append((n_vis, az, el, it))
        scored.sort(key=lambda s: -s[0])

        photos, used_az = [], []
        for n_vis, az, el, it in scored:
            if n_vis < 5 or any(abs((az - a + 180) % 360 - 180) < 25 for a in used_az):
                continue                                 # distinct view directions only
            name = f"sol{sol}_az{az:03.0f}_{it['imageid'][:13]}.jpg"
            annotate(it, Camera(it), T, lines, stops, OUT / name)
            photos.append({"file": f"photos/{name}", "image_sol": it["sol"], "imageid": it["imageid"],
                           "azimuth_deg": round(az, 1), "elevation_deg": round(el, 1),
                           "lmst": it["extended"].get("lmst"), "source": it["https_url"],
                           "visible_route_points": n_vis})
            used_az.append(az)
            print(f"  wrote {name}: looking {az:.0f} deg, {n_vis} visible route points")
            if len(photos) >= MAX_PER_SITE:
                break
        sites.append({"sol": sol, "label": label, "lat": lat0, "lon": lon0, "photos": photos})

    (OUT / "photos.json").write_text(json.dumps({"credit": "NASA/JPL-Caltech", "sites": sites}, indent=1))


if __name__ == "__main__":
    main()
