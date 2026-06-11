#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_sample_packages.py — generate three ready-to-import MESA asset packages.

Each package is a self-contained folder under sample_data/:
    <theme>/<theme>.gpkg     one layer per asset group (layer name = group name),
                             synthetic but deliberately OVERLAPPING MultiPolygons
                             in a real geographic region (EPSG:4326).
    <theme>/settings.xlsx    the 'vulnerability' sheet MESA reads to assign
                             importance/susceptibility (matched on name_original).
    <theme>/README.md        how to use the package.

The overlaps are intentional: where several asset groups stack on the same cells,
the per-cell (importance, susceptibility) histogram gains depth, which is what makes
the Classification tool's types and certainty meaningful (a single-asset cell is
compositionally unambiguous — see the wiki "low-overlap regime" note).

Run:  python devtools/make_sample_packages.py
Deterministic (no RNG) so re-runs reproduce identical geometry.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
from shapely.affinity import rotate
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union
import openpyxl

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sample_data"

# Organic outlines harvested from OpenStreetMap (natural=wood/wetland/water,
# landuse=forest/farmland) around the sample regions, normalised to centroid 0 and
# sqrt(area)=1. Re-fetch with devtools/fetch_osm_templates.py. These give every asset
# patch a real, irregular shoreline instead of a circle.
TEMPLATES = json.loads((Path(__file__).resolve().parent / "osm_shape_templates.json").read_text())

# MESA A–E thresholds (mirror config.ini [A]..[E]).
def code_for(sens: int) -> tuple[str, str]:
    if sens >= 21: return "A", "Very high"
    if sens >= 16: return "B", "High"
    if sens >= 11: return "C", "Moderate"
    if sens >= 6:  return "D", "Low"
    return "E", "Very low"


def km_deg(lat: float) -> tuple[float, float]:
    """Degrees per km in (lon, lat) at a given latitude."""
    return 1.0 / (111.320 * math.cos(math.radians(lat))), 1.0 / 110.574


# Organic templates are normalised to sqrt(area)=1 (≈ 1/√π the linear size of a unit
# circle), so scale up to keep patch sizes — and thus the designed overlaps — similar
# to the original circular layout.
ORGANIC_SCALE = 2.85


def organic(tmpl, cx, cy, dx_km, dy_km, r_km, lat, ratio=1.0, rot=0.0):
    """Place a real OSM outline as an organic 'habitat patch' of ~r_km radius, offset
    (dx,dy) km from centre, with optional aspect ratio + rotation."""
    kx, ky = km_deg(lat)
    ox, oy = cx + dx_km * kx, cy + dy_km * ky
    sx = r_km * kx * ORGANIC_SCALE
    sy = r_km * ky * ratio * ORGANIC_SCALE
    p = Polygon([(ox + x * sx, oy + y * sy) for x, y in tmpl]).buffer(0)
    return rotate(p, rot, origin=(ox, oy)) if rot else p


def river(cx, cy, dx_km, dy_km, length_km, width_km, lat, rot=0.0, wiggle_km=1.4, waves=2.5):
    """A sinuous linear feature (river / channel / trail) as a meandering buffered
    centreline, built along +x then rotated — natural, not a straight rectangle."""
    kx, ky = km_deg(lat)
    ox, oy = cx + dx_km * kx, cy + dy_km * ky
    steps = 64
    pts = []
    for i in range(steps + 1):
        t = i / steps - 0.5
        perp = wiggle_km * math.sin((t + 0.5) * waves * 2 * math.pi)
        pts.append((ox + (t * length_km) * kx, oy + perp * ky))
    poly = LineString(pts).buffer(width_km * kx / 2, cap_style=1, join_style=1)
    return rotate(poly, rot, origin=(ox, oy)) if rot else poly


def group_geom(primitives) -> "object":
    """Union a list of shapely primitives into one (Multi)Polygon, cleaned."""
    return unary_union(primitives).buffer(0)


# Each category is written as SEVERAL organic objects, not one blob: the designed
# patches plus a few small scattered satellites. That gives more spatial dynamics
# (a category recurs across the area) and lifts total overlap. Deterministic.
N_SCATTER = 7
SCATTER_SPREAD_KM = 7.0


def _h(*xs) -> float:
    """Deterministic pseudo-random value in [0,1) from the inputs (no RNG)."""
    v = 0.0
    for i, x in enumerate(xs):
        v += math.sin((x + 1.0) * (12.9898 + i * 7.137)) * 43758.5453
    return v - math.floor(v)


def scatter_objects(cx, cy, lat, gi, n=N_SCATTER):
    """Small organic satellite patches for group index gi, scattered over the AOI."""
    out = []
    for k in range(n):
        dx = (_h(gi, k, 1) - 0.5) * 2 * SCATTER_SPREAD_KM
        dy = (_h(gi, k, 2) - 0.5) * 2 * SCATTER_SPREAD_KM
        r = 1.1 + _h(gi, k, 3) * 1.7
        ratio = 0.65 + _h(gi, k, 4) * 0.9
        rot = _h(gi, k, 5) * 360.0
        tmpl = TEMPLATES[int(_h(gi, k, 6) * len(TEMPLATES)) % len(TEMPLATES)]
        out.append(organic(tmpl, cx, cy, dx, dy, r, lat, ratio, rot))
    return out


def as_multipolygons(geoms):
    """Normalise a mix of Polygon/MultiPolygon to a clean list of MultiPolygons
    (one GPKG layer = one geometry type), dropping empties."""
    out = []
    for g in geoms:
        g = g.buffer(0)
        if g.is_empty:
            continue
        if g.geom_type == "Polygon":
            out.append(MultiPolygon([g]))
        elif g.geom_type == "MultiPolygon":
            out.append(g)
    return out


# ---------------------------------------------------------------------------
# Package definitions: (centre lon/lat, region label, groups).
# Each group: name -> (importance, susceptibility, [primitive specs]).
# Primitive spec: ('blob', dx, dy, r[, ratio, rot]) or ('strip', dx, dy, len, wid, rot).
# Offsets are km from the centre; overlaps are designed by sharing nearby offsets.
# ---------------------------------------------------------------------------
def P(cx, cy):  # bind centre+lat into a primitive builder
    counter = [0]

    def make(spec):
        kind = spec[0]
        idx = counter[0]
        counter[0] += 1                       # deterministic template/rotation variety
        if kind == "blob":
            dx, dy, r = spec[1], spec[2], spec[3]
            ratio = spec[4] if len(spec) > 4 else 1.0
            rot = spec[5] if len(spec) > 5 else (idx * 53) % 360
            tmpl = TEMPLATES[idx % len(TEMPLATES)]
            return organic(tmpl, cx, cy, dx, dy, r, cy, ratio, rot)
        dx, dy, ln, wd, rot = spec[1], spec[2], spec[3], spec[4], spec[5]
        return river(cx, cy, dx, dy, ln, wd, cy, rot)
    return make


PACKAGES = {
    "coastal_zone": {
        "centre": (39.30, -6.20),  # Zanzibar Channel, Tanzania
        "region": "Zanzibar Channel, Tanzania (tropical coast)",
        "groups": {
            # name: (importance, susceptibility, primitives)
            "Coral reef":          (5, 5, [("blob",  4,  2, 3.0), ("blob",  7,  0, 2.4)]),
            "Seagrass meadow":     (4, 4, [("blob",  3,  1, 3.2), ("blob",  6, -2, 2.6)]),
            "Mangrove":            (5, 4, [("blob", -3, -1, 2.8), ("blob", -1,  2, 2.4)]),
            "Saltmarsh":           (3, 4, [("blob", -4,  1, 2.2), ("blob", -2, -2, 1.8)]),
            "Fish breeding ground":(4, 3, [("blob", -1,  0, 3.6)]),                     # core overlap
            "Tourism beach":       (3, 2, [("blob", -5,  3, 1.6), ("blob", -4,  4, 1.4)]),
            "Aquaculture":         (2, 2, [("blob", -2,  3, 1.6), ("blob",  0,  3, 1.4)]),
            "Shipping lane":       (2, 1, [("strip", 3,  0, 22, 1.2, 25)]),             # cuts across reefs
        },
    },
    "river_system": {
        "centre": (38.40, -7.85),  # Rufiji River, Tanzania
        "region": "Rufiji River, Tanzania (large lowland river)",
        "groups": {
            "Fish spawning reach": (5, 5, [("blob",  0,  0, 2.2), ("blob",  4, -1, 1.8)]),
            "Wetland floodplain":  (5, 4, [("blob", -2,  2, 3.2), ("blob",  2,  2, 3.0), ("blob", 0, -2, 2.6)]),
            "Riparian forest":     (4, 4, [("strip", 0,  0, 26, 2.2, 18)]),            # along channel
            "Drinking water intake":(5, 3,[("blob",  6,  1, 1.4)]),
            "River channel":       (4, 3, [("strip", 0,  0, 30, 0.8, 18)]),            # the river itself
            "Irrigation abstraction":(3, 3,[("blob",  3,  3, 2.2), ("blob",  5,  2, 1.8)]),
            "Hydropower reservoir":(3, 2, [("blob", -7, -2, 3.0)]),
            "Agricultural land":   (2, 2, [("blob",  4,  4, 3.4), ("blob", -3,  4, 3.0)]),
            "Settlement urban":    (2, 1, [("blob",  6,  3, 1.8)]),
            "Sand mining":         (1, 1, [("blob", -1, -1, 1.2), ("blob",  2, -1, 1.0)]),
        },
    },
    "mountain_area": {
        "centre": (8.30, 61.60),   # Jotunheimen, Norway (alpine)
        "region": "Jotunheimen, Norway (alpine range)",
        "groups": {
            "Endemic species habitat":(5, 5, [("blob",  0,  1, 2.0), ("blob", -2, 3, 1.6)]),
            "Glacier snowfield":   (4, 5, [("blob",  2,  4, 3.0), ("blob",  5,  4, 2.4)]),
            "Old-growth forest":   (5, 4, [("blob", -3, -2, 3.0), ("blob",  0, -1, 2.6)]),
            "Headwater stream":    (4, 4, [("strip", 0,  1, 24, 0.7, -35)]),           # down the valley
            "Alpine meadow":       (4, 3, [("blob",  0,  2, 3.2)]),                     # core overlap
            "Grazing pasture":     (2, 3, [("blob", -2,  0, 2.4), ("blob",  1,  0, 2.0)]),
            "Ski resort":          (2, 2, [("blob",  3,  3, 1.8), ("strip", 4, 2, 6, 0.6, -50)]),
            "Hiking trail corridor":(2, 1, [("strip", 0, 0, 28, 0.5, 60)]),
            "Mining concession":   (1, 2, [("blob", -5, -3, 2.2)]),
            "Hydropower intake":   (2, 1, [("blob",  4,  5, 1.4)]),
        },
    },
    "mount_kenya": {
        "centre": (37.30, -0.15),  # Mount Kenya, Kenya (equatorial afro-alpine)
        "region": "Mount Kenya, Kenya (equatorial afro-alpine massif)",
        "groups": {
            # Importance/susceptibility set by the operator (revised settings.xlsx).
            "Endemic species habitat":(3, 4, [("blob",  0,  1, 2.4), ("blob",  1.5, 2, 1.8)]),
            "Montane cloud forest":(5, 5, [("blob", -2, -3, 3.4), ("blob",  2, -3, 3.2)]),
            "Glacier ice cap":     (1, 1, [("blob",  0,  0.5, 1.6), ("blob", -1, 0, 1.2)]),
            "Afro-alpine moorland":(5, 4, [("blob",  0,  2, 3.2)]),                     # overlaps endemic
            "Headwater tarn":      (5, 2, [("strip", 0,  0, 22, 0.7, -40), ("blob", 1, 1, 1.0)]),
            "Bamboo zone":         (3, 2, [("blob",  0, -4, 3.0)]),                      # below forest
            "Grazing pasture":     (2, 3, [("blob", -3,  2, 2.4), ("blob",  3,  3, 2.0)]),
            "Smallholder farmland":(4, 3, [("blob", -4, -5, 3.2), ("blob",  4, -5, 3.0)]),
            "Trekking route":      (2, 3, [("strip", 1,  0, 26, 0.5, 55)]),             # Sirimon/Chogoria
            "Logging concession":  (4, 3, [("blob", -3, -4, 2.4)]),                     # overlaps forest
        },
    },
}

VULN_HEADER = ["id", "name_original", "susceptibility", "importance",
               "sensitivity", "sensitivity_code", "sensitivity_description"]


def write_settings_xlsx(path: Path, groups: dict):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "vulnerability"
    ws.append(VULN_HEADER)
    for i, (name, (imp, sus, _prim)) in enumerate(groups.items(), start=1):
        sens = imp * sus
        code, desc = code_for(sens)
        ws.append([i, name, sus, imp, sens, code, desc])
    wb.save(path)


def write_readme(path: Path, theme: str, region: str, groups: dict):
    lines = [f"# MESA sample package — {theme.replace('_', ' ')}", "",
             f"Region: **{region}**. Synthetic, deliberately overlapping asset data for "
             "testing MESA end-to-end (processing, sensitivity, segmentation, Classification).", "",
             "## Use", "",
             "1. Copy `" + theme + ".gpkg` into `input/asset/` (replace or alongside existing assets).",
             "2. Copy `settings.xlsx` into `input/` (it carries the importance/susceptibility per group).",
             "3. Run Import assets → set up geocodes → process. Importance/susceptibility are matched",
             "   to each gpkg layer by `name_original`.", "",
             "## Asset groups (layer = group name)", "",
             "| Group | Importance | Susceptibility | Sensitivity | Code |",
             "|---|---:|---:|---:|:--:|"]
    for name, (imp, sus, _p) in groups.items():
        sens = imp * sus
        code, _ = code_for(sens)
        lines.append(f"| {name} | {imp} | {sus} | {sens} | {code} |")
    lines += ["", "Groups overlap by design, so stacked cells carry a mix of "
              "(importance, susceptibility) pairs — giving the Classification tool real "
              "histogram depth to cluster on.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def build():
    OUT.mkdir(exist_ok=True)
    summary = []
    for theme, spec in PACKAGES.items():
        cx, cy = spec["centre"]
        make = P(cx, cy)
        folder = OUT / theme
        folder.mkdir(parents=True, exist_ok=True)
        gpkg = folder / f"{theme}.gpkg"
        if gpkg.exists():
            gpkg.unlink()
        total_obj = 0
        for gi, (name, (_imp, _sus, prims)) in enumerate(spec["groups"].items()):
            objs = [make(p) for p in prims] + scatter_objects(cx, cy, cy, gi)
            objs = as_multipolygons(objs)
            total_obj += len(objs)
            gdf = gpd.GeoDataFrame({"class": [name] * len(objs)}, geometry=objs, crs="EPSG:4326")
            gdf.to_file(gpkg, layer=name, driver="GPKG")
        write_settings_xlsx(folder / "settings.xlsx", spec["groups"])
        write_readme(folder / "README.md", theme, spec["region"], spec["groups"])
        summary.append((theme, len(spec["groups"]), gpkg))
        print(f"[{theme}] {len(spec['groups'])} groups, {total_obj} objects -> {gpkg}")

    top = ["# MESA sample data packages", "",
           "Three self-contained, ready-to-import asset packages with reasonable "
           "importance/susceptibility parameters and deliberately overlapping geometry.", ""]
    for theme, n, _g in summary:
        top.append(f"- **{theme.replace('_', ' ')}** — {n} asset groups "
                   f"(`{theme}/{theme}.gpkg` + `{theme}/settings.xlsx`)")
    top += ["", "See each folder's README for usage. Generated by "
            "`devtools/make_sample_packages.py` (deterministic).", ""]
    (OUT / "README.md").write_text("\n".join(top), encoding="utf-8")
    print(f"Done -> {OUT}")


if __name__ == "__main__":
    build()
