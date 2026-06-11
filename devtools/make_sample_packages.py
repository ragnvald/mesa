#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_sample_packages.py — generate sample MESA asset packages.

Each package is a self-contained folder under sample_data/:
    <theme>/<theme>.gpkg     one layer per asset group (layer name = group name),
                             several procedurally-generated, deliberately OVERLAPPING
                             organic MultiPolygons in a real region (EPSG:4326).
    <theme>/settings.xlsx    the 'vulnerability' sheet MESA reads to assign
                             importance/susceptibility (matched on name_original).
    <theme>/README.md        how to use the package.

Geometry design (see learning.md "Sample-data generator: organic + scattered"):
  * Shapes are NOT a few templates rotated/copy-pasted. Every patch is a unique
    organic outline built from a random multi-harmonic radial function, so no two
    patches are identical. Linear features (rivers/trails/lanes) are meandering
    buffered centrelines with their OWN random orientation — never rotated around
    the AOI centre (that produced the old radial "star" of spokes).
  * Placement spreads across the whole AOI. A handful of shared overlap *hotspots*
    sit at mid-radius (distributed, not one central pile), and each group also throws
    several small *satellites* into the outer ring. Overlaps where groups share a
    hotspot give the per-cell (importance, susceptibility) histogram the depth the
    Classification tool clusters on; the satellites give spatial dynamics.

Run:  python devtools/make_sample_packages.py
Deterministic: a fixed master seed per (theme, group) reproduces identical geometry.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union
import openpyxl

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sample_data"

# Fixed master seed -> deterministic re-runs. Bump to reshuffle every package.
SEED = 20260611

# AOI half-extent. Features spread across ~2*AOI_RADIUS_KM (~26 km), matching the
# ~23 km basic_mosaic footprint of the shipped sample regions.
AOI_RADIUS_KM = 13.0
N_HOTSPOTS = 6            # shared overlap anchors, spread across the AOI
HOTSPOT_SPREAD = 0.72     # fraction of AOI radius the hotspots reach out to
HOTSPOT_MIN_SEP = 0.34    # min hotspot separation (fraction of AOI radius)
SAT_RING = (0.42, 1.0)    # satellites land in this radial band -> real outliers


# MESA A-E thresholds (mirror config.ini [A]..[E]).
def code_for(sens: int) -> tuple[str, str]:
    if sens >= 21: return "A", "Very high"
    if sens >= 16: return "B", "High"
    if sens >= 11: return "C", "Moderate"
    if sens >= 6:  return "D", "Low"
    return "E", "Very low"


def km_deg(lat: float) -> tuple[float, float]:
    """Degrees per km in (lon, lat) at a given latitude."""
    return 1.0 / (111.320 * math.cos(math.radians(lat))), 1.0 / 110.574


def _off(cx, cy, dx_km, dy_km, lat):
    """km offset from package centre -> (lon, lat)."""
    kx, ky = km_deg(lat)
    return cx + dx_km * kx, cy + dy_km * ky


# ---------------------------------------------------------------------------
# Procedural organic shapes — each call returns a UNIQUE outline (no templates).
# ---------------------------------------------------------------------------
def organic_blob(rng, lon0, lat0, r_km, lat):
    """A natural lobed patch of ~r_km radius centred at (lon0, lat0).

    The boundary radius is 1 + a sum of a few sine harmonics with random
    amplitude/phase, so every patch has its own irregular, organic shape.
    """
    kx, ky = km_deg(lat)
    n_pts = rng.randint(10, 18)
    n_harm = rng.randint(2, 5)
    amps = [rng.uniform(0.06, 0.42) / (h + 1) for h in range(n_harm)]
    phs = [rng.uniform(0, 2 * math.pi) for _ in range(n_harm)]
    # mild anisotropy + a free rotation so patches are not radially symmetric
    ax = rng.uniform(0.72, 1.38)
    ay = rng.uniform(0.72, 1.38)
    rot = rng.uniform(0, 2 * math.pi)
    cr, sr = math.cos(rot), math.sin(rot)
    start = rng.uniform(0, 2 * math.pi)
    pts = []
    for i in range(n_pts):
        a = start + 2 * math.pi * i / n_pts
        rad = 1.0 + sum(A * math.sin((h + 1) * a + p) for h, (A, p) in enumerate(zip(amps, phs)))
        rad = max(0.32, rad) * r_km
        x, y = rad * math.cos(a) * ax, rad * math.sin(a) * ay
        xr, yr = x * cr - y * sr, x * sr + y * cr
        pts.append((lon0 + xr * kx, lat0 + yr * ky))
    poly = Polygon(pts).buffer(0)
    # round off the sharpest concavities -> more natural coastline feel
    s = 0.05 * kx
    return poly.buffer(s).buffer(-s) if not poly.is_empty else poly


def organic_line(rng, lon0, lat0, length_km, width_km, lat):
    """A meandering buffered centreline with its OWN random orientation.

    Built along a random heading through (lon0, lat0); the perpendicular meander is
    a couple of low-frequency waves. Never rotated about the AOI centre.
    """
    kx, ky = km_deg(lat)
    ang = rng.uniform(0, 2 * math.pi)
    ca, sa = math.cos(ang), math.sin(ang)
    waves = rng.uniform(1.3, 3.4)
    wiggle = rng.uniform(0.7, 2.1)
    phase = rng.uniform(0, 2 * math.pi)
    drift = rng.uniform(-0.5, 0.5)            # gentle overall curve
    steps = 56
    pts = []
    for i in range(steps + 1):
        t = i / steps - 0.5
        along = t * length_km
        perp = wiggle * math.sin(t * waves * 2 * math.pi + phase) + drift * (t * length_km)
        x, y = along * ca - perp * sa, along * sa + perp * ca
        pts.append((lon0 + x * kx, lat0 + y * ky))
    return LineString(pts).buffer(width_km * kx / 2.0, cap_style=1, join_style=1).buffer(0)


# ---------------------------------------------------------------------------
# Placement helpers — spread features across the AOI, not piled at the centre.
# ---------------------------------------------------------------------------
def _disc_point(rng, r_lo, r_hi):
    """Random km offset (x, y) in an annulus [r_lo, r_hi] of the AOI (area-uniform)."""
    ang = rng.uniform(0, 2 * math.pi)
    u = rng.random()
    r = math.sqrt(r_lo * r_lo + u * (r_hi * r_hi - r_lo * r_lo))
    return r * math.cos(ang), r * math.sin(ang)


def make_hotspots(rng):
    """Distributed mid-AOI overlap anchors (km offsets), kept apart by min-separation."""
    spread = AOI_RADIUS_KM * HOTSPOT_SPREAD
    sep = AOI_RADIUS_KM * HOTSPOT_MIN_SEP
    pts = []
    for _ in range(4000):
        if len(pts) >= N_HOTSPOTS:
            break
        x, y = _disc_point(rng, 0.15 * AOI_RADIUS_KM, spread)
        if all(math.hypot(x - px, y - py) > sep for px, py in pts):
            pts.append((x, y))
    return pts


def build_group(rng, cx, cy, lat, kind, hotspots, assigned, scale):
    """Return a list of shapely geometries for one asset group."""
    geoms = []
    if kind == "linear":
        # one main meandering feature anchored on a hotspot, random heading
        hx, hy = hotspots[assigned[0]]
        ax, ay = hx + rng.uniform(-1.5, 1.5), hy + rng.uniform(-1.5, 1.5)
        lon0, lat0 = _off(cx, cy, ax, ay, lat)
        geoms.append(organic_line(rng, lon0, lat0,
                                  length_km=rng.uniform(13, 24) * scale,
                                  width_km=rng.uniform(0.4, 0.9) * scale, lat=lat))
        # a few short tributaries / pools scattered into the outer ring
        for _ in range(rng.randint(2, 4)):
            dx, dy = _disc_point(rng, SAT_RING[0] * AOI_RADIUS_KM, SAT_RING[1] * AOI_RADIUS_KM)
            lon0, lat0 = _off(cx, cy, dx, dy, lat)
            if rng.random() < 0.5:
                geoms.append(organic_line(rng, lon0, lat0,
                                          length_km=rng.uniform(3, 7) * scale,
                                          width_km=rng.uniform(0.3, 0.6) * scale, lat=lat))
            else:
                geoms.append(organic_blob(rng, lon0, lat0, rng.uniform(0.5, 1.1) * scale, lat))
        return geoms

    # area kind: 1-2 medium core patches at shared hotspots (overlap), then satellites
    for hi in assigned:
        hx, hy = hotspots[hi]
        for _ in range(rng.randint(1, 2)):
            jx, jy = hx + rng.uniform(-2.2, 2.2), hy + rng.uniform(-2.2, 2.2)
            lon0, lat0 = _off(cx, cy, jx, jy, lat)
            geoms.append(organic_blob(rng, lon0, lat0, rng.uniform(1.7, 3.2) * scale, lat))
    for _ in range(rng.randint(4, 8)):
        dx, dy = _disc_point(rng, SAT_RING[0] * AOI_RADIUS_KM, SAT_RING[1] * AOI_RADIUS_KM)
        lon0, lat0 = _off(cx, cy, dx, dy, lat)
        geoms.append(organic_blob(rng, lon0, lat0, rng.uniform(0.5, 1.5) * scale, lat))
    return geoms


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
# Each group: name -> (importance, susceptibility, kind[, scale]).
#   kind  = "area"   organic patches + outer-ring satellites
#         = "linear" meandering channel/trail/lane + scattered short segments
#   scale = optional size multiplier (small features < 1.0).
# Overlap hotspots and satellite placement are generated procedurally per theme,
# so groups overlap where they share hotspots — distributed across the AOI.
# ---------------------------------------------------------------------------
PACKAGES = {
    "coastal_zone": {
        "centre": (39.30, -6.20),  # Zanzibar Channel, Tanzania
        "region": "Zanzibar Channel, Tanzania (tropical coast)",
        "groups": {
            "Coral reef":           (5, 5, "area"),
            "Seagrass meadow":      (4, 4, "area"),
            "Mangrove":             (5, 4, "area"),
            "Saltmarsh":            (3, 4, "area"),
            "Fish breeding ground": (4, 3, "area"),
            "Tourism beach":        (3, 2, "linear"),
            "Aquaculture":          (2, 2, "area", 0.7),
            "Shipping lane":        (2, 1, "linear"),
        },
    },
    "river_system": {
        "centre": (38.40, -7.85),  # Rufiji River, Tanzania
        "region": "Rufiji River, Tanzania (large lowland river)",
        "groups": {
            "Fish spawning reach":   (5, 5, "area"),
            "Wetland floodplain":    (5, 4, "area"),
            "Riparian forest":       (4, 4, "linear"),
            "Drinking water intake": (5, 3, "area", 0.6),
            "River channel":         (4, 3, "linear"),
            "Irrigation abstraction":(3, 3, "area"),
            "Hydropower reservoir":  (3, 2, "area"),
            "Agricultural land":     (2, 2, "area"),
            "Settlement urban":      (2, 1, "area", 0.8),
            "Sand mining":           (1, 1, "area", 0.6),
        },
    },
    "mountain_area": {
        "centre": (8.30, 61.60),   # Jotunheimen, Norway (alpine)
        "region": "Jotunheimen, Norway (alpine range)",
        "groups": {
            "Endemic species habitat":(5, 5, "area", 0.8),
            "Glacier snowfield":      (4, 5, "area"),
            "Old-growth forest":      (5, 4, "area"),
            "Headwater stream":       (4, 4, "linear"),
            "Alpine meadow":          (4, 3, "area"),
            "Grazing pasture":        (2, 3, "area"),
            "Ski resort":             (2, 2, "area", 0.7),
            "Hiking trail corridor":  (2, 1, "linear"),
            "Mining concession":      (1, 2, "area", 0.8),
            "Hydropower intake":      (2, 1, "area", 0.6),
        },
    },
    "mount_kenya": {
        "centre": (37.30, -0.15),  # Mount Kenya, Kenya (equatorial afro-alpine)
        "region": "Mount Kenya, Kenya (equatorial afro-alpine massif)",
        "groups": {
            "Endemic species habitat":(3, 4, "area", 0.8),
            "Montane cloud forest":   (5, 5, "area"),
            "Glacier ice cap":        (1, 1, "area", 0.5),
            "Afro-alpine moorland":   (5, 4, "area"),
            "Headwater tarn":         (5, 2, "linear"),
            "Bamboo zone":            (3, 2, "area"),
            "Grazing pasture":        (2, 3, "area"),
            "Smallholder farmland":   (4, 3, "area"),
            "Trekking route":         (2, 3, "linear"),
            "Logging concession":     (4, 3, "area"),
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
    for i, (name, spec) in enumerate(groups.items(), start=1):
        imp, sus = spec[0], spec[1]
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
    for name, spec in groups.items():
        imp, sus = spec[0], spec[1]
        sens = imp * sus
        code, _ = code_for(sens)
        lines.append(f"| {name} | {imp} | {sus} | {sens} | {code} |")
    lines += ["", "Each group is scattered across the area as a few organic patches plus "
              "outer satellites, overlapping other groups at shared hotspots — so stacked "
              "cells carry a mix of (importance, susceptibility) pairs, giving the "
              "Classification tool real histogram depth to cluster on.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def build():
    OUT.mkdir(exist_ok=True)
    summary = []
    for theme_i, (theme, spec) in enumerate(PACKAGES.items()):
        cx, cy = spec["centre"]
        folder = OUT / theme
        folder.mkdir(parents=True, exist_ok=True)
        gpkg = folder / f"{theme}.gpkg"
        if gpkg.exists():
            gpkg.unlink()

        # Theme-level RNG fixes the shared hotspots all groups overlap on.
        theme_rng = random.Random(SEED + theme_i * 1009)
        hotspots = make_hotspots(theme_rng)
        n_hot = len(hotspots)

        total_obj = 0
        for gi, (name, gspec) in enumerate(spec["groups"].items()):
            kind = gspec[2]
            scale = gspec[3] if len(gspec) > 3 else 1.0
            grp_rng = random.Random(SEED + theme_i * 1009 + (gi + 1) * 131)
            # assign 1-2 shared hotspots -> groups that share one overlap there
            k = 1 if kind == "linear" else grp_rng.randint(1, 2)
            assigned = grp_rng.sample(range(n_hot), k=min(k, n_hot))
            objs = build_group(grp_rng, cx, cy, cy, kind, hotspots, assigned, scale)
            objs = as_multipolygons(objs)
            total_obj += len(objs)
            gdf = gpd.GeoDataFrame({"class": [name] * len(objs)}, geometry=objs, crs="EPSG:4326")
            gdf.to_file(gpkg, layer=name, driver="GPKG")

        write_settings_xlsx(folder / "settings.xlsx", spec["groups"])
        write_readme(folder / "README.md", theme, spec["region"], spec["groups"])
        summary.append((theme, len(spec["groups"]), gpkg))
        print(f"[{theme}] {len(spec['groups'])} groups, {total_obj} objects -> {gpkg}")

    top = ["# MESA sample data packages", "",
           "Four self-contained, ready-to-import asset packages with reasonable "
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
