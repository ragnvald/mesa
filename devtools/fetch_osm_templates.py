#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fetch_osm_templates.py — harvest organic outlines from OpenStreetMap.

Queries Overpass for natural/landuse polygons around the sample-data regions and
normalises each outline (centroid → 0, sqrt(area) → 1) into a reusable shape template.
make_sample_packages.py loads the result so every asset patch gets a real, irregular
shoreline instead of a circle.

Run:  python devtools/fetch_osm_templates.py
Writes devtools/osm_shape_templates.json. Network-dependent; Overpass rate-limits, so
it keeps whatever it manages to fetch (existing templates are enough to regenerate).
"""
from __future__ import annotations

import json
import math
import time
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from shapely.geometry import Polygon  # noqa: E402

OUT = Path(__file__).resolve().parent / "osm_shape_templates.json"

# (south, west, north, east) per sample region.
BBOX = {
    "coast": (-6.32, 39.18, -6.08, 39.42),   # Zanzibar Channel
    "river": (-7.98, 38.28, -7.72, 38.52),   # Rufiji River
    "alps":  (61.48, 8.16, 61.72, 8.44),     # Jotunheimen
    "kenya": (-0.26, 37.18, -0.04, 37.42),   # Mount Kenya
}
TAGS = ["natural=wood", "natural=wetland", "natural=water", "natural=scrub",
        "natural=glacier", "landuse=forest", "landuse=farmland"]
PER_SOURCE = 4
MAX_TEMPLATES = 16


def fetch(bbox):
    s, w, n, e = bbox
    parts = "".join(f'way["{k}"="{v}"]({s},{w},{n},{e});'
                    for k, v in (t.split("=") for t in TAGS))
    q = f"[out:json][timeout:60];({parts});out geom;"
    req = urllib.request.Request("https://overpass-api.de/api/interpreter",
                                 data=q.encode(), headers={"User-Agent": "mesa-sample/1.0"})
    return json.loads(urllib.request.urlopen(req, timeout=90).read()).get("elements", [])


def normalise(geom):
    coords = [(p["lon"], p["lat"]) for p in geom]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    poly = Polygon(coords)
    if not poly.is_valid or poly.area <= 0:
        return None
    cx, cy = poly.centroid.x, poly.centroid.y
    sc = math.sqrt(poly.area)
    norm = [[round((x - cx) / sc, 4), round((y - cy) / sc, 4)] for x, y in poly.exterior.coords]
    return norm if 12 <= len(norm) <= 200 else None


def main():
    rings = []
    for name, bb in BBOX.items():
        try:
            els = fetch(bb)
        except Exception as ex:
            print(f"{name}: fetch failed ({ex})")
            time.sleep(2)
            continue
        kept = 0
        for el in els:
            g = el.get("geometry")
            if not g or len(g) < 10:
                continue
            norm = normalise(g)
            if norm:
                rings.append({"src": name, "n": len(norm), "coords": norm})
                kept += 1
        print(f"{name}: {kept} usable rings")
        time.sleep(2)  # be polite to Overpass

    rings.sort(key=lambda r: -r["n"])           # prefer the most detailed outlines
    sel, per = [], {}
    for r in rings:
        if per.get(r["src"], 0) >= PER_SOURCE:
            continue
        sel.append(r["coords"])
        per[r["src"]] = per.get(r["src"], 0) + 1
        if len(sel) >= MAX_TEMPLATES:
            break
    if not sel:
        print("No templates fetched; keeping existing file.")
        return
    OUT.write_text(json.dumps(sel))
    print(f"Saved {len(sel)} templates -> {OUT} (sources: {per})")


if __name__ == "__main__":
    main()
