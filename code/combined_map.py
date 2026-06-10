#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""combined_map.py — unified Overview (Results) + Segmentation map (tabbed).

One pywebview window with two tabs, each its own Leaflet map:
  - Overview      : index / MBTiles layers (basemap only for now — next increment)
  - Segmentation  : raster MBTiles when present (<group>_seg_signatures.mbtiles /
                    <group>_seg_clusters.mbtiles, produced by the Tiles stage), with
                    a small vector fallback for levels that have no tiles yet.

The Asset map is deliberately kept as its own separate window (asset_map_view.py)
so its heavy asset layers don't load alongside Results — see docs/UNIFIED_MAP_PLAN.md.

A "Link zoom & pan" toggle in the header (right of the tabs, left of Exit) keeps
the two maps' view in lockstep when enabled.

The UI and the MBTiles tiles are served from one loopback origin so WebView2 /
Windows does not block loopback tile requests from an opaque html= origin (same
approach as map_overview.py).

CALLED BY
    mesa.py -> _launch_helper_subprocess("combined_map")
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sqlite3
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import webview  # pip install pywebview
except Exception:
    webview = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segmentation as seg  # noqa: E402

# 1x1 transparent PNG, returned for tile misses so image loads never break.
BLANK_PNG = (b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
             b'\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01'
             b'\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82')

_BASE_DIR: Path = Path.cwd()
HTML: str = ""


def base_dir() -> Path:
    candidates = []
    if os.environ.get("MESA_BASE_DIR"):
        candidates.append(Path(os.environ["MESA_BASE_DIR"]))
    candidates.append(Path.cwd())
    candidates.append(Path(__file__).resolve().parent.parent)
    for c in candidates:
        if (c / "output" / "geoparquet").exists() or (c / "config.ini").exists():
            return c
    return candidates[0]


def _mbtiles_dir() -> Path:
    return _BASE_DIR / "output" / "mbtiles"


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "", str(name))


def _mbtiles_meta(path: Path) -> dict:
    out = {"bounds": None, "minzoom": None, "maxzoom": None, "format": "image/png"}
    try:
        con = sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True, timeout=5.0)
        try:
            for k in ("bounds", "minzoom", "maxzoom", "format"):
                r = con.execute("SELECT value FROM metadata WHERE name=?", (k,)).fetchone()
                if r and r[0] is not None:
                    out[k] = r[0]
        finally:
            con.close()
    except Exception:
        pass
    if out["bounds"]:
        try:
            out["bounds"] = [float(x) for x in str(out["bounds"]).split(",")]
        except Exception:
            out["bounds"] = None
    for k in ("minzoom", "maxzoom"):
        try:
            out[k] = int(out[k]) if out[k] is not None else None
        except Exception:
            out[k] = None
    if not str(out["format"]).startswith("image/"):
        out["format"] = "image/png"
    return out


# ---------------------------------------------------------------------------
# GetFeatureInfo payload builders (module-level so the bridge stays lean)
# ---------------------------------------------------------------------------
# Results overlay suffix -> human label (mirrors the Overview layer radios).
_FI_RESULTS_OVERLAY_LABELS = {
    "sensitivity_max": "Sensitivity (max)",
    "importance_max": "Importance (max)",
    "index_owa": "OWA index",
    "groupstotal": "# asset groups",
    "assetstotal": "# asset objects",
}
# Always-shown context metrics. (label, column, transform, unit)
# OWA index is here (not just under its own overlay) so the value is visible on
# every results popup; _fi_metrics dedups by column so the OWA overlay won't repeat it.
_FI_RESULTS_GENERAL = [
    ("OWA index", "index_owa", None, None),
    ("# asset groups", "asset_groups_total", None, None),
    ("# asset objects", "assets_overlap_total", None, None),
    ("Area", "area_m2", lambda v: float(v) / 1_000_000.0, "km²"),
]
# Overlay-specific metrics layered on top of the general ones.
_FI_RESULTS_SPECIFIC = {
    "sensitivity_max": [
        ("Sensitivity (max)", "sensitivity_max", None, None),
        ("Sensitivity code", "sensitivity_code_max", None, None),
        ("Sensitivity description", "sensitivity_description_max", None, None),
        ("Importance (max)", "importance_max", None, None),
        ("Susceptibility (max)", "susceptibility_max", None, None),
    ],
    "importance_max": [
        ("Importance (max)", "importance_max", None, None),
        ("Importance code", "importance_code_max", None, None),
        ("Importance description", "importance_description_max", None, None),
    ],
    "index_owa": [("OWA index", "index_owa", None, None)],
    "groupstotal": [("Asset group names", "asset_group_names", None, None)],
    "assetstotal": [("# asset objects", "assets_overlap_total", None, None)],
}


def _fi_clean_value(val):
    try:
        import pandas as pd
        if pd.isna(val):
            return None
    except Exception:
        pass
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        try:
            f = float(val)
        except Exception:
            return val
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return int(f) if f.is_integer() else f
    s = str(val).strip()
    return s or None


def _fi_metrics(row, fields):
    out, seen = [], set()
    for label, col, tx, unit in fields:
        if col in seen:
            continue
        seen.add(col)
        val = row.get(col)
        if tx is not None and val is not None:
            try:
                val = tx(val)
            except Exception:
                continue
        cl = _fi_clean_value(val)
        if cl is not None and cl != "":
            out.append({"label": label, "value": cl, "unit": unit})
    return out


def _fi_results_info(row, overlay, group):
    overlay = overlay if overlay in _FI_RESULTS_OVERLAY_LABELS else "sensitivity_max"
    fields = _FI_RESULTS_GENERAL + _FI_RESULTS_SPECIFIC.get(overlay, [])
    return {
        "title": str(row.get("code") or "Cell"),
        "subtitle": f"{group} · {_FI_RESULTS_OVERLAY_LABELS[overlay]}",
        "metrics": _fi_metrics(row, fields),
    }


def _fi_seg_info(row, level, mode):
    if mode == "clusters":
        cid = row.get("cluster_id")
        title = f"Cluster {cid}" if cid is not None and str(cid) != "" else "Cluster"
        fields = [
            ("Cluster id", "cluster_id", None, None),
            ("Method", "cluster_method", None, None),
            ("# assets", "n_assets", None, None),
            ("Mean sensitivity", "sens_mean", None, None),
        ]
    else:
        title = str(row.get("signature") or "Signature")
        fields = [
            ("Signature", "signature", None, None),
            ("# assets", "n_assets", None, None),
            ("Mean sensitivity", "sens_mean", None, None),
        ]
    return {
        "title": title,
        "subtitle": f"{level} · {'Clusters' if mode == 'clusters' else 'Signatures'}",
        "metrics": _fi_metrics(row, fields),
    }


def _fi_json_ready(v):
    try:
        import numpy as _np
        if isinstance(v, (_np.integer,)):
            return int(v)
        if isinstance(v, (_np.floating,)):
            return float(v)
    except Exception:
        pass
    if isinstance(v, dict):
        return {k: _fi_json_ready(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_fi_json_ready(x) for x in v]
    return v


# ---------------------------------------------------------------------------
# Python <-> JS bridge
# ---------------------------------------------------------------------------

class _Api:
    def __init__(self, base: Path):
        self.base = Path(base)
        self.gpq = self.base / "output" / "geoparquet"
        self._window = None
        # GetFeatureInfo: lazily-built, per-(mode, layer) GeoDataFrame cache (4326,
        # with a geopandas spatial index) so repeated map clicks don't re-read parquet.
        self._fi_cache: dict = {}

    def set_window(self, w):
        self._window = w

    def save_png(self, data_url: str) -> dict:
        """Save a data-URL PNG (from html2canvas) via a Save dialog — mirrors
        map_overview / asset_map_view."""
        try:
            b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
            data = base64.b64decode(b64)
            win = webview.windows[0]
            path = win.create_file_dialog(
                webview.FileDialog.SAVE,
                save_filename="mesa_map_export.png",
                file_types=("PNG Files (*.png)",),
            )
            if not path:
                return {"ok": False, "error": "cancelled"}
            if isinstance(path, (list, tuple)):
                path = path[0]
            with open(path, "wb") as f:
                f.write(data)
            return {"ok": True, "path": str(path)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def exit_app(self):
        # Match the proven pattern from map_overview / asset_map_view: the
        # module-level destroy_window() reliably closes the active window across
        # backends; fall back to a hard exit if it raises.
        try:
            webview.destroy_window()
        except Exception:
            try:
                if self._window is not None:
                    self._window.destroy()
            except Exception:
                pass
            os._exit(0)

    # -- overview (index MBTiles) -------------------------------------------
    def area_stats(self) -> dict:
        """A–E area distribution (output/area_stats.json) + config A–E colours,
        for the small bar chart on the Overview tab (as the old map viewer had)."""
        import configparser
        out = {"labels": [], "values": [], "colors": []}
        p = self.base / "output" / "area_stats.json"
        if not p.exists():
            return out
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return out
        labels = list(d.get("labels", []))
        values = list(d.get("values", []))
        # NB: only ';' as inline-comment prefix — '#' would strip hex colour
        # values like '#bd0026' down to empty.
        cfg = configparser.ConfigParser(inline_comment_prefixes=(";",), strict=False)
        try:
            cfg.read(self.base / "config.ini", encoding="utf-8")
        except Exception:
            pass
        colors, descriptions = [], []
        for lab in labels:
            col, desc = "#cccccc", ""
            if cfg.has_section(lab):
                col = (cfg[lab].get("category_colour") or cfg[lab].get("category_color") or "#cccccc")
                desc = cfg[lab].get("description") or ""
            colors.append(str(col).strip())
            descriptions.append(str(desc).strip())
        total = float(sum(v for v in values if isinstance(v, (int, float))))
        return {"labels": labels, "values": values, "colors": colors,
                "descriptions": descriptions, "total": total}

    def mbtiles_catalog(self) -> dict:
        """Scan output/mbtiles for index layers → {groups: {group: {kinds:
        {suffix: {name, label, bounds, minzoom, maxzoom}}}}}. seg_* tiles are
        excluded (they belong to the Segmentation tab)."""
        suffixes = [
            ("sensitivity_max", "Sensitivity"),
            ("index_owa", "OWA index"),
            ("importance_max", "Importance (max)"),
            ("groupstotal", "# asset groups"),
            ("assetstotal", "# asset objects"),
        ]
        groups: dict = {}
        d = _mbtiles_dir()
        if not d.is_dir():
            return {"groups": groups}
        for p in sorted(d.glob("*.mbtiles")):
            stem = p.stem
            for suf, label in suffixes:
                if stem.endswith("_" + suf):
                    grp = stem[: -(len(suf) + 1)]
                    meta = _mbtiles_meta(p)
                    groups.setdefault(grp, {"kinds": {}})
                    groups[grp]["kinds"][suf] = {
                        "name": stem, "label": label, "bounds": meta["bounds"],
                        "minzoom": meta["minzoom"], "maxzoom": meta["maxzoom"],
                    }
                    break
        return {"groups": groups}

    # -- assets (lazy) -------------------------------------------------------
    def asset_groups(self) -> list[dict]:
        """Asset-group metadata only (no geometry): {gid, label, count, color}.
        Colour from styling.fill_color, else the config A–E ramp by
        sensitivity_code, else a default. Lightweight — for the toggle list."""
        import json
        import configparser
        import pandas as pd
        p = self.gpq / "tbl_asset_group.parquet"
        if not p.exists():
            return []
        try:
            g = pd.read_parquet(p)
        except Exception:
            return []
        cfg = configparser.ConfigParser(inline_comment_prefixes=(";",), strict=False)
        try:
            cfg.read(self.base / "config.ini", encoding="utf-8")
        except Exception:
            pass

        def _color(row):
            st = row.get("styling")
            if isinstance(st, str) and st.strip():
                try:
                    d = json.loads(st)
                    fc = d.get("fill_color") or d.get("fill") or d.get("color")
                    if fc:
                        return str(fc)
                except Exception:
                    pass
            code = str(row.get("sensitivity_code", "") or "").strip().upper()
            if code and cfg.has_section(code):
                c = cfg[code].get("category_colour")
                if c:
                    return str(c).strip()
            return "#7f8c9b"

        out = []
        for _, r in g.iterrows():
            gid = r.get("id")
            label = str(r.get("title_fromuser") or r.get("name_original")
                        or r.get("name_gis_assetgroup") or ("Group " + str(gid)))
            cnt = r.get("total_asset_objects")
            out.append({
                "gid": int(gid) if pd.notna(gid) else gid,
                "label": label,
                "count": int(cnt) if pd.notna(cnt) else None,
                "color": _color(r),
            })
        out.sort(key=lambda d: d["label"].lower())
        return out

    def asset_layer(self, gid) -> dict:
        """GeoJSON for one asset group (filtered read of tbl_asset_object on
        ref_asset_group) — loaded only when the operator toggles the group."""
        import geopandas as gpd
        p = self.gpq / "tbl_asset_object.parquet"
        if not p.exists():
            return {"type": "FeatureCollection", "features": []}

        def _read(val):
            return gpd.read_parquet(p, columns=["ref_asset_group", "geometry", "id"],
                                    filters=[("ref_asset_group", "=", val)])
        try:
            gdf = _read(gid)
            if gdf.empty:
                gdf = _read(str(gid))  # ref_asset_group may be string-typed
        except Exception as exc:
            return {"error": str(exc)}
        if gdf is None or gdf.empty:
            return {"type": "FeatureCollection", "features": []}
        try:
            if gdf.crs is not None and "4326" not in str(gdf.crs):
                gdf = gdf.to_crs(4326)
        except Exception:
            pass
        feats = []
        for _, r in gdf.iterrows():
            geom = r.geometry
            if geom is None or geom.is_empty:
                continue
            feats.append({"type": "Feature", "geometry": geom.__geo_interface__,
                          "properties": {"id": str(r.get("id", "") or "")}})
        return {"type": "FeatureCollection", "features": feats}

    def generate_ai_styles(self, group_ids) -> dict:
        """AI cartography styling: distinct per-group colours, persisted to
        tbl_asset_group. (Local generator — same path asset_map_view uses when
        no OpenAI key is configured.)"""
        try:
            import asset_styling
            styles = asset_styling.apply_ai_styles(self.gpq / "tbl_asset_group.parquet", group_ids or [])
            return {"ok": True, "styles": styles, "mode": "local"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def clear_asset_styles(self, group_ids) -> dict:
        try:
            import asset_styling
            asset_styling.clear_styles(self.gpq / "tbl_asset_group.parquet", group_ids or [])
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def segments_geojson(self) -> dict:
        """Line segments (tbl_segment_flat) as GeoJSON, coloured by the config
        A–E ramp on sensitivity_code_max — the old Results map's 'Sensitivity
        lines' overlay."""
        import json
        import configparser
        try:
            import geopandas as gpd
        except Exception as exc:
            return {"error": str(exc)}
        p = self.gpq / "tbl_segment_flat.parquet"
        if not p.exists():
            return {"type": "FeatureCollection", "features": []}
        try:
            g = gpd.read_parquet(p)
        except Exception as exc:
            return {"error": str(exc)}
        if g.empty:
            return {"type": "FeatureCollection", "features": []}
        cfg = configparser.ConfigParser(inline_comment_prefixes=(";",), strict=False)
        try:
            cfg.read(self.base / "config.ini", encoding="utf-8")
        except Exception:
            pass
        cmap = {c: (cfg[c].get("category_colour") if cfg.has_section(c) else "#cccccc")
                for c in "ABCDE"}
        try:
            if g.crs is not None and "4326" not in str(g.crs):
                g = g.to_crs(4326)
        except Exception:
            pass
        feats = []
        for _, r in g.iterrows():
            geom = r.geometry
            if geom is None or geom.is_empty:
                continue
            code = str(r.get("sensitivity_code_max", "") or "").strip().upper()
            feats.append({"type": "Feature", "geometry": geom.__geo_interface__,
                          "properties": {
                              "name": str(r.get("name_gis", "") or ""),
                              "seg": str(r.get("segment_id", "") or ""),
                              "code": code,
                              "sens": float(r.get("sensitivity_max") or 0),
                              "fill": str(cmap.get(code, "#888888")).strip()}})
        return {"type": "FeatureCollection", "features": feats}

    # -- discovery -----------------------------------------------------------
    def seg_levels(self) -> list[dict]:
        d = self.gpq / "tbl_segmentation"
        if not d.is_dir():
            return []
        import pyarrow.parquet as pq
        out = []
        for p in sorted(d.glob("*.parquet")):
            try:
                n = pq.ParquetFile(p).metadata.num_rows
            except Exception:
                n = 0
            out.append({"name": p.stem, "cells": int(n)})
        out.sort(key=lambda r: (r["name"] != "basic_mosaic", r["name"]))
        return out

    def seg_modes(self, level: str) -> list[str]:
        try:
            return seg.overview_modes(self.gpq, level)
        except Exception:
            return ["signatures"]

    def seg_tile_layers(self, level: str) -> dict:
        """Which seg MBTiles exist for the level. Returns {modes: {signatures:
        {name, bounds, minzoom, maxzoom} | None, clusters: ...}}."""
        modes: dict = {"signatures": None, "clusters": None}
        for mode in ("signatures", "clusters"):
            name = f"{_safe_name(level)}_seg_{mode}"
            path = _mbtiles_dir() / f"{name}.mbtiles"
            if path.exists():
                meta = _mbtiles_meta(path)
                modes[mode] = {"name": name, "bounds": meta["bounds"],
                               "minzoom": meta["minzoom"], "maxzoom": meta["maxzoom"]}
        return {"modes": modes}

    def seg_panel(self, level: str, mode: str) -> dict:
        """Legend + zones table from tbl_segmentation_profiles for (level, mode),
        sorted big→small by area. Fill colours computed from the zone name."""
        import pandas as pd
        prof = self.gpq / "tbl_segmentation_profiles.parquet"
        if not prof.exists():
            return {"zones": []}
        try:
            df = pd.read_parquet(prof)
        except Exception:
            return {"zones": []}
        df = df[df["name_gis_geocodegroup"].astype(str) == str(level)]
        if df.empty:
            return {"zones": []}
        if mode == "signatures":
            df = df[df["method"].astype(str) == "signatures"]
        else:
            df = df[df["method"].astype(str) != "signatures"]
        if df.empty:
            return {"zones": []}
        if "total_area_km2" in df.columns and df["total_area_km2"].notna().any():
            df = df.sort_values("total_area_km2", ascending=False, na_position="last")
        else:
            df = df.sort_values("n_polygons", ascending=False)
        zones = []
        for _, r in df.iterrows():
            zone = str(r["zone"])
            fill = seg._overview_colour(zone, mode) if mode == "clusters" else \
                ("#d2d2d2" if zone in ("", "(no overlap)") else seg._hex(seg._signature_colour(zone)))
            zones.append({
                "zone": zone, "fill": fill,
                "total_area_km2": (float(r["total_area_km2"]) if pd.notna(r.get("total_area_km2")) else None),
                "n_polygons": int(r["n_polygons"]) if pd.notna(r.get("n_polygons")) else 0,
                "sens_mean": float(r["sens_mean"]) if pd.notna(r.get("sens_mean")) else None,
                "mean_n_assets": float(r["mean_n_assets"]) if pd.notna(r.get("mean_n_assets")) else None,
            })
        return {"zones": zones}

    # -- vector fallback (small levels with no tiles) ------------------------
    def seg_needs_build(self, level: str, mode: str) -> bool:
        try:
            cache = seg.overview_cache_path(self.gpq, level, mode)
            part = self.gpq / "tbl_segmentation" / f"{level}.parquet"
            return (not cache.exists()) or (part.exists() and cache.stat().st_mtime < part.stat().st_mtime)
        except Exception:
            return True

    def seg_overview(self, level: str, mode: str) -> dict:
        try:
            cache = seg.overview_cache_path(self.gpq, level, mode)
            part = self.gpq / "tbl_segmentation" / f"{level}.parquet"
            stale = (not cache.exists()) or (part.exists() and cache.stat().st_mtime < part.stat().st_mtime)
            if stale:
                res = seg.build_overview_geojson(self.gpq, level, mode, out_path=cache)
                if res is None:
                    return {"error": f"No segmentation available for {level} / {mode}."}
                if res.get("too_large"):
                    return {"too_large": True, "cells": res["cells"], "max_cells": res["max_cells"]}
                return {"geojson": res["geojson"], "legend": res["legend"]}
            fc = json.loads(cache.read_text(encoding="utf-8"))
            legend = sorted(
                ({"zone": f["properties"]["zone"], "fill": f["properties"]["fill"],
                  "total_area_km2": f["properties"].get("total_area_km2"),
                  "n_polygons": f["properties"].get("n_polygons"),
                  "sens_mean": f["properties"].get("sens_mean"),
                  "mean_n_assets": f["properties"].get("mean_n_assets")}
                 for f in fc.get("features", [])),
                key=lambda d: (d["total_area_km2"] is None, -(d["total_area_km2"] or 0.0)),
            )
            return {"geojson": fc, "legend": legend}
        except Exception as exc:
            return {"error": str(exc)}

    # -- multivariate generalisation (Classifications tab) -------------------
    # Polygon results from "Sensitivity generalisation" (segmentation_run.py):
    # tbl_seg_mv (cluster_id per code) joined to tbl_geocode_object geometry.
    # Rendered as a vector layer — there is no MBTiles pipeline for this output.
    SEGMV_MAX_FEATURES = 250_000

    def segmv_runs(self) -> list[dict]:
        """Available (run_id, method, n_clusters) results from tbl_seg_mv, newest
        run first. Each combination is one selectable Classifications layer."""
        import pandas as pd
        p = self.gpq / "tbl_seg_mv.parquet"
        if not p.exists():
            return []
        try:
            df = pd.read_parquet(p, columns=["run_id", "name_gis_geocodegroup", "method", "n_clusters"])
        except Exception:
            return []
        if df.empty:
            return []
        combos = df.drop_duplicates(["run_id", "name_gis_geocodegroup", "method", "n_clusters"])
        out = []
        for _, r in combos.iterrows():
            rid, layer = str(r["run_id"]), str(r["name_gis_geocodegroup"])
            method, k = str(r["method"]), int(r["n_clusters"])
            out.append({"run_id": rid, "layer": layer, "method": method, "n_clusters": k,
                        "label": f"{rid} · {layer} · {method}"})
        out.sort(key=lambda d: (d["run_id"], d["method"]), reverse=True)
        return out

    def _segmv_profile(self, run_id: str, method: str, k):
        """Profile rows + legend for one result, largest area first."""
        import pandas as pd
        prof = self.gpq / "tbl_seg_mv_profile.parquet"
        if not prof.exists():
            return [], []
        try:
            df = pd.read_parquet(prof)
        except Exception:
            return [], []
        df = df[(df["run_id"].astype(str) == str(run_id)) & (df["method"].astype(str) == str(method))]
        if k is not None and "n_clusters" in df.columns:
            df = df[pd.to_numeric(df["n_clusters"], errors="coerce") == k]
        if df.empty:
            return [], []
        if "total_area_km2" in df.columns and df["total_area_km2"].notna().any():
            df = df.sort_values("total_area_km2", ascending=False, na_position="last")
        else:
            df = df.sort_values("n_polygons", ascending=False)
        rows, legend = [], []
        for _, r in df.iterrows():
            try:
                cid = int(r.get("cluster_id"))
            except Exception:
                cid = -1
            fill = seg._overview_colour(str(cid), "clusters")
            label = str(r.get("cluster_label") or f"type {cid + 1}")
            rows.append({
                "cluster_label": label, "fill": fill,
                "n_polygons": int(r["n_polygons"]) if pd.notna(r.get("n_polygons")) else 0,
                "total_area_km2": float(r["total_area_km2"]) if pd.notna(r.get("total_area_km2")) else None,
                "sens_mean": float(r["sens_mean"]) if pd.notna(r.get("sens_mean")) else None,
                "top_asset_groups": str(r.get("top_asset_groups") or ""),
            })
            legend.append({"zone": label, "fill": fill})
        return rows, legend

    def segmv_layer(self, run_id: str, method: str, n_clusters) -> dict:
        """Polygon GeoJSON + legend + profile for one generalisation result.
        Geometry joined from tbl_geocode_object; fill colour per cluster id
        (qualitative palette, reused from the Segmentation clusters view)."""
        import pandas as pd
        try:
            import geopandas as gpd
        except Exception as exc:
            return {"error": str(exc)}
        p = self.gpq / "tbl_seg_mv.parquet"
        go = self.gpq / "tbl_geocode_object.parquet"
        if not p.exists() or not go.exists():
            return {"error": "No classification results found. Run Classification first."}
        try:
            k = int(n_clusters)
        except Exception:
            k = None
        try:
            df = pd.read_parquet(p)
        except Exception as exc:
            return {"error": str(exc)}
        m = df[(df["run_id"].astype(str) == str(run_id)) & (df["method"].astype(str) == str(method))]
        if k is not None and "n_clusters" in m.columns:
            m = m[pd.to_numeric(m["n_clusters"], errors="coerce") == k]
        m = m[m["cluster_id"].notna()]
        if "no_data" in m.columns:
            m = m[~m["no_data"].astype(bool)]
        if m.empty:
            return {"geojson": {"type": "FeatureCollection", "features": []}, "legend": [], "profile": []}
        if len(m) > self.SEGMV_MAX_FEATURES:
            return {"too_large": True, "features": int(len(m)), "max_features": self.SEGMV_MAX_FEATURES}
        layer = str(m["name_gis_geocodegroup"].iloc[0])
        try:
            geo = gpd.read_parquet(go, columns=["code", "geometry"],
                                   filters=[("name_gis_geocodegroup", "=", layer)])
        except Exception:
            geo = gpd.read_parquet(go, columns=["code", "name_gis_geocodegroup", "geometry"])
            geo = geo[geo["name_gis_geocodegroup"].astype(str) == layer][["code", "geometry"]]
        geo["code"] = geo["code"].astype(str)
        m = m.copy(); m["code"] = m["code"].astype(str)
        gdf = gpd.GeoDataFrame(m.merge(geo, on="code", how="inner"),
                               geometry="geometry", crs=getattr(geo, "crs", None))
        if gdf.empty:
            return {"geojson": {"type": "FeatureCollection", "features": []}, "legend": [], "profile": []}
        try:
            if gdf.crs is not None and "4326" not in str(gdf.crs):
                gdf = gdf.to_crs(4326)
        except Exception:
            pass
        feats = []
        for _, r in gdf.iterrows():
            geom = r.geometry
            if geom is None or geom.is_empty:
                continue
            try:
                cid = int(r.get("cluster_id"))
            except Exception:
                cid = -1
            feats.append({"type": "Feature", "geometry": geom.__geo_interface__,
                          "properties": {
                              "code": str(r.get("code", "") or ""),
                              "cluster_id": cid,
                              "cluster_label": str(r.get("cluster_label") or f"type {cid + 1}"),
                              "sens_mean": float(r.get("sens_mean")) if pd.notna(r.get("sens_mean")) else None,
                              "fill": seg._overview_colour(str(cid), "clusters")}})
        profile, legend = self._segmv_profile(run_id, method, k)
        return {"geojson": {"type": "FeatureCollection", "features": feats},
                "legend": legend, "profile": profile, "layer": layer}

    # -- GetFeatureInfo (click-to-query) -------------------------------------
    # Even though the displayed layers are raster MBTiles, the underlying cell
    # data lives in parquet. A left-click sends lat/lng here; we point-in-polygon
    # against the cells of the *currently displayed* layer and return attributes
    # + geometry for a popup and an outline highlight. Ported from the old
    # map_overview.lookup_tile_info, trimmed to today's tbl_flat schema.
    def _fi_load(self, mode: str, layer: str):
        """Return a 4326 GeoDataFrame of cells for (mode, layer), cached. The
        Results layer reads tbl_flat filtered to the group; Segmentation joins
        tbl_segmentation/<level> to geometry from tbl_geocode_object."""
        key = (mode, str(layer))
        if key in self._fi_cache:
            return self._fi_cache[key]
        import geopandas as gpd
        import pandas as pd
        import pyarrow.parquet as pq
        gdf = None
        try:
            if mode == "results":
                p = self.gpq / "tbl_flat.parquet"
                if not p.exists():
                    self._fi_cache[key] = None
                    return None
                want = ["name_gis_geocodegroup", "code", "geometry",
                        "sensitivity_max", "sensitivity_code_max", "sensitivity_description_max",
                        "importance_max", "importance_code_max", "importance_description_max",
                        "susceptibility_max", "index_owa", "asset_group_names",
                        "asset_groups_total", "assets_overlap_total", "area_m2"]
                have = set(pq.ParquetFile(p).schema_arrow.names)
                cols = [c for c in want if c in have]
                if "geometry" not in cols:
                    cols.append("geometry")
                try:
                    gdf = gpd.read_parquet(p, columns=cols, filters=[("name_gis_geocodegroup", "=", str(layer))])
                except Exception:
                    gdf = gpd.read_parquet(p, columns=cols)
                    gdf = gdf[gdf["name_gis_geocodegroup"].astype(str) == str(layer)]
            else:  # segmentation
                part = self.gpq / "tbl_segmentation" / f"{layer}.parquet"
                go = self.gpq / "tbl_geocode_object.parquet"
                if not part.exists() or not go.exists():
                    self._fi_cache[key] = None
                    return None
                seg_df = pd.read_parquet(part)
                try:
                    geo = gpd.read_parquet(go, columns=["code", "geometry"],
                                           filters=[("name_gis_geocodegroup", "=", str(layer))])
                except Exception:
                    geo = gpd.read_parquet(go, columns=["code", "name_gis_geocodegroup", "geometry"])
                    geo = geo[geo["name_gis_geocodegroup"].astype(str) == str(layer)][["code", "geometry"]]
                gdf = gpd.GeoDataFrame(seg_df.merge(geo, on="code", how="inner"),
                                       geometry="geometry", crs=getattr(geo, "crs", None))
            if gdf is not None and not gdf.empty:
                try:
                    if gdf.crs is not None and "4326" not in str(gdf.crs):
                        gdf = gdf.to_crs(4326)
                except Exception:
                    pass
        except Exception:
            gdf = None
        self._fi_cache[key] = gdf
        return gdf

    def _fi_point_lookup(self, gdf, lng: float, lat: float):
        from shapely.geometry import Point
        pt = Point(float(lng), float(lat))
        try:
            idxs = list(gdf.sindex.query(pt, predicate="intersects"))
        except Exception:
            idxs = []
        for pos in idxs:
            try:
                geom = gdf.geometry.iloc[pos]
            except Exception:
                continue
            if geom is not None and not geom.is_empty and geom.intersects(pt):
                return gdf.iloc[pos]
        # Fallback: linear scan (small layers, or if the spatial index is unavailable).
        try:
            sub = gdf[gdf.geometry.apply(lambda g: g is not None and not g.is_empty and g.intersects(pt))]
            if not sub.empty:
                return sub.iloc[0]
        except Exception:
            pass
        return None

    def query_feature_info(self, lat, lng, mode, layer, overlay=None) -> dict:
        """Point-in-polygon lookup for the cell under a map click.

        mode: 'results' (tbl_flat) or 'seg' (segmentation). layer: geocode group
        / segmentation level name. overlay: results suffix (sensitivity_max, …) or
        seg mode (signatures/clusters) — selects which attributes to surface."""
        try:
            if not layer:
                return {"ok": False, "error": "No active layer."}
            gdf = self._fi_load(str(mode), str(layer))
            if gdf is None or gdf.empty:
                return {"ok": False, "error": "No queryable data for this layer."}
            try:
                row = self._fi_point_lookup(gdf, float(lng), float(lat))
            except Exception:
                return {"ok": False, "error": "Invalid coordinates."}
            if row is None:
                return {"ok": False, "error": "No cell at that location."}
            if str(mode) == "results":
                info = _fi_results_info(row, str(overlay or "sensitivity_max"), str(layer))
            else:
                info = _fi_seg_info(row, str(layer), str(overlay or "signatures"))
            geom = row.geometry
            try:
                info["geometry"] = geom.__geo_interface__ if (geom is not None and not geom.is_empty) else None
            except Exception:
                info["geometry"] = None
            return {"ok": True, "info": _fi_json_ready(info)}
        except Exception as e:
            return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Loopback server: HTML at / and MBTiles tiles at /tiles/<name>/{z}/{x}/{y}.png
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        return

    def _png(self, data: bytes, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except Exception:
            pass

    def _html(self):
        data = (HTML or "").encode("utf-8", errors="replace")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except Exception:
            pass

    def _tile(self):
        con = None
        try:
            parts = [p for p in self.path.split("?", 1)[0].split("/") if p]
            # /tiles/<name>/<z>/<x>/<y>.png  -> 5 path segments
            if len(parts) != 5 or parts[0] != "tiles":
                self._png(BLANK_PNG)
                return
            name = _safe_name(parts[1])
            z = int(parts[2]); x = int(parts[3]); y = int(parts[4].rsplit(".", 1)[0])
            db = _mbtiles_dir() / f"{name}.mbtiles"
            if not db.exists():
                self._png(BLANK_PNG)
                return
            con = sqlite3.connect(f"file:{db.resolve()}?mode=ro&immutable=1", uri=True, timeout=5.0)
            cur = con.cursor()
            tms_row = (1 << z) - 1 - y
            row = cur.execute(
                "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                (z, x, tms_row)).fetchone()
            if not row:
                row = cur.execute(
                    "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                    (z, x, y)).fetchone()
            if not row:
                self._png(BLANK_PNG)
                return
            self._png(row[0])
        except Exception:
            self._png(BLANK_PNG)
        finally:
            if con:
                try:
                    con.close()
                except Exception:
                    pass

    def do_GET(self):
        p = self.path.split("?", 1)[0]
        if p in ("/", "/index.html"):
            self._html()
            return
        if p == "/favicon.ico":
            self.send_response(204); self.end_headers(); return
        if p.startswith("/tiles/"):
            self._tile()
            return
        self.send_response(404); self.end_headers()


def _start_server() -> str:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}"


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8">
<title>MESA maps</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  /* Aligned with the MESA PyQt main window: warm tan/parchment palette
     (bg #f3ecdf, buttons #e6dac2, accent #d9bd7d/#9b7c3d, text #3f3528/#5c4a2f). */
  html,body{height:100%;margin:0;font-family:system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;font-size:13px;color:#3f3528}
  #app{display:flex;flex-direction:column;height:100vh}
  #bar{display:flex;align-items:center;gap:8px;padding:8px 12px;background:#f3ecdf;border-bottom:2px solid #cbb791}
  .tab{background:#e6dac2;color:#5c4a2f;border:1px solid #c6b089;padding:6px 12px;cursor:pointer;border-radius:6px;font-size:13px}
  .tab:hover{background:#eadbbd}
  .tab:active{transform:translateY(1px)}
  .tab.active{background:#d9bd7d;color:#3f3018;border-color:#9b7c3d;font-weight:600}
  #spacer{flex:1}
  #opwrap{display:flex;align-items:center;gap:6px;margin-right:12px;font-size:12px;color:#3f3528}
  #opacity{width:120px;accent-color:#9b7c3d}
  #linkwrap{display:flex;align-items:center;gap:6px;margin-right:8px;font-size:12px;color:#3f3528}
  .info{font-size:11px;color:#5c4a2f;background:#f3ecdf;border:1px solid #e3d7be;border-radius:5px;padding:8px;margin:8px 0;line-height:1.35}
  .info p{margin:0 0 5px}
  #exit,#export{background:#e6dac2;color:#5c4a2f;border:1px solid #c6b089;padding:6px 12px;border-radius:6px;cursor:pointer}
  #exit:hover,#export:hover{background:#eadbbd}
  #exit:active,#export:active{transform:translateY(1px)}
  .map.exporting .leaflet-control-zoom{display:none !important}
  #views{position:relative;flex:1}
  .view{position:absolute;inset:0;display:none}
  .view.active{display:block}
  .map{position:absolute;inset:0;background:#ddd}
  #view-seg.active, #view-results.active, #view-asset.active, #view-class.active{display:grid;grid-template-columns:1fr 340px}
  .map-wrap{position:relative;height:100%}
  /* floating layer-control panel over the map (frees the right column) */
  .mapctl{position:absolute;top:10px;right:10px;z-index:1000;background:rgba(250,246,238,.96);
          border:1px solid #cbb791;border-radius:6px;padding:8px 10px;max-width:240px;font-size:12px;
          box-shadow:0 1px 6px rgba(0,0,0,.25);color:#3f3528}
  .mapctl select{width:100%;margin-top:2px}
  .ctlrow{margin:5px 0}
  .mapctl-hd{display:flex;justify-content:space-between;align-items:center;cursor:pointer;
             font-weight:600;color:#5c4a2f;user-select:none}
  .mapctl-arrow{font-size:13px;line-height:1}
  .mapctl.collapsed .mapctl-body{display:none}
  .mapctl.collapsed{max-width:none}
  #seg-panel, #res-panel, #asset-panel, #class-panel{border-left:2px solid #cbb791;padding:12px;overflow:auto;background:#faf6ee}
  #seg-panel h2, #res-panel h2, #asset-panel h2, #class-panel h2{font-size:14px;margin:0 0 8px;color:#5c4a2f}
  .assetlist{max-height:240px;overflow:auto;margin:4px 0}
  .assetlist label{display:flex;align-items:center;gap:6px;margin:2px 0;font-size:12px;cursor:pointer}
  .assetlist .sw{flex:0 0 auto}
  .pbtn{background:#e6dac2;color:#5c4a2f;border:1px solid #c6b089;padding:5px 10px;border-radius:6px;cursor:pointer;font-size:12px;margin-right:6px}
  .pbtn:hover{background:#eadbbd}
  .pbtn:active{transform:translateY(1px)}
  .row{margin:8px 0}
  select{width:100%;padding:4px;background:#fff;border:1px solid #c6b089;border-radius:4px;color:#3f3528}
  table{border-collapse:collapse;width:100%;font-size:12px;margin-top:6px}
  th,td{border-bottom:1px solid #e3d7be;padding:3px 4px;text-align:right}
  th:first-child,td:first-child{text-align:left}
  th{color:#715a36;font-weight:600}
  #segZones thead th, #classTable thead th{cursor:pointer;user-select:none;white-space:nowrap}
  #segZones thead th:hover, #classTable thead th:hover{color:#3f3528}
  tfoot td{font-weight:600;border-top:1px solid #cbb791}
  .sw{display:inline-block;width:12px;height:12px;border:1px solid #b9a87f;vertical-align:middle;margin-right:6px}
  .muted{color:#8a7c63}
  .stub{position:absolute;top:14px;left:14px;z-index:500;background:#fff;border:1px solid #ccc;
        border-radius:6px;padding:8px 12px;box-shadow:0 1px 6px rgba(0,0,0,.15)}
  #loading{position:absolute;top:50%;left:40%;background:#fff;border:1px solid #ccc;border-radius:6px;
           padding:10px 16px;box-shadow:0 2px 8px rgba(0,0,0,.2);display:none;z-index:1000}
  #segMsg{font-size:12px;color:#666;margin-top:6px}
</style></head>
<body>
<div id="app">
  <div id="bar">
    <button class="tab active" data-tab="results">Overview</button>
    <button class="tab" data-tab="seg">Segmentation</button>
    <button class="tab" data-tab="class">Classifications</button>
    <button class="tab" data-tab="asset">Assets</button>
    <span id="spacer"></span>
    <label id="opwrap">Opacity <input type="range" id="opacity" min="0" max="100" value="85"></label>
    <label id="linkwrap"><input type="checkbox" id="linkChk"> Link zoom &amp; pan</label>
    <button id="export" title="Export the current tab's map to PNG">Export PNG</button>
    <button id="exit">Exit</button>
  </div>
  <div id="views">
    <div class="view" id="view-asset">
      <div class="map-wrap">
        <div class="map" id="map-asset"></div>
        <div class="mapctl">
          <div class="mapctl-hd"><span>Layers</span><span class="mapctl-arrow">▾</span></div>
          <div class="mapctl-body">
            <div class="ctlrow"><b>Asset groups</b> <span class="muted">(none selected)</span></div>
            <div id="assetList" class="assetlist muted">…</div>
            <div class="ctlrow"><b>Basemap</b><br>
              <select id="assetBasemap">
                <option value="osm">OpenStreetMap</option>
                <option value="topo">OpenTopoMap</option>
                <option value="sat">Satellite (Esri)</option>
              </select></div>
            <div id="assetMsg" class="muted"></div>
          </div>
        </div>
      </div>
      <div id="asset-panel">
        <h2>Assets</h2>
        <div class="info">
          <p>Toggle <b>asset groups</b> on the map to load them. Nothing is loaded by default — assets
             can be large, so each group is fetched only when you select it.</p>
          <p>Use the <b>Opacity</b> slider in the header to blend with the basemap.</p>
        </div>
        <div class="row"><b>Cartography</b><br>
          <button id="aiStyleBtn" class="pbtn" title="Generate visually distinct colours per asset group">AI styling</button>
          <button id="clearStyleBtn" class="pbtn" title="Reset to default / sensitivity colours">Clear styling</button>
          <div id="styleMsg" class="muted" style="margin-top:4px"></div>
        </div>
        <div id="assetLegend"></div>
      </div>
    </div>
    <div class="view active" id="view-results">
      <div class="map-wrap">
        <div class="map" id="map-results"></div>
        <div class="mapctl">
          <div class="mapctl-hd"><span>Layers</span><span class="mapctl-arrow">▾</span></div>
          <div class="mapctl-body">
            <div class="ctlrow"><b>Geocode group</b><br><select id="resGroup"></select></div>
            <div class="ctlrow"><b>Layer</b> <span class="muted">(one at a time)</span><br><span id="resKinds"></span></div>
            <label class="ctlrow"><input type="checkbox" id="resSegChk"> Line segments</label>
            <div class="ctlrow"><b>Basemap</b><br>
              <select id="resBasemap">
                <option value="osm">OpenStreetMap</option>
                <option value="topo">OpenTopoMap</option>
                <option value="sat">Satellite (Esri)</option>
              </select></div>
            <div id="resMsg" class="muted"></div>
          </div>
        </div>
      </div>
      <div id="res-panel">
        <h2>Overview</h2>
        <div class="info" id="resInfo">
          <p>Pick a <b>geocode group</b> and <b>layer</b> on the map. <b>Sensitive areas (A–E)</b> shows the
             highest sensitivity class per cell (worst-case). The <b>OWA index</b>
             layer is a 0–100 ranking <i>within the selected group</i>. <b># asset groups</b> = diversity,
             <b># asset objects</b> = density per cell. <b>Line segments</b> overlays the line sensitivity.</p>
          <p>Use the <b>Opacity</b> slider in the header to blend a layer with the basemap.</p>
        </div>
        <div class="row"><b>Totals by sensitivity</b> <span class="muted">(basic_mosaic)</span>
          <div id="resStatsTable" class="muted">–</div>
          <div id="resChartBox" style="height:200px;margin-top:8px"><canvas id="resChart"></canvas></div>
        </div>
        <div class="row muted" style="font-size:11px">The Asset map is a separate window.</div>
      </div>
    </div>
    <div class="view" id="view-seg">
      <div class="map-wrap">
        <div class="map" id="map-seg"></div>
        <div id="loading">Building view…</div>
        <div class="mapctl">
          <div class="mapctl-hd"><span>Layers</span><span class="mapctl-arrow">▾</span></div>
          <div class="mapctl-body">
            <div class="ctlrow"><b>Level</b><br><select id="segLevel"></select></div>
            <div class="ctlrow"><b>Mode</b><br><span id="segModes"></span></div>
            <div class="ctlrow"><b>Basemap</b><br>
              <select id="segBasemap">
                <option value="osm">OpenStreetMap</option>
                <option value="topo">OpenTopoMap</option>
                <option value="sat">Satellite (Esri)</option>
              </select></div>
            <div id="segMsg" class="muted"></div>
          </div>
        </div>
      </div>
      <div id="seg-panel">
        <h2>Segmentation</h2>
        <div class="row"><b>Legend</b><div id="segLegend" class="muted">–</div></div>
        <div class="row"><b>Zones</b> <span class="muted">(largest area first)</span>
          <table id="segZones"><thead><tr><th data-key="zone">Zone</th><th data-key="total_area_km2">Area km²</th><th data-key="n_polygons">Cells</th><th data-key="sens_mean">Sens</th><th data-key="mean_n_assets">Assets</th></tr></thead>
          <tbody></tbody></table>
        </div>
      </div>
    </div>
    <div class="view" id="view-class">
      <div class="map-wrap">
        <div class="map" id="map-class"></div>
        <div id="classLoading" class="stub" style="display:none">Building view…</div>
        <div class="mapctl">
          <div class="mapctl-hd"><span>Layers</span><span class="mapctl-arrow">▾</span></div>
          <div class="mapctl-body">
            <div class="ctlrow"><b>Run</b><br><select id="classRun"></select></div>
            <div class="ctlrow"><b>Basemap</b><br>
              <select id="classBasemap">
                <option value="osm">OpenStreetMap</option>
                <option value="topo">OpenTopoMap</option>
                <option value="sat">Satellite (Esri)</option>
              </select></div>
            <div id="classMsg" class="muted"></div>
          </div>
        </div>
      </div>
      <div id="class-panel">
        <h2>Classifications</h2>
        <div class="info">
          <p>Polygon results from <b>Classification</b>. Each colour is a
             <b>type</b> of sensitivity pattern — <i>“what kind of place is this?”</i> —
             complementary to the A–E sensitivity classes. Pick a run on the map, then
             click a polygon to identify it.</p>
        </div>
        <div class="row"><b>Legend</b><div id="classLegend" class="muted">–</div></div>
        <div class="row"><b>Types</b> <span class="muted">(largest area first)</span>
          <table id="classTable"><thead><tr><th data-key="cluster_label">Type</th><th data-key="total_area_km2">Area km²</th><th data-key="n_polygons">Cells</th><th data-key="sens_mean">Sens</th><th data-key="top_asset_groups" style="text-align:left">Top asset groups</th></tr></thead>
          <tbody></tbody></table>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
  function api(){ return window.pywebview.api; }
  // Devtools hook: ?tab=seg&sort=<col> lets the screenshot tool open a tab and
  // pre-sort the Zones table so the feature is visible without OS-level clicks.
  // See devtools/capture_ui_active_batch.py.
  var _qp=new URLSearchParams(location.search);
  var START_TAB=_qp.get('tab')||'';
  var START_SORT=_qp.get('sort')||'';
  // Basemaps (same set as the old Results map).
  var BASEMAPS=[
    {id:'osm', label:'OpenStreetMap', url:'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
     opts:{maxZoom:19, attribution:'© OpenStreetMap'}},
    {id:'topo', label:'OpenTopoMap', url:'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
     opts:{maxZoom:17, subdomains:['a','b','c'], attribution:'© OpenStreetMap, © OpenTopoMap (CC-BY-SA)'}},
    {id:'sat', label:'Satellite (Esri)',
     url:'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
     opts:{maxZoom:19, attribution:'Esri, Maxar, Earthstar Geographics'}}
  ];
  function mkMap(id, opts){ opts=opts||{}; opts.zoomControl=true; var m=L.map(id,opts); m.setView([0,0],2); return m; }
  var maps = {asset:mkMap('map-asset',{preferCanvas:true}), results:mkMap('map-results'), seg:mkMap('map-seg'), class:mkMap('map-class',{preferCanvas:true})};
  var baseLayer={};
  function setBasemap(key, id){
    var def=BASEMAPS[0], i;
    for(i=0;i<BASEMAPS.length;i++){ if(BASEMAPS[i].id===id){ def=BASEMAPS[i]; break; } }
    if(baseLayer[key]) maps[key].removeLayer(baseLayer[key]);
    var o={}; for(var k in def.opts) o[k]=def.opts[k];
    baseLayer[key]=L.tileLayer(def.url, o).addTo(maps[key]);
    baseLayer[key].bringToBack();
  }
  setBasemap('asset','osm'); setBasemap('results','osm'); setBasemap('seg','osm'); setBasemap('class','osm');
  (function(){
    var rb=document.getElementById('resBasemap'); if(rb) rb.addEventListener('change', function(){ setBasemap('results', this.value); });
    var sb=document.getElementById('segBasemap'); if(sb) sb.addEventListener('change', function(){ setBasemap('seg', this.value); });
    var ab=document.getElementById('assetBasemap'); if(ab) ab.addEventListener('change', function(){ setBasemap('asset', this.value); });
    var cb=document.getElementById('classBasemap'); if(cb) cb.addEventListener('change', function(){ setBasemap('class', this.value); });
  })();
  var segVec = L.geoJSON(null).addTo(maps.seg);   // vector fallback layer
  var segVecGj = null;                            // inner styled vector layer (for opacity)
  var segTile = null;                              // raster tile layer
  var resTile = null;                              // overview index raster layer
  var segOverlay = null;                           // line-segments vector overlay (Overview tab)
  var classVecGj = null;                           // Classifications (seg_mv) vector layer (for opacity)
  var assetLayers = {};                            // gid -> asset L.geoJSON layer
  var resKind = null;                              // active Overview overlay suffix (for click-to-query)
  var fiHighlight = null;                          // GetFeatureInfo: clicked-cell outline overlay
  var fiHighlightMap = null;                       // map the highlight currently lives on
  var current = 'results';
  var opacity = 0.85;                             // shared layer opacity (header slider)

  function applyOpacity(){
    if(resTile) resTile.setOpacity(opacity);
    if(segTile) segTile.setOpacity(opacity);
    if(segVecGj) segVecGj.setStyle({fillOpacity: opacity*0.9});
    if(segOverlay) segOverlay.setStyle({fillOpacity: opacity*0.9});
    if(classVecGj) classVecGj.setStyle({fillOpacity: opacity*0.9});
    for(var k in assetLayers){ if(assetLayers[k]) assetLayers[k].setStyle({fillOpacity: opacity*0.9}); }
  }
  // ---- collapsible floating layer panels ----
  document.querySelectorAll('.mapctl-hd').forEach(function(hd){
    hd.addEventListener('click', function(){
      var box=hd.parentElement;
      var collapsed=box.classList.toggle('collapsed');
      var arrow=hd.querySelector('.mapctl-arrow');
      if(arrow) arrow.textContent = collapsed ? '▸' : '▾';
    });
  });

  // ---- line segments overlay (Overview tab) ----
  document.getElementById('resSegChk').addEventListener('change', function(){
    if(segOverlay){ maps.results.removeLayer(segOverlay); segOverlay=null; }
    if(!this.checked) return;
    api().segments_geojson().then(function(fc){
      if(fc.error || !fc.features || !fc.features.length){
        document.getElementById('resMsg').textContent = 'No line segments found (run Lines processing).';
        return;
      }
      segOverlay=L.geoJSON(fc, {
        style:function(f){ return {color:'#333',weight:0.6,fillColor:f.properties.fill,fillOpacity:opacity*0.9}; },
        onEachFeature:function(f,l){ var p=f.properties;
          l.bindPopup('<b>'+(p.name||'line')+'</b> · seg '+p.seg+'<br>Sensitivity: '+(p.code||'–')+' ('+p.sens+')'); }
      }).addTo(maps.results);
      try{ maps.results.fitBounds(segOverlay.getBounds(),{padding:[20,20]}); if(linking) syncFrom('results'); }catch(e){}
    });
  });
  document.getElementById('opacity').addEventListener('input', function(){
    opacity = (this.value||0)/100; applyOpacity();
  });

  // ---- linked zoom/pan ----
  var linking=false, syncing=false;
  function syncFrom(srcKey){
    if(!linking || syncing) return;
    syncing=true;
    try{
      var src=maps[srcKey], c=src.getCenter(), z=src.getZoom();
      Object.keys(maps).forEach(function(k){
        if(k!==srcKey){ try{ maps[k].setView(c,z,{animate:false}); }catch(e){} }
      });
    } finally { syncing=false; }   // never leave the guard stuck (would block all linking)
  }
  Object.keys(maps).forEach(function(k){ maps[k].on('moveend zoomend', function(){ if(k===current) syncFrom(k); }); });
  document.getElementById('linkChk').addEventListener('change', function(){ linking=this.checked; if(linking) syncFrom(current); });

  // ---- GetFeatureInfo: left-click a raster cell to identify it ----
  // The map layers are raster MBTiles, so the cell attributes are fetched from
  // the source parquet via the Python bridge (query_feature_info) on each click.
  var fiPopup = null;
  function fiClear(){
    if(fiPopup){ try{ fiPopup.remove(); }catch(e){} fiPopup=null; }
    if(fiHighlight && fiHighlightMap){ try{ fiHighlightMap.removeLayer(fiHighlight); }catch(e){} }
    fiHighlight=null; fiHighlightMap=null;
  }
  function fiEsc(s){ return String(s==null?'':s).replace(/[&<>]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;'}[c];}); }
  function fiFmt(v){ return (typeof v==='number') ? v.toLocaleString(undefined,{maximumFractionDigits:3}) : fiEsc(v); }
  function fiPopupHtml(info){
    var h='<div style="min-width:140px"><div style="font-weight:600;font-size:13px;margin-bottom:1px">'+fiEsc(info.title||'Cell')+'</div>';
    if(info.subtitle) h+='<div style="color:#777;font-size:11px;margin-bottom:5px">'+fiEsc(info.subtitle)+'</div>';
    var m=info.metrics||[];
    if(m.length){ h+='<table style="border-collapse:collapse;font-size:12px">';
      for(var i=0;i<m.length;i++){
        h+='<tr><td style="padding:1px 10px 1px 0;color:#555;vertical-align:top">'+fiEsc(m[i].label)+
           '</td><td style="padding:1px 0;text-align:right;font-weight:500">'+fiFmt(m[i].value)+(m[i].unit?(' '+fiEsc(m[i].unit)):'')+'</td></tr>';
      }
      h+='</table>'; }
    h+='</div>'; return h;
  }
  function fiShow(mapKey, latlng, info){
    var m=maps[mapKey];
    fiClear();
    if(info.geometry){
      try{ fiHighlight=L.geoJSON(info.geometry,{interactive:false,
            style:{color:'#1d4ed8',weight:2,fill:false,opacity:0.95}}).addTo(m); fiHighlightMap=m; }catch(e){}
    }
    fiPopup=L.popup({maxWidth:340,autoPan:true}).setLatLng(latlng).setContent(fiPopupHtml(info)).openOn(m);
  }
  function fiQuery(mapKey, latlng, mode, layer, overlay){
    if(!layer) return;
    api().query_feature_info(latlng.lat, latlng.lng, mode, layer, overlay).then(function(res){
      if(res && res.ok && res.info){ fiShow(mapKey, latlng, res.info); }
      else {
        fiClear();
        fiPopup=L.popup({maxWidth:240}).setLatLng(latlng)
          .setContent('<div style="color:#777;font-size:11px">'+fiEsc((res&&res.error)||'No cell here.')+'</div>')
          .openOn(maps[mapKey]);
      }
    }).catch(function(){});
  }
  // Results raster → tbl_flat for the active group + overlay.
  maps.results.on('click', function(e){ if(resTile && resGroup && resKind) fiQuery('results', e.latlng, 'results', resGroup, resKind); });
  // Segmentation raster → the level's cells. The vector fallback keeps its own per-zone popups.
  maps.seg.on('click', function(e){ if(segTile && segLevel && segMode) fiQuery('seg', e.latlng, 'seg', segLevel, segMode); });

  // ---- tabs ----
  function showTab(tab){
    var prev=current;                 // capture BEFORE switching
    current=tab;
    fiClear();                        // drop any feature popup/highlight from the previous tab
    document.querySelectorAll('.tab').forEach(function(b){ b.classList.toggle('active', b.dataset.tab===tab); });
    document.querySelectorAll('.view').forEach(function(v){ v.classList.remove('active'); });
    document.getElementById('view-'+tab).classList.add('active');
    var m=maps[tab];
    setTimeout(function(){
      m.invalidateSize();
      // When linked, copy the view from the PREVIOUSLY active map to this one.
      if(linking && prev && prev!==tab){
        syncing=true;
        m.setView(maps[prev].getCenter(), maps[prev].getZoom(), {animate:false});
        syncing=false;
      }
    }, 60);
    if(tab==='seg' && !segLoaded) initSeg();
    if(tab==='results' && !resLoaded) initResults();
    if(tab==='asset' && !assetLoaded) initAssets();
    if(tab==='class' && !classLoaded) initClass();
  }
  document.querySelectorAll('.tab').forEach(function(b){ b.addEventListener('click', function(){ showTab(this.dataset.tab); }); });
  document.getElementById('exit').addEventListener('click', function(){
    if(window.pywebview && window.pywebview.api && window.pywebview.api.exit_app){ window.pywebview.api.exit_app(); }
  });
  document.getElementById('export').addEventListener('click', function(){
    var node=document.getElementById('map-'+current);   // active tab's map
    if(!node || typeof html2canvas==='undefined') return;
    node.classList.add('exporting');
    html2canvas(node,{useCORS:true,allowTaint:false,backgroundColor:'#ffffff',scale:2})
      .then(function(canvas){ node.classList.remove('exporting'); return window.pywebview.api.save_png(canvas.toDataURL('image/png')); })
      .catch(function(e){ node.classList.remove('exporting'); });
  });

  // ---- assets tab (lazy; nothing selected by default) ----
  var assetLoaded=false, assetGroups={}, assetSwatches={};
  function initAssets(){
    assetLoaded=true;
    api().asset_groups().then(function(list){
      var box=document.getElementById('assetList'); box.innerHTML=''; assetGroups={}; assetSwatches={};
      if(!list || !list.length){ box.innerHTML='<span class="muted">No asset groups found.</span>'; return; }
      box.classList.remove('muted');
      list.forEach(function(g){
        assetGroups[g.gid]=g;
        var lab=document.createElement('label');
        var cnt=(g.count!=null)?(' ('+Number(g.count).toLocaleString()+')'):'';
        lab.innerHTML='<input type="checkbox"><span class="sw" style="background:'+g.color+'"></span>'+
                      g.label+'<span class="muted">'+cnt+'</span>';
        var cb=lab.querySelector('input');
        assetSwatches[g.gid]=lab.querySelector('.sw');
        cb.addEventListener('change', function(){ toggleAsset(g, this.checked, this); });
        box.appendChild(lab);
      });
    });
  }
  // Re-read effective colours (after styling/clear) and apply to swatches + layers.
  function refreshAssetColors(){
    api().asset_groups().then(function(list){
      (list||[]).forEach(function(g){
        if(assetGroups[g.gid]) assetGroups[g.gid].color=g.color;
        if(assetSwatches[g.gid]) assetSwatches[g.gid].style.background=g.color;
        if(assetLayers[g.gid]) assetLayers[g.gid].setStyle({fillColor:g.color});
      });
    });
  }
  (function(){
    var ai=document.getElementById('aiStyleBtn'), cl=document.getElementById('clearStyleBtn'),
        msg=document.getElementById('styleMsg');
    function gids(){ return Object.keys(assetGroups); }
    if(ai) ai.addEventListener('click', function(){
      var ids=gids(); if(!ids.length){ if(msg) msg.textContent='No asset groups to style.'; return; }
      ai.disabled=true; if(msg) msg.textContent='Generating styles…';
      api().generate_ai_styles(ids).then(function(res){
        ai.disabled=false;
        if(!res || !res.ok){ if(msg) msg.textContent='Styling failed: '+((res&&res.error)||''); return; }
        refreshAssetColors();
        if(msg) msg.textContent='Re-styled '+Object.keys(res.styles||{}).length+' groups'+(res.mode==='local'?' (local cartography).':'.');
      });
    });
    if(cl) cl.addEventListener('click', function(){
      var ids=gids(); if(!ids.length){ return; }
      cl.disabled=true; if(msg) msg.textContent='Clearing styles…';
      api().clear_asset_styles(ids).then(function(res){
        cl.disabled=false; refreshAssetColors();
        if(msg) msg.textContent=(res&&res.ok)?'Styles cleared.':'Clear failed.';
      });
    });
  })();
  function toggleAsset(g, on, cb){
    var gid=g.gid;
    if(!on){ if(assetLayers[gid]){ maps.asset.removeLayer(assetLayers[gid]); delete assetLayers[gid]; } return; }
    if(cb) cb.disabled=true;
    document.getElementById('assetMsg').textContent='Loading '+g.label+'…';
    api().asset_layer(gid).then(function(fc){
      if(cb) cb.disabled=false;
      document.getElementById('assetMsg').textContent='';
      if(fc.error || !fc.features || !fc.features.length){
        document.getElementById('assetMsg').textContent=(fc.error||(g.label+': no geometry.'));
        if(cb) cb.checked=false; return;
      }
      var lyr=L.geoJSON(fc, {
        style:function(){ return {color:'#444',weight:0.4,fillColor:g.color,fillOpacity:opacity*0.9}; },
        onEachFeature:function(f,l){ l.bindPopup('<b>'+g.label+'</b><br>id '+(f.properties.id||'')); }
      }).addTo(maps.asset);
      assetLayers[gid]=lyr;
      try{ maps.asset.invalidateSize(); maps.asset.fitBounds(lyr.getBounds(),{padding:[20,20]}); if(linking) syncFrom('asset'); }catch(e){}
    });
  }

  // ---- overview (index MBTiles) tab ----
  var resLoaded=false, resGroup=null, resCatalog={};
  var KIND_ORDER=['sensitivity_max','importance_max','index_owa','groupstotal','assetstotal'];
  var KIND_DESC={
    sensitivity_max:'Highest sensitivity class (A–E) among assets overlapping each cell — a worst-case view.',
    importance_max:'Highest importance class (1–5) per cell.',
    index_owa:'OWA-weighted index 0–100, ranked within this geocode group.',
    groupstotal:'Count of distinct asset groups per cell (diversity).',
    assetstotal:'Count of asset objects per cell (density).'
  };
  function showResLayer(group, suf){
    var info=((resCatalog[group]||{}).kinds||{})[suf];
    if(!info) return;
    resKind=suf;                       // remember active overlay for click-to-query
    fiClear();                         // a new layer invalidates the previous popup/highlight
    if(resTile){ maps.results.removeLayer(resTile); resTile=null; }
    resTile=L.tileLayer('/tiles/'+info.name+'/{z}/{x}/{y}.png',
      {opacity:opacity, maxNativeZoom:(info.maxzoom||14), minNativeZoom:(info.minzoom||0), maxZoom:19, tms:false}).addTo(maps.results);
    if(info.bounds && info.bounds.length===4){
      try{ maps.results.fitBounds([[info.bounds[1],info.bounds[0]],[info.bounds[3],info.bounds[2]]],{padding:[20,20]}); if(linking) syncFrom('results'); }catch(e){}
    }
    var d=KIND_DESC[suf]||'';
    document.getElementById('resMsg').innerHTML='<b>'+info.label+'</b> — '+group+(d?('<br>'+d):'')+
      '<br><span class="muted">Click a cell to identify it.</span>';
  }
  function setResKinds(group){
    var box=document.getElementById('resKinds'); box.innerHTML='';
    var kinds=(resCatalog[group]||{}).kinds||{};
    var avail=KIND_ORDER.filter(function(k){ return kinds[k]; });
    if(!avail.length){ box.innerHTML='<span class="muted">no layers</span>'; return; }
    avail.forEach(function(k,i){
      var lab=document.createElement('label'); lab.style.display='block'; lab.style.margin='2px 0';
      lab.innerHTML='<input type="radio" name="reskind" value="'+k+'" '+(i===0?'checked':'')+'> '+kinds[k].label;
      box.appendChild(lab);
    });
    box.querySelectorAll('input[name=reskind]').forEach(function(r){
      r.addEventListener('change', function(){ showResLayer(group, this.value); });
    });
    showResLayer(group, avail[0]);
  }
  var resChartObj=null;
  function _km2(v){ return Number(v||0).toLocaleString(undefined,{maximumFractionDigits:1}); }
  function renderAreaStats(){
    api().area_stats().then(function(s){
      var labels=s.labels||[], values=s.values||[], colors=s.colors||[], descs=s.descriptions||[];
      var total=s.total||values.reduce(function(a,b){return a+(b||0);},0);
      var tableEl=document.getElementById('resStatsTable');
      if(!labels.length){ tableEl.innerHTML='<span class="muted">no area stats</span>'; return; }
      var h='<table><thead><tr><th></th><th>Code</th><th>Description</th><th>Area km²</th><th>Share</th></tr></thead><tbody>';
      for(var i=0;i<labels.length;i++){
        var p = total>0 ? (values[i]/total*100) : 0;
        h+='<tr><td><span class="sw" style="background:'+(colors[i]||'#999')+'"></span></td><td>'+labels[i]+
           '</td><td>'+(descs[i]||'')+'</td><td>'+_km2(values[i])+'</td><td>'+p.toFixed(1)+'%</td></tr>';
      }
      h+='</tbody><tfoot><tr><td></td><td></td><td>Total</td><td>'+_km2(total)+'</td><td>100%</td></tr></tfoot></table>';
      tableEl.innerHTML=h;
      if(typeof Chart!=='undefined'){
        var ctx=document.getElementById('resChart').getContext('2d');
        if(resChartObj) resChartObj.destroy();
        resChartObj=new Chart(ctx,{type:'bar',
          data:{labels:labels,datasets:[{data:values,backgroundColor:colors,borderColor:'#fff',borderWidth:1}]},
          options:{responsive:true,maintainAspectRatio:false,animation:false,
            plugins:{legend:{display:false},title:{display:true,text:'Area by sensitivity (km²)'}},
            scales:{y:{beginAtZero:true}}}});
      }
    });
  }
  function initResults(){
    resLoaded=true;
    renderAreaStats();
    api().mbtiles_catalog().then(function(cat){
      resCatalog=cat.groups||{};
      var names=Object.keys(resCatalog);
      var sel=document.getElementById('resGroup');
      if(!names.length){ document.getElementById('resMsg').innerHTML='<span class="muted">No map tiles found. Run the <b>Tiles</b> stage to generate them.</span>'; return; }
      names.sort(function(a,b){ return (a!=='basic_mosaic')-(b!=='basic_mosaic') || a.localeCompare(b); });
      names.forEach(function(g){ var o=document.createElement('option'); o.value=g; o.text=g; sel.appendChild(o); });
      sel.addEventListener('change', function(){ resGroup=this.value; setResKinds(resGroup); });
      resGroup=names[0]; setResKinds(resGroup);
    });
  }

  // ---- segmentation tab ----
  var segLoaded=false, segLevel=null, segMode=null, segCells={};
  var WARN_CELLS=1000000, HARD_CAP=2000000;
  function fmt(v,d){ return (v===null||v===undefined)?'–':Number(v).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d}); }
  function setLoading(on){ document.getElementById('loading').style.display=on?'block':'none'; }
  function clearSegLayers(){ if(segTile){ maps.seg.removeLayer(segTile); segTile=null; } segVec.clearLayers(); segVecGj=null; fiClear(); }
  function setMsg(t){ document.getElementById('segMsg').innerHTML=t||''; }

  // Zone table sort state. Legend keeps the server order (largest area first);
  // only the table reorders when a header is clicked.
  var segZonesData=[], segSort={key:null, dir:1};

  function renderPanel(zones){
    var leg=document.getElementById('segLegend'); leg.innerHTML='';
    segZonesData = zones ? zones.slice() : [];
    document.querySelector('#segZones tbody').innerHTML='';
    if(!zones || !zones.length){ leg.innerHTML='<span class="muted">–</span>'; renderSegRows(); return; }
    zones.forEach(function(z){
      var d=document.createElement('div'); d.innerHTML='<span class="sw" style="background:'+z.fill+'"></span>'+z.zone; leg.appendChild(d);
    });
    // Devtools: apply a one-time initial sort so the screenshot shows the arrow.
    if(START_SORT && !segSort.key){ segSort={key:START_SORT, dir:(START_SORT==='zone')?1:-1}; }
    renderSegRows();
  }

  function renderSegRows(){
    var tb=document.querySelector('#segZones tbody'); tb.innerHTML='';
    var rows=segZonesData.slice();
    if(segSort.key){
      var k=segSort.key, dir=segSort.dir;
      rows.sort(function(a,b){
        var va=a[k], vb=b[k];
        if(k==='zone'){ return String(va==null?'':va).localeCompare(String(vb==null?'':vb))*dir; }
        va=(va==null||isNaN(va))?-Infinity:Number(va); vb=(vb==null||isNaN(vb))?-Infinity:Number(vb);
        return (va<vb?-1:va>vb?1:0)*dir;
      });
    }
    rows.forEach(function(z){
      var tr=document.createElement('tr');
      tr.innerHTML='<td>'+z.zone+'</td><td>'+fmt(z.total_area_km2,2)+'</td><td>'+fmt(z.n_polygons,0)+'</td><td>'+fmt(z.sens_mean,1)+'</td><td>'+fmt(z.mean_n_assets,1)+'</td>';
      tb.appendChild(tr);
    });
    document.querySelectorAll('#segZones thead th').forEach(function(th){
      var base=th.getAttribute('data-label');
      if(base===null){ base=th.textContent; th.setAttribute('data-label', base); }
      th.textContent = (th.dataset.key===segSort.key) ? base+(segSort.dir>0?' ▲':' ▼') : base;
    });
  }

  (function(){
    var thead=document.querySelector('#segZones thead');
    if(!thead) return;
    thead.addEventListener('click', function(e){
      var th=e.target.closest('th'); if(!th || !th.dataset.key) return;
      var k=th.dataset.key;
      // Toggle direction on the active column; new column starts ascending for
      // the zone name, descending for the numeric metrics (largest first).
      if(segSort.key===k){ segSort.dir=-segSort.dir; }
      else { segSort.key=k; segSort.dir=(k==='zone')?1:-1; }
      renderSegRows();
    });
  })();

  function showTiles(info, level, mode){
    clearSegLayers();
    var url='/tiles/'+info.name+'/{z}/{x}/{y}.png';
    var opts={opacity:opacity, maxNativeZoom:(info.maxzoom||14), minNativeZoom:(info.minzoom||0), maxZoom:19, tms:false};
    segTile=L.tileLayer(url, opts).addTo(maps.seg);
    if(info.bounds && info.bounds.length===4){
      var b=[[info.bounds[1],info.bounds[0]],[info.bounds[3],info.bounds[2]]];
      // invalidateSize first: the seg map is hidden until its tab is shown, so the
      // container can still report a stale size when this async chain resolves —
      // fitBounds against a 0-sized map yields the wrong zoom.
      try{ maps.seg.invalidateSize(); maps.seg.fitBounds(b,{padding:[20,20]}); if(linking) syncFrom('seg'); }catch(e){}
    }
    setMsg('Raster tiles ('+info.name+'). <span class="muted">Click a cell to identify it.</span>');
    api().seg_panel(level, mode).then(function(p){ renderPanel(p.zones||[]); });
  }

  function renderVector(res){
    clearSegLayers();
    if(res.error){ setMsg(res.error); return; }
    if(res.too_large){ setMsg('Level too large for the vector fallback (~'+Number(res.cells).toLocaleString()+
       ' cells). Run the Tiles stage to generate segmentation MBTiles for this level.'); return; }
    var gj=L.geoJSON(res.geojson, {
      style:function(f){ return {fillColor:f.properties.fill,color:'#666',weight:0.4,fillOpacity:opacity*0.9}; },
      onEachFeature:function(f,lyr){ var p=f.properties;
        lyr.bindPopup('<b>'+p.zone+'</b><br>Area: '+fmt(p.total_area_km2,2)+' km²<br>Cells: '+fmt(p.n_polygons,0)+
                      '<br>Mean sensitivity: '+fmt(p.sens_mean,2)+'<br>Mean # assets: '+fmt(p.mean_n_assets,1)); }
    });
    segVecGj=gj; segVec.addLayer(gj);
    try{ maps.seg.invalidateSize(); maps.seg.fitBounds(gj.getBounds(),{padding:[20,20]}); if(linking) syncFrom('seg'); }catch(e){}
    setMsg('Vector view (no MBTiles for this level yet).');
    renderPanel((res.legend||[]));
  }

  function loadVector(level, mode){
    var n=segCells[level]||0;
    if(n>HARD_CAP){ clearSegLayers();
      setMsg('Level "'+level+'" has ~'+n.toLocaleString()+' cells — too large for the on-the-fly vector view. '+
             'Run the Tiles stage to generate segmentation MBTiles for it.'); renderPanel([]); return; }
    api().seg_needs_build(level,mode).then(function(needs){
      if(needs && n>WARN_CELLS){ if(!confirm('Level "'+level+'" has ~'+n.toLocaleString()+
        ' cells. Building the vector view can take a while and use a lot of memory the first time. Continue?')) return; }
      setLoading(true);
      api().seg_overview(level,mode).then(function(res){ setLoading(false); renderVector(res); });
    });
  }

  function loadSeg(level, mode){
    // Prefer raster MBTiles; fall back to vector when none exist for this level.
    api().seg_tile_layers(level).then(function(tl){
      var info=(tl.modes||{})[mode];
      if(info){ showTiles(info, level, mode); }
      else { loadVector(level, mode); }
    });
  }

  function setSegModes(level){
    api().seg_modes(level).then(function(modes){
      var box=document.getElementById('segModes'); box.innerHTML='';
      modes.forEach(function(m,i){
        var lab=document.createElement('label'); lab.style.marginRight='10px';
        lab.innerHTML='<input type="radio" name="segmode" value="'+m+'" '+(i===0?'checked':'')+'> '+(m==='signatures'?'Signatures':'Clusters');
        box.appendChild(lab);
      });
      box.querySelectorAll('input[name=segmode]').forEach(function(r){
        r.addEventListener('change', function(){ segMode=this.value; loadSeg(segLevel, segMode); });
      });
      segMode=modes[0]||'signatures'; loadSeg(segLevel, segMode);
    });
  }

  function initSeg(){
    segLoaded=true;
    api().seg_levels().then(function(levels){
      var sel=document.getElementById('segLevel');
      if(!levels.length){ setMsg('No segmentation found. Run the Segment stage (5), then the Tiles stage.'); return; }
      levels.forEach(function(l){ segCells[l.name]=l.cells;
        var o=document.createElement('option'); o.value=l.name; o.text=l.name+' ('+Number(l.cells).toLocaleString()+' cells)'; sel.appendChild(o); });
      sel.addEventListener('change', function(){ segLevel=this.value; setSegModes(segLevel); });
      segLevel=levels[0].name; setSegModes(segLevel);
    });
  }

  // ---- classifications (sensitivity generalisation) tab ----
  var classLoaded=false, classRuns=[], classVec=L.geoJSON(null).addTo(maps.class);
  // Types table sort state (legend keeps server order; only the table reorders).
  var classProfileData=[], classSort={key:null, dir:1};
  function classSetLoading(on){ var e=document.getElementById('classLoading'); if(e) e.style.display=on?'block':'none'; }
  function classMsg(t){ document.getElementById('classMsg').innerHTML=t||''; }
  function clearClassLayers(){ classVec.clearLayers(); classVecGj=null; }
  function renderClassPanel(legend, profile){
    var leg=document.getElementById('classLegend'); leg.innerHTML='';
    if(!legend || !legend.length){ leg.innerHTML='<span class="muted">–</span>'; }
    else legend.forEach(function(z){ var d=document.createElement('div');
      d.innerHTML='<span class="sw" style="background:'+z.fill+'"></span>'+z.zone; leg.appendChild(d); });
    classProfileData = profile ? profile.slice() : [];
    renderClassRows();
  }
  function renderClassRows(){
    var tb=document.querySelector('#classTable tbody'); tb.innerHTML='';
    var rows=classProfileData.slice();
    if(classSort.key){
      var k=classSort.key, dir=classSort.dir, txt=(k==='cluster_label'||k==='top_asset_groups');
      rows.sort(function(a,b){
        var va=a[k], vb=b[k];
        if(txt){ return String(va==null?'':va).localeCompare(String(vb==null?'':vb))*dir; }
        va=(va==null||isNaN(va))?-Infinity:Number(va); vb=(vb==null||isNaN(vb))?-Infinity:Number(vb);
        return (va<vb?-1:va>vb?1:0)*dir;
      });
    }
    rows.forEach(function(p){
      var tr=document.createElement('tr');
      tr.innerHTML='<td>'+p.cluster_label+'</td><td>'+fmt(p.total_area_km2,2)+'</td><td>'+fmt(p.n_polygons,0)+
                   '</td><td>'+fmt(p.sens_mean,1)+'</td><td style="text-align:left">'+(p.top_asset_groups||'')+'</td>';
      tb.appendChild(tr);
    });
    document.querySelectorAll('#classTable thead th').forEach(function(th){
      var base=th.getAttribute('data-label');
      if(base===null){ base=th.textContent; th.setAttribute('data-label', base); }
      th.textContent = (th.dataset.key===classSort.key) ? base+(classSort.dir>0?' ▲':' ▼') : base;
    });
  }
  (function(){
    var thead=document.querySelector('#classTable thead');
    if(!thead) return;
    thead.addEventListener('click', function(e){
      var th=e.target.closest('th'); if(!th || !th.dataset.key) return;
      var k=th.dataset.key, txt=(k==='cluster_label'||k==='top_asset_groups');
      // New column: text ascending, numbers descending (largest first); same
      // column: flip direction.
      if(classSort.key===k){ classSort.dir=-classSort.dir; }
      else { classSort.key=k; classSort.dir=txt?1:-1; }
      renderClassRows();
    });
  })();
  function loadClass(run){
    if(!run) return;
    clearClassLayers(); classMsg('Loading…'); classSetLoading(true);
    api().segmv_layer(run.run_id, run.method, run.n_clusters).then(function(res){
      classSetLoading(false);
      if(!res || res.error){ classMsg((res&&res.error)||'Could not load results.'); renderClassPanel([],[]); return; }
      if(res.too_large){ classMsg('This result has '+Number(res.features).toLocaleString()+
        ' polygons — too many to draw as vectors (cap '+Number(res.max_features).toLocaleString()+').'); renderClassPanel([],[]); return; }
      var gj=L.geoJSON(res.geojson, {
        style:function(f){ return {fillColor:f.properties.fill,color:'#555',weight:0.3,fillOpacity:opacity*0.9}; },
        onEachFeature:function(f,lyr){ var p=f.properties;
          lyr.bindPopup('<b>'+(p.cluster_label||'type')+'</b><br>Cell: '+(p.code||'')+
                        '<br>Mean sensitivity: '+fmt(p.sens_mean,2)); }
      });
      classVecGj=gj; classVec.addLayer(gj);
      try{ maps.class.invalidateSize(); maps.class.fitBounds(gj.getBounds(),{padding:[20,20]}); if(linking) syncFrom('class'); }catch(e){}
      classMsg(res.layer ? ('<b>'+res.layer+'</b> · click a polygon to identify it.') : 'Click a polygon to identify it.');
      renderClassPanel(res.legend||[], res.profile||[]);
    });
  }
  function initClass(){
    classLoaded=true;
    api().segmv_runs().then(function(runs){
      classRuns=runs||[];
      var sel=document.getElementById('classRun');
      if(!classRuns.length){ classMsg('No classification results yet. Run <b>Classification</b> '+
        '(Workflows tab → Process), then reopen this map.'); renderClassPanel([],[]); return; }
      classRuns.forEach(function(r,i){ var o=document.createElement('option'); o.value=String(i); o.text=r.label; sel.appendChild(o); });
      sel.addEventListener('change', function(){ loadClass(classRuns[+this.value]); });
      loadClass(classRuns[0]);
    });
  }

  // Overview is the default active tab — init it once the bridge is ready.
  window.addEventListener('pywebviewready', function(){
    if(!resLoaded) initResults();
    if(START_TAB && START_TAB!=='results'){ showTab(START_TAB); }
  });
</script>
</body></html>"""


def run(base: str | None = None) -> None:
    global _BASE_DIR
    if webview is None:
        raise RuntimeError("'pywebview' is not installed. Install it with: pip install pywebview")
    _BASE_DIR = Path(base) if base else base_dir()
    api = _Api(_BASE_DIR)
    url = _start_server()
    # Devtools screenshot hook: open a tab / pre-sort the Zones table on load.
    # MESA_DEVTOOLS_MAP=seg:sens_mean -> ?tab=seg&sort=sens_mean
    dev = os.environ.get("MESA_DEVTOOLS_MAP", "").strip()
    if dev:
        tab, _, srt = dev.partition(":")
        q = "tab=" + tab + (("&sort=" + srt) if srt else "")
        url = url + ("&" if "?" in url else "?") + q
    window = webview.create_window(
        title="MESA maps (Assets / Overview / Segmentation / Classifications)",
        url=url,
        js_api=api,
        width=1360, height=860, resizable=True,
    )
    api.set_window(window)
    # No gui= override → cross-platform (Cocoa on macOS, EdgeChromium on Windows).
    webview.start(debug=False)


if __name__ == "__main__":
    if webview is None:
        sys.stderr.write("ERROR: 'pywebview' is not installed in this environment.\n")
        raise SystemExit(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_working_directory", default=None)
    args, _ = ap.parse_known_args()
    run(args.original_working_directory)
