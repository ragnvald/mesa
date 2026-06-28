#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""segmentation.py — per-geocode-layer segmentation of tbl_stacked.

Promotes the experimental devtools proof of concept (devtools/test_segmentation.py,
devtools/signature_analysis.py) into a reusable, pipeline-grade module. Two modes:

  signatures  — deterministic typology: each polygon is labelled by the *set* of
                MESA sensitivity codes (A..E) its overlapping assets carry, e.g.
                "B+C+D+E". No tuning, fully reproducible, cheap. The only mode.

Algorithmic clustering used to live here too, but the Classification tool
(code/segmentation_run.py) now owns that — it clusters in the (importance,
susceptibility) plane, which signatures deliberately do not. Signatures remain the
deterministic reference the Classification clustering is validated against (ARI/NMI).

MEMORY DISCIPLINE (see CLAUDE.md + learning.md "Parent-side memory in the pipeline")
    The heavy per-layer read happens here, and this module is meant to run *inside a
    spawned worker*, never in the pipeline orchestrator. tbl_stacked is read
    partition-by-partition with a pyarrow filter on name_gis_geocodegroup, so only one
    layer is ever in memory. The output table carries no geometry (join to
    tbl_geocode_object on `code` at render time), keeping it slim.

OUTPUTS (under <gpq_dir>/)
    tbl_segmentation/<layer>.parquet     — slim per-polygon table (one file per layer)
    tbl_segmentation_profiles.parquet    — one row per (layer, method, zone); tiny

Columns of tbl_segmentation/<layer>.parquet:
    code, name_gis_geocodegroup, signature, n_assets, cluster_id, cluster_method, sens_mean
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# Heavy libs (pandas/pyarrow/geopandas/sklearn) are imported lazily inside the
# functions so importing this module is cheap and --help-safe.

CODE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
VALID_CODES = ("A", "B", "C", "D", "E")

# Slim output schema — the contract the report engine and data model rely on.
OUTPUT_COLUMNS = [
    "code", "name_gis_geocodegroup", "signature", "n_assets",
    "cluster_id", "cluster_method", "sens_mean",
]


def _safe_layer_name(layer: str) -> str:
    """Filesystem-safe partition filename for a geocode-layer name."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(layer))[:120] or "layer"


def list_geocode_layers(gpq_dir: Path) -> list[str]:
    """Distinct geocode-layer names from tbl_geocode_object (small read).

    pyarrow-only (no pandas) so the lightweight Classification setup UI that calls
    this can exclude pandas from its bundle. See build_all.py
    helper_exclude_modules["segmentation_setup"]."""
    import pyarrow.dataset as ds
    poly_path = Path(gpq_dir) / "tbl_geocode_object.parquet"
    if not poly_path.exists():
        return []
    try:
        col = "name_gis_geocodegroup"
        tbl = ds.dataset(str(poly_path), format="parquet").to_table(columns=[col])
        return sorted({str(v) for v in tbl.column(col).to_pylist() if v is not None})
    except Exception:
        return []


def _read_layer_stacked(gpq_dir: Path, layer: str, log_fn=None):
    """Read tbl_stacked rows for one layer with a pyarrow filter. Returns a
    pandas DataFrame with code, sensitivity (float), sensitivity_code (A..E) and,
    when present, area_m2 (per-cell area, backfilled — constant across a code's rows).

    Robust to a corrupt/unreadable partition: if the fast whole-dataset read
    fails (e.g. one truncated parquet), it re-reads file-by-file and skips the
    bad partition(s) with a warning rather than aborting the whole stage. See
    learning.md "Segmentation robust to a corrupt tbl_stacked partition"."""
    import pandas as pd
    import pyarrow.dataset as ds

    gpq_dir = Path(gpq_dir)
    stacked_dir = gpq_dir / "tbl_stacked"
    stacked_file = gpq_dir / "tbl_stacked.parquet"
    if stacked_dir.exists():
        source = str(stacked_dir)
    elif stacked_file.exists():
        source = str(stacked_file)
    else:
        return pd.DataFrame(columns=["code", "sensitivity", "sensitivity_code", "area_m2"])

    dataset = ds.dataset(source, format="parquet")
    present = set(dataset.schema.names)
    wanted = [c for c in ("code", "name_gis_geocodegroup", "sensitivity", "sensitivity_code", "area_m2") if c in present]
    fld = ds.field("name_gis_geocodegroup") == layer
    try:
        df = dataset.to_table(columns=wanted, filter=fld).to_pandas()
    except Exception:
        src = Path(source)
        files = sorted(src.glob("*.parquet")) if src.is_dir() else [src]
        frames, skipped = [], []
        for f in files:
            try:
                frames.append(ds.dataset(str(f), format="parquet").to_table(columns=wanted, filter=fld).to_pandas())
            except Exception:
                skipped.append(f.name)
        if skipped and callable(log_fn):
            log_fn(f"[segment] WARNING: skipped {len(skipped)} unreadable tbl_stacked "
                   f"partition(s) (corrupt parquet): {', '.join(skipped[:5])}"
                   + (" …" if len(skipped) > 5 else "")
                   + " — re-run Intersect (Stage 2) to rebuild them.")
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=wanted)
    if "sensitivity" in df.columns:
        df["sensitivity"] = pd.to_numeric(df["sensitivity"], errors="coerce").fillna(0.0).astype("float64")
    else:
        df["sensitivity"] = 0.0
    if "sensitivity_code" in df.columns:
        df["sensitivity_code"] = df["sensitivity_code"].astype(str).str.strip().str.upper()
    else:
        df["sensitivity_code"] = ""
    if "area_m2" in df.columns:
        df["area_m2"] = pd.to_numeric(df["area_m2"], errors="coerce")
    df["code"] = df["code"].astype(str)
    return df


def _per_code_area(gpq_dir: Path, layer: str, stacked):
    """Return a Series (index = code, value = per-cell area in m²) for the layer.

    Primary source: the area_m2 already carried in tbl_stacked (backfilled, so it
    is constant across a code's rows → take the max). Fallback: tbl_flat filtered
    to the layer. Returns an empty Series if area is unavailable everywhere — the
    caller then shows '–' and sorts by polygon count.
    """
    import pandas as pd

    if "area_m2" in stacked.columns:
        a = pd.to_numeric(stacked["area_m2"], errors="coerce")
        if a.notna().any():
            return stacked.assign(_a=a).groupby("code")["_a"].max()

    # Fallback: authoritative per-cell area from tbl_flat.
    try:
        import pyarrow.dataset as ds
        gpq_dir = Path(gpq_dir)
        flat_file = gpq_dir / "tbl_flat.parquet"
        flat_dir = gpq_dir / "tbl_flat"
        src = str(flat_file) if flat_file.exists() else (str(flat_dir) if flat_dir.exists() else None)
        if src:
            d = ds.dataset(src, format="parquet")
            present = set(d.schema.names)
            if {"code", "area_m2"} <= present:
                cols = [c for c in ("code", "area_m2", "name_gis_geocodegroup") if c in present]
                filt = (ds.field("name_gis_geocodegroup") == layer) if "name_gis_geocodegroup" in present else None
                fdf = d.to_table(columns=cols, filter=filt).to_pandas()
                fdf["code"] = fdf["code"].astype(str)
                a = pd.to_numeric(fdf["area_m2"], errors="coerce")
                if a.notna().any():
                    return fdf.assign(_a=a).groupby("code")["_a"].max()
    except Exception:
        pass
    return pd.Series(dtype="float64")


def _signatures_for_layer(stacked):
    """Per-code signature, asset count and mean sensitivity from stacked rows.
    Returns a DataFrame indexed by code with columns signature, n_assets, sens_mean."""
    import pandas as pd

    coded = stacked[stacked["sensitivity_code"].isin(VALID_CODES)]
    n_assets = stacked.groupby("code").size().rename("n_assets")
    sens_mean = stacked.groupby("code")["sensitivity"].mean().rename("sens_mean")

    def _sig(s):
        codes = sorted(set(s), key=lambda c: CODE_ORDER.get(c, 99))
        return "+".join(codes)

    if coded.empty:
        sig = pd.Series(dtype=str, name="signature")
    else:
        sig = coded.groupby("code")["sensitivity_code"].apply(_sig).rename("signature")

    out = pd.concat([sig, n_assets, sens_mean], axis=1)
    out["signature"] = out["signature"].fillna("")
    out["n_assets"] = out["n_assets"].fillna(0).astype("int64")
    out["sens_mean"] = pd.to_numeric(out["sens_mean"], errors="coerce").fillna(0.0).round(4)
    return out


def segment_layer(gpq_dir, layer: str, *, mode: str = "signatures",
                  n_clusters: int = 0, spatial_method: str = "agglomerative",
                  log_fn=None) -> dict:
    """Segment one geocode layer and write tbl_segmentation/<layer>.parquet.

    Returns a summary dict: {layer, n_polygons, n_written, profiles: [...]}.
    profiles is a small list of per-zone dicts safe to ship back to the parent.

    Designed to run inside a spawned worker — the only large allocation
    (per-layer stacked + slim output) lives in this process.
    """
    import pandas as pd

    def _log(msg):
        if callable(log_fn):
            try:
                log_fn(msg)
            except Exception:
                pass

    gpq_dir = Path(gpq_dir)
    stacked = _read_layer_stacked(gpq_dir, layer, log_fn=_log)
    if stacked.empty:
        _log(f"[segment] layer '{layer}': 0 stacked rows — skipped")
        return {"layer": layer, "n_polygons": 0, "n_written": 0, "profiles": []}

    sig_df = _signatures_for_layer(stacked)
    n_polygons = len(sig_df)

    # Per-cell area for the per-zone total-area column (sorted on in the report).
    area_by_code = _per_code_area(gpq_dir, layer, stacked)
    if area_by_code.empty:
        _log(f"[segment] layer '{layer}': area_m2 unavailable — zones will show '–' for area")

    out = sig_df.reset_index().rename(columns={"index": "code"})
    out["name_gis_geocodegroup"] = layer
    out["cluster_id"] = pd.NA
    out["cluster_method"] = pd.NA

    # Algorithmic clustering moved to the Classification tool (segmentation_run.py).
    # Segment is signatures-only; cluster_id/cluster_method stay null for schema
    # stability. A caller still requesting "clusters"/"both" gets signatures.
    method_label = ""
    if mode in ("clusters", "both"):
        _log(f"[segment] layer '{layer}': 'clusters' mode retired — see the "
             f"Classification tool; writing signatures only.")

    for c in OUTPUT_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[OUTPUT_COLUMNS]

    # Write the slim per-layer partition (no geometry) — ZSTD-3 like the pipeline.
    seg_dir = gpq_dir / "tbl_segmentation"
    seg_dir.mkdir(parents=True, exist_ok=True)
    part_path = seg_dir / f"{_safe_layer_name(layer)}.parquet"
    out.to_parquet(part_path, index=False, compression="zstd", compression_level=3)
    _log(f"[segment] layer '{layer}': wrote {len(out):,} rows -> {part_path.name}")

    profiles = _build_profiles(out, mode, method_label, area_by_code)
    return {"layer": layer, "n_polygons": n_polygons, "n_written": len(out), "profiles": profiles}


def _zone_area_km2(codes, area_by_code):
    """Total area (km²) for a set of codes, or None when area is unavailable."""
    if area_by_code is None or len(area_by_code) == 0:
        return None
    total_m2 = float(area_by_code.reindex(codes).sum())
    return round(total_m2 / 1_000_000.0, 4)


def _build_profiles(out, mode: str, method_label: str, area_by_code=None) -> list[dict]:
    """One small row per zone (signature group, and per cluster if clustered).

    Each row carries total_area_km2 (None when per-cell area is unavailable). Rows
    are sorted within each method by total area descending — the report reads them
    big-to-small — falling back to polygon count when area is missing.
    """
    import pandas as pd

    layer = str(out["name_gis_geocodegroup"].iloc[0]) if len(out) else ""
    rows: list[dict] = []

    # Signature zones (always present).
    for sig, grp in out.groupby("signature"):
        rows.append({
            "name_gis_geocodegroup": layer,
            "method": "signatures",
            "zone": sig if sig else "(no overlap)",
            "n_polygons": int(len(grp)),
            "total_area_km2": _zone_area_km2(grp["code"], area_by_code),
            "sens_mean": round(float(pd.to_numeric(grp["sens_mean"], errors="coerce").mean() or 0.0), 4),
            "mean_n_assets": round(float(pd.to_numeric(grp["n_assets"], errors="coerce").mean() or 0.0), 2),
        })

    # Cluster zones (only if clustering ran).
    if method_label and out["cluster_id"].notna().any():
        for cid, grp in out.dropna(subset=["cluster_id"]).groupby("cluster_id"):
            rows.append({
                "name_gis_geocodegroup": layer,
                "method": method_label,
                "zone": f"cluster {int(cid)}",
                "n_polygons": int(len(grp)),
                "total_area_km2": _zone_area_km2(grp["code"], area_by_code),
                "sens_mean": round(float(pd.to_numeric(grp["sens_mean"], errors="coerce").mean() or 0.0), 4),
                "mean_n_assets": round(float(pd.to_numeric(grp["n_assets"], errors="coerce").mean() or 0.0), 2),
            })

    def _sort_key(r):
        a = r.get("total_area_km2")
        area_rank = a if isinstance(a, (int, float)) and a == a else float("-inf")  # NaN/None last
        return (r["method"], -area_rank, -r["n_polygons"])

    rows.sort(key=_sort_key)
    return rows


def write_profiles(gpq_dir, profile_rows: list[dict]) -> Optional[Path]:
    """Write the tiny tbl_segmentation_profiles.parquet from collected rows."""
    import pandas as pd
    if not profile_rows:
        return None
    df = pd.DataFrame(profile_rows)
    path = Path(gpq_dir) / "tbl_segmentation_profiles.parquet"
    df.to_parquet(path, index=False, compression="zstd", compression_level=3)
    return path


# ---------------------------------------------------------------------------
# Multiprocessing worker entry — one geocode layer per call. Top-level so it is
# picklable under the 'spawn' start method.
# ---------------------------------------------------------------------------

def segment_layer_worker(args: tuple) -> dict:
    """args = (gpq_dir_str, layer, options_dict). Returns the summary dict."""
    gpq_dir_str, layer, opts = args
    return segment_layer(
        gpq_dir_str, layer,
        mode=str(opts.get("mode", "signatures")),
        n_clusters=int(opts.get("n_clusters", 0)),
        spatial_method=str(opts.get("spatial_method", "agglomerative")),
    )


# ---------------------------------------------------------------------------
# Report helper — Marimekko/"domination overview" mosaic (promoted from the
# devtool). Renders a PNG from tbl_segmentation_profiles for a given layer.
# ---------------------------------------------------------------------------

_RAMP = {
    "A": (178, 24, 43), "B": (239, 138, 98), "C": (200, 200, 200),
    "D": (103, 169, 207), "E": (33, 102, 172),
}


def _signature_colour(sig: str):
    import numpy as np
    if not sig or sig == "(no overlap)":
        return (210, 210, 210)
    parts = [p for p in sig.split("+") if p in _RAMP]
    if not parts:
        return (180, 180, 180)
    if len(parts) == 1:
        return _RAMP[parts[0]]
    weights = np.array([{"A": 3.0, "B": 2.0, "C": 1.0, "D": 1.0, "E": 1.0}[p] for p in parts])
    rgb = np.zeros(3)
    for p, w in zip(parts, weights):
        rgb += np.array(_RAMP[p]) * w
    rgb /= weights.sum()
    return tuple(int(round(v)) for v in rgb)


def make_signature_mosaic(gpq_dir, layer: str, out_png, top_n: int = 16) -> bool:
    """Marimekko mosaic of signature frequencies for `layer`, from the profiles
    table. Column width = frequency, internal stack = codes present. Returns True
    on success, False if matplotlib/data is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        import pandas as pd
    except Exception:
        return False

    prof_path = Path(gpq_dir) / "tbl_segmentation_profiles.parquet"
    if not prof_path.exists():
        return False
    try:
        df = pd.read_parquet(prof_path)
    except Exception:
        return False
    df = df[(df["name_gis_geocodegroup"].astype(str) == str(layer)) & (df["method"] == "signatures")]
    if df.empty:
        return False
    df = df.sort_values("n_polygons", ascending=False).head(top_n)
    total = float(df["n_polygons"].sum()) or 1.0

    fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(df) + 2), 6))
    x_left = 0.0
    for _, row in df.iterrows():
        sig = str(row["zone"])
        w = float(row["n_polygons"]) / total
        parts = [p for p in sig.split("+") if p in _RAMP] if sig not in ("", "(no overlap)") else []
        if not parts:
            rgb = tuple(c / 255 for c in _signature_colour(sig))
            ax.add_patch(plt.Rectangle((x_left, 0), w, 1, color=rgb + (0.6,), ec="white", lw=0.8))
        else:
            seg_h = 1.0 / len(parts)
            y_bot = 0.0
            for code in parts:
                rgb = tuple(c / 255 for c in _RAMP[code])
                ax.add_patch(plt.Rectangle((x_left, y_bot), w, seg_h, color=rgb, ec="white", lw=0.8))
                ax.text(x_left + w / 2, y_bot + seg_h / 2, code, ha="center", va="center",
                        fontsize=9, color="white" if code in ("A", "E") else "black", fontweight="bold")
                y_bot += seg_h
        ax.text(x_left + w / 2, -0.04, sig, ha="center", va="top", fontsize=8)
        ax.text(x_left + w / 2, -0.10, f"{int(row['n_polygons']):,}", ha="center", va="top",
                fontsize=7, color="#444")
        x_left += w
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.16, 1.02)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Overlap signatures — {layer}\ncolumn width = frequency, stack = codes present")
    legend = [Patch(facecolor=tuple(c / 255 for c in _RAMP[k]), label=k) for k in VALID_CODES]
    legend.append(Patch(facecolor=(210 / 255,) * 3 + (0.6,), label="(no overlap)"))
    ax.legend(handles=legend, title="MESA code", loc="upper right",
              bbox_to_anchor=(1.0, -0.02), ncol=6, frameon=False)
    fig.tight_layout()
    try:
        fig.savefig(str(out_png), dpi=130)
    finally:
        plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Map-layer renderer — dissolved, render-ready GeoJSON per (layer, mode).
# A handful of multipolygons (one per signature / cluster) instead of millions
# of cells, so a map can draw the segmentation overlay instantly. This is the
# spatial renderer for segmentation: the standalone viewer window was parked
# (2026-06-06) in favour of folding segmentation in as a layer in the planned
# unified Asset + Results map app. The heavy dissolve runs once in the caller's
# process and the result is cached under output/cache/segmentation_overview/.
# ---------------------------------------------------------------------------

# Qualitative palette for cluster ids (ColorBrewer Set3 + Paired-ish).
_QUAL_PALETTE = [
    "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462",
    "#b3de69", "#fccde5", "#bc80bd", "#ccebc5", "#ffed6f", "#a6cee3",
    "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f",
    "#ff7f00", "#cab2d6", "#6a3d9a", "#b15928",
]


def _hex(rgb) -> str:
    r, g, b = (int(round(v)) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _overview_colour(category: str, mode: str) -> str:
    """Fill colour for a dissolved zone: MESA A–E ramp for signatures, a
    qualitative palette for clusters; grey for unassigned/noise."""
    if mode == "clusters":
        s = str(category).replace("cluster", "").strip()
        try:
            cid = int(s)
        except ValueError:
            return "#bdbdbd"
        if cid in (-1, -999):
            return "#dddddd"
        return _QUAL_PALETTE[cid % len(_QUAL_PALETTE)]
    # signatures
    if not category or category == "(no overlap)":
        return "#d2d2d2"
    return _hex(_signature_colour(category))


def overview_cache_path(gpq_dir, layer: str, mode: str) -> Path:
    """Where the dissolved GeoJSON for (layer, mode) is cached."""
    base = Path(gpq_dir).parent  # output/
    return base / "cache" / "segmentation_overview" / f"{_safe_layer_name(layer)}__{mode}.geojson"


def overview_modes(gpq_dir, layer: str) -> list[str]:
    """Modes available for a layer: always signatures; clusters only if present."""
    import pandas as pd
    modes = ["signatures"]
    part = Path(gpq_dir) / "tbl_segmentation" / f"{_safe_layer_name(layer)}.parquet"
    if part.exists():
        try:
            df = pd.read_parquet(part, columns=["cluster_id"])
            if "cluster_id" in df.columns and df["cluster_id"].notna().any():
                modes.append("clusters")
        except Exception:
            pass
    return modes


def build_overview_geojson(gpq_dir, layer: str, mode: str = "signatures",
                           simplify_tolerance: float | None = None,
                           out_path=None, max_cells: int = 2_000_000) -> Optional[dict]:
    """Dissolve a layer's segmentation into one multipolygon per category and
    return a GeoJSON FeatureCollection (+ a legend sorted big→small by area).

    Returns None when the layer/mode is unavailable. Writes the GeoJSON to
    out_path when given. Designed to run in the viewer's subprocess; reads one
    layer's geometry at a time.
    """
    import json
    import numpy as np
    import pandas as pd
    import geopandas as gpd

    gpq_dir = Path(gpq_dir)
    seg_path = gpq_dir / "tbl_segmentation" / f"{_safe_layer_name(layer)}.parquet"
    if not seg_path.exists():
        return None
    # Size guard: dissolving millions of cells to vector OOMs the viewer's
    # process (basic_mosaic ~9M cells crashed it silently). Refuse above
    # max_cells; such levels need pre-rendered MBTiles instead of vector.
    if max_cells and max_cells > 0:
        try:
            import pyarrow.parquet as pq
            n_cells = int(pq.ParquetFile(seg_path).metadata.num_rows)
            if n_cells > max_cells:
                return {"too_large": True, "cells": n_cells, "max_cells": int(max_cells)}
        except Exception:
            pass
    seg = pd.read_parquet(seg_path)
    seg["code"] = seg["code"].astype(str)

    if mode == "clusters":
        if "cluster_id" not in seg.columns or not seg["cluster_id"].notna().any():
            return None
        seg = seg[seg["cluster_id"].notna()].copy()
        seg["_cat"] = seg["cluster_id"].astype("Int64").astype("int64").map(lambda c: f"cluster {c}")
    else:
        seg["_cat"] = seg["signature"].fillna("").map(lambda s: s if s else "(no overlap)")

    poly = gpd.read_parquet(
        gpq_dir / "tbl_geocode_object.parquet",
        columns=["code", "geometry", "name_gis_geocodegroup"],
        filters=[("name_gis_geocodegroup", "=", layer)],
    )
    if poly.empty:
        return None
    poly["code"] = poly["code"].astype(str)
    g = poly.merge(seg[["code", "_cat", "sens_mean", "n_assets"]], on="code", how="inner")
    if g.empty:
        return None
    src_crs = g.crs

    # Per-zone stats (area via global equal-area EPSG:6933).
    try:
        areas = g.geometry.to_crs(6933).area
    except Exception:
        areas = pd.Series(np.zeros(len(g)), index=g.index)
    g["_area_m2"] = np.asarray(areas)
    stats = g.groupby("_cat").agg(
        n_polygons=("code", "size"),
        total_area_km2=("_area_m2", lambda s: round(float(s.sum()) / 1_000_000.0, 4)),
        sens_mean=("sens_mean", lambda s: round(float(pd.to_numeric(s, errors="coerce").mean() or 0.0), 4)),
        mean_n_assets=("n_assets", lambda s: round(float(pd.to_numeric(s, errors="coerce").mean() or 0.0), 2)),
    ).reset_index()

    # Dissolve to one (multi)polygon per category.
    diss = g[["_cat", "geometry"]].dissolve(by="_cat", as_index=False)

    # Auto simplify by size to keep the GeoJSON light (categories are coarse
    # zones, so a small tolerance is invisible at overview zoom). EPSG:4326:
    # 0.0001 ≈ 11 m. Bigger layers get a slightly larger tolerance.
    if simplify_tolerance is None:
        simplify_tolerance = 0.0005 if len(g) > 500_000 else 0.0001
    if simplify_tolerance and simplify_tolerance > 0:
        try:
            diss["geometry"] = diss.geometry.simplify(simplify_tolerance, preserve_topology=True)
        except Exception:
            pass

    diss = diss.merge(stats, on="_cat", how="left")
    diss["_fill"] = diss["_cat"].map(lambda c: _overview_colour(c, mode))

    # Leaflet wants WGS84.
    try:
        if src_crs is not None and "4326" not in str(src_crs):
            diss = diss.to_crs(4326)
    except Exception:
        pass

    feats = []
    for _, r in diss.iterrows():
        geom = r["geometry"]
        if geom is None or geom.is_empty:
            continue
        feats.append({
            "type": "Feature",
            "geometry": geom.__geo_interface__,
            "properties": {
                "zone": r["_cat"],
                "fill": r["_fill"],
                "n_polygons": int(r["n_polygons"]) if pd.notna(r["n_polygons"]) else 0,
                "total_area_km2": (round(float(r["total_area_km2"]), 4)
                                   if pd.notna(r["total_area_km2"]) else None),
                "sens_mean": round(float(r["sens_mean"]), 4) if pd.notna(r["sens_mean"]) else None,
                "mean_n_assets": round(float(r["mean_n_assets"]), 2) if pd.notna(r["mean_n_assets"]) else None,
            },
        })
    fc = {"type": "FeatureCollection", "features": feats}

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(fc), encoding="utf-8")

    legend = sorted(
        ({"zone": f["properties"]["zone"], "fill": f["properties"]["fill"],
          "total_area_km2": f["properties"]["total_area_km2"],
          "n_polygons": f["properties"]["n_polygons"],
          "sens_mean": f["properties"]["sens_mean"],
          "mean_n_assets": f["properties"]["mean_n_assets"]} for f in feats),
        key=lambda d: (d["total_area_km2"] is None, -(d["total_area_km2"] or 0.0)),
    )
    return {"path": str(out_path) if out_path else None, "geojson": fc,
            "legend": legend, "n_zones": len(feats)}
