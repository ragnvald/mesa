#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""segmentation_run.py — Multivariate spatial generalisation of sensitivity.

PURPOSE
    Additive, complementary view to MESA's univariate A–E sensitivity
    classification. The classification answers "how sensitive is this place?"
    (a level/intensity, one polygon at a time, neighbour-blind). This helper
    answers "what kind of sensitivity pattern is this place part of?" — it
    operates on the *stacked* per-asset profile of each polygon and groups
    polygons into a configurable number of sensitivity *types* (composition /
    character), optionally as spatially-contiguous regions. See the methods
    paper section "Generalisation of sensitivity patterns" (Frelat-style
    decomposition; Blaschke-style OBIA analogy) and docs/segmentation.md.

INPUTS  (output/geoparquet/, read at runtime — columns inspected, not assumed)
    tbl_stacked            per-asset rows per polygon (read partition-by-partition
                           with a pyarrow name_gis_geocodegroup filter — never
                           materialised whole; see CLAUDE.md / learning.md
                           "Parent-side memory in the pipeline").
    tbl_geocode_object     polygon geometries (join key: code) + full code list.
    tbl_geocode_group      geocode-layer catalogue.
    tbl_asset_group        asset-category labels for cluster profiling.
    config.ini             segmv_* keys (see segmentation_setup.py).

OUTPUTS
    output/geoparquet/tbl_seg_mv.parquet          one row per (code, method,
                                                  n_clusters, run_id) — ZSTD-3.
    output/geoparquet/tbl_seg_mv_profile.parquet  one row per (run_id, method,
                                                  n_clusters, cluster_id) — ZSTD-3.
    output/segmentation_mv/<run_id>/segmentation_results.gpkg  one layer per
                                                  (method, n_clusters) for QGIS.
    output/segmentation_mv/<run_id>/summary.md    parameters, methods + path run,
                                                  quality metrics, profile table.
    output/segmentation_mv/<run_id>/*.png         optional per-result maps (off
                                                  by default).

CALLED BY
    code/segmentation_setup.py ("Run now"), the mesa.exe launcher, or directly
    from a terminal:  python code/segmentation_run.py --original_working_directory <dir>

CALLS
    code/segmentation.py (list_geocode_layers, _read_layer_stacked pattern),
    code/mesa_shared.py (find_base_dir, read_config, parquet_dir).
    Heavy libs (scikit-learn, libpysal, spopt, hdbscan) are imported lazily.

NOTES
    MESA v5+ feature. CPU-only, no GPU. Does NOT touch the shipped tbl_segmentation*
    tables, the Segment pipeline stage, the Maps Segmentation tab, or the existing
    report section — it lives in its own tbl_seg_mv* namespace. spopt/hdbscan are
    optional; a missing one degrades gracefully (SKATER → KMeans+contiguity
    fallback; HDBSCAN method skipped). Reproducible: a fixed run_id + fixed seeds
    reproduce identical cluster assignments.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Make sibling helpers importable when run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mesa_shared  # stdlib-only, always safe to import early

# ---------------------------------------------------------------------------
# Canonical column names (verified against the MESA GeoParquet store). Code that
# depends on a column still checks presence at runtime before using it.
# ---------------------------------------------------------------------------
COL_CODE = "code"
COL_LAYER = "name_gis_geocodegroup"
COL_SENS = "sensitivity"
COL_SENS_CODE = "sensitivity_code"
COL_AREA = "area_m2"
COL_GROUP_ID = "ref_asset_group"        # FK to tbl_asset_group.id
COL_GROUP_NAME = "name_gis_assetgroup"  # human-ish label carried in tbl_stacked

# Equal-area CRS for honest km² (matches segmentation.py).
EQUAL_AREA_EPSG = 6933

DEFAULT_FEATURES = ("sum", "mean", "max", "std", "depth", "group_sums")
ALL_FEATURES = ("sum", "mean", "max", "std", "depth", "group_sums", "dominant")


# ---------------------------------------------------------------------------
# Logging — standard MESA log.txt append + stdout, with section headers.
# ---------------------------------------------------------------------------
def _ts() -> str:
    return datetime.now().strftime("%Y.%m.%d %H:%M:%S")


def make_logger(base_dir: Path):
    log_path = Path(base_dir) / "log.txt"
    # When stdout is a pipe (launched by the GUI's QProcess) rather than a real
    # console, Python picks the locale code page (cp1252 on Windows), which can't
    # encode glyphs we use in log lines like '≥' or '²' — that aborts the run.
    # Force UTF-8 with replacement; the GUI also decodes our stdout as UTF-8.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    def _log(msg: str) -> None:
        line = f"{_ts()} - [segmv] {msg}"
        try:
            print(line, flush=True)
        except Exception:
            # Last-resort: write bytes directly so a console-encoding quirk never
            # crashes a run (the file copy below is always UTF-8 regardless).
            try:
                sys.stdout.buffer.write((line + "\n").encode("utf-8", "replace"))
                sys.stdout.flush()
            except Exception:
                pass
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass

    return _log


def _section(log, title: str) -> None:
    log("")
    log("=" * 70)
    log(title)
    log("=" * 70)


def make_progress(total: int):
    """Return an emitter for machine-readable progress markers the Classification
    setup GUI parses to drive its bar. Lines look like
    '@@SEGMV_PROGRESS <done> <total> <label>'. Harmless when run standalone — just
    extra stdout lines the user (or nothing) reads."""
    state = {"done": 0, "total": max(1, int(total))}

    def _emit(label: str = "", advance: int = 1, done: Optional[int] = None) -> None:
        if done is not None:
            state["done"] = done
        else:
            state["done"] += advance
        d = max(0, min(state["done"], state["total"]))
        print(f"@@SEGMV_PROGRESS {d} {state['total']} {label}", flush=True)

    return _emit


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
class Params:
    """Full, reproducible parameter set for one run. Serialised into every
    output row and into summary.md so a run_id round-trips to identical output."""

    def __init__(self, **kw):
        self.run_id: str = kw.get("run_id") or datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.layer: str = kw.get("layer") or ""          # resolved later if blank
        self.n_clusters: list[int] = list(kw.get("n_clusters") or [8])
        self.method: str = (kw.get("method") or "attribute").lower()  # attribute|spatial|both
        self.pressure: str = kw.get("pressure") or ""    # "" / "all" = aggregate
        self.features: list[str] = list(kw.get("features") or DEFAULT_FEATURES)
        self.min_area_m2: float = float(kw.get("min_area_m2") or 0.0)
        self.skater_max_polys: int = int(kw.get("skater_max_polys") or 50_000)
        self.ai_enabled: bool = bool(kw.get("ai_enabled") or False)
        self.ollama_url: str = kw.get("ollama_url") or "http://localhost:11434/api/generate"
        self.ollama_model: str = kw.get("ollama_model") or "mistral"
        self.make_png: bool = bool(kw.get("make_png") or False)
        self.seed: int = int(kw.get("seed") or 42)

    def methods_to_run(self) -> list[str]:
        if self.method == "both":
            return ["attribute", "spatial"]
        return [self.method]

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id, "layer": self.layer, "n_clusters": self.n_clusters,
            "method": self.method, "pressure": self.pressure, "features": self.features,
            "min_area_m2": self.min_area_m2, "skater_max_polys": self.skater_max_polys,
            "ai_enabled": self.ai_enabled, "ollama_model": self.ollama_model,
            "make_png": self.make_png, "seed": self.seed,
        }


def params_from_config(cfg, **overrides) -> Params:
    """Build Params from config.ini segmv_* keys, with explicit overrides winning."""
    def g(key, default=""):
        try:
            return (cfg["DEFAULT"].get(key, default) or default).split("#", 1)[0].strip()
        except Exception:
            return default

    n_raw = overrides.get("n_clusters") or g("segmv_n_clusters", "8")
    if isinstance(n_raw, str):
        n_list = [int(x) for x in n_raw.replace(";", ",").split(",") if x.strip().isdigit()] or [8]
    else:
        n_list = list(n_raw)

    feat_raw = overrides.get("features") or g("segmv_features", ",".join(DEFAULT_FEATURES))
    if isinstance(feat_raw, str):
        feats = [f.strip() for f in feat_raw.split(",") if f.strip() in ALL_FEATURES] or list(DEFAULT_FEATURES)
    else:
        feats = list(feat_raw)

    def _truthy(v):
        return str(v).strip().lower() in ("1", "true", "yes", "on")

    return Params(
        run_id=overrides.get("run_id"),
        layer=overrides.get("layer") if overrides.get("layer") is not None else g("segmv_geocode_layer", ""),
        n_clusters=n_list,
        method=overrides.get("method") or g("segmv_method", "attribute"),
        pressure=overrides.get("pressure") if overrides.get("pressure") is not None else g("segmv_pressure", ""),
        features=feats,
        min_area_m2=overrides.get("min_area_m2") if overrides.get("min_area_m2") is not None else g("segmv_min_area_m2", "0"),
        skater_max_polys=overrides.get("skater_max_polys") or g("segmv_skater_max_polys", "50000"),
        ai_enabled=overrides.get("ai_enabled") if overrides.get("ai_enabled") is not None else _truthy(g("segmv_ai_enabled", "0")),
        ollama_url=overrides.get("ollama_url") or g("segmv_ollama_url", "http://localhost:11434/api/generate"),
        ollama_model=overrides.get("ollama_model") or g("segmv_ollama_model", "mistral"),
        make_png=overrides.get("make_png") if overrides.get("make_png") is not None else _truthy(g("segmv_make_png", "0")),
        seed=overrides.get("seed") or 42,
    )


# ---------------------------------------------------------------------------
# Input reads
# ---------------------------------------------------------------------------
def detect_pressure_columns(gpq: Path) -> list[str]:
    """Return tbl_stacked columns that look like a pressure identifier. The
    canonical MESA stack has none today, so this is usually empty — the setup
    UI then offers only 'all pressures (aggregate)'."""
    import pyarrow.dataset as ds
    src = _stacked_source(gpq)
    if src is None:
        return []
    try:
        names = ds.dataset(src, format="parquet").schema.names
    except Exception:
        return []
    return [n for n in names if "pressure" in n.lower()]


def _stacked_source(gpq: Path) -> Optional[str]:
    d, f = Path(gpq) / "tbl_stacked", Path(gpq) / "tbl_stacked.parquet"
    if d.exists():
        return str(d)
    if f.exists():
        return str(f)
    return None


def read_layer_stacked(gpq: Path, layer: str, pressure: str, log) -> "object":
    """Partitioned, filtered read of tbl_stacked for one layer. Mirrors
    segmentation.py:_read_layer_stacked (memory discipline) but also pulls the
    asset-group columns needed for the multivariate feature vector."""
    import pandas as pd
    import pyarrow.dataset as ds

    src = _stacked_source(gpq)
    cols = [COL_CODE, COL_LAYER, COL_SENS, COL_SENS_CODE, COL_AREA, COL_GROUP_ID, COL_GROUP_NAME]
    if src is None:
        return pd.DataFrame(columns=cols)

    dataset = ds.dataset(src, format="parquet")
    present = set(dataset.schema.names)
    wanted = [c for c in cols if c in present]
    press_cols = [c for c in present if "pressure" in c.lower()]
    if pressure and pressure.lower() not in ("", "all") and press_cols:
        wanted = list(dict.fromkeys(wanted + press_cols))
    flt = ds.field(COL_LAYER) == layer
    try:
        df = dataset.to_table(columns=wanted, filter=flt).to_pandas()
    except Exception:
        # Robust to a corrupt partition: re-read file-by-file, skip bad ones.
        srcp = Path(src)
        files = sorted(srcp.glob("*.parquet")) if srcp.is_dir() else [srcp]
        frames, skipped = [], []
        for fp in files:
            try:
                frames.append(ds.dataset(str(fp), format="parquet")
                              .to_table(columns=wanted, filter=flt).to_pandas())
            except Exception:
                skipped.append(fp.name)
        if skipped:
            log(f"WARNING: skipped {len(skipped)} unreadable tbl_stacked partition(s): "
                f"{', '.join(skipped[:5])} — re-run Intersect to rebuild.")
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=wanted)

    # Optional pressure filter (only if a pressure column actually exists).
    if pressure and pressure.lower() not in ("", "all") and press_cols:
        pc = press_cols[0]
        df = df[df[pc].astype(str) == str(pressure)]

    df[COL_SENS] = pd.to_numeric(df.get(COL_SENS), errors="coerce").fillna(0.0).astype("float64")
    if COL_AREA in df.columns:
        df[COL_AREA] = pd.to_numeric(df[COL_AREA], errors="coerce")
    df[COL_CODE] = df[COL_CODE].astype(str)
    if COL_GROUP_ID in df.columns:
        df[COL_GROUP_ID] = df[COL_GROUP_ID].astype(str)
    return df


def layer_codes(gpq: Path, layer: str):
    """All polygon codes for the layer (so empty-stack polygons surface as no_data)."""
    import pandas as pd
    p = Path(gpq) / "tbl_geocode_object.parquet"
    if not p.exists():
        return pd.Index([], name=COL_CODE)
    try:
        s = pd.read_parquet(p, columns=[COL_CODE, COL_LAYER])
    except Exception:
        return pd.Index([], name=COL_CODE)
    s = s[s[COL_LAYER].astype(str) == str(layer)]
    return pd.Index(s[COL_CODE].astype(str).unique(), name=COL_CODE)


def asset_group_labels(gpq: Path) -> dict:
    """Map ref_asset_group (==tbl_asset_group.id, as str) → human label."""
    import pandas as pd
    p = Path(gpq) / "tbl_asset_group.parquet"
    if not p.exists():
        return {}
    try:
        df = pd.read_parquet(p, columns=["id", "title_fromuser", "name_gis_assetgroup"])
    except Exception:
        return {}
    out = {}
    for _, r in df.iterrows():
        label = (str(r.get("title_fromuser") or "").strip()
                 or str(r.get("name_gis_assetgroup") or "").strip()
                 or f"group {r.get('id')}")
        out[str(r.get("id"))] = label
    return out


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------
def build_feature_matrix(stacked, all_codes, features: list[str], log):
    """Aggregate stacked rows into one fixed-length vector per polygon.

    Returns (feat_df, no_data_mask, sens_mean, group_sum_cols) where feat_df is
    indexed by code over *all_codes*. Empty-stack polygons are zero rows flagged
    in no_data_mask."""
    import numpy as np
    import pandas as pd

    feat = pd.DataFrame(index=pd.Index(all_codes, name=COL_CODE))

    if stacked is None or stacked.empty:
        log("No stacked rows for this layer — every polygon is no_data.")
        feat["sens_sum"] = 0.0
        no_data = pd.Series(True, index=feat.index)
        return feat, no_data, pd.Series(0.0, index=feat.index), []

    grp = stacked.groupby(COL_CODE)[COL_SENS]
    agg = pd.DataFrame({
        "sens_sum": grp.sum(),
        "sens_mean": grp.mean(),
        "sens_max": grp.max(),
        "sens_std": grp.std().fillna(0.0),
        "depth": stacked.groupby(COL_CODE).size().astype("float64"),
    })

    group_sum_cols: list[str] = []
    if "group_sums" in features and COL_GROUP_ID in stacked.columns:
        piv = (stacked.pivot_table(index=COL_CODE, columns=COL_GROUP_ID,
                                   values=COL_SENS, aggfunc="sum", fill_value=0.0))
        piv.columns = [f"grp_{c}" for c in piv.columns]
        group_sum_cols = list(piv.columns)
        agg = agg.join(piv, how="left")

    if "dominant" in features and COL_GROUP_ID in stacked.columns:
        dom = (stacked.groupby([COL_CODE, COL_GROUP_ID])[COL_SENS].sum()
               .reset_index().sort_values(COL_SENS, ascending=False)
               .drop_duplicates(COL_CODE).set_index(COL_CODE)[COL_GROUP_ID])
        oh = pd.get_dummies(dom, prefix="dom").astype("float64")
        agg = agg.join(oh, how="left")

    # Select the requested scalar aggregates (group_sums/dominant handled above).
    scalar_map = {"sum": "sens_sum", "mean": "sens_mean", "max": "sens_max",
                  "std": "sens_std", "depth": "depth"}
    keep = [scalar_map[f] for f in features if f in scalar_map]
    keep += group_sum_cols + [c for c in agg.columns if c.startswith("dom_")]
    keep = [c for c in dict.fromkeys(keep) if c in agg.columns] or ["sens_sum"]
    agg = agg[keep]

    feat = feat.join(agg, how="left")
    no_data = feat[keep[0]].isna() if keep else pd.Series(True, index=feat.index)
    # Per-cell mean sensitivity carried for report colouring (0 where no_data).
    sens_mean = feat["sens_mean"] if "sens_mean" in feat.columns else (
        stacked.groupby(COL_CODE)[COL_SENS].mean().reindex(feat.index))
    sens_mean = sens_mean.fillna(0.0)
    feat = feat.fillna(0.0)
    log(f"Feature matrix: {feat.shape[0]} polygons × {feat.shape[1]} features "
        f"({int(no_data.sum())} no_data) — columns: {', '.join(keep[:8])}"
        + (" …" if len(keep) > 8 else ""))
    return feat, no_data.fillna(True), sens_mean, group_sum_cols


# ---------------------------------------------------------------------------
# Geometry / contiguity
# ---------------------------------------------------------------------------
def read_layer_geometry(gpq: Path, layer: str, codes):
    """GeoDataFrame (code, geometry) for the given codes, in the file's native CRS."""
    import geopandas as gpd
    p = Path(gpq) / "tbl_geocode_object.parquet"
    g = gpd.read_parquet(p, columns=[COL_CODE, COL_LAYER, "geometry"])
    g = g[g[COL_LAYER].astype(str) == str(layer)].copy()
    g[COL_CODE] = g[COL_CODE].astype(str)
    g = g[g[COL_CODE].isin(set(map(str, codes)))]
    return g.drop_duplicates(COL_CODE).set_index(COL_CODE)


def per_code_area_km2(stacked, geom_gdf):
    """km² per code: prefer backfilled area_m2 in stacked (constant per code →
    max), fall back to equal-area geometry area."""
    import pandas as pd
    if stacked is not None and not stacked.empty and COL_AREA in stacked.columns \
            and stacked[COL_AREA].notna().any():
        a = stacked.groupby(COL_CODE)[COL_AREA].max() / 1e6
        return a
    try:
        g = geom_gdf.to_crs(EQUAL_AREA_EPSG)
        return (g.geometry.area / 1e6)
    except Exception:
        return pd.Series(dtype="float64")


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def _scale(X):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(X.to_numpy(dtype=float))


def fit_attribute(Xdf, k: int, seed: int, log):
    """KMeans on standardised features. Returns (labels, method_label, metrics)."""
    import numpy as np
    from sklearn.cluster import KMeans
    Xs = _scale(Xdf)
    k = max(2, min(int(k), len(Xdf) - 1)) if len(Xdf) > 2 else 1
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(Xs)
    metrics = {}
    try:
        from sklearn.metrics import silhouette_score
        if 1 < k < len(Xdf):
            metrics["silhouette"] = round(float(silhouette_score(Xs, labels)), 4)
    except Exception:
        pass
    metrics["inertia"] = round(float(getattr(km, "inertia_", float("nan"))), 2)
    return np.asarray(labels), f"kmeans_k{k}", metrics


def fit_hdbscan(Xdf, log):
    """Optional emergent-count clustering. Returns None if hdbscan missing."""
    try:
        import hdbscan
    except Exception:
        log("HDBSCAN unavailable — skipping emergent-count comparison.")
        return None
    import numpy as np
    Xs = _scale(Xdf)
    cl = hdbscan.HDBSCAN(min_cluster_size=max(5, len(Xdf) // 200))
    labels = cl.fit_predict(Xs)  # -1 = noise
    n = len(set(labels) - {-1})
    return np.asarray(labels), f"hdbscan_n{n}", {"emergent_clusters": n}


def _queen_weights(geom_gdf, log):
    """Queen contiguity over the geometries. Returns (w, ordered_codes) or None."""
    try:
        from libpysal.weights import Queen
    except Exception:
        log("libpysal unavailable — cannot build contiguity graph.")
        return None
    try:
        g = geom_gdf.reset_index()
        w = Queen.from_dataframe(g, ids=g[COL_CODE].tolist(), use_index=False)
        return w, g[COL_CODE].tolist()
    except Exception as exc:
        log(f"Queen weights failed: {exc}")
        return None


def fit_spatial(Xdf, k: int, geom_gdf, skater_max: int, seed: int, log):
    """Spatial+attribute regionalisation. SKATER (spopt) when feasible, else
    KMeans + post-hoc contiguity enforcement. Always logs which path ran."""
    import numpy as np
    n = len(Xdf)
    use_skater = n <= skater_max
    if not use_skater:
        log(f"SKATER skipped: {n} polygons > segmv_skater_max_polys={skater_max}. "
            f"Using KMeans + contiguity fallback.")
    if use_skater:
        try:
            from spopt.region import Skater
            from sklearn.preprocessing import StandardScaler
            wq = _queen_weights(geom_gdf, log)
            if wq is None:
                raise RuntimeError("no contiguity graph")
            w, ordered = wq
            g = geom_gdf.reindex(ordered).copy()
            cols = list(Xdf.columns)
            Xs = StandardScaler().fit_transform(Xdf.reindex(ordered).to_numpy(dtype=float))
            for i, c in enumerate(cols):
                g[c] = Xs[:, i]
            k_eff = max(2, min(int(k), n - 1))
            model = Skater(g, w, cols, n_clusters=k_eff,
                           floor=1, trace=False, islands="increase")
            model.solve()
            labels = np.asarray(model.labels_)
            # Re-align to Xdf index order.
            lbl = {c: int(l) for c, l in zip(ordered, labels)}
            out = np.array([lbl.get(c, -1) for c in Xdf.index], dtype=int)
            log(f"SKATER ran (spopt) for k={k_eff}.")
            return out, f"skater_k{k_eff}", {"path": "skater", "objective": _spatial_objective(Xdf, out)}
        except Exception as exc:
            log(f"SKATER path failed ({exc}); falling back to KMeans + contiguity.")

    # Fallback: attribute KMeans, then merge non-contiguous fragments.
    labels, _, _ = fit_attribute(Xdf, k, seed, log)
    labels = _enforce_contiguity(labels, Xdf.index.tolist(), geom_gdf, log)
    return labels, f"kmeans_contig_k{int(max(labels)+1) if len(labels) else 0}", {
        "path": "kmeans_contiguity", "objective": _spatial_objective(Xdf, labels)}


def _spatial_objective(Xdf, labels) -> float:
    """Within-cluster sum of squared deviations on standardised features (lower
    = tighter). A SKATER-style coherence proxy comparable across methods."""
    import numpy as np
    try:
        Xs = _scale(Xdf)
        total = 0.0
        for lab in set(labels):
            if lab < 0:
                continue
            m = labels == lab
            if m.sum() == 0:
                continue
            total += float(((Xs[m] - Xs[m].mean(axis=0)) ** 2).sum())
        return round(total, 2)
    except Exception:
        return float("nan")


def _enforce_contiguity(labels, codes, geom_gdf, log):
    """Merge each non-contiguous cluster fragment into its majority neighbour.
    Keeps the largest connected component per label; reassigns the rest."""
    import numpy as np
    wq = _queen_weights(geom_gdf, log)
    if wq is None:
        log("Contiguity enforcement skipped (no graph) — labels left as-is.")
        return np.asarray(labels)
    w, ordered = wq
    pos = {c: i for i, c in enumerate(codes)}
    lab = {c: int(labels[pos[c]]) for c in codes if c in pos}
    neigh = w.neighbors

    # Connected components within each label via BFS over same-label neighbours.
    seen, comp_of, comps = set(), {}, []
    for c in ordered:
        if c in seen or c not in lab:
            continue
        stack, comp = [c], []
        seen.add(c)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in neigh.get(u, []):
                if v not in seen and v in lab and lab[v] == lab[u]:
                    seen.add(v)
                    stack.append(v)
        cid = len(comps)
        comps.append(comp)
        for u in comp:
            comp_of[u] = cid

    # For each label, keep its biggest component; reassign smaller ones.
    from collections import defaultdict, Counter
    by_label = defaultdict(list)
    for cid, comp in enumerate(comps):
        by_label[lab[comp[0]]].append(cid)
    keep = set()
    for _label, cids in by_label.items():
        keep.add(max(cids, key=lambda i: len(comps[i])))

    changed = 0
    for cid, comp in enumerate(comps):
        if cid in keep:
            continue
        # majority label among boundary neighbours not in this component
        votes = Counter()
        for u in comp:
            for v in neigh.get(u, []):
                if comp_of.get(v) != cid:
                    votes[lab[v]] += 1
        if votes:
            new = votes.most_common(1)[0][0]
            for u in comp:
                lab[u] = new
            changed += len(comp)
    if changed:
        log(f"Contiguity enforcement reassigned {changed} polygon(s) in fragments.")
    return np.asarray([lab[c] for c in codes])


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------
def build_profiles(params, method_label, n_clusters, codes, labels, no_data,
                   stacked, sens_mean, area_km2, group_labels, log):
    """One profile row per cluster: stats, top-3 asset groups, count, area."""
    import numpy as np
    import pandas as pd

    rows = []
    fit_mask = ~no_data.reindex(codes).fillna(True).to_numpy()
    label_arr = np.asarray(labels)

    # Per-(code) dominant-group contribution for top-3 naming.
    grp_contrib = None
    if stacked is not None and not stacked.empty and COL_GROUP_ID in stacked.columns:
        grp_contrib = stacked.groupby([COL_CODE, COL_GROUP_ID])[COL_SENS].sum()

    sm = sens_mean.reindex(codes).fillna(0.0).to_numpy()
    ar = area_km2.reindex(codes).fillna(0.0).to_numpy() if area_km2 is not None else np.zeros(len(codes))

    for lab in sorted(set(label_arr.tolist())):
        sel = (label_arr == lab) & fit_mask
        if sel.sum() == 0:
            continue
        sub_codes = [c for c, s in zip(codes, sel) if s]
        s_vals = sm[sel]
        top3 = []
        if grp_contrib is not None:
            sub = grp_contrib[grp_contrib.index.get_level_values(0).isin(sub_codes)]
            if not sub.empty:
                agg = sub.groupby(level=1).sum().sort_values(ascending=False).head(3)
                top3 = [group_labels.get(str(gid), str(gid)) for gid in agg.index]
        rows.append({
            "run_id": params.run_id,
            "name_gis_geocodegroup": params.layer,
            "method": method_label,
            "n_clusters": int(n_clusters),
            "cluster_id": int(lab),
            "cluster_label": (f"noise" if lab == -1 else f"type {int(lab)+1}"),
            "n_polygons": int(sel.sum()),
            "total_area_km2": round(float(ar[sel].sum()), 4),
            "sens_mean": round(float(s_vals.mean()), 4) if len(s_vals) else 0.0,
            "sens_max": round(float(s_vals.max()), 4) if len(s_vals) else 0.0,
            "sens_std": round(float(s_vals.std()), 4) if len(s_vals) else 0.0,
            "top_asset_groups": ", ".join(top3),
            "description_ai": None,
        })
    return rows


# ---------------------------------------------------------------------------
# AI descriptions (optional, default off)
# ---------------------------------------------------------------------------
def _ai_context(profile_row) -> str:
    return (
        f"This is a sensitivity-pattern type from a spatial generalisation of an "
        f"environmental sensitivity analysis. Cluster '{profile_row['cluster_label']}' "
        f"covers {profile_row['n_polygons']} polygons ({profile_row['total_area_km2']} km²). "
        f"Mean sensitivity {profile_row['sens_mean']} (max {profile_row['sens_max']}, "
        f"std {profile_row['sens_std']}). Dominant asset groups: "
        f"{profile_row['top_asset_groups'] or 'none'}. "
        f"Write ONE short plain-language paragraph (<=60 words) describing what kind of "
        f"area this represents for a spatial planner. No preamble."
    )


def _ollama_describe(prompt: str, url: str, model: str, log) -> Optional[str]:
    try:
        import urllib.request
        payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = (data.get("response") or "").strip()
        return text or None
    except Exception as exc:
        log(f"Ollama description failed ({exc}); leaving null.")
        return None


def _openai_describe(prompt: str, base_dir: Path, log) -> Optional[str]:
    """Fallback to the existing OpenAI integration if a key is configured.
    Reuses asset_map_view's key resolution pattern (env / config / secrets)."""
    try:
        key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if not key:
            kp = Path(base_dir) / "secrets" / "openai.key"
            if kp.exists():
                key = kp.read_text(encoding="utf-8").strip()
        if not key:
            return None
        import urllib.request
        payload = json.dumps({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data["choices"][0]["message"]["content"] or "").strip() or None
    except Exception as exc:
        log(f"OpenAI description failed ({exc}); leaving null.")
        return None


def add_ai_descriptions(profile_rows, params, base_dir, log):
    if not params.ai_enabled or not profile_rows:
        return profile_rows
    _section(log, "AI cluster descriptions")
    for row in profile_rows:
        prompt = _ai_context(row)
        txt = _ollama_describe(prompt, params.ollama_url, params.ollama_model, log)
        if not txt:
            txt = _openai_describe(prompt, base_dir, log)
        row["description_ai"] = txt
    return profile_rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def _write_parquet_coexist(path: Path, new_df, run_id: str, log):
    """Append rows for this run_id, replacing any existing rows with the same
    run_id so a re-run is idempotent and multiple runs co-exist."""
    import pandas as pd
    path = Path(path)
    if path.exists():
        try:
            old = pd.read_parquet(path)
            old = old[old.get("run_id").astype(str) != str(run_id)] if "run_id" in old.columns else old
            new_df = pd.concat([old, new_df], ignore_index=True)
        except Exception as exc:
            log(f"WARNING: could not merge existing {path.name} ({exc}); overwriting run rows only.")
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_parquet(path, index=False, compression="zstd", compression_level=3)
    log(f"Wrote {path.name}: {len(new_df)} total row(s).")


def export_gpkg(out_dir: Path, results, geom_by_code, log):
    """One GPKG layer per (method, n_clusters): polygons + cluster_id."""
    import geopandas as gpd
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg = out_dir / "segmentation_results.gpkg"
    wrote = 0
    for r in results:
        df = r["assignments"]  # code, cluster_id, cluster_label, no_data, sens_mean
        try:
            g = geom_by_code.join(df.set_index("code"), how="inner").reset_index()
            gdf = gpd.GeoDataFrame(g, geometry="geometry", crs=geom_by_code.crs)
            layer_name = f"{r['method_label']}"[:60]
            gdf.to_file(gpkg, layer=layer_name, driver="GPKG")
            wrote += 1
        except Exception as exc:
            log(f"WARNING: GPKG layer for {r['method_label']} failed ({exc}).")
    if wrote:
        log(f"Wrote {gpkg.name}: {wrote} layer(s).")
    return gpkg


def write_summary_md(out_dir: Path, params, results, log):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Sensitivity generalisation — run summary", "",
             f"- **run_id**: `{params.run_id}`",
             f"- **geocode layer**: {params.layer}",
             f"- **method(s)**: {params.method}",
             f"- **cluster counts**: {', '.join(map(str, params.n_clusters))}",
             f"- **pressure filter**: {params.pressure or 'all (aggregate)'}",
             f"- **features**: {', '.join(params.features)}",
             f"- **min polygon area (m²)**: {params.min_area_m2}",
             f"- **seed**: {params.seed}", "",
             "This run answers *\"what kind of sensitivity pattern is this place part "
             "of?\"* — complementary to the A–E *\"how sensitive is this place?\"* "
             "classification. See methods paper, \"Generalisation of sensitivity patterns\".",
             ""]
    for r in results:
        lines.append(f"## {r['method_label']}")
        m = r.get("metrics", {})
        if m:
            lines.append("Quality metrics: " + ", ".join(f"{k}={v}" for k, v in m.items()))
            lines.append("")
        lines.append("| Cluster | Polygons | Area km² | Mean sens | Top asset groups | AI description |")
        lines.append("|---|---:|---:|---:|---|---|")
        for p in r["profiles"]:
            lines.append(f"| {p['cluster_label']} | {p['n_polygons']} | {p['total_area_km2']} | "
                         f"{p['sens_mean']} | {p['top_asset_groups']} | {p.get('description_ai') or ''} |")
        lines.append("")
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"Wrote {path.name}.")
    return path


def write_png_maps(out_dir: Path, results, geom_by_code, log):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import geopandas as gpd
    except Exception as exc:
        log(f"PNG maps skipped (matplotlib/geopandas unavailable: {exc}).")
        return
    out_dir = Path(out_dir)
    for r in results:
        try:
            df = r["assignments"]
            g = geom_by_code.join(df.set_index("code"), how="inner").reset_index()
            gdf = gpd.GeoDataFrame(g, geometry="geometry", crs=geom_by_code.crs)
            ax = gdf.plot(column="cluster_id", categorical=True, legend=True, figsize=(8, 8))
            ax.set_axis_off()
            ax.set_title(r["method_label"])
            fig = ax.get_figure()
            fig.savefig(out_dir / f"{r['method_label']}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            log(f"PNG for {r['method_label']} failed ({exc}).")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_segmentation(base_dir, params: Params, log=None) -> dict:
    """Execute one segmentation run. Returns a small result dict (paths + counts)."""
    import pandas as pd

    base_dir = Path(base_dir)
    log = log or make_logger(base_dir)
    cfg = mesa_shared.read_config(base_dir)
    gpq = mesa_shared.parquet_dir(base_dir, cfg)

    _section(log, f"Sensitivity generalisation — run_id {params.run_id}")
    log(f"Parameters: {json.dumps(params.as_dict())}")

    # Resolve layer (blank → basic_mosaic if present, else first available).
    import segmentation as _seg
    layers = _seg.list_geocode_layers(gpq)
    if not layers:
        log("No geocode layers found — nothing to segment.")
        return {"ok": False, "reason": "no layers"}
    if not params.layer:
        params.layer = "basic_mosaic" if "basic_mosaic" in layers else layers[0]
    if params.layer not in layers:
        log(f"Layer '{params.layer}' not found. Available: {', '.join(layers)}")
        return {"ok": False, "reason": "layer not found"}
    log(f"Operating on layer: {params.layer}")

    # Progress milestones: read inputs, build features, one per (method × k) fit,
    # write outputs. The GUI maps these onto its bar; standalone runs ignore them.
    total_steps = len(params.methods_to_run()) * len(params.n_clusters) + 3
    progress = make_progress(total_steps)
    progress("Reading inputs", done=0)

    # Inputs.
    stacked = read_layer_stacked(gpq, params.layer, params.pressure, log)
    all_codes = list(layer_codes(gpq, params.layer))
    if not all_codes:
        # Fall back to codes present in the stack.
        all_codes = sorted(stacked[COL_CODE].unique().tolist()) if not stacked.empty else []
    if not all_codes:
        log("No polygons for this layer.")
        return {"ok": False, "reason": "no polygons"}
    group_labels = asset_group_labels(gpq)

    # Optional min-area sliver filter (basic_mosaic). Drops tiny polygons up front.
    geom = read_layer_geometry(gpq, params.layer, all_codes)
    area_km2 = per_code_area_km2(stacked, geom)
    if params.min_area_m2 and params.min_area_m2 > 0 and not area_km2.empty:
        keep_codes = set(area_km2[area_km2 * 1e6 >= params.min_area_m2].index.astype(str))
        before = len(all_codes)
        all_codes = [c for c in all_codes if c in keep_codes]
        log(f"Min-area filter (≥{params.min_area_m2:.0f} m²): kept {len(all_codes)}/{before} polygons.")
        geom = geom.loc[geom.index.intersection(all_codes)]

    progress("Building feature vectors")
    feat, no_data, sens_mean, _gcols = build_feature_matrix(stacked, all_codes, params.features, log)

    # Fit set excludes no_data polygons.
    fit_codes = [c for c, nd in zip(all_codes, no_data.to_numpy()) if not nd]
    if len(fit_codes) < 2:
        log("Fewer than 2 polygons with stacked data — cannot cluster.")
        return {"ok": False, "reason": "insufficient data"}
    Xfit = feat.loc[fit_codes]
    geom_fit = geom.loc[geom.index.intersection(fit_codes)]

    results = []
    for method in params.methods_to_run():
        for k in params.n_clusters:
            _section(log, f"method={method}  k={k}")
            if method == "attribute":
                labels, mlabel, metrics = fit_attribute(Xfit, k, params.seed, log)
            elif method == "spatial":
                labels, mlabel, metrics = fit_spatial(
                    Xfit, k, geom_fit, params.skater_max_polys, params.seed, log)
            else:
                log(f"Unknown method '{method}', skipping.")
                continue

            # Assignments over ALL codes (no_data → NaN cluster).
            assign = pd.DataFrame({"code": all_codes})
            lab_map = {c: int(l) for c, l in zip(fit_codes, labels)}
            assign["cluster_id"] = assign["code"].map(lab_map)
            assign["no_data"] = assign["code"].map(lambda c: bool(no_data.get(c, True)))
            assign["cluster_label"] = assign["cluster_id"].map(
                lambda v: ("type %d" % (int(v) + 1)) if pd.notna(v) and v >= 0
                else ("noise" if v == -1 else None))
            assign["sens_mean"] = assign["code"].map(sens_mean.to_dict()).fillna(0.0)
            assign["method"] = mlabel
            assign["n_clusters"] = int(k)
            assign["run_id"] = params.run_id

            profiles = build_profiles(params, mlabel, k, all_codes, assign["cluster_id"].to_numpy(),
                                      no_data, stacked, sens_mean, area_km2, group_labels, log)
            results.append({"method": method, "method_label": mlabel, "k": k,
                            "metrics": metrics, "assignments": assign, "profiles": profiles})
            log(f"{mlabel}: {assign['cluster_id'].notna().sum()} polygons assigned, "
                f"{len(profiles)} cluster(s). metrics={metrics}")
            progress(f"Clustered {mlabel}")

    if not results:
        return {"ok": False, "reason": "no results"}

    # Optional AI descriptions per cluster.
    for r in results:
        r["profiles"] = add_ai_descriptions(r["profiles"], params, base_dir, log)

    # ---- Write tables ----
    _section(log, "Writing outputs")
    seg_df = pd.concat([r["assignments"] for r in results], ignore_index=True)
    if "name_gis_geocodegroup" not in seg_df.columns:
        seg_df.insert(1, "name_gis_geocodegroup", params.layer)
    prof_df = pd.DataFrame([p for r in results for p in r["profiles"]])
    _write_parquet_coexist(gpq / "tbl_seg_mv.parquet", seg_df, params.run_id, log)
    _write_parquet_coexist(gpq / "tbl_seg_mv_profile.parquet", prof_df, params.run_id, log)

    # ---- File exports ----
    out_dir = base_dir / "output" / "segmentation_mv" / params.run_id
    geom4326 = geom
    try:
        if geom.crs is not None and "4326" not in str(geom.crs):
            geom4326 = geom.to_crs(4326)
    except Exception:
        pass
    export_gpkg(out_dir, results, geom4326, log)
    write_summary_md(out_dir, params, results, log)
    if params.make_png:
        write_png_maps(out_dir, results, geom4326, log)
    # Persist the exact parameter set for reproducibility / re-runs.
    try:
        (out_dir / "params.json").write_text(json.dumps(params.as_dict(), indent=2), encoding="utf-8")
    except Exception:
        pass

    progress("Finished", done=total_steps)
    _section(log, "Done")
    log(f"run_id={params.run_id}  results={len(results)}  output={out_dir}")
    return {"ok": True, "run_id": params.run_id, "results": len(results),
            "out_dir": str(out_dir), "layer": params.layer}


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def run(base_dir: str, master=None, **params):
    """In-process entry (kept for symmetry with other helpers; this helper is
    normally launched as a subprocess so its heavy deps stay out of mesa.exe)."""
    bd = Path(mesa_shared.find_base_dir(base_dir))
    cfg = mesa_shared.read_config(bd)
    p = params_from_config(cfg, **params)
    return run_segmentation(bd, p)


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="MESA — Sensitivity generalisation (run)")
    ap.add_argument("--original_working_directory", required=False, default=None)
    ap.add_argument("--layer", default=None, help="geocode layer (blank = basic_mosaic/first)")
    ap.add_argument("--n-clusters", dest="n_clusters", default=None, help="int or list, e.g. 4,8,16")
    ap.add_argument("--method", default=None, choices=[None, "attribute", "spatial", "both"])
    ap.add_argument("--pressure", default=None)
    ap.add_argument("--features", default=None, help="comma list of: " + ",".join(ALL_FEATURES))
    ap.add_argument("--min-area-m2", dest="min_area_m2", default=None)
    ap.add_argument("--skater-max-polys", dest="skater_max_polys", default=None)
    ap.add_argument("--ai", dest="ai_enabled", action="store_true", default=None)
    ap.add_argument("--png", dest="make_png", action="store_true", default=None)
    ap.add_argument("--run-id", dest="run_id", default=None, help="reuse to reproduce a run")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    base_dir = Path(mesa_shared.find_base_dir(args.original_working_directory))
    cfg = mesa_shared.read_config(base_dir)
    overrides = {k: v for k, v in vars(args).items()
                 if k != "original_working_directory" and v is not None}
    params = params_from_config(cfg, **overrides)
    log = make_logger(base_dir)
    t0 = time.time()
    try:
        res = run_segmentation(base_dir, params, log)
    except Exception as exc:
        log(f"FATAL: {exc!r}")
        raise
    log(f"Total wall time: {time.time() - t0:.1f}s")
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
