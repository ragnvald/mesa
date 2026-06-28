#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""segmentation_run.py — Classification engine (sensitivity-pattern clustering).

WHAT THIS DOES
    Groups geocode cells by the *shape* of their sensitivity composition in the
    (importance, susceptibility) plane, rather than by the A–E sensitivity product.
    Sensitivity is a many-to-one product of importance × susceptibility, so a cell
    holding an (importance 5, susceptibility 1) asset and one holding (importance 1,
    susceptibility 5) carry the *same* code yet describe very different places. This
    engine clusters directly on the joint (importance, susceptibility) histogram, so
    those two cells can land in different types. See the wiki page
    "Segmentation and clustering".

METHOD (count-based, area-weighted aggregation)
    Per cell: a normalised histogram over the (importance, susceptibility) bins
    (counts of overlapping asset rows, → proportions), plus a coverage/intensity
    index (stack depth). Per-asset intersection areas are not persisted upstream, so
    within a cell each overlap is weighted equally; cross-cell comparability comes
    from area-weighting at the cluster-aggregation step (cell area_m2 IS available).
    The histogram is Hellinger- (or CLR-) transformed, an optional standardised
    coverage feature is appended, and Gaussian Mixture Models are fit for every k in
    a range; k is chosen by minimum BIC. Soft posteriors give per-cell certainty
    (p_max, entropy). Results are validated against the deterministic signatures
    (tbl_segmentation) with ARI and NMI.

CALLED BY
    mesa.exe launcher / Classification setup (subprocess), or standalone:
      python code/segmentation_run.py --original_working_directory <dir>

OUTPUTS (geometry-free; join to tbl_geocode_object on `code` at render time)
    output/geoparquet/tbl_seg_mv.parquet          per-cell assignments + certainty
    output/geoparquet/tbl_seg_mv_profile.parquet  per-cluster fingerprint + stats
    output/segmentation_mv/<run_id>/              summary.md, GeoPackage, params.json,
                                                  optional PNG fingerprints/maps

MEMORY DISCIPLINE (see CLAUDE.md + learning.md "Parent-side memory in the pipeline")
    Reads tbl_stacked one layer at a time with a pyarrow filter; never materialises
    the whole stack. The heavy compute libs (scikit-learn) are imported lazily.
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
COL_IMP = "importance"
COL_SUS = "susceptibility"
COL_GROUP_ID = "ref_asset_group"        # FK to tbl_asset_group.id
COL_GROUP_NAME = "name_gis_assetgroup"  # human-ish label carried in tbl_stacked

# Equal-area CRS for honest km² (matches segmentation.py).
EQUAL_AREA_EPSG = 6933

# Fallback valuation scale when config has none.
DEFAULT_SCALE = (1, 2, 3, 4, 5)


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
    """Emit machine-readable progress markers the Classification setup GUI parses
    to drive its bar: '@@SEGMV_PROGRESS <done> <total> <label>'. Harmless when run
    standalone."""
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
    """Full, reproducible parameter set for one run. Serialised into params.json
    and summary.md so a run_id round-trips to identical output."""

    def __init__(self, **kw):
        self.run_id: str = kw.get("run_id") or datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.layer: str = kw.get("layer") or ""          # resolved later if blank
        self.k_min: int = int(kw.get("k_min") or 2)
        self.k_max: int = int(kw.get("k_max") or 15)
        self.transform: str = (kw.get("transform") or "hellinger").lower()  # hellinger|clr
        self.coverage_weight: float = float(kw.get("coverage_weight", 1.0))
        self.pressure: str = kw.get("pressure") or ""    # "" / "all" = aggregate
        self.min_area_m2: float = float(kw.get("min_area_m2") or 0.0)
        self.ai_enabled: bool = bool(kw.get("ai_enabled") or False)
        self.ollama_url: str = kw.get("ollama_url") or "http://localhost:11434/api/generate"
        self.ollama_model: str = kw.get("ollama_model") or "mistral"
        self.make_png: bool = bool(kw.get("make_png") or False)
        self.seed: int = int(kw.get("seed") or 42)
        if self.k_max < self.k_min:
            self.k_min, self.k_max = self.k_max, self.k_min
        self.k_min = max(2, self.k_min)
        if self.transform not in ("hellinger", "clr"):
            self.transform = "hellinger"

    def k_values(self) -> list[int]:
        return list(range(self.k_min, self.k_max + 1))

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id, "layer": self.layer,
            "k_range": f"{self.k_min}-{self.k_max}",
            "transform": self.transform, "coverage_weight": self.coverage_weight,
            "pressure": self.pressure, "min_area_m2": self.min_area_m2,
            "ai_enabled": self.ai_enabled, "ollama_model": self.ollama_model,
            "make_png": self.make_png, "seed": self.seed,
        }


def _parse_k_range(raw, default=(2, 15)) -> tuple[int, int]:
    """Parse 'min-max' (or a single int) into (k_min, k_max)."""
    if raw is None:
        return default
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        return int(raw[0]), int(raw[1])
    s = str(raw).strip()
    for sep in ("-", ":", ".."):
        if sep in s:
            a, _, b = s.partition(sep)
            if a.strip().isdigit() and b.strip().isdigit():
                return int(a), int(b)
    if s.isdigit():
        return int(s), int(s)
    return default


def params_from_config(cfg, **overrides) -> Params:
    """Build Params from config.ini segmv_* keys, with explicit overrides winning."""
    def g(key, default=""):
        try:
            return (cfg["DEFAULT"].get(key, default) or default).split("#", 1)[0].strip()
        except Exception:
            return default

    def _truthy(v):
        return str(v).strip().lower() in ("1", "true", "yes", "on")

    k_raw = overrides.get("k_range") if overrides.get("k_range") is not None else g("segmv_k_range", "2-15")
    k_min, k_max = _parse_k_range(k_raw)

    return Params(
        run_id=overrides.get("run_id"),
        layer=overrides.get("layer") if overrides.get("layer") is not None else g("segmv_geocode_layer", ""),
        k_min=k_min, k_max=k_max,
        transform=overrides.get("transform") or g("segmv_transform", "hellinger"),
        coverage_weight=(overrides.get("coverage_weight") if overrides.get("coverage_weight") is not None
                         else g("segmv_coverage_weight", "1.0")),
        pressure=overrides.get("pressure") if overrides.get("pressure") is not None else g("segmv_pressure", ""),
        min_area_m2=overrides.get("min_area_m2") if overrides.get("min_area_m2") is not None else g("segmv_min_area_m2", "0"),
        ai_enabled=overrides.get("ai_enabled") if overrides.get("ai_enabled") is not None else _truthy(g("segmv_ai_enabled", "0")),
        ollama_url=overrides.get("ollama_url") or g("segmv_ollama_url", "http://localhost:11434/api/generate"),
        ollama_model=overrides.get("ollama_model") or g("segmv_ollama_model", "mistral"),
        make_png=overrides.get("make_png") if overrides.get("make_png") is not None else _truthy(g("segmv_make_png", "0")),
        seed=overrides.get("seed") or 42,
    )


def valuation_scale(cfg) -> list[int]:
    """The registered importance/susceptibility scale (shared), from
    config.ini [VALID_VALUES] valid_input. Falls back to 1..5. Both axes use it."""
    try:
        raw = cfg["VALID_VALUES"]["valid_input"]
        vals = sorted({int(x.strip()) for x in str(raw).split(",") if x.strip().lstrip("-").isdigit()})
        vals = [v for v in vals if 0 <= v <= 9999]
        return vals or list(DEFAULT_SCALE)
    except Exception:
        return list(DEFAULT_SCALE)


# ---------------------------------------------------------------------------
# Input reads
# ---------------------------------------------------------------------------
def _stacked_source(gpq: Path) -> Optional[str]:
    d, f = Path(gpq) / "tbl_stacked", Path(gpq) / "tbl_stacked.parquet"
    if d.exists():
        return str(d)
    if f.exists():
        return str(f)
    return None


def detect_pressure_columns(gpq: Path) -> list[str]:
    """tbl_stacked columns that look like a pressure identifier (usually none)."""
    import pyarrow.dataset as ds
    src = _stacked_source(gpq)
    if src is None:
        return []
    try:
        names = ds.dataset(src, format="parquet").schema.names
    except Exception:
        return []
    return [n for n in names if "pressure" in n.lower()]


def _asset_group_valuation(gpq: Path):
    """ref_asset_group → (importance, susceptibility) lookup from tbl_asset_group,
    used to backfill rows where tbl_stacked didn't materialise the numeric fields."""
    import pandas as pd
    p = Path(gpq) / "tbl_asset_group.parquet"
    if not p.exists():
        return pd.DataFrame(columns=[COL_GROUP_ID, COL_IMP, COL_SUS])
    try:
        df = pd.read_parquet(p)
    except Exception:
        return pd.DataFrame(columns=[COL_GROUP_ID, COL_IMP, COL_SUS])
    if "id" in df.columns and COL_GROUP_ID not in df.columns:
        df = df.rename(columns={"id": COL_GROUP_ID})
    keep = [c for c in (COL_GROUP_ID, COL_IMP, COL_SUS) if c in df.columns]
    df = df[keep].copy()
    if COL_GROUP_ID in df.columns:
        df[COL_GROUP_ID] = df[COL_GROUP_ID].astype(str)
    return df


def read_layer_stacked(gpq: Path, layer: str, pressure: str, log) -> "object":
    """Partitioned, filtered read of tbl_stacked for one layer. Returns a DataFrame
    with code, importance, susceptibility (backfilled from tbl_asset_group where the
    stack didn't materialise them) and ref_asset_group for top-group naming."""
    import pandas as pd
    import pyarrow.dataset as ds

    src = _stacked_source(gpq)
    base_cols = [COL_CODE, COL_LAYER, COL_IMP, COL_SUS, COL_GROUP_ID, COL_GROUP_NAME]
    if src is None:
        return pd.DataFrame(columns=base_cols)

    dataset = ds.dataset(src, format="parquet")
    present = set(dataset.schema.names)
    wanted = [c for c in base_cols if c in present]
    press_cols = [c for c in present if "pressure" in c.lower()]
    if pressure and pressure.lower() not in ("", "all") and press_cols:
        wanted = list(dict.fromkeys(wanted + press_cols))
    flt = ds.field(COL_LAYER) == layer
    try:
        df = dataset.to_table(columns=wanted, filter=flt).to_pandas()
    except Exception:
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

    if df.empty:
        return df
    df[COL_CODE] = df[COL_CODE].astype(str)
    if COL_GROUP_ID in df.columns:
        df[COL_GROUP_ID] = df[COL_GROUP_ID].astype(str)

    # Optional pressure filter (only if a pressure column actually exists).
    if pressure and pressure.lower() not in ("", "all") and press_cols:
        pc = press_cols[0]
        df = df[df[pc].astype(str) == str(pressure)]

    # Backfill importance/susceptibility from tbl_asset_group when missing/blank.
    need_imp = COL_IMP not in df.columns or pd.to_numeric(df.get(COL_IMP), errors="coerce").isna().all()
    need_sus = COL_SUS not in df.columns or pd.to_numeric(df.get(COL_SUS), errors="coerce").isna().all()
    if (need_imp or need_sus) and COL_GROUP_ID in df.columns:
        lut = _asset_group_valuation(gpq)
        if not lut.empty:
            df = df.merge(lut, on=COL_GROUP_ID, how="left", suffixes=("", "_grp"))
            for col in (COL_IMP, COL_SUS):
                gcol = f"{col}_grp"
                if gcol in df.columns:
                    base = pd.to_numeric(df.get(col), errors="coerce") if col in df.columns else None
                    grp = pd.to_numeric(df[gcol], errors="coerce")
                    df[col] = grp if base is None else base.fillna(grp)
                    df.drop(columns=[gcol], inplace=True)
    for col in (COL_IMP, COL_SUS):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
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


def read_layer_geometry(gpq: Path, layer: str, codes):
    """GeoDataFrame (code, geometry) for the given codes, in the file's native CRS."""
    import geopandas as gpd
    p = Path(gpq) / "tbl_geocode_object.parquet"
    g = gpd.read_parquet(p, columns=[COL_CODE, COL_LAYER, "geometry"])
    g = g[g[COL_LAYER].astype(str) == str(layer)].copy()
    g[COL_CODE] = g[COL_CODE].astype(str)
    g = g[g[COL_CODE].isin(set(map(str, codes)))]
    return g.drop_duplicates(COL_CODE).set_index(COL_CODE)


def per_code_area_km2(gpq: Path, layer: str, geom_gdf):
    """km² per code from the geocode geometry in an equal-area CRS."""
    import pandas as pd
    try:
        g = geom_gdf.to_crs(EQUAL_AREA_EPSG)
        return (g.geometry.area / 1e6)
    except Exception:
        return pd.Series(dtype="float64")


# ---------------------------------------------------------------------------
# Histograms over the (importance, susceptibility) plane
# ---------------------------------------------------------------------------
def _nearest_index(values, scale_arr):
    """Map each value to the index of the nearest scale level; NaN → -1."""
    import numpy as np
    vals = np.asarray(values, dtype="float64")
    out = np.full(vals.shape, -1, dtype="int64")
    ok = ~np.isnan(vals)
    if ok.any():
        out[ok] = np.abs(vals[ok][:, None] - scale_arr[None, :]).argmin(axis=1)
    return out


def hist_columns(scale: list[int]) -> list[str]:
    """Column names for the joint histogram, importance-major order."""
    return [f"h_i{i}_s{s}" for i in scale for s in scale]


def build_histograms(stacked, all_codes, scale: list[int], log):
    """Per-cell normalised (importance, susceptibility) histogram + stack depth.

    Returns (prop_df, depth, no_data) where prop_df is indexed by code over
    *all_codes* (rows summing to 1, or 0 for no_data) with one column per (i,s) bin,
    depth is the overlap count per code, and no_data flags empty-stack cells."""
    import numpy as np
    import pandas as pd

    cols = hist_columns(scale)
    nv = len(scale)
    idx = pd.Index(all_codes, name=COL_CODE)

    if stacked is None or stacked.empty:
        log("No stacked rows for this layer — every polygon is no_data.")
        prop = pd.DataFrame(0.0, index=idx, columns=cols)
        return prop, pd.Series(0, index=idx), pd.Series(True, index=idx)

    scale_arr = np.asarray(scale, dtype="float64")
    df = stacked[[COL_CODE, COL_IMP, COL_SUS]].copy()
    ii = _nearest_index(df[COL_IMP].to_numpy(), scale_arr)
    si = _nearest_index(df[COL_SUS].to_numpy(), scale_arr)
    keep = (ii >= 0) & (si >= 0)
    df = df.loc[keep].copy()
    df["_bin"] = ii[keep] * nv + si[keep]            # importance-major, matches hist_columns
    df[COL_CODE] = df[COL_CODE].astype(str)

    # Counts per (code, bin) → wide proportions.
    ct = (df.groupby([COL_CODE, "_bin"]).size().unstack(fill_value=0)
          .reindex(columns=range(nv * nv), fill_value=0))
    ct.columns = cols
    ct = ct.reindex(idx, fill_value=0)
    depth = ct.sum(axis=1).astype("int64")
    no_data = depth <= 0
    denom = depth.replace(0, 1)
    prop = ct.div(denom, axis=0).astype("float64")
    log(f"Histograms: {len(idx)} cells × {len(cols)} bins "
        f"({int(no_data.sum())} no_data); scale {scale[0]}..{scale[-1]} ({nv}×{nv}).")
    return prop, depth, no_data


# ---------------------------------------------------------------------------
# Transform + GMM clustering (BIC-selected k, soft assignments)
# ---------------------------------------------------------------------------
def _transform(prop_df, kind: str):
    """Hellinger (sqrt) or CLR on histogram proportions. Returns an ndarray."""
    import numpy as np
    P = prop_df.to_numpy(dtype="float64")
    if kind == "clr":
        eps = 1e-6
        L = np.log(P + eps)
        return L - L.mean(axis=1, keepdims=True)
    return np.sqrt(P)  # Hellinger; handles zeros without pseudocounts


def fit_gmm_bic(Xfit, k_values, seed, log):
    """Fit diagonal-covariance GMMs for every k, pick min-BIC. Returns
    (labels, proba, best_k, bic_table). Deterministic for a fixed seed."""
    import warnings
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.mixture import GaussianMixture

    n = Xfit.shape[0]
    ks = [k for k in k_values if 1 < k < n] or [min(2, max(1, n - 1))]
    best = None
    bic_table = []
    for k in ks:
        gm = GaussianMixture(n_components=k, covariance_type="diag",
                             random_state=seed, n_init=5, reg_covar=1e-6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            gm.fit(Xfit)
        bic = float(gm.bic(Xfit))
        bic_table.append((k, round(bic, 2)))
        log(f"  k={k:>3}  BIC={bic:,.1f}")
        if best is None or bic < best[1]:
            best = (k, bic, gm)
    best_k, _best_bic, gm = best
    proba = gm.predict_proba(Xfit)
    labels = proba.argmax(axis=1)
    log(f"Selected k={best_k} by minimum BIC.")
    return np.asarray(labels), proba, best_k, bic_table


def _posterior_stats(proba):
    """p_max and Shannon entropy (nats) of each cell's posterior vector."""
    import numpy as np
    p = np.clip(proba, 1e-12, 1.0)
    p_max = proba.max(axis=1)
    entropy = -(proba * np.log(p)).sum(axis=1)
    return p_max, entropy


# ---------------------------------------------------------------------------
# Validation against the deterministic signatures
# ---------------------------------------------------------------------------
def validate_against_signatures(gpq: Path, layer: str, assign_df, log) -> dict:
    """ARI + NMI between GMM cluster labels and the Segment signatures for the same
    cells. Returns {} when tbl_segmentation for the layer is unavailable."""
    import pandas as pd
    try:
        import segmentation as _seg
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    except Exception:
        return {}
    part = Path(gpq) / "tbl_segmentation" / f"{_seg._safe_layer_name(layer)}.parquet"
    if not part.exists():
        log("Signatures table (tbl_segmentation) not found for this layer — "
            "run the Segment sub-stage to enable ARI/NMI validation.")
        return {}
    try:
        sig = pd.read_parquet(part, columns=[COL_CODE, "signature"])
    except Exception:
        return {}
    sig[COL_CODE] = sig[COL_CODE].astype(str)
    a = assign_df[[COL_CODE, "cluster_id"]].dropna(subset=["cluster_id"]).copy()
    a[COL_CODE] = a[COL_CODE].astype(str)
    m = a.merge(sig, on=COL_CODE, how="inner")
    m = m[m["signature"].astype(str) != ""]
    if len(m) < 2 or m["cluster_id"].nunique() < 2 or m["signature"].nunique() < 2:
        return {}
    sig_codes = m["signature"].astype("category").cat.codes.to_numpy()
    clu = m["cluster_id"].astype(int).to_numpy()
    return {
        "ari": round(float(adjusted_rand_score(sig_codes, clu)), 4),
        "nmi": round(float(normalized_mutual_info_score(sig_codes, clu)), 4),
        "n": int(len(m)),
    }


# ---------------------------------------------------------------------------
# Profiles (per-cluster fingerprint)
# ---------------------------------------------------------------------------
def build_profiles(params, best_k, all_codes, labels, no_data, prop_df, depth,
                   stacked, area_km2, mean_imp, mean_sus, group_labels, scale, log):
    """One profile row per cluster: area-weighted mean histogram (fingerprint),
    area-weighted mean importance/susceptibility, mean coverage, area + count, and
    top-3 asset groups."""
    import numpy as np
    import pandas as pd

    cols = hist_columns(scale)
    rows = []
    code_idx = pd.Index(all_codes, name=COL_CODE)
    fit_mask = ~no_data.reindex(code_idx).fillna(True).to_numpy()
    label_arr = np.asarray(labels)

    ar = (area_km2.reindex(code_idx).fillna(0.0).to_numpy()
          if area_km2 is not None else np.zeros(len(code_idx)))
    dp = depth.reindex(code_idx).fillna(0).to_numpy()
    mi = mean_imp.reindex(code_idx).to_numpy()
    ms = mean_sus.reindex(code_idx).to_numpy()
    P = prop_df.reindex(code_idx)[cols].to_numpy()

    grp_contrib = None
    if stacked is not None and not stacked.empty and COL_GROUP_ID in stacked.columns:
        grp_contrib = stacked.groupby([COL_CODE, COL_GROUP_ID]).size()

    for lab in sorted(set(label_arr.tolist())):
        sel = (label_arr == lab) & fit_mask
        if sel.sum() == 0:
            continue
        sub_codes = [c for c, s in zip(all_codes, sel) if s]
        w = ar[sel]
        wsum = float(w.sum())
        if wsum <= 0:                      # area unavailable → equal weights
            w = np.ones(int(sel.sum())); wsum = float(w.sum())
        fp = (P[sel] * w[:, None]).sum(axis=0) / wsum     # area-weighted mean histogram
        top3 = []
        if grp_contrib is not None:
            sub = grp_contrib[grp_contrib.index.get_level_values(0).isin(sub_codes)]
            if not sub.empty:
                agg = sub.groupby(level=1).sum().sort_values(ascending=False).head(3)
                top3 = [group_labels.get(str(gid), str(gid)) for gid in agg.index]
        row = {
            "run_id": params.run_id,
            "name_gis_geocodegroup": params.layer,
            "n_clusters": int(best_k),
            "cluster_id": int(lab),
            "cluster_label": f"type {int(lab) + 1}",
            "n_polygons": int(sel.sum()),
            "total_area_km2": round(float(ar[sel].sum()), 4),
            "mean_importance": round(float(np.nansum(mi[sel] * w) / wsum), 3),
            "mean_susceptibility": round(float(np.nansum(ms[sel] * w) / wsum), 3),
            "mean_coverage_index": round(float((dp[sel] * w).sum() / wsum), 3),
            "top_asset_groups": ", ".join(top3),
            "description_ai": None,
        }
        for c, v in zip(cols, fp):
            row[c] = round(float(v), 6)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# AI descriptions (optional, default off)
# ---------------------------------------------------------------------------
def _ai_context(profile_row) -> str:
    return (
        f"This is a sensitivity-pattern type from a classification of an environmental "
        f"sensitivity analysis, clustered in the (importance, susceptibility) plane. "
        f"Type '{profile_row['cluster_label']}' covers {profile_row['n_polygons']} cells "
        f"({profile_row['total_area_km2']} km²), area-weighted mean importance "
        f"{profile_row['mean_importance']} and susceptibility {profile_row['mean_susceptibility']}, "
        f"mean overlap depth {profile_row['mean_coverage_index']}. Dominant asset groups: "
        f"{profile_row['top_asset_groups'] or 'none'}. Write ONE short plain-language "
        f"paragraph (<=60 words) describing what kind of area this is for a spatial "
        f"planner. No preamble."
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


def _ai_secrets(base_dir) -> dict:
    """AI connection settings from secrets/ai_connection.parquet — the central
    store written by the Config tab's "AI connection" box. Returns {} if absent.
    Keeps the OpenAI token out of config.ini/output/ and survives "Clear output"."""
    try:
        p = Path(base_dir) / "secrets" / "ai_connection.parquet"
        if not p.exists():
            return {}
        import pandas as pd
        df = pd.read_parquet(p)
        if df is None or df.empty:
            return {}
        row = df.iloc[-1].to_dict()
        return {k: ("" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v))
                for k, v in row.items()}
    except Exception:
        return {}


def _ai_overrides(base_dir) -> dict:
    """Ollama endpoint overrides from the AI connection store, so the Config tab's
    "AI connection" box wins over config.ini's segmv_ollama_* defaults. The OpenAI
    token is consumed separately in _openai_*; it is not surfaced as a param."""
    s = _ai_secrets(base_dir)
    out = {}
    if s.get("ollama_url"):
        out["ollama_url"] = s["ollama_url"]
    if s.get("ollama_model"):
        out["ollama_model"] = s["ollama_model"]
    return out


def _openai_describe(prompt: str, base_dir: Path, log) -> Optional[str]:
    try:
        key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if not key:  # central store (Config → AI connection) is the primary home
            key = (_ai_secrets(base_dir).get("openai_token") or "").strip()
        if not key:  # legacy fallback
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


def _ollama_status(url: str, model: str, timeout: float = 2.0) -> str:
    """Fast liveness/model probe. Returns 'ok', 'down' (no server answering),
    or 'no_model' (server up but `model` not pulled). One quick check up front
    avoids a 60s-per-cluster urlopen timeout when Ollama is unavailable."""
    try:
        import urllib.request
        from urllib.parse import urlsplit, urlunsplit
        parts = urlsplit(url)
        tags_url = urlunsplit((parts.scheme, parts.netloc, "/api/tags", "", ""))
        with urllib.request.urlopen(tags_url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return "down"
    names = [str(m.get("name", "")) for m in (data.get("models") or [])]
    base = (model or "").split(":")[0]
    if any(n == model or n.split(":")[0] == base for n in names):
        return "ok"
    return "no_model"


def _openai_available(base_dir: Path) -> bool:
    if (os.environ.get("OPENAI_API_KEY") or "").strip():
        return True
    if (_ai_secrets(base_dir).get("openai_token") or "").strip():
        return True
    try:
        return (Path(base_dir) / "secrets" / "openai.key").exists()
    except Exception:
        return False


def add_ai_descriptions(profile_rows, params, base_dir, log):
    if not params.ai_enabled or not profile_rows:
        return profile_rows

    # Quiet no-AI fallback: probe the backends once. If neither Ollama nor an
    # OpenAI key is available, log a single line and leave descriptions null —
    # never hammer a dead endpoint with one timing-out call per cluster.
    status = _ollama_status(params.ollama_url, params.ollama_model)
    use_ollama = status == "ok"
    use_openai = _openai_available(base_dir)
    if not use_ollama and not use_openai:
        reason = (f"Ollama not reachable at {params.ollama_url}" if status == "down"
                  else f"Ollama model '{params.ollama_model}' not installed")
        log(f"AI descriptions skipped: {reason}, and no OpenAI key. "
            f"Leaving descriptions null.")
        return profile_rows

    _section(log, "AI cluster descriptions")
    for row in profile_rows:
        prompt = _ai_context(row)
        txt = _ollama_describe(prompt, params.ollama_url, params.ollama_model, log) if use_ollama else None
        if not txt and use_openai:
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
            # Schema migration: prior Classification runs used a different engine and
            # column set. Mixing schemas would yield a half-NaN union table, so when
            # the columns no longer match we replace the file rather than merge.
            if set(old.columns) != set(new_df.columns):
                log(f"{path.name}: schema changed since the previous engine — "
                    f"replacing {len(old)} obsolete row(s) from earlier runs.")
            else:
                old = old[old.get("run_id").astype(str) != str(run_id)] if "run_id" in old.columns else old
                new_df = pd.concat([old, new_df], ignore_index=True)
        except Exception as exc:
            log(f"WARNING: could not merge existing {path.name} ({exc}); overwriting run rows only.")
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_parquet(path, index=False, compression="zstd", compression_level=3)
    log(f"Wrote {path.name}: {len(new_df)} total row(s).")


def export_gpkg(out_dir: Path, assign_df, geom_by_code, log):
    """A single GPKG layer: polygons + cluster_id/label/certainty."""
    import geopandas as gpd
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg = out_dir / "classification_results.gpkg"
    try:
        g = geom_by_code.join(assign_df.set_index(COL_CODE), how="inner").reset_index()
        gdf = gpd.GeoDataFrame(g, geometry="geometry", crs=geom_by_code.crs)
        gdf.to_file(gpkg, layer="classification", driver="GPKG")
        log(f"Wrote {gpkg.name}.")
    except Exception as exc:
        log(f"WARNING: GPKG export failed ({exc}).")
    return gpkg


def write_summary_md(out_dir: Path, params, best_k, bic_table, profiles, validation,
                     scale, log):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Classification — run summary", "",
             f"- **run_id**: `{params.run_id}`",
             f"- **geocode layer**: {params.layer}",
             f"- **k range**: {params.k_min}–{params.k_max}  → **chosen k = {best_k}** (min BIC)",
             f"- **transform**: {params.transform}",
             f"- **coverage weight**: {params.coverage_weight}",
             f"- **valuation scale**: {scale[0]}..{scale[-1]} ({len(scale)}×{len(scale)} bins)",
             f"- **pressure filter**: {params.pressure or 'all (aggregate)'}",
             f"- **min polygon area (m²)**: {params.min_area_m2}",
             f"- **seed**: {params.seed}", "",
             "Cells are clustered by the *shape* of their (importance, susceptibility) "
             "histogram — answering *\"what kind of sensitivity pattern is this place "
             "part of?\"*, complementary to the A–E *\"how sensitive is this place?\"* "
             "classification.", ""]

    lines += ["## Model selection (BIC per k)", "", "| k | BIC |", "|---:|---:|"]
    for k, bic in bic_table:
        mark = "  ← chosen" if k == best_k else ""
        lines.append(f"| {k} | {bic:,.1f}{mark} |")
    lines.append("")

    lines += ["## Validation against signatures", ""]
    if validation:
        lines.append(f"- **Adjusted Rand Index (ARI)**: {validation['ari']}")
        lines.append(f"- **Normalised Mutual Information (NMI)**: {validation['nmi']}  "
                     f"(over {validation['n']} cells)")
        agree = "high" if validation["nmi"] >= 0.5 else ("moderate" if validation["nmi"] >= 0.2 else "low")
        if agree == "low":
            interp = ("Low agreement: the clustering sees structure the A–E sensitivity "
                      "categories do not capture.")
        elif agree == "high":
            interp = ("High agreement: the sensitivity categories already capture most of "
                      "this structure.")
        else:
            interp = ("Moderate agreement: the clustering partly overlaps the sensitivity "
                      "categories but adds distinctions of its own.")
        lines += ["", interp, ""]
    else:
        lines += ["_Signatures (tbl_segmentation) unavailable for this layer — run the "
                  "Segment sub-stage to enable ARI/NMI._", ""]

    lines += ["## Types", "",
              "| Type | Cells | Area km² | Mean imp | Mean sus | Coverage | Top asset groups | AI |",
              "|---|---:|---:|---:|---:|---:|---|---|"]
    for p in profiles:
        lines.append(
            f"| {p['cluster_label']} | {p['n_polygons']} | {p['total_area_km2']} | "
            f"{p['mean_importance']} | {p['mean_susceptibility']} | {p['mean_coverage_index']} | "
            f"{p['top_asset_groups']} | {p.get('description_ai') or ''} |")
    lines.append("")
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"Wrote {path.name}.")
    return path


def _fingerprint_png(ax, fp_vec, scale):
    """Draw a cluster's mean histogram as a scale×scale heatmap (importance rows,
    susceptibility columns) onto a matplotlib axis."""
    import numpy as np
    nv = len(scale)
    grid = np.asarray(fp_vec, dtype="float64").reshape(nv, nv)  # rows=importance, cols=susceptibility
    ax.imshow(grid, cmap="magma", origin="lower", vmin=0.0,
              vmax=max(1e-6, float(grid.max())), aspect="equal")
    ax.set_xticks(range(nv)); ax.set_xticklabels(scale, fontsize=7)
    ax.set_yticks(range(nv)); ax.set_yticklabels(scale, fontsize=7)
    ax.set_xlabel("susceptibility", fontsize=7)
    ax.set_ylabel("importance", fontsize=7)


def write_png_outputs(out_dir: Path, profiles, assign_df, geom_by_code, scale, log):
    """Per-type fingerprint heatmaps (always) + a categorical Types map (if geometry)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        log(f"PNG output skipped (matplotlib unavailable: {exc}).")
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = hist_columns(scale)

    # Fingerprint grid: one small heatmap per type.
    try:
        n = len(profiles)
        if n:
            ncol = min(5, n)
            nrow = (n + ncol - 1) // ncol
            fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.4 * nrow), squeeze=False)
            for i, p in enumerate(profiles):
                ax = axes[i // ncol][i % ncol]
                _fingerprint_png(ax, [p[c] for c in cols], scale)
                ax.set_title(p["cluster_label"], fontsize=8)
            for j in range(n, nrow * ncol):
                axes[j // ncol][j % ncol].axis("off")
            fig.suptitle("Type fingerprints — mean (importance, susceptibility) histogram", fontsize=10)
            fig.tight_layout()
            fig.savefig(out_dir / "fingerprints.png", dpi=130)
            plt.close(fig)
            log("Wrote fingerprints.png.")
    except Exception as exc:
        log(f"Fingerprint PNG failed ({exc}).")

    # Categorical Types map.
    try:
        import geopandas as gpd
        g = geom_by_code.join(assign_df.set_index(COL_CODE), how="inner").reset_index()
        gdf = gpd.GeoDataFrame(g, geometry="geometry", crs=geom_by_code.crs)
        ax = gdf.plot(column="cluster_label", categorical=True, legend=True, figsize=(8, 8))
        ax.set_axis_off(); ax.set_title("Classification — types")
        ax.get_figure().savefig(out_dir / "types_map.png", dpi=120, bbox_inches="tight")
        plt.close(ax.get_figure())
        log("Wrote types_map.png.")
    except Exception as exc:
        log(f"Types map PNG failed ({exc}).")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_segmentation(base_dir, params: Params, log=None) -> dict:
    """Execute one classification run. Returns a small result dict (paths + counts)."""
    import numpy as np
    import pandas as pd

    base_dir = Path(base_dir)
    log = log or make_logger(base_dir)
    cfg = mesa_shared.read_config(base_dir)
    gpq = mesa_shared.parquet_dir(base_dir, cfg)
    scale = valuation_scale(cfg)

    _section(log, f"Classification — run_id {params.run_id}")
    log(f"Parameters: {json.dumps(params.as_dict())}")

    import segmentation as _seg
    layers = _seg.list_geocode_layers(gpq)
    if not layers:
        log("No geocode layers found — nothing to classify.")
        return {"ok": False, "reason": "no layers"}
    if not params.layer:
        params.layer = "basic_mosaic" if "basic_mosaic" in layers else layers[0]
    if params.layer not in layers:
        log(f"Layer '{params.layer}' not found. Available: {', '.join(layers)}")
        return {"ok": False, "reason": "layer not found"}
    log(f"Operating on layer: {params.layer}")

    total_steps = len(params.k_values()) + 4
    progress = make_progress(total_steps)
    progress("Reading inputs", done=0)

    stacked = read_layer_stacked(gpq, params.layer, params.pressure, log)
    all_codes = list(layer_codes(gpq, params.layer))
    if not all_codes:
        all_codes = sorted(stacked[COL_CODE].unique().tolist()) if not stacked.empty else []
    if not all_codes:
        log("No polygons for this layer.")
        return {"ok": False, "reason": "no polygons"}
    group_labels = asset_group_labels(gpq)

    geom = read_layer_geometry(gpq, params.layer, all_codes)
    area_km2 = per_code_area_km2(gpq, params.layer, geom)
    if params.min_area_m2 and params.min_area_m2 > 0 and not area_km2.empty:
        keep_codes = set(area_km2[area_km2 * 1e6 >= params.min_area_m2].index.astype(str))
        before = len(all_codes)
        all_codes = [c for c in all_codes if c in keep_codes]
        log(f"Min-area filter (≥{params.min_area_m2:.0f} m²): kept {len(all_codes)}/{before} polygons.")
        geom = geom.loc[geom.index.intersection(all_codes)]

    progress("Building histograms")
    prop_df, depth, no_data = build_histograms(stacked, all_codes, scale, log)

    # Per-cell mean importance/susceptibility (for area-weighted profile stats).
    if stacked is not None and not stacked.empty:
        mean_imp = stacked.groupby(COL_CODE)[COL_IMP].mean()
        mean_sus = stacked.groupby(COL_CODE)[COL_SUS].mean()
    else:
        mean_imp = pd.Series(dtype="float64")
        mean_sus = pd.Series(dtype="float64")

    fit_codes = [c for c, nd in zip(all_codes, no_data.to_numpy()) if not nd]
    if len(fit_codes) < 2:
        log("Fewer than 2 polygons with stacked data — cannot cluster.")
        return {"ok": False, "reason": "insufficient data"}

    # Feature matrix: transformed histogram + optional standardised coverage feature.
    Xhist = _transform(prop_df.loc[fit_codes], params.transform)
    feat_blocks = [Xhist]
    if params.coverage_weight and params.coverage_weight != 0:
        from sklearn.preprocessing import StandardScaler
        cov = depth.reindex(fit_codes).to_numpy(dtype="float64").reshape(-1, 1)
        cov_z = StandardScaler().fit_transform(cov) * float(params.coverage_weight)
        feat_blocks.append(cov_z)
    Xfit = np.hstack(feat_blocks)
    log(f"Feature matrix: {Xfit.shape[0]} cells × {Xfit.shape[1]} dims "
        f"(transform={params.transform}, coverage_weight={params.coverage_weight}).")

    _section(log, f"GMM model selection (k = {params.k_min}..{params.k_max}, diag, n_init=5)")
    labels_fit, proba, best_k, bic_table = fit_gmm_bic(Xfit, params.k_values(), params.seed, log)
    p_max_fit, entropy_fit = _posterior_stats(proba)
    for _k in params.k_values():
        progress(f"Fitted models")

    # Assignments over ALL codes (no_data → null cluster).
    lab_map = {c: int(v) for c, v in zip(fit_codes, labels_fit)}
    pmax_map = {c: float(v) for c, v in zip(fit_codes, p_max_fit)}
    ent_map = {c: float(v) for c, v in zip(fit_codes, entropy_fit)}
    assign = pd.DataFrame({COL_CODE: all_codes})
    assign["cluster_id"] = assign[COL_CODE].map(lab_map)
    assign["cluster_label"] = assign["cluster_id"].map(
        lambda v: f"type {int(v) + 1}" if pd.notna(v) else "no_data")
    assign["p_max"] = assign[COL_CODE].map(pmax_map)
    assign["entropy"] = assign[COL_CODE].map(ent_map)
    assign["coverage_index"] = assign[COL_CODE].map(depth.to_dict()).fillna(0).astype("int64")
    assign["top_bins"] = assign[COL_CODE].map(_top_bins_series(prop_df.reindex(all_codes), scale).to_dict())
    assign["name_gis_geocodegroup"] = params.layer
    assign["run_id"] = params.run_id

    validation = validate_against_signatures(gpq, params.layer, assign, log)
    if validation:
        log(f"Validation vs signatures: ARI={validation['ari']} NMI={validation['nmi']} "
            f"(n={validation['n']}).")

    progress("Building profiles")
    profiles = build_profiles(params, best_k, all_codes, assign["cluster_id"].to_numpy(),
                              no_data, prop_df, depth, stacked, area_km2,
                              mean_imp, mean_sus, group_labels, scale, log)
    profiles = add_ai_descriptions(profiles, params, base_dir, log)
    log(f"Clustered into {best_k} type(s); {int((~no_data).sum())} cells assigned.")

    # ---- Write tables ----
    _section(log, "Writing outputs")
    seg_cols = [COL_CODE, "name_gis_geocodegroup", "run_id", "cluster_id",
                "cluster_label", "p_max", "entropy", "coverage_index", "top_bins"]
    seg_df = assign[seg_cols].copy()
    prof_df = pd.DataFrame(profiles)
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
    export_gpkg(out_dir, assign[[COL_CODE, "cluster_id", "cluster_label", "p_max",
                                 "entropy", "coverage_index"]], geom4326, log)
    write_summary_md(out_dir, params, best_k, bic_table, profiles, validation, scale, log)
    if params.make_png:
        write_png_outputs(out_dir, profiles, assign[[COL_CODE, "cluster_label"]], geom4326, scale, log)
    try:
        (out_dir / "params.json").write_text(json.dumps(params.as_dict(), indent=2), encoding="utf-8")
    except Exception:
        pass

    progress("Finished", done=total_steps)
    _section(log, "Done")
    log(f"run_id={params.run_id}  k={best_k}  output={out_dir}")
    return {"ok": True, "run_id": params.run_id, "k": best_k,
            "out_dir": str(out_dir), "layer": params.layer,
            "validation": validation}


def _top_bins_series(prop_df, scale, top=3):
    """Compact 'iMPxSUS:prop' top-N string per cell for the Maps tooltip."""
    import numpy as np
    import pandas as pd
    cols = hist_columns(scale)
    nv = len(scale)
    P = prop_df[cols].to_numpy(dtype="float64")
    out = []
    for row in P:
        if not np.any(row > 0):
            out.append("")
            continue
        order = np.argsort(row)[::-1][:top]
        parts = []
        for b in order:
            if row[b] <= 0:
                continue
            imp = scale[b // nv]; sus = scale[b % nv]
            parts.append(f"{imp}×{sus}:{row[b]:.2f}")
        out.append("; ".join(parts))
    return pd.Series(out, index=prop_df.index)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def run(base_dir: str, master=None, **params):
    """In-process entry (kept for symmetry; normally launched as a subprocess)."""
    bd = Path(mesa_shared.find_base_dir(base_dir))
    cfg = mesa_shared.read_config(bd)
    p = params_from_config(cfg, **{**_ai_overrides(bd), **params})
    return run_segmentation(bd, p)


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="MESA — Classification (run)")
    ap.add_argument("--original_working_directory", required=False, default=None)
    ap.add_argument("--layer", default=None, help="geocode layer (blank = basic_mosaic/first)")
    ap.add_argument("--k-range", dest="k_range", default=None, help="min-max, e.g. 2-15")
    ap.add_argument("--transform", default=None, choices=[None, "hellinger", "clr"])
    ap.add_argument("--coverage-weight", dest="coverage_weight", default=None)
    ap.add_argument("--pressure", default=None)
    ap.add_argument("--min-area-m2", dest="min_area_m2", default=None)
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
    overrides = {**_ai_overrides(base_dir), **overrides}  # explicit CLI args win
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
