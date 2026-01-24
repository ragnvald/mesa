#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""process_all.py — Unified processing runner (Data / Lines / Analysis).

Goal
- Provide a single UI for batch-processing:
  - data_process (presentation processing)
  - lines_process (segment processing)
  - analysis_process (study area analysis)

Notes
- This script does NOT modify mesa.py wiring yet (by request).
- It tries to be safe for both dev (.py) runs and future frozen builds.

UI behavior
- Shows three checkboxes (Data / Lines / Analysis).
- A checkbox is disabled (greyed out) when required input data is missing.
- Default: all available processes are checked.

CLI behavior
- `--list` prints availability and exits.
- `--dry-run` prints what would run and exits.
- `--headless` runs without UI.

"""

from __future__ import annotations

import argparse
import configparser
import datetime
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


# ---------------------------------------------------------------------------
# Time / logging
# ---------------------------------------------------------------------------
try:
    from zoneinfo import ZoneInfo

    _OSLO_TZ = ZoneInfo("Europe/Oslo")
except Exception:
    _OSLO_TZ = None


def _ts() -> str:
    try:
        if _OSLO_TZ is not None:
            return datetime.datetime.now(_OSLO_TZ).strftime("%Y.%m.%d %H:%M:%S")
    except Exception:
        pass
    return datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------

def resolve_base_dir(original_working_directory: str | None) -> Path:
    """Resolve the mesa root folder in all modes (.py, frozen, tools/ launch)."""
    candidates: list[Path] = []

    if original_working_directory:
        candidates.append(Path(original_working_directory))

    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent)
    else:
        if "__file__" in globals():
            candidates.append(Path(__file__).resolve().parent)

    candidates.append(Path(os.getcwd()).resolve())

    def normalize(p: Path) -> Path:
        p = p.resolve()
        if p.name.lower() in ("tools", "system", "code"):
            p = p.parent
        q = p
        for _ in range(5):
            if (q / "config.ini").exists() and (q / "output").exists() and (q / "input").exists():
                return q
            q = q.parent
        return p

    for c in candidates:
        root = normalize(c)
        if (root / "config.ini").exists():
            return root

    return normalize(candidates[0])


def read_config(base_dir: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(base_dir / "config.ini", encoding="utf-8")
    return cfg


def parquet_dir(base_dir: Path, cfg: configparser.ConfigParser) -> Path:
    rel = "output/geoparquet"
    try:
        if "DEFAULT" in cfg:
            rel = str(cfg["DEFAULT"].get("parquet_folder", rel)).strip() or rel
    except Exception:
        pass
    out = (base_dir / rel).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _parquet_num_rows(path: Path) -> int | None:
    """Fast-ish row count for Parquet file or dataset folder.

    Returns None when the path does not exist or row count can't be determined.
    Uses pyarrow metadata to avoid loading full tables.
    """

    try:
        if not path.exists():
            return None
        from pyarrow import parquet as pq  # type: ignore
    except Exception:
        return None

    try:
        if path.is_file():
            return int(pq.ParquetFile(str(path)).metadata.num_rows)
        if path.is_dir():
            total = 0
            any_files = False
            for f in sorted(path.glob("*.parquet")):
                any_files = True
                try:
                    total += int(pq.ParquetFile(str(f)).metadata.num_rows)
                except Exception:
                    # Skip unreadable part files rather than failing the UI.
                    continue
            return total if any_files else None
    except Exception:
        return None
    return None


def _format_stats(flat_rows: int | None, stacked_rows: int | None) -> str:
    if flat_rows is None and stacked_rows is None:
        return ""
    parts: list[str] = []
    if flat_rows is not None:
        parts.append(f"flat={flat_rows:,}")
    if stacked_rows is not None:
        parts.append(f"stacked={stacked_rows:,}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Availability model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessAvailability:
    available: bool
    reasons: list[str]


@dataclass(frozen=True)
class ProcessPlan:
    run_data: bool
    run_lines: bool
    run_analysis: bool


def _exists_any(paths: Iterable[Path]) -> bool:
    for p in paths:
        if p.exists():
            return True
    return False


def detect_data_processing(base_dir: Path, cfg: configparser.ConfigParser) -> ProcessAvailability:
    gpq = parquet_dir(base_dir, cfg)

    required: list[tuple[str, list[Path]]] = [
        ("tbl_flat.parquet", [gpq / "tbl_flat.parquet"]),
        ("tbl_stacked dataset", [gpq / "tbl_stacked", gpq / "tbl_stacked.parquet"]),
    ]

    missing: list[str] = []
    for label, paths in required:
        if not _exists_any(paths):
            missing.append(label)

    if missing:
        return ProcessAvailability(False, ["Missing: " + ", ".join(missing)])

    return ProcessAvailability(True, [])


def detect_lines_processing(base_dir: Path, cfg: configparser.ConfigParser) -> ProcessAvailability:
    gpq = parquet_dir(base_dir, cfg)

    # lines_process can create default tbl_lines, but it needs geocode groups to derive extent.
    required = [
        ("tbl_geocode_group.parquet", [gpq / "tbl_geocode_group.parquet"]),
    ]

    missing: list[str] = []
    for label, paths in required:
        if not _exists_any(paths):
            missing.append(label)

    if missing:
        return ProcessAvailability(False, ["Missing: " + ", ".join(missing)])

    return ProcessAvailability(True, [])


def detect_analysis_processing(base_dir: Path, cfg: configparser.ConfigParser) -> ProcessAvailability:
    gpq = parquet_dir(base_dir, cfg)

    required = [
        ("tbl_analysis_group.parquet", [gpq / "tbl_analysis_group.parquet"]),
        ("tbl_analysis_polygons.parquet", [gpq / "tbl_analysis_polygons.parquet"]),
        # The analysis processing also needs the base presentation tables to clip.
        ("tbl_flat.parquet", [gpq / "tbl_flat.parquet"]),
        ("tbl_stacked dataset", [gpq / "tbl_stacked", gpq / "tbl_stacked.parquet"]),
    ]

    missing: list[str] = []
    for label, paths in required:
        if not _exists_any(paths):
            missing.append(label)

    if missing:
        return ProcessAvailability(False, ["Missing: " + ", ".join(missing)])

    return ProcessAvailability(True, [])


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------


def _log_line(base_dir: Path, log_fn: Callable[[str], None], msg: str) -> None:
    line = f"{_ts()} - {msg}"
    log_fn(line)
    try:
        with open(base_dir / "log.txt", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def _run_subprocess(
    base_dir: Path,
    log_fn: Callable[[str], None],
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
) -> int:
    try:
        _log_line(base_dir, log_fn, "Running: " + " ".join(argv))
        p = subprocess.run(
            argv,
            cwd=str(base_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if p.stdout:
            for line in p.stdout.splitlines():
                _log_line(base_dir, log_fn, f"[child] {line}")
        if p.returncode != 0:
            _log_line(base_dir, log_fn, f"ERROR: process failed (exit={p.returncode})")
        return int(p.returncode)
    except FileNotFoundError as e:
        _log_line(base_dir, log_fn, f"ERROR: failed to start process: {e}")
        return 2
    except Exception as e:
        _log_line(base_dir, log_fn, f"ERROR: failed to run process: {e}")
        return 2


# ---------------------------------------------------------------------------
# Individual processing implementations
# ---------------------------------------------------------------------------


def run_data_process(
    base_dir: Path,
    log_fn: Callable[[str], None],
    progress_fn: Callable[[float], None],
) -> None:
    """Run data_process in headless mode via subprocess.

    Rationale: code/data_process.py is large and relies on many globals initialized
    in its __main__ guard. Calling it via subprocess keeps it isolated and closer
    to how mesa.exe launches helpers.
    """

    progress_fn(0.0)

    script = base_dir / "code" / "data_process.py"
    if not script.exists():
        # fallback for legacy layout
        script = Path(__file__).resolve().parent / "data_process.py"

    argv = [
        sys.executable,
        str(script),
        "--original_working_directory",
        str(base_dir),
        "--headless",
    ]

    _log_line(base_dir, log_fn, "DATA PROCESS START")
    rc = _run_subprocess(base_dir, log_fn, argv)
    if rc != 0:
        raise RuntimeError(f"data_process failed (exit={rc})")

    progress_fn(100.0)
    _log_line(base_dir, log_fn, "DATA PROCESS COMPLETED")


# ---- Lines processing: integrated from code/lines_process.py (Parquet-only)

def run_lines_process(
    base_dir: Path,
    cfg: configparser.ConfigParser,
    log_fn: Callable[[str], None],
    progress_fn: Callable[[float], None],
) -> None:
    import math

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import pyproj
    from shapely import wkb as _shp_wkb
    from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
    from shapely.ops import polygonize, transform, unary_union

    gpq = parquet_dir(base_dir, cfg)

    def parquet_path(name: str) -> Path:
        return gpq / f"{name}.parquet"

    def _maybe_decode_wkb_geometry(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if "geometry" not in df.columns:
            return df

        s = df["geometry"]
        # Common cases:
        # - geopandas/pyarrow stores shapely objects directly -> OK
        # - pandas reads WKB bytes/memoryview -> decode
        sample = None
        try:
            sample = next((v for v in s.values[:50] if v is not None), None)
        except Exception:
            try:
                sample = s.dropna().iloc[0]
            except Exception:
                sample = None

        if isinstance(sample, (bytes, bytearray, memoryview)):
            def _loads(v):
                if v is None:
                    return None
                if isinstance(v, memoryview):
                    v = v.tobytes()
                if isinstance(v, bytearray):
                    v = bytes(v)
                if isinstance(v, bytes):
                    try:
                        return _shp_wkb.loads(v)
                    except Exception:
                        return None
                return v

            try:
                df = df.copy()
                df["geometry"] = df["geometry"].map(_loads)
            except Exception:
                pass
        return df

    def read_parquet_or_none(name: str, *, columns: list[str] | None = None, needs_geometry: bool = True):
        p = parquet_path(name)
        if not p.exists():
            return None
        try:
            if needs_geometry:
                return gpd.read_parquet(p, columns=columns)
            # Read as a plain table when geometry is not needed (avoids GeoParquet metadata requirements).
            return pd.read_parquet(p, columns=columns)
        except Exception as e:
            _log_line(base_dir, log_fn, f"Failed reading {p.name}: {e} (falling back to pandas)")
            try:
                df = pd.read_parquet(p, columns=columns)
                df = _maybe_decode_wkb_geometry(df)
                if needs_geometry and "geometry" in df.columns:
                    try:
                        gdf = gpd.GeoDataFrame(df, geometry="geometry")
                        try:
                            gdf = gdf.set_crs(workingprojection_epsg, allow_override=True)
                        except Exception:
                            pass
                        return gdf
                    except Exception as geo_exc:
                        _log_line(base_dir, log_fn, f"GeoDataFrame coercion failed for {p.name}: {geo_exc}")
                return df
            except Exception as e2:
                _log_line(base_dir, log_fn, f"Fallback read failed for {p.name}: {e2}")
                return None

    def write_parquet(name: str, gdf: gpd.GeoDataFrame) -> None:
        p = parquet_path(name)
        try:
            gdf.to_parquet(p, index=False)
            _log_line(base_dir, log_fn, f"Saved {p.name} (rows={len(gdf)})")
        except Exception as e:
            _log_line(base_dir, log_fn, f"Parquet write failed {p.name}: {e}")

    def read_config_classification(config_path: Path) -> dict:
        ccfg = configparser.ConfigParser()
        ccfg.read(config_path, encoding="utf-8")
        classification = {}
        for section in ccfg.sections():
            rng = ccfg[section].get("range", "").strip()
            if not rng:
                continue
            try:
                start, end = map(int, rng.split("-"))
            except Exception:
                continue
            classification[section] = {
                "range": range(start, end + 1),
                "description": ccfg[section].get("description", ""),
            }
        return classification

    def _coerce_int_or_none(v):
        try:
            if pd.isna(v):
                return None
            return int(round(float(v)))
        except Exception:
            return None

    def apply_classification_to_gdf(gdf, column_name, classes_dict, code_suffix=""):
        base_name, *suffix = column_name.rsplit("_", 1)
        suffix = suffix[0] if suffix else ""
        final_suffix = suffix if not code_suffix else code_suffix

        new_code_col = f"{base_name}_code_{final_suffix}" if final_suffix else f"{base_name}_code"
        new_desc_col = (
            f"{base_name}_description_{final_suffix}" if final_suffix else f"{base_name}_description"
        )

        if not classes_dict:
            gdf[new_code_col] = "Unknown"
            gdf[new_desc_col] = "No description available"
            return gdf, new_code_col, new_desc_col

        def classify_value(v):
            iv = _coerce_int_or_none(v)
            if iv is None:
                return "Unknown", "No description available"
            for label, info in classes_dict.items():
                rng = info.get("range", range(0))
                if iv in rng:
                    return label, info.get("description", "")
            return "Unknown", "No description available"

        codes, descs = zip(*gdf[column_name].apply(classify_value))
        gdf[new_code_col] = list(codes)
        gdf[new_desc_col] = list(descs)
        return gdf, new_code_col, new_desc_col

    workingprojection_epsg = f"EPSG:{cfg['DEFAULT'].get('workingprojection_epsg', '4326')}" if "DEFAULT" in cfg else "EPSG:4326"

    def load_lines_table():
        gdf = read_parquet_or_none("tbl_lines")
        if gdf is not None and not getattr(gdf, "empty", True):
            return gdf
        return None

    def create_lines_table_and_lines():
        geocode_group = read_parquet_or_none("tbl_geocode_group")
        if geocode_group is None or geocode_group.empty:
            _log_line(base_dir, log_fn, "Cannot derive extent (no geocode groups in Parquet).")
            return
        _log_line(base_dir, log_fn, "Creating template lines (Parquet).")
        minx, miny, maxx, maxy = geocode_group.total_bounds
        lines = []
        for _ in range(3):
            sx = np.random.uniform(minx, maxx)
            sy = np.random.uniform(miny, maxy)
            ex = np.random.uniform(minx, maxx)
            ey = np.random.uniform(miny, maxy)
            lines.append(LineString([(sx, sy), (ex, ey)]))
        gdf_lines = gpd.GeoDataFrame(
            {
                "name_gis": [f"line_{i:03d}" for i in range(1, 4)],
                "name_user": [f"line_{i:03d}" for i in range(1, 4)],
                "segment_length": [15, 30, 10],
                "segment_width": [1000, 20000, 5000],
                "description": ["auto line", "auto line", "auto line"],
                "geometry": lines,
            },
            geometry="geometry",
            crs=geocode_group.crs or workingprojection_epsg,
        )
        gdf_lines = gdf_lines.set_crs(workingprojection_epsg, allow_override=True)
        write_parquet("tbl_lines", gdf_lines)

    def process_and_buffer_lines():
        target_crs = "EPSG:4087"
        lines_df = load_lines_table()

        if lines_df is None:
            _log_line(base_dir, log_fn, "No lines found; creating defaults.")
            create_lines_table_and_lines()
            lines_df = load_lines_table()
            if lines_df is None:
                _log_line(base_dir, log_fn, "Aborting: lines still missing.")
                return

        buffered_records = []
        for idx, row in lines_df.iterrows():
            try:
                geom = row.geometry
                seg_len = int(row["segment_length"])
                seg_w = int(row["segment_width"])
                name_gis = row["name_gis"]
                name_usr = row["name_user"]
                desc = row.get("description", "")

                _log_line(base_dir, log_fn, f"Buffering {name_gis}")
                tmp = gpd.GeoDataFrame([{ "geometry": geom }], geometry="geometry", crs=workingprojection_epsg).to_crs(target_crs)
                tmp["geometry"] = tmp.buffer(seg_w, cap_style=2)
                back = tmp.to_crs(workingprojection_epsg)
                gbuf = back.iloc[0].geometry
                if not isinstance(gbuf, (Polygon, MultiPolygon)):
                    _log_line(base_dir, log_fn, f"Unexpected buffered geom type: {type(gbuf)}")
                buffered_records.append(
                    {
                        "fid": idx,
                        "name_gis": name_gis,
                        "name_user": name_usr,
                        "segment_length": seg_len,
                        "segment_width": seg_w,
                        "description": desc,
                        "geometry": gbuf,
                    }
                )
            except Exception as e:
                _log_line(base_dir, log_fn, f"Line {idx} failed: {e}")

        if not buffered_records:
            _log_line(base_dir, log_fn, "No buffered lines produced.")
            return

        buffered_gdf = gpd.GeoDataFrame(buffered_records, geometry="geometry", crs=workingprojection_epsg)
        write_parquet("tbl_lines_buffered", buffered_gdf)
        _log_line(base_dir, log_fn, "Buffered lines ready (Parquet).")

    def create_perpendicular_lines(line_input: LineString, segment_width, segment_length):
        segment_width = float(segment_width)
        segment_length = float(segment_length)

        transformer_to_4087 = pyproj.Transformer.from_crs(workingprojection_epsg, "EPSG:4087", always_xy=True)
        transformer_back = pyproj.Transformer.from_crs("EPSG:4087", workingprojection_epsg, always_xy=True)

        line_trans = transform(transformer_to_4087.transform, line_input)
        full_len = line_trans.length
        num_segments = math.ceil(full_len / segment_length)

        perpendicular_lines = []
        for i in range(num_segments + 1):
            d = min(i * segment_length, full_len)
            point = line_trans.interpolate(d)

            if d < segment_width:
                seg = LineString([line_trans.interpolate(0), line_trans.interpolate(segment_width)])
            elif d > full_len - segment_width:
                seg = LineString([line_trans.interpolate(full_len - segment_width), line_trans.interpolate(full_len)])
            else:
                seg = LineString([
                    line_trans.interpolate(d - segment_width / 2),
                    line_trans.interpolate(d + segment_width / 2),
                ])

            dx = seg.coords[1][0] - seg.coords[0][0]
            dy = seg.coords[1][1] - seg.coords[0][1]
            angle = math.atan2(-dx, dy) if dx != 0 else (math.pi / 2 if dy > 0 else -math.pi / 2)
            length = (segment_width / 2) * 3
            dxp = math.cos(angle) * length
            dyp = math.sin(angle) * length

            p1 = Point(point.x - dxp, point.y - dyp)
            p2 = Point(point.x + dxp, point.y + dyp)

            p1b = transform(transformer_back.transform, p1)
            p2b = transform(transformer_back.transform, p2)
            perpendicular_lines.append(LineString([p1b, p2b]))

        return MultiLineString(perpendicular_lines)

    def cut_into_segments(perpendicular_lines: MultiLineString, buffered_line_geometry: Polygon):
        line_list = [line for line in perpendicular_lines.geoms]
        combined_lines = unary_union([buffered_line_geometry.boundary] + line_list)
        result_polygons = list(polygonize(combined_lines))
        return gpd.GeoDataFrame(geometry=result_polygons)

    def create_segments_from_buffered_lines():
        lines_df = load_lines_table()
        if lines_df is None:
            _log_line(base_dir, log_fn, "No lines for segment creation.")
            return

        buffered = read_parquet_or_none("tbl_lines_buffered")
        if buffered is None or buffered.empty:
            _log_line(base_dir, log_fn, "No buffered lines found; run buffering first.")
            return

        all_segments = []
        counter: dict[str, int] = {}
        for _, row in lines_df.iterrows():
            name_gis = row["name_gis"]
            seg_w = row["segment_width"]
            seg_l = row["segment_length"]
            geom = row.geometry
            if name_gis not in counter:
                counter[name_gis] = 1

            try:
                seg_w_f = float(seg_w)
                seg_l_f = float(seg_l)
            except Exception:
                _log_line(base_dir, log_fn, f"Invalid segment parameters for {name_gis}: length={seg_l} width={seg_w}")
                continue
            if not (seg_l_f > 0 and seg_w_f > 0):
                _log_line(base_dir, log_fn, f"Non-positive segment parameters for {name_gis}: length={seg_l_f} width={seg_w_f}")
                continue

            try:
                perp = create_perpendicular_lines(geom, seg_w_f, seg_l_f)
            except Exception as e:
                _log_line(base_dir, log_fn, f"Perpendicular gen failed {name_gis}: {e}")
                continue
            blines = buffered[buffered["name_gis"] == name_gis]
            if blines.empty:
                continue
            t0 = datetime.datetime.now(_OSLO_TZ) if _OSLO_TZ else datetime.datetime.now()
            _log_line(base_dir, log_fn, f"Segmenting {name_gis}")
            for _, brow in blines.iterrows():
                bgeom = brow.geometry
                if not isinstance(bgeom, Polygon):
                    continue
                try:
                    segs = cut_into_segments(perp, bgeom)
                except Exception as e:
                    _log_line(base_dir, log_fn, f"Segmentation failed {name_gis}: {e}")
                    continue
                try:
                    segs = segs[segs.is_valid]
                except Exception:
                    pass
                if segs.empty:
                    continue
                segs["segment_id"] = [f"{name_gis}_{counter[name_gis] + i}" for i in range(len(segs))]
                counter[name_gis] += len(segs)
                segs["name_gis"] = name_gis
                segs["name_user"] = row["name_user"]
                segs["segment_length"] = seg_l_f
                all_segments.append(segs)

            try:
                dt_s = (datetime.datetime.now(_OSLO_TZ) if _OSLO_TZ else datetime.datetime.now()) - t0
                _log_line(base_dir, log_fn, f"Segmented {name_gis} in {dt_s.total_seconds():.2f}s")
            except Exception:
                pass

        if not all_segments:
            _log_line(base_dir, log_fn, "No segments produced.")
            return

        seg_all = gpd.GeoDataFrame(pd.concat(all_segments, ignore_index=True), geometry="geometry", crs=lines_df.crs)
        write_parquet("tbl_segments", seg_all)
        _log_line(base_dir, log_fn, "Segments saved (Parquet).")

    def intersection_with_geocode_data(asset_df, segment_df, geom_type):
        _log_line(base_dir, log_fn, f"Processing {geom_type} intersections")
        asset_filtered = asset_df[asset_df.geometry.geom_type == geom_type]
        if asset_filtered.empty:
            return gpd.GeoDataFrame()
        return gpd.sjoin(segment_df, asset_filtered, how="inner", predicate="intersects")

    def build_stacked_data():
        _log_line(base_dir, log_fn, "Building tbl_segment_stacked (Parquet)…")

        asset_data = read_parquet_or_none("tbl_asset_object", needs_geometry=True)
        # Asset group is a lookup table; geometry is not required for the merge.
        group_cols = [
            "id",
            "name_gis_assetgroup",
            "total_asset_objects",
            "importance",
            "susceptibility",
            "sensitivity",
            "sensitivity_code",
            "sensitivity_description",
        ]
        group_data = read_parquet_or_none("tbl_asset_group", columns=group_cols, needs_geometry=False)
        segments = read_parquet_or_none("tbl_segments", needs_geometry=True)

        if asset_data is None or asset_data.empty or segments is None or segments.empty:
            _log_line(base_dir, log_fn, "Missing assets or segments; cannot build segment stacked.")
            return

        if group_data is not None and not group_data.empty:
            merge_cols = [c for c in group_cols if c in group_data.columns]
            if "id" in merge_cols:
                asset_data = asset_data.merge(
                    group_data[merge_cols],
                    left_on="ref_asset_group",
                    right_on="id",
                    how="left",
                )
            else:
                _log_line(base_dir, log_fn, "WARNING: tbl_asset_group is missing 'id'; skipping group merge.")
        else:
            _log_line(base_dir, log_fn, "WARNING: tbl_asset_group missing/unreadable; skipping group merge.")

        segments = segments.set_crs(workingprojection_epsg, allow_override=True)
        asset_data = asset_data.set_crs(workingprojection_epsg, allow_override=True)

        parts = []
        for gt in asset_data.geometry.geom_type.unique():
            parts.append(intersection_with_geocode_data(asset_data, segments, gt))
        stacked = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if stacked.empty:
            _log_line(base_dir, log_fn, "No segment intersections.")
            return

        stacked = gpd.GeoDataFrame(stacked, geometry="geometry", crs=workingprojection_epsg)
        stacked.reset_index(drop=True, inplace=True)
        stacked["fid"] = stacked.index

        classes = read_config_classification(base_dir / "config.ini")
        if "sensitivity" in stacked.columns:
            stacked, codecol, desccol = apply_classification_to_gdf(stacked, "sensitivity", classes)
            if codecol != "sensitivity_code":
                stacked.rename(columns={codecol: "sensitivity_code"}, inplace=True)
            if desccol != "sensitivity_description":
                stacked.rename(columns={desccol: "sensitivity_description"}, inplace=True)

        write_parquet("tbl_segment_stacked", stacked)
        _log_line(base_dir, log_fn, f"tbl_segment_stacked rows: {len(stacked)}")

    def build_flat_data():
        _log_line(base_dir, log_fn, "Building tbl_segment_flat (Parquet)…")
        stacked = read_parquet_or_none("tbl_segment_stacked")
        if stacked is None or stacked.empty:
            _log_line(base_dir, log_fn, "No stacked segment data.")
            return

        # Only aggregate fields that exist (if group merge was skipped, these may be missing).
        agg: dict[str, object] = {
            "name_gis": "first",
            "segment_id": "first",
            "geometry": "first",
        }
        for col in ("importance", "sensitivity", "susceptibility"):
            if col in stacked.columns:
                agg[col] = ["min", "max"]
            else:
                _log_line(base_dir, log_fn, f"NOTE: '{col}' missing in tbl_segment_stacked; skipping in flat stats.")
        flat = stacked.groupby("segment_id").agg(agg)
        flat.columns = ["_".join(c).strip() for c in flat.columns]
        rename_map = {"name_gis_first": "name_gis", "geometry_first": "geometry"}
        flat.rename(columns=rename_map, inplace=True)
        flat.reset_index(inplace=True)
        if "segment_id_first" in flat.columns:
            flat.drop(columns=["segment_id_first"], inplace=True)

        flat = gpd.GeoDataFrame(flat, geometry="geometry", crs=stacked.crs)

        classes = read_config_classification(base_dir / "config.ini")
        if "sensitivity_min_min" in flat.columns:
            flat["sensitivity_min"] = flat.get("sensitivity_min", flat["sensitivity_min_min"])
            flat.drop(columns=["sensitivity_min_min"], inplace=True)
        if "sensitivity_max_max" in flat.columns:
            flat["sensitivity_max"] = flat.get("sensitivity_max", flat["sensitivity_max_max"])
            flat.drop(columns=["sensitivity_max_max"], inplace=True)

        if "sensitivity_min" in flat.columns:
            flat, cmin, dmin = apply_classification_to_gdf(flat, "sensitivity_min", classes, code_suffix="min")
            if cmin != "sensitivity_code_min":
                flat.rename(columns={cmin: "sensitivity_code_min"}, inplace=True)
            if dmin != "sensitivity_description_min":
                flat.rename(columns={dmin: "sensitivity_description_min"}, inplace=True)
        if "sensitivity_max" in flat.columns:
            flat, cmax, dmax = apply_classification_to_gdf(flat, "sensitivity_max", classes, code_suffix="max")
            if cmax != "sensitivity_code_max":
                flat.rename(columns={cmax: "sensitivity_code_max"}, inplace=True)
            if dmax != "sensitivity_description_max":
                flat.rename(columns={dmax: "sensitivity_description_max"}, inplace=True)

        write_parquet("tbl_segment_flat", flat)
        _log_line(base_dir, log_fn, f"tbl_segment_flat rows: {len(flat)}")

    _log_line(base_dir, log_fn, "LINES PROCESS START (Parquet)")
    progress_fn(1.0)
    process_and_buffer_lines()
    progress_fn(35.0)
    create_segments_from_buffered_lines()
    progress_fn(70.0)
    build_stacked_data()
    progress_fn(85.0)
    build_flat_data()
    progress_fn(100.0)
    _log_line(base_dir, log_fn, "LINES PROCESS COMPLETED")


def run_analysis_process(
    base_dir: Path,
    cfg: configparser.ConfigParser,
    log_fn: Callable[[str], None],
    progress_fn: Callable[[float], None],
) -> None:
    from analysis_process import AssetAnalyzer
    from data_analysis_setup import AnalysisStorage

    def set_progress(v: float) -> None:
        try:
            v = float(v)
        except Exception:
            return
        progress_fn(max(0.0, min(100.0, v)))

    _log_line(base_dir, log_fn, "ANALYSIS PROCESS START")
    set_progress(1.0)

    storage = AnalysisStorage(base_dir, cfg)
    analyzer = AssetAnalyzer(base_dir, cfg)
    set_progress(5.0)

    groups = storage.list_groups()
    polygon_counts: dict[str, int] = {}
    total_polygons = 0
    for g in groups:
        try:
            count = len(storage.list_records(g.identifier))
        except Exception:
            count = 0
        polygon_counts[g.identifier] = count
        total_polygons += count

    _log_line(base_dir, log_fn, f"Found {len(groups)} analysis group(s) with {total_polygons} polygon(s) total.")

    work_groups = [g for g in groups if polygon_counts.get(g.identifier, 0) > 0]
    total_work = len(work_groups)
    if total_work == 0:
        _log_line(base_dir, log_fn, "No study area polygons found; nothing to do. Use the 'Set up analysis' tool first.")
        set_progress(0.0)
        return

    GROUP_PHASE_MAX = 90.0
    group_range = GROUP_PHASE_MAX / float(total_work)

    processed = 0
    skipped = len(groups) - total_work
    failed = 0

    for idx, group in enumerate(work_groups, start=1):
        count = polygon_counts.get(group.identifier, 0)
        group_start = (idx - 1) * group_range
        group_end = idx * group_range
        set_progress(group_start)
        _log_line(base_dir, log_fn, f"Processing group '{group.name}' ({group.identifier}) with {count} polygon(s)...")

        def _on_record_progress(done: int, total: int) -> None:
            if total <= 0:
                return
            within_group = done / float(total)
            set_progress(group_start + (within_group * group_range))

        try:
            result = analyzer.run_group_analysis(
                group,
                storage.list_records(group.identifier),
                geocode=getattr(group, "default_geocode", None),
                progress_callback=_on_record_progress,
            )
            _log_line(
                base_dir,
                log_fn,
                f"Completed group '{group.name}': flat_rows={result.get('flat_rows')} stacked_rows={result.get('stacked_rows')}",
            )
            processed += 1
        except Exception as exc:
            failed += 1
            _log_line(base_dir, log_fn, f"ERROR processing group '{group.name}' ({group.identifier}): {exc}")
        finally:
            set_progress(group_end)

    set_progress(GROUP_PHASE_MAX)
    _log_line(base_dir, log_fn, f"COMPLETED: processed_groups={processed} skipped_groups={skipped} failed_groups={failed}")
    set_progress(95.0)
    set_progress(100.0)
    _log_line(base_dir, log_fn, "ANALYSIS PROCESS COMPLETED")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_selected(
    base_dir: Path,
    cfg: configparser.ConfigParser,
    plan: ProcessPlan,
    log_fn: Callable[[str], None],
    progress_fn: Callable[[float], None],
) -> None:
    selected = [
        ("data", plan.run_data),
        ("lines", plan.run_lines),
        ("analysis", plan.run_analysis),
    ]
    active = [name for name, enabled in selected if enabled]
    if not active:
        _log_line(base_dir, log_fn, "Nothing selected; exiting.")
        progress_fn(0.0)
        return

    # Allocate progress evenly across selected processes.
    slice_size = 100.0 / float(len(active))

    def make_slice_progress(offset: float) -> Callable[[float], None]:
        def _slice(p: float) -> None:
            p = max(0.0, min(100.0, float(p)))
            progress_fn(min(100.0, offset + (p / 100.0) * slice_size))

        return _slice

    current_offset = 0.0

    if plan.run_data:
        try:
            run_data_process(base_dir, log_fn, make_slice_progress(current_offset))
        except Exception as exc:
            _log_line(base_dir, log_fn, f"ERROR: data processing failed: {exc}")
        current_offset += slice_size

    if plan.run_lines:
        try:
            run_lines_process(base_dir, cfg, log_fn, make_slice_progress(current_offset))
        except Exception as exc:
            _log_line(base_dir, log_fn, f"ERROR: lines processing failed: {exc}")
        current_offset += slice_size

    if plan.run_analysis:
        try:
            run_analysis_process(base_dir, cfg, log_fn, make_slice_progress(current_offset))
        except Exception as exc:
            _log_line(base_dir, log_fn, f"ERROR: analysis processing failed: {exc}")
        current_offset += slice_size

    progress_fn(100.0)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def run_ui(base_dir: Path, cfg: configparser.ConfigParser) -> None:
    import tkinter as tk
    import tkinter.scrolledtext as scrolledtext

    try:
        import ttkbootstrap as tb
    except Exception:
        tb = None

    theme = "flatly"
    try:
        if "DEFAULT" in cfg:
            theme = cfg["DEFAULT"].get("ttk_bootstrap_theme", theme) or theme
    except Exception:
        pass

    root = tb.Window(themename=theme) if tb is not None else tk.Tk()
    root.title("MESA – Process all")
    try:
        ico = base_dir / "system_resources" / "mesa.ico"
        if ico.exists():
            root.iconbitmap(str(ico))
    except Exception:
        pass
    root.geometry("820x520")

    # Make the log area the resizable region so buttons never get pushed off-screen.
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # Availability
    avail_data = detect_data_processing(base_dir, cfg)
    avail_lines = detect_lines_processing(base_dir, cfg)
    avail_analysis = detect_analysis_processing(base_dir, cfg)

    # Defaults: checked if available
    var_data = tk.BooleanVar(value=avail_data.available)
    var_lines = tk.BooleanVar(value=avail_lines.available)
    var_analysis = tk.BooleanVar(value=avail_analysis.available)

    # --- layout (match other processing tools)
    main_frame = tk.Frame(root)
    main_frame.grid(row=0, column=0, sticky="nsew")
    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_rowconfigure(0, weight=1)

    # Log output in a labeled frame
    if tb is not None:
        log_frame = tb.LabelFrame(main_frame, text="Log output", bootstyle="info")
    else:
        from tkinter import ttk

        log_frame = ttk.LabelFrame(main_frame, text="Log output")
    log_frame.grid(row=0, column=0, padx=10, pady=(10, 6), sticky="nsew")

    log_widget = scrolledtext.ScrolledText(log_frame, height=8)
    log_widget.pack(fill=tk.BOTH, expand=True)

    # Progress bar row
    progress_frame = tk.Frame(main_frame)
    progress_frame.grid(row=1, column=0, pady=(0, 6))

    progress_var = tk.DoubleVar(value=0.0)
    if tb is not None:
        progress_bar = tb.Progressbar(
            progress_frame,
            orient="horizontal",
            length=280,
            mode="determinate",
            variable=progress_var,
            bootstyle="info",
        )
    else:
        from tkinter import ttk

        progress_bar = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            length=280,
            mode="determinate",
            variable=progress_var,
        )
    progress_bar.pack(side=tk.LEFT)

    progress_label = tk.Label(progress_frame, text="0%", bg="light grey")
    progress_label.pack(side=tk.LEFT, padx=5)

    def ui_log(line: str) -> None:
        log_widget.insert(tk.END, line + "\n")
        log_widget.see(tk.END)

    def ui_progress(p: float) -> None:
        p = max(0.0, min(100.0, float(p)))
        progress_var.set(p)
        progress_label.config(text=f"{int(p)}%")

    # Info text (like other tools)
    info_label_text = (
        "Run one or more processing steps in one batch. "
        "Unavailable steps are disabled when required input tables are missing."
    )
    info_label = tk.Label(main_frame, text=info_label_text, wraplength=760, justify="left", anchor="w")
    info_label.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 6))

    # Controls (checkbox rows)
    controls = tk.Frame(main_frame)
    controls.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 6))
    controls.grid_columnconfigure(0, weight=0)
    controls.grid_columnconfigure(1, weight=1)
    controls.grid_columnconfigure(2, weight=1)

    # Header row
    hdr1 = tk.Label(controls, text="Process", anchor="w")
    hdr2 = tk.Label(controls, text="Status", anchor="w")
    hdr3 = tk.Label(controls, text="Current results", anchor="w")
    hdr1.grid(row=0, column=0, sticky="w")
    hdr2.grid(row=0, column=1, sticky="w", padx=10)
    hdr3.grid(row=0, column=2, sticky="w", padx=10)

    gpq = parquet_dir(base_dir, cfg)
    stats_data = _format_stats(
        _parquet_num_rows(gpq / "tbl_flat.parquet"),
        _parquet_num_rows(gpq / "tbl_stacked") or _parquet_num_rows(gpq / "tbl_stacked.parquet"),
    )
    stats_lines = _format_stats(
        _parquet_num_rows(gpq / "tbl_segment_flat.parquet"),
        _parquet_num_rows(gpq / "tbl_segment_stacked.parquet"),
    )
    stats_analysis = _format_stats(
        _parquet_num_rows(gpq / "tbl_analysis_flat.parquet"),
        _parquet_num_rows(gpq / "tbl_analysis_stacked.parquet"),
    )

    def _mk_row(row: int, text: str, var: tk.BooleanVar, avail: ProcessAvailability, stats: str):
        cb = tk.Checkbutton(controls, text=text, variable=var)
        cb.grid(row=row, column=0, sticky="w")
        status = "Ready" if avail.available else ("; ".join(avail.reasons) if avail.reasons else "Missing inputs")
        lbl = tk.Label(controls, text=status, anchor="w")
        lbl.grid(row=row, column=1, sticky="w", padx=10)
        st = tk.Label(controls, text=stats, anchor="w")
        st.grid(row=row, column=2, sticky="w", padx=10)
        if not avail.available:
            cb.configure(state=tk.DISABLED)
            var.set(False)

    _mk_row(1, "Data processing (presentation)", var_data, avail_data, stats_data)
    _mk_row(2, "Lines processing (segments)", var_lines, avail_lines, stats_lines)
    _mk_row(3, "Analysis processing (study areas)", var_analysis, avail_analysis, stats_analysis)

    buttons = tk.Frame(main_frame)
    buttons.grid(row=4, column=0, sticky="ew", padx=20, pady=(0, 10))

    if tb is not None:
        Button = tb.Button
    else:
        Button = tk.Button

    button_width = 18
    button_padx = 7
    button_pady = 7

    if tb is not None:
        process_btn = Button(buttons, text="Process selected", width=button_width, bootstyle="primary")
        exit_btn = Button(buttons, text="Exit", command=root.destroy, width=button_width, bootstyle="warning")
    else:
        process_btn = Button(buttons, text="Process selected", width=button_width)
        exit_btn = Button(buttons, text="Exit", command=root.destroy, width=button_width)

    process_btn.pack(side=tk.LEFT, padx=button_padx, pady=button_pady)
    exit_btn.pack(side=tk.RIGHT, padx=button_padx, pady=button_pady)

    def worker() -> None:
        process_btn.configure(state=tk.DISABLED)
        try:
            plan = ProcessPlan(
                run_data=bool(var_data.get()),
                run_lines=bool(var_lines.get()),
                run_analysis=bool(var_analysis.get()),
            )

            def log_from_worker(msg: str) -> None:
                root.after(0, lambda: ui_log(msg))

            def progress_from_worker(v: float) -> None:
                root.after(0, lambda: ui_progress(v))

            run_selected(base_dir, cfg, plan, log_from_worker, progress_from_worker)
            root.after(0, lambda: ui_log(f"{_ts()} - ALL SELECTED PROCESSING COMPLETED"))
        except Exception as e:
            # IMPORTANT: don't close over `e` in a deferred callback; the exception
            # variable is cleared after the except block.
            err_msg = f"{_ts()} - ERROR: {e}"
            root.after(0, lambda m=err_msg: ui_log(m))
        finally:
            root.after(0, lambda: process_btn.configure(state=tk.NORMAL))

    def on_click() -> None:
        t = threading.Thread(target=worker, daemon=True)
        t.start()

    process_btn.configure(command=on_click)

    # Initial log
    ui_log(f"{_ts()} - Base dir: {base_dir}")

    root.mainloop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MESA – Unified processing runner (data/lines/analysis)")
    p.add_argument("--original_working_directory", default=None, help="Mesa base directory")
    p.add_argument("--headless", action="store_true", help="Run without GUI")
    p.add_argument("--list", action="store_true", help="Print availability and exit")
    p.add_argument("--dry-run", action="store_true", help="Print planned actions and exit")

    p.add_argument("--no-data", action="store_true", help="Do not run data processing")
    p.add_argument("--no-lines", action="store_true", help="Do not run lines processing")
    p.add_argument("--no-analysis", action="store_true", help="Do not run analysis processing")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = resolve_base_dir(args.original_working_directory)
    cfg = read_config(base_dir)

    avail_data = detect_data_processing(base_dir, cfg)
    avail_lines = detect_lines_processing(base_dir, cfg)
    avail_analysis = detect_analysis_processing(base_dir, cfg)

    default_plan = ProcessPlan(
        run_data=avail_data.available and not bool(args.no_data),
        run_lines=avail_lines.available and not bool(args.no_lines),
        run_analysis=avail_analysis.available and not bool(args.no_analysis),
    )

    if args.list:
        print("Base dir:", base_dir)
        print("Data:", "OK" if avail_data.available else "NO", "; ".join(avail_data.reasons))
        print("Lines:", "OK" if avail_lines.available else "NO", "; ".join(avail_lines.reasons))
        print("Analysis:", "OK" if avail_analysis.available else "NO", "; ".join(avail_analysis.reasons))
        return

    if args.dry_run:
        selected = []
        if default_plan.run_data:
            selected.append("data")
        if default_plan.run_lines:
            selected.append("lines")
        if default_plan.run_analysis:
            selected.append("analysis")
        print("Base dir:", base_dir)
        print("Would run:", ", ".join(selected) if selected else "(nothing)")
        return

    if args.headless:
        def log_print(line: str) -> None:
            print(line)

        def progress_print(p: float) -> None:
            # keep it simple; caller can watch log.txt for details
            print(f"progress={int(max(0, min(100, float(p))))}%")

        run_selected(base_dir, cfg, default_plan, log_print, progress_print)
        return

    run_ui(base_dir, cfg)


if __name__ == "__main__":
    main()
