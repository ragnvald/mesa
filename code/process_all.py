#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""process_all.py — Unified processing runner (Data / Lines / Analysis).

Goal
- Provide a single UI for batch-processing:
    - Data processing (presentation processing)
    - Lines processing (segment processing)
    - Analysis processing (study area analysis)

Notes
- This tool is intended to replace the separate processing helpers.
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
import uuid
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
    explode_flat_multipolygons: bool
    run_tiles: bool
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
        ("tbl_asset_object.parquet", [gpq / "tbl_asset_object.parquet"]),
        ("tbl_geocode_object.parquet", [gpq / "tbl_geocode_object.parquet"]),
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
    ]

    missing: list[str] = []
    for label, paths in required:
        if not _exists_any(paths):
            missing.append(label)

    if missing:
        return ProcessAvailability(False, ["Missing: " + ", ".join(missing)])

    return ProcessAvailability(True, [])


def detect_tiles_processing(base_dir: Path, cfg: configparser.ConfigParser) -> ProcessAvailability:
    runner_path, _is_exe = _find_tiles_runner(base_dir)
    if not runner_path:
        return ProcessAvailability(False, ["Missing: create_raster_tiles helper"])

    gpq = parquet_dir(base_dir, cfg)
    flat_exists = _exists_any([gpq / "tbl_flat.parquet"])
    if not flat_exists:
        return ProcessAvailability(True, ["tbl_flat.parquet missing (run data first)"])

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


def _run_subprocess_streaming(
    base_dir: Path,
    log_fn: Callable[[str], None],
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    line_prefix: str = "[child]",
) -> int:
    try:
        _log_line(base_dir, log_fn, "Running: " + " ".join(argv))
        proc = subprocess.Popen(
            argv,
            cwd=str(base_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if proc.stdout:
            prefix = line_prefix or ""
            for line in proc.stdout:
                if line:
                    _log_line(base_dir, log_fn, f"{prefix} {line.rstrip()}".rstrip())
                else:
                    _log_line(base_dir, log_fn, prefix)
        proc.wait()
        if proc.returncode != 0:
            _log_line(base_dir, log_fn, f"ERROR: process failed (exit={proc.returncode})")
        return int(proc.returncode or 0)
    except FileNotFoundError as e:
        _log_line(base_dir, log_fn, f"ERROR: failed to start process: {e}")
        return 2
    except Exception as e:
        _log_line(base_dir, log_fn, f"ERROR: failed to run process: {e}")
        return 2


def _find_tiles_runner(base_dir: Path) -> tuple[Path | None, bool]:
    """Find create_raster_tiles (returns path, is_executable)."""
    frozen = bool(getattr(sys, "frozen", False))

    def _dedup(paths: Iterable[Path | None]) -> list[Path]:
        out: list[Path] = []
        seen: set[Path] = set()
        for p in paths:
            if p is None:
                continue
            try:
                resolved = p.resolve()
            except Exception:
                resolved = p
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(resolved)
        return out

    exe_candidates = _dedup(
        [
            base_dir / "tools" / "create_raster_tiles.exe",
            base_dir / "create_raster_tiles.exe",
            base_dir / "code" / "create_raster_tiles.exe",
            base_dir / "system" / "create_raster_tiles.exe",
            Path(sys.executable).resolve().parent / "create_raster_tiles.exe" if frozen else None,
        ]
    )
    py_candidates = _dedup(
        [
            base_dir / "create_raster_tiles.py",
            base_dir / "system" / "create_raster_tiles.py",
            base_dir / "code" / "create_raster_tiles.py",
            (Path(__file__).resolve().parent / "create_raster_tiles.py") if "__file__" in globals() else None,
        ]
    )

    def _pick(candidates: list[Path]) -> Path | None:
        for c in candidates:
            try:
                if c.exists():
                    return c
            except Exception:
                continue
        return None

    if frozen:
        exe_path = _pick(exe_candidates)
        if exe_path:
            return exe_path, True
        script_path = _pick(py_candidates)
        if script_path:
            return script_path, False
        return None, False

    script_path = _pick(py_candidates)
    if script_path:
        return script_path, False

    exe_path = _pick(exe_candidates)
    if exe_path:
        return exe_path, True

    return None, False


def run_tiles_process(
    base_dir: Path,
    cfg: configparser.ConfigParser,
    log_fn: Callable[[str], None],
    progress_fn: Callable[[float], None],
) -> None:
    """Run the raster tiles (MBTiles) helper."""
    progress_fn(0.0)
    _log_line(base_dir, log_fn, "TILES PROCESS START")

    runner_path, is_exe = _find_tiles_runner(base_dir)
    if not runner_path:
        _log_line(base_dir, log_fn, "ERROR: create_raster_tiles helper not found")
        raise RuntimeError("Missing create_raster_tiles helper")

    args = [str(runner_path)] if is_exe else [sys.executable, str(runner_path)]

    try:
        minzoom = int(str(cfg["DEFAULT"].get("tiles_minzoom", "")).strip() or 0)
        maxzoom = int(str(cfg["DEFAULT"].get("tiles_maxzoom", "")).strip() or 0)
        if minzoom > 0:
            args += ["--minzoom", str(minzoom)]
        if maxzoom > 0:
            args += ["--maxzoom", str(maxzoom)]
    except Exception:
        pass

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["MESA_BASE_DIR"] = str(base_dir)

    code = _run_subprocess_streaming(base_dir, log_fn, args, env=env, line_prefix="[tiles]")
    if code != 0:
        raise RuntimeError(f"Raster tiles failed (exit={code})")

    progress_fn(100.0)
    _log_line(base_dir, log_fn, "TILES PROCESS COMPLETED")


# ---------------------------------------------------------------------------
# Individual processing implementations
# ---------------------------------------------------------------------------


def run_data_process(
    base_dir: Path,
    log_fn: Callable[[str], None],
    progress_fn: Callable[[float], None],
    *,
    explode_flat_multipolygons: bool = False,
) -> None:
    """Run the data-processing pipeline.

    Important: when frozen (PyInstaller), calling a .py via `sys.executable` does not work.
    We therefore run the pipeline in-process by importing the internal module.
    """

    progress_fn(0.0)
    _log_line(base_dir, log_fn, "DATA PROCESS START")

    try:
        # Import on-demand so the main UI stays light until the user actually
        # runs the Area step.
        import _data_process_internal as dpi

        dpi.run_headless(str(base_dir), explode_flat_multipolygons=bool(explode_flat_multipolygons))
    except Exception as exc:
        _log_line(base_dir, log_fn, f"ERROR: data processing failed: {exc}")
        raise

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
    # Embedded (minimal) analysis processor so we can remove analysis_process.py.
    # This intentionally avoids importing heavy GIS modules at process_all import time.
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    from data_analysis_setup import (
        DEFAULT_ANALYSIS_GEOCODE,
        AnalysisStorage,
        analysis_flat_path,
        analysis_stacked_path,
        find_dataset_dir,
        find_parquet_file,
    )

    class AssetAnalyzer:
        """Run the actual analysis processing (clipping tbl_flat/tbl_stacked to polygons)."""

        def __init__(self, base_dir: Path, cfg_local, storage_epsg: int = 4326) -> None:
            self.base_dir = base_dir
            self.cfg = cfg_local
            self.storage_epsg = storage_epsg or 4326
            try:
                self.area_epsg = int(str(cfg_local["DEFAULT"].get("area_projection_epsg", "3035")))
            except Exception:
                self.area_epsg = 3035

            self._DEFAULT_GEOCODE = DEFAULT_ANALYSIS_GEOCODE
            self._find_parquet_file = find_parquet_file
            self._find_dataset_dir = find_dataset_dir
            self.analysis_flat_path = analysis_flat_path(base_dir, cfg_local)
            self.analysis_stacked_path = analysis_stacked_path(base_dir, cfg_local)

            self._flat_dataset: gpd.GeoDataFrame | None = None
            self._stacked_dataset: gpd.GeoDataFrame | None = None

        def _load_flat_dataset(self) -> gpd.GeoDataFrame:
            if self._flat_dataset is not None:
                return self._flat_dataset
            path = self._find_parquet_file(self.base_dir, self.cfg, "tbl_flat.parquet")
            if not path:
                raise FileNotFoundError("Presentation table missing (tbl_flat.parquet).")
            gdf = gpd.read_parquet(path)
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
            self._flat_dataset = gdf
            return gdf

        def _load_stacked_dataset(self) -> gpd.GeoDataFrame:
            if self._stacked_dataset is not None:
                return self._stacked_dataset
            path = self._find_dataset_dir(self.base_dir, self.cfg, "tbl_stacked")
            if not path:
                raise FileNotFoundError("Stacked table missing (tbl_stacked).")
            gdf = gpd.read_parquet(path)
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
            self._stacked_dataset = gdf
            return gdf

        @staticmethod
        def _subset_by_polygon(gdf: gpd.GeoDataFrame, polygon) -> gpd.GeoDataFrame:
            if gdf.empty:
                return gdf.iloc[0:0].copy()
            try:
                sindex = gdf.sindex
                candidates = list(sindex.intersection(polygon.bounds))
                subset = gdf.iloc[candidates].copy() if candidates else gdf.iloc[0:0].copy()
            except Exception:
                subset = gdf.copy()
            if subset.empty:
                return subset
            subset = subset[subset.geometry.intersects(polygon)]
            return subset.copy()

        def _clip_to_polygon(
            self,
            base_gdf: gpd.GeoDataFrame,
            polygon,
            group,
            record,
            geocode: str,
            run_id: str,
            run_ts: str,
        ) -> gpd.GeoDataFrame:
            subset = self._subset_by_polygon(base_gdf, polygon)
            if subset.empty:
                return subset.iloc[0:0].copy()

            clipped_geoms = []
            for geom in subset.geometry:
                try:
                    inter = geom.intersection(polygon)
                except Exception:
                    try:
                        inter = geom.buffer(0).intersection(polygon)
                    except Exception:
                        inter = None
                if inter is None or inter.is_empty:
                    clipped_geoms.append(None)
                else:
                    clipped_geoms.append(inter)

            subset = subset.assign(geometry=clipped_geoms)
            subset = subset[subset.geometry.notna()]
            subset = subset[~subset.geometry.is_empty]
            if subset.empty:
                return subset.iloc[0:0].copy()

            subset = gpd.GeoDataFrame(subset, geometry="geometry", crs=base_gdf.crs).copy()
            metric = subset.to_crs(self.area_epsg)
            subset["analysis_area_m2"] = metric.geometry.area.astype("float64")
            subset = subset[subset["analysis_area_m2"] > 0]
            if subset.empty:
                return subset.iloc[0:0].copy()

            base_area = pd.to_numeric(subset.get("area_m2"), errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                subset["analysis_area_fraction"] = np.where(
                    base_area > 0,
                    subset["analysis_area_m2"] / base_area.astype("float64"),
                    np.nan,
                )

            subset["analysis_group_id"] = getattr(group, "identifier", "")
            subset["analysis_group_name"] = getattr(group, "name", "")
            subset["analysis_polygon_id"] = getattr(record, "identifier", "")
            subset["analysis_polygon_title"] = getattr(record, "title", "")
            subset["analysis_polygon_notes"] = getattr(record, "notes", "")
            subset["analysis_geocode"] = geocode
            subset["analysis_run_id"] = run_id
            subset["analysis_timestamp"] = run_ts
            return subset.reset_index(drop=True)

        def _write_analysis_output(self, path: Path, group_id: str, new_gdf: gpd.GeoDataFrame) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                try:
                    existing = gpd.read_parquet(path)
                except Exception:
                    existing = gpd.GeoDataFrame(columns=new_gdf.columns, geometry="geometry", crs=new_gdf.crs)
            else:
                existing = gpd.GeoDataFrame(columns=new_gdf.columns, geometry="geometry", crs=new_gdf.crs)

            if "analysis_group_id" in existing.columns:
                mask = existing["analysis_group_id"].astype(str).fillna("") != str(group_id)
                existing = existing.loc[mask].reset_index(drop=True)
            else:
                existing = existing.iloc[0:0].copy()

            combined = existing if new_gdf.empty else pd.concat([existing, new_gdf], ignore_index=True)
            if not isinstance(combined, gpd.GeoDataFrame):
                combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=new_gdf.crs)
            combined.to_parquet(path, index=False)

        def run_group_analysis(
            self,
            group,
            records,
            geocode: str | None = None,
            progress_callback: Callable[[int, int], None] | None = None,
        ) -> dict:
            if not records:
                raise ValueError("No analysis polygons selected.")

            category = self._DEFAULT_GEOCODE
            flat_base = self._load_flat_dataset()
            stacked_base = self._load_stacked_dataset()

            if "name_gis_geocodegroup" in flat_base.columns:
                flat_base = flat_base[flat_base["name_gis_geocodegroup"].astype(str).str.strip() == category].copy()
            if "name_gis_geocodegroup" in stacked_base.columns:
                stacked_base = stacked_base[stacked_base["name_gis_geocodegroup"].astype(str).str.strip() == category].copy()

            run_id = uuid.uuid4().hex
            run_ts = datetime.datetime.utcnow().isoformat()

            total_records = len(records)
            if progress_callback is not None:
                try:
                    progress_callback(0, total_records)
                except Exception:
                    pass

            flat_results: list[gpd.GeoDataFrame] = []
            stacked_results: list[gpd.GeoDataFrame] = []

            for record_index, record in enumerate(records, start=1):
                poly_storage = gpd.GeoDataFrame(
                    [{"geometry": getattr(record, "geometry", None)}],
                    geometry="geometry",
                    crs=f"EPSG:{self.storage_epsg}",
                )
                polygon = poly_storage.to_crs(flat_base.crs).geometry.iloc[0]

                clipped_flat = self._clip_to_polygon(flat_base, polygon, group, record, category, run_id, run_ts)
                if not clipped_flat.empty:
                    flat_results.append(clipped_flat)

                clipped_stacked = self._clip_to_polygon(stacked_base, polygon, group, record, category, run_id, run_ts)
                if not clipped_stacked.empty:
                    stacked_results.append(clipped_stacked)

                if progress_callback is not None:
                    try:
                        progress_callback(record_index, total_records)
                    except Exception:
                        pass

            extra_cols = [
                "analysis_group_id",
                "analysis_group_name",
                "analysis_polygon_id",
                "analysis_polygon_title",
                "analysis_polygon_notes",
                "analysis_geocode",
                "analysis_run_id",
                "analysis_timestamp",
                "analysis_area_m2",
                "analysis_area_fraction",
            ]

            if flat_results:
                flat_gdf = gpd.GeoDataFrame(
                    pd.concat(flat_results, ignore_index=True),
                    geometry="geometry",
                    crs=flat_results[0].crs,
                )
            else:
                flat_gdf = gpd.GeoDataFrame(
                    columns=list(flat_base.columns) + extra_cols,
                    geometry="geometry",
                    crs=flat_base.crs,
                )

            if stacked_results:
                stacked_gdf = gpd.GeoDataFrame(
                    pd.concat(stacked_results, ignore_index=True),
                    geometry="geometry",
                    crs=stacked_results[0].crs,
                )
            else:
                stacked_gdf = gpd.GeoDataFrame(
                    columns=list(stacked_base.columns) + extra_cols,
                    geometry="geometry",
                    crs=stacked_base.crs,
                )

            flat_gdf = flat_gdf.reset_index(drop=True)
            stacked_gdf = stacked_gdf.reset_index(drop=True)

            self._write_analysis_output(self.analysis_flat_path, getattr(group, "identifier", ""), flat_gdf)
            self._write_analysis_output(self.analysis_stacked_path, getattr(group, "identifier", ""), stacked_gdf)

            return {
                "analysis_group_id": getattr(group, "identifier", ""),
                "analysis_group_name": getattr(group, "name", ""),
                "analysis_geocode": category,
                "flat_path": str(self.analysis_flat_path),
                "stacked_path": str(self.analysis_stacked_path),
                "flat_rows": int(len(flat_gdf)),
                "stacked_rows": int(len(stacked_gdf)),
                "run_id": run_id,
                "run_timestamp": run_ts,
            }

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
        ("tiles", plan.run_tiles),
        ("lines", plan.run_lines),
        ("analysis", plan.run_analysis),
    ]
    active = [name for name, enabled in selected if enabled]
    if not active:
        _log_line(base_dir, log_fn, "Nothing selected; exiting.")
        progress_fn(0.0)
        return

    had_errors = False
    _log_line(base_dir, log_fn, "[Process] STARTED")

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
            run_data_process(
                base_dir,
                log_fn,
                make_slice_progress(current_offset),
                explode_flat_multipolygons=bool(plan.explode_flat_multipolygons),
            )
        except Exception as exc:
            _log_line(base_dir, log_fn, f"ERROR: data processing failed: {exc}")
            had_errors = True
        current_offset += slice_size

    if plan.run_tiles:
        try:
            gpq = parquet_dir(base_dir, cfg)
            if not _exists_any([gpq / "tbl_flat.parquet"]):
                raise FileNotFoundError("tbl_flat.parquet is missing; run data processing first")
            run_tiles_process(base_dir, cfg, log_fn, make_slice_progress(current_offset))
        except Exception as exc:
            _log_line(base_dir, log_fn, f"ERROR: raster tiles failed: {exc}")
            had_errors = True
        current_offset += slice_size

    if plan.run_lines:
        try:
            run_lines_process(base_dir, cfg, log_fn, make_slice_progress(current_offset))
        except Exception as exc:
            _log_line(base_dir, log_fn, f"ERROR: lines processing failed: {exc}")
            had_errors = True
        current_offset += slice_size

    if plan.run_analysis:
        try:
            run_analysis_process(base_dir, cfg, log_fn, make_slice_progress(current_offset))
        except Exception as exc:
            _log_line(base_dir, log_fn, f"ERROR: analysis processing failed: {exc}")
            had_errors = True
        current_offset += slice_size

    progress_fn(100.0)
    _log_line(base_dir, log_fn, "[Process] FAILED" if had_errors else "[Process] COMPLETED")


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
    # Slightly bigger default window (requested): more room for options and log.
    root.geometry("900x560")

    # Make the log area the resizable region so buttons never get pushed off-screen.
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # Availability
    avail_data = detect_data_processing(base_dir, cfg)
    avail_tiles = detect_tiles_processing(base_dir, cfg)
    avail_lines = detect_lines_processing(base_dir, cfg)
    avail_analysis = detect_analysis_processing(base_dir, cfg)

    # Defaults: checked if available
    var_data = tk.BooleanVar(value=avail_data.available)
    var_data_explode = tk.BooleanVar(value=False)
    gpq = parquet_dir(base_dir, cfg)
    tiles_flat_exists = _exists_any([gpq / "tbl_flat.parquet"])
    tiles_default = bool(avail_tiles.available and (tiles_flat_exists or avail_data.available))
    var_tiles = tk.BooleanVar(value=tiles_default)
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

    # Tail base_dir/log.txt for detailed progress output (data processing writes there).
    _tail_state: dict[str, int] = {}

    def _start_log_tailer(interval_ms: int = 750) -> None:
        candidates = [base_dir / "log.txt", base_dir / "code" / "log.txt"]

        def _tail_once() -> None:
            try:
                for p in candidates:
                    try:
                        if not p.exists():
                            continue
                        key = str(p)
                        with open(p, "r", encoding="utf-8", errors="replace") as f:
                            pos = _tail_state.get(key, 0)
                            try:
                                f.seek(pos)
                            except Exception:
                                pos = 0
                                f.seek(0)
                            data = f.read()
                            _tail_state[key] = f.tell()
                        if data:
                            for line in data.splitlines():
                                if line.strip():
                                    ui_log(line)
                    except Exception:
                        pass
            finally:
                try:
                    root.after(interval_ms, _tail_once)
                except Exception:
                    pass

        # Start at current EOF to avoid dumping old logs
        for p in candidates:
            try:
                if p.exists():
                    _tail_state[str(p)] = p.stat().st_size
            except Exception:
                pass
        root.after(interval_ms, _tail_once)

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
    controls.grid_columnconfigure(2, weight=0)

    # Header row
    hdr1 = tk.Label(controls, text="Process", anchor="w")
    hdr2 = tk.Label(controls, text="Status", anchor="w")
    hdr3 = tk.Label(controls, text="Options", anchor="w")
    hdr1.grid(row=0, column=0, sticky="w")
    hdr2.grid(row=0, column=1, sticky="w", padx=10)
    hdr3.grid(row=0, column=2, sticky="w", padx=10)

    def _mk_row(
        row: int,
        text: str,
        var: tk.BooleanVar,
        avail: ProcessAvailability,
        status_override: str | None = None,
    ):
        cb = tk.Checkbutton(controls, text=text, variable=var)
        cb.grid(row=row, column=0, sticky="w")
        if status_override is not None:
            status = status_override
        else:
            status = "Ready" if avail.available else ("; ".join(avail.reasons) if avail.reasons else "Missing inputs")
        lbl = tk.Label(controls, text=status, anchor="w")
        lbl.grid(row=row, column=1, sticky="w", padx=10)
        if not avail.available:
            cb.configure(state=tk.DISABLED)
            var.set(False)

        return cb

    cb_data = _mk_row(1, "Data processing (presentation)", var_data, avail_data)
    tiles_status = "Run data processing first" if not tiles_flat_exists else None
    cb_tiles = _mk_row(2, "Tiles processing (MBTiles)", var_tiles, avail_tiles, tiles_status)
    _mk_row(3, "Lines processing (segments)", var_lines, avail_lines)
    _mk_row(4, "Analysis processing (study areas)", var_analysis, avail_analysis)

    # Optional data-processing settings.
    options_frame = tk.Frame(controls)
    options_frame.grid(row=1, column=2, sticky="w", padx=10)

    # Explode MultiPolygons into individual Polygon rows in tbl_flat.
    opt_data = tk.Checkbutton(
        options_frame,
        text="Split MultiPolygons in tbl_flat",
        variable=var_data_explode,
    )
    opt_data.pack(anchor="w")

    def _sync_data_option_state(*_args) -> None:
        try:
            enabled = bool(var_data.get()) and bool(avail_data.available)
            opt_data.configure(state=(tk.NORMAL if enabled else tk.DISABLED))
            if not enabled:
                var_data_explode.set(False)
        except Exception:
            pass

    def _sync_tiles_state(*_args) -> None:
        try:
            helper_ok = bool(avail_tiles.available)
            data_ok = bool(var_data.get()) and bool(avail_data.available)
            flat_ok = bool(tiles_flat_exists)
            enabled = helper_ok and (data_ok or flat_ok)
            cb_tiles.configure(state=(tk.NORMAL if enabled else tk.DISABLED))
            if not enabled:
                var_tiles.set(False)
        except Exception:
            pass

    try:
        var_data.trace_add("write", _sync_data_option_state)
        var_data.trace_add("write", _sync_tiles_state)
    except Exception:
        pass
    _sync_data_option_state()
    _sync_tiles_state()

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
        map_btn = Button(buttons, text="Progress map", width=button_width, bootstyle="info")
        exit_btn = Button(buttons, text="Exit", command=root.destroy, width=button_width, bootstyle="warning")
    else:
        process_btn = Button(buttons, text="Process selected", width=button_width)
        map_btn = Button(buttons, text="Progress map", width=button_width)
        exit_btn = Button(buttons, text="Exit", command=root.destroy, width=button_width)

    process_btn.pack(side=tk.LEFT, padx=button_padx, pady=button_pady)
    map_btn.pack(side=tk.LEFT, padx=button_padx, pady=button_pady)
    exit_btn.pack(side=tk.RIGHT, padx=button_padx, pady=button_pady)

    def open_progress_map() -> None:
        try:
            import _data_process_internal as dpi
            try:
                import importlib.util as _importlib_util
                if _importlib_util.find_spec("webview") is None:
                    ui_log(f"{_ts()} - Progress map requires pywebview (Edge WebView2). It is not available in this build.")
                    return
            except Exception:
                pass
            try:
                os.environ["MESA_BASE_DIR"] = str(base_dir)
            except Exception:
                pass
            try:
                dpi.original_working_directory = str(base_dir)
            except Exception:
                pass
            try:
                # Ensure status path is initialized so the minimap has something to read.
                dpi.MINIMAP_STATUS_PATH = dpi.gpq_dir() / "__chunk_status.json"
                dpi._init_idle_status()
            except Exception:
                pass
            dpi.open_minimap_window()
        except Exception as exc:
            ui_log(f"{_ts()} - Progress map unavailable: {exc}")

    def worker() -> None:
        process_btn.configure(state=tk.DISABLED)
        try:
            plan = ProcessPlan(
                run_data=bool(var_data.get()),
                explode_flat_multipolygons=bool(var_data_explode.get()),
                run_tiles=bool(var_tiles.get()),
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
    map_btn.configure(command=open_progress_map)

    # Initial log
    ui_log(f"{_ts()} - Base dir: {base_dir}")

    _start_log_tailer()

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
    p.add_argument(
        "--explode-flat-multipolygons",
        action="store_true",
        help="When running data processing, split MultiPolygon geometries in tbl_flat into individual Polygon rows",
    )
    p.add_argument("--no-tiles", action="store_true", help="Do not run raster tiles (MBTiles) after data processing")
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
        explode_flat_multipolygons=bool(args.explode_flat_multipolygons),
        run_tiles=avail_data.available and not bool(args.no_data) and not bool(args.no_tiles),
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
        if default_plan.run_tiles:
            selected.append("tiles")
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
