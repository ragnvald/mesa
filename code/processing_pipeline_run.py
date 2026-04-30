#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""processing_pipeline_run.py — Unified processing runner (Data / Lines / Analysis).

Goal
- Provide a single UI for batch-processing:
    - Data processing (presentation processing)
    - Lines processing (segment processing)
    - Analysis processing (study area analysis)

Notes
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

from mesa_shared import find_base_dir as resolve_base_dir, read_config, parquet_dir

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QPlainTextEdit, QCheckBox, QProgressBar,
    QFrame, QSizePolicy, QRadioButton, QButtonGroup,
)
from PySide6.QtGui import QIcon, QFont
from PySide6.QtCore import Qt, QTimer, Signal, QObject

from asset_manage import apply_shared_stylesheet

import argparse
import configparser
import datetime
import os
import subprocess
import sys
import threading
import time
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



# ---------------------------------------------------------------------------
# Availability model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessAvailability:
    available: bool
    reasons: list[str]


@dataclass(frozen=True)
class ProcessPlan:
    # Data-processing sub-stages. Any True triggers run_data_process; the
    # internal pipeline honours each flag independently with soft validation.
    run_prep: bool
    run_intersect: bool
    run_flatten: bool
    run_backfill: bool
    explode_flat_multipolygons: bool
    cleanup_slivers: bool
    run_tiles: bool
    run_lines: bool
    run_analysis: bool

    @property
    def run_data(self) -> bool:
        """True if any of the four data sub-stages is selected."""
        return bool(self.run_prep or self.run_intersect
                    or self.run_flatten or self.run_backfill)


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
        return ProcessAvailability(False, ["Missing: tiles_create_raster helper"])

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


def _start_progress_pulse(
    progress_fn: Callable[[float], None],
    *,
    start: float,
    cap: float,
    step: float = 1.0,
    interval_s: float = 0.8,
) -> tuple[threading.Event, threading.Thread]:
    """Emit small monotone progress jumps while a long step is running."""
    stop_event = threading.Event()
    state = {"value": max(0.0, float(start))}
    progress_fn(state["value"])

    def _worker() -> None:
        while not stop_event.wait(max(0.2, float(interval_s))):
            nxt = min(float(cap), state["value"] + max(0.2, float(step)))
            if nxt <= state["value"]:
                continue
            state["value"] = nxt
            progress_fn(state["value"])

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return stop_event, t


def _find_tiles_runner(base_dir: Path) -> tuple[Path | None, bool]:
    """Find tiles_create_raster (returns path, is_executable)."""
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
            base_dir / "tools" / "tiles_create_raster.exe",
            base_dir / "tiles_create_raster.exe",
            base_dir / "code" / "tiles_create_raster.exe",
            base_dir / "system" / "tiles_create_raster.exe",
            Path(sys.executable).resolve().parent / "tiles_create_raster.exe" if frozen else None,
        ]
    )
    py_candidates = _dedup(
        [
            base_dir / "tiles_create_raster.py",
            base_dir / "system" / "tiles_create_raster.py",
            base_dir / "code" / "tiles_create_raster.py",
            (Path(__file__).resolve().parent / "tiles_create_raster.py") if "__file__" in globals() else None,
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
    progress_fn(5.0)
    _log_line(base_dir, log_fn, "TILES PROCESS START")

    runner_path, is_exe = _find_tiles_runner(base_dir)
    if not runner_path:
        _log_line(base_dir, log_fn, "ERROR: tiles_create_raster helper not found")
        raise RuntimeError("Missing tiles_create_raster helper")

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

    stop_pulse, pulse_thread = _start_progress_pulse(
        progress_fn,
        start=10.0,
        cap=92.0,
        step=1.8,
        interval_s=0.7,
    )
    try:
        code = _run_subprocess_streaming(base_dir, log_fn, args, env=env, line_prefix="[tiles]")
        if code != 0:
            raise RuntimeError(f"Raster tiles failed (exit={code})")
    finally:
        stop_pulse.set()
        try:
            pulse_thread.join(timeout=0.2)
        except Exception:
            pass

    progress_fn(97.0)
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
    run_prep: bool = True,
    run_intersect: bool = True,
    run_flatten: bool = True,
    run_backfill: bool = True,
    cleanup_slivers: bool = True,
) -> None:
    """Run the data-processing pipeline.

    Sub-stages (prep / intersect / flatten / backfill) can be skipped
    independently via the run_* flags; the internal pipeline does soft
    validation and logs a clear skip line when an upstream artifact is
    missing.

    Important: when frozen (PyInstaller), calling a .py via `sys.executable` does not work.
    We therefore run the pipeline in-process by importing the internal module.
    """

    progress_fn(0.0)
    progress_fn(5.0)
    _log_line(base_dir, log_fn, "DATA PROCESS START")

    try:
        progress_fn(12.0)
        # Import on-demand so the main UI stays light until the user actually
        # runs the Area step.
        import processing_internal as dpi
        progress_fn(18.0)

        stop_pulse, pulse_thread = _start_progress_pulse(
            progress_fn,
            start=22.0,
            cap=93.0,
            step=1.2,
            interval_s=0.8,
        )
        dpi.run_headless(
            str(base_dir),
            explode_flat_multipolygons=bool(explode_flat_multipolygons),
            log_fn=log_fn,
            run_prep=bool(run_prep),
            run_intersect=bool(run_intersect),
            run_flatten=bool(run_flatten),
            run_backfill=bool(run_backfill),
            cleanup_slivers=bool(cleanup_slivers),
        )
        stop_pulse.set()
        try:
            pulse_thread.join(timeout=0.2)
        except Exception:
            pass
    except Exception as exc:
        _log_line(base_dir, log_fn, f"ERROR: data processing failed: {exc}")
        raise

    progress_fn(97.0)
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

    def _cfg_int(*keys: str, default: int) -> int:
        if "DEFAULT" not in cfg:
            return int(default)
        for key in keys:
            try:
                raw = str(cfg["DEFAULT"].get(key, "")).strip()
                if raw:
                    return int(round(float(raw)))
            except Exception:
                continue
        return int(default)

    workingprojection_epsg = f"EPSG:{_cfg_int('working_projection_epsg', 'workingprojection_epsg', default=4326)}"
    default_segment_width = max(1, _cfg_int("segment_width", default=600))
    default_segment_length = max(1, _cfg_int("segment_length", default=1000))

    def _looks_like_legacy_auto_lines(gdf: gpd.GeoDataFrame) -> bool:
        required = {"name_gis", "description", "segment_width", "segment_length"}
        if gdf is None or getattr(gdf, "empty", True):
            return False
        if len(gdf) != 3 or not required.issubset(set(gdf.columns)):
            return False
        try:
            names = sorted(str(v).strip() for v in gdf["name_gis"].tolist())
            desc = {str(v).strip().lower() for v in gdf["description"].tolist()}
            widths = sorted(pd.to_numeric(gdf["segment_width"], errors="coerce").dropna().astype(int).tolist())
            lengths = sorted(pd.to_numeric(gdf["segment_length"], errors="coerce").dropna().astype(int).tolist())
            return (
                names == ["line_001", "line_002", "line_003"]
                and desc.issubset({"auto line", ""})
                and widths == [1000, 5000, 20000]
                and lengths == [10, 15, 30]
            )
        except Exception:
            return False

    def load_lines_table():
        gdf = read_parquet_or_none("tbl_lines")
        if gdf is not None and not getattr(gdf, "empty", True):
            if _looks_like_legacy_auto_lines(gdf):
                gdf = gdf.copy()
                gdf["segment_width"] = int(default_segment_width)
                gdf["segment_length"] = int(default_segment_length)
                _log_line(
                    base_dir,
                    log_fn,
                    (
                        "Detected legacy auto-line defaults "
                        f"(1000/5000/20000 m). Normalizing to config defaults: "
                        f"segment_width={default_segment_width} m, "
                        f"segment_length={default_segment_length} m."
                    ),
                )
                write_parquet("tbl_lines", gdf)
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
                "segment_length": [int(default_segment_length)] * 3,
                "segment_width": [int(default_segment_width)] * 3,
                "description": ["auto line", "auto line", "auto line"],
                "geometry": lines,
            },
            geometry="geometry",
            crs=geocode_group.crs or workingprojection_epsg,
        )
        _log_line(
            base_dir,
            log_fn,
            (
                "Template lines use config defaults: "
                f"segment_width={default_segment_width} m, "
                f"segment_length={default_segment_length} m."
            ),
        )
        if str(gdf_lines.crs).upper() != str(workingprojection_epsg).upper():
            gdf_lines = gdf_lines.to_crs(workingprojection_epsg)
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
                projected_line = gpd.GeoDataFrame([{"geometry": geom}], geometry="geometry", crs=workingprojection_epsg).to_crs(target_crs)
                projected_line["geometry"] = projected_line.buffer(seg_w, cap_style=2)
                buffered_back = projected_line.to_crs(workingprojection_epsg)
                buffered_geom = buffered_back.iloc[0].geometry
                if not isinstance(buffered_geom, (Polygon, MultiPolygon)):
                    _log_line(base_dir, log_fn, f"Unexpected buffered geom type: {type(buffered_geom)}")
                buffered_records.append(
                    {
                        "fid": idx,
                        "name_gis": name_gis,
                        "name_user": name_usr,
                        "segment_length": seg_len,
                        "segment_width": seg_w,
                        "description": desc,
                        "geometry": buffered_geom,
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
    progress_fn(8.0)
    process_and_buffer_lines()
    progress_fn(22.0)
    progress_fn(35.0)
    progress_fn(52.0)
    create_segments_from_buffered_lines()
    progress_fn(64.0)
    progress_fn(70.0)
    progress_fn(78.0)
    build_stacked_data()
    progress_fn(82.0)
    progress_fn(85.0)
    progress_fn(94.0)
    build_flat_data()
    progress_fn(98.0)
    progress_fn(100.0)
    _log_line(base_dir, log_fn, "LINES PROCESS COMPLETED")


def run_analysis_process(
    base_dir: Path,
    cfg: configparser.ConfigParser,
    log_fn: Callable[[str], None],
    progress_fn: Callable[[float], None],
) -> None:
    # Embedded (minimal) analysis processor so we can remove analysis_process.py.
    # This intentionally avoids importing heavy GIS modules at processing_pipeline_run import time.
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    from analysis_setup import (
        DEFAULT_ANALYSIS_GEOCODE,
        AnalysisStorage,
        analysis_flat_path,
        analysis_stacked_path,
        find_dataset_dir,
        find_parquet_file,
        _load_runtime_data_stack as _analysis_load_deps,
    )
    # analysis_setup keeps pd/gpd/np as lazy module-level globals (None until loaded).
    # Populate them now so AnalysisStorage and friends can call pd.DataFrame etc.
    _analysis_load_deps()

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

    # Weighted progress allocation (normalized across selected processes).
    # This provides more realistic overall movement than fixed 25% blocks.
    weights: dict[str, float] = {
        "data": 4.0,
        "tiles": 1.0,
        "lines": 2.5,
        "analysis": 2.5,
    }

    total_weight = sum(weights.get(name, 1.0) for name in active)
    if total_weight <= 0:
        total_weight = float(len(active))

    ranges: dict[str, tuple[float, float]] = {}
    cursor = 0.0
    for name in ["data", "tiles", "lines", "analysis"]:
        if name not in active:
            continue
        w = max(0.0, float(weights.get(name, 1.0)))
        span = (w / total_weight) * 100.0
        start = cursor
        end = min(100.0, start + span)
        ranges[name] = (start, end)
        cursor = end

    def make_slice_progress(start: float, end: float) -> Callable[[float], None]:
        span = max(0.0, float(end) - float(start))

        def _slice(p: float) -> None:
            p = max(0.0, min(100.0, float(p)))
            progress_fn(min(100.0, float(start) + (p / 100.0) * span))

        return _slice

    if plan.run_data:
        try:
            s, e = ranges.get("data", (0.0, 100.0))
            sub_summary = ", ".join([
                name for name, on in [("prep", plan.run_prep),
                                       ("intersect", plan.run_intersect),
                                       ("flatten", plan.run_flatten),
                                       ("backfill", plan.run_backfill)]
                if on
            ]) or "(none)"
            _log_line(base_dir, log_fn,
                      f"[Process] Progress range data: {s:.1f}% -> {e:.1f}% (sub-stages: {sub_summary})")
            run_data_process(
                base_dir,
                log_fn,
                make_slice_progress(s, e),
                explode_flat_multipolygons=bool(plan.explode_flat_multipolygons),
                run_prep=bool(plan.run_prep),
                run_intersect=bool(plan.run_intersect),
                run_flatten=bool(plan.run_flatten),
                run_backfill=bool(plan.run_backfill),
                cleanup_slivers=bool(plan.cleanup_slivers),
            )
        except Exception as exc:
            _log_line(base_dir, log_fn, f"ERROR: data processing failed: {exc}")
            had_errors = True

    if plan.run_tiles:
        try:
            s, e = ranges.get("tiles", (0.0, 100.0))
            _log_line(base_dir, log_fn, f"[Process] Progress range tiles: {s:.1f}% -> {e:.1f}%")
            gpq = parquet_dir(base_dir, cfg)
            if not _exists_any([gpq / "tbl_flat.parquet"]):
                # Soft skip: user opted in to tiles but the upstream artifact
                # is not on disk. Honour the rerun-parts intent rather than
                # failing the whole batch.
                _log_line(base_dir, log_fn,
                          "Tiles: skipped - tbl_flat.parquet is missing. "
                          "Run Flatten (or Data processing end-to-end) first.")
            else:
                run_tiles_process(base_dir, cfg, log_fn, make_slice_progress(s, e))
        except Exception as exc:
            _log_line(base_dir, log_fn, f"ERROR: raster tiles failed: {exc}")
            had_errors = True

    if plan.run_lines:
        try:
            s, e = ranges.get("lines", (0.0, 100.0))
            _log_line(base_dir, log_fn, f"[Process] Progress range lines: {s:.1f}% -> {e:.1f}%")
            run_lines_process(base_dir, cfg, log_fn, make_slice_progress(s, e))
        except Exception as exc:
            _log_line(base_dir, log_fn, f"ERROR: lines processing failed: {exc}")
            had_errors = True

    if plan.run_analysis:
        try:
            s, e = ranges.get("analysis", (0.0, 100.0))
            _log_line(base_dir, log_fn, f"[Process] Progress range analysis: {s:.1f}% -> {e:.1f}%")
            run_analysis_process(base_dir, cfg, log_fn, make_slice_progress(s, e))
        except Exception as exc:
            _log_line(base_dir, log_fn, f"ERROR: analysis processing failed: {exc}")
            had_errors = True

    progress_fn(100.0)
    _log_line(base_dir, log_fn, "[Process] FAILED" if had_errors else "[Process] COMPLETED")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Qt signal bridge for thread-safe UI updates
# ---------------------------------------------------------------------------


class _RunnerSignals(QObject):
    log_message = Signal(str)
    progress_update = Signal(float)
    task_finished = Signal()


def _shared_window_icon(base_dir: Path) -> QIcon:
    for candidate in (
        base_dir / "system_resources" / "icon.png",
        base_dir / "system_resources" / "mesa.ico",
    ):
        try:
            if candidate.exists():
                icon = QIcon(str(candidate))
                if not icon.isNull():
                    return icon
        except Exception:
            pass
    return QIcon()


# ---------------------------------------------------------------------------
# ProcessRunnerWindow (PySide6)
# ---------------------------------------------------------------------------


class ProcessRunnerWindow(QMainWindow):
    def __init__(self, base_dir: Path, cfg: configparser.ConfigParser, parent=None):
        super().__init__(parent)
        self._base_dir = base_dir
        self._cfg = cfg
        self._signals = _RunnerSignals()
        self._tail_state: dict[str, int] = {}

        self.setWindowTitle("MESA \u2013 Process all")
        self.resize(900, 560)
        self.setMinimumSize(700, 440)

        try:
            icon = _shared_window_icon(base_dir)
            if not icon.isNull():
                self.setWindowIcon(icon)
        except Exception:
            pass

        # Central widget
        central = QWidget(objectName="CentralHost")
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Log area
        log_group = QGroupBox("Log output")
        log_lay = QVBoxLayout(log_group)
        self._log_widget = QPlainTextEdit()
        self._log_widget.setReadOnly(True)
        self._log_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        log_lay.addWidget(self._log_widget)
        layout.addWidget(log_group, stretch=1)

        # Progress bar row
        prog_row = QHBoxLayout()
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%p%")
        self._progress_bar.setAlignment(Qt.AlignCenter)
        self._progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        prog_row.addWidget(self._progress_bar, stretch=1)
        layout.addLayout(prog_row)

        # Availability
        avail_data = detect_data_processing(base_dir, cfg)
        avail_tiles = detect_tiles_processing(base_dir, cfg)
        avail_lines = detect_lines_processing(base_dir, cfg)
        avail_analysis = detect_analysis_processing(base_dir, cfg)

        gpq = parquet_dir(base_dir, cfg)
        tiles_flat_exists = _exists_any([gpq / "tbl_flat.parquet"])
        tiles_default = bool(avail_tiles.available and (tiles_flat_exists or avail_data.available))

        # Stash availability for the Normal-mode worker, which builds a plan
        # straight from these instead of reading checkbox state.
        self._avail_data = avail_data
        self._avail_tiles = avail_tiles
        self._avail_lines = avail_lines
        self._avail_analysis = avail_analysis
        self._tiles_flat_exists = tiles_flat_exists

        # Mode selector: Normal hides the checkbox grid and runs everything
        # available in one click. Advanced shows the per-stage checkboxes.
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._rb_normal   = QRadioButton("Normal")
        self._rb_advanced = QRadioButton("Advanced")
        self._rb_normal.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._rb_normal)
        self._mode_group.addButton(self._rb_advanced)
        mode_row.addWidget(self._rb_normal)
        mode_row.addWidget(self._rb_advanced)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # Info label adapts to the active mode.
        self._info_label = QLabel()
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)

        # Checkbox grid (only visible in Advanced mode).
        grid_widget = QWidget()
        self._advanced_panel = grid_widget
        grid = QGridLayout(grid_widget)
        grid.setContentsMargins(10, 0, 10, 0)

        # Two-column layout:
        #   Left column (cols 0-1): data sub-stages (Prep / Intersect / Flatten / Backfill)
        #   Right column (cols 3-4): post-data stages (Tiles / Lines / Analysis)
        #   Col 2: visual gap, with a directional arrow on the header row.
        # Row 0 is a directional banner replacing the old generic
        # "Process / Status" headers — it tells the operator that geometry
        # stages live on the left and parameter-driven downstream stages live
        # on the right, so they can pick a sensible re-run point after a
        # parameter-only change.
        left_hdr = QLabel("Geometry stages — rerun when inputs change")
        left_hdr.setWordWrap(True)
        left_hdr.setStyleSheet("font-weight: bold; color: #6a5533; padding-bottom: 4px;")
        grid.addWidget(left_hdr, 0, 0, 1, 2)

        arrow_hdr = QLabel("→")
        arrow_hdr.setAlignment(Qt.AlignCenter)
        arrow_hdr.setStyleSheet("color: #6a5533; font-size: 14pt; font-weight: bold;")
        grid.addWidget(arrow_hdr, 0, 2)

        right_hdr = QLabel("Downstream stages — rerun for parameter changes")
        right_hdr.setWordWrap(True)
        right_hdr.setStyleSheet("font-weight: bold; color: #6a5533; padding-bottom: 4px;")
        grid.addWidget(right_hdr, 0, 3, 1, 2)

        # Master "Process" checkbox + flatten options live in row 1 spanning
        # the whole grid.

        grid.setColumnMinimumWidth(2, 32)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(4, 1)

        # Row 1: master "Process" checkbox (cascades to data sub-stages + tiles).
        self._cb_data_master = QCheckBox("Process (prep + intersect + flatten + backfill + tiles)")
        self._cb_data_master.setTristate(True)
        self._cb_data_master.setEnabled(avail_data.available)
        self._cb_data_master.setChecked(avail_data.available)
        grid.addWidget(self._cb_data_master, 1, 0, 1, 2)
        master_status = "Ready" if avail_data.available else (
            "; ".join(avail_data.reasons) if avail_data.reasons else "Missing inputs")
        grid.addWidget(QLabel(master_status), 1, 3, 1, 2)

        # Flatten options stacked vertically below the master row, spanning
        # the whole grid so they sit clearly under the master.
        opts_col = QVBoxLayout()
        opts_col.setContentsMargins(0, 0, 0, 0)
        opts_col.setSpacing(2)
        self._cb_data_explode = QCheckBox("Split MultiPolygons in tbl_flat")
        self._cb_data_explode.setChecked(False)
        opts_col.addWidget(self._cb_data_explode)
        self._cb_cleanup_slivers = QCheckBox("Drop sliver cells (<1 m²)")
        self._cb_cleanup_slivers.setChecked(True)
        opts_col.addWidget(self._cb_cleanup_slivers)
        opts_host = QWidget()
        opts_host.setLayout(opts_col)

        # Left column: data sub-stages.
        def _mk_sub(row, text, default_checked):
            cb = QCheckBox("    " + text)  # indent for visual hierarchy
            cb.setEnabled(avail_data.available)
            cb.setChecked(default_checked and avail_data.available)
            grid.addWidget(cb, row, 0)
            grid.addWidget(QLabel("Ready" if avail_data.available else "Missing inputs"),
                           row, 1)
            return cb

        self._cb_prep      = _mk_sub(2, "Prep (workspace, status)", avail_data.available)
        self._cb_intersect = _mk_sub(3, "Intersect (build tbl_stacked)", avail_data.available)
        self._cb_flatten   = _mk_sub(4, "Flatten (build tbl_flat)", avail_data.available)
        self._cb_backfill  = _mk_sub(5, "Backfill (area_m2 → tbl_stacked)",
                                     avail_data.available)

        # Right column: post-data stages. Custom helper because _mk_row
        # writes to col 0/1 and we want col 3/4.
        def _mk_row_right(row, text, checked, avail, status_override=None):
            cb = QCheckBox(text)
            cb.setChecked(checked and avail.available)
            grid.addWidget(cb, row, 3)
            if status_override is not None:
                status = status_override
            else:
                status = "Ready" if avail.available else (
                    "; ".join(avail.reasons) if avail.reasons else "Missing inputs")
            lbl = QLabel(status)
            grid.addWidget(lbl, row, 4)
            if not avail.available:
                cb.setEnabled(False)
                cb.setChecked(False)
            return cb

        tiles_status = "Run Flatten (or full data) first" if not tiles_flat_exists else None
        self._cb_tiles    = _mk_row_right(2, "Tiles processing (MBTiles)",
                                          tiles_default, avail_tiles, tiles_status)
        self._cb_lines    = _mk_row_right(3, "Lines processing (segments)",
                                          avail_lines.available, avail_lines)
        self._cb_analysis = _mk_row_right(4, "Analysis processing (study areas)",
                                          avail_analysis.available, avail_analysis)

        # Options under the grid (spans all columns) so they're clearly the
        # flatten / data options, not specific to either stage column.
        layout.addWidget(grid_widget)
        opts_label = QLabel("Options")
        opts_label.setStyleSheet("color: #6a5533; font-size: 9pt; padding-top: 6px;")
        layout.addWidget(opts_label)
        layout.addWidget(opts_host)

        # ----- Master <-> sub cascade -----
        # Use `clicked` (fires only on user interaction, not setCheckState) so
        # programmatic updates don't recurse - no guard flag needed.
        sub_cbs = [self._cb_prep, self._cb_intersect, self._cb_flatten,
                   self._cb_backfill, self._cb_tiles]

        def _on_master_clicked():
            state = self._cb_data_master.checkState()
            # Tristate cycles user clicks through Unchecked → PartiallyChecked → Checked;
            # treat Partial as "user wants all on".
            if state == Qt.PartiallyChecked:
                state = Qt.Checked
                self._cb_data_master.setCheckState(Qt.Checked)
            new_checked = (state == Qt.Checked)
            for cb in sub_cbs:
                if cb.isEnabled():
                    cb.setChecked(new_checked)
            _sync_data_option()
            _sync_tiles()

        def _refresh_master_state():
            enabled_subs = [cb for cb in sub_cbs if cb.isEnabled()]
            if not enabled_subs:
                self._cb_data_master.setCheckState(Qt.Unchecked)
                return
            states = [cb.isChecked() for cb in enabled_subs]
            if all(states):
                self._cb_data_master.setCheckState(Qt.Checked)
            elif any(states):
                self._cb_data_master.setCheckState(Qt.PartiallyChecked)
            else:
                self._cb_data_master.setCheckState(Qt.Unchecked)

        def _sync_data_option():
            # Explode option requires Flatten; it's the stage that writes tbl_flat.
            enabled = self._cb_flatten.isChecked() and avail_data.available
            self._cb_data_explode.setEnabled(enabled)
            if not enabled:
                self._cb_data_explode.setChecked(False)

        def _sync_tiles():
            helper_ok = avail_tiles.available
            flatten_ok = self._cb_flatten.isChecked() and avail_data.available
            flat_ok = tiles_flat_exists
            enabled = helper_ok and (flatten_ok or flat_ok)
            self._cb_tiles.setEnabled(enabled)
            if not enabled:
                self._cb_tiles.setChecked(False)

        self._cb_data_master.clicked.connect(_on_master_clicked)
        for cb in sub_cbs:
            cb.clicked.connect(_refresh_master_state)
        # Flatten controls availability of the explode option and (along with
        # an on-disk tbl_flat) controls Tiles availability.
        self._cb_flatten.clicked.connect(_sync_data_option)
        self._cb_flatten.clicked.connect(_sync_tiles)

        _sync_data_option()
        _sync_tiles()
        _refresh_master_state()

        # Button row
        btn_row = QHBoxLayout()
        self._process_btn = QPushButton("Process")
        self._map_btn = QPushButton("Progress map")
        exit_btn = QPushButton("Exit")
        exit_btn.setObjectName("CornerExitButton")
        exit_btn.setStyleSheet("""
            QPushButton#CornerExitButton {
                background: #eadfc8; border: 1px solid #b79f73;
                border-radius: 4px; color: #453621;
                padding: 6px 18px;
            }
            QPushButton#CornerExitButton:hover { background: #e1d1ae; }
            QPushButton#CornerExitButton:pressed { background: #d4c094; }
        """)

        btn_row.addWidget(self._process_btn)
        btn_row.addWidget(self._map_btn)
        btn_row.addStretch()
        btn_row.addWidget(exit_btn)
        layout.addLayout(btn_row)

        # Connect buttons
        self._process_btn.clicked.connect(self._on_process_click)
        self._map_btn.clicked.connect(self._open_progress_map)
        exit_btn.clicked.connect(self.close)

        # Mode toggle: hide the per-stage grid + rename the button in Normal.
        def _apply_mode():
            advanced = self._rb_advanced.isChecked()
            self._advanced_panel.setVisible(advanced)
            self._process_btn.setText("Process all selected" if advanced else "Process")
            self._info_label.setText(
                "Advanced mode: pick exactly which stages to run. Unavailable "
                "stages stay disabled. Useful for reruns of a single sub-step."
                if advanced else
                "Normal mode: one click runs every stage that has its inputs "
                "ready. Switch to Advanced to pick individual sub-stages."
            )

        self._rb_normal.toggled.connect(_apply_mode)
        self._rb_advanced.toggled.connect(_apply_mode)
        _apply_mode()

        # Connect signals
        self._signals.log_message.connect(self._append_log)
        self._signals.progress_update.connect(self._set_progress)
        self._signals.task_finished.connect(self._on_task_finished)

        # Log tail timer
        self._tail_timer = QTimer(self)
        self._tail_timer.setInterval(750)
        self._tail_timer.timeout.connect(self._tail_once)

        # Start at current EOF to avoid dumping old logs
        log_path = base_dir / "log.txt"
        try:
            if log_path.exists():
                self._tail_state[str(log_path)] = log_path.stat().st_size
        except Exception:
            pass
        self._tail_timer.start()

        # Initial log
        self._append_log(f"{_ts()} - Base dir: {base_dir}")

        # Apply stylesheet
        # (applied on the window; app-level stylesheet set in run_ui)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _append_log(self, line: str) -> None:
        self._log_widget.appendPlainText(line)

    def _set_progress(self, p: float) -> None:
        p = max(0.0, min(100.0, float(p)))
        self._progress_bar.setValue(int(p))

    def _on_task_finished(self) -> None:
        self._process_btn.setEnabled(True)
        # Resume log-file tailing; skip to current EOF so we don't replay
        # lines already shown via the worker's direct signal path.
        log_path = self._base_dir / "log.txt"
        try:
            if log_path.exists():
                self._tail_state[str(log_path)] = log_path.stat().st_size
        except Exception:
            pass
        self._tail_timer.start()

    # ------------------------------------------------------------------
    # Log tailer
    # ------------------------------------------------------------------

    def _tail_once(self) -> None:
        candidates = [self._base_dir / "log.txt"]
        for p in candidates:
            try:
                if not p.exists():
                    continue
                key = str(p)
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    pos = self._tail_state.get(key, 0)
                    try:
                        f.seek(pos)
                    except Exception:
                        pos = 0
                        f.seek(0)
                    data = f.read()
                    self._tail_state[key] = f.tell()
                if data:
                    for line in data.splitlines():
                        if line.strip():
                            self._append_log(line)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Progress map
    # ------------------------------------------------------------------

    def _open_progress_map(self) -> None:
        try:
            import processing_internal as dpi
            try:
                import importlib.util as _importlib_util
                if _importlib_util.find_spec("webview") is None:
                    self._append_log(f"{_ts()} - Progress map requires pywebview (Edge WebView2). It is not available in this build.")
                    return
            except Exception:
                pass
            try:
                os.environ["MESA_BASE_DIR"] = str(self._base_dir)
            except Exception:
                pass
            try:
                dpi.original_working_directory = str(self._base_dir)
            except Exception:
                pass
            try:
                dpi.MINIMAP_STATUS_PATH = dpi.gpq_dir() / "__chunk_status.json"
                dpi._init_idle_status()
            except Exception:
                pass
            dpi.open_minimap_window()
        except Exception as exc:
            self._append_log(f"{_ts()} - Progress map unavailable: {exc}")

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _on_process_click(self) -> None:
        self._process_btn.setEnabled(False)
        # Pause log-file tailing while the worker is running — the worker
        # sends lines directly via signals, and _log_line also writes to
        # log.txt, so tailing would duplicate every line.
        self._tail_timer.stop()
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def _worker(self) -> None:
        try:
            if self._rb_advanced.isChecked():
                plan = ProcessPlan(
                    run_prep=self._cb_prep.isChecked(),
                    run_intersect=self._cb_intersect.isChecked(),
                    run_flatten=self._cb_flatten.isChecked(),
                    run_backfill=self._cb_backfill.isChecked(),
                    explode_flat_multipolygons=self._cb_data_explode.isChecked(),
                    cleanup_slivers=self._cb_cleanup_slivers.isChecked(),
                    run_tiles=self._cb_tiles.isChecked(),
                    run_lines=self._cb_lines.isChecked(),
                    run_analysis=self._cb_analysis.isChecked(),
                )
            else:
                # Normal mode: run everything available, no advanced toggles.
                data_on = self._avail_data.available
                plan = ProcessPlan(
                    run_prep=data_on,
                    run_intersect=data_on,
                    run_flatten=data_on,
                    run_backfill=data_on,
                    explode_flat_multipolygons=False,
                    cleanup_slivers=True,
                    run_tiles=self._avail_tiles.available and (data_on or self._tiles_flat_exists),
                    run_lines=self._avail_lines.available,
                    run_analysis=self._avail_analysis.available,
                )

            def log_from_worker(msg: str) -> None:
                self._signals.log_message.emit(msg)

            def progress_from_worker(v: float) -> None:
                self._signals.progress_update.emit(v)

            run_selected(self._base_dir, self._cfg, plan, log_from_worker, progress_from_worker)
            self._signals.log_message.emit(f"{_ts()} - ALL SELECTED PROCESSING COMPLETED")
        except Exception as e:
            self._signals.log_message.emit(f"{_ts()} - ERROR: {e}")
        finally:
            self._signals.task_finished.emit()


def run_ui(base_dir: Path, cfg: configparser.ConfigParser, master=None) -> None:
    """Create and show the ProcessRunnerWindow."""
    app = QApplication.instance()
    own_app = False
    if app is None:
        app = QApplication(sys.argv)
        own_app = True

    apply_shared_stylesheet(app)
    try:
        icon = _shared_window_icon(base_dir)
        if not icon.isNull():
            app.setWindowIcon(icon)
    except Exception:
        pass

    win = ProcessRunnerWindow(base_dir, cfg, parent=None)
    win.show()

    if own_app:
        app.exec()
    return win


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MESA \u2013 Unified processing runner (data/lines/analysis)")
    p.add_argument("--original_working_directory", default=None, help="Mesa base directory")
    p.add_argument("--headless", action="store_true", help="Run without GUI")
    p.add_argument("--list", action="store_true", help="Print availability and exit")
    p.add_argument("--dry-run", action="store_true", help="Print planned actions and exit")

    p.add_argument("--no-data", action="store_true", help="Do not run any data sub-stage")
    # Per-sub-stage skips (data processing). Default is to run all three when
    # --no-data is not set; pass --no-prep / --no-intersect / --no-flatten to
    # rerun a subset.
    p.add_argument("--no-prep", action="store_true", help="Skip Prep sub-stage of data processing")
    p.add_argument("--no-intersect", action="store_true", help="Skip Intersect sub-stage of data processing")
    p.add_argument("--no-flatten", action="store_true", help="Skip Flatten sub-stage of data processing")
    p.add_argument("--no-backfill", action="store_true", help="Skip Backfill sub-stage (area_m2 enrichment of tbl_stacked)")
    p.add_argument(
        "--explode-flat-multipolygons",
        action="store_true",
        help="When running data processing, split MultiPolygon geometries in tbl_flat into individual Polygon rows",
    )
    p.add_argument(
        "--no-cleanup-slivers",
        action="store_true",
        help="Keep zero-area / sub-1-m^2 sliver rows in tbl_flat (default: drop them)",
    )
    p.add_argument("--no-tiles", action="store_true", help="Do not run raster tiles (MBTiles) after data processing")
    p.add_argument("--no-lines", action="store_true", help="Do not run lines processing")
    p.add_argument("--no-analysis", action="store_true", help="Do not run analysis processing")

    return p.parse_args()


def run(base_dir: str, master=None) -> None:
    """In-process entry point called by mesa.py via lazy import. Launches the GUI."""
    resolved = resolve_base_dir(base_dir)
    cfg = read_config(resolved)
    return run_ui(resolved, cfg, master=master)


def main() -> None:
    args = parse_args()
    base_dir = resolve_base_dir(args.original_working_directory)
    cfg = read_config(base_dir)

    avail_data = detect_data_processing(base_dir, cfg)
    avail_lines = detect_lines_processing(base_dir, cfg)
    avail_analysis = detect_analysis_processing(base_dir, cfg)

    data_on = avail_data.available and not bool(args.no_data)
    default_plan = ProcessPlan(
        run_prep=data_on and not bool(args.no_prep),
        run_intersect=data_on and not bool(args.no_intersect),
        run_flatten=data_on and not bool(args.no_flatten),
        run_backfill=data_on and not bool(args.no_backfill),
        explode_flat_multipolygons=bool(args.explode_flat_multipolygons),
        cleanup_slivers=not bool(args.no_cleanup_slivers),
        run_tiles=data_on and not bool(args.no_tiles),
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
        if default_plan.run_prep:
            selected.append("data:prep")
        if default_plan.run_intersect:
            selected.append("data:intersect")
        if default_plan.run_flatten:
            selected.append("data:flatten")
        if default_plan.run_backfill:
            selected.append("data:backfill")
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
    try:
        import multiprocessing as _mp
        _mp.freeze_support()
    except Exception:
        pass
    main()
