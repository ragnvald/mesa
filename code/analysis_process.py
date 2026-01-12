#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analysis_process.py — Run study area analysis processing.

UI aligned with other Mesa processing tools:
- ttkbootstrap window
- log output window
- determinate progress bar + percent label
- Process + Exit buttons

This tool processes ALL analysis groups and ALL polygons in each group.

Inputs (created via data_analysis_setup.py):
- tbl_analysis_group.parquet
- tbl_analysis_polygons.parquet

Outputs (consumed by data_analysis_presentation.py):
- tbl_analysis_flat.parquet
- tbl_analysis_stacked.parquet
"""

from __future__ import annotations

import argparse
import datetime
import locale
import os
import queue
import threading
import uuid
from pathlib import Path
from typing import Callable

import tkinter as tk
import tkinter.scrolledtext as scrolledtext

import geopandas as gpd
import numpy as np
import pandas as pd



def _patch_locale_setlocale_for_windows() -> None:
    """Make locale.setlocale resilient on Windows.

    Some Windows machines/environments raise locale.Error('unsupported locale setting')
    for calls like setlocale(LC_TIME, ""). ttkbootstrap calls setlocale during import
    (e.g. DatePickerDialog), so we must not allow this to crash the app.
    """
    try:
        if os.name != "nt":
            return
        _orig = locale.setlocale

        def _safe_setlocale(category, value=None):
            try:
                if value is None:
                    return _orig(category)
                return _orig(category, value)
            except locale.Error:
                for fallback in ("", "C"):
                    try:
                        return _orig(category, fallback)
                    except Exception:
                        pass
                try:
                    return _orig(category)
                except Exception:
                    return "C"

        locale.setlocale = _safe_setlocale  # type: ignore[assignment]
    except Exception:
        pass


# Locale is surprisingly fragile on Windows:
# - Many machines do not support POSIX locale names like "en_US.UTF-8".
# - Some environments set LANG/LC_ALL to POSIX values, and `setlocale(LC_ALL, "")`
#   can raise locale.Error("unsupported locale setting").
try:
    if os.name == "nt":
        for _k in ("LC_ALL", "LC_CTYPE", "LANG"):
            _v = os.environ.get(_k)
            if _v and ("utf-8" in _v.lower()) and ("_" in _v) and ("." in _v):
                os.environ.pop(_k, None)
except Exception:
    pass

_patch_locale_setlocale_for_windows()

try:
    locale.setlocale(locale.LC_ALL, "")
except Exception:
    try:
        locale.setlocale(locale.LC_ALL, "C")
    except Exception:
        pass

import ttkbootstrap as tb


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run study area analysis processing")
    parser.add_argument(
        "--original_working_directory",
        default=None,
        help="Base directory for resolving config.ini and output paths.",
    )
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")


class AssetAnalyzer:
    """Run the actual analysis processing (clipping tbl_flat/tbl_stacked to polygons)."""

    def __init__(self, base_dir: Path, cfg, storage_epsg: int = 4326) -> None:
        self.base_dir = base_dir
        self.cfg = cfg
        self.storage_epsg = storage_epsg or 4326
        try:
            self.area_epsg = int(str(cfg["DEFAULT"].get("area_projection_epsg", "3035")))
        except Exception:
            self.area_epsg = 3035

        from data_analysis_setup import (
            DEFAULT_ANALYSIS_GEOCODE,
            analysis_flat_path,
            analysis_stacked_path,
            find_dataset_dir,
            find_parquet_file,
        )

        self._DEFAULT_GEOCODE = DEFAULT_ANALYSIS_GEOCODE
        self._find_parquet_file = find_parquet_file
        self._find_dataset_dir = find_dataset_dir
        self.analysis_flat_path = analysis_flat_path(base_dir, cfg)
        self.analysis_stacked_path = analysis_stacked_path(base_dir, cfg)

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
                base_area > 0, subset["analysis_area_m2"] / base_area.astype("float64"), np.nan
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

        # Filter by geocode group if column exists (setup currently uses one category).
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

        if flat_results:
            flat_gdf = gpd.GeoDataFrame(pd.concat(flat_results, ignore_index=True), geometry="geometry", crs=flat_results[0].crs)
        else:
            flat_gdf = gpd.GeoDataFrame(columns=list(flat_base.columns) + [
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
            ], geometry="geometry", crs=flat_base.crs)

        if stacked_results:
            stacked_gdf = gpd.GeoDataFrame(
                pd.concat(stacked_results, ignore_index=True),
                geometry="geometry",
                crs=stacked_results[0].crs,
            )
        else:
            stacked_gdf = gpd.GeoDataFrame(columns=list(stacked_base.columns) + [
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
            ], geometry="geometry", crs=stacked_base.crs)

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


def main() -> None:
    args = _parse_args()

    from data_analysis_setup import (
        AnalysisStorage,
        read_config,
        resolve_base_dir,
    )

    base_dir = resolve_base_dir(args.original_working_directory)
    cfg = read_config(base_dir)
    ttk_theme = cfg["DEFAULT"].get("ttk_bootstrap_theme", "flatly") if "DEFAULT" in cfg else "flatly"

    root = tb.Window(themename=ttk_theme)
    root.title("MESA – Analysis processing")
    try:
        ico = base_dir / "system_resources" / "mesa.ico"
        if ico.exists():
            root.iconbitmap(str(ico))
    except Exception:
        pass
    root.geometry("780x420")

    log_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
    progress_queue: queue.SimpleQueue[float] = queue.SimpleQueue()

    def log_to_gui(message: str) -> None:
        formatted = f"{_timestamp()} - {message}"
        log_queue.put(formatted)
        try:
            with open(base_dir / "log.txt", "a", encoding="utf-8") as fh:
                fh.write(formatted + "\n")
        except Exception:
            pass

    def update_progress(value: float) -> None:
        try:
            progress_queue.put(float(value))
        except Exception:
            pass

    def run_processing() -> None:
        last_progress = 0.0

        def set_progress(v: float) -> None:
            nonlocal last_progress
            try:
                v = float(v)
            except Exception:
                return
            v = max(0.0, min(100.0, v))
            if v < last_progress:
                return
            last_progress = v
            update_progress(v)

        try:
            set_progress(1.0)
            log_to_gui("ANALYSIS PROCESS START")
            log_to_gui("This run processes all analysis groups and all polygons within each group.")

            storage = AnalysisStorage(base_dir, cfg)
            set_progress(3.0)
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

            log_to_gui(f"Found {len(groups)} analysis group(s) with {total_polygons} polygon(s) total.")
            set_progress(7.0)

            work_groups = [g for g in groups if polygon_counts.get(g.identifier, 0) > 0]
            total_work = len(work_groups)
            if total_work == 0:
                log_to_gui("No study area polygons found; nothing to do. Use the 'Set up analysis' tool first.")
                set_progress(0.0)
                return

            # Split progress:
            # - Groups + per-polygon: 0..90
            # - Finalization: 90..100
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
                log_to_gui(f"Processing group '{group.name}' ({group.identifier}) with {count} polygon(s)...")

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
                    log_to_gui(
                        f"Completed group '{group.name}': flat_rows={result.get('flat_rows')} stacked_rows={result.get('stacked_rows')}"
                    )
                    processed += 1
                except Exception as exc:
                    failed += 1
                    log_to_gui(f"ERROR processing group '{group.name}' ({group.identifier}): {exc}")
                finally:
                    set_progress(group_end)

            set_progress(GROUP_PHASE_MAX)
            log_to_gui(f"COMPLETED: processed_groups={processed} skipped_groups={skipped} failed_groups={failed}")

            # Finalization bucket.
            set_progress(95.0)
            set_progress(100.0)
        except Exception as exc:
            log_to_gui(f"FATAL ERROR: {exc}")
        finally:
            try:
                root.after(0, lambda: process_button.config(state=tk.NORMAL))
            except Exception:
                pass

    def on_click() -> None:
        process_button.config(state=tk.DISABLED)
        thread = threading.Thread(target=run_processing, daemon=True)
        thread.start()

    def exit_program() -> None:
        try:
            root.destroy()
        except Exception:
            pass

    button_width = 18
    button_padx = 7
    button_pady = 7

    main_frame = tk.Frame(root)
    main_frame.pack(fill="both", expand=True, pady=10)

    log_frame = tb.LabelFrame(main_frame, text="Log output", bootstyle="info")
    log_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    log_widget = scrolledtext.ScrolledText(log_frame, height=10)
    log_widget.pack(fill=tk.BOTH, expand=True)
    log_widget.see(tk.END)

    progress_frame = tk.Frame(main_frame)
    progress_frame.pack(pady=5)

    progress_var = tk.DoubleVar(value=0.0)
    progress_bar = tb.Progressbar(
        progress_frame,
        orient="horizontal",
        length=280,
        mode="determinate",
        variable=progress_var,
        bootstyle="info",
    )
    progress_bar.pack(side=tk.LEFT)
    progress_label = tk.Label(progress_frame, text="0%", bg="light grey")
    progress_label.pack(side=tk.LEFT, padx=5)

    buttons_frame = tk.Frame(main_frame)
    buttons_frame.pack(side="left", fill="both", padx=20, pady=5)

    info_label_text = (
        "Process configured study areas into tbl_analysis_flat.parquet and tbl_analysis_stacked.parquet. "
        "This processes all analysis groups and all polygons in each group."
    )
    info_label = tk.Label(root, text=info_label_text, wraplength=720, justify="left", anchor="w")
    info_label.pack(fill=tk.X, padx=10, pady=8, anchor="w")

    process_button = tb.Button(
        buttons_frame,
        text="Process analysis",
        command=on_click,
        width=button_width,
        bootstyle="primary",
    )
    process_button.grid(row=0, column=0, padx=button_padx, pady=button_pady, sticky="w")

    exit_btn = tb.Button(
        buttons_frame,
        text="Exit",
        command=exit_program,
        width=button_width,
        bootstyle="warning",
    )
    exit_btn.grid(row=0, column=1, padx=button_padx, pady=button_pady, sticky="w")

    def _pump_queues() -> None:
        try:
            while True:
                line = log_queue.get_nowait()
                log_widget.insert(tk.END, line + "\n")
                log_widget.see(tk.END)
        except Exception:
            pass

        try:
            while True:
                v = progress_queue.get_nowait()
                v = max(0.0, min(100.0, float(v)))
                progress_var.set(v)
                progress_label.config(text=f"{int(v)}%")
        except Exception:
            pass

        root.after(100, _pump_queues)

    _pump_queues()
    root.mainloop()


if __name__ == "__main__":
    main()
