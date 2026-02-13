# -*- coding: utf-8 -*-
"""Unified atlas manager: create/import tiles + edit metadata."""

from locale_bootstrap import harden_locale_for_ttkbootstrap

harden_locale_for_ttkbootstrap()

import argparse
import configparser
import datetime
import io
import math
import os
import tempfile
import threading
import urllib.request
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

try:
    import ttkbootstrap as tb
    from ttkbootstrap.constants import PRIMARY, WARNING, SUCCESS
except Exception:
    tb = None
    PRIMARY = "primary"
    WARNING = "warning"
    SUCCESS = "success"

import pandas as pd
import geopandas as gpd
from shapely.geometry import box

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from PIL import Image as PILImage

try:
    import contextily as ctx
except Exception:
    ctx = None


BASE_DIR: Path = Path(".").resolve()
_CFG: configparser.ConfigParser | None = None
_CFG_PATH: Path | None = None
_PARQUET_SUBDIR = "output/geoparquet"
_PARQUET_OVERRIDE: Path | None = None


def _exists(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


def _has_config_at(root: Path) -> bool:
    return _exists(root / "config.ini") or _exists(root / "system" / "config.ini")


def find_base_dir(cli_workdir: str | None = None) -> Path:
    candidates: list[Path] = []

    def _add(path_like):
        if not path_like:
            return
        try:
            candidates.append(Path(path_like))
        except Exception:
            pass

    env_base = os.environ.get("MESA_BASE_DIR")
    if env_base:
        _add(env_base)
    if cli_workdir:
        _add(cli_workdir)

    exe_path: Path | None = None
    try:
        exe_path = Path(os.path.abspath(os.path.realpath(os.sys.executable))).resolve()
    except Exception:
        exe_path = None
    if exe_path:
        _add(exe_path.parent)
        _add(exe_path.parent.parent)
        _add(exe_path.parent.parent.parent)

    meipass = getattr(os.sys, "_MEIPASS", None)
    if meipass:
        _add(meipass)

    here = Path(__file__).resolve()
    _add(here.parent)
    _add(here.parent.parent)
    _add(here.parent.parent.parent)

    cwd = Path.cwd()
    _add(cwd)
    _add(cwd / "code")
    _add(cwd.parent)
    _add(cwd.parent / "code")

    seen = set()
    uniq = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved not in seen:
            seen.add(resolved)
            uniq.append(resolved)

    for candidate in uniq:
        if _has_config_at(candidate):
            return candidate

    if here.parent.name.lower() == "system":
        return here.parent.parent
    if exe_path:
        return exe_path.parent
    if env_base:
        return Path(env_base)
    return here.parent


def _ensure_cfg() -> configparser.ConfigParser:
    global _CFG, _CFG_PATH
    if _CFG is not None:
        return _CFG

    cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    flat = BASE_DIR / "config.ini"
    legacy = BASE_DIR / "system" / "config.ini"

    loaded = False
    if flat.exists():
        try:
            cfg.read(flat, encoding="utf-8")
            _CFG_PATH = flat
            loaded = True
        except Exception:
            pass
    if not loaded and legacy.exists():
        try:
            cfg.read(legacy, encoding="utf-8")
            _CFG_PATH = legacy
            loaded = True
        except Exception:
            pass

    if "DEFAULT" not in cfg:
        cfg["DEFAULT"] = {}

    d = cfg["DEFAULT"]
    d.setdefault("parquet_folder", _PARQUET_SUBDIR)
    d.setdefault("ttk_bootstrap_theme", "flatly")
    d.setdefault("workingprojection_epsg", "4326")
    d.setdefault("input_folder_atlas", "input/atlas")
    d.setdefault("atlas_parquet_file", "tbl_atlas.parquet")
    d.setdefault("atlas_lon_size_km", "10")
    d.setdefault("atlas_lat_size_km", "10")
    d.setdefault("atlas_overlap_percent", "10")

    _CFG = cfg
    return _CFG


def _abs_path_like(value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (BASE_DIR / p).resolve()


def _parquet_candidate_dirs() -> list[Path]:
    cfg = _ensure_cfg()
    sub_cfg = cfg["DEFAULT"].get("parquet_folder", _PARQUET_SUBDIR)
    sub_path = Path(sub_cfg)

    if sub_path.is_absolute():
        return [sub_path.resolve()]

    base = BASE_DIR.resolve()
    candidates: list[Path] = []
    if base.name.lower() == "code":
        parent = base.parent
        if parent:
            candidates.append((parent / sub_path).resolve())
        candidates.append((base / sub_path).resolve())
    else:
        candidates.append((base / sub_path).resolve())
        candidates.append((base / "code" / sub_path).resolve())

    uniq = []
    seen = set()
    for d in candidates:
        if d in seen:
            continue
        seen.add(d)
        uniq.append(d)
    return uniq


def _select_parquet_dir(prefer_file: str | None = None, *, for_write: bool = False) -> Path:
    global _PARQUET_OVERRIDE

    candidates = _parquet_candidate_dirs()
    primary = candidates[0]

    if _PARQUET_OVERRIDE is None:
        _PARQUET_OVERRIDE = primary

    if for_write:
        try:
            _PARQUET_OVERRIDE.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return _PARQUET_OVERRIDE

    if prefer_file:
        for cand in candidates:
            if (cand / prefer_file).exists():
                return cand

    for cand in candidates:
        try:
            if cand.exists() and any(cand.glob("*.parquet")):
                return cand
        except Exception:
            pass

    return primary


def _parquet_path(name: str, *, for_write: bool = False) -> Path:
    directory = _select_parquet_dir(None if for_write else name, for_write=for_write)
    if for_write:
        directory.mkdir(parents=True, exist_ok=True)
    return directory / name


class AtlasManagerApp:
    def __init__(self, root: tk.Tk):
        self.root = root

        self.cfg = _ensure_cfg()
        d = self.cfg["DEFAULT"]

        self.input_folder_atlas = _abs_path_like(d.get("input_folder_atlas", "input/atlas"))
        self.atlas_file_name = d.get("atlas_parquet_file", "tbl_atlas.parquet")

        try:
            self.atlas_lon_size_km = float(d.get("atlas_lon_size_km", "10"))
        except Exception:
            self.atlas_lon_size_km = 10.0
        try:
            self.atlas_lat_size_km = float(d.get("atlas_lat_size_km", "10"))
        except Exception:
            self.atlas_lat_size_km = 10.0
        try:
            self.atlas_overlap_percent = float(d.get("atlas_overlap_percent", "10"))
        except Exception:
            self.atlas_overlap_percent = 10.0

        self.df = pd.DataFrame()
        self.gdf = gpd.GeoDataFrame()
        self.current_index = 0

        self.name_gis_var = tk.StringVar()
        self.title_user_var = tk.StringVar()
        self.description_var = tk.StringVar()
        self.image_name_1_var = tk.StringVar()
        self.image_desc_1_var = tk.StringVar()
        self.image_name_2_var = tk.StringVar()
        self.image_desc_2_var = tk.StringVar()

        self.tile_mode_var = tk.StringVar(value="config")
        self.tile_lon_var = tk.StringVar(value=f"{self.atlas_lon_size_km:.2f}")
        self.tile_lat_var = tk.StringVar(value=f"{self.atlas_lat_size_km:.2f}")
        self.tile_overlap_var = tk.StringVar(value=f"{self.atlas_overlap_percent:.2f}")
        self.tile_count_var = tk.StringVar(value="4")
        self.tile_tolerance_var = tk.StringVar(value="5")
        self.tile_count_overlap_var = tk.StringVar(value=f"{self.atlas_overlap_percent:.2f}")

        self.progress_var = tk.DoubleVar(value=0.0)

        self.map_fig = None
        self.map_ax = None
        self.map_canvas = None
        self.map_gdf_plot = gpd.GeoDataFrame()
        self.map_tile_limit = 400
        self.map_user_agent = "Mozilla/5.0 (compatible; AtlasManager/1.0)"

        self._build_ui()
        self._refresh_edit_data()

    def _atlas_path(self, for_write: bool = False) -> Path:
        return _parquet_path(self.atlas_file_name, for_write=for_write)

    def _log_to_gui(self, message: str):
        timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
        line = f"{timestamp} - {message}"
        try:
            self.log_widget.insert(tk.END, line + "\n")
            self.log_widget.see(tk.END)
        except Exception:
            pass
        try:
            with open(BASE_DIR / "log.txt", "a", encoding="utf-8", errors="replace") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _update_progress(self, value: float):
        try:
            v = max(0.0, min(100.0, float(value)))
            self.progress_var.set(v)
            self.progress_label.config(text=f"{int(v)}%")
        except Exception:
            pass

    def _build_ui(self):
        self.root.title("Atlas")
        try:
            ico = BASE_DIR / "system_resources" / "mesa.ico"
            if ico.exists() and hasattr(self.root, "iconbitmap"):
                self.root.iconbitmap(str(ico))
        except Exception:
            pass

        shell = (tb.Frame(self.root, padding=10) if tb is not None else ttk.Frame(self.root, padding=10))
        shell.pack(fill="both", expand=True)

        self.notebook = ttk.Notebook(shell)
        self.notebook.pack(fill="both", expand=True)

        self.create_tab = (tb.Frame(self.notebook, padding=10) if tb is not None else ttk.Frame(self.notebook, padding=10))
        self.edit_tab = (tb.Frame(self.notebook, padding=10) if tb is not None else ttk.Frame(self.notebook, padding=10))
        self.notebook.add(self.create_tab, text="Create / Import")
        self.notebook.add(self.edit_tab, text="Edit metadata")

        self._build_create_tab(self.create_tab)
        self._build_edit_tab(self.edit_tab)

        bottom = (tb.Frame(shell) if tb is not None else ttk.Frame(shell))
        bottom.pack(fill="x", pady=(8, 0))

        self.summary_label = (tb.Label(bottom, text="") if tb is not None else ttk.Label(bottom, text=""))
        self.summary_label.pack(side="left")

        self.exit_btn = (
            tb.Button(bottom, text="Exit", bootstyle=WARNING, command=self.root.destroy)
            if tb is not None
            else ttk.Button(bottom, text="Exit", command=self.root.destroy)
        )
        self.exit_btn.pack(side="right")

    def _build_create_tab(self, parent):
        options_frame = (tb.LabelFrame(parent, text="Tile generation options") if tb is not None else ttk.LabelFrame(parent, text="Tile generation options"))
        options_frame.pack(fill="x", pady=(0, 8))

        ttk.Radiobutton(options_frame, text="Use config.ini defaults", variable=self.tile_mode_var, value="config").grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Radiobutton(options_frame, text="Custom tile size (km)", variable=self.tile_mode_var, value="custom").grid(row=1, column=0, columnspan=4, sticky="w")

        ttk.Label(options_frame, text="Width (lon km):").grid(row=2, column=0, sticky="w", padx=(20, 4))
        self.entry_lon = ttk.Entry(options_frame, textvariable=self.tile_lon_var, width=10)
        self.entry_lon.grid(row=2, column=1, sticky="w")

        ttk.Label(options_frame, text="Height (lat km):").grid(row=2, column=2, sticky="w", padx=(10, 4))
        self.entry_lat = ttk.Entry(options_frame, textvariable=self.tile_lat_var, width=10)
        self.entry_lat.grid(row=2, column=3, sticky="w")

        ttk.Label(options_frame, text="Overlap % (custom):").grid(row=3, column=0, sticky="w", padx=(20, 4))
        self.entry_overlap = ttk.Entry(options_frame, textvariable=self.tile_overlap_var, width=10)
        self.entry_overlap.grid(row=3, column=1, sticky="w")

        ttk.Radiobutton(options_frame, text="Distribute fixed number of tiles", variable=self.tile_mode_var, value="count").grid(row=4, column=0, columnspan=4, sticky="w", pady=(6, 0))

        ttk.Label(options_frame, text="Tiles:").grid(row=5, column=0, sticky="w", padx=(20, 4))
        self.entry_count = ttk.Entry(options_frame, textvariable=self.tile_count_var, width=10)
        self.entry_count.grid(row=5, column=1, sticky="w")

        ttk.Label(options_frame, text="Padding %:").grid(row=5, column=2, sticky="w", padx=(10, 4))
        self.entry_tol = ttk.Entry(options_frame, textvariable=self.tile_tolerance_var, width=10)
        self.entry_tol.grid(row=5, column=3, sticky="w")

        ttk.Label(options_frame, text="Overlap % (count):").grid(row=6, column=0, sticky="w", padx=(20, 4))
        self.entry_count_overlap = ttk.Entry(options_frame, textvariable=self.tile_count_overlap_var, width=10)
        self.entry_count_overlap.grid(row=6, column=1, sticky="w")

        self.create_mode_custom_widgets = [self.entry_lon, self.entry_lat, self.entry_overlap]
        self.create_mode_count_widgets = [self.entry_count, self.entry_tol, self.entry_count_overlap]

        self.tile_mode_var.trace_add("write", lambda *args: self._update_tile_mode_ui())
        self._update_tile_mode_ui()

        self.log_widget = scrolledtext.ScrolledText(parent, height=10)
        self.log_widget.pack(fill="both", expand=True, pady=(0, 8))

        progress_frame = (tb.Frame(parent) if tb is not None else ttk.Frame(parent))
        progress_frame.pack(fill="x")
        progress = ttk.Progressbar(progress_frame, orient="horizontal", length=300, mode="determinate", variable=self.progress_var)
        progress.pack(side="left")
        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.pack(side="left", padx=8)

        btn_frame = (tb.Frame(parent) if tb is not None else ttk.Frame(parent))
        btn_frame.pack(fill="x", pady=(8, 0))

        create_btn = (
            tb.Button(btn_frame, text="Create", bootstyle=PRIMARY, command=lambda: self._spawn(self._run_create))
            if tb is not None
            else ttk.Button(btn_frame, text="Create", command=lambda: self._spawn(self._run_create))
        )
        create_btn.pack(side="left")

        import_btn = (
            tb.Button(btn_frame, text="Import", bootstyle=PRIMARY, command=lambda: self._spawn(self._run_import_action))
            if tb is not None
            else ttk.Button(btn_frame, text="Import", command=lambda: self._spawn(self._run_import_action))
        )
        import_btn.pack(side="left", padx=(8, 0))

        reload_edit_btn = (
            tb.Button(btn_frame, text="Refresh editor", command=self._refresh_edit_data)
            if tb is not None
            else ttk.Button(btn_frame, text="Refresh editor", command=self._refresh_edit_data)
        )
        reload_edit_btn.pack(side="left", padx=(8, 0))

    def _build_edit_tab(self, parent):
        top = (tb.Frame(parent) if tb is not None else ttk.Frame(parent))
        top.pack(fill="x", pady=(0, 8))

        self.edit_state_label = (tb.Label(top, text="") if tb is not None else ttk.Label(top, text=""))
        self.edit_state_label.pack(side="left")

        content = (tb.Frame(parent) if tb is not None else ttk.Frame(parent))
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        form = (tb.Frame(content) if tb is not None else ttk.Frame(content))
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="GIS Name").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Label(form, textvariable=self.name_gis_var).grid(row=0, column=1, sticky="w", pady=4)

        ttk.Label(form, text="Title").grid(row=1, column=0, sticky="w", pady=4)
        self.title_entry = ttk.Entry(form, textvariable=self.title_user_var)
        self.title_entry.grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(form, text="Image Name 1").grid(row=2, column=0, sticky="w", pady=4)
        image1_wrap = (tb.Frame(form) if tb is not None else ttk.Frame(form))
        image1_wrap.grid(row=2, column=1, sticky="ew", pady=4)
        image1_wrap.columnconfigure(0, weight=1)
        self.image1_entry = ttk.Entry(image1_wrap, textvariable=self.image_name_1_var)
        self.image1_entry.grid(row=0, column=0, sticky="ew")
        ttk.Button(image1_wrap, text="Browse", command=self._browse_image_1).grid(row=0, column=1, padx=(6, 0))

        ttk.Label(form, text="Image 1 description").grid(row=3, column=0, sticky="w", pady=4)
        self.image1_desc_entry = ttk.Entry(form, textvariable=self.image_desc_1_var)
        self.image1_desc_entry.grid(row=3, column=1, sticky="ew", pady=4)

        ttk.Label(form, text="Image Name 2").grid(row=4, column=0, sticky="w", pady=4)
        image2_wrap = (tb.Frame(form) if tb is not None else ttk.Frame(form))
        image2_wrap.grid(row=4, column=1, sticky="ew", pady=4)
        image2_wrap.columnconfigure(0, weight=1)
        self.image2_entry = ttk.Entry(image2_wrap, textvariable=self.image_name_2_var)
        self.image2_entry.grid(row=0, column=0, sticky="ew")
        ttk.Button(image2_wrap, text="Browse", command=self._browse_image_2).grid(row=0, column=1, padx=(6, 0))

        ttk.Label(form, text="Image 2 description").grid(row=5, column=0, sticky="w", pady=4)
        self.image2_desc_entry = ttk.Entry(form, textvariable=self.image_desc_2_var)
        self.image2_desc_entry.grid(row=5, column=1, sticky="ew", pady=4)

        ttk.Label(form, text="Description").grid(row=6, column=0, sticky="w", pady=4)
        self.description_entry = ttk.Entry(form, textvariable=self.description_var)
        self.description_entry.grid(row=6, column=1, sticky="ew", pady=4)

        map_wrap = (tb.Frame(content) if tb is not None else ttk.Frame(content))
        map_wrap.grid(row=0, column=1, sticky="nsew")
        map_wrap.columnconfigure(0, weight=1)
        map_wrap.rowconfigure(1, weight=1)

        ttk.Label(map_wrap, text="Atlas preview").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self._build_map_widget(map_wrap)

        controls = (tb.Frame(parent) if tb is not None else ttk.Frame(parent))
        controls.pack(fill="x", pady=(8, 0))

        self.prev_btn = ttk.Button(controls, text="Previous", command=lambda: self._navigate(-1))
        self.prev_btn.pack(side="left")
        self.next_btn = ttk.Button(controls, text="Next", command=lambda: self._navigate(1))
        self.next_btn.pack(side="left", padx=(6, 0))

        save_btn = (
            tb.Button(controls, text="Save", bootstyle=SUCCESS, command=self._save_current)
            if tb is not None
            else ttk.Button(controls, text="Save", command=self._save_current)
        )
        save_btn.pack(side="right")

        self.reload_btn = ttk.Button(controls, text="Reload", command=self._refresh_edit_data)
        self.reload_btn.pack(side="right", padx=(0, 6))

    def _spawn(self, fn):
        threading.Thread(target=fn, daemon=True).start()

    def _update_tile_mode_ui(self):
        mode = (self.tile_mode_var.get() or "config").lower()
        custom_state = tk.NORMAL if mode == "custom" else tk.DISABLED
        count_state = tk.NORMAL if mode == "count" else tk.DISABLED
        for widget in self.create_mode_custom_widgets:
            widget.configure(state=custom_state)
        for widget in self.create_mode_count_widgets:
            widget.configure(state=count_state)

    def _read_asset_group_parquet(self) -> gpd.GeoDataFrame:
        p = _parquet_path("tbl_asset_group.parquet")
        if not p.exists():
            self._log_to_gui(f"Missing tbl_asset_group.parquet at {p}")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        try:
            gdf = gpd.read_parquet(p)
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            return gdf
        except Exception as e:
            self._log_to_gui(f"Failed reading tbl_asset_group.parquet: {e}")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    def _read_flat_parquet(self) -> gpd.GeoDataFrame:
        p = _parquet_path("tbl_flat.parquet")
        if not p.exists():
            self._log_to_gui(f"Missing tbl_flat.parquet at {p}")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        try:
            gdf = gpd.read_parquet(p)
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            return gdf
        except Exception as e:
            self._log_to_gui(f"Failed reading tbl_flat.parquet: {e}")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    def _write_atlas_parquet(self, gdf: gpd.GeoDataFrame):
        p = self._atlas_path(for_write=True)
        gdf.to_parquet(p, index=False)
        self._log_to_gui(f"Saved atlas GeoParquet -> {p} (rows={len(gdf)})")

    def _atlas_bounds_from_asset_groups(self):
        ag = self._read_asset_group_parquet()
        if ag.empty or "geometry" not in ag:
            return None
        try:
            geom = ag.geometry
            mask = geom.notna()
            try:
                mask &= ~geom.is_empty
            except Exception:
                pass
            try:
                mask &= geom.geom_type.isin(["Polygon", "MultiPolygon", "GeometryCollection"])
            except Exception:
                pass
            valid = geom[mask]
            if valid.empty:
                return None
            minx, miny, maxx, maxy = valid.total_bounds
            return (minx, miny, maxx, maxy)
        except Exception:
            return None

    def _determine_atlas_bounds(self, tbl_flat: gpd.GeoDataFrame | None):
        bounds = self._atlas_bounds_from_asset_groups()
        if bounds:
            self._log_to_gui("Atlas extent derived from tbl_asset_group polygons.")
            return bounds
        if tbl_flat is not None and not tbl_flat.empty:
            try:
                minx, miny, maxx, maxy = tbl_flat.total_bounds
                self._log_to_gui("tbl_asset_group unavailable; using tbl_flat extent.")
                return (minx, miny, maxx, maxy)
            except Exception:
                pass
        self._log_to_gui("No extent available from tbl_asset_group or tbl_flat.")
        return None

    def _generate_atlas_geometries(self, bounds, lon_km, lat_km, overlap_pct):
        self._log_to_gui("Generating atlas geometries...")
        lon_size_deg = float(lon_km) / 111.0
        lat_size_deg = float(lat_km) / 111.0
        overlap = float(overlap_pct) / 100.0

        if not bounds:
            return []

        minx, miny, maxx, maxy = bounds
        if maxx <= minx:
            maxx = minx + 1e-6
        if maxy <= miny:
            maxy = miny + 1e-6

        atlas_geometries = []
        idx = 1
        y = miny
        step_y = max(1e-9, lat_size_deg * (1.0 - overlap))
        step_x = max(1e-9, lon_size_deg * (1.0 - overlap))

        while y < maxy:
            x = minx
            while x < maxx:
                atlas_geometries.append(
                    {
                        "id": idx,
                        "name_gis": f"atlas{idx:03d}",
                        "title_user": f"Map title for {idx:03d}",
                        "geom": box(x, y, x + lon_size_deg, y + lat_size_deg),
                        "description": "",
                        "image_name_1": "",
                        "image_desc_1": "",
                        "image_name_2": "",
                        "image_desc_2": "",
                    }
                )
                idx += 1
                x += step_x
            y += step_y
        return atlas_geometries

    def _generate_atlas_geometries_by_count(self, bounds, tile_count: int, tolerance_pct: float, overlap_pct: float):
        self._log_to_gui(
            f"Generating atlas geometries for {tile_count} tiles (tolerance {tolerance_pct}%, overlap {overlap_pct}%)."
        )
        if not bounds:
            return []

        minx, miny, maxx, maxy = bounds
        if maxx <= minx:
            maxx = minx + 1e-6
        if maxy <= miny:
            maxy = miny + 1e-6

        width = maxx - minx
        height = maxy - miny
        tile_count = max(2, int(tile_count))
        tolerance_pct = max(0.0, float(tolerance_pct))
        overlap_pct = max(0.0, min(float(overlap_pct), 90.0))

        pad_x = width * (tolerance_pct / 100.0) / 2.0
        pad_y = height * (tolerance_pct / 100.0) / 2.0
        expanded_minx = minx - pad_x
        expanded_maxx = maxx + pad_x
        expanded_miny = miny - pad_y
        expanded_maxy = maxy + pad_y

        total_width = expanded_maxx - expanded_minx
        total_height = expanded_maxy - expanded_miny
        ratio = max(total_width / total_height if total_height > 0 else 1.0, 1e-6)

        cols = max(1, math.ceil(math.sqrt(tile_count * ratio)))
        rows = max(1, math.ceil(tile_count / cols))
        while cols * rows < tile_count:
            rows += 1

        tile_width = total_width / cols
        tile_height = total_height / rows
        overlap_fraction = min(overlap_pct / 100.0, 0.9)
        expand_x = tile_width * overlap_fraction * 0.5
        expand_y = tile_height * overlap_fraction * 0.5

        atlas_geometries = []
        idx = 1
        for row in range(rows):
            y0 = expanded_miny + row * tile_height
            y1 = expanded_miny + (row + 1) * tile_height
            if row == rows - 1:
                y1 = expanded_maxy
            for col in range(cols):
                x0 = expanded_minx + col * tile_width
                x1 = expanded_minx + (col + 1) * tile_width
                if col == cols - 1:
                    x1 = expanded_maxx
                atlas_geometries.append(
                    {
                        "id": idx,
                        "name_gis": f"atlas{idx:03d}",
                        "title_user": f"Map title for {idx:03d}",
                        "geom": box(
                            max(expanded_minx, x0 - expand_x),
                            max(expanded_miny, y0 - expand_y),
                            min(expanded_maxx, x1 + expand_x),
                            min(expanded_maxy, y1 + expand_y),
                        ),
                        "description": "",
                        "image_name_1": "",
                        "image_desc_1": "",
                        "image_name_2": "",
                        "image_desc_2": "",
                    }
                )
                idx += 1
        return atlas_geometries

    def _filter_and_update_atlas_geometries(self, atlas_geometries, tbl_flat: gpd.GeoDataFrame | None):
        atlas_gdf = gpd.GeoDataFrame(
            atlas_geometries,
            columns=[
                "id",
                "name_gis",
                "title_user",
                "geom",
                "description",
                "image_name_1",
                "image_desc_1",
                "image_name_2",
                "image_desc_2",
            ],
        )
        atlas_gdf.set_geometry("geom", inplace=True)

        target_crs = None
        if tbl_flat is not None and hasattr(tbl_flat, "crs"):
            target_crs = tbl_flat.crs
        try:
            if target_crs is not None:
                if atlas_gdf.crs is None:
                    atlas_gdf.set_crs(target_crs, inplace=True)
                elif atlas_gdf.crs != target_crs:
                    atlas_gdf = atlas_gdf.to_crs(target_crs)
        except Exception:
            pass

        if tbl_flat is None or tbl_flat.empty:
            intersecting = atlas_gdf.copy()
        else:
            try:
                filtered = atlas_gdf.geometry.apply(lambda geom: tbl_flat.intersects(geom).any())
            except Exception:
                filtered = [True] * len(atlas_gdf)
            intersecting = atlas_gdf[filtered].copy()

        idx = 1
        for row_idx in intersecting.index:
            intersecting.loc[row_idx, "name_gis"] = f"atlas_{idx:03d}"
            intersecting.loc[row_idx, "title_user"] = f"Map title for {idx:03d}"
            idx += 1

        return intersecting.rename(columns={"geom": "geometry"}).set_geometry("geometry")

    def _run_create(self):
        self._log_to_gui("Starting atlas generation (GeoParquet).")
        self._update_progress(10)

        tbl_flat = self._read_flat_parquet()
        tbl_flat_data = None if tbl_flat.empty else tbl_flat

        bounds = self._determine_atlas_bounds(tbl_flat_data)
        if not bounds:
            self._log_to_gui("Cannot derive atlas extent (need tbl_asset_group polygons or tbl_flat bounds).")
            self._update_progress(100)
            return

        self._update_progress(30)

        mode = (self.tile_mode_var.get() or "config").lower()
        atlas_geometries = []

        if mode == "custom":
            try:
                lon_km = float(self.tile_lon_var.get())
                lat_km = float(self.tile_lat_var.get())
                overlap = float(self.tile_overlap_var.get())
            except Exception:
                self._log_to_gui("Invalid custom tile values.")
                self._update_progress(100)
                return
            if lon_km <= 0 or lat_km <= 0:
                self._log_to_gui("Tile sizes must be positive.")
                self._update_progress(100)
                return
            overlap = max(0.0, min(overlap, 90.0))
            atlas_geometries = self._generate_atlas_geometries(bounds, lon_km, lat_km, overlap)
        elif mode == "count":
            try:
                requested_tiles = max(2, int(self.tile_count_var.get()))
                tolerance = max(0.0, float(self.tile_tolerance_var.get()))
                overlap_pct = max(0.0, min(float(self.tile_count_overlap_var.get()), 90.0))
            except Exception:
                self._log_to_gui("Invalid count-mode values.")
                self._update_progress(100)
                return
            atlas_geometries = self._generate_atlas_geometries_by_count(bounds, requested_tiles, tolerance, overlap_pct)
        else:
            atlas_geometries = self._generate_atlas_geometries(
                bounds,
                self.atlas_lon_size_km,
                self.atlas_lat_size_km,
                self.atlas_overlap_percent,
            )

        if not atlas_geometries:
            self._log_to_gui("No atlas tiles generated.")
            self._update_progress(100)
            return

        self._update_progress(60)
        updated = self._filter_and_update_atlas_geometries(atlas_geometries, tbl_flat_data)
        self._update_progress(80)

        if not updated.empty:
            self._write_atlas_parquet(updated)
        else:
            fallback_crs = tbl_flat_data.crs if tbl_flat_data is not None else "EPSG:4326"
            empty = gpd.GeoDataFrame(
                columns=[
                    "id",
                    "name_gis",
                    "title_user",
                    "description",
                    "image_name_1",
                    "image_desc_1",
                    "image_name_2",
                    "image_desc_2",
                    "geometry",
                ],
                geometry="geometry",
                crs=fallback_crs,
            )
            self._write_atlas_parquet(empty)

        self._update_progress(100)
        self._log_to_gui("COMPLETED: Atlas creation saved to GeoParquet.")
        self.root.after(0, self._refresh_edit_data)

    def _process_spatial_file(self, filepath: Path, atlas_objects: list[dict], atlas_id_counter: int) -> int:
        try:
            gdf = gpd.read_file(filepath)
            polys = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
            for _, row in polys.iterrows():
                atlas_objects.append(
                    {
                        "id": atlas_id_counter,
                        "name_gis": f"atlas{atlas_id_counter:03d}",
                        "title_user": f"Map title for {atlas_id_counter:03d}",
                        "geom": row.geometry,
                        "description": "",
                        "image_name_1": "",
                        "image_desc_1": "",
                        "image_name_2": "",
                        "image_desc_2": "",
                    }
                )
                atlas_id_counter += 1
            self._log_to_gui(f"Processed file: {filepath}")
        except Exception as e:
            self._log_to_gui(f"Error processing file {filepath}: {e}")
        return atlas_id_counter

    def _run_import_action(self):
        self._log_to_gui("Starting atlas import (GeoParquet)…")
        self._update_progress(10)

        atlas_objects = []
        atlas_id_counter = 1
        patterns = ("*.shp", "*.gpkg", "*.parquet")

        files = []
        for pat in patterns:
            files.extend(self.input_folder_atlas.rglob(pat))
        total = len(files)
        step = (70.0 / total) if total else 70.0

        processed = 0
        for fp in files:
            try:
                if fp.suffix.lower() == ".parquet":
                    gdf = gpd.read_parquet(fp)
                    polys = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])] if not gdf.empty else gdf
                    for _, row in polys.iterrows():
                        atlas_objects.append(
                            {
                                "id": atlas_id_counter,
                                "name_gis": f"atlas{atlas_id_counter:03d}",
                                "title_user": f"Map title for {atlas_id_counter:03d}",
                                "geom": row.geometry,
                                "description": "",
                                "image_name_1": "",
                                "image_desc_1": "",
                                "image_name_2": "",
                                "image_desc_2": "",
                            }
                        )
                        atlas_id_counter += 1
                    self._log_to_gui(f"Processed file: {fp}")
                else:
                    atlas_id_counter = self._process_spatial_file(fp, atlas_objects, atlas_id_counter)
            except Exception as e:
                self._log_to_gui(f"Error processing file {fp}: {e}")
            finally:
                processed += 1
                self._update_progress(10 + processed * step)

        if atlas_objects:
            atlas_objects_gdf = gpd.GeoDataFrame(atlas_objects, geometry="geom").rename(columns={"geom": "geometry"}).set_geometry("geometry")
        else:
            atlas_objects_gdf = gpd.GeoDataFrame(
                columns=[
                    "id",
                    "name_gis",
                    "title_user",
                    "description",
                    "image_name_1",
                    "image_desc_1",
                    "image_name_2",
                    "image_desc_2",
                    "geometry",
                ],
                geometry="geometry",
            )

        self._write_atlas_parquet(atlas_objects_gdf)
        self._update_progress(100)
        self._log_to_gui("COMPLETED: Atlas polygons imported (GeoParquet).")
        self.root.after(0, self._refresh_edit_data)

    def _browse_image_1(self):
        fp = filedialog.askopenfilename(
            initialdir=str((BASE_DIR / "input" / "images")),
            title="Select file",
            filetypes=(("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png"), ("All files", "*.*")),
        )
        if fp:
            self.image_name_1_var.set(fp)

    def _browse_image_2(self):
        fp = filedialog.askopenfilename(
            initialdir=str((BASE_DIR / "input" / "images")),
            title="Select file",
            filetypes=(("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png"), ("All files", "*.*")),
        )
        if fp:
            self.image_name_2_var.set(fp)

    def _set_edit_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        for w in [
            self.title_entry,
            self.image1_entry,
            self.image1_desc_entry,
            self.image2_entry,
            self.image2_desc_entry,
            self.description_entry,
            self.prev_btn,
            self.next_btn,
            self.reload_btn,
        ]:
            try:
                w.configure(state=state)
            except Exception:
                pass

    def _build_map_widget(self, parent):
        self.map_fig = Figure(figsize=(4.0, 4.0), dpi=100)
        self.map_ax = self.map_fig.add_subplot(111)
        self.map_ax.set_title("Atlas overview", fontsize=10)
        self.map_ax.set_xticks([])
        self.map_ax.set_yticks([])
        self.map_ax.set_aspect("equal", adjustable="box")
        self.map_canvas = FigureCanvasTkAgg(self.map_fig, master=parent)
        self.map_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

    def _merc_to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        radius = 6378137.0
        lon = (x / radius) * 180.0 / math.pi
        lat = (2.0 * math.atan(math.exp(y / radius)) - math.pi / 2.0) * 180.0 / math.pi
        return lon, lat

    def _lonlat_to_merc(self, lon: float, lat: float) -> tuple[float, float]:
        radius = 6378137.0
        x = lon * math.pi / 180.0 * radius
        lat_clamped = max(-85.05112878, min(85.05112878, float(lat)))
        y = math.log(math.tan((90.0 + lat_clamped) * math.pi / 360.0)) * radius
        return x, y

    def _lonlat_to_tile(self, lon: float, lat: float, zoom: int) -> tuple[int, int]:
        n = 2 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        lat_clamped = max(-85.05112878, min(85.05112878, float(lat)))
        lat_rad = math.radians(lat_clamped)
        y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    def _tile_bounds_lonlat(self, zoom: int, x: int, y: int) -> tuple[float, float, float, float]:
        n = 2.0 ** zoom
        minlon = x / n * 360.0 - 180.0
        maxlon = (x + 1) / n * 360.0 - 180.0

        def tiley_to_lat(tile_y: int) -> float:
            y_ratio = tile_y / n
            lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y_ratio)))
            return math.degrees(lat_rad)

        maxlat = tiley_to_lat(y)
        minlat = tiley_to_lat(y + 1)
        return (minlon, minlat, maxlon, maxlat)

    def _fetch_osm_tiles(self, xlim: tuple[float, float], ylim: tuple[float, float]):
        minx, maxx = xlim
        miny, maxy = ylim
        lon_left, lat_bottom = self._merc_to_lonlat(minx, miny)
        lon_right, lat_top = self._merc_to_lonlat(maxx, maxy)

        lon_min, lon_max = sorted((lon_left, lon_right))
        lat_min, lat_max = sorted((lat_bottom, lat_top))

        def tile_range(zoom_level: int) -> tuple[int, int, int, int]:
            tx0, ty0 = self._lonlat_to_tile(lon_min, lat_min, zoom_level)
            tx1, ty1 = self._lonlat_to_tile(lon_max, lat_max, zoom_level)
            return min(tx0, tx1), max(tx0, tx1), min(ty0, ty1), max(ty0, ty1)

        zoom = 3
        target_px = 512
        for test_zoom in range(3, 20):
            xmin, xmax, ymin, ymax = tile_range(test_zoom)
            width = (abs(xmax - xmin) + 1) * 256
            height = (abs(ymax - ymin) + 1) * 256
            if width >= target_px and height >= target_px:
                zoom = test_zoom
                break

        xmin, xmax, ymin, ymax = tile_range(zoom)
        tile_span = (xmax - xmin + 1) * (ymax - ymin + 1)
        while tile_span > self.map_tile_limit and zoom > 3:
            zoom -= 1
            xmin, xmax, ymin, ymax = tile_range(zoom)
            tile_span = (xmax - xmin + 1) * (ymax - ymin + 1)

        if tile_span > self.map_tile_limit:
            return None

        tile_size = 256
        canvas_w = (xmax - xmin + 1) * tile_size
        canvas_h = (ymax - ymin + 1) * tile_size
        mosaic = PILImage.new("RGB", (canvas_w, canvas_h), (240, 240, 240))

        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", self.map_user_agent)]
        urllib.request.install_opener(opener)

        for xi, x in enumerate(range(xmin, xmax + 1)):
            for yi, y in enumerate(range(ymin, ymax + 1)):
                try:
                    url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
                    with urllib.request.urlopen(url, timeout=5) as response:
                        data = response.read()
                    tile_img = PILImage.open(io.BytesIO(data)).convert("RGB")
                    mosaic.paste(tile_img, (xi * tile_size, yi * tile_size))
                except Exception:
                    pass

        west, _, _, north = self._tile_bounds_lonlat(zoom, xmin, ymin)
        east = self._tile_bounds_lonlat(zoom, xmax, ymax)[2]
        south = self._tile_bounds_lonlat(zoom, xmax, ymax)[1]
        lx, north_y = self._lonlat_to_merc(west, north)
        rx, south_y = self._lonlat_to_merc(east, south)

        return mosaic, [lx, rx, south_y, north_y]

    def _draw_basemap(self, xlim: tuple[float, float], ylim: tuple[float, float]):
        if self.map_ax is None:
            return
        self.map_ax.set_facecolor("#e9edf4")
        crs_epsg = None
        try:
            crs_epsg = self.map_gdf_plot.crs.to_epsg() if self.map_gdf_plot is not None else None
        except Exception:
            crs_epsg = None

        if crs_epsg == 3857 and ctx is not None:
            try:
                ctx.add_basemap(self.map_ax, crs="EPSG:3857", source=ctx.providers.OpenStreetMap.Mapnik, reset_extent=False)
                return
            except Exception:
                pass

        if crs_epsg != 3857:
            return

        fetched = self._fetch_osm_tiles(xlim, ylim)
        if fetched is None:
            return
        mosaic, extent = fetched
        self.map_ax.imshow(mosaic, extent=extent, origin="upper", interpolation="bilinear", zorder=0)

    def _refresh_map(self):
        if self.map_ax is None or self.map_canvas is None:
            return

        self.map_ax.clear()
        self.map_ax.set_title("Atlas overview", fontsize=10)
        self.map_ax.set_xticks([])
        self.map_ax.set_yticks([])
        self.map_ax.set_aspect("equal", adjustable="box")

        if self.gdf is None or len(self.gdf) == 0:
            self.map_ax.text(0.5, 0.5, "No atlas data available.", transform=self.map_ax.transAxes, ha="center", va="center")
            self.map_canvas.draw_idle()
            return

        try:
            g = self.gdf.copy()
            if g.crs is None:
                g = g.set_crs("EPSG:4326")
            self.map_gdf_plot = g.to_crs(3857)
        except Exception:
            self.map_gdf_plot = self.gdf.copy()

        try:
            minx, miny, maxx, maxy = self.map_gdf_plot.total_bounds
            if not all(math.isfinite(v) for v in (minx, miny, maxx, maxy)):
                raise ValueError("invalid bounds")
            if maxx - minx <= 0:
                maxx = minx + 1.0
            if maxy - miny <= 0:
                maxy = miny + 1.0
            pad_x = (maxx - minx) * 0.05
            pad_y = (maxy - miny) * 0.05
            xlim = (minx - pad_x, maxx + pad_x)
            ylim = (miny - pad_y, maxy + pad_y)
            self.map_ax.set_xlim(*xlim)
            self.map_ax.set_ylim(*ylim)
            self._draw_basemap(xlim, ylim)
        except Exception:
            pass

        try:
            self.map_gdf_plot.plot(
                ax=self.map_ax,
                edgecolor="#6b8aa1",
                facecolor="#d5e2f0",
                linewidth=0.5,
                alpha=0.85,
                zorder=1,
            )
            if 0 <= self.current_index < len(self.map_gdf_plot):
                geom = self.map_gdf_plot.iloc[self.current_index].geometry
                if geom is not None and not geom.is_empty:
                    gpd.GeoSeries([geom], crs=self.map_gdf_plot.crs).plot(
                        ax=self.map_ax,
                        edgecolor="#e74c3c",
                        facecolor="#f8b4aa",
                        linewidth=1.2,
                        alpha=0.9,
                        zorder=2,
                    )
        except Exception:
            self.map_ax.text(0.5, 0.5, "Unable to render atlas preview.", transform=self.map_ax.transAxes, ha="center", va="center")

        self.map_canvas.draw_idle()

    def _refresh_edit_data(self):
        path = self._atlas_path(for_write=False)
        if not path.exists():
            self.df = pd.DataFrame()
            self.gdf = gpd.GeoDataFrame()
            self.current_index = 0
            self._clear_form()
            self._set_edit_enabled(False)
            self.edit_state_label.config(text="No atlas data. Create or import atlas first.")
            self.summary_label.config(text=f"Atlas file: {path} | pages: 0")
            self._refresh_map()
            return

        try:
            self.gdf = gpd.read_parquet(path)
            if self.gdf.crs is None:
                self.gdf = self.gdf.set_crs("EPSG:4326")
            geom_col = self.gdf.geometry.name if hasattr(self.gdf, "geometry") else "geometry"
            self.df = pd.DataFrame(self.gdf.drop(columns=[geom_col], errors="ignore"))
            for col in ["name_gis", "title_user", "description", "image_name_1", "image_desc_1", "image_name_2", "image_desc_2"]:
                if col not in self.df.columns:
                    self.df[col] = ""
            self.current_index = min(self.current_index, max(len(self.df) - 1, 0))
            self._load_record()
            self._set_edit_enabled(len(self.df) > 0)
            self.edit_state_label.config(text=f"Loaded {len(self.df)} atlas page(s).")
            self.summary_label.config(text=f"Atlas file: {path} | pages: {len(self.df)}")
            self._refresh_map()
        except Exception as exc:
            messagebox.showerror("Error", f"Could not read atlas data:\n{exc}")
            self.df = pd.DataFrame()
            self.gdf = gpd.GeoDataFrame()
            self._clear_form()
            self._set_edit_enabled(False)
            self.edit_state_label.config(text="Failed to load atlas data.")
            self.summary_label.config(text=f"Atlas file: {path} | pages: 0")
            self._refresh_map()

    def _clear_form(self):
        self.name_gis_var.set("")
        self.title_user_var.set("")
        self.description_var.set("")
        self.image_name_1_var.set("")
        self.image_desc_1_var.set("")
        self.image_name_2_var.set("")
        self.image_desc_2_var.set("")

    def _load_record(self):
        if self.df.empty:
            self._clear_form()
            self._refresh_map()
            return
        row = self.df.iloc[self.current_index]
        self.name_gis_var.set("" if pd.isna(row.get("name_gis")) else str(row.get("name_gis")))
        self.title_user_var.set("" if pd.isna(row.get("title_user")) else str(row.get("title_user")))
        self.description_var.set("" if pd.isna(row.get("description")) else str(row.get("description")))
        self.image_name_1_var.set("" if pd.isna(row.get("image_name_1")) else str(row.get("image_name_1")))
        self.image_desc_1_var.set("" if pd.isna(row.get("image_desc_1")) else str(row.get("image_desc_1")))
        self.image_name_2_var.set("" if pd.isna(row.get("image_name_2")) else str(row.get("image_name_2")))
        self.image_desc_2_var.set("" if pd.isna(row.get("image_desc_2")) else str(row.get("image_desc_2")))
        self.edit_state_label.config(text=f"Record {self.current_index + 1} of {len(self.df)}")
        self._refresh_map()

    def _save_current(self):
        if self.df.empty:
            return
        key = self.name_gis_var.get().strip()
        if not key:
            messagebox.showerror("Error", "Missing name_gis")
            return

        self.df.at[self.current_index, "title_user"] = self.title_user_var.get()
        self.df.at[self.current_index, "description"] = self.description_var.get()
        self.df.at[self.current_index, "image_name_1"] = self.image_name_1_var.get()
        self.df.at[self.current_index, "image_desc_1"] = self.image_desc_1_var.get()
        self.df.at[self.current_index, "image_name_2"] = self.image_name_2_var.get()
        self.df.at[self.current_index, "image_desc_2"] = self.image_desc_2_var.get()

        path = self._atlas_path(for_write=False)
        if not path.exists():
            messagebox.showerror("Error", f"File not found:\n{path}")
            return

        try:
            gdf_local = gpd.read_parquet(path)
            if "name_gis" not in gdf_local.columns:
                messagebox.showerror("Error", "GeoParquet does not contain name_gis.")
                return

            idx = gdf_local.index[gdf_local["name_gis"].astype(str) == key]
            if len(idx) == 0:
                messagebox.showerror("Error", f"No record found for name_gis='{key}'.")
                return

            for col in ["title_user", "description", "image_name_1", "image_desc_1", "image_name_2", "image_desc_2"]:
                if col in gdf_local.columns:
                    gdf_local.loc[idx, col] = self.df.at[self.current_index, col]

            write_path = self._atlas_path(for_write=True)
            write_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=write_path.parent, suffix=".parquet", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                gdf_local.to_parquet(tmp_path, index=False)
                os.replace(tmp_path, write_path)
            finally:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass

            self.edit_state_label.config(text="Saved.")
            self.summary_label.config(text=f"Atlas file: {write_path} | pages: {len(self.df)}")
        except Exception as exc:
            messagebox.showerror("Error", f"Save failed:\n{exc}")

    def _navigate(self, direction: int):
        if self.df.empty:
            return
        nxt = self.current_index + int(direction)
        nxt = max(0, min(len(self.df) - 1, nxt))
        if nxt == self.current_index:
            return
        self.current_index = nxt
        self._load_record()


def main():
    global BASE_DIR

    parser = argparse.ArgumentParser(description="Unified atlas create/import + edit metadata")
    parser.add_argument("--original_working_directory", required=False, help="Path to running folder")
    args = parser.parse_args()

    BASE_DIR = find_base_dir(args.original_working_directory)
    cfg = _ensure_cfg()

    if tb is not None:
        try:
            theme = cfg["DEFAULT"].get("ttk_bootstrap_theme", "flatly")
            root = tb.Window(themename=theme)
        except Exception:
            root = tb.Window(themename="flatly")
    else:
        root = tk.Tk()

    AtlasManagerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
