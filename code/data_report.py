import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except Exception:
    # Fall back silently if locale isn't available on this system
    pass

import geopandas as gpd
import pandas as pd
import configparser
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm  # still used for ScalarMappable (not deprecated)
from matplotlib import colormaps as mpl_cmaps  # modern colormap access
import matplotlib.colors as mcolors
import numpy as np
import argparse
import os
import sys
try:
    import contextily as ctx  # optional; heavy deps (rasterio) may be absent in EXE
except Exception:
    ctx = None
from matplotlib.patches import Rectangle

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Image, Spacer,
    Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm as RL_CM
from reportlab.lib.colors import HexColor
from PIL import Image as PILImage
import re
import tkinter as tk
from tkinter import scrolledtext
import ttkbootstrap as tb
from ttkbootstrap.constants import PRIMARY, WARNING
import threading
from pathlib import Path
import subprocess
import io, math, time, urllib.request

# ---------------- UI / sizing constants ----------------
MAX_MAP_PX_HEIGHT = 2000           # hard cap for saved map PNG height (px)
MAX_MAP_CM_HEIGHT = 10.0           # map display cap inside PDF (cm)
RIBBON_CM_HEIGHT   = 0.6           # ribbon display height inside PDF (cm)

TILE_CACHE_MAX_AGE_DAYS = 30       # discard cached OSM tiles older than this (<=0 keeps forever)

# ---------------- GUI / globals ----------------
log_widget = None
progress_var = None
progress_label = None
last_report_path = None
link_var = None  # hyperlink label StringVar
report_mode_var = None  # radio button selection for report detail level
atlas_geocode_var = None  # Combobox selection for atlas geocode group
_atlas_geocode_choices: list[str] = []  # cached list for GUI

SENSITIVITY_ORDER = ['A', 'B', 'C', 'D', 'E']
SENSITIVITY_UNKNOWN_COLOR = "#FF00F2"
_SENSITIVITY_NUMERIC_RANGES: list[tuple[str, float, float]] = []

class ReportEngine:
    """
    Helper responsible for rendering all cartographic artefacts and tracking temporary files.
    """

    def __init__(self,
                 base_dir: str,
                 tmp_dir: str,
                 palette: dict,
                 desc: dict,
                 config_path: str):
        self.base_dir = base_dir
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.palette = palette
        self.desc = desc
        self.config_path = config_path
        self.generated_paths: list[Path] = []

    def register(self, path: str | Path):
        self.generated_paths.append(Path(path))

    def make_path(self, *parts: str, suffix: str = ".png") -> str:
        name = "_".join(parts) + suffix
        full = self.tmp_dir / name
        self.register(full)
        return str(full)

    def cleanup(self):
        for path in self.generated_paths:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
        self.generated_paths.clear()

    def render_geocode_maps(self,
                            flat_df: gpd.GeoDataFrame,
                            set_progress_callback) -> tuple[list, list, list]:
        pages: list = []
        intro_table = None
        groups = []
        if flat_df.empty or 'name_gis_geocodegroup' not in flat_df.columns:
            return pages, intro_table, groups

        groups = (flat_df['name_gis_geocodegroup']
                  .astype('string')
                  .dropna().unique().tolist())
        groups = sorted(groups)

        counts = (flat_df.groupby('name_gis_geocodegroup')
                  .size()
                  .rename('count')
                  .reset_index())
        intro_table = [["Geocode category", "Geocode objects"]]
        for _, row in counts.sort_values('name_gis_geocodegroup').iterrows():
            intro_table.append([str(row['name_gis_geocodegroup']), int(row['count'])])

        fixed_bounds_3857 = compute_fixed_bounds_3857(flat_df, base_dir=self.base_dir)

        done = 0
        for gname in groups:
            sub = flat_df[flat_df['name_gis_geocodegroup'] == gname].copy()

            safe = _safe_name(gname)
            sens_png = self.make_path("geocode", safe, "sens")
            env_png = self.make_path("geocode", safe, "env")

            ok_sens = draw_group_map_sensitivity(sub, gname, self.palette, self.desc, sens_png, fixed_bounds_3857, base_dir=self.base_dir)
            ok_env = draw_group_map_envindex(sub, gname, env_png, fixed_bounds_3857, base_dir=self.base_dir)

            if ok_sens and _file_ok(sens_png):
                pages += [
                    ('heading(2)', f"Geocode group: {gname}"),
                    ('text', "Sensitivity (A–E palette)."),
                    ('image', ("Sensitivity map", sens_png)),
                    ('new_page', None),
                ]
            if ok_env and _file_ok(env_png):
                pages += [
                    ('heading(2)', f"Geocode group: {gname}"),
                    ('text', "Environment index (0–100)."),
                    ('image', ("Environment index map", env_png)),
                    ('new_page', None),
                ]

            done += 1
            if set_progress_callback:
                set_progress_callback(done, max(1, len(groups)))

        return pages, intro_table, groups

    def render_atlas_maps(self,
                          flat_df: gpd.GeoDataFrame,
                          atlas_df: gpd.GeoDataFrame,
                          atlas_geocode_pref: str | None,
                          include_atlas_maps: bool,
                          set_progress_callback) -> tuple[list, str | None]:
        pages = []
        atlas_geocode_selected = None
        if not include_atlas_maps or atlas_df is None or atlas_df.empty:
            return pages, atlas_geocode_selected

        flat_geos = flat_df if flat_df is not None else gpd.GeoDataFrame()
        flat_polys_3857 = gpd.GeoDataFrame()
        if not flat_geos.empty and 'geometry' in flat_geos.columns:
            flat_polys = flat_geos[flat_geos.geometry.type.isin(['Polygon','MultiPolygon'])].copy()
            if 'name_gis_geocodegroup' in flat_polys.columns:
                flat_polys['name_gis_geocodegroup'] = (
                    flat_polys['name_gis_geocodegroup']
                    .astype('string')
                    .str.strip()
                )
            flat_polys_3857 = _safe_to_3857(flat_polys)

        atlas_crs = atlas_df.crs
        atlas_df = atlas_df[atlas_df['geometry'].notna()].copy()

        def _pick_geocode_level():
            nonlocal atlas_geocode_selected
            levels = []
            if not flat_df.empty and 'name_gis_geocodegroup' in flat_df.columns:
                levels = sorted(flat_df['name_gis_geocodegroup'].astype('string').dropna().unique().tolist())
            if atlas_geocode_pref and atlas_geocode_pref in levels:
                atlas_geocode_selected = atlas_geocode_pref
            elif 'basic_mosaic' in levels:
                atlas_geocode_selected = 'basic_mosaic'
            elif levels:
                atlas_geocode_selected = levels[0]

        _pick_geocode_level()

        polys_for_atlas = flat_polys_3857
        if atlas_geocode_selected and not flat_polys_3857.empty:
            if 'name_gis_geocodegroup' in flat_polys_3857.columns:
                atlas_mask = flat_polys_3857['name_gis_geocodegroup'].astype('string').str.lower() == atlas_geocode_selected.lower()
                filtered = flat_polys_3857[atlas_mask].copy()
                if filtered.empty:
                    write_to_log(f"No atlas polygons found for geocode '{atlas_geocode_selected}'. Using all polygons.", self.base_dir)
                else:
                    polys_for_atlas = filtered
            else:
                write_to_log("Atlas polygons missing 'name_gis_geocodegroup'; using all polygons.", self.base_dir)

        bounds = compute_fixed_bounds_3857(flat_df, base_dir=self.base_dir)
        overview_png = self.make_path("atlas", "overview")
        ok_overview = draw_atlas_overview_map(atlas_df, atlas_crs, polys_for_atlas, overview_png, bounds, base_dir=self.base_dir)
        if ok_overview and _file_ok(overview_png):
            text = "Overview of all atlas tiles within the study area."
            if atlas_geocode_selected:
                text += f" Geocode level shown: <b>{atlas_geocode_selected}</b>."
            pages += [
                ('heading(3)', "Atlas overview"),
                ('text', text),
                ('image', ("Atlas tiles overview", overview_png)),
                ('new_page', None),
            ]

        atlas_total = len(atlas_df)
        for idx, tile_row in atlas_df.iterrows():
            safe_tile = _safe_name(tile_row.get('name_gis') or f"atlas_{idx+1}")
            sens_png = self.make_path("atlas", safe_tile, "sens")
            env_png = self.make_path("atlas", safe_tile, "env")

            ok_sens = draw_atlas_map(tile_row, atlas_crs, polys_for_atlas, self.palette, self.desc, sens_png, bounds, base_dir=self.base_dir)
            ok_env = draw_atlas_map(tile_row, atlas_crs, polys_for_atlas, self.palette, self.desc, env_png, bounds, base_dir=self.base_dir, metric="env_index")

            has_entries = False
            title_raw = tile_row.get('title_user') or tile_row.get('name_gis') or safe_tile
            tile_id = tile_row.get('name_gis') or safe_tile
            heading = f"Atlas tile: {title_raw}" if str(title_raw) == str(tile_id) else f"Atlas tile: {title_raw} ({tile_id})"
            pages.append(('heading(3)', heading))
            info_parts = []
            if isinstance(tile_row.get('description'), str) and tile_row.get('description').strip():
                info_parts.append(tile_row.get('description').strip())
            if atlas_geocode_selected:
                info_parts.append(f"Geocode level: <b>{atlas_geocode_selected}</b>.")
            info_parts.append("Inset highlights tile within the study area.")
            pages.append(('text', " ".join(info_parts)))

            if ok_sens and _file_ok(sens_png):
                pages.append(('text', "Sensitivity (A–E palette)."))
                pages.append(('image', ("Sensitivity atlas map", sens_png)))
                has_entries = True
            if ok_env and _file_ok(env_png):
                pages.append(('text', "Environment index (0–100)."))
                pages.append(('image', ("Environment index atlas map", env_png)))
                has_entries = True

            if has_entries:
                pages.append(('new_page', None))
            else:
                pages.pop()  # remove heading
                pages.pop()  # remove text

            if set_progress_callback:
                set_progress_callback(idx+1, max(1, atlas_total))

        return pages, atlas_geocode_selected

    def render_segments(self,
                        lines_df: gpd.GeoDataFrame,
                        segments_df: gpd.GeoDataFrame,
                        palette: dict,
                        base_dir: str,
                        set_progress_callback):
        pages_lines = []
        log_data = []
        if (lines_df.empty or segments_df.empty or
            {'name_gis','segment_id','sensitivity_code_max','sensitivity_code_min'}.issubset(segments_df.columns) is False or
            'length_m' not in lines_df.columns or 'geometry' not in lines_df.columns):
            return pages_lines, log_data

        total = len(lines_df)
        for idx, line in lines_df.iterrows():
            ln_visible = line['name_gis']
            ln_safe = _safe_name(ln_visible)
            length_m = float(line.get('length_m', 0) or 0)
            length_km = length_m / 1000.0
            segment_records = sort_segments_numerically(segments_df[segments_df['name_gis'] == ln_visible]).copy()

            if not segment_records.empty:
                segment_records['__sens_code_max'] = segment_records.apply(
                    lambda row: _normalize_sensitivity_code(
                        row.get('sensitivity_code_max'),
                        row.get('sensitivity_max')
                    ),
                    axis=1
                )
                segment_records['__sens_code_min'] = segment_records.apply(
                    lambda row: _normalize_sensitivity_code(
                        row.get('sensitivity_code_min'),
                        row.get('sensitivity_min')
                    ),
                    axis=1
                )

            context_img = self.make_path("line", ln_safe, "context")
            seg_map_max = self.make_path("line", ln_safe, "segments_max")
            seg_map_min = self.make_path("line", ln_safe, "segments_min")

            ok_context = draw_line_context_map(line, context_img, pad_ratio=1.0, rect_buffer_ratio=0.03, base_dir=base_dir)
            ok_max = draw_line_segments_map(segments_df, ln_visible, palette, seg_map_max, mode='max', pad_ratio=0.20, base_dir=base_dir)
            ok_min = draw_line_segments_map(segments_df, ln_visible, palette, seg_map_min, mode='min', pad_ratio=0.20, base_dir=base_dir)

            max_stats_img = self.make_path("line", ln_safe, "max_dist")
            min_stats_img = self.make_path("line", ln_safe, "min_dist")
            max_codes = segment_records.get('__sens_code_max', segment_records.get('sensitivity_code_max'))
            min_codes = segment_records.get('__sens_code_min', segment_records.get('sensitivity_code_min'))
            if max_codes is None:
                max_codes = pd.Series(dtype=object)
            if min_codes is None:
                min_codes = pd.Series(dtype=object)
            create_sensitivity_summary(max_codes, palette, max_stats_img)
            create_sensitivity_summary(min_codes, palette, min_stats_img)

            max_img = self.make_path("line", ln_safe, "max_ribbon")
            min_img = self.make_path("line", ln_safe, "min_ribbon")
            create_line_statistic_image(ln_visible, max_codes, palette, length_m, max_img)
            create_line_statistic_image(ln_visible, min_codes, palette, length_m, min_img)

            first_page = [
                ('heading(2)', f"Line: {ln_visible}"),
                ('text', f"This section summarizes sensitivity along the line <b>{ln_visible}</b> "
                         f"(total length <b>{length_km:.2f} km</b>, segments: <b>{len(segment_records)}</b>). "
                         "Below: geographical context and segments maps colored by sensitivity values, "
                         "followed by the distribution and a ribbon (maximum sensitivity).")
            ]

            if ok_context and _file_ok(context_img):
                first_page.append(('image', ("Geographical context", context_img)))
            if ok_max and _file_ok(seg_map_max):
                first_page.append(('image', ("Segments colored by maximum sensitivity", seg_map_max)))
            first_page.append(('image', ("Maximum sensitivity – distribution", max_stats_img)))
            first_page.append(('text', f"Distance (km): 0 – {length_km/2:.1f} – {length_km:.1f}"))
            first_page.append(('image_ribbon', ("Maximum sensitivity – along line", max_img)))
            first_page.append(('new_page', None))

            second_page = [
                ('heading(2)', f"Line: {ln_visible} (continued)"),
                ('text', "Minimum sensitivity map, distribution, and ribbon.")
            ]
            if ok_min and _file_ok(seg_map_min):
                second_page.append(('image', ("Segments colored by minimum sensitivity", seg_map_min)))
            second_page.append(('image', ("Minimum sensitivity – distribution", min_stats_img)))
            second_page.append(('text', f"Distance (km): 0 – {length_km/2:.1f} – {length_km:.1f}"))
            second_page.append(('image_ribbon', ("Minimum sensitivity – along line", min_img)))
            second_page.append(('new_page', None))

            pages_lines += first_page + second_page

            for _, seg in segment_records.iterrows():
                log_data.append({
                    'line_name': ln_visible,
                    'segment_id': seg['segment_id'],
                    'sensitivity_code_max': seg['sensitivity_code_max'],
                    'sensitivity_code_min': seg['sensitivity_code_min']
                })

            if set_progress_callback:
                set_progress_callback(idx+1, total)

        return pages_lines, log_data

# Primary UI color (Steel Blue by default)
PRIMARY_HEX = "#4682B4"        # Steel blue
LIGHT_PRIMARY_HEX = "#6fa6cf"  # lighter steel-blue

# ---------------- Config + path helpers ----------------
def read_config(file_name: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    try:
        cfg.read(file_name, encoding="utf-8")
    except Exception:
        pass
    if "DEFAULT" not in cfg:
        cfg["DEFAULT"] = {}
    return cfg

def config_path(base_dir: str) -> str:
    """Flat config: <base>/config.ini"""
    return os.path.join(base_dir, "config.ini")

def parquet_dir_from_cfg(base_dir: str, cfg: configparser.ConfigParser) -> str:
    sub = cfg["DEFAULT"].get("parquet_folder", "output/geoparquet")
    required = ("tbl_asset_object.parquet", "tbl_asset_group.parquet", "tbl_flat.parquet")

    base_path = Path(base_dir)
    sub_path = Path(sub)

    candidates: list[Path] = []
    seen: set[str] = set()

    def _register(path: Path):
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            candidates.append(resolved)

    if sub_path.is_absolute():
        _register(sub_path)
    else:
        ancestors = [base_path]
        ancestors.extend(list(base_path.parents)[:4])
        for ancestor in ancestors:
            _register(ancestor / sub_path)
            _register(ancestor / "code" / sub_path)

    primary = candidates[0] if candidates else (base_path / sub_path)
    for cand in candidates:
        if all((cand / req).exists() for req in required):
            if cand != primary:
                write_to_log(f"Using parquet folder at {cand}", base_dir)
            return str(cand)

    try:
        primary.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    if all((primary / req).exists() for req in required):
        return str(primary)

    write_to_log(f"Expected parquet tables not found; using {primary}", base_dir)
    return str(primary)

# ---------------- Base dir normalization ----------------
def normalize_base_dir(path_str: str) -> str:
    """Return the logical app root for both .py and packaged .exe runs.
    - If started inside tools/ or system/ or code/, climb to parent.
    - Then climb up a few levels to find a folder containing 'output' & 'input'
      or containing 'tools' and 'config.ini'.
    """
    try:
        p = Path(path_str).resolve()
    except Exception:
        return path_str

    if p.name.lower() in ("tools", "system", "code"):
        p = p.parent

    q = p
    for _ in range(4):
        try:
            if (q / "output").exists() and (q / "input").exists():
                return str(q)
            if (q / "tools").exists() and (q / "config.ini").exists():
                return str(q)
        except Exception:
            pass
        q = q.parent
    return str(p)

# ---------------- Logging ----------------
def write_to_log(message: str, base_dir: str | None = None):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted = f"{timestamp} - {message}"
    try:
        if base_dir:
            with open(os.path.join(base_dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write(formatted + "\n")
    except Exception:
        pass
    try:
        if log_widget and log_widget.winfo_exists():
            log_widget.insert(tk.END, formatted + "\n")
            log_widget.see(tk.END)
    except Exception:
        pass

# ---------------- Small utilities ----------------
def _file_ok(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except Exception:
        return False

def _safe_name(name: str) -> str:
    # Windows-safe file name (keep visible name unchanged in titles)
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name))

def _cfg_getboolean(cfg: configparser.ConfigParser,
                    section: str,
                    option: str,
                    default: bool = False) -> bool:
    """
    Robust boolean reader for configparser with permissive parsing.
    Accepts truthy values like '1', 'true', 'yes', 'on'.
    """
    try:
        raw = cfg.get(section, option, fallback=None)
    except Exception:
        raw = None
    if raw is None:
        return default
    val = str(raw).strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return default

def _available_geocode_levels(base_dir: str,
                              config_file: str) -> list[str]:
    """
    Return sorted list of geocode group names available in tbl_flat.
    Used for GUI combo box population. Falls back to an empty list on failure.
    """
    try:
        cfg = read_config(config_file)
        gpq_dir = parquet_dir_from_cfg(base_dir, cfg)
        flat_df = load_tbl_flat(gpq_dir, base_dir=base_dir)
        if flat_df.empty or 'name_gis_geocodegroup' not in flat_df.columns:
            return []
        levels = (flat_df['name_gis_geocodegroup']
                  .astype('string')
                  .dropna()
                  .unique()
                  .tolist())
        return sorted(levels)
    except Exception as exc:
        write_to_log(f"Failed to enumerate geocode levels for UI: {exc}", base_dir)
        return []

def _dpi_for_fig_height(fig_height_in: float, px_cap: int = MAX_MAP_PX_HEIGHT, min_dpi: int = 110, max_dpi: int = 300) -> int:
    """
    Choose a DPI so that fig_height_in * dpi <= px_cap, bounded by min/max dpi.
    """
    try:
        dpi_by_cap = int(px_cap / max(fig_height_in, 1e-6))
        return max(min_dpi, min(max_dpi, dpi_by_cap))
    except Exception:
        return 150

# ---------------- Palette / descriptions ----------------
def read_sensitivity_palette_and_desc(file_name: str):
    """
    Reads color palette and descriptions for A-E from config.ini.
    Returns: (colors: dict{A..E->hex, 'UNKNOWN'->hex}, desc: dict{A..E->str})
    """
    global SENSITIVITY_UNKNOWN_COLOR, _SENSITIVITY_NUMERIC_RANGES
    cfg = configparser.ConfigParser()
    try:
        cfg.read(file_name, encoding="utf-8")
    except Exception:
        pass

    unknown_col = '#BDBDBD'
    if cfg.has_section('VALID_VALUES'):
        try:
            unknown_col = cfg['VALID_VALUES'].get('category_colour_unknown', unknown_col).strip() or unknown_col
        except Exception:
            pass
    SENSITIVITY_UNKNOWN_COLOR = unknown_col

    ranges: list[tuple[str, float, float]] = []

    def _parse_range(range_str: str):
        try:
            parts = re.split(r'\s*[-]\s*', range_str.strip())
            if len(parts) == 2:
                lo = float(parts[0])
                hi = float(parts[1])
                if lo > hi:
                    lo, hi = hi, lo
                return lo, hi
        except Exception:
            return None
        return None

    colors_map, desc_map = {}, {}
    for code in ['A','B','C','D','E']:
        if cfg.has_section(code):
            col = cfg[code].get('category_colour', '').strip() or unknown_col
            colors_map[code] = col
            desc_map[code] = cfg[code].get('description', '').strip()
            range_str = cfg[code].get('range', '').strip()
            parsed = _parse_range(range_str) if range_str else None
            if parsed:
                ranges.append((code, parsed[0], parsed[1]))
        else:
            colors_map[code] = unknown_col
            desc_map[code] = ''
    colors_map['UNKNOWN'] = unknown_col
    _SENSITIVITY_NUMERIC_RANGES = ranges
    return colors_map, desc_map

def _normalize_sensitivity_code(code_val, numeric_val):
    if code_val is not None and not pd.isna(code_val):
        try:
            code_str = str(code_val).strip().upper()
        except Exception:
            code_str = str(code_val).upper()
        if code_str in SENSITIVITY_ORDER:
            return code_str
    numeric_code = _sensitivity_code_from_numeric(numeric_val)
    if numeric_code in SENSITIVITY_ORDER:
        return numeric_code
    return 'UNKNOWN'

def _prepare_sensitivity_annotations(df: gpd.GeoDataFrame,
                                     code_column: str,
                                     numeric_column: str) -> gpd.GeoDataFrame:
    if df.empty:
        return df.copy()
    codes = df.apply(
        lambda row: _normalize_sensitivity_code(
            row.get(code_column),
            row.get(numeric_column)
        ),
        axis=1
    )
    out = df.copy()
    out['__sens_code'] = codes.fillna('UNKNOWN')
    return out


def _colors_from_annotations(gdf: gpd.GeoDataFrame,
                             palette: dict[str, str]) -> pd.Series:
    """
    Map normalized sensitivity codes stored in '__sens_code' to palette colors.
    Falls back to the configured UNKNOWN color when missing.
    """
    fallback = palette.get('UNKNOWN', SENSITIVITY_UNKNOWN_COLOR)

    def _lookup(code):
        if pd.isna(code):
            return fallback
        try:
            key = str(code).strip().upper()
        except Exception:
            key = str(code).upper()
        return palette.get(key, fallback)

    return gdf['__sens_code'].apply(_lookup)

def _sensitivity_code_from_numeric(value: float | int | None) -> str | None:
    try:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return None
        val = float(value)
    except Exception:
        return None
    for code, lo, hi in _SENSITIVITY_NUMERIC_RANGES:
        if lo <= val <= hi:
            return code
    return None

def _resolve_sensitivity_color(code_value,
                               numeric_value,
                               palette: dict[str, str]) -> str:
    if code_value is not None:
        try:
            code_str = str(code_value).strip().upper()
        except Exception:
            code_str = str(code_value).upper()
        if code_str:
            color = palette.get(code_str)
            if color:
                return color
    numeric_code = _sensitivity_code_from_numeric(numeric_value)
    if numeric_code:
        color = palette.get(numeric_code)
        if color:
            return color
    return palette.get('UNKNOWN', SENSITIVITY_UNKNOWN_COLOR)



# ---------------- Generic plotting utils ----------------
def _safe_to_3857(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    g = gdf.copy()
    try:
        if g.crs is None:
            g = g.set_crs(4326)
        try:
            epsg = g.crs.to_epsg()
        except Exception:
            epsg = None
        if epsg != 3857:
            g = g.to_crs(3857)
    except Exception:
        try:
            g = g.set_crs(3857, allow_override=True)
        except Exception:
            pass
    return g

def _tile_cache_root(base_dir: str | None) -> Path:
    """
    Resolve the cache directory used for XYZ tiles. Defaults to <base>/output/tile_cache.
    """
    try:
        base = Path(base_dir) if base_dir else Path.cwd()
    except Exception:
        base = Path.cwd()
    cache = base / "output" / "tile_cache"
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return cache

def _load_cached_tile(cache_path: Path):
    if not cache_path.exists():
        return None
    if TILE_CACHE_MAX_AGE_DAYS > 0:
        try:
            age = time.time() - cache_path.stat().st_mtime
            if age > TILE_CACHE_MAX_AGE_DAYS * 86400:
                try:
                    cache_path.unlink()
                except Exception:
                    pass
                return None
        except Exception:
            return None
    try:
        with PILImage.open(cache_path) as pil_img:
            return pil_img.convert("RGB")
    except Exception:
        try:
            cache_path.unlink()
        except Exception:
            pass
        return None

def _save_tile_to_cache(cache_path: Path, data: bytes) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            f.write(data)
    except Exception:
        pass

def _plot_basemap(ax, crs_epsg=3857, base_dir: str | None = None):
    """Add a basemap under current axes.
    Prefers contextily; if unavailable, fetches OSM Web Mercator tiles and composites them.
    """
    # Preferred path: contextily
    if ctx is not None:
        try:
            ctx.add_basemap(ax, crs=f"EPSG:{crs_epsg}", source=ctx.providers.OpenStreetMap.Mapnik)
            return
        except Exception as e:
            write_to_log(f"Basemap via contextily failed, falling back to tiles: {e}", base_dir)

    # Fallback: simple OSM tile fetch/composite in EPSG:3857 only
    if int(crs_epsg) != 3857:
        write_to_log("Basemap fallback only supports EPSG:3857; skipping.", base_dir)
        return

    try:
        # Current view in meters (WebMercator)
        minx, maxx = ax.get_xlim()
        miny, maxy = ax.get_ylim()

        def merc_to_lonlat(x, y):
            R = 6378137.0
            lon = (x / R) * 180.0 / math.pi
            lat = (2.0 * math.atan(math.exp(y / R)) - math.pi/2.0) * 180.0 / math.pi
            return lon, lat

        def lonlat_to_tile(lon, lat, z):
            n = 2 ** z
            x = int((lon + 180.0) / 360.0 * n)
            lat = max(-85.05112878, min(85.05112878, float(lat)))
            lat_rad = math.radians(lat)
            y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
            return x, y

        def tile_bounds_lonlat(z, x, y):
            n = 2.0 ** z
            minlon = x / n * 360.0 - 180.0
            maxlon = (x + 1) / n * 360.0 - 180.0
            def tiley_to_lat(t):
                Y = t / n
                lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * Y)))
                return math.degrees(lat_rad)
            maxlat = tiley_to_lat(y)
            minlat = tiley_to_lat(y + 1)
            return (minlon, minlat, maxlon, maxlat)

        def lonlat_to_merc(lon, lat):
            R = 6378137.0
            x = lon * math.pi / 180.0 * R
            lat = max(-85.05112878, min(85.05112878, float(lat)))
            y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) * R
            return x, y

        # Decide a zoom level based on axis pixel size to keep labels crisp.
        # Aim for mosaic pixel dims >= 1.5x axes pixels (oversample to reduce blur).
        bbox = ax.get_window_extent()
        try:
            ax_pix_w = int(bbox.width)
            ax_pix_h = int(bbox.height)
        except Exception:
            # Fallback: estimate from figure size
            ax_pix_w = int(ax.figure.dpi * ax.figure.get_size_inches()[0])
            ax_pix_h = int(ax.figure.dpi * ax.figure.get_size_inches()[1])

        lon0, lat1 = merc_to_lonlat(minx, maxy)
        lon1, lat0 = merc_to_lonlat(maxx, miny)

        target_w = max(512, int(ax_pix_w * 1.5))
        target_h = max(512, int(ax_pix_h * 1.5))

        z = 3
        for test in range(3, 20):
            x0, y0 = lonlat_to_tile(lon0, lat0, test)
            x1, y1 = lonlat_to_tile(lon1, lat1, test)
            w = (abs(x1 - x0) + 1) * 256
            h = (abs(y1 - y0) + 1) * 256
            if w >= target_w and h >= target_h:
                z = test
                break

        tile_limit = 400 if getattr(sys, "frozen", False) else 900
        def _tile_range(z_level):
            tx0, ty0 = lonlat_to_tile(lon0, lat0, z_level)
            tx1, ty1 = lonlat_to_tile(lon1, lat1, z_level)
            return (min(tx0, tx1), max(tx0, tx1), min(ty0, ty1), max(ty0, ty1))

        xmin, xmax, ymin, ymax = _tile_range(z)
        tile_span = (xmax - xmin + 1) * (ymax - ymin + 1)
        while tile_span > tile_limit and z > 3:
            z -= 1
            xmin, xmax, ymin, ymax = _tile_range(z)
            tile_span = (xmax - xmin + 1) * (ymax - ymin + 1)

        if tile_span > tile_limit:
            try:
                ax.set_facecolor("#f5f5f5")
            except Exception:
                pass
            write_to_log(f"Basemap fallback skipped; tile span too large ({tile_span} tiles @ z{z}).", base_dir)
            return

        if tile_span > tile_limit * 0.6:
            write_to_log(f"Basemap fallback using reduced zoom z{z} ({tile_span} tiles).", base_dir)

        TILE = 256
        W, H = (xmax - xmin + 1) * TILE, (ymax - ymin + 1) * TILE

        mosaic = PILImage.new("RGB", (W, H), (240, 240, 240))

        ua = ("Mozilla/5.0 (compatible; MESA-Report/1.0)")
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", ua)]
        urllib.request.install_opener(opener)

        def tile_url(z, x, y):
            return f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"

        cache_root = _tile_cache_root(base_dir)
        cache_hits = 0
        fetched = 0

        for xi, x in enumerate(range(xmin, xmax+1)):
            for yi, y in enumerate(range(ymin, ymax+1)):
                cache_path = cache_root / str(z) / str(x) / f"{y}.png"
                img = _load_cached_tile(cache_path)
                if img is not None:
                    cache_hits += 1
                else:
                    try:
                        url = tile_url(z, x, y)
                        with urllib.request.urlopen(url, timeout=5) as resp:
                            data = resp.read()
                        img = PILImage.open(io.BytesIO(data)).convert("RGB")
                        _save_tile_to_cache(cache_path, data)
                        fetched += 1
                    except Exception:
                        img = None
                if img is not None:
                    mosaic.paste(img, (xi*TILE, yi*TILE))

        if (cache_hits or fetched) and base_dir:
            try:
                write_to_log(f"Basemap tiles z{z}: {cache_hits} cached, {fetched} fetched ({tile_span} total).", base_dir)
            except Exception:
                pass

        # Extent of mosaic in Mercator meters (XYZ scheme)
        west  = tile_bounds_lonlat(z, xmin, ymin)[0]  # minlon of top-left tile
        north = tile_bounds_lonlat(z, xmin, ymin)[3]  # maxlat of top-left tile
        east  = tile_bounds_lonlat(z, xmax, ymax)[2]  # maxlon of bottom-right tile
        south = tile_bounds_lonlat(z, xmax, ymax)[1]  # minlat of bottom-right tile

        lx, north_y = lonlat_to_merc(west,  north)
        rx, south_y = lonlat_to_merc(east,  south)

        # Show with correct extent; origin='upper' so row 0 is at 'north'
        ax.imshow(mosaic, extent=[lx, rx, south_y, north_y], origin="upper",
                  interpolation='bilinear', resample=True)
    except Exception as e:
        write_to_log(f"Basemap fallback failed: {e}", base_dir)

def _expand_bounds(bounds, pad_ratio=0.08):
    minx, miny, maxx, maxy = bounds
    dx, dy = maxx - minx, maxy - miny
    if dx <= 0 or dy <= 0:
        return bounds
    pad_x, pad_y = dx * pad_ratio, dy * pad_ratio
    return (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)

# ---------------- tbl_flat (per-geocode maps) ----------------
def load_tbl_flat(parquet_dir: str, base_dir: str | None = None) -> gpd.GeoDataFrame:
    fp = os.path.join(parquet_dir, "tbl_flat.parquet")
    if not os.path.exists(fp):
        write_to_log("tbl_flat.parquet not found.", base_dir)
        return gpd.GeoDataFrame()
    try:
        gdf = gpd.read_parquet(fp)
        if 'geometry' not in gdf.columns or gdf.geometry.is_empty.all():
            write_to_log("tbl_flat has no valid geometry.", base_dir)
            return gpd.GeoDataFrame()
        return gdf
    except Exception as e:
        write_to_log(f"Failed reading tbl_flat: {e}", base_dir)
        return gpd.GeoDataFrame()

def compute_fixed_bounds_3857(flat_df: gpd.GeoDataFrame, base_dir: str | None = None):
    try:
        poly = flat_df[flat_df.geometry.type.isin(['Polygon','MultiPolygon'])].copy()
        if poly.empty:
            write_to_log("No polygons in tbl_flat to compute fixed bounds; using all geometry.", base_dir)
            poly = flat_df.copy()
        g3857 = _safe_to_3857(poly)
        if g3857.empty:
            return None
        b = g3857.total_bounds
        b = _expand_bounds(b, pad_ratio=0.08)
        return b
    except Exception as e:
        write_to_log(f"Failed computing fixed bounds: {e}", base_dir)
        return None

def _legend_for_sensitivity(ax, palette: dict, desc: dict, base_dir: str | None = None):
    try:
        y0 = 0.02
        dy = 0.06
        for i, code in enumerate(['A','B','C','D','E']):
            y = y0 + i*dy
            ax.add_patch(plt.Rectangle((0.02, y), 0.04, 0.035,
                                       transform=ax.transAxes,
                                       color=palette.get(code, '#BDBDBD'),
                                       ec='white', lw=0.8))
            label = f"{code}: {desc.get(code,'')}".strip() or code
            ax.text(0.07, y+0.005, label, transform=ax.transAxes,
                    fontsize=9, color='white',
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.35, ec="none"))
    except Exception as e:
        write_to_log(f"Sensitivity legend failed: {e}", base_dir)

def draw_group_map_sensitivity(gdf_group: gpd.GeoDataFrame,
                               group_name: str,
                               palette: dict,
                               desc: dict,
                               out_path: str,
                               fixed_bounds_3857=None,
                               base_dir: str | None = None):
    try:
        g = gdf_group.copy()
        g = g[g.geometry.type.isin(['Polygon','MultiPolygon'])]
        if g.empty:
            write_to_log(f"[{group_name}] No polygon geometries for sensitivity map.", base_dir)
            return False

        g = _safe_to_3857(g)
        fig_h_in = 10.0  # square to preserve detail
        fig_w_in = 10.0
        dpi = _dpi_for_fig_height(fig_h_in)
        fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
        ax.set_axis_off()

        if fixed_bounds_3857:
            minx, miny, maxx, maxy = fixed_bounds_3857
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)

        # Draw basemap first, then polygons on top for clarity
        _plot_basemap(ax, crs_epsg=3857, base_dir=base_dir)

        annotated = _prepare_sensitivity_annotations(g, 'sensitivity_code_max', 'sensitivity_max')
        if annotated.empty:
            write_to_log(f"[{group_name}] No sensitivity records to plot.", base_dir)
            plt.close(fig)
            return False

        colors = _colors_from_annotations(annotated, palette)
        annotated.plot(ax=ax,
                       color=colors,
                       edgecolor='white',
                       linewidth=0.3,
                       alpha=0.95,
                       zorder=10)

        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        write_to_log(f"Sensitivity map saved: {out_path}", base_dir)
        return True
    except Exception as e:
        write_to_log(f"Sensitivity map failed for {group_name}: {e}", base_dir)
        plt.close('all')
        return False

def draw_group_map_envindex(gdf_group: gpd.GeoDataFrame,
                            group_name: str,
                            out_path: str,
                            fixed_bounds_3857=None,
                            base_dir: str | None = None):
    try:
        if 'env_index' not in gdf_group.columns:
            write_to_log(f"[{group_name}] env_index missing; skipping env map.", base_dir)
            return False

        g = gdf_group.copy()
        g = g[g.geometry.type.isin(['Polygon','MultiPolygon'])]
        if g.empty:
            write_to_log(f"[{group_name}] No polygon geometries for env_index map.", base_dir)
            return False

        g['env_index'] = pd.to_numeric(g['env_index'], errors='coerce')
        g = g[np.isfinite(g['env_index'])]
        if g.empty:
            write_to_log(f"[{group_name}] env_index has no finite values.", base_dir)
            return False

        g = _safe_to_3857(g)
        fig_h_in = 10.0
        fig_w_in = 10.0
        dpi = _dpi_for_fig_height(fig_h_in)
        fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
        ax.set_axis_off()

        if fixed_bounds_3857:
            minx, miny, maxx, maxy = fixed_bounds_3857
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)

        norm = mcolors.Normalize(vmin=0, vmax=100)
        cmap = mpl_cmaps.get_cmap('YlOrRd')

        # Basemap first, colored polygons above
        _plot_basemap(ax, crs_epsg=3857, base_dir=base_dir)

        colors_arr = cmap(norm(g['env_index'].values))
        g.plot(ax=ax, color=colors_arr, edgecolor='white', linewidth=0.25, alpha=0.95, zorder=10)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label('env_index')

        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        write_to_log(f"Env index map saved: {out_path}", base_dir)
        return True
    except Exception as e:
        write_to_log(f"Env index map failed for {group_name}: {e}", base_dir)
        plt.close('all')
        return False

def draw_atlas_map(tile_row: pd.Series,
                   atlas_crs,
                   atlas_polys_3857: gpd.GeoDataFrame,
                   palette: dict,
                   desc: dict,
                   out_path: str,
                   global_bounds_3857=None,
                   base_dir: str | None = None,
                   metric: str = "sensitivity") -> bool:
    """
    Render a single atlas tile map (sensitivity or env_index) using shared cartography.
    """
    try:
        geom = tile_row.get('geometry', None)
        tile_name = tile_row.get('name_gis', '?')
        if geom is None or getattr(geom, "is_empty", True):
            write_to_log(f"[Atlas {tile_name}] Missing geometry; skipping map.", base_dir)
            return False

        tile_gdf = gpd.GeoDataFrame([{'geometry': geom}], geometry='geometry', crs=atlas_crs)
        if tile_gdf.crs is None:
            tile_gdf.set_crs(4326, inplace=True)
        tile_3857 = _safe_to_3857(tile_gdf)
        if tile_3857.empty or tile_3857.geometry.is_empty.all():
            write_to_log(f"[Atlas {tile_name}] Reprojection failed; skipping map.", base_dir)
            return False

        tile_bounds = _expand_bounds(tile_3857.total_bounds, pad_ratio=0.05)

        subset = gpd.GeoDataFrame()
        if atlas_polys_3857 is not None and not atlas_polys_3857.empty:
            try:
                mask = atlas_polys_3857.geometry.intersects(tile_3857.iloc[0].geometry)
                subset = atlas_polys_3857[mask].copy()
            except Exception as e:
                write_to_log(f"[Atlas {tile_name}] Intersection failed: {e}", base_dir)
                subset = gpd.GeoDataFrame()

        fig_h_in = 10.0
        fig_w_in = 10.0
        dpi = _dpi_for_fig_height(fig_h_in)
        fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
        ax.set_axis_off()
        ax.set_xlim(tile_bounds[0], tile_bounds[2])
        ax.set_ylim(tile_bounds[1], tile_bounds[3])

        _plot_basemap(ax, crs_epsg=3857, base_dir=base_dir)

        metric = (metric or "sensitivity").lower()
        drew_data = False
        if metric == "env_index":
            if subset.empty or "env_index" not in subset.columns:
                write_to_log(f"[Atlas {tile_name}] env_index missing for atlas map.", base_dir)
            else:
                subset = subset.copy()
                subset["env_index"] = pd.to_numeric(subset["env_index"], errors="coerce")
                subset = subset[np.isfinite(subset["env_index"])]
                if subset.empty:
                    write_to_log(f"[Atlas {tile_name}] env_index has no finite values.", base_dir)
                else:
                    norm = mcolors.Normalize(vmin=0, vmax=100)
                    cmap = mpl_cmaps.get_cmap("YlOrRd")
                    colors_arr = cmap(norm(subset["env_index"].values))
                    subset.plot(ax=ax, color=colors_arr, edgecolor="none", linewidth=0.0, alpha=0.95, zorder=10)
                    drew_data = True
        else:
            if not subset.empty and "sensitivity_code_max" in subset.columns:
                subset_ann = _prepare_sensitivity_annotations(subset, "sensitivity_code_max", "sensitivity_max")
                if subset_ann.empty:
                    write_to_log(f"[Atlas {tile_name}] Sensitivity annotations empty.", base_dir)
                else:
                    colors = _colors_from_annotations(subset_ann, palette)
                    subset_ann.plot(ax=ax,
                                    color=colors,
                                    edgecolor="none",
                                    linewidth=0.0,
                                    alpha=1.0,
                                    zorder=10)
                    drew_data = True
            elif not subset.empty:
                subset.plot(ax=ax,
                            color=SENSITIVITY_UNKNOWN_COLOR,
                            edgecolor="none",
                            linewidth=0.0,
                            alpha=1.0,
                            zorder=10)
                drew_data = True
        if not drew_data:
            plt.close(fig)
            write_to_log(f"[Atlas {tile_name}] No drawable data for metric '{metric}'.", base_dir)
            return False

        tile_3857.boundary.plot(ax=ax, edgecolor=PRIMARY_HEX, linewidth=1.6, zorder=12)

        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        write_to_log(f"Atlas map saved: {out_path}", base_dir)
        return True
    except Exception as e:
        write_to_log(f"Atlas map failed for {tile_row.get('name_gis', '?')}: {e}", base_dir)
        plt.close('all')
        return False

def draw_atlas_overview_map(atlas_df: gpd.GeoDataFrame,
                            atlas_crs,
                            atlas_polys_3857: gpd.GeoDataFrame,
                            out_path: str,
                            global_bounds_3857=None,
                            base_dir: str | None = None) -> bool:
    """
    Draw an overview map showing every atlas tile in relation to the full dataset extent.
    """
    try:
        if atlas_df.empty or 'geometry' not in atlas_df.columns:
            write_to_log("Atlas overview skipped (no geometries).", base_dir)
            return False

        atlas_gdf = atlas_df.copy()
        atlas_gdf = atlas_gdf[atlas_gdf['geometry'].notna()]
        if atlas_gdf.empty:
            write_to_log("Atlas overview skipped (all geometries empty).", base_dir)
            return False

        if atlas_crs is not None:
            try:
                atlas_gdf = atlas_gdf.set_crs(atlas_crs, allow_override=True)
            except Exception:
                pass
        atlas_3857 = _safe_to_3857(atlas_gdf)
        if atlas_3857.empty:
            write_to_log("Atlas overview reprojection failed; skipping.", base_dir)
            return False

        if global_bounds_3857:
            bounds = global_bounds_3857
        else:
            bounds = _expand_bounds(atlas_3857.total_bounds, pad_ratio=0.08)

        fig_h_in = 10.0
        fig_w_in = 10.0
        dpi = _dpi_for_fig_height(fig_h_in)
        fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
        ax.set_axis_off()
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])

        _plot_basemap(ax, crs_epsg=3857, base_dir=base_dir)

        if atlas_polys_3857 is not None and not atlas_polys_3857.empty:
            atlas_polys_3857.plot(ax=ax, facecolor="#d9d9d9", edgecolor='white',
                                 linewidth=0.15, alpha=0.5, zorder=8)

        atlas_3857.boundary.plot(ax=ax, edgecolor=PRIMARY_HEX, linewidth=1.0, alpha=0.9, zorder=12)

        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        write_to_log(f"Atlas overview map saved: {out_path}", base_dir)
        return True
    except Exception as e:
        write_to_log(f"Atlas overview map failed: {e}", base_dir)
        plt.close('all')
        return False

# ---------------- Lines: context + segments map ----------------
def draw_line_context_map(line_row: pd.Series, out_path: str,
                          pad_ratio: float = 1.0,
                          rect_buffer_ratio: float = 0.03,
                          base_dir: str | None = None):
    """
    Draw a geographical context map for the line.
    pad_ratio = 1.0 means a 100% buffer around the line's bounding box.
    """
    try:
        geom = line_row.get('geometry', None)
        if geom is None:
            write_to_log(f"[{line_row.get('name_gis','?')}] Missing geometry; cannot draw context map.", base_dir)
            return False

        g = gpd.GeoDataFrame([{'name_gis': line_row.get('name_gis','(line)'), 'geometry': geom}], crs=None)
        g = _safe_to_3857(g)
        if g.empty or g.geometry.is_empty.all():
            write_to_log(f"[{line_row.get('name_gis','?')}] Invalid geometry after reprojection.", base_dir)
            return False

        fig_h_in = 7.5
        fig_w_in = 10.0
        dpi = _dpi_for_fig_height(fig_h_in)
        fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
        ax.set_axis_off()

        b = g.total_bounds
        minx, miny, maxx, maxy = _expand_bounds(b, pad_ratio)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        _plot_basemap(ax, crs_epsg=3857, base_dir=base_dir)

        # Red rectangle around line bbox (slightly expanded)
        bx0, by0, bx1, by1 = b
        bw, bh = (bx1 - bx0), (by1 - by0)
        rx0 = bx0 - bw * rect_buffer_ratio
        ry0 = by0 - bh * rect_buffer_ratio
        rw  = bw + 2 * bw * rect_buffer_ratio
        rh  = bh + 2 * bh * rect_buffer_ratio
        rect = Rectangle((rx0, ry0), rw, rh, fill=False, edgecolor='red', linewidth=2.0, alpha=0.95, zorder=20)
        ax.add_patch(rect)

        # Halo + line on top
        try:
            g.plot(ax=ax, color='none', edgecolor='white', linewidth=5.0, alpha=0.9, zorder=21)
        except Exception:
            pass
        g.plot(ax=ax, color='none', edgecolor=PRIMARY_HEX, linewidth=2.2, alpha=1.0, zorder=22)

        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        write_to_log(f"Line context map saved: {out_path}", base_dir)
        return True
    except Exception as e:
        write_to_log(f"Context map failed for line: {e}", base_dir)
        plt.close('all')
        return False

def draw_line_segments_map(segments_df: gpd.GeoDataFrame,
                           line_name: str,
                           palette: dict,
                           out_path: str,
                           mode: str = 'max',
                           pad_ratio: float = 0.20,
                           base_dir: str | None = None):
    """
    Draw segments for a given line, colored by sensitivity (max or min), with basemap.
    """
    try:
        if segments_df.empty or 'geometry' not in segments_df.columns:
            write_to_log(f"[{line_name}] No geometry in segments; skipping segments map.", base_dir)
            return False

        col = 'sensitivity_code_max' if mode == 'max' else 'sensitivity_code_min'
        segs = segments_df[(segments_df['name_gis'] == line_name) & (segments_df['geometry'].notna())].copy()
        if segs.empty:
            write_to_log(f"[{line_name}] No segments found for segments map.", base_dir)
            return False

        segs = _safe_to_3857(segs)
        if segs.empty:
            write_to_log(f"[{line_name}] Segments reprojection failed.", base_dir)
            return False

        fig_h_in = 7.5
        fig_w_in = 10.0
        dpi = _dpi_for_fig_height(fig_h_in)
        fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
        ax.set_axis_off()

        b = segs.total_bounds
        minx, miny, maxx, maxy = _expand_bounds(b, pad_ratio)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        _plot_basemap(ax, crs_epsg=3857, base_dir=base_dir)

        polys = segs[segs.geometry.type.isin(['Polygon','MultiPolygon'])]
        lines = segs[segs.geometry.type.isin(['LineString','MultiLineString'])]
        value_col = 'sensitivity_max' if mode == 'max' else 'sensitivity_min'

        if not polys.empty:
            polys_ann = _prepare_sensitivity_annotations(polys, col, value_col)
            colors_polys = _colors_from_annotations(polys_ann, palette)
            polys_ann.plot(ax=ax,
                           color=colors_polys,
                           edgecolor='white',
                           linewidth=0.4,
                           alpha=0.95,
                           zorder=12)

        if not lines.empty:
            lines_ann = _prepare_sensitivity_annotations(lines, col, value_col)
            line_colors = _colors_from_annotations(lines_ann, palette)
            line_colors_rgba = [mcolors.to_rgba(c, alpha=1.0) for c in line_colors]
            try:
                lines.plot(ax=ax, color='white', linewidth=4.2, alpha=0.9, zorder=13)
            except Exception:
                pass
            lines.plot(ax=ax, color=line_colors_rgba, linewidth=2.4, alpha=1.0, zorder=14)

        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        write_to_log(f"Segments map ({mode}) saved: {out_path}", base_dir)
        return True
    except Exception as e:
        write_to_log(f"Segments map failed for {line_name}: {e}", base_dir)
        plt.close('all')
        return False

# ---------------- Existing (kept / refined) ----------------
def get_color_from_code(code, color_codes):
    return _resolve_sensitivity_color(code, None, color_codes)

def sort_segments_numerically(segments):
    def extract_number(segment_id):
        m = re.search(r'_(\d+)$', str(segment_id))
        return int(m.group(1)) if m else float('inf')
    s = segments.copy()
    s['sort_key'] = s['segment_id'].apply(extract_number)
    return s.sort_values(by='sort_key').drop(columns=['sort_key'])

def create_line_statistic_image(line_name, sensitivity_series, color_codes, length_m, output_path):
    """
    Generate ribbon PNG with NO axes/ticks/labels.
    Axes fill the full figure area; saved with bbox_inches=None and pad_inches=0
    so every PNG has an identical pixel canvas regardless of content.
    """
    segment_count = max(1, len(sensitivity_series))
    fig_h_in = 0.236  # ~6 mm
    dpi = 300
    fig_w_in = 12
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)

    # Fill figure with axes; no margins
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

    for i, code in enumerate(sensitivity_series):
        color = get_color_from_code(code, color_codes)
        ax.add_patch(plt.Rectangle((i/segment_count, 0), 1/segment_count, 1, color=color))

    # Save with constant canvas (no tight bbox, no padding)
    plt.savefig(output_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)

def resize_image(image_path, max_width_px, max_height_px):
    """
    Resample the stored image file to be within (max_width_px, max_height_px) in *pixels*.
    """
    try:
        with PILImage.open(image_path) as img:
            width, height = img.size
            ratio = min(float(max_width_px) / width, float(max_height_px) / height)
            if ratio >= 1.0:
                return  # already within limits
            new_w, new_h = max(1, int(width * ratio)), max(1, int(height * ratio))
            img = img.resize((new_w, new_h), PILImage.LANCZOS)
            img.save(image_path)
    except Exception as e:
        write_to_log(f"Image resize failed for {image_path}: {e}")

def create_sensitivity_summary(sensitivity_series, color_codes, output_path):
    counts = sensitivity_series.value_counts().reindex(['A','B','C','D','E']).fillna(0)
    total = max(1, len(sensitivity_series))
    fallback_color = color_codes.get('UNKNOWN', SENSITIVITY_UNKNOWN_COLOR)

    fig, (ax_text, ax_bar) = plt.subplots(2, 1, figsize=(10, 1.7), height_ratios=[0.45, 0.55])
    plt.subplots_adjust(hspace=0.15)

    parts = [f"{c}: {int(counts[c])} ({counts[c]/total*100:.1f}%)" for c in ['A','B','C','D','E']]
    summary = "Distribution: " + " | ".join(parts)
    ax_text.text(0.5, 0.5, summary, ha='center', va='center', fontsize=10)
    ax_text.axis('off')

    left = 0
    for c in ['A','B','C','D','E']:
        w = counts[c] / total
        ax_bar.barh(0, w, left=left, color=color_codes.get(c, fallback_color), edgecolor='white')
        left += w
    ax_bar.set_xlim(0, 1); ax_bar.set_ylim(-0.5, 0.5); ax_bar.axis('off')

    plt.savefig(output_path, bbox_inches='tight', dpi=110)
    plt.close(fig)

def debug_atlas_sample(base_dir: str,
                       cfg: configparser.ConfigParser,
                       palette: dict[str, str],
                       desc: dict[str, str],
                       tile_name: str,
                       geocode_level: str | None = None,
                       sample_size: int | None = None) -> str | None:
    """
    Render a quick atlas sensitivity map for a single tile using an optional
    subset of polygons. Helpful when debugging palette application.
    """
    gpq_dir = parquet_dir_from_cfg(base_dir, cfg)
    flat_df = load_tbl_flat(gpq_dir, base_dir=base_dir)
    if flat_df.empty:
        write_to_log("debug_atlas_sample: tbl_flat missing or empty.", base_dir)
        return None

    atlas_path = os.path.join(gpq_dir, "tbl_atlas.parquet")
    if not os.path.exists(atlas_path):
        write_to_log("debug_atlas_sample: tbl_atlas.parquet not found.", base_dir)
        return None

    try:
        atlas_df = gpd.read_parquet(atlas_path)
    except Exception as e:
        write_to_log(f"debug_atlas_sample: failed to read tbl_atlas ({e})", base_dir)
        return None

    if atlas_df.empty or 'geometry' not in atlas_df.columns:
        write_to_log("debug_atlas_sample: atlas table empty or missing geometry.", base_dir)
        return None

    tile_row = atlas_df[atlas_df['name_gis'] == tile_name]
    if tile_row.empty:
        write_to_log(f"debug_atlas_sample: tile '{tile_name}' not found in atlas.", base_dir)
        return None
    tile_row = tile_row.iloc[0]

    flat_polys = flat_df[flat_df.geometry.type.isin(['Polygon','MultiPolygon'])].copy()
    if flat_polys.empty:
        write_to_log("debug_atlas_sample: no polygon geometries in tbl_flat.", base_dir)
        return None

    if 'name_gis_geocodegroup' in flat_polys.columns:
        flat_polys['name_gis_geocodegroup'] = (
            flat_polys['name_gis_geocodegroup']
            .astype('string')
            .str.strip()
        )
    else:
        geocode_level = None  # cannot filter by level if column missing

    if geocode_level:
        mask_lvl = flat_polys['name_gis_geocodegroup'].astype('string').str.lower() == geocode_level.lower()
        filtered = flat_polys[mask_lvl].copy()
        if filtered.empty:
            write_to_log(f"debug_atlas_sample: geocode '{geocode_level}' produced no polygons; using full set.", base_dir)
        else:
            flat_polys = filtered

    flat_polys_3857 = _safe_to_3857(flat_polys)
    if flat_polys_3857.empty:
        write_to_log("debug_atlas_sample: reprojection yielded empty polygon set.", base_dir)
        return None

    tile_gdf = gpd.GeoDataFrame([tile_row], geometry='geometry', crs=atlas_df.crs)
    tile_3857 = _safe_to_3857(tile_gdf)
    if tile_3857.empty or tile_3857.geometry.is_empty.all():
        write_to_log(f"debug_atlas_sample: tile '{tile_name}' reprojection failed.", base_dir)
        return None

    try:
        intersects_mask = flat_polys_3857.geometry.intersects(tile_3857.iloc[0].geometry)
    except Exception as e:
        write_to_log(f"debug_atlas_sample: intersection failed ({e}).", base_dir)
        return None

    subset = flat_polys_3857[intersects_mask].copy()
    if subset.empty:
        write_to_log(f"debug_atlas_sample: no polygons intersect tile '{tile_name}'.", base_dir)
        return None

    if sample_size and sample_size > 0 and len(subset) > sample_size:
        subset = subset.sample(sample_size, random_state=0)
        write_to_log(f"debug_atlas_sample: down-sampled to {len(subset)} polygons.", base_dir)
    else:
        write_to_log(f"debug_atlas_sample: using {len(subset)} polygons.", base_dir)

    counts = _prepare_sensitivity_annotations(subset, 'sensitivity_code_max', 'sensitivity_max')['__sens_code'].value_counts(dropna=False)
    write_to_log(f"debug_atlas_sample sensitivity distribution: {counts.to_dict()}", base_dir)

    bounds = compute_fixed_bounds_3857(flat_df, base_dir=base_dir)
    out_path = os.path.join(base_dir, "output", "tmp", f"debug_atlas_{_safe_name(tile_name)}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ok = draw_atlas_map(tile_row, atlas_df.crs, subset, palette, desc, out_path, bounds, base_dir=base_dir)
    if ok:
        write_to_log(f"debug_atlas_sample: saved {out_path}", base_dir)
        return out_path
    return None

def fetch_asset_group_statistics(asset_group_df: gpd.GeoDataFrame, asset_object_df: gpd.GeoDataFrame):
    """
    Returns per sensitivity code:
      - Number of Asset Objects
      - Active asset groups (distinct groups having ≥1 object)
    """
    if asset_group_df.empty:
        return pd.DataFrame(columns=['Sensitivity Code','Sensitivity Description','Active asset groups','Number of Asset Objects'])

    grp = asset_group_df[['id','sensitivity_code','sensitivity_description']].copy()

    if not asset_object_df.empty and 'ref_asset_group' in asset_object_df.columns:
        cnt = asset_object_df.groupby('ref_asset_group').size().rename('asset_objects_nr')
        merged = grp.merge(cnt, left_on='id', right_index=True, how='left')
    else:
        merged = grp.copy()
        merged['asset_objects_nr'] = 0

    merged['asset_objects_nr'] = merged['asset_objects_nr'].fillna(0).astype(int)

    active_groups = (merged[merged['asset_objects_nr'] > 0]
                     .groupby(['sensitivity_code','sensitivity_description'])
                     .agg(active_groups=('id','nunique'))
                     .reset_index())

    objects_per_code = (merged.groupby(['sensitivity_code','sensitivity_description'])
                        .agg(asset_objects=('asset_objects_nr','sum'))
                        .reset_index())

    out = objects_per_code.merge(active_groups,
                                 on=['sensitivity_code','sensitivity_description'],
                                 how='left').fillna({'active_groups':0})

    out = out.rename(columns={
        'sensitivity_code':'Sensitivity Code',
        'sensitivity_description':'Sensitivity Description',
        'active_groups':'Active asset groups',
        'asset_objects':'Number of Asset Objects'
    }).sort_values('Sensitivity Code')

    return out

def format_area(row):
    if row['geometry_type'] in ['Polygon', 'MultiPolygon']:
        try:
            area = float(row['total_area'])
        except Exception:
            return "-"
        if area < 1_000_000:
            return f"{area:.0f} m²"
        return f"{area/1_000_000:.2f} km²"
    return "-"

def calculate_group_statistics(asset_object_df: gpd.GeoDataFrame, asset_group_df: gpd.GeoDataFrame):
    if asset_object_df.empty or asset_group_df.empty:
        return pd.DataFrame(columns=['Title','Code','Description','Type','Total area','# objects'])
    base = asset_group_df.drop(columns=[c for c in ['geometry'] if c in asset_group_df.columns])
    gdf = asset_object_df.merge(base, left_on='ref_asset_group', right_on='id', how='left')
    gdf['geometry_type'] = gdf.geometry.type
    try:
        gdf = gdf.to_crs('ESRI:54009')
    except Exception:
        pass
    polys = gdf[gdf.geometry.type.isin(['Polygon','MultiPolygon'])].copy()
    others = gdf[~gdf.geometry.type.isin(['Polygon','MultiPolygon'])].copy()
    polys.loc[:, 'area'] = polys['geometry'].area
    others.loc[:, 'area'] = np.nan
    gdf2 = pd.concat([polys, others], ignore_index=True)

    stats = gdf2.groupby(
        ['title_fromuser','sensitivity_code','sensitivity_description','geometry_type'],
        dropna=False
    ).agg(total_area=('area','sum'), object_count=('geometry','size')).reset_index()

    stats['total_area'] = stats['total_area'].fillna(0)
    stats['total_area'] = stats.apply(format_area, axis=1)
    stats = stats.rename(columns={
        'title_fromuser':'Title',
        'sensitivity_code':'Code',
        'sensitivity_description':'Description',
        'geometry_type':'Type',
        'object_count':'# objects'
    }).sort_values('Code')
    return stats

def export_to_excel(df, fp):
    if not fp.lower().endswith('.xlsx'):
        fp += '.xlsx'
    df.to_excel(fp, index=False)

def fetch_lines_and_segments(parquet_dir: str):
    lines_pq    = os.path.join(parquet_dir, "tbl_lines.parquet")
    segments_pq = os.path.join(parquet_dir, "tbl_segment_flat.parquet")
    try:
        lines_df = gpd.read_parquet(lines_pq) if os.path.exists(lines_pq) else gpd.GeoDataFrame()
    except Exception:
        lines_df = gpd.GeoDataFrame()
    try:
        segments_df = gpd.read_parquet(segments_pq) if os.path.exists(segments_pq) else gpd.GeoDataFrame()
    except Exception:
        segments_df = gpd.GeoDataFrame()
    return lines_df, segments_df

# ---------------- PDF helpers ----------------
def line_up_to_pdf(order_list):
    elements = []
    styles = getSampleStyleSheet()

    primary = HexColor(PRIMARY_HEX)
    light_primary = HexColor(LIGHT_PRIMARY_HEX)
    heading_styles = {
        'H1': ParagraphStyle(
            name='H1', fontSize=20, leading=24, spaceBefore=6, spaceAfter=8,
            alignment=TA_CENTER, textColor=colors.black
        ),
        'H2Bar': ParagraphStyle(
            name='H2Bar', fontSize=16, leading=20, spaceBefore=10, spaceAfter=6,
            textColor=colors.white, backColor=light_primary, leftIndent=0, rightIndent=0
        ),
        'H3': ParagraphStyle(
            name='H3', fontSize=13, leading=16, spaceBefore=4, spaceAfter=4,
            textColor=primary
        ),
        'Body': styles['Normal']
    }

    # A4 text area constraints (points)
    max_image_width_pts = 16 * RL_CM                 # usable frame width
    default_max_image_height_pts = 24 * RL_CM
    map_max_height_pts = MAX_MAP_CM_HEIGHT * RL_CM

    ribbon_height_pts = RIBBON_CM_HEIGHT * RL_CM
    ribbon_width_pts  = max_image_width_pts          # <-- force ribbons to EXACT frame width

    def _add_heading(level, text):
        if level == 1:
            elements.append(Paragraph(text, heading_styles['H1']))
            elements.append(HRFlowable(width="100%", thickness=0.8, color=primary))
        elif level == 2:
            elements.append(Paragraph(text, heading_styles['H2Bar']))
        else:
            elements.append(Paragraph(text, heading_styles['H3']))
        elements.append(Spacer(1, 6))

    def _is_map_image(path: str) -> bool:
        name = os.path.basename(path).lower()
        return any(tok in name for tok in ['map_', '_segments_', '_context'])

    for item in order_list:
        itype, ival = item

        if itype == 'text':
            if isinstance(ival, str) and os.path.isfile(ival):
                with open(ival, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                text = str(ival)
            text = text.replace("\n", "<br/>")
            elements.append(Paragraph(text, heading_styles['Body']))
            elements.append(Spacer(1, 8))

        elif itype == 'image':
            # ival: (heading_str, path)
            heading_str, path = ival
            if not _file_ok(path):
                elements.append(Paragraph(f"<i>(image missing: {os.path.basename(path)})</i>", heading_styles['Body']))
                elements.append(Spacer(1, 6))
                continue

            _add_heading(3, heading_str)

            # Cap stored pixel height for maps to keep file sizes reasonable.
            if _is_map_image(path):
                try:
                    resize_image(path, max_width_px=1_000_000, max_height_px=MAX_MAP_PX_HEIGHT)
                except Exception:
                    pass

            img = Image(path)
            img.hAlign = 'CENTER'

            # Dual constraint: width cap + height cap (maps only)
            max_h_pts = map_max_height_pts if _is_map_image(path) else default_max_image_height_pts
            w0, h0 = float(getattr(img, 'imageWidth', 0) or 0), float(getattr(img, 'imageHeight', 0) or 0)
            if w0 > 0 and h0 > 0:
                scale = min(max_image_width_pts / w0, max_h_pts / h0, 1.0)
                img.drawWidth = w0 * scale
                img.drawHeight = h0 * scale

            elements.append(img)
            elements.append(Spacer(1, 6))

        elif itype == 'image_ribbon':
            # ival: (heading_str, path)
            heading_str, path = ival
            if not _file_ok(path):
                elements.append(Paragraph(f"<i>(image missing: {os.path.basename(path)})</i>", heading_styles['Body']))
                elements.append(Spacer(1, 6))
                continue
            _add_heading(3, heading_str)

            img = Image(path)
            # FORCE exact same display size for all ribbons (no aspect scaling surprises)
            img.drawWidth  = ribbon_width_pts
            img.drawHeight = ribbon_height_pts
            img.hAlign = 'CENTER'

            elements.append(img)
            elements.append(Spacer(1, 6))

        elif itype == 'table':
            df = pd.read_excel(ival)
            data = [df.columns.tolist()] + df.values.tolist()
            table = Table(data, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), primary),
                ('TEXTCOLOR',  (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('ALIGN',      (0, 0), (-1,-1), 'CENTER'),
                ('GRID',       (0, 0), (-1,-1), 0.5, colors.HexColor("#A9A9A9")),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.whitesmoke, colors.Color(0.97,0.97,0.97)])
            ]))
            elements.append(table)
            elements.append(Spacer(1, 8))

        elif itype == 'table_data':
            title, data_ll = ival
            _add_heading(3, title)
            table = Table(data_ll, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), primary),
                ('TEXTCOLOR',  (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('ALIGN',      (0, 0), (-1,-1), 'CENTER'),
                ('GRID',       (0, 0), (-1,-1), 0.5, colors.HexColor("#A9A9A9")),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.whitesmoke, colors.Color(0.97,0.97,0.97)])
            ]))
            elements.append(table)
            elements.append(Spacer(1, 8))

        elif itype.startswith('heading'):
            level = int(itype[-2])
            _add_heading(level, ival)

        elif itype == 'rule':
            elements.append(HRFlowable(width="100%", thickness=0.8, color=primary))
            elements.append(Spacer(1, 6))

        elif itype == 'spacer':
            lines = ival if ival else 1
            elements.append(Spacer(1, lines * 10))

        elif itype == 'new_page':
            elements.append(PageBreak())

    return elements

def compile_pdf(output_pdf, elements):
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    primary = HexColor(PRIMARY_HEX)

    def add_header_footer(canvas, doc_obj):
        width, height = A4
        # Top steel-blue band
        canvas.saveState()
        canvas.setFillColor(primary)
        canvas.rect(0, height - 16, width, 16, fill=1, stroke=0)
        canvas.setFont('Helvetica-Bold', 10)
        canvas.setFillColor(colors.white)
        canvas.drawString(24, height - 12, "MESA – Report")
        canvas.restoreState()

        # Footer with page number
        canvas.saveState()
        canvas.setStrokeColor(primary)
        canvas.setLineWidth(0.5)
        canvas.line(24, 36, width - 24, 36)
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.gray)
        page_num = canvas.getPageNumber()
        canvas.drawRightString(width - 24, 24, f"Page {page_num}")
        canvas.restoreState()

    doc.build(elements, onFirstPage=add_header_footer, onLaterPages=add_header_footer)

def set_progress(pct: float, message: str | None = None):
    try:
        pct = max(0, min(100, int(pct)))
        if progress_var is not None:
            progress_var.set(pct)
        if progress_label is not None:
            progress_label.config(text=f"{pct}%")
        if message:
            # base_dir is not known here; GUI will show it anyway
            write_to_log(message)
    except Exception:
        pass

# ---------------- Core: generate report ----------------
def generate_report(base_dir: str,
                    config_file: str,
                    palette_A2E: dict,
                    desc_A2E: dict,
                    report_mode: str | None = None,
                    atlas_geocode_level: str | None = None):
    engine: ReportEngine | None = None
    try:
        set_progress(3, "Initializing report generation …")
        cfg       = read_config(config_file)
        include_atlas_cfg = _cfg_getboolean(cfg, 'DEFAULT', 'report_include_atlas_maps', default=False)
        if report_mode is None:
            include_atlas_maps = include_atlas_cfg
        else:
            include_atlas_maps = (str(report_mode).lower() == "detailed")
        atlas_geocode_pref = (atlas_geocode_level or
                              cfg['DEFAULT'].get('atlas_report_geocode_level', '') or '').strip()
        atlas_geocode_selected: str | None = None
        write_to_log(f"Report mode selected: {'Detailed (atlas included)' if include_atlas_maps else 'General maps only'}", base_dir)
        if include_atlas_maps:
            if atlas_geocode_pref:
                write_to_log(f"Atlas geocode preference: {atlas_geocode_pref}", base_dir)
        gpq_dir   = parquet_dir_from_cfg(base_dir, cfg)
        tmp_dir   = os.path.join(base_dir, 'output', 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        set_progress(6, "Paths prepared.")

        # Load key tables
        asset_object_pq = os.path.join(gpq_dir, "tbl_asset_object.parquet")
        asset_group_pq  = os.path.join(gpq_dir, "tbl_asset_group.parquet")
        flat_pq         = os.path.join(gpq_dir, "tbl_flat.parquet")

        try: asset_objects_df = gpd.read_parquet(asset_object_pq) if os.path.exists(asset_object_pq) else gpd.GeoDataFrame()
        except Exception: asset_objects_df = gpd.GeoDataFrame()
        try: asset_groups_df  = gpd.read_parquet(asset_group_pq)  if os.path.exists(asset_group_pq)  else gpd.GeoDataFrame()
        except Exception: asset_groups_df  = gpd.GeoDataFrame()

        set_progress(10, "Loaded asset tables.")

        # ---- Assets statistics ----
        write_to_log("Computing asset object statistics …", base_dir)
        group_stats_df = calculate_group_statistics(asset_objects_df, asset_groups_df)
        object_stats_xlsx = os.path.join(tmp_dir, 'asset_object_statistics.xlsx')
        export_to_excel(group_stats_df, object_stats_xlsx)

        ag_stats_df = fetch_asset_group_statistics(asset_groups_df, asset_objects_df)
        ag_stats_xlsx = os.path.join(tmp_dir, 'asset_group_statistics.xlsx')
        export_to_excel(ag_stats_df, ag_stats_xlsx)
        set_progress(22, "Asset stats ready.")

        flat_df = load_tbl_flat(gpq_dir, base_dir=base_dir)
                # ---- Lines & segments (grouped per line with context & segments maps) ----
        lines_df, segments_df = fetch_lines_and_segments(gpq_dir)
        engine = ReportEngine(base_dir, tmp_dir, palette_A2E, desc_A2E, config_file)

        def _progress_segments(done: int, total: int):
            set_progress(30 + int(10 * done / max(1, total)), f"Rendering line segment maps ({done}/{total})")

        pages_lines, log_data = engine.render_segments(lines_df, segments_df, palette_A2E, base_dir, _progress_segments)

        if log_data:
            log_df = pd.DataFrame(log_data)
            log_xlsx = os.path.join(tmp_dir, 'line_segment_log.xlsx')
            export_to_excel(log_df, log_xlsx)
            write_to_log(f"Segments log exported to {log_xlsx}", base_dir)
        else:
            write_to_log("Skipping line/segment pages (missing/empty).", base_dir)
        set_progress(40, "Lines/segments processed.")

        # ---- Atlas maps ----
        atlas_df = gpd.GeoDataFrame()
        atlas_geocode_selected = None
        if include_atlas_maps:
            atlas_pq = os.path.join(gpq_dir, "tbl_atlas.parquet")
            if os.path.exists(atlas_pq):
                try:
                    atlas_df = gpd.read_parquet(atlas_pq)
                except Exception as e:
                    write_to_log(f"Failed reading tbl_atlas: {e}", base_dir)
                    atlas_df = gpd.GeoDataFrame()
            else:
                write_to_log("Parquet table tbl_atlas not found; skipping atlas maps.", base_dir)

        atlas_geocode_pref = None
        if include_atlas_maps:
            atlas_geocode_pref = (atlas_geocode_level or cfg['DEFAULT'].get('atlas_report_geocode_level', '') or '').strip() or None
            if atlas_geocode_pref:
                write_to_log(f"Atlas geocode preference: {atlas_geocode_pref}", base_dir)

        def _progress_atlas(done: int, total: int):
            set_progress(45 + int(10 * done / max(1, total)), f"Rendering atlas tiles ({done}/{total})")

        atlas_pages, atlas_geocode_selected = engine.render_atlas_maps(flat_df, atlas_df, atlas_geocode_pref, include_atlas_maps, _progress_atlas)
        if atlas_pages:
            write_to_log("Per-atlas maps created.", base_dir)
        elif include_atlas_maps:
            write_to_log("Atlas maps requested but none were rendered.", base_dir)
        set_progress(55, "Atlas maps processed.")

        # ---- Per-geocode maps ----
        def _progress_geocode(done: int, total: int):
            set_progress(55 + int(15 * done / max(1, total)), f"Rendered maps for group {done}/{total}")

        geocode_pages, geocode_intro_table, geocode_groups = engine.render_geocode_maps(flat_df, _progress_geocode)
        if geocode_pages:
            write_to_log("Per-geocode maps created.", base_dir)
        else:
            write_to_log("tbl_flat missing or no 'name_gis_geocodegroup'; skipping per-geocode maps.", base_dir)
        set_progress(70, "Per-geocode maps completed.")
        # ---- Compose PDF ----
        timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")

        assets_area_note = (
            "Note on areas: areas are computed only for polygon geometries. "
            "Points and lines have no surface area, therefore their 'Total area' is shown as “–”. "
            "Line and point assets are still counted in '# objects' and can be visualized on maps."
        )

        contents_lines = [
            "- Assets overview & statistics",
            "- Per-geocode maps (Sensitivity & Environment Index, shared scale)"
        ]
        if atlas_pages:
            contents_lines.append("- Atlas tile maps (per atlas object with inset)")
        contents_lines.append("- Lines & segments (grouped by line; context & segments maps, distributions, ribbons)")
        contents_text = "<br/>".join(contents_lines)

        order_list = [
            ('heading(1)', "MESA report"),
            ('text', f"Timestamp: {timestamp}"),
            ('rule', None),

            ('heading(2)', "Contents"),
            ('text', contents_text),
            ('new_page', None),

            ('heading(2)', "Assets – overview"),
            ('text', "Asset objects by group (count and area, where applicable)."),
            ('text', assets_area_note),
            ('table', os.path.join(tmp_dir, 'asset_object_statistics.xlsx')),
            ('spacer', 1),
            ('text', "Asset groups – sensitivity distribution (count of objects) and number of active groups (distinct groups with ≥1 object)."),
            ('table', os.path.join(tmp_dir, 'asset_group_statistics.xlsx')),
            ('new_page', None),

            ('heading(2)', "Per-geocode maps"),
            ('text',
             "About geocode categories: <br/><br/>"
             "<b>basic_mosaic</b> is the baseline geocoding used for area statistics in MESA. "
             "It is a consistent partition intended for repeatable reporting and comparison across runs. "
             "In contrast, <b>H3</b> geocodes are hexagonal cells from Uber’s H3 hierarchical indexing system. "
             "H3 provides multi-resolution grids that are excellent for scalable analysis and visualization. "
             "In this report, we render each available geocode category independently. "
             "All maps below share the same cartographic scale/extent for easier visual comparison."),
        ]

        if geocode_intro_table is not None:
            order_list.append(('table_data', ("Available geocode categories and object counts", geocode_intro_table)))
            order_list.append(('rule', None))

        order_list.extend(geocode_pages)
        if atlas_pages:
            order_list.append(('heading(2)', "Atlas maps"))
            atlas_intro = ("Each atlas tile focuses on a subset of the study area. "
                           "An inset map indicates where the tile sits relative to the full extent.")
            if atlas_geocode_selected:
                atlas_intro += f" Geocode level shown: <b>{atlas_geocode_selected}</b>."
            order_list.append(('text', atlas_intro))
            order_list.append(('rule', None))
            order_list.extend(atlas_pages)

        if pages_lines:
            order_list.append(('heading(2)', "Lines and segments"))
            order_list.append(('text', "Images contain only cartography—no embedded titles. "
                                       "Ribbons are fixed to 0.6 cm high and the full text width. "
                                       "Distance markers are written as text before each ribbon."))
            order_list.append(('rule', None))
            order_list.append(('new_page', None))
            order_list.extend(pages_lines)

        set_progress(86, "Composing PDF …")
        elements = line_up_to_pdf(order_list)

        ts_pdf = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
        output_pdf = os.path.join(base_dir, f'output/MESA-report_{ts_pdf}.pdf')
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
        compile_pdf(output_pdf, elements)
        engine.cleanup()
        engine = None
        set_progress(100, "Report completed.")

        global last_report_path
        last_report_path = output_pdf
        write_to_log(f"PDF report created: {output_pdf}", base_dir)

        try:
            if link_var:
                link_var.set("Open report folder")
        except Exception:
            pass

    except Exception as e:
        write_to_log(f"ERROR during report generation: {e}", base_dir)
        set_progress(100, "Report failed.")
    finally:
        if engine is not None:
            try:
                engine.cleanup()
            except Exception:
                pass

# ---------------- GUI runner ----------------
def _start_report_thread(base_dir, config_file, palette, desc, report_mode, atlas_geocode):
    threading.Thread(
        target=generate_report,
        args=(base_dir, config_file, palette, desc, report_mode, atlas_geocode),
        daemon=True
    ).start()

def launch_gui(base_dir: str, config_file: str, palette: dict, desc: dict, theme: str):
    global log_widget, progress_var, progress_label, link_var, report_mode_var, atlas_geocode_var, _atlas_geocode_choices
    root = tb.Window(themename=theme)
    root.title("MESA – Report generator")
    try:
        ico = Path(base_dir) / "system_resources" / "mesa.ico"
        if ico.exists():
            root.iconbitmap(str(ico))
    except Exception:
        pass

    frame = tb.LabelFrame(root, text="Log", bootstyle="info")
    frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    log_widget = scrolledtext.ScrolledText(frame, height=14)
    log_widget.pack(fill=tk.BOTH, expand=True)

    pframe = tk.Frame(root); pframe.pack(pady=6)
    progress_var = tk.DoubleVar(value=0.0)
    pbar = tb.Progressbar(pframe, orient="horizontal", length=300, mode="determinate",
                          variable=progress_var, bootstyle="info")
    pbar.pack(side=tk.LEFT, padx=6)
    progress_label = tk.Label(pframe, text="0%", width=5, anchor="w")
    progress_label.pack(side=tk.LEFT)

    mode_frame = tb.LabelFrame(root, text="Report detail level", bootstyle="secondary")
    mode_frame.pack(padx=10, pady=(0, 10), fill=tk.X)
    report_mode_var = tk.StringVar(value="general")
    rb_general = tb.Radiobutton(mode_frame, text="General maps", variable=report_mode_var,
                                value="general", bootstyle="info-toolbutton")
    rb_general.grid(row=0, column=0, padx=6, pady=4, sticky="w")
    rb_detailed = tb.Radiobutton(mode_frame, text="Detailed maps / overviews", variable=report_mode_var,
                                 value="detailed", bootstyle="info-toolbutton")
    rb_detailed.grid(row=0, column=1, padx=6, pady=4, sticky="w")

    atlas_frame = tb.LabelFrame(root, text="Atlas geocode level", bootstyle="secondary")
    atlas_frame.pack(padx=10, pady=(0, 10), fill=tk.X)
    _atlas_geocode_choices = _available_geocode_levels(base_dir, config_file)
    default_level = None
    if _atlas_geocode_choices:
        default_level = 'basic_mosaic' if 'basic_mosaic' in _atlas_geocode_choices else _atlas_geocode_choices[0]
    atlas_geocode_var = tk.StringVar(value=default_level or '')
    atlas_combo = tb.Combobox(atlas_frame, textvariable=atlas_geocode_var,
                              values=tuple(_atlas_geocode_choices),
                              state="readonly" if _atlas_geocode_choices else "disabled",
                              width=30)
    atlas_combo.grid(row=0, column=0, padx=6, pady=4, sticky="w")
    if _atlas_geocode_choices:
        tk.Label(atlas_frame, text="Used for atlas sensitivity and environment maps.",
                 anchor="w").grid(row=0, column=1, padx=6, pady=4, sticky="w")
    else:
        tk.Label(atlas_frame, text="(No geocode levels detected yet)", anchor="w")\
          .grid(row=0, column=1, padx=6, pady=4, sticky="w")

    btn_frame = tk.Frame(root); btn_frame.pack(pady=4)
    tb.Button(btn_frame, text="Generate report", bootstyle=PRIMARY,
              command=lambda: _start_report_thread(base_dir, config_file, palette, desc,
                                                   report_mode_var.get(),
                                                   atlas_geocode_var.get())
              ).grid(row=0, column=0, padx=6)
    tb.Button(btn_frame, text="Exit", bootstyle=WARNING, command=root.destroy).grid(row=0, column=1, padx=6)

    # Live link to report folder
    def open_report_folder(event=None):
        if not last_report_path:
            return
        folder = os.path.dirname(last_report_path)
        if os.path.isdir(folder):
            try:
                os.startfile(folder)  # Windows
            except Exception:
                try:
                    subprocess.Popen(["explorer", folder])
                except Exception as ee:
                    write_to_log(f"Failed to open folder: {ee}", base_dir)
        else:
            write_to_log("Report folder not found.", base_dir)

    link_var = tk.StringVar(value="")
    link_label = tk.Label(root, textvariable=link_var, fg="#4ea3ff", cursor="hand2",
                          font=("Segoe UI", 10, "underline"))
    link_label.pack(pady=(2, 8))
    link_label.bind("<Button-1>", open_report_folder)

    write_to_log(f"Working directory: {base_dir}", base_dir)
    write_to_log("Ready. Press 'Generate report' to start.", base_dir)
    root.mainloop()

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Presentation report (GeoParquet per geocode, same-scale maps, line context + segments maps with buffers)')
    parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
    parser.add_argument('--no-gui', action='store_true', help='Run directly without GUI')
    parser.add_argument('--report-mode', choices=['general', 'detailed'],
                        help='Select report detail level (general maps or detailed with atlas overviews).')
    parser.add_argument('--atlas-geocode',
                        help='Override geocode level to use for atlas maps (e.g. basic_mosaic, H3_R7).')
    parser.add_argument('--debug-atlas-sample',
                        help='Render a standalone sensitivity atlas map for the specified tile (name_gis).')
    parser.add_argument('--debug-atlas-geocode',
                        help='Optional geocode level to filter polygons when using --debug-atlas-sample.')
    parser.add_argument('--debug-atlas-size', type=int,
                        help='Optional max polygon count for --debug-atlas-sample (down-sample if exceeded).')
    args = parser.parse_args()

    base_dir = args.original_working_directory
    if not base_dir:
        base_dir = os.getcwd()
    base_dir = normalize_base_dir(base_dir)

    cfg_path = config_path(base_dir)
    cfg = read_config(cfg_path)
    theme = cfg['DEFAULT'].get('ttk_bootstrap_theme', 'flatly')

    # Sensitivity palette + descriptions from config (A–E)
    palette_A2E, desc_A2E = read_sensitivity_palette_and_desc(cfg_path)

    if args.debug_atlas_sample:
        out = debug_atlas_sample(
            base_dir,
            cfg,
            palette_A2E,
            desc_A2E,
            args.debug_atlas_sample,
            geocode_level=args.debug_atlas_geocode,
            sample_size=args.debug_atlas_size
        )
        if out:
            print(f"Debug atlas sample saved to: {out}")
        else:
            print("Debug atlas sample failed. Check log for details.")
        sys.exit(0)

    # Optional override for brand color from config
    PRIMARY_HEX = cfg['DEFAULT'].get('ui_primary_color', PRIMARY_HEX).strip() or PRIMARY_HEX
    try:
        def _lighten(hexcol, amt=0.30):
            hexcol = hexcol.lstrip('#')
            r = int(hexcol[0:2],16); g = int(hexcol[2:4],16); b = int(hexcol[4:6],16)
            r = int(r + (255 - r)*amt); g = int(g + (255 - g)*amt); b = int(b + (255 - b)*amt)
            return f"#{r:02x}{g:02x}{b:02x}"
        LIGHT_PRIMARY_HEX = _lighten(PRIMARY_HEX, 0.30)
    except Exception:
        pass

    if args.no_gui:
        generate_report(base_dir, cfg_path, palette_A2E, desc_A2E,
                        report_mode=args.report_mode,
                        atlas_geocode_level=args.atlas_geocode)
    else:
        launch_gui(base_dir, cfg_path, palette_A2E, desc_A2E, theme)

