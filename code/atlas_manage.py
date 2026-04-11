# -*- coding: utf-8 -*-
"""Unified atlas manager: create/import tiles + edit metadata.

PySide6 UI (migrated from ttkbootstrap).
"""

from mesa_shared import find_base_dir
from mesa_constants import TABLE_ATLAS, TABLE_ASSET_GROUP, TABLE_FLAT

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

import pandas as pd
import geopandas as gpd
from shapely.geometry import box

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PIL import Image as PILImage

try:
    import contextily as ctx
except Exception:
    ctx = None

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QPlainTextEdit, QLineEdit,
    QRadioButton, QProgressBar, QFileDialog, QMessageBox,
    QSizePolicy, QButtonGroup,
)
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt, QTimer, Signal, QObject

from asset_manage import ASSET_STYLESHEET as _SHARED_STYLESHEET


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
    d.setdefault("atlas_parquet_file", TABLE_ATLAS)
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


# =====================================================================
# Thread-safe signal bridge
# =====================================================================
class _AtlasSignals(QObject):
    log_message = Signal(str)
    progress_update = Signal(float)
    task_finished = Signal()


# =====================================================================
# Main window
# =====================================================================
class AtlasManagerWindow(QMainWindow):

    def __init__(self, base_dir: Path):
        super().__init__()
        self.base_dir = base_dir

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

        # Plain instance attributes (replacing tk.StringVar)
        self._name_gis = ""
        self._title_user = ""
        self._description = ""
        self._image_name_1 = ""
        self._image_desc_1 = ""
        self._image_name_2 = ""
        self._image_desc_2 = ""

        self._tile_mode = "config"
        self._tile_lon = f"{self.atlas_lon_size_km:.2f}"
        self._tile_lat = f"{self.atlas_lat_size_km:.2f}"
        self._tile_overlap = f"{self.atlas_overlap_percent:.2f}"
        self._tile_count = "4"
        self._tile_tolerance = "5"
        self._tile_count_overlap = f"{self.atlas_overlap_percent:.2f}"

        self.map_fig = None
        self.map_ax = None
        self.map_canvas = None
        self.map_gdf_plot = gpd.GeoDataFrame()
        self.map_tile_limit = 400
        self.map_user_agent = "Mozilla/5.0 (compatible; AtlasManager/1.0)"

        # Thread-safe signals
        self._signals = _AtlasSignals()
        self._signals.log_message.connect(self._append_log_line)
        self._signals.progress_update.connect(self._set_progress)
        self._signals.task_finished.connect(self._refresh_edit_data)

        self._build_ui()
        self._refresh_edit_data()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _atlas_path(self, for_write: bool = False) -> Path:
        return _parquet_path(self.atlas_file_name, for_write=for_write)

    def _ts(self) -> str:
        return datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")

    # ------------------------------------------------------------------
    # Logging / progress (thread-safe via signals)
    # ------------------------------------------------------------------
    def _log_to_gui(self, message: str):
        line = f"{self._ts()} - {message}"
        self._signals.log_message.emit(line)
        try:
            with open(BASE_DIR / "log.txt", "a", encoding="utf-8", errors="replace") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _append_log_line(self, line: str):
        """Slot: appends text to the log widget (runs on UI thread)."""
        try:
            self.log_widget.appendPlainText(line)
        except Exception:
            pass

    def _update_progress(self, value: float):
        v = max(0.0, min(100.0, float(value)))
        self._signals.progress_update.emit(v)

    def _set_progress(self, value: float):
        """Slot: updates the progress bar (runs on UI thread)."""
        try:
            self.progress_bar.setValue(int(value))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # UI build
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.setWindowTitle("Atlas")
        self.resize(900, 640)
        self.setMinimumSize(700, 500)

        try:
            ico = self.base_dir / "system_resources" / "mesa.ico"
            if ico.exists():
                self.setWindowIcon(QIcon(str(ico)))
        except Exception:
            pass

        central = QWidget()
        central.setObjectName("CentralHost")
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 8, 10, 8)
        main_layout.setSpacing(6)

        # Tab bar row with Exit button
        tab_row = QHBoxLayout()
        tab_row.setContentsMargins(0, 0, 0, 0)
        tab_row.setSpacing(0)

        self.tabs = QTabWidget()
        tab_row.addWidget(self.tabs, stretch=1)

        exit_btn = QPushButton("Exit")
        exit_btn.setFixedSize(72, 28)
        exit_btn.setStyleSheet("""
            QPushButton {
                background: #eadfc8; border: 1px solid #b79f73;
                border-radius: 4px; color: #453621; font-size: 9pt;
                padding: 2px 8px;
            }
            QPushButton:hover { background: #e1d1ae; }
            QPushButton:pressed { background: #d4c094; }
        """)
        exit_btn.clicked.connect(self.close)
        tab_row.addWidget(exit_btn, alignment=Qt.AlignTop)

        main_layout.addLayout(tab_row, stretch=1)

        # --- Create / Import tab ---
        create_tab = QWidget()
        create_layout = QVBoxLayout(create_tab)
        create_layout.setContentsMargins(10, 10, 10, 10)
        create_layout.setSpacing(8)
        self._build_create_tab(create_layout)
        self.tabs.addTab(create_tab, "Create / Import")

        # --- Edit tab ---
        edit_tab = QWidget()
        edit_layout = QVBoxLayout(edit_tab)
        edit_layout.setContentsMargins(10, 10, 10, 10)
        edit_layout.setSpacing(8)
        self._build_edit_tab(edit_layout)
        self.tabs.addTab(edit_tab, "Edit metadata")

        # --- Bottom status ---
        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("color: #9a8a6e; font-size: 8pt;")
        main_layout.addWidget(self.summary_label)

    # ------------------------------------------------------------------
    # Create / Import tab
    # ------------------------------------------------------------------
    def _build_create_tab(self, layout):
        options_group = QGroupBox("Tile generation options")
        options_layout = QGridLayout(options_group)
        options_layout.setHorizontalSpacing(10)
        options_layout.setVerticalSpacing(6)

        # Radio buttons
        self._radio_config = QRadioButton("Use config.ini defaults")
        self._radio_custom = QRadioButton("Custom tile size (km)")
        self._radio_count = QRadioButton("Distribute fixed number of tiles")
        self._radio_config.setChecked(True)

        self._tile_mode_group = QButtonGroup(self)
        self._tile_mode_group.addButton(self._radio_config)
        self._tile_mode_group.addButton(self._radio_custom)
        self._tile_mode_group.addButton(self._radio_count)
        self._tile_mode_group.buttonClicked.connect(self._update_tile_mode_ui)

        options_layout.addWidget(self._radio_config, 0, 0, 1, 4)
        options_layout.addWidget(self._radio_custom, 1, 0, 1, 4)

        options_layout.addWidget(QLabel("Width (lon km):"), 2, 0)
        self.entry_lon = QLineEdit(self._tile_lon)
        self.entry_lon.setMaximumWidth(80)
        options_layout.addWidget(self.entry_lon, 2, 1)

        options_layout.addWidget(QLabel("Height (lat km):"), 2, 2)
        self.entry_lat = QLineEdit(self._tile_lat)
        self.entry_lat.setMaximumWidth(80)
        options_layout.addWidget(self.entry_lat, 2, 3)

        options_layout.addWidget(QLabel("Overlap % (custom):"), 3, 0)
        self.entry_overlap = QLineEdit(self._tile_overlap)
        self.entry_overlap.setMaximumWidth(80)
        options_layout.addWidget(self.entry_overlap, 3, 1)

        options_layout.addWidget(self._radio_count, 4, 0, 1, 4)

        options_layout.addWidget(QLabel("Tiles:"), 5, 0)
        self.entry_count = QLineEdit(self._tile_count)
        self.entry_count.setMaximumWidth(80)
        options_layout.addWidget(self.entry_count, 5, 1)

        options_layout.addWidget(QLabel("Padding %:"), 5, 2)
        self.entry_tol = QLineEdit(self._tile_tolerance)
        self.entry_tol.setMaximumWidth(80)
        options_layout.addWidget(self.entry_tol, 5, 3)

        options_layout.addWidget(QLabel("Overlap % (count):"), 6, 0)
        self.entry_count_overlap = QLineEdit(self._tile_count_overlap)
        self.entry_count_overlap.setMaximumWidth(80)
        options_layout.addWidget(self.entry_count_overlap, 6, 1)

        self.create_mode_custom_widgets = [self.entry_lon, self.entry_lat, self.entry_overlap]
        self.create_mode_count_widgets = [self.entry_count, self.entry_tol, self.entry_count_overlap]

        self._update_tile_mode_ui()
        layout.addWidget(options_group)

        # Log widget
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_widget = QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMinimumHeight(80)
        self.log_widget.setMaximumHeight(180)
        log_layout.addWidget(self.log_widget)
        layout.addWidget(log_group, stretch=1)

        # Progress
        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        progress_row.addWidget(self.progress_bar)
        layout.addLayout(progress_row)

        # Action buttons
        btn_row = QHBoxLayout()
        create_btn = QPushButton("Create")
        create_btn.setProperty("role", "primary")
        create_btn.clicked.connect(lambda: self._spawn(self._run_create))
        btn_row.addWidget(create_btn)

        import_btn = QPushButton("Import")
        import_btn.setProperty("role", "primary")
        import_btn.clicked.connect(lambda: self._spawn(self._run_import_action))
        btn_row.addWidget(import_btn)

        refresh_btn = QPushButton("Refresh editor")
        refresh_btn.clicked.connect(self._refresh_edit_data)
        btn_row.addWidget(refresh_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Edit tab
    # ------------------------------------------------------------------
    def _build_edit_tab(self, layout):
        self.edit_state_label = QLabel("")
        self.edit_state_label.setStyleSheet("color: #6a5533; font-size: 9pt;")
        layout.addWidget(self.edit_state_label)

        # Horizontal split: form (left) + map (right)
        content_row = QHBoxLayout()
        content_row.setSpacing(10)

        # Form
        form_group = QGroupBox("Atlas record details")
        form = QGridLayout(form_group)
        form.setColumnStretch(1, 1)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)

        form.addWidget(QLabel("GIS Name"), 0, 0, Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_name_gis = QLabel("")
        self.lbl_name_gis.setStyleSheet("font-weight: 600;")
        form.addWidget(self.lbl_name_gis, 0, 1)

        form.addWidget(QLabel("Title"), 1, 0, Qt.AlignRight | Qt.AlignVCenter)
        self.title_entry = QLineEdit()
        form.addWidget(self.title_entry, 1, 1)

        form.addWidget(QLabel("Image Name 1"), 2, 0, Qt.AlignRight | Qt.AlignVCenter)
        image1_row = QHBoxLayout()
        self.image1_entry = QLineEdit()
        image1_row.addWidget(self.image1_entry)
        browse1_btn = QPushButton("Browse")
        browse1_btn.clicked.connect(self._browse_image_1)
        image1_row.addWidget(browse1_btn)
        form.addLayout(image1_row, 2, 1)

        form.addWidget(QLabel("Image 1 description"), 3, 0, Qt.AlignRight | Qt.AlignVCenter)
        self.image1_desc_entry = QLineEdit()
        form.addWidget(self.image1_desc_entry, 3, 1)

        form.addWidget(QLabel("Image Name 2"), 4, 0, Qt.AlignRight | Qt.AlignVCenter)
        image2_row = QHBoxLayout()
        self.image2_entry = QLineEdit()
        image2_row.addWidget(self.image2_entry)
        browse2_btn = QPushButton("Browse")
        browse2_btn.clicked.connect(self._browse_image_2)
        image2_row.addWidget(browse2_btn)
        form.addLayout(image2_row, 4, 1)

        form.addWidget(QLabel("Image 2 description"), 5, 0, Qt.AlignRight | Qt.AlignVCenter)
        self.image2_desc_entry = QLineEdit()
        form.addWidget(self.image2_desc_entry, 5, 1)

        form.addWidget(QLabel("Description"), 6, 0, Qt.AlignRight | Qt.AlignVCenter)
        self.description_entry = QLineEdit()
        form.addWidget(self.description_entry, 6, 1)

        content_row.addWidget(form_group, stretch=3)

        # Map widget
        map_group = QGroupBox("Atlas preview")
        map_layout = QVBoxLayout(map_group)
        self._build_map_widget(map_layout)
        content_row.addWidget(map_group, stretch=2)

        layout.addLayout(content_row, stretch=1)

        # Navigation controls
        controls_row = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(lambda: self._navigate(-1))
        controls_row.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(lambda: self._navigate(1))
        controls_row.addWidget(self.next_btn)

        controls_row.addStretch()

        self.reload_btn = QPushButton("Reload")
        self.reload_btn.clicked.connect(self._refresh_edit_data)
        controls_row.addWidget(self.reload_btn)

        save_btn = QPushButton("Save")
        save_btn.setProperty("role", "success")
        save_btn.clicked.connect(self._save_current)
        controls_row.addWidget(save_btn)

        layout.addLayout(controls_row)

    # ------------------------------------------------------------------
    # Map widget (matplotlib + FigureCanvasQTAgg)
    # ------------------------------------------------------------------
    def _build_map_widget(self, layout):
        self.map_fig = Figure(figsize=(4.0, 4.0), dpi=100)
        self.map_ax = self.map_fig.add_subplot(111)
        self.map_ax.set_title("Atlas overview", fontsize=10)
        self.map_ax.set_xticks([])
        self.map_ax.set_yticks([])
        self.map_ax.set_aspect("equal", adjustable="box")
        self.map_canvas = FigureCanvasQTAgg(self.map_fig)
        self.map_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.map_canvas)

    # ------------------------------------------------------------------
    # Tile mode UI
    # ------------------------------------------------------------------
    def _spawn(self, fn):
        threading.Thread(target=fn, daemon=True).start()

    def _update_tile_mode_ui(self, *_args):
        if self._radio_custom.isChecked():
            self._tile_mode = "custom"
        elif self._radio_count.isChecked():
            self._tile_mode = "count"
        else:
            self._tile_mode = "config"

        custom_enabled = self._tile_mode == "custom"
        count_enabled = self._tile_mode == "count"
        for widget in self.create_mode_custom_widgets:
            widget.setEnabled(custom_enabled)
        for widget in self.create_mode_count_widgets:
            widget.setEnabled(count_enabled)

    # ------------------------------------------------------------------
    # Edit helpers
    # ------------------------------------------------------------------
    def _set_edit_enabled(self, enabled: bool):
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
                w.setEnabled(enabled)
            except Exception:
                pass

    def _browse_image_1(self):
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Select file",
            str(BASE_DIR / "input" / "images"),
            "JPEG files (*.jpg *.jpeg);;PNG files (*.png);;All files (*.*)",
        )
        if fp:
            self.image1_entry.setText(fp)

    def _browse_image_2(self):
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Select file",
            str(BASE_DIR / "input" / "images"),
            "JPEG files (*.jpg *.jpeg);;PNG files (*.png);;All files (*.*)",
        )
        if fp:
            self.image2_entry.setText(fp)

    # ------------------------------------------------------------------
    # Map refresh
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Edit data refresh / load / save / navigate
    # ------------------------------------------------------------------
    def _refresh_edit_data(self):
        path = self._atlas_path(for_write=False)
        if not path.exists():
            self.df = pd.DataFrame()
            self.gdf = gpd.GeoDataFrame()
            self.current_index = 0
            self._clear_form()
            self._set_edit_enabled(False)
            self.edit_state_label.setText("No atlas data. Create or import atlas first.")
            self.summary_label.setText(f"Atlas file: {path} | pages: 0")
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
            self.edit_state_label.setText(f"Loaded {len(self.df)} atlas page(s).")
            self.summary_label.setText(f"Atlas file: {path} | pages: {len(self.df)}")
            self._refresh_map()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Could not read atlas data:\n{exc}")
            self.df = pd.DataFrame()
            self.gdf = gpd.GeoDataFrame()
            self._clear_form()
            self._set_edit_enabled(False)
            self.edit_state_label.setText("Failed to load atlas data.")
            self.summary_label.setText(f"Atlas file: {path} | pages: 0")
            self._refresh_map()

    def _clear_form(self):
        self.lbl_name_gis.setText("")
        self.title_entry.setText("")
        self.description_entry.setText("")
        self.image1_entry.setText("")
        self.image1_desc_entry.setText("")
        self.image2_entry.setText("")
        self.image2_desc_entry.setText("")

    def _load_record(self):
        if self.df.empty:
            self._clear_form()
            self._refresh_map()
            return
        row = self.df.iloc[self.current_index]
        self.lbl_name_gis.setText("" if pd.isna(row.get("name_gis")) else str(row.get("name_gis")))
        self.title_entry.setText("" if pd.isna(row.get("title_user")) else str(row.get("title_user")))
        self.description_entry.setText("" if pd.isna(row.get("description")) else str(row.get("description")))
        self.image1_entry.setText("" if pd.isna(row.get("image_name_1")) else str(row.get("image_name_1")))
        self.image1_desc_entry.setText("" if pd.isna(row.get("image_desc_1")) else str(row.get("image_desc_1")))
        self.image2_entry.setText("" if pd.isna(row.get("image_name_2")) else str(row.get("image_name_2")))
        self.image2_desc_entry.setText("" if pd.isna(row.get("image_desc_2")) else str(row.get("image_desc_2")))
        self.edit_state_label.setText(f"Record {self.current_index + 1} of {len(self.df)}")
        self._refresh_map()

    def _save_current(self):
        if self.df.empty:
            return
        key = self.lbl_name_gis.text().strip()
        if not key:
            QMessageBox.critical(self, "Error", "Missing name_gis")
            return

        self.df.at[self.current_index, "title_user"] = self.title_entry.text()
        self.df.at[self.current_index, "description"] = self.description_entry.text()
        self.df.at[self.current_index, "image_name_1"] = self.image1_entry.text()
        self.df.at[self.current_index, "image_desc_1"] = self.image1_desc_entry.text()
        self.df.at[self.current_index, "image_name_2"] = self.image2_entry.text()
        self.df.at[self.current_index, "image_desc_2"] = self.image2_desc_entry.text()

        path = self._atlas_path(for_write=False)
        if not path.exists():
            QMessageBox.critical(self, "Error", f"File not found:\n{path}")
            return

        try:
            gdf_local = gpd.read_parquet(path)
            if "name_gis" not in gdf_local.columns:
                QMessageBox.critical(self, "Error", "GeoParquet does not contain name_gis.")
                return

            idx = gdf_local.index[gdf_local["name_gis"].astype(str) == key]
            if len(idx) == 0:
                QMessageBox.critical(self, "Error", f"No record found for name_gis='{key}'.")
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

            self.edit_state_label.setText("Saved.")
            self.summary_label.setText(f"Atlas file: {write_path} | pages: {len(self.df)}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Save failed:\n{exc}")

    def _navigate(self, direction: int):
        if self.df.empty:
            return
        nxt = self.current_index + int(direction)
        nxt = max(0, min(len(self.df) - 1, nxt))
        if nxt == self.current_index:
            return
        self.current_index = nxt
        self._load_record()

    # ------------------------------------------------------------------
    # Business logic (parquet reading helpers)
    # ------------------------------------------------------------------
    def _read_asset_group_parquet(self) -> gpd.GeoDataFrame:
        p = _parquet_path(TABLE_ASSET_GROUP)
        if not p.exists():
            self._log_to_gui(f"Missing {TABLE_ASSET_GROUP} at {p}")
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
        p = _parquet_path(TABLE_FLAT)
        if not p.exists():
            self._log_to_gui(f"Missing {TABLE_FLAT} at {p}")
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

    # ------------------------------------------------------------------
    # Atlas bounds / geometry generation
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Create action (runs in background thread)
    # ------------------------------------------------------------------
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

        mode = self._tile_mode
        atlas_geometries = []

        if mode == "custom":
            try:
                lon_km = float(self.entry_lon.text())
                lat_km = float(self.entry_lat.text())
                overlap = float(self.entry_overlap.text())
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
                requested_tiles = max(2, int(self.entry_count.text()))
                tolerance = max(0.0, float(self.entry_tol.text()))
                overlap_pct = max(0.0, min(float(self.entry_count_overlap.text()), 90.0))
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
        self._signals.task_finished.emit()

    # ------------------------------------------------------------------
    # Import action (runs in background thread)
    # ------------------------------------------------------------------
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
        self._log_to_gui("Starting atlas import (GeoParquet)...")
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
        self._signals.task_finished.emit()


# =====================================================================
# Entry points
# =====================================================================
def run(base_dir: str, master=None):
    """In-process entry point called by mesa.py via lazy import.

    Note: with PySide6, in-process embedding from a Tk master is no longer
    supported. This launches a standalone Qt window instead.
    """
    global BASE_DIR, _CFG, _CFG_PATH, _PARQUET_OVERRIDE
    BASE_DIR = find_base_dir(base_dir)
    _CFG = None
    _CFG_PATH = None
    _PARQUET_OVERRIDE = None
    _ensure_cfg()

    # If QApplication already exists (e.g. launched from mesa.py),
    # just create the window. Otherwise create a new app.
    app = QApplication.instance()
    own_app = False
    if app is None:
        app = QApplication([])
        app.setStyleSheet(_SHARED_STYLESHEET)
        own_app = True
    window = AtlasManagerWindow(BASE_DIR)
    window.show()
    if own_app:
        app.exec()
    return window


def main():
    global BASE_DIR, _CFG, _CFG_PATH, _PARQUET_OVERRIDE

    parser = argparse.ArgumentParser(description="Unified atlas create/import + edit metadata")
    parser.add_argument("--original_working_directory", required=False, help="Path to running folder")
    args = parser.parse_args()

    BASE_DIR = find_base_dir(args.original_working_directory)
    _CFG = None
    _CFG_PATH = None
    _PARQUET_OVERRIDE = None
    _ensure_cfg()

    app = QApplication([])
    app.setStyleSheet(_SHARED_STYLESHEET)

    try:
        ico = BASE_DIR / "system_resources" / "mesa.ico"
        if ico.exists():
            app.setWindowIcon(QIcon(str(ico)))
    except Exception:
        pass

    window = AtlasManagerWindow(BASE_DIR)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
