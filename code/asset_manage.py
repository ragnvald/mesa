# -*- coding: utf-8 -*-
"""Unified asset manager: import assets + edit asset-group metadata.

PySide6 UI (migrated from ttkbootstrap).
"""

from mesa_shared import find_base_dir

import argparse
import configparser
import datetime
import os
import tempfile
import threading
import time
import warnings
from pathlib import Path

import fiona
import geopandas as gpd
import pandas as pd
from shapely import wkb as _shp_wkb
from shapely.geometry import box

try:
    from shapely import force_2d as _shp_force_2d
except Exception:
    _shp_force_2d = None

try:
    from shapely.validation import make_valid as shapely_make_valid
except Exception:
    shapely_make_valid = None

warnings.filterwarnings(
    "ignore",
    message=r"Measured \(M\) geometry types are not supported\..*",
    category=UserWarning,
    module=r"pyogrio\..*",
)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QPlainTextEdit, QLineEdit,
    QCheckBox, QProgressBar, QFrame, QSizePolicy,
    QMessageBox,
)
from PySide6.QtGui import QIcon, QFont
from PySide6.QtCore import Qt, QTimer, Signal, QObject

# =====================================================================
# Shared stylesheet (same warm palette as mesa.py)
# =====================================================================
ASSET_STYLESHEET = """
QMainWindow {
    background-color: #f3ecdf;
}
QWidget#CentralHost {
    background-color: #f3ecdf;
}
QWidget {
    background-color: transparent;
    color: #3f3528;
    font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif;
    font-size: 10pt;
}
QTabWidget::pane {
    border: 1px solid #cbb791;
    border-top: none;
    background: #f3ecdf;
}
QTabBar {
    background: #e6dac2;
}
QTabBar::tab {
    background: #e6dac2;
    color: #5c4a2f;
    padding: 8px 18px;
    margin-right: 2px;
    border: 1px solid #c6b089;
    border-bottom: none;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    font-weight: 500;
    font-size: 10pt;
}
QTabBar::tab:selected {
    background: #f8f3e9;
    color: #3f3528;
    font-weight: 600;
}
QTabBar::tab:!selected {
    margin-top: 2px;
}
QTabBar::tab:hover:!selected {
    background: #efe3cc;
}
QGroupBox {
    font-weight: 600;
    font-size: 10pt;
    background: #faf6ee;
    border: 1px solid #d5c3a4;
    border-radius: 8px;
    margin-top: 12px;
    padding: 18px 16px 14px 16px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 3px 10px;
    color: #715a36;
}
QPushButton {
    background-color: #f2e7d3;
    color: #4d4029;
    border: 1px solid #c7b18c;
    border-radius: 7px;
    padding: 6px 16px;
    font-weight: 500;
    min-width: 80px;
}
QPushButton:hover {
    background-color: #eadbbd;
    border-color: #b99763;
}
QPushButton:pressed {
    background-color: #ddc89f;
}
QPushButton:disabled {
    background-color: #eee5d7;
    color: #a28f71;
    border-color: #d4c6af;
}
QPushButton[role="primary"] {
    background-color: #d9bd7d;
    color: #3f3018;
    border: 1px solid #9b7c3d;
    font-weight: 600;
}
QPushButton[role="primary"]:hover {
    background-color: #e1c78d;
    border-color: #8c6d31;
}
QPushButton[role="primary"]:pressed {
    background-color: #cfb06f;
}
QPushButton[role="success"] {
    background-color: #e6ecd8;
    color: #34482a;
    border: 1px solid #9cad83;
}
QPushButton[role="success"]:hover {
    background-color: #edf2e3;
    border-color: #899b72;
}
QPushButton[role="success"]:pressed {
    background-color: #d9e3c7;
}
QLabel {
    background: transparent;
    color: #3f3528;
}
QLineEdit {
    background: #fffdf8;
    border: 1px solid #d9cab1;
    border-radius: 6px;
    padding: 5px 8px;
    font-size: 10pt;
    selection-background-color: #d7bb7f;
    selection-color: #2f2517;
}
QLineEdit:focus {
    border-color: #b99763;
}
QPlainTextEdit {
    background: #fffdf8;
    border: 1px solid #d9cab1;
    border-radius: 6px;
    padding: 6px;
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 9pt;
    selection-background-color: #d7bb7f;
    selection-color: #2f2517;
}
QPlainTextEdit:focus {
    border-color: #b99763;
}
QCheckBox {
    spacing: 6px;
    font-size: 10pt;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border-radius: 3px;
}
QCheckBox::indicator:unchecked {
    background: #dfc792;
    border: 1px solid #684d24;
}
QCheckBox::indicator:checked {
    background: #9a7230;
    border: 1px solid #513912;
}
QProgressBar {
    background: #e7dbc4;
    border: 1px solid #d3c29f;
    border-radius: 5px;
    height: 14px;
    text-align: center;
    color: #4f4129;
    font-size: 8pt;
}
QProgressBar::chunk {
    background: #b79b67;
    border-radius: 4px;
}
QScrollBar:vertical {
    background: transparent;
    width: 8px;
}
QScrollBar::handle:vertical {
    background: #d0bc97;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #b99763;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QToolTip {
    background: #3f3528;
    color: #faf6ee;
    border: none;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 9pt;
}
"""

BASE_DIR: Path = Path(".").resolve()
_CFG: configparser.ConfigParser | None = None
_CFG_PATH: Path | None = None
_PARQUET_SUBDIR = "output/geoparquet"
_PARQUET_OVERRIDE: Path | None = None

PURPOSE_COLUMN = "purpose_description"
STYLING_COLUMN = "styling"
REQUIRED_COLUMNS = [
    "id",
    "name_gis_assetgroup",
    "name_original",
    "title_fromuser",
    PURPOSE_COLUMN,
    STYLING_COLUMN,
    "importance",
    "susceptibility",
    "sensitivity",
    "sensitivity_code",
    "sensitivity_description",
    "total_asset_objects",
]

ASSET_OBJECT_COLUMNS = [
    "id",
    "asset_group_name",
    "attributes",
    "process",
    "ref_asset_group",
    "geometry",
]

ASSET_GROUP_COLUMNS = [
    "id",
    "name_original",
    "name_gis_assetgroup",
    "title_fromuser",
    "date_import",
    "geometry",
    "total_asset_objects",
    "importance",
    "susceptibility",
    "sensitivity",
    "sensitivity_code",
    "sensitivity_description",
    PURPOSE_COLUMN,
    STYLING_COLUMN,
]


def _force_2d_geom(geom):
    if geom is None:
        return None
    try:
        if getattr(geom, "is_empty", False):
            return geom
    except Exception:
        return geom
    if _shp_force_2d is not None:
        try:
            return _shp_force_2d(geom)
        except Exception:
            pass
    try:
        return _shp_wkb.loads(_shp_wkb.dumps(geom, output_dimension=2))
    except Exception:
        return geom


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
    d.setdefault("input_folder_asset", "input/asset")
    d.setdefault("asset_group_parquet_file", "tbl_asset_group.parquet")
    d.setdefault("import_validate_geometries", "false")
    d.setdefault("import_simplify_geometries", "false")
    d.setdefault("import_simplify_tolerance_m", "1.0")
    d.setdefault("import_simplify_preserve_topology", "true")

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
        _PARQUET_OVERRIDE.mkdir(parents=True, exist_ok=True)
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


def _bool_setting(value: str, default: bool = False) -> bool:
    try:
        v = str(value if value is not None else default).strip().lower()
        return v in ("1", "true", "yes", "on")
    except Exception:
        return bool(default)


def load_settings() -> dict:
    d = _ensure_cfg()["DEFAULT"]
    return {
        "input_folder_asset": d.get("input_folder_asset", "input/asset"),
        "working_epsg": int(d.get("workingprojection_epsg", "4326")),
        "ttk_theme": d.get("ttk_bootstrap_theme", "flatly"),
        "asset_group_file": d.get("asset_group_parquet_file", "tbl_asset_group.parquet"),
        "import_validate_geometries": _bool_setting(d.get("import_validate_geometries", "false"), False),
        "import_simplify_geometries": _bool_setting(d.get("import_simplify_geometries", "false"), False),
        "import_simplify_tolerance_m": float(d.get("import_simplify_tolerance_m", "1.0")),
        "import_simplify_preserve_topology": _bool_setting(d.get("import_simplify_preserve_topology", "true"), True),
    }


def _atomic_write_parquet(df: pd.DataFrame, path: Path):
    out_dir = path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise


def load_asset_group_df(file_name: str) -> pd.DataFrame:
    path = _parquet_path(file_name)
    if path.exists():
        try:
            df = pd.read_parquet(path)
        except Exception:
            df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    else:
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    str_cols = [
        "name_original",
        "title_fromuser",
        "name_gis_assetgroup",
        "sensitivity_code",
        "sensitivity_description",
        PURPOSE_COLUMN,
        STYLING_COLUMN,
    ]
    for col in str_cols:
        df[col] = df[col].astype("string").fillna("")

    if "id" not in df.columns or df["id"].isna().all():
        df["id"] = range(1, len(df) + 1)
    return df


def save_asset_group_df(file_name: str, df: pd.DataFrame) -> bool:
    path = _parquet_path(file_name, for_write=True)
    try:
        _atomic_write_parquet(df, path)
        return True
    except Exception:
        return False


# =====================================================================
# Thread-safe signal bridge for import progress/log
# =====================================================================
class _ImportSignals(QObject):
    log_message = Signal(str)
    progress_update = Signal(float)
    import_finished = Signal()


# =====================================================================
# Main window
# =====================================================================
class AssetManagerWindow(QMainWindow):

    def __init__(self, base_dir: Path):
        super().__init__()
        self.base_dir = base_dir
        self._import_running = False

        self.settings = load_settings()
        self.input_folder_asset = _abs_path_like(self.settings["input_folder_asset"])
        self.working_epsg = int(self.settings["working_epsg"])
        self.asset_group_file = self.settings["asset_group_file"]

        self.df = pd.DataFrame()
        self.idx = 0

        # Thread-safe signals
        self._signals = _ImportSignals()
        self._signals.log_message.connect(self._append_log_line)
        self._signals.progress_update.connect(self._set_progress)
        self._signals.import_finished.connect(self._on_import_finished)

        self._build_ui()
        self._log_import_diagnostics()
        self._refresh_edit_data()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _ts(self) -> str:
        return datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")

    def _log(self, message: str, level: str = "INFO"):
        line = f"{self._ts()} [{level}] - {message}"
        # Thread-safe: emit signal instead of direct widget access
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
        self.setWindowTitle("Asset manager")
        self.resize(740, 520)
        self.setMinimumSize(600, 400)

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
        exit_btn.clicked.connect(self._request_close)
        tab_row.addWidget(exit_btn, alignment=Qt.AlignTop)

        main_layout.addLayout(tab_row, stretch=1)

        # --- Import tab ---
        import_tab = QWidget()
        import_layout = QVBoxLayout(import_tab)
        import_layout.setContentsMargins(10, 10, 10, 10)
        import_layout.setSpacing(8)
        self._build_import_tab(import_layout)
        self.tabs.addTab(import_tab, "Import assets")

        # --- Edit tab ---
        edit_tab = QWidget()
        edit_layout = QVBoxLayout(edit_tab)
        edit_layout.setContentsMargins(10, 10, 10, 10)
        edit_layout.setSpacing(8)
        self._build_edit_tab(edit_layout)
        self.tabs.addTab(edit_tab, "Edit assets")

        # --- Bottom status ---
        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("color: #9a8a6e; font-size: 8pt;")
        main_layout.addWidget(self.summary_label)

    def _request_close(self):
        if self._import_running:
            QMessageBox.warning(self, "Import running",
                                "Import is still running. Please wait for completion before exiting.")
            return
        self.close()

    def _log_import_diagnostics(self):
        cfg_display = _CFG_PATH if _CFG_PATH is not None else (self.base_dir / "config.ini")
        self._log(f"BASE_DIR: {self.base_dir}")
        self._log(f"Config used: {cfg_display}")
        self._log(f"GeoParquet out: {_select_parquet_dir()}")
        self._log(f"Assets in: {self.input_folder_asset}")
        self._log(f"EPSG: {self.working_epsg}")

    # ------------------------------------------------------------------
    # Import tab
    # ------------------------------------------------------------------
    def _build_import_tab(self, layout):
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_widget = QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMinimumHeight(100)
        self.log_widget.setMaximumHeight(200)
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

        # Options
        opt_row = QHBoxLayout()
        self.validate_check = QCheckBox("Validate geometries")
        self.validate_check.setChecked(bool(self.settings["import_validate_geometries"]))
        opt_row.addWidget(self.validate_check)

        self.simplify_check = QCheckBox("Simplify geometries")
        self.simplify_check.setChecked(bool(self.settings["import_simplify_geometries"]))
        opt_row.addWidget(self.simplify_check)
        opt_row.addStretch()
        layout.addLayout(opt_row)

        # Action buttons
        btn_row = QHBoxLayout()
        import_btn = QPushButton("Import assets")
        import_btn.setProperty("role", "primary")
        import_btn.clicked.connect(self._start_import)
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

        form_group = QGroupBox("Asset group details")
        form = QGridLayout(form_group)
        form.setColumnStretch(1, 1)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)

        # GIS name (read-only)
        form.addWidget(QLabel("GIS name"), 0, 0, Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_name_gis = QLabel("")
        self.lbl_name_gis.setStyleSheet("font-weight: 600;")
        form.addWidget(self.lbl_name_gis, 0, 1)

        # Original name
        form.addWidget(QLabel("Original name"), 1, 0, Qt.AlignRight | Qt.AlignVCenter)
        self.edit_name_original = QLineEdit()
        form.addWidget(self.edit_name_original, 1, 1)

        # Title
        form.addWidget(QLabel("Title (for presentation)"), 2, 0, Qt.AlignRight | Qt.AlignVCenter)
        self.edit_title = QLineEdit()
        form.addWidget(self.edit_title, 2, 1)

        # Purpose
        form.addWidget(QLabel("Layer purpose / notes"), 3, 0, Qt.AlignRight | Qt.AlignTop)
        self.txt_purpose = QPlainTextEdit()
        self.txt_purpose.setMaximumHeight(120)
        form.addWidget(self.txt_purpose, 3, 1)

        # Styling
        form.addWidget(QLabel("Styling JSON (optional)"), 4, 0, Qt.AlignRight | Qt.AlignTop)
        self.txt_style = QPlainTextEdit()
        self.txt_style.setMaximumHeight(100)
        form.addWidget(self.txt_style, 4, 1)

        layout.addWidget(form_group, stretch=1)

        # Navigation + save controls
        controls = QHBoxLayout()
        controls.setSpacing(8)

        self.counter_label = QLabel("0 / 0")
        self.counter_label.setStyleSheet("font-weight: 600; min-width: 60px;")
        controls.addWidget(self.counter_label)

        prev_btn = QPushButton("Previous")
        prev_btn.clicked.connect(lambda: self._navigate(-1))
        controls.addWidget(prev_btn)

        next_btn = QPushButton("Next")
        next_btn.clicked.connect(lambda: self._navigate(1))
        controls.addWidget(next_btn)

        controls.addStretch()

        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._refresh_edit_data)
        controls.addWidget(reload_btn)

        save_next_btn = QPushButton("Save && Next")
        save_next_btn.setProperty("role", "primary")
        save_next_btn.clicked.connect(self._save_and_next)
        controls.addWidget(save_next_btn)

        save_btn = QPushButton("Save")
        save_btn.setProperty("role", "success")
        save_btn.clicked.connect(self._save_current)
        controls.addWidget(save_btn)

        layout.addLayout(controls)

    # ------------------------------------------------------------------
    # Import logic (all business logic preserved from original)
    # ------------------------------------------------------------------
    def _sync_import_options_to_config(self):
        try:
            cfg = _ensure_cfg()
            cfg["DEFAULT"]["import_validate_geometries"] = "true" if self.validate_check.isChecked() else "false"
            cfg["DEFAULT"]["import_simplify_geometries"] = "true" if self.simplify_check.isChecked() else "false"
        except Exception:
            pass

    def _rglob_many(self, folder: Path, patterns: tuple[str, ...]) -> list[Path]:
        files: list[Path] = []
        for pat in patterns:
            files.extend(folder.rglob(pat))
        return files

    def _scan_for_files(self, label: str, folder: Path, patterns: tuple[str, ...]) -> list[Path]:
        if not folder.exists():
            self._log(f"{label} folder does not exist: {folder}", "WARN")
            return []
        self._log(f"{label}: scanning {folder} ...")
        t0 = time.time()
        files = self._rglob_many(folder, patterns)
        self._log(f"{label}: scan finished in {time.time() - t0:.1f}s -> {len(files)} file(s).")
        return files

    def _read_and_reproject(self, filepath: Path, layer: str | None) -> gpd.GeoDataFrame:
        try:
            gdf = gpd.read_file(filepath, layer=layer) if layer else gpd.read_file(filepath)
            if gdf.crs is None:
                gdf.set_crs(epsg=self.working_epsg, inplace=True)
            elif (gdf.crs.to_epsg() or self.working_epsg) != self.working_epsg:
                gdf = gdf.to_crs(epsg=self.working_epsg)
            if gdf.geometry.name != "geometry":
                gdf = gdf.set_geometry(gdf.geometry.name).rename_geometry("geometry")
            try:
                gdf["geometry"] = gdf.geometry.apply(_force_2d_geom)
            except Exception:
                pass
            return gdf
        except Exception as exc:
            self._log(f"Read fail {filepath} (layer={layer}): {exc}", "ERROR")
            return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{self.working_epsg}")

    def _read_parquet_vector(self, filepath: Path) -> gpd.GeoDataFrame:
        try:
            gdf = gpd.read_parquet(filepath)
            if gdf.crs is None:
                gdf.set_crs(epsg=self.working_epsg, inplace=True)
            elif (gdf.crs.to_epsg() or self.working_epsg) != self.working_epsg:
                gdf = gdf.to_crs(epsg=self.working_epsg)
            if gdf.geometry.name != "geometry":
                gdf = gdf.set_geometry(gdf.geometry.name).rename_geometry("geometry")
            try:
                gdf["geometry"] = gdf.geometry.apply(_force_2d_geom)
            except Exception:
                pass
            return gdf
        except Exception as exc:
            self._log(f"Read fail (parquet) {filepath}: {exc}", "ERROR")
            return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{self.working_epsg}")

    def _validate_geometry(self, geom):
        if geom is None:
            return None
        try:
            if geom.is_empty:
                return geom
        except Exception:
            pass
        try:
            if shapely_make_valid is not None:
                return shapely_make_valid(geom)
        except Exception:
            pass
        try:
            return geom.buffer(0)
        except Exception:
            return geom

    def _apply_quality_controls(self, gdf: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
        if gdf is None or gdf.empty or "geometry" not in gdf.columns:
            return gdf
        validate = self.validate_check.isChecked()
        simplify = self.simplify_check.isChecked()
        tol_m = float(self.settings.get("import_simplify_tolerance_m", 1.0))
        preserve = bool(self.settings.get("import_simplify_preserve_topology", True))

        out = gdf
        if validate:
            try:
                out = out.copy()
                out["geometry"] = out.geometry.apply(self._validate_geometry)
                out = out[out.geometry.notna()].copy()
            except Exception as exc:
                self._log(f"{label}: validate failed: {exc}", "WARN")

        if simplify and tol_m > 0:
            try:
                tol = tol_m / 111_320.0 if int(self.working_epsg) == 4326 else tol_m
                out = out.copy()
                out["geometry"] = out.geometry.simplify(tol, preserve_topology=preserve)
                out = out[out.geometry.notna()].copy()
            except Exception as exc:
                self._log(f"{label}: simplify failed: {exc}", "WARN")
        return out

    def _ensure_geo_gdf(self, records_or_gdf, crs_str: str, expected_columns: list[str] | None = None) -> gpd.GeoDataFrame:
        expected = list(expected_columns or [])

        if isinstance(records_or_gdf, gpd.GeoDataFrame):
            gdf = records_or_gdf.copy()
            if gdf.geometry.name != "geometry":
                gdf = gdf.set_geometry(gdf.geometry.name).rename_geometry("geometry")
            if gdf.crs is None and crs_str:
                gdf.set_crs(crs_str, inplace=True)
            elif str(gdf.crs) != crs_str and crs_str:
                gdf = gdf.to_crs(crs_str)
        else:
            df = records_or_gdf.copy() if isinstance(records_or_gdf, pd.DataFrame) else pd.DataFrame(records_or_gdf)
            if "geometry" not in df.columns:
                df["geometry"] = None
            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs_str)

        for col in expected:
            if col not in gdf.columns:
                gdf[col] = pd.NA
        if expected:
            ordered = [col for col in expected if col in gdf.columns]
            remaining = [col for col in gdf.columns if col not in ordered]
            gdf = gdf[ordered + remaining]
        return gdf

    def _save_parquet(self, name: str, gdf: gpd.GeoDataFrame):
        path = _parquet_path(f"{name}.parquet", for_write=True)
        _atomic_write_parquet(gdf, path)
        self._log(f"Saved {name} -> {path} (rows={len(gdf)})")

    def _import_spatial_data_asset(self):
        asset_objects, asset_groups = [], []
        group_id, object_id = 1, 1
        files = self._scan_for_files("Assets", self.input_folder_asset, ("*.shp", "*.gpkg", "*.parquet"))
        if not files:
            self._log("No supported asset files found (*.shp, *.gpkg, *.parquet).", "WARN")
        self._update_progress(2)

        for i, fp in enumerate(files, start=1):
            self._update_progress(5 + 85 * (i / max(1, len(files))))
            if fp.suffix.lower() == ".gpkg":
                try:
                    layers = fiona.listlayers(fp)
                except Exception:
                    layers = []
                for layer in layers:
                    gdf = self._read_and_reproject(fp, layer)
                    if gdf.empty:
                        continue
                    gdf = self._apply_quality_controls(gdf, f"Assets:{fp.name}:{layer}")
                    bbox_polygon = box(*gdf.total_bounds)
                    count = len(gdf)
                    asset_groups.append({
                        "id": group_id,
                        "name_original": layer,
                        "name_gis_assetgroup": f"layer_{group_id:03d}",
                        "title_fromuser": fp.stem,
                        "date_import": datetime.datetime.now(),
                        "geometry": bbox_polygon,
                        "total_asset_objects": int(count),
                        "importance": 0,
                        "susceptibility": 0,
                        "sensitivity": 0,
                        "sensitivity_code": "",
                        "sensitivity_description": "",
                        PURPOSE_COLUMN: "",
                        STYLING_COLUMN: "",
                    })
                    for _, row in gdf.iterrows():
                        attrs = "; ".join([f"{c}: {row[c]}" for c in gdf.columns if c != gdf.geometry.name])
                        asset_objects.append({
                            "id": object_id,
                            "asset_group_name": layer,
                            "attributes": attrs,
                            "process": True,
                            "ref_asset_group": group_id,
                            "geometry": row.geometry,
                        })
                        object_id += 1
                    group_id += 1
            else:
                gdf = self._read_parquet_vector(fp) if fp.suffix.lower() == ".parquet" else self._read_and_reproject(fp, None)
                if gdf.empty:
                    continue
                gdf = self._apply_quality_controls(gdf, f"Assets:{fp.name}")
                layer = fp.stem
                bbox_polygon = box(*gdf.total_bounds)
                count = len(gdf)
                asset_groups.append({
                    "id": group_id,
                    "name_original": layer,
                    "name_gis_assetgroup": f"layer_{group_id:03d}",
                    "title_fromuser": layer,
                    "date_import": datetime.datetime.now(),
                    "geometry": bbox_polygon,
                    "total_asset_objects": int(count),
                    "importance": 0,
                    "susceptibility": 0,
                    "sensitivity": 0,
                    "sensitivity_code": "",
                    "sensitivity_description": "",
                    PURPOSE_COLUMN: "",
                    STYLING_COLUMN: "",
                })
                for _, row in gdf.iterrows():
                    attrs = "; ".join([f"{c}: {row[c]}" for c in gdf.columns if c != gdf.geometry.name])
                    asset_objects.append({
                        "id": object_id,
                        "asset_group_name": layer,
                        "attributes": attrs,
                        "process": True,
                        "ref_asset_group": group_id,
                        "geometry": row.geometry,
                    })
                    object_id += 1
                group_id += 1

        crs = f"EPSG:{self.working_epsg}"
        return (
            self._ensure_geo_gdf(asset_objects, crs, expected_columns=ASSET_OBJECT_COLUMNS),
            self._ensure_geo_gdf(asset_groups, crs, expected_columns=ASSET_GROUP_COLUMNS),
        )

    def _run_import_asset(self):
        self._update_progress(0)
        self._log("Step [Assets] STARTED")
        try:
            asset_objects, asset_groups = self._import_spatial_data_asset()
            self._save_parquet("tbl_asset_object", asset_objects)
            self._save_parquet("tbl_asset_group", asset_groups)
            self._log("Step [Assets] COMPLETED")
        except Exception as exc:
            self._log(f"Step [Assets] FAILED: {exc}", "ERROR")
        finally:
            self._update_progress(100)

    def _start_import(self):
        if self._import_running:
            self._log("Import already running.", "WARN")
            return

        def _job():
            self._import_running = True
            self._sync_import_options_to_config()
            try:
                self._run_import_asset()
            finally:
                self._import_running = False
                self._signals.import_finished.emit()

        threading.Thread(target=_job, daemon=False).start()

    def _on_import_finished(self):
        """Slot: called on UI thread when import completes."""
        self._refresh_edit_data()

    # ------------------------------------------------------------------
    # Edit logic (all business logic preserved from original)
    # ------------------------------------------------------------------
    def _update_counter(self):
        total = len(self.df)
        current = (self.idx + 1) if total else 0
        self.counter_label.setText(f"{current} / {total}")

    def _clear_editor(self):
        self.lbl_name_gis.setText("")
        self.edit_name_original.setText("")
        self.edit_title.setText("")
        self.txt_purpose.setPlainText("")
        self.txt_style.setPlainText("")
        self._update_counter()

    def _load_record(self):
        if len(self.df) == 0:
            self._clear_editor()
            return
        self.idx = max(0, min(self.idx, len(self.df) - 1))
        row = self.df.iloc[self.idx]
        self.lbl_name_gis.setText(str(row.get("name_gis_assetgroup", "") or ""))
        self.edit_name_original.setText(str(row.get("name_original", "") or ""))
        self.edit_title.setText(str(row.get("title_fromuser", "") or ""))
        self.txt_purpose.setPlainText(str(row.get(PURPOSE_COLUMN, "") or ""))
        self.txt_style.setPlainText(str(row.get(STYLING_COLUMN, "") or ""))
        self._update_counter()
        self.edit_state_label.setText(f"Record {self.idx + 1} of {len(self.df)}")

    def _write_back_to_df(self):
        if len(self.df) == 0:
            return
        self.df.at[self.idx, "name_original"] = (self.edit_name_original.text() or "").strip()
        self.df.at[self.idx, "title_fromuser"] = (self.edit_title.text() or "").strip()
        self.df.at[self.idx, PURPOSE_COLUMN] = self.txt_purpose.toPlainText().strip()
        self.df.at[self.idx, STYLING_COLUMN] = self.txt_style.toPlainText().strip()

    def _save_current(self) -> bool:
        if len(self.df) == 0:
            QMessageBox.information(self, "Nothing to save", "There are no asset groups to save.")
            return False
        self._write_back_to_df()
        ok = save_asset_group_df(self.asset_group_file, self.df)
        if ok:
            self.edit_state_label.setText("Saved.")
            self._log("Asset group record saved from asset_manage.")
            return True
        QMessageBox.critical(self, "Save failed", "Could not write asset group GeoParquet.")
        return False

    def _save_and_next(self):
        if self._save_current():
            self._navigate(+1)

    def _navigate(self, step: int):
        if len(self.df) == 0:
            return
        self._write_back_to_df()
        self.idx = max(0, min(self.idx + int(step), len(self.df) - 1))
        self._load_record()

    def _refresh_edit_data(self):
        self.df = load_asset_group_df(self.asset_group_file)
        self.idx = min(self.idx, max(len(self.df) - 1, 0))
        if len(self.df) == 0:
            self._clear_editor()
            self.edit_state_label.setText("No asset groups found. Import assets first.")
        else:
            self._load_record()
        parquet_path = _parquet_path(self.asset_group_file)
        self.summary_label.setText(f"Asset group file: {parquet_path}  |  rows: {len(self.df)}")


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
        app.setStyleSheet(ASSET_STYLESHEET)
        own_app = True
    window = AssetManagerWindow(BASE_DIR)
    window.show()
    if own_app:
        app.exec()
    return window


def main():
    global BASE_DIR, _CFG, _CFG_PATH, _PARQUET_OVERRIDE

    parser = argparse.ArgumentParser(description="Unified asset import + asset-group editor")
    parser.add_argument("--original_working_directory", required=False, help="Path to running folder")
    args = parser.parse_args()

    BASE_DIR = find_base_dir(args.original_working_directory)
    _CFG = None
    _CFG_PATH = None
    _PARQUET_OVERRIDE = None
    _ensure_cfg()

    app = QApplication([])
    app.setStyleSheet(ASSET_STYLESHEET)

    try:
        ico = BASE_DIR / "system_resources" / "mesa.ico"
        if ico.exists():
            app.setWindowIcon(QIcon(str(ico)))
    except Exception:
        pass

    window = AssetManagerWindow(BASE_DIR)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
