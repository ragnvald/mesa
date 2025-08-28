# -*- coding: utf-8 -*-
# 08_edit_lines.py — GeoParquet-native editor for line parameters
# - Reads/writes output/geoparquet/tbl_lines.parquet
# - Edits: name_user (title), segment_length, segment_width, description
# - Displays (read-only): name_gis
#
# Key: keep numeric columns as Int64 (nullable) to avoid pyarrow "bytes vs int" issues.

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import os
import sys
import argparse
import configparser
import datetime

import pandas as pd
import geopandas as gpd

import tkinter as tk
from tkinter import messagebox

import ttkbootstrap as tb
from ttkbootstrap.constants import PRIMARY, SUCCESS, WARNING

# ---------------------------
# Paths / config helpers
# ---------------------------
def resolve_base_dir(passed: str | None) -> str:
    base = passed or os.getcwd()
    # If launched from /system, step up one directory to the project root
    if os.path.basename(base).lower() == "system":
        base = os.path.abspath(os.path.join(base, ".."))
    return base

def gpq_dir(base_dir: str) -> str:
    d = os.path.join(base_dir, "output", "geoparquet")
    os.makedirs(d, exist_ok=True)
    return d

def lines_parquet_path(base_dir: str) -> str:
    return os.path.join(gpq_dir(base_dir), "tbl_lines.parquet")

def config_path(base_dir: str) -> str:
    return os.path.join(base_dir, "system", "config.ini")

def read_config(path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")
    if "DEFAULT" not in cfg:
        cfg["DEFAULT"] = {}
    return cfg

# ---------------------------
# Logging
# ---------------------------
def write_to_log(base_dir: str, message: str):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    line = f"{timestamp} - {message}"
    try:
        with open(os.path.join(base_dir, "log.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# ---------------------------
# Data IO (GeoParquet)
# ---------------------------
REQUIRED_COLUMNS = [
    "name_gis",          # display (read-only, system name)
    "name_user",         # editable title
    "segment_length",    # editable Int64 (meters)
    "segment_width",     # editable Int64 (meters)
    "description"        # editable string
    # geometry is present; we won't touch it
]

def _ensure_schema_types(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Make sure editable numeric columns are Int64 (nullable), and text columns are string dtype.
    This prevents object dtype from leaking into Parquet as 'binary' and causing save errors.
    """
    # Create missing columns if needed
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Normalize dtypes
    # Text columns
    for col in ["name_gis", "name_user", "description"]:
        df[col] = df[col].astype("string").fillna("")

    # Numeric columns
    for col in ["segment_length", "segment_width"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        else:
            df[col] = pd.Series([pd.NA] * len(df), dtype="Int64")

    return df

def load_lines_df(base_dir: str) -> gpd.GeoDataFrame:
    pq = lines_parquet_path(base_dir)
    if not os.path.exists(pq):
        raise FileNotFoundError(f"GeoParquet file not found: {pq}")
    try:
        df = gpd.read_parquet(pq)
    except Exception as e:
        write_to_log(base_dir, f"Error reading parquet {pq}: {e}")
        raise

    # Ensure required columns and proper dtypes for editing
    df = _ensure_schema_types(df)
    return df

def save_lines_df(base_dir: str, df: gpd.GeoDataFrame) -> bool:
    """
    Save to GeoParquet with stable schema. Always call _ensure_schema_types before saving.
    """
    pq = lines_parquet_path(base_dir)
    try:
        df = _ensure_schema_types(df)
        df.to_parquet(pq, index=False)
        write_to_log(base_dir, f"Line data saved: {os.path.relpath(pq, start=base_dir)} ({len(df)} rows)")
        return True
    except Exception as e:
        write_to_log(base_dir, f"Error saving data: {e}")
        messagebox.showerror("Save failed", f"Could not write GeoParquet.\n\n{e}")
        return False

# ---------------------------
# Parsing helpers
# ---------------------------
def _parse_int_or_na(text: str):
    """
    Convert user text → Int or <NA>.
    - Empty / whitespace -> <NA>
    - Non-numeric -> raises ValueError (handled by caller to show validation message)
    """
    t = (text or "").strip()
    if t == "":
        return pd.NA
    # Allow decimals but cast to int if they are integral (e.g., "1000" or "1000.0")
    val = pd.to_numeric(t, errors="raise")
    if pd.isna(val):
        return pd.NA
    # No negative lengths/widths
    if float(val) < 0:
        raise ValueError("Negative values are not allowed.")
    ival = int(round(float(val)))
    return ival

# ---------------------------
# UI Controller
# ---------------------------
class LinesEditor:
    def __init__(self, root: tb.Window, base_dir: str, theme: str):
        self.root = root
        self.base_dir = base_dir
        self.theme = theme

        # Data
        self.df = load_lines_df(base_dir)
        self.idx = 0

        # Vars
        self.var_name_gis = tk.StringVar(value="")
        self.var_name_user = tk.StringVar(value="")
        self.var_segment_length = tk.StringVar(value="")
        self.var_segment_width = tk.StringVar(value="")
        self.var_description = tk.StringVar(value="")
        self.var_counter = tk.StringVar(value="0 / 0")
        self.var_status = tk.StringVar(value="")  # validation / status line

        # Window
        self.root.title("Edit lines / segments (GeoParquet)")
        try:
            icon = os.path.join(base_dir, "system_resources", "mesa.ico")
            if os.path.exists(icon):
                self.root.iconbitmap(icon)
        except Exception:
            pass

        self._build_ui()
        if len(self.df) == 0:
            messagebox.showinfo("No data", "No lines found in GeoParquet (tbl_lines.parquet).")
            self._update_counter()
        else:
            self._load_record()

    def _build_ui(self):
        pad = dict(padx=10, pady=8)

        # About
        top = tb.LabelFrame(self.root, text="About", bootstyle="info")
        top.grid(row=0, column=0, columnspan=4, sticky="ew", **pad)
        info = ("Registered lines are used to create segment polygons.\n"
                "You can edit their title and the default segment size here.\n"
                "Numeric fields accept integer meters (leave empty to unset).")
        tk.Label(top, text=info, justify="left", wraplength=640).pack(anchor="w", padx=10, pady=8)

        # GIS name (read-only)
        tk.Label(self.root, text="GIS name").grid(row=1, column=0, sticky="w", **pad)
        tk.Label(self.root, textvariable=self.var_name_gis, relief="sunken", anchor="w", width=58)\
            .grid(row=1, column=1, columnspan=3, sticky="ew", **pad)

        # Title
        tk.Label(self.root, text="Title").grid(row=2, column=0, sticky="w", **pad)
        tb.Entry(self.root, textvariable=self.var_name_user, width=60)\
            .grid(row=2, column=1, columnspan=3, sticky="ew", **pad)

        # Segment length (meters)
        tk.Label(self.root, text="Length of segments (m)").grid(row=3, column=0, sticky="w", **pad)
        tb.Entry(self.root, textvariable=self.var_segment_length, width=24)\
            .grid(row=3, column=1, sticky="w", **pad)

        # Segment width (meters)
        tk.Label(self.root, text="Segments width (m)").grid(row=3, column=2, sticky="w", **pad)
        tb.Entry(self.root, textvariable=self.var_segment_width, width=24)\
            .grid(row=3, column=3, sticky="w", **pad)

        # Description
        tk.Label(self.root, text="Description").grid(row=4, column=0, sticky="w", **pad)
        tb.Entry(self.root, textvariable=self.var_description, width=60)\
            .grid(row=4, column=1, columnspan=3, sticky="ew", **pad)

        # Status / validation line
        self.status_label = tk.Label(self.root, textvariable=self.var_status, fg="#c0392b", anchor="w")
        self.status_label.grid(row=5, column=0, columnspan=4, sticky="ew", padx=10)

        # Counter + buttons
        tk.Label(self.root, textvariable=self.var_counter).grid(row=6, column=0, sticky="w", **pad)

        btns = tk.Frame(self.root); btns.grid(row=6, column=1, columnspan=3, sticky="e", **pad)
        tb.Button(btns, text="⟵ Previous", bootstyle=PRIMARY,
                  command=lambda: self._navigate(-1)).pack(side="left", padx=4)
        tb.Button(btns, text="Save", bootstyle=SUCCESS,
                  command=self._save_current).pack(side="left", padx=4)
        tb.Button(btns, text="Save & Next ⟶", bootstyle=PRIMARY,
                  command=self._save_and_next).pack(side="left", padx=4)
        tb.Button(btns, text="Exit", bootstyle=WARNING,
                  command=self.root.destroy).pack(side="left", padx=4)

        # Make middle columns stretch
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(3, weight=1)

    def _update_counter(self):
        total = len(self.df)
        idx1 = (self.idx + 1) if total else 0
        self.var_counter.set(f"{idx1} / {total}")

    def _load_record(self):
        self.idx = max(0, min(self.idx, max(0, len(self.df) - 1)))
        self._update_counter()
        self.var_status.set("")
        if len(self.df) == 0:
            self.var_name_gis.set("")
            self.var_name_user.set("")
            self.var_segment_length.set("")
            self.var_segment_width.set("")
            self.var_description.set("")
            return
        row = self.df.iloc[self.idx]
        self.var_name_gis.set(str(row.get("name_gis", "") or ""))
        self.var_name_user.set(str(row.get("name_user", "") or ""))
        # Show numeric Int64 values as text, blank if NA
        sl = row.get("segment_length", pd.NA)
        sw = row.get("segment_width", pd.NA)
        self.var_segment_length.set("" if pd.isna(sl) else str(int(sl)))
        self.var_segment_width.set("" if pd.isna(sw) else str(int(sw)))
        self.var_description.set(str(row.get("description", "") or ""))

    def _write_back_to_df(self, validate: bool = True) -> bool:
        """
        Pull values from UI, validate/cast, and write into the current row.
        Returns True if ok, False if validation failed.
        """
        if len(self.df) == 0:
            return False

        name_user = (self.var_name_user.get() or "").strip()
        desc = (self.var_description.get() or "").strip()

        # Parse numbers with validation
        try:
            sl = _parse_int_or_na(self.var_segment_length.get())
            sw = _parse_int_or_na(self.var_segment_width.get())
        except ValueError as e:
            if validate:
                self.var_status.set(f"Invalid number: {e}")
                return False
            else:
                sl = pd.NA; sw = pd.NA

        # Write back
        rix = self.idx
        self.df.at[rix, "name_user"] = name_user
        self.df.at[rix, "description"] = desc
        self.df.at[rix, "segment_length"] = sl
        self.df.at[rix, "segment_width"] = sw

        # Keep dtypes stable
        self.df = _ensure_schema_types(self.df)
        return True

    def _save_current(self) -> bool:
        if not self._write_back_to_df(validate=True):
            return False
        return save_lines_df(self.base_dir, self.df)

    def _save_and_next(self):
        if self._save_current():
            self._navigate(+1)

    def _navigate(self, step: int):
        if len(self.df) == 0:
            return
        # Save silently but do not block on validation; we still normalize dtypes
        self._write_back_to_df(validate=False)
        # Move
        self.idx = max(0, min(self.idx + step, len(self.df) - 1))
        self._load_record()


# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edit line attributes (GeoParquet)")
    parser.add_argument("--original_working_directory", required=False, help="Path to running folder")
    args = parser.parse_args()

    base_dir = resolve_base_dir(args.original_working_directory)
    cfg = read_config(config_path(base_dir))
    theme = cfg["DEFAULT"].get("ttk_bootstrap_theme", "flatly")

    app = tb.Window(themename=theme)
    editor = LinesEditor(app, base_dir, theme)
    app.mainloop()
