# -*- coding: utf-8 -*-
# 04_edit_asset_group.py — GeoParquet-native editor for tbl_asset_group
# - Reads/writes output/geoparquet/tbl_asset_group.parquet
# - Edits: name_original, title_fromuser (display-only: name_gis_assetgroup)
# - ttkbootstrap UI

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import os
import sys
import argparse
import configparser
import datetime
import pandas as pd

import tkinter as tk
from tkinter import messagebox

import ttkbootstrap as tb
from ttkbootstrap.constants import PRIMARY, SUCCESS, WARNING

# ---------------------------
# Paths / config helpers
# ---------------------------
def resolve_base_dir(passed: str | None) -> str:
    """
    Determine the original working directory:
    - If PyInstaller (_MEIPASS) is present, work relative to the parent of the 'system' folder next to the exe.
    - If invoked with --original_working_directory, prefer that.
    - If running from /system/, go one level up.
    """
    if passed:
        base = passed
    else:
        base = os.getcwd()

    # If user runs directly from system/, go one level up
    if os.path.basename(base).lower() == "system":
        base = os.path.abspath(os.path.join(base, ".."))
    return base

def config_path(base_dir: str) -> str:
    return os.path.join(base_dir, "system", "config.ini")

def gpq_dir(base_dir: str) -> str:
    d = os.path.join(base_dir, "output", "geoparquet")
    os.makedirs(d, exist_ok=True)
    return d

def asset_group_parquet(base_dir: str) -> str:
    return os.path.join(gpq_dir(base_dir), "tbl_asset_group.parquet")

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
    "id",
    "name_gis_assetgroup",  # system name (display only)
    "name_original",        # editable
    "title_fromuser",       # editable
    # The following may exist; we won’t edit them here but we keep them if present:
    "importance", "susceptibility", "sensitivity",
    "sensitivity_code", "sensitivity_description",
    "total_asset_objects"
]

def load_asset_group_df(base_dir: str) -> pd.DataFrame:
    """
    Read tbl_asset_group.parquet; if missing, return an empty frame with required columns.
    Ensure editable columns exist as strings (no NaN in the editor).
    """
    pq = asset_group_parquet(base_dir)
    if os.path.exists(pq):
        try:
            df = pd.read_parquet(pq)
        except Exception as e:
            write_to_log(base_dir, f"Error reading parquet {pq}: {e}")
            df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    else:
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)

    # Ensure required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Coerce editor columns to string (avoid NaN in UI)
    for col in ["name_original", "title_fromuser", "name_gis_assetgroup"]:
        df[col] = df[col].astype("string").fillna("")

    # If id is missing/empty, create a simple incremental id (do not overwrite if present)
    if df["id"].isna().all():
        df["id"] = range(1, len(df) + 1)

    return df

def save_asset_group_df(base_dir: str, df: pd.DataFrame) -> bool:
    """
    Save entire asset group table back to GeoParquet.
    """
    pq = asset_group_parquet(base_dir)
    try:
        df.to_parquet(pq, index=False)
        write_to_log(base_dir, f"Saved {os.path.relpath(pq, start=base_dir)} ({len(df)} rows)")
        return True
    except Exception as e:
        write_to_log(base_dir, f"Error writing parquet {pq}: {e}")
        return False

# ---------------------------
# UI
# ---------------------------
class AssetGroupEditor:
    def __init__(self, root: tb.Window, base_dir: str, theme: str):
        self.root = root
        self.base_dir = base_dir
        self.theme = theme

        # Data
        self.df = load_asset_group_df(base_dir)
        self.idx = 0

        # Vars
        self.var_name_gis = tk.StringVar(value="")
        self.var_name_original = tk.StringVar(value="")
        self.var_title_fromuser = tk.StringVar(value="")
        self.var_counter = tk.StringVar(value="0 / 0")

        # Window
        self.root.title("Edit asset groups (GeoParquet)")
        try:
            icon = os.path.join(base_dir, "system_resources", "mesa.ico")
            if os.path.exists(icon):
                self.root.iconbitmap(icon)
        except Exception:
            pass

        # Layout
        self._build_ui()

        # Load first record
        if len(self.df) == 0:
            messagebox.showinfo("No data", "No asset groups found in GeoParquet.\n"
                                            "Import assets first (or create tbl_asset_group.parquet).")
            self._update_counter()
        else:
            self._load_record()

    def _build_ui(self):
        pad = dict(padx=10, pady=8)

        # Top info
        top = tb.LabelFrame(self.root, text="About", bootstyle="info")
        top.grid(row=0, column=0, columnspan=4, sticky="ew", **pad)
        info = ("You can give asset groups nicer display names here.\n"
                "‘GIS name’ is the internal, system-generated name and cannot be edited.")
        tk.Label(top, text=info, justify="left", wraplength=640).pack(anchor="w", padx=10, pady=8)

        # Form
        tk.Label(self.root, text="GIS name").grid(row=1, column=0, sticky="w", **pad)
        tk.Label(self.root, textvariable=self.var_name_gis, relief="sunken", anchor="w", width=60)\
            .grid(row=1, column=1, columnspan=3, sticky="ew", **pad)

        tk.Label(self.root, text="Original name").grid(row=2, column=0, sticky="w", **pad)
        tb.Entry(self.root, textvariable=self.var_name_original, width=62)\
            .grid(row=2, column=1, columnspan=3, sticky="ew", **pad)

        tk.Label(self.root, text="Title (for presentation)").grid(row=3, column=0, sticky="w", **pad)
        tb.Entry(self.root, textvariable=self.var_title_fromuser, width=62)\
            .grid(row=3, column=1, columnspan=3, sticky="ew", **pad)

        # Counter
        tk.Label(self.root, textvariable=self.var_counter).grid(row=4, column=0, sticky="w", **pad)

        # Buttons
        nav = tk.Frame(self.root)
        nav.grid(row=4, column=1, columnspan=3, sticky="e", **pad)

        tb.Button(nav, text="⟵ Previous", bootstyle=PRIMARY,
                  command=lambda: self._navigate(-1)).pack(side="left", padx=4)
        tb.Button(nav, text="Save", bootstyle=SUCCESS,
                  command=self._save_current).pack(side="left", padx=4)
        tb.Button(nav, text="Save & Next ⟶", bootstyle=PRIMARY,
                  command=self._save_and_next).pack(side="left", padx=4)
        tb.Button(nav, text="Exit", bootstyle=WARNING,
                  command=self.root.destroy).pack(side="left", padx=4)

        # Make column 1 expandable
        self.root.grid_columnconfigure(1, weight=1)

    def _update_counter(self):
        total = len(self.df)
        idx1 = (self.idx + 1) if total else 0
        self.var_counter.set(f"{idx1} / {total}")

    def _load_record(self):
        # Clamp index
        self.idx = max(0, min(self.idx, max(0, len(self.df) - 1)))
        self._update_counter()
        if len(self.df) == 0:
            self.var_name_gis.set("")
            self.var_name_original.set("")
            self.var_title_fromuser.set("")
            return
        row = self.df.iloc[self.idx]
        self.var_name_gis.set(str(row.get("name_gis_assetgroup", "") or ""))
        self.var_name_original.set(str(row.get("name_original", "") or ""))
        self.var_title_fromuser.set(str(row.get("title_fromuser", "") or ""))

    def _write_back_to_df(self):
        if len(self.df) == 0:
            return
        # Basic sanitation: strip whitespace
        name_original = (self.var_name_original.get() or "").strip()
        title_user = (self.var_title_fromuser.get() or "").strip()

        self.df.at[self.idx, "name_original"] = name_original
        self.df.at[self.idx, "title_fromuser"] = title_user

    def _save_current(self) -> bool:
        try:
            if len(self.df) == 0:
                messagebox.showinfo("Nothing to save", "There are no rows to save.")
                return False
            self._write_back_to_df()
            ok = save_asset_group_df(self.base_dir, self.df)
            if ok:
                write_to_log(self.base_dir, "Asset group record saved.")
            else:
                messagebox.showerror("Save failed", "Could not write the GeoParquet file.")
            return ok
        except Exception as e:
            write_to_log(self.base_dir, f"Error during save: {e}")
            messagebox.showerror("Save failed", str(e))
            return False

    def _save_and_next(self):
        if self._save_current():
            self._navigate(+1)

    def _navigate(self, step: int):
        if len(self.df) == 0:
            return
        # Save silently before leaving the record
        self._write_back_to_df()
        # move
        self.idx = max(0, min(self.idx + step, len(self.df) - 1))
        self._load_record()


# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edit asset-group titles (GeoParquet)")
    parser.add_argument("--original_working_directory", required=False, help="Path to running folder")
    args = parser.parse_args()

    base_dir = resolve_base_dir(args.original_working_directory)
    cfg = read_config(config_path(base_dir))
    theme = cfg["DEFAULT"].get("ttk_bootstrap_theme", "flatly")

    app = tb.Window(themename=theme)
    editor = AssetGroupEditor(app, base_dir, theme)
    app.mainloop()
