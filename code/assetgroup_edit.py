# -*- coding: utf-8 -*-
# 04_edit_asset_group.py — GeoParquet-native editor for tbl_asset_group
# - Stable BASE_DIR resolution (env/CLI/script/CWD)
# - Reads/writes <parquet_folder>/<asset_group_parquet_file> (defaults: output/geoparquet/tbl_asset_group.parquet)
# - Atomic writes; logs to <BASE_DIR>/log.txt
# - ttkbootstrap optional (falls back to standard Tk)

import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except Exception:
    try:
        locale.setlocale(locale.LC_ALL, '')
    except Exception:
        pass

import os, argparse, configparser, datetime, tempfile
from pathlib import Path
import pandas as pd

import tkinter as tk
from tkinter import messagebox, scrolledtext
try:
    import ttkbootstrap as tb
    from ttkbootstrap.constants import PRIMARY, SUCCESS, WARNING, INFO
except Exception:
    tb = None
    PRIMARY = SUCCESS = WARNING = INFO = None
from tkinter import ttk as ttk_std

# ------------------------ Base dir / config ------------------------
BASE_DIR: Path = Path(".").resolve()
_CFG: configparser.ConfigParser | None = None
_CFG_PATH: Path | None = None
ASSET_GROUP_FILE_NAME: str = "tbl_asset_group.parquet"

def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

def _has_config_at(root: Path) -> bool:
    return _exists(root / "config.ini") or _exists(root / "system" / "config.ini")

def resolve_base_dir(cli_workdir: str | None) -> Path:
    """
    Priority:
      1) env MESA_BASE_DIR
      2) --original_working_directory (CLI)
      3) script folder & parents (prefer repo/code root)
      4) CWD (treat cwd/system -> parent)
    """
    env_base = os.environ.get("MESA_BASE_DIR")
    if env_base:
        return Path(env_base).resolve()

    if cli_workdir:
        return Path(cli_workdir).resolve()

    here = Path(__file__).resolve()
    candidates = [
        here.parent,                # .../system
        here.parent.parent,         # .../code
        here.parent.parent.parent,  # repo root (if any)
        Path(os.getcwd()),
        Path(os.getcwd()) / "code",
    ]

    seen, uniq = set(), []
    for c in candidates:
        try:
            r = c.resolve()
        except Exception:
            r = c
        if r not in seen:
            seen.add(r)
            uniq.append(r)

    for c in uniq:
        if _has_config_at(c):
            return c

    # Fallback heuristics
    if here.parent.name.lower() == "system":
        return here.parent.parent
    return here.parent

def _ensure_cfg() -> configparser.ConfigParser:
    """Load config, preferring <base>/config.ini over <base>/system/config.ini. Provide sane defaults."""
    global _CFG, _CFG_PATH, ASSET_GROUP_FILE_NAME
    if _CFG is not None:
        return _CFG

    cfg = configparser.ConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
    flat = BASE_DIR / "config.ini"
    legacy = BASE_DIR / "system" / "config.ini"

    loaded = False
    if flat.exists():
        try:
            cfg.read(flat, encoding="utf-8"); _CFG_PATH = flat; loaded = True
        except Exception:
            pass
    if not loaded and legacy.exists():
        try:
            cfg.read(legacy, encoding="utf-8"); _CFG_PATH = legacy; loaded = True
        except Exception:
            pass

    if "DEFAULT" not in cfg:
        cfg["DEFAULT"] = {}

    d = cfg["DEFAULT"]
    d.setdefault("parquet_folder", "output/geoparquet")
    d.setdefault("ttk_bootstrap_theme", "flatly")
    d.setdefault("asset_group_parquet_file", "tbl_asset_group.parquet")

    ASSET_GROUP_FILE_NAME = d.get("asset_group_parquet_file", "tbl_asset_group.parquet")
    _CFG = cfg
    return _CFG

def config_path(base_dir: Path) -> Path:
    cfg = base_dir / "config.ini"
    if cfg.exists():
        return cfg
    return base_dir / "system" / "config.ini"

def gpq_dir(base_dir: Path) -> Path:
    cfg = _ensure_cfg()
    sub = cfg["DEFAULT"].get("parquet_folder", "output/geoparquet")
    d = base_dir / sub
    d.mkdir(parents=True, exist_ok=True)
    return d

def asset_group_parquet(base_dir: Path) -> Path:
    return gpq_dir(base_dir) / ASSET_GROUP_FILE_NAME

# ------------------------ Stats / logging ------------------------
def increment_stat_value(cfg_path: Path | str, stat_name: str, increment_value: int):
    try:
        cfg_path = Path(cfg_path)
        if not cfg_path.is_file():
            return
        with open(cfg_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        updated = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f'{stat_name} ='):
                parts = line.split('=', 1)
                if len(parts) == 2:
                    current_value = parts[1].strip()
                    try:
                        new_value = int(current_value) + int(increment_value)
                        lines[i] = f"{stat_name} = {new_value}\n"
                        updated = True
                        break
                    except ValueError:
                        return
        if updated:
            with open(cfg_path, 'w', encoding='utf-8', errors='replace') as f:
                f.writelines(lines)
    except Exception:
        pass

def write_to_log(base_dir: Path, message: str):
    ts = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    try:
        with open(base_dir / "log.txt", "a", encoding="utf-8") as f:
            f.write(f"{ts} - {message}\n")
    except Exception:
        pass

# ------------------------ Data I/O ------------------------
REQUIRED_COLUMNS = [
    "id",
    "name_gis_assetgroup",
    "name_original",
    "title_fromuser",
    "importance", "susceptibility", "sensitivity",
    "sensitivity_code", "sensitivity_description",
    "total_asset_objects",
]

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

def load_asset_group_df(base_dir: Path) -> pd.DataFrame:
    pq = asset_group_parquet(base_dir)
    if pq.exists():
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

    # Coerce types for key presentation columns
    for col in ["name_original", "title_fromuser", "name_gis_assetgroup", "sensitivity_code", "sensitivity_description"]:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("")

    # Ensure ID exists
    if "id" not in df.columns or df["id"].isna().all():
        df["id"] = range(1, len(df) + 1)

    return df

def save_asset_group_df(base_dir: Path, df: pd.DataFrame) -> bool:
    pq = asset_group_parquet(base_dir)
    try:
        _atomic_write_parquet(df, pq)
        write_to_log(base_dir, f"Saved {pq.relative_to(base_dir)} ({len(df)} rows)")
        return True
    except Exception as e:
        write_to_log(base_dir, f"Error writing parquet {pq}: {e}")
        return False

# ------------------------ UI ------------------------
class AssetGroupEditor:
    def __init__(self, root, base_dir: Path, theme: str):
        self.root = root
        self.base_dir = base_dir
        self.theme = theme

        self.df = load_asset_group_df(base_dir)
        self.idx = 0

        self.var_name_gis = tk.StringVar(value="")
        self.var_name_original = tk.StringVar(value="")
        self.var_title_fromuser = tk.StringVar(value="")
        self.var_counter = tk.StringVar(value="0 / 0")

        self.root.title("Edit asset groups (GeoParquet)")
        try:
            icon = base_dir / "system_resources" / "mesa.ico"
            if icon.exists() and hasattr(self.root, "iconbitmap"):
                self.root.iconbitmap(str(icon))
        except Exception:
            pass

        self._build_ui()

        if len(self.df) == 0:
            messagebox.showinfo(
                "No data",
                "No asset groups found in GeoParquet.\n"
                "Import assets first (or create the asset group table)."
            )
            self._update_counter()
        else:
            self._load_record()

    def _build_ui(self):
        pad = dict(padx=10, pady=8)

        # Diagnostics
        diag_txt = (
            f"BASE_DIR: {self.base_dir}\n"
            f"GeoParquet: {asset_group_parquet(self.base_dir)}\n"
            f"Exists: {asset_group_parquet(self.base_dir).exists()}"
        )
        top = (tb.LabelFrame(self.root, text="Diagnostics", bootstyle=INFO)
               if tb else ttk_std.LabelFrame(self.root, text="Diagnostics"))
        top.grid(row=0, column=0, columnspan=4, sticky="ew", **pad)
        diag = scrolledtext.ScrolledText(top, height=4)
        diag.insert(tk.END, diag_txt)
        diag.configure(state="disabled")
        diag.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        about = (tb.LabelFrame(self.root, text="About", bootstyle=INFO)
                 if tb else ttk_std.LabelFrame(self.root, text="About"))
        about.grid(row=1, column=0, columnspan=4, sticky="ew", **pad)
        tk.Label(
            about,
            text=("Give asset groups nicer display names here.\n"
                  "‘GIS name’ is the internal, system-generated name and cannot be edited."),
            justify="left", wraplength=640
        ).pack(anchor="w", padx=10, pady=8)

        tk.Label(self.root, text="GIS name").grid(row=2, column=0, sticky="w", **pad)
        tk.Label(self.root, textvariable=self.var_name_gis, relief="sunken", anchor="w", width=60)\
            .grid(row=2, column=1, columnspan=3, sticky="ew", **pad)

        entry_cls = (tb.Entry if tb else ttk_std.Entry)
        tk.Label(self.root, text="Original name").grid(row=3, column=0, sticky="w", **pad)
        entry_cls(self.root, textvariable=self.var_name_original, width=62)\
            .grid(row=3, column=1, columnspan=3, sticky="ew", **pad)

        tk.Label(self.root, text="Title (for presentation)").grid(row=4, column=0, sticky="w", **pad)
        entry_cls(self.root, textvariable=self.var_title_fromuser, width=62)\
            .grid(row=4, column=1, columnspan=3, sticky="ew", **pad)

        tk.Label(self.root, textvariable=self.var_counter).grid(row=5, column=0, sticky="w", **pad)

        nav = tk.Frame(self.root); nav.grid(row=5, column=1, columnspan=3, sticky="e", **pad)
        def _btn(widget, text, style, cmd):
            if tb: return tb.Button(widget, text=text, bootstyle=style, command=cmd)
            return ttk_std.Button(widget, text=text, command=cmd)
        _btn(nav, "⟵ Previous", PRIMARY, lambda: self._navigate(-1)).pack(side="left", padx=4)
        _btn(nav, "Save", SUCCESS, self._save_current).pack(side="left", padx=4)
        _btn(nav, "Save & Next ⟶", PRIMARY, self._save_and_next).pack(side="left", padx=4)
        _btn(nav, "Exit", WARNING, self.root.destroy).pack(side="left", padx=4)

        self.root.grid_columnconfigure(1, weight=1)

    def _update_counter(self):
        total = len(self.df)
        self.var_counter.set(f"{(self.idx + 1) if total else 0} / {total}")

    def _load_record(self):
        self.idx = max(0, min(self.idx, max(0, len(self.df) - 1)))
        self._update_counter()
        if len(self.df) == 0:
            self.var_name_gis.set(""); self.var_name_original.set(""); self.var_title_fromuser.set(""); return
        row = self.df.iloc[self.idx]
        self.var_name_gis.set(str(row.get("name_gis_assetgroup", "") or ""))
        self.var_name_original.set(str(row.get("name_original", "") or ""))
        self.var_title_fromuser.set(str(row.get("title_fromuser", "") or ""))

    def _write_back_to_df(self):
        if len(self.df) == 0: return
        self.df.at[self.idx, "name_original"] = (self.var_name_original.get() or "").strip()
        self.df.at[self.idx, "title_fromuser"] = (self.var_title_fromuser.get() or "").strip()

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
        if len(self.df) == 0: return
        self._write_back_to_df()
        self.idx = max(0, min(self.idx + step, len(self.df) - 1))
        self._load_record()

# ------------------------ Entrypoint ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edit asset-group titles (GeoParquet)")
    parser.add_argument("--original_working_directory", required=False, help="Path to running folder")
    args = parser.parse_args()

    BASE_DIR = resolve_base_dir(args.original_working_directory)
    _ensure_cfg()
    cfg_used = _CFG_PATH if _CFG_PATH else config_path(BASE_DIR)

    theme = _ensure_cfg()["DEFAULT"].get("ttk_bootstrap_theme", "flatly")
    increment_stat_value(cfg_used, 'mesa_stat_edit_asset_group', 1)

    app = (tb.Window(themename=theme) if tb else tk.Tk())
    editor = AssetGroupEditor(app, BASE_DIR, theme)
    app.mainloop()
