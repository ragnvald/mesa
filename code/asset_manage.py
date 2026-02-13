# -*- coding: utf-8 -*-
"""Unified asset manager: import assets + edit asset-group metadata."""

from locale_bootstrap import harden_locale_for_ttkbootstrap

harden_locale_for_ttkbootstrap()

import argparse
import configparser
import datetime
import os
import tempfile
import threading
import time
import warnings
from pathlib import Path

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

try:
	import ttkbootstrap as tb
	from ttkbootstrap.constants import PRIMARY, SUCCESS, WARNING, INFO
except Exception:
	tb = None
	PRIMARY = "primary"
	SUCCESS = "success"
	WARNING = "warning"
	INFO = "info"

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


class AssetManagerApp:
	def __init__(self, root: tk.Tk, base_dir: Path):
		self.root = root
		self.base_dir = base_dir

		self.settings = load_settings()
		self.input_folder_asset = _abs_path_like(self.settings["input_folder_asset"])
		self.working_epsg = int(self.settings["working_epsg"])
		self.asset_group_file = self.settings["asset_group_file"]

		self.validate_var = tk.BooleanVar(value=bool(self.settings["import_validate_geometries"]))
		self.simplify_var = tk.BooleanVar(value=bool(self.settings["import_simplify_geometries"]))

		self.progress_var = tk.DoubleVar(value=0.0)
		self.progress_label = None
		self.log_widget = None

		self.df = pd.DataFrame()
		self.idx = 0
		self.var_name_gis = tk.StringVar(value="")
		self.var_name_original = tk.StringVar(value="")
		self.var_title_fromuser = tk.StringVar(value="")
		self.var_counter = tk.StringVar(value="0 / 0")
		self.txt_purpose = None
		self.txt_style = None
		self.edit_state_label = None
		self.summary_label = None

		self._build_ui()
		self._log_import_diagnostics()
		self._refresh_edit_data()

	def _ts(self) -> str:
		return datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")

	def _log(self, message: str, level: str = "INFO"):
		line = f"{self._ts()} [{level}] - {message}"
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
		self.root.title("Asset")
		try:
			ico = self.base_dir / "system_resources" / "mesa.ico"
			if ico.exists() and hasattr(self.root, "iconbitmap"):
				self.root.iconbitmap(str(ico))
		except Exception:
			pass

		shell = tb.Frame(self.root, padding=10) if tb is not None else ttk.Frame(self.root, padding=10)
		shell.pack(fill="both", expand=True)

		notebook = ttk.Notebook(shell)
		notebook.pack(fill="both", expand=True)

		import_tab = tb.Frame(notebook, padding=10) if tb is not None else ttk.Frame(notebook, padding=10)
		edit_tab = tb.Frame(notebook, padding=10) if tb is not None else ttk.Frame(notebook, padding=10)
		notebook.add(import_tab, text="Import assets")
		notebook.add(edit_tab, text="Edit asset groups")

		self._build_import_tab(import_tab)
		self._build_edit_tab(edit_tab)

		bottom = tb.Frame(shell) if tb is not None else ttk.Frame(shell)
		bottom.pack(fill="x", pady=(8, 0))
		self.summary_label = tb.Label(bottom, text="") if tb is not None else ttk.Label(bottom, text="")
		self.summary_label.pack(side="left")
		exit_btn = (
			tb.Button(bottom, text="Exit", bootstyle=WARNING, command=self.root.destroy)
			if tb is not None
			else ttk.Button(bottom, text="Exit", command=self.root.destroy)
		)
		exit_btn.pack(side="right")

	def _log_import_diagnostics(self):
		cfg_display = _CFG_PATH if _CFG_PATH is not None else (self.base_dir / "config.ini")
		self._log(f"BASE_DIR: {self.base_dir}")
		self._log(f"Config used: {cfg_display}")
		self._log(f"GeoParquet out: {_select_parquet_dir()}")
		self._log(f"Assets in: {self.input_folder_asset}")
		self._log(f"EPSG: {self.working_epsg}")

	def _build_import_tab(self, parent):
		log_frame = tb.LabelFrame(parent, text="Log", bootstyle=INFO) if tb is not None else ttk.LabelFrame(parent, text="Log")
		log_frame.pack(fill="both", expand=True)

		self.log_widget = scrolledtext.ScrolledText(log_frame, height=12)
		self.log_widget.pack(fill="both", expand=True)

		progress_frame = tb.Frame(parent) if tb is not None else ttk.Frame(parent)
		progress_frame.pack(fill="x", pady=(8, 0))
		progress = ttk.Progressbar(progress_frame, orient="horizontal", length=260, mode="determinate", variable=self.progress_var)
		progress.pack(side="left")
		self.progress_label = ttk.Label(progress_frame, text="0%")
		self.progress_label.pack(side="left", padx=8)

		opt = tb.Frame(parent) if tb is not None else ttk.Frame(parent)
		opt.pack(fill="x", pady=(8, 0))
		if tb is not None:
			tb.Checkbutton(opt, text="Validate geometries", variable=self.validate_var, bootstyle=PRIMARY).pack(side="left", padx=(0, 10))
			tb.Checkbutton(opt, text="Simplify geometries", variable=self.simplify_var, bootstyle=PRIMARY).pack(side="left")
		else:
			tk.Checkbutton(opt, text="Validate geometries", variable=self.validate_var).pack(side="left", padx=(0, 10))
			tk.Checkbutton(opt, text="Simplify geometries", variable=self.simplify_var).pack(side="left")

		btns = tb.Frame(parent) if tb is not None else ttk.Frame(parent)
		btns.pack(fill="x", pady=(8, 0))

		import_btn = tb.Button(btns, text="Import assets", bootstyle=PRIMARY, command=self._start_import) if tb is not None else ttk.Button(btns, text="Import assets", command=self._start_import)
		import_btn.pack(side="left")

		refresh_btn = tb.Button(btns, text="Refresh editor", command=self._refresh_edit_data) if tb is not None else ttk.Button(btns, text="Refresh editor", command=self._refresh_edit_data)
		refresh_btn.pack(side="left", padx=(8, 0))

	def _sync_import_options_to_config(self):
		try:
			cfg = _ensure_cfg()
			cfg["DEFAULT"]["import_validate_geometries"] = "true" if bool(self.validate_var.get()) else "false"
			cfg["DEFAULT"]["import_simplify_geometries"] = "true" if bool(self.simplify_var.get()) else "false"
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
		validate = bool(self.validate_var.get())
		simplify = bool(self.simplify_var.get())
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

	def _ensure_geo_gdf(self, records_or_gdf, crs_str: str) -> gpd.GeoDataFrame:
		if isinstance(records_or_gdf, gpd.GeoDataFrame):
			gdf = records_or_gdf.copy()
			if gdf.geometry.name != "geometry":
				gdf = gdf.set_geometry(gdf.geometry.name).rename_geometry("geometry")
			if gdf.crs is None and crs_str:
				gdf.set_crs(crs_str, inplace=True)
			elif str(gdf.crs) != crs_str and crs_str:
				gdf = gdf.to_crs(crs_str)
			return gdf
		df = pd.DataFrame(records_or_gdf)
		if "geometry" not in df.columns:
			df["geometry"] = gpd.GeoSeries([], dtype="geometry")
		return gpd.GeoDataFrame(df, geometry="geometry", crs=crs_str)

	def _save_parquet(self, name: str, gdf: gpd.GeoDataFrame):
		path = _parquet_path(f"{name}.parquet", for_write=True)
		gdf.to_parquet(path, index=False)
		self._log(f"Saved {name} -> {path} (rows={len(gdf)})")

	def _import_spatial_data_asset(self):
		asset_objects, asset_groups = [], []
		group_id, object_id = 1, 1
		files = self._scan_for_files("Assets", self.input_folder_asset, ("*.shp", "*.gpkg", "*.parquet"))
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
		return self._ensure_geo_gdf(asset_objects, crs), self._ensure_geo_gdf(asset_groups, crs)

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
		def _job():
			self._sync_import_options_to_config()
			self._run_import_asset()
			self.root.after(0, self._refresh_edit_data)

		threading.Thread(target=_job, daemon=True).start()

	def _build_edit_tab(self, parent):
		self.edit_state_label = tb.Label(parent, text="") if tb is not None else ttk.Label(parent, text="")
		self.edit_state_label.pack(fill="x", pady=(0, 8))

		form = tb.Frame(parent) if tb is not None else ttk.Frame(parent)
		form.pack(fill="both", expand=True)
		form.columnconfigure(1, weight=1)

		ttk.Label(form, text="GIS name").grid(row=0, column=0, sticky="w", padx=4, pady=4)
		ttk.Label(form, textvariable=self.var_name_gis).grid(row=0, column=1, sticky="w", padx=4, pady=4)
		ttk.Label(form, text="Original name").grid(row=1, column=0, sticky="w", padx=4, pady=4)
		ttk.Entry(form, textvariable=self.var_name_original).grid(row=1, column=1, sticky="ew", padx=4, pady=4)
		ttk.Label(form, text="Title (for presentation)").grid(row=2, column=0, sticky="w", padx=4, pady=4)
		ttk.Entry(form, textvariable=self.var_title_fromuser).grid(row=2, column=1, sticky="ew", padx=4, pady=4)

		ttk.Label(form, text="Layer purpose / notes").grid(row=3, column=0, sticky="nw", padx=4, pady=4)
		self.txt_purpose = scrolledtext.ScrolledText(form, height=6, wrap="word")
		self.txt_purpose.grid(row=3, column=1, sticky="ew", padx=4, pady=4)

		ttk.Label(form, text="Styling JSON (optional)").grid(row=4, column=0, sticky="nw", padx=4, pady=4)
		self.txt_style = scrolledtext.ScrolledText(form, height=5, wrap="word")
		self.txt_style.grid(row=4, column=1, sticky="ew", padx=4, pady=4)

		controls = tb.Frame(parent) if tb is not None else ttk.Frame(parent)
		controls.pack(fill="x", pady=(8, 0))
		ttk.Label(controls, textvariable=self.var_counter).pack(side="left")

		ttk.Button(controls, text="Previous", command=lambda: self._navigate(-1)).pack(side="left", padx=(12, 0))
		ttk.Button(controls, text="Next", command=lambda: self._navigate(1)).pack(side="left", padx=(6, 0))

		save_btn = tb.Button(controls, text="Save", bootstyle=SUCCESS, command=self._save_current) if tb is not None else ttk.Button(controls, text="Save", command=self._save_current)
		save_btn.pack(side="right")
		save_next_btn = tb.Button(controls, text="Save & Next", bootstyle=PRIMARY, command=self._save_and_next) if tb is not None else ttk.Button(controls, text="Save & Next", command=self._save_and_next)
		save_next_btn.pack(side="right", padx=(0, 6))
		ttk.Button(controls, text="Reload", command=self._refresh_edit_data).pack(side="right", padx=(0, 6))

	def _update_counter(self):
		total = len(self.df)
		current = (self.idx + 1) if total else 0
		self.var_counter.set(f"{current} / {total}")

	def _clear_editor(self):
		self.var_name_gis.set("")
		self.var_name_original.set("")
		self.var_title_fromuser.set("")
		if self.txt_purpose is not None:
			self.txt_purpose.delete("1.0", tk.END)
		if self.txt_style is not None:
			self.txt_style.delete("1.0", tk.END)
		self._update_counter()

	def _load_record(self):
		if len(self.df) == 0:
			self._clear_editor()
			return
		self.idx = max(0, min(self.idx, len(self.df) - 1))
		row = self.df.iloc[self.idx]
		self.var_name_gis.set(str(row.get("name_gis_assetgroup", "") or ""))
		self.var_name_original.set(str(row.get("name_original", "") or ""))
		self.var_title_fromuser.set(str(row.get("title_fromuser", "") or ""))
		if self.txt_purpose is not None:
			self.txt_purpose.delete("1.0", tk.END)
			self.txt_purpose.insert(tk.END, str(row.get(PURPOSE_COLUMN, "") or ""))
		if self.txt_style is not None:
			self.txt_style.delete("1.0", tk.END)
			self.txt_style.insert(tk.END, str(row.get(STYLING_COLUMN, "") or ""))
		self._update_counter()
		self.edit_state_label.config(text=f"Record {self.idx + 1} of {len(self.df)}")

	def _write_back_to_df(self):
		if len(self.df) == 0:
			return
		self.df.at[self.idx, "name_original"] = (self.var_name_original.get() or "").strip()
		self.df.at[self.idx, "title_fromuser"] = (self.var_title_fromuser.get() or "").strip()
		if self.txt_purpose is not None:
			self.df.at[self.idx, PURPOSE_COLUMN] = self.txt_purpose.get("1.0", tk.END).strip()
		if self.txt_style is not None:
			self.df.at[self.idx, STYLING_COLUMN] = self.txt_style.get("1.0", tk.END).strip()

	def _save_current(self) -> bool:
		if len(self.df) == 0:
			messagebox.showinfo("Nothing to save", "There are no asset groups to save.")
			return False
		self._write_back_to_df()
		ok = save_asset_group_df(self.asset_group_file, self.df)
		if ok:
			self.edit_state_label.config(text="Saved.")
			self._log("Asset group record saved from asset_manage.")
			return True
		messagebox.showerror("Save failed", "Could not write asset group GeoParquet.")
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
			self.edit_state_label.config(text="No asset groups found. Import assets first.")
		else:
			self._load_record()
		parquet_path = _parquet_path(self.asset_group_file)
		self.summary_label.config(text=f"Asset group file: {parquet_path} | rows: {len(self.df)}")


def main():
	global BASE_DIR, _CFG, _CFG_PATH, _PARQUET_OVERRIDE

	parser = argparse.ArgumentParser(description="Unified asset import + asset-group editor")
	parser.add_argument("--original_working_directory", required=False, help="Path to running folder")
	args = parser.parse_args()

	BASE_DIR = find_base_dir(args.original_working_directory)
	_CFG = None
	_CFG_PATH = None
	_PARQUET_OVERRIDE = None
	cfg = _ensure_cfg()
	theme = cfg["DEFAULT"].get("ttk_bootstrap_theme", "flatly")

	if tb is not None:
		try:
			root = tb.Window(themename=theme)
		except Exception:
			root = tb.Window(themename="flatly")
	else:
		root = tk.Tk()

	AssetManagerApp(root, BASE_DIR)
	root.mainloop()


if __name__ == "__main__":
	main()
