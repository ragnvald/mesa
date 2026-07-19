# -*- coding: utf-8 -*-
"""Shared utilities for MESA helper scripts.

This module centralises functions that were previously duplicated across every
helper (find_base_dir, read_config, parquet_dir).  New helpers should import
from here; existing helpers can migrate incrementally.

Intentionally stdlib-only so it is always safe to import early, before
optional GIS dependencies are loaded.
"""

from __future__ import annotations

import base64
import configparser
import os
import re
import sys
from pathlib import Path
from typing import Optional


def ensure_bundled_geo_data() -> None:
    """Point PROJ/GDAL at the data bundled with pyproj/pyogrio, so a machine-wide
    ``PROJ_LIB`` or ``GDAL_DATA`` from another install (PostgreSQL/PostGIS, QGIS,
    OSGeo4W) cannot feed MESA an incompatible ``proj.db`` — the cause of
    "DATABASE.LAYOUT.VERSION.MINOR ... a number >= 6 is expected. It comes from
    another PROJ installation." Idempotent and fully guarded; must run before
    geopandas/pyproj/rasterio load. Uses importlib to locate the packages without
    importing them (keeps this stdlib-only module light and avoids early PROJ init).
    """
    import importlib.util
    try:
        spec = importlib.util.find_spec("pyproj")
        if spec and spec.origin:
            proj_dir = os.path.join(os.path.dirname(spec.origin), "proj_dir", "share", "proj")
            if os.path.exists(os.path.join(proj_dir, "proj.db")):
                os.environ["PROJ_DATA"] = proj_dir
                os.environ["PROJ_LIB"] = proj_dir  # legacy name still read by some GDAL builds
    except Exception:
        pass
    try:
        for _pkg in ("pyogrio", "rasterio"):
            spec = importlib.util.find_spec(_pkg)
            if spec and spec.origin:
                gd = os.path.join(os.path.dirname(spec.origin), "gdal_data")
                if os.path.isdir(gd):
                    os.environ["GDAL_DATA"] = gd
                    break
    except Exception:
        pass


# Run at import: helpers import mesa_shared before geopandas, so this repairs a
# polluted PROJ/GDAL environment before the GIS stack initialises.
ensure_bundled_geo_data()


def find_base_dir(cli_arg: Optional[str] = None) -> Path:
    """Locate the MESA project root in all execution modes.

    Search order:
    1. ``MESA_BASE_DIR`` environment variable (set by mesa.py launcher).
    2. ``cli_arg`` (value passed via ``--workdir`` or similar CLI flag).
    3. Directory of the frozen executable (PyInstaller onedir/onefile).
    4. Directory of the calling script (``__file__``).
    5. Current working directory.
    6. Walking up 5 levels from each candidate looking for a directory that
       contains *all three* of ``config.ini``, ``output/``, and ``input/``.

    Returns the best match or the first candidate if no match is found.
    """
    candidates: list[Path] = []

    env_base = os.environ.get("MESA_BASE_DIR")
    if env_base:
        candidates.append(Path(env_base))
    if cli_arg:
        candidates.append(Path(cli_arg))

    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent)

    try:
        frame = sys._getframe(1)
        caller_file = frame.f_globals.get("__file__")
        if caller_file:
            here = Path(caller_file).resolve()
            candidates.append(here.parent)
            candidates.append(here.parent.parent)
    except Exception:
        pass

    candidates.append(Path(os.getcwd()).resolve())

    def _normalize(p: Path) -> Path:
        """Walk up to the project root if *p* is inside a known sub-folder."""
        try:
            p = p.resolve()
        except Exception:
            pass
        if p.name.lower() in ("tools", "system", "code"):
            p = p.parent
        q = p
        for _ in range(5):
            if (q / "config.ini").exists() and (q / "output").exists() and (q / "input").exists():
                return q
            q = q.parent
        return p

    seen: set[Path] = set()
    for candidate in candidates:
        root = _normalize(candidate)
        try:
            key = root.resolve()
        except Exception:
            key = root
        if key in seen:
            continue
        seen.add(key)
        if (root / "config.ini").exists():
            return root

    # Fallback: return normalised first candidate
    return _normalize(candidates[0]) if candidates else Path.cwd()


def read_config(base_dir: Path) -> configparser.ConfigParser:
    """Read ``config.ini`` from *base_dir* (flat layout), falling back to the
    legacy ``system/config.ini`` location.

    Uses ``inline_comment_prefixes=(';', '#')`` and ``strict=False`` to
    tolerate the hand-edited config files used in this project.
    """
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    for candidate in (base_dir / "config.ini", base_dir / "system" / "config.ini"):
        if candidate.exists():
            try:
                cfg.read(candidate, encoding="utf-8")
            except Exception:
                cfg.read(candidate)
            break
    # config.ini holds the version-controlled defaults; a per-project settings
    # table (seeded from it, written by Tune processing) overrides them when
    # present. Absent table -> config.ini stays authoritative (the fallback for
    # helpers run before mesa.py has seeded anything).
    apply_settings_overlay(cfg, base_dir)
    return cfg


# ---------------------------------------------------------------------------
# Settings store (config.ini = defaults; tbl_settings = live per-project store)
# ---------------------------------------------------------------------------
# Design: config.ini ships the version-controlled defaults. At runtime they are
# overlaid by a per-project key/value table (output/geoparquet/tbl_settings.parquet)
# that Tune processing writes to. read_config() applies the overlay, so every
# existing `cfg["DEFAULT"].get(...)` call site honours it with no change.
#
# Stdlib-import-safety: this module stays stdlib-only, so pyarrow is imported
# lazily *inside* these functions — and only when the table actually exists. An
# untuned project (or a helper run before mesa.py seeds anything) pays just a
# Path.exists() and falls back to config.ini. Results are cached per process,
# keyed on the file mtime, so repeated read_config() calls stay cheap.
_SETTINGS_OVERLAY_CACHE: dict = {}


def settings_table_path(base_dir: Path) -> Path:
    """Path to the per-project settings store (parquet key/value table)."""
    return Path(base_dir) / "output" / "geoparquet" / "tbl_settings.parquet"


def read_settings_overlay(base_dir: Path) -> dict:
    """Return {key: value} from the settings table, or {} if absent/unreadable.

    Cached per process on the file mtime; lazy-imports pyarrow only when the
    table exists. Any failure returns {} so config.ini remains authoritative.
    """
    try:
        p = settings_table_path(base_dir)
        if not p.exists():
            return {}
        mtime = p.stat().st_mtime
        key = str(p)
        cached = _SETTINGS_OVERLAY_CACHE.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        import pyarrow.parquet as pq  # lazy: only when a table exists
        tbl = pq.read_table(p, columns=["key", "value"])
        keys = tbl.column("key").to_pylist()
        vals = tbl.column("value").to_pylist()
        out = {str(k): ("" if v is None else str(v)) for k, v in zip(keys, vals) if k is not None}
        _SETTINGS_OVERLAY_CACHE[key] = (mtime, out)
        return out
    except Exception:
        return {}


def apply_settings_overlay(cfg: configparser.ConfigParser, base_dir: Path) -> configparser.ConfigParser:
    """Overlay the settings table onto cfg['DEFAULT'] in place and return cfg.

    No-op when the table is absent (config.ini stays authoritative) — this is the
    standalone-helper fallback. Never raises.
    """
    try:
        overlay = read_settings_overlay(base_dir)
        if overlay:
            for k, v in overlay.items():
                try:
                    cfg.set("DEFAULT", str(k), str(v))
                except Exception:
                    pass
    except Exception:
        pass
    return cfg


def write_settings(base_dir: Path, updates: dict) -> bool:
    """Upsert key/value pairs into the settings table (created if absent).

    Returns True on success. Lazy-imports pyarrow. Atomic write via temp+replace.
    """
    if not updates:
        return False
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        merged = dict(read_settings_overlay(base_dir))
        for k, v in updates.items():
            merged[str(k)] = "" if v is None else str(v)
        keys = list(merged.keys())
        vals = [merged[k] for k in keys]
        tbl = pa.table({"key": keys, "value": vals})
        p = settings_table_path(base_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        pq.write_table(tbl, tmp)
        os.replace(tmp, p)
        _SETTINGS_OVERLAY_CACHE.pop(str(p), None)  # invalidate cache
        return True
    except Exception:
        return False


def mesa_version_label(cfg: configparser.ConfigParser) -> str:
    """Return a filesystem-safe MESA version string from config, or ``"dev"``.

    Reads ``mesa_version`` (preferred) or ``version`` from ``[DEFAULT]`` and
    normalises spaces to underscores so it is safe in a User-Agent string.
    """
    try:
        default = cfg["DEFAULT"] if "DEFAULT" in cfg else {}
        for option in ("mesa_version", "version"):
            value = default.get(option)  # type: ignore[union-attr]
            if value:
                cleaned = str(value).strip().replace(" ", "_")
                if cleaned:
                    return cleaned
    except Exception:
        pass
    return "dev"


def parquet_dir(base_dir: Path, cfg: Optional[configparser.ConfigParser] = None) -> Path:
    """Return (and create if needed) the GeoParquet output directory.

    Respects the ``parquet_folder`` key under ``[DEFAULT]`` in *cfg*, falling
    back to ``output/geoparquet``.
    """
    rel = "output/geoparquet"
    if cfg is not None:
        try:
            if "DEFAULT" in cfg:
                rel = str(cfg["DEFAULT"].get("parquet_folder", rel)).strip() or rel
        except Exception:
            pass
    out = (base_dir / rel).resolve()
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return out


# ---------------------------------------------------------------------------
# Leaflet vendor inlining
#
# All map views in MESA load HTML inline via pywebview (`html=...`), which
# means the document has no base URL and relative paths to JS/CSS/images do
# not resolve. We therefore inline the vendored Leaflet bundle as <style>
# and <script> blocks, with image references rewritten to data: URLs so the
# whole UI loads with zero network access.
# ---------------------------------------------------------------------------

_LEAFLET_BUNDLE_CACHE: dict[str, "LeafletBundle"] = {}

_IMAGE_MIME = {
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
}


def choose_primary_geocode(available_groups, *, prefer: str = "basic_mosaic") -> str:
    """Pick the geocode group that downstream consumers should treat as the
    project's primary analytical unit.

    When `prefer` (default `basic_mosaic`) is in `available_groups`, return it.
    Otherwise return the first sorted group name — sorted lexicographically,
    which puts H3_R6 < H3_R7 < … < H3_R10 ahead of arbitrary imported set
    names. That matches operator intuition that the coarsest-resolution H3
    grid covers the most area and is the safest fallback when the
    asset-shaped mosaic is absent.

    Returns an empty string when `available_groups` is empty so callers can
    short-circuit cleanly without raising.
    """
    if not available_groups:
        return ""
    groups = [str(g).strip() for g in available_groups if str(g).strip()]
    if prefer and prefer in groups:
        return prefer
    return sorted(groups)[0] if groups else ""


class LeafletBundle:
    """Inline-ready Leaflet + Leaflet-Draw assets, plus an offline banner.

    Two ready-to-paste fragments:

    - ``head_block``: drop inside ``<head>`` (CSS + JS for Leaflet[+draw],
      banner CSS, and the online/offline wiring script).
    - ``body_open``: drop right after ``<body>`` (the banner ``<div>``).
    """

    def __init__(self, head_block: str, body_open: str) -> None:
        self.head_block = head_block
        self.body_open = body_open


def _vendor_leaflet_dir(base_dir: Optional[Path] = None) -> Path:
    """Resolve the vendored Leaflet folder, searching common layouts."""
    candidates: list[Path] = []
    if base_dir is not None:
        candidates.append(Path(base_dir) / "system_resources" / "vendor" / "leaflet")
    here = Path(__file__).resolve().parent
    candidates.extend([
        here.parent / "system_resources" / "vendor" / "leaflet",
        here / "system_resources" / "vendor" / "leaflet",
    ])
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.extend([
            exe_dir / "system_resources" / "vendor" / "leaflet",
            exe_dir.parent / "system_resources" / "vendor" / "leaflet",
        ])
    for cand in candidates:
        if cand.is_dir() and (cand / "leaflet.js").is_file():
            return cand
    # Fallback to the first candidate so the FileNotFoundError below names a useful path.
    return candidates[0]


def _inline_css_images(css_text: str, images_dir: Path) -> str:
    """Rewrite ``url(...)`` references in *css_text* to base64 data URLs.

    Only rewrites references whose path is ``images/<filename>`` (Leaflet's
    own convention). Anything else (e.g. ``url(#default#VML)``) is left alone.
    """

    pattern = re.compile(r"url\(\s*(['\"]?)images/([^'\")\s]+)\1\s*\)")

    def _repl(match: "re.Match[str]") -> str:
        filename = match.group(2)
        path = images_dir / filename
        if not path.is_file():
            return match.group(0)
        mime = _IMAGE_MIME.get(path.suffix.lower(), "application/octet-stream")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"url('data:{mime};base64,{encoded}')"

    return pattern.sub(_repl, css_text)


def leaflet_bundle(base_dir: Optional[Path] = None, *, include_draw: bool = True) -> LeafletBundle:
    """Return inline-ready Leaflet assets, cached per (base_dir, include_draw).

    Pass ``include_draw=False`` for views that don't need the line/polygon
    editing controls — saves ~70 KB of inlined JS.
    """
    cache_key = f"{base_dir or ''}|{int(include_draw)}"
    cached = _LEAFLET_BUNDLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    vendor = _vendor_leaflet_dir(base_dir)
    images = vendor / "images"

    leaflet_css = _inline_css_images((vendor / "leaflet.css").read_text(encoding="utf-8"), images)
    leaflet_js = (vendor / "leaflet.js").read_text(encoding="utf-8")

    head_parts = [f"<style>\n{leaflet_css}\n</style>"]
    script_parts = [f"<script>\n{leaflet_js}\n</script>"]

    if include_draw:
        draw_css = _inline_css_images(
            (vendor / "leaflet.draw.css").read_text(encoding="utf-8"), images
        )
        draw_js = (vendor / "leaflet.draw.js").read_text(encoding="utf-8")
        head_parts.append(f"<style>\n{draw_css}\n</style>")
        script_parts.append(f"<script>\n{draw_js}\n</script>")

    # Banner shown when the WebView reports offline. Kept stylistically neutral
    # so it works regardless of the host page's theme.
    banner_css = """
<style>
#mesa-offline-banner {
  position: fixed; top: 0; left: 0; right: 0; z-index: 10000;
  display: none; padding: 6px 12px;
  background: #b95c00; color: #fff;
  font: 12px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
  text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,.25);
}
#mesa-offline-banner.is-visible { display: block; }
body.mesa-has-offline-banner { padding-top: 28px; }
</style>
""".strip()
    head_parts.append(banner_css)

    banner_html = (
        '<div id="mesa-offline-banner" role="status" aria-live="polite">'
        'Offline - showing cached map tiles only. Pan to areas you have viewed online before.'
        '</div>'
    )

    banner_script = """
<script>
(function () {
  function applyState() {
    var el = document.getElementById('mesa-offline-banner');
    if (!el) return;
    var offline = (typeof navigator !== 'undefined') && navigator.onLine === false;
    el.classList.toggle('is-visible', offline);
    document.body.classList.toggle('mesa-has-offline-banner', offline);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', applyState);
  } else {
    applyState();
  }
  window.addEventListener('online', applyState);
  window.addEventListener('offline', applyState);
})();
</script>
""".strip()

    head_block = "\n".join(head_parts + script_parts + [banner_script])
    bundle = LeafletBundle(head_block=head_block, body_open=banner_html)
    _LEAFLET_BUNDLE_CACHE[cache_key] = bundle
    return bundle
