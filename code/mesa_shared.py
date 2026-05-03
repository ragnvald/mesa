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
    """Read ``config.ini`` from *base_dir*.

    Uses ``inline_comment_prefixes=(';', '#')`` and ``strict=False`` to
    tolerate the hand-edited config files used in this project.
    """
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    config_path = base_dir / "config.ini"
    if config_path.exists():
        cfg.read(config_path, encoding="utf-8")
    return cfg


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
