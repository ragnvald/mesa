# -*- coding: utf-8 -*-
"""Shared utilities for MESA helper scripts.

This module centralises functions that were previously duplicated across every
helper (find_base_dir, read_config, parquet_dir).  New helpers should import
from here; existing helpers can migrate incrementally.

Intentionally stdlib-only so it is always safe to import early, before
optional GIS dependencies are loaded.
"""

from __future__ import annotations

import configparser
import os
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
