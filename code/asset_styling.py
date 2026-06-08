#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""asset_styling.py — shared asset cartography styling (no data load).

Promoted from asset_map_view.py so the unified Maps window (combined_map.py)
can offer the same "AI styling" feature without importing asset_map_view (which
reads the whole asset table at import). Generates visually distinct per-group
styles and persists them to tbl_asset_group.parquet's `styling` column.

OpenAI note: asset_map_view uses OpenAI when a key is configured and otherwise
generates local distinct styles. This module ships the local generator (the
path that runs when no key is set); callers can layer an OpenAI path on top.
"""
from __future__ import annotations

import colorsys
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_STYLE_PAYLOAD: Dict[str, Any] = {
    "fill_color": "#9fa4b0",
    "border_color": "#2c3342",
    "fill_opacity": 0.65,
    "border_weight": 1.2,
}


def _rgb_to_hex(rgb) -> str:
    r, g, b = rgb
    return "#%02X%02X%02X" % (
        max(0, min(255, int(round(r * 255.0)))),
        max(0, min(255, int(round(g * 255.0)))),
        max(0, min(255, int(round(b * 255.0)))),
    )


def generate_distinct_styles(group_ids) -> Dict[str, Dict[str, Any]]:
    """Local, visually distinct styles — golden-ratio hue dispersion, tuned for
    readability. Randomised per call but unique across the requested set."""
    ids = [str(g) for g in (group_ids or [])]
    if not ids:
        return {}
    rnd = random.Random()
    hue0 = rnd.random()
    step = 0.618033988749895  # golden ratio conjugate
    out: Dict[str, Dict[str, Any]] = {}
    for idx, gid in enumerate(ids):
        hue = (hue0 + idx * step) % 1.0
        fill = colorsys.hsv_to_rgb(hue, 0.62, 0.85)
        border = colorsys.hsv_to_rgb(hue, 0.78, 0.42)
        s = dict(DEFAULT_STYLE_PAYLOAD)
        s["fill_color"] = _rgb_to_hex(fill)
        s["border_color"] = _rgb_to_hex(border)
        out[gid] = s
    return out


def _style_to_json(style: Optional[Dict[str, Any]]) -> Optional[str]:
    if not style:
        return None
    payload = {k: v for k, v in style.items() if v not in (None, "")}
    return json.dumps(payload, ensure_ascii=False) if payload else None


def persist_styles(asset_group_path, style_updates: Dict[str, Optional[Dict[str, Any]]]) -> bool:
    """Write the `styling` column in tbl_asset_group.parquet for the given ids.
    style_updates maps id(str) -> style dict (or None to clear)."""
    import pandas as pd
    p = Path(asset_group_path)
    if not style_updates or not p.exists():
        return False
    try:
        df = pd.read_parquet(p)
    except Exception:
        return False
    if "styling" not in df.columns:
        df["styling"] = pd.NA
    ids = df["id"].astype(str)
    changed = False
    for gid, style in style_updates.items():
        mask = ids == str(gid)
        if not mask.any():
            continue
        ser = _style_to_json(style)
        df.loc[mask, "styling"] = ser if ser is not None else pd.NA
        changed = True
    if not changed:
        return False
    try:
        df.to_parquet(p, index=False, compression="zstd", compression_level=3)
        return True
    except Exception:
        try:
            df.to_parquet(p, index=False)
            return True
        except Exception:
            return False


def apply_ai_styles(asset_group_path, group_ids) -> Dict[str, Dict[str, Any]]:
    """Generate distinct styles for group_ids, persist, return {id -> style}."""
    updates = generate_distinct_styles(group_ids)
    persist_styles(asset_group_path, updates)
    return updates


def clear_styles(asset_group_path, group_ids) -> Dict[str, None]:
    """Clear styling for group_ids (reverts to the fallback colour). Returns the map."""
    updates = {str(g): None for g in (group_ids or [])}
    persist_styles(asset_group_path, updates)
    return updates
