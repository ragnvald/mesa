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
import hashlib
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


# ---------------------------------------------------------------------------
# Title-aware styling (semantic heuristic + optional LLM)
# ---------------------------------------------------------------------------
# (category, keyword substrings, (hue, sat, val)) for the fill. First match wins,
# so more specific/important categories (oil) are ordered before broad ones.
_CATEGORIES = [
    ("oil",           ("oil", "petroleum", "hydrocarbon", "refiner", "pipeline", "drill", " well", "rig"), (0.07, 0.30, 0.22)),
    ("water",         ("water", "sea", "ocean", "marine", "river", "lake", "coast", "hydro", "bay", "lagoon", "delta", "aquatic"), (0.58, 0.55, 0.82)),
    ("wetland",       ("wetland", "marsh", "mangrove", "swamp", "estuar", "bog", "fen", "floodplain"), (0.46, 0.45, 0.66)),
    ("fishery",       ("fish", "aquacult", "spawn", "trawl", "fisher"), (0.50, 0.50, 0.74)),
    ("protected",     ("protect", "reserve", "conserv", "sanctuary", "national park", "park", "wildlife", "habitat"), (0.38, 0.52, 0.50)),
    ("forest",        ("forest", "wood", "tree", "vegetat", "flora", "jungle", "canopy", "savanna"), (0.33, 0.55, 0.62)),
    ("agriculture",   ("agricult", "farm", "crop", "cultivat", "plantation", "arable", "pasture", "graz", "livestock"), (0.22, 0.55, 0.78)),
    ("mining",        ("mine", "mining", "quarry", "mineral", "extraction", " ore"), (0.08, 0.45, 0.42)),
    ("energy",        ("power", "energy", "electric", "grid", "wind", "solar", "turbine"), (0.13, 0.55, 0.85)),
    ("port",          ("port", "harbour", "harbor", "shipping", "quay", "dock", "terminal", "vessel", " ship", "maritime"), (0.55, 0.22, 0.42)),
    ("infrastructure", ("road", "rail", "transport", "airport", "highway", "bridge", "infrastructure", "utility"), (0.58, 0.10, 0.48)),
    ("industry",      ("industr", "factory", "plant", "manufactur"), (0.60, 0.15, 0.40)),
    ("urban",         ("urban", "built", "building", "settlement", "city", "town", "resident", "village", "housing"), (0.08, 0.08, 0.62)),
    ("tourism",       ("tourism", "recreat", "beach", "resort", "leisure", "hotel"), (0.07, 0.60, 0.90)),
    ("cultural",      ("cultural", "heritage", "historic", "archaeolog", "sacred", "monument", "religi"), (0.78, 0.35, 0.62)),
    ("waste",         ("waste", "landfill", "pollut", "sewage", "dump", "contaminat"), (0.11, 0.35, 0.40)),
]


def _match_category(title: str):
    t = (title or "").lower()
    for name, kws, hsv in _CATEGORIES:
        if any(k in t for k in kws):
            return name, hsv
    return None, None


def _style_from_hsv(hue: float, sat: float, val: float) -> Dict[str, Any]:
    sat = max(0.0, min(1.0, sat)); val = max(0.0, min(1.0, val))
    fill = colorsys.hsv_to_rgb(hue % 1.0, sat, val)
    border = colorsys.hsv_to_rgb(hue % 1.0, min(1.0, sat * 1.2 + 0.05), max(0.0, val * 0.5))
    s = dict(DEFAULT_STYLE_PAYLOAD)
    s["fill_color"] = _rgb_to_hex(fill)
    s["border_color"] = _rgb_to_hex(border)
    return s


def generate_styles_from_titles(id_to_title) -> Dict[str, Dict[str, Any]]:
    """Deterministic, title-aware styles. Recognised categories get a semantic
    colour family (lightness spread so multiples stay distinct); unrecognised
    titles get a stable golden-ratio hue from the title. Works from titles alone."""
    items = [(str(k), str(v or "")) for k, v in (id_to_title or {}).items()]
    if not items:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    cat_n: Dict[str, int] = {}
    fb = 0
    for gid, title in items:
        _name, hsv = _match_category(title)
        if hsv is not None:
            n = cat_n.get(_name, 0); cat_n[_name] = n + 1
            hue, sat, val = hsv
            val = max(0.18, min(0.95, val + ((n % 4) - 1.5) * 0.07))  # spread lightness
            hue = hue + (n // 4) * 0.015                              # nudge hue if >4
            out[gid] = _style_from_hsv(hue, sat, val)
        else:
            seed = int(hashlib.md5((title or gid).encode("utf-8")).hexdigest()[:6], 16) / float(0xFFFFFF)
            out[gid] = _style_from_hsv((seed + fb * 0.618033988749895) % 1.0, 0.62, 0.85)
            fb += 1
    return out


# -- optional LLM path (Config -> AI connection) -----------------------------
def _read_ai_connection(base_dir) -> Dict[str, str]:
    try:
        import pandas as pd
        p = Path(base_dir) / "secrets" / "ai_connection.parquet"
        if not p.exists():
            return {}
        df = pd.read_parquet(p)
        if df is None or df.empty:
            return {}
        row = df.iloc[-1].to_dict()
        return {k: ("" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v))
                for k, v in row.items()}
    except Exception:
        return {}


def _is_hex(v: str) -> bool:
    v = (v or "").strip()
    return len(v) == 7 and v[0] == "#" and all(c in "0123456789abcdefABCDEF" for c in v[1:])


def _darken(hex_color: str, factor: float = 0.5) -> str:
    try:
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return "#%02X%02X%02X" % (int(r * factor), int(g * factor), int(b * factor))
    except Exception:
        return "#2c3342"


def _parse_color_json(raw: str) -> Dict[str, str]:
    import re
    if not raw:
        return {}
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except Exception:
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}


def _openai_chat(prompt: str, token: str, timeout: float = 40.0) -> Optional[str]:
    import urllib.request
    payload = json.dumps({"model": "gpt-4o-mini",
                          "messages": [{"role": "user", "content": prompt}],
                          "temperature": 0.2}).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data["choices"][0]["message"]["content"] or "").strip() or None


def _ollama_chat(prompt: str, url: str, model: str, timeout: float = 60.0) -> Optional[str]:
    import urllib.request
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("response") or "").strip() or None


def generate_styles_via_llm(id_to_title, base_dir) -> Dict[str, Dict[str, Any]]:
    """Ask the configured AI backend (Config -> AI connection) for a semantic hex
    colour per title. Returns {} on no-token/unreachable/parse-failure so the
    caller falls back to the local heuristic."""
    conn = _read_ai_connection(base_dir)
    if not conn:
        return {}
    items = [(str(k), str(v or "")) for k, v in (id_to_title or {}).items() if str(v or "").strip()]
    if not items:
        return {}
    titles = sorted({t for _, t in items})
    prompt = (
        "You are a GIS cartographer. For each map layer title below, pick ONE fill "
        "colour a map reader would intuitively associate with it (water=blue, "
        "forest/vegetation=green, oil/petroleum=near-black, agriculture=olive, "
        "urban=grey, etc.). Keep the colours distinct from one another. Respond "
        "with ONLY a JSON object mapping each exact title to a \"#RRGGBB\" hex "
        "string.\nTitles:\n" + "\n".join("- " + t for t in titles))
    provider = (conn.get("provider") or "openai").lower()
    token = (conn.get("openai_token") or "").strip()
    raw = None
    try:
        if provider == "openai" and token:
            raw = _openai_chat(prompt, token)
        elif provider == "ollama":
            raw = _ollama_chat(prompt,
                               conn.get("ollama_url") or "http://localhost:11434/api/generate",
                               conn.get("ollama_model") or "mistral")
    except Exception:
        return {}
    mapping = _parse_color_json(raw or "")
    if not mapping:
        return {}
    norm = {k.strip().lower(): v for k, v in mapping.items()}
    out: Dict[str, Dict[str, Any]] = {}
    for gid, title in items:
        hexv = mapping.get(title) or norm.get(title.strip().lower())
        if hexv and _is_hex(hexv):
            s = dict(DEFAULT_STYLE_PAYLOAD)
            s["fill_color"] = hexv.upper()
            s["border_color"] = _darken(hexv)
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


def apply_ai_styles(asset_group_path, group_ids, *, titles=None, base_dir=None):
    """Title-aware AI styles, persisted. Preference order:
    1. LLM (Config -> AI connection token) when titles + base_dir are available,
    2. local semantic heuristic from titles,
    3. index-based golden-ratio (no titles, last resort).
    Returns (updates: {id -> style}, mode: 'llm' | 'heuristic' | 'index')."""
    id_to_title = {str(k): str(v or "") for k, v in (titles or {}).items()}
    updates: Dict[str, Dict[str, Any]] = {}
    mode = "index"
    if id_to_title and base_dir:
        updates = generate_styles_via_llm(id_to_title, base_dir)
        if updates:
            mode = "llm"
            missing = {k: v for k, v in id_to_title.items() if k not in updates}
            if missing:  # LLM skipped some titles — fill them from the heuristic
                updates.update(generate_styles_from_titles(missing))
    if not updates and id_to_title:
        updates = generate_styles_from_titles(id_to_title)
        mode = "heuristic"
    if not updates:
        updates = generate_distinct_styles(group_ids)
        mode = "index"
    persist_styles(asset_group_path, updates)
    return updates, mode


def clear_styles(asset_group_path, group_ids) -> Dict[str, None]:
    """Clear styling for group_ids (reverts to the fallback colour). Returns the map."""
    updates = {str(g): None for g in (group_ids or [])}
    persist_styles(asset_group_path, updates)
    return updates
