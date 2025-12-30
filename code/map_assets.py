#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive asset layer viewer dedicated to per-group asset overlays."""

from __future__ import annotations

import base64
import colorsys
import configparser
import hashlib
import io
import json
import locale
import os
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from urllib import error as urlerror
from urllib import request as urlrequest

import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping

try:
    import webview  # type: ignore
except ModuleNotFoundError:
    sys.stderr.write(
        "ERROR: 'pywebview' is not installed in the Python environment launching map_assets.py.\n"
        "Install it in that environment, e.g.:  pip install pywebview\n"
    )
    sys.exit(1)

try:
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
except Exception:
    pass


def base_dir() -> Path:
    """Resolve the Mesa repo root regardless of how the script is launched."""

    candidates: List[Path] = []
    try:
      owdir = globals().get("original_working_directory")  # type: ignore[name-defined]
      if owdir:
        candidates.append(Path(owdir))
    except Exception:
      pass
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent)
    else:
        if "__file__" in globals():
            candidates.append(Path(__file__).resolve().parent)
    candidates.append(Path(os.getcwd()).resolve())

    def normalize(p: Path) -> Path:
        p = p.resolve()
        if p.name.lower() in {"tools", "system", "code"}:
            if not ((p / "config.ini").exists() or (p / "output").exists()):
                p = p.parent
        q = p
        for _ in range(4):
            if (q / "output").exists() and (q / "input").exists():
                return q
            if (q / "tools").exists() and (q / "config.ini").exists():
                return q
            code_candidate = q / "code"
            if code_candidate.exists() and (code_candidate / "config.ini").exists():
                return code_candidate
            q = q.parent
        if (p / "config.ini").exists():
            return p
        code_alt = p / "code"
        if code_alt.exists() and (code_alt / "config.ini").exists():
            return code_alt
        return p

    for candidate in candidates:
        root = normalize(candidate)
        if (root / "tools").exists() or ((root / "output").exists() and (root / "input").exists()):
            return root
    return normalize(candidates[0])


SCRIPT_DIR = Path(__file__).resolve().parent
APP_DIR = base_dir()
os.chdir(APP_DIR)

CONFIG_FILE = APP_DIR / "config.ini"
FALLBACK_CONFIG_FILE: Optional[Path] = None
if APP_DIR.name.lower() == "code":
  parent_config = APP_DIR.parent / "config.ini"
  if parent_config.exists():
    FALLBACK_CONFIG_FILE = parent_config
OUTPUT_DIR = APP_DIR / "output"
PARQUET_DIR = OUTPUT_DIR / "geoparquet"
ASSET_OBJECT_FILE = PARQUET_DIR / "tbl_asset_object.parquet"
ASSET_GROUP_FILE = PARQUET_DIR / "tbl_asset_group.parquet"
ASSET_HIERARCHY_FILE = PARQUET_DIR / "tbl_asset_hierarchy.parquet"
LOG_FILE = SCRIPT_DIR / "log.txt"
STYLE_QUERY_FILE = SCRIPT_DIR / "style_query.txt"

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"

STYLE_VALUE_LIMITS = {
  "fill_opacity": (0.1, 0.9),
  "border_weight": (0.3, 3.0),
}

GROUP_METADATA: Dict[str, Dict[str, Any]] = {}
GROUP_NAME_LOOKUP: Dict[str, str] = {}
GROUP_ID_SET: set[str] = set()

DEFAULT_STYLE_PAYLOAD: Dict[str, Any] = {
    "fill_color": "#9fa4b0",
    "border_color": "#2c3342",
    "fill_opacity": 0.65,
    "border_weight": 1.2,
}


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
  r, g, b = rgb
  ri = max(0, min(255, int(round(r * 255.0))))
  gi = max(0, min(255, int(round(g * 255.0))))
  bi = max(0, min(255, int(round(b * 255.0))))
  return f"#{ri:02X}{gi:02X}{bi:02X}"


def _generate_distinct_style_payloads(group_ids: List[str]) -> Dict[str, Dict[str, Any]]:
  """Generate local, visually distinct styles.

  Used as a fallback when an OpenAI key is not configured.
  """

  if not group_ids:
    return {}

  # Randomise per invocation, but keep colors unique across the requested set.
  rnd = random.Random()
  hue0 = rnd.random()
  step = 0.618033988749895  # golden ratio conjugate for good dispersion

  updates: Dict[str, Dict[str, Any]] = {}
  for idx, gid in enumerate(group_ids):
    hue = (hue0 + idx * step) % 1.0

    # Tuned for readability against dark basemap (#0f172a).
    # Fill: fairly bright but not neon; Border: darker, slightly more saturated.
    fill_rgb = colorsys.hsv_to_rgb(hue, 0.62, 0.85)
    border_rgb = colorsys.hsv_to_rgb(hue, 0.78, 0.42)

    style = _default_style_payload()
    style["fill_color"] = _rgb_to_hex(fill_rgb)
    style["border_color"] = _rgb_to_hex(border_rgb)
    updates[gid] = style

  return updates


def _cfg_default_get(config: Optional[configparser.ConfigParser], option: str) -> Optional[str]:
  if not config or "DEFAULT" not in config:
    return None
  try:
    value = config["DEFAULT"].get(option)
  except Exception:
    return None
  return value.strip() if value else None


def _derive_obfuscation_key() -> bytes:
  current_cfg = globals().get("cfg")
  fallback_cfg = globals().get("FALLBACK_CFG")
  seed = _cfg_default_get(current_cfg, "id_uuid") or _cfg_default_get(fallback_cfg, "id_uuid")
  if not seed:
    seed = os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "mesa"
  payload = (str(seed) + "|openai-style-salt").encode("utf-8")
  return hashlib.sha256(payload).digest()


def _xor_bytes(data: bytes, key: bytes) -> bytes:
  if not data or not key:
    return data
  key_len = len(key)
  return bytes(b ^ key[idx % key_len] for idx, b in enumerate(data))


def _obfuscate_secret(value: str) -> str:
  key = _derive_obfuscation_key()
  payload = value.encode("utf-8")
  obfuscated = _xor_bytes(payload, key)
  return "ENC::" + base64.b64encode(obfuscated).decode("ascii")


def _maybe_deobfuscate_secret(secret: str) -> str:
  text = secret.strip()
  if not text:
    return text
  if text.startswith("B64::"):
    try:
      return base64.b64decode(text[5:].strip()).decode("utf-8")
    except Exception:
      return text
  if text.startswith("ENC::"):
    payload = text[5:].strip()
    try:
      raw = base64.b64decode(payload)
      key = _derive_obfuscation_key()
      plain = _xor_bytes(raw, key)
      return plain.decode("utf-8")
    except Exception as exc:
      log_event(f"Failed to deobfuscate OpenAI key: {exc}")
      return ""
  return text


def _read_secret_file(path_value: Optional[str]) -> Optional[str]:
  if not path_value:
    return None
  raw = path_value.strip()
  if not raw:
    return None
  candidate = Path(raw).expanduser()
  candidates: List[Path] = [candidate]
  if not candidate.is_absolute():
    candidates.append(APP_DIR / candidate)
    if APP_DIR.name.lower() == "code":
      candidates.append(APP_DIR.parent / candidate)
  seen: set[Path] = set()
  for path in candidates:
    try:
      resolved = path.resolve()
    except Exception:
      resolved = path
    if resolved in seen:
      continue
    seen.add(resolved)
    if not resolved.exists():
      continue
    try:
      secret = resolved.read_text(encoding="utf-8").strip()
      if secret:
        return _maybe_deobfuscate_secret(secret)
    except Exception as exc:
      log_event(f"Failed to read secret file {resolved}: {exc}")
  return None
def _normalize_identifier(value: Any) -> str:
  if value is None:
    return ""
  text = str(value).strip().lower()
  return "".join(ch for ch in text if ch.isalnum())


def _refresh_group_name_lookup(records: List[Dict[str, Any]]) -> None:
  global GROUP_NAME_LOOKUP, GROUP_ID_SET
  lookup: Dict[str, str] = {}
  ids: set[str] = set()
  for rec in records:
    gid_raw = rec.get("id")
    gid = str(gid_raw).strip() if gid_raw not in (None, "", "None") else None
    if not gid:
      continue
    ids.add(gid)
    meta = GROUP_METADATA.get(gid)
    candidates = [rec.get("name")]
    if meta:
      candidates.append(meta.get("title"))
    for candidate in candidates:
      key = _normalize_identifier(candidate)
      if key and key not in lookup:
        lookup[key] = gid
  GROUP_NAME_LOOKUP = lookup
  GROUP_ID_SET = ids


def _resolve_group_identifier(identifier: Any) -> Optional[str]:
  if identifier is None:
    return None
  gid = str(identifier).strip()
  if not gid:
    return None
  if gid in GROUP_ID_SET:
    return gid
  key = _normalize_identifier(gid)
  if key and key in GROUP_NAME_LOOKUP:
    return GROUP_NAME_LOOKUP[key]
  return None


def _coerce_message_content(content: Any) -> str:
  if content is None:
    return ""
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    parts: List[str] = []
    for chunk in content:
      if isinstance(chunk, str):
        parts.append(chunk)
        continue
      if isinstance(chunk, dict):
        text_field = chunk.get("text")
        if isinstance(text_field, str):
          parts.append(text_field)
        elif isinstance(text_field, list):
          parts.extend(str(item) for item in text_field if isinstance(item, str))
    return "".join(parts)
  return str(content)


def log_event(message: str) -> None:
    try:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass


log_event(f"map_assets.py starting (APP_DIR={APP_DIR})")


def read_config(path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";",), strict=False)
    try:
        cfg.read(path, encoding="utf-8")
    except Exception:
        pass
    return cfg


def _safe_hex(value: str | None, fallback: str = "#BDBDBD") -> str:
    return (value or "").strip() or fallback


def get_color_mapping(cfg: configparser.ConfigParser) -> Dict[str, str]:
    default_unknown = _safe_hex(cfg["DEFAULT"].get("category_colour_unknown", "#BDBDBD"))
    colors: Dict[str, str] = {}
    for code in "ABCDE":
        if cfg.has_section(code):
            colors[code] = _safe_hex(cfg[code].get("category_colour", default_unknown), default_unknown)
        else:
            colors[code] = default_unknown
    colors["UNKNOWN"] = default_unknown
    return colors


def to_epsg4326(gdf: gpd.GeoDataFrame, cfg: configparser.ConfigParser) -> gpd.GeoDataFrame:
    if gdf.empty:
        if gdf.crs is None:
            return gdf.set_crs(4326, allow_override=True)
        return gdf.to_crs(4326)
    if gdf.crs is None:
        try:
            epsg = int(cfg["DEFAULT"].get("workingprojection_epsg", "4326"))
            gdf = gdf.set_crs(epsg=epsg, allow_override=True)
        except Exception:
            gdf = gdf.set_crs(4326, allow_override=True)
    try:
        return gdf.to_crs(4326)
    except Exception:
        return gdf


def bounds_to_leaflet(bounds: Tuple[float, float, float, float]) -> List[List[float]]:
    minx, miny, maxx, maxy = [float(x) for x in bounds]
    dx, dy = maxx - minx, maxy - miny
    if dx <= 0 or dy <= 0:
        pad = 0.1
        minx -= pad
        maxx += pad
        miny -= pad
        maxy += pad
    else:
        minx -= dx * 0.1
        maxx += dx * 0.1
        miny -= dy * 0.1
        maxy += dy * 0.1
    minx = max(-180.0, minx)
    maxx = min(180.0, maxx)
    miny = max(-85.0, miny)
    maxy = min(85.0, maxy)
    return [[miny, minx], [maxy, maxx]]


def gdf_to_geojson_min(gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    features: List[Dict[str, Any]] = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        props = {
            "name_asset_object": row.get("name_asset_object"),
            "id_asset_object": row.get("id"),
            "sensitivity_code_max": row.get("sensitivity_code_max") or row.get("sensitivity_code"),
            "area_km2": (float(row.get("area_m2", 0)) / 1_000_000.0) if row.get("area_m2") is not None else None,
        }
        features.append({"type": "Feature", "geometry": mapping(geom), "properties": props})
    return {"type": "FeatureCollection", "features": features}


def _normalize_hex_color(value: Any) -> Optional[str]:
  if value is None:
    return None
  text = str(value).strip()
  if not text:
    return None
  if text.startswith("#"):
    text = text[1:]
  if len(text) != 6:
    return None
  if not all(ch in "0123456789abcdefABCDEF" for ch in text):
    return None
  return f"#{text.upper()}"


def _coerce_float(value: Any) -> Optional[float]:
  if value is None:
    return None
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def _sanitize_style_payload(payload: Any) -> Optional[Dict[str, Any]]:
  if payload is None:
    return None
  if isinstance(payload, str):
    text = payload.strip()
    if not text:
      return None
    try:
      payload = json.loads(text)
    except Exception:
      return None
  if not isinstance(payload, dict):
    return None

  result: Dict[str, Any] = {}
  fill_color = _normalize_hex_color(payload.get("fill_color") or payload.get("fill") or payload.get("color"))
  border_color = _normalize_hex_color(payload.get("border_color") or payload.get("stroke_color"))
  if fill_color:
    result["fill_color"] = fill_color
  if border_color:
    result["border_color"] = border_color

  fill_opacity = _coerce_float(payload.get("fill_opacity") or payload.get("opacity"))
  if fill_opacity is not None:
    low, high = STYLE_VALUE_LIMITS["fill_opacity"]
    result["fill_opacity"] = max(low, min(high, fill_opacity))

  border_weight = _coerce_float(payload.get("border_weight") or payload.get("stroke_weight") or payload.get("weight"))
  if border_weight is not None:
    low, high = STYLE_VALUE_LIMITS["border_weight"]
    result["border_weight"] = max(low, min(high, border_weight))

  dash_array = payload.get("dash_array") or payload.get("dashArray")
  if isinstance(dash_array, str):
    result["dash_array"] = dash_array.strip() or None

  return result or None


def _default_style_payload() -> Dict[str, Any]:
  return dict(DEFAULT_STYLE_PAYLOAD)


def _style_to_json(style: Optional[Dict[str, Any]]) -> Optional[str]:
  if not style:
    return None
  payload = {k: v for k, v in style.items() if v not in (None, "")}
  if not payload:
    return None
  return json.dumps(payload, ensure_ascii=False)


def _set_group_style(group_id: str, style: Optional[Dict[str, Any]]) -> None:
  normalized = _sanitize_style_payload(style) if style else None
  meta = GROUP_METADATA.setdefault(group_id, {"title": f"Group {group_id}", "purpose": "", "styling": None})
  meta["styling"] = normalized
  for record in ASSET_LAYERS:
    if str(record.get("id")) == group_id:
      if normalized is None:
        record.pop("styling", None)
      else:
        record["styling"] = normalized
      break


def _update_group_styles_on_disk(style_updates: Dict[str, Optional[Dict[str, Any]]]) -> bool:
  if not style_updates:
    return False
  if not ASSET_GROUP_FILE.exists():
    log_event("Asset group parquet missing; cannot update styling")
    return False
  try:
    df = pd.read_parquet(ASSET_GROUP_FILE)
  except Exception as exc:
    log_event(f"Failed to read asset group parquet for styling update: {exc}")
    return False
  if "styling" not in df.columns:
    df["styling"] = pd.NA
  id_strings = df["id"].astype(str)
  changed = False
  for group_id, style in style_updates.items():
    mask = id_strings == group_id
    if not mask.any():
      continue
    serialized = _style_to_json(style)
    df.loc[mask, "styling"] = serialized if serialized is not None else pd.NA
    changed = True
  if not changed:
    return False
  try:
    df.to_parquet(ASSET_GROUP_FILE, index=False)
    return True
  except Exception as exc:
    log_event(f"Failed to persist styling updates: {exc}")
    return False


def _build_style_prompt(layers: List[Dict[str, str]]) -> str:
  header = (
    "You are an assistant that designs coordinated map styles for a dark basemap. "
    "Use cartographic cues so each thematic layer is recognisable: forest/woodland/vegetation layers should use deep greens with subtle dark speckles or dashed outlines; "
    "wetlands, rivers, lagoons, and marine habitats should use bluish palettes with pale fill and darker strokes or stippling; urban/infrastructure layers can use warmer ambers/oranges; protected areas can use muted magentas/purples. "
    "All layers must remain highly distinguishable when stacked, readable for colorblind users, and pass WCAG contrast guidelines against #0f172a. "
    "Return only JSON matching {\"layers\":[{\"group_id\":\"string\",\"fill_color\":\"#RRGGBB\",\"border_color\":\"#RRGGBB\",\"fill_opacity\":0.15-0.9,\"border_weight\":0.4-2.5,\"dash_array\":\"pattern optional\"}]}. "
    "Provide entries only for the layers listed below."
  )
  lines = [header, "", "Layers:"]
  for layer in layers:
    purpose = layer.get("purpose") or "No description provided."
    lines.append(f"- id={layer['id']} title={layer['title']} purpose={purpose}")
  lines.append("")
  lines.append("Output JSON only with the schema above.")
  return "\n".join(lines)


def _write_style_query(prompt: str) -> None:
  try:
    STYLE_QUERY_FILE.write_text(prompt, encoding="utf-8")
  except Exception as exc:
    log_event(f"Failed to write style query file: {exc}")


def _resolve_openai_key() -> Optional[str]:
  env_value = os.environ.get("OPENAI_API_KEY")
  if env_value:
    return env_value.strip()

  cfg_env_key = _cfg_default_get(cfg, "openai_api_key_env") or _cfg_default_get(FALLBACK_CFG, "openai_api_key_env")
  if cfg_env_key:
    alt_env = os.environ.get(cfg_env_key.strip())
    if alt_env:
      return alt_env.strip()

  file_setting = _cfg_default_get(cfg, "openai_api_key_file") or _cfg_default_get(FALLBACK_CFG, "openai_api_key_file")
  secret_from_file = _read_secret_file(file_setting)
  if secret_from_file:
    return secret_from_file

  # Default secret location bundled with builds
  default_secret = _read_secret_file("secrets/openai.key")
  if default_secret:
    return default_secret

  inline = _cfg_default_get(cfg, "openai_api_key") or _cfg_default_get(FALLBACK_CFG, "openai_api_key")
  return inline


def _resolve_openai_model() -> str:
  if "DEFAULT" in cfg and cfg["DEFAULT"].get("openai_style_model"):
    return cfg["DEFAULT"].get("openai_style_model")  # type: ignore[return-value]
  if FALLBACK_CFG and "DEFAULT" in FALLBACK_CFG and FALLBACK_CFG["DEFAULT"].get("openai_style_model"):
    return FALLBACK_CFG["DEFAULT"].get("openai_style_model")  # type: ignore[return-value]
  return os.environ.get("OPENAI_STYLE_MODEL", DEFAULT_OPENAI_MODEL)


def _call_openai(prompt: str, api_key: str, model: str) -> Optional[str]:
  payload = json.dumps(
    {
      "model": model,
      "temperature": 0.2,
      "response_format": {"type": "json_object"},
      "messages": [
        {
          "role": "system",
          "content": "Design distinctive, readable map styles. Respond with strict JSON only.",
        },
        {"role": "user", "content": prompt},
      ],
    }
  ).encode("utf-8")
  request = urlrequest.Request(
    OPENAI_ENDPOINT,
    data=payload,
    headers={
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}",
    },
    method="POST",
  )
  try:
    with urlrequest.urlopen(request, timeout=90) as response:
      raw = response.read().decode("utf-8")
  except urlerror.HTTPError as exc:
    detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else str(exc)
    log_event(f"OpenAI HTTP error: {detail}")
    return None
  except Exception as exc:
    log_event(f"OpenAI request failed: {exc}")
    return None
  try:
    parsed = json.loads(raw)
  except Exception as exc:
    log_event(f"Failed to parse OpenAI envelope: {exc}")
    return None
  choices = parsed.get("choices") or []
  if not choices:
    return None
  message = choices[0].get("message") or {}
  if not isinstance(message, dict):
    return None
  content = _coerce_message_content(message.get("content"))
  text = content.strip()
  if not text:
    log_event("AI style response was empty after coercion")
    return None
  preview = text.replace("\n", " ")
  if len(preview) > 240:
    preview = preview[:240] + "…"
  log_event(f"AI style response snippet={preview!r}")
  return text


def _extract_json_blob(text: str) -> Optional[str]:
  snippet = text.strip()
  if not snippet:
    return None
  if snippet.startswith("```"):
    parts = snippet.split("```")
    snippet = parts[1] if len(parts) > 1 else snippet
  start = snippet.find("{")
  end = snippet.rfind("}")
  if start == -1 or end == -1 or end <= start:
    return None
  return snippet[start : end + 1]


def _parse_style_response(content: Optional[str]) -> Dict[str, Dict[str, Any]]:
  if not content:
    return {}
  blob = _extract_json_blob(content) or content
  try:
    data = json.loads(blob)
  except Exception:
    return {}
  layers = data.get("layers") if isinstance(data, dict) else None
  if not isinstance(layers, list):
    return {}
  result: Dict[str, Dict[str, Any]] = {}
  for entry in layers:
    if not isinstance(entry, dict):
      continue
    group_id = (
      entry.get("group_id")
      or entry.get("id")
      or entry.get("layer_id")
      or entry.get("layer")
      or entry.get("name")
      or entry.get("title")
    )
    resolved = _resolve_group_identifier(group_id)
    if not resolved:
      log_event(f"Skipping AI style entry with unknown group reference: {group_id!r}")
      continue
    sanitized = _sanitize_style_payload(entry)
    if sanitized:
      result[resolved] = sanitized
  return result


def _collect_prompt_layers(group_ids: List[str]) -> List[Dict[str, str]]:
  layers: List[Dict[str, str]] = []
  for gid in group_ids:
    meta = GROUP_METADATA.get(gid)
    if not meta:
      fallback = next((rec for rec in ASSET_LAYERS if str(rec.get("id")) == gid), None)
      title = (fallback or {}).get("name") or f"Group {gid}"
      meta = {"title": str(title), "purpose": "", "styling": None}
      GROUP_METADATA[gid] = meta
    title = meta.get("title") or f"Group {gid}"
    purpose = meta.get("purpose") or ""
    layers.append({"id": gid, "title": str(title), "purpose": str(purpose)})
  return layers


def _apply_style_updates(updates: Dict[str, Optional[Dict[str, Any]]]) -> None:
  if not updates:
    return
  for gid, style in updates.items():
    _set_group_style(gid, style)
  _update_group_styles_on_disk(updates)


def _load_asset_layers(
  cfg: configparser.ConfigParser,
) -> Tuple[List[Dict[str, Any]], List[List[float]] | None, Dict[str, Dict[str, Any]]]:
    log_event("Loading asset layers from GeoParquet")
    if not ASSET_OBJECT_FILE.exists():
        sys.stderr.write("tbl_asset_object.parquet not found – nothing to display.\n")
        log_event("tbl_asset_object.parquet missing; aborting load")
        return [], None, {}

    try:
        asset_objects = gpd.read_parquet(ASSET_OBJECT_FILE)
    except Exception as exc:
        sys.stderr.write(f"Failed to read asset objects: {exc}\n")
        log_event(f"Failed to read asset objects: {exc}")
        return [], None, {}

    if asset_objects.empty or "geometry" not in asset_objects.columns:
        log_event("Asset object table empty or lacks geometry column")
        return [], None, {}

    asset_objects = to_epsg4326(asset_objects, cfg)
    asset_objects = asset_objects[asset_objects.geometry.notna()].copy()
    if asset_objects.empty:
        log_event("Asset object table empty after dropping missing geometries")
        return [], None, {}

    group_names: Dict[str, str] = {}
    global GROUP_METADATA
    GROUP_METADATA = {}
    if ASSET_GROUP_FILE.exists():
      try:
        groups_df = pd.read_parquet(ASSET_GROUP_FILE)
        if not groups_df.empty and "id" in groups_df.columns:
          for _, row in groups_df.iterrows():
            key_raw = row.get("id")
            if key_raw in (None, "", "None"):
              continue
            key = str(key_raw)
            title = (
              row.get("title_fromuser")
              or row.get("name_user")
              or row.get("name_gis")
              or f"Group {key}"
            )
            purpose_raw = row.get("purpose_description") or row.get("description") or ""
            purpose = str(purpose_raw).strip()
            styling_payload = _sanitize_style_payload(row.get("styling"))
            meta = {
              "title": str(title),
              "purpose": purpose,
              "styling": styling_payload,
            }
            GROUP_METADATA[key] = meta
            group_names[key] = meta["title"]
      except Exception as exc:
        sys.stderr.write(f"Failed to read asset groups (continuing without names): {exc}\n")
        log_event(f"Failed to read asset group names: {exc}")

    if "ref_asset_group" not in asset_objects.columns:
        sys.stderr.write("Column 'ref_asset_group' missing in asset objects; cannot build per-group layers.\n")
        log_event("ref_asset_group column missing; cannot build groups")
        return [], bounds_to_leaflet(tuple(asset_objects.total_bounds)), {}

    records: List[Dict[str, Any]] = []
    geojson_by_group: Dict[str, Dict[str, Any]] = {}
    for group_id, subset in asset_objects.groupby("ref_asset_group"):
        subset = subset.copy()
        if subset.empty:
            continue
        gid_key = str(group_id)
        meta = GROUP_METADATA.setdefault(
          gid_key,
          {"title": f"Group {gid_key}", "purpose": "", "styling": None},
        )
        display_name = meta.get("title") or group_names.get(gid_key) or f"Group {gid_key}"
        geojson = gdf_to_geojson_min(subset)
        records.append(
            {
                "id": gid_key,
                "name": display_name,
                "count": int(len(subset)),
                "bounds": bounds_to_leaflet(tuple(subset.total_bounds)),
            "styling": meta.get("styling"),
            "purpose": meta.get("purpose"),
            }
        )
        geojson_by_group[gid_key] = geojson

    records.sort(key=lambda r: r["name"].lower())
    home_bounds = bounds_to_leaflet(tuple(asset_objects.total_bounds))
    log_event(f"Prepared {len(records)} asset layers (ids={[rec['id'] for rec in records]})")
    return records, home_bounds, geojson_by_group


def _read_hierarchy() -> List[Dict[str, Any]]:
    if not ASSET_HIERARCHY_FILE.exists():
        return []
    try:
        df = pd.read_parquet(ASSET_HIERARCHY_FILE)
    except Exception as exc:
        log_event(f"Failed to read asset hierarchy: {exc}")
        return []
    nodes: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        node_id = str(row.get("node_id") or "").strip()
        if not node_id:
            continue
        parent_id_val = row.get("parent_id")
        parent_id = str(parent_id_val).strip() if parent_id_val not in (None, "", "None") else None
        ref_group_val = row.get("ref_asset_group")
        ref_group = str(ref_group_val).strip() if ref_group_val not in (None, "", "None") else None
        node_type = (row.get("node_type") or "group").strip().lower()
        nodes.append(
            {
                "node_id": node_id,
                "parent_id": parent_id,
                "node_type": node_type if node_type in {"folder", "group"} else "folder",
                "ref_asset_group": ref_group,
                "title": row.get("title") or "",
                "sort_order": int(row.get("sort_order") or 0),
            }
        )
    return nodes


def _normalize_hierarchy(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lookup = {node["node_id"]: node for node in nodes}
    for node in nodes:
      parent_id = node.get("parent_id") or None
      if parent_id:
        parent_node = lookup.get(parent_id)
        if parent_node is None or parent_node.get("node_type") != "folder":
          node["parent_id"] = None
    children: Dict[str | None, List[Dict[str, Any]]] = {}
    for node in nodes:
        parent = node.get("parent_id") or None
        children.setdefault(parent, []).append(node)
    for parent, bucket in children.items():
        bucket.sort(key=lambda item: item.get("sort_order") or 0)
        for idx, item in enumerate(bucket):
            item["sort_order"] = idx
    return nodes


def _ensure_group_hierarchy(records: List[Dict[str, Any]], nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid_group_ids = {str(rec["id"]) for rec in records}
    nodes = [
        node
        for node in nodes
        if node["node_type"] != "group" or (node.get("ref_asset_group") in valid_group_ids)
    ]
    node_ids = {node["node_id"] for node in nodes}
    existing_group_refs = {
        node.get("ref_asset_group"): node for node in nodes if node["node_type"] == "group" and node.get("ref_asset_group")
    }
    root_count = len([node for node in nodes if not node.get("parent_id")])
    for idx, rec in enumerate(records):
        gid = str(rec["id"])
        if gid in existing_group_refs:
            existing_group_refs[gid]["title"] = rec["name"]
            continue
        node_id = f"group:{gid}"
        suffix = 1
        while node_id in node_ids:
            suffix += 1
            node_id = f"group:{gid}:{suffix}"
        node = {
            "node_id": node_id,
            "parent_id": None,
            "node_type": "group",
            "ref_asset_group": gid,
            "title": rec["name"],
            "sort_order": root_count + idx,
        }
        nodes.append(node)
        node_ids.add(node_id)
    return _normalize_hierarchy(nodes)


def _write_hierarchy(nodes: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(nodes)
    df = df[["node_id", "parent_id", "node_type", "ref_asset_group", "title", "sort_order"]]
    ASSET_HIERARCHY_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ASSET_HIERARCHY_FILE, index=False)


def _sanitize_hierarchy_payload(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("Hierarchy payload must be a list")
    valid_group_ids = {str(rec["id"]) for rec in ASSET_LAYERS}
    nodes: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("node_id") or "").strip()
        if not node_id or node_id in seen:
            continue
        node_type = (item.get("node_type") or "folder").strip().lower()
        if node_type not in {"folder", "group"}:
            node_type = "folder"
        parent_id_val = item.get("parent_id")
        parent_id = str(parent_id_val).strip() if parent_id_val not in (None, "", "None") else None
        ref_group_val = item.get("ref_asset_group")
        ref_group = (
            str(ref_group_val).strip() if ref_group_val not in (None, "", "None") else None
        )
        if node_type == "group":
            if ref_group not in valid_group_ids:
                continue
        else:
            ref_group = None
        nodes.append(
            {
                "node_id": node_id,
                "parent_id": parent_id,
                "node_type": node_type,
                "ref_asset_group": ref_group,
                "title": (item.get("title") or "").strip(),
                "sort_order": int(item.get("sort_order") or 0),
            }
        )
        seen.add(node_id)
    nodes = _normalize_hierarchy(nodes)
    nodes = _ensure_group_hierarchy(ASSET_LAYERS, nodes)
    return nodes


cfg = read_config(CONFIG_FILE)
FALLBACK_CFG = read_config(FALLBACK_CONFIG_FILE) if FALLBACK_CONFIG_FILE else None
COLOR_MAP = get_color_mapping(cfg)
ASSET_LAYERS, HOME_BOUNDS, ASSET_GEOJSON = _load_asset_layers(cfg)
_refresh_group_name_lookup(ASSET_LAYERS)
ASSET_HIERARCHY = _ensure_group_hierarchy(ASSET_LAYERS, _read_hierarchy())


class Api:
    def get_state(self) -> Dict[str, Any]:
        log_event(f"get_state called; {len(ASSET_LAYERS)} layers ready")
        return {
            "asset_layers": ASSET_LAYERS,
            "home_bounds": HOME_BOUNDS,
            "colors": COLOR_MAP,
            "hierarchy": ASSET_HIERARCHY,
            "style_query_file": str(STYLE_QUERY_FILE),
        }

    def get_asset_layer(self, group_id: str | int | None = None) -> Dict[str, Any]:
        if group_id is None:
            log_event("get_asset_layer called without group id")
            return {"ok": False, "error": "Missing asset group id"}
        data = ASSET_GEOJSON.get(str(group_id))
        if data is None:
            log_event(f"get_asset_layer miss for id={group_id}")
            return {"ok": False, "error": f"No asset layer found for group {group_id}"}
        log_event(f"get_asset_layer hit for id={group_id}")
        return {"ok": True, "geojson": data}

    def exit_app(self) -> None:
        try:
            webview.destroy_window()
        except Exception:
            os._exit(0)

    def save_png(self, data_url: str) -> Dict[str, Any]:
        try:
            if "," in data_url:
                _, payload = data_url.split(",", 1)
            else:
                payload = data_url
            data = base64.b64decode(payload)
            win = webview.windows[0]
            target = win.create_file_dialog(
                webview.FileDialog.SAVE,
                save_filename="asset_map.png",
                file_types=("PNG Files (*.png)",),
            )
            if not target:
                return {"ok": False, "error": "User cancelled"}
            if isinstance(target, (list, tuple)):
                target = target[0]
            with open(target, "wb") as handle:
                handle.write(data)
            return {"ok": True, "path": target}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def save_hierarchy(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        global ASSET_HIERARCHY
        try:
            normalized = _sanitize_hierarchy_payload(nodes)
            _write_hierarchy(normalized)
            ASSET_HIERARCHY = normalized
            log_event(f"Saved asset hierarchy ({len(normalized)} nodes)")
            return {"ok": True}
        except Exception as exc:
            log_event(f"Failed to save asset hierarchy: {exc}")
            return {"ok": False, "error": str(exc)}

    def generate_ai_styles(self, payload: Dict[str, Any]) -> Dict[str, Any]:
      group_ids_raw = (payload or {}).get("group_ids", [])
      if not isinstance(group_ids_raw, list):
        group_ids_raw = [group_ids_raw]
      requested = []
      seen: set[str] = set()
      valid_ids = {str(rec.get("id")) for rec in ASSET_LAYERS}
      for raw in group_ids_raw:
        gid = str(raw)
        if gid and gid in valid_ids and gid not in seen:
          requested.append(gid)
          seen.add(gid)
      if not requested:
        return {"ok": False, "error": "No active layers supplied"}
      api_key = _resolve_openai_key()
      if not api_key:
        log_event(f"OpenAI API key missing; generating local random styles for groups={requested}")
        updates = _generate_distinct_style_payloads(requested)
        _apply_style_updates(updates)
        return {
          "ok": True,
          "styles": updates,
          "prompt_file": None,
          "raw": None,
          "fallback_groups": [],
          "mode": "random",
        }
      model = _resolve_openai_model()
      prompt_layers = _collect_prompt_layers(requested)
      prompt = _build_style_prompt(prompt_layers)
      _write_style_query(prompt)
      log_event(f"Requesting AI styles for groups={requested}")
      response_text = _call_openai(prompt, api_key, model)
      if not response_text:
        return {"ok": False, "error": "No response from OpenAI"}
      parsed = _parse_style_response(response_text)
      if not parsed:
        preview = response_text[:200].replace("\n", " ") if isinstance(response_text, str) else str(response_text)
        log_event(f"AI style response unparsable; preview={preview!r}")
      updates: Dict[str, Optional[Dict[str, Any]]] = {}
      missing: List[str] = []
      for gid in requested:
        style = parsed.get(gid)
        if not style:
          style = _default_style_payload()
          missing.append(gid)
        updates[gid] = style
      if missing:
        log_event(f"AI response missing groups {missing}; applied default styling")
      _apply_style_updates(updates)
      log_event(f"Applied AI styles to groups={requested}")
      return {
        "ok": True,
        "styles": updates,
        "prompt_file": str(STYLE_QUERY_FILE),
        "raw": response_text,
        "fallback_groups": missing,
        "mode": "openai",
      }

    def clear_styles(self, payload: Dict[str, Any]) -> Dict[str, Any]:
      group_ids_raw = (payload or {}).get("group_ids", [])
      if not isinstance(group_ids_raw, list):
        group_ids_raw = [group_ids_raw]
      requested = []
      seen: set[str] = set()
      for raw in group_ids_raw:
        gid = str(raw)
        if gid and gid not in seen:
          requested.append(gid)
          seen.add(gid)
      if not requested:
        return {"ok": False, "error": "No layers specified"}
      updates = {gid: None for gid in requested}
      _apply_style_updates(updates)
      log_event(f"Cleared styles for groups={requested}")
      return {"ok": True, "cleared": requested, "styles": updates}


def _ensure_stdio_utf8() -> None:
    try:
        encoding = os.environ.get("PYTHONIOENCODING") or getattr(sys.stdout, "encoding", None) or "utf-8"
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding=encoding, errors="replace")
            sys.stderr.reconfigure(encoding=encoding, errors="replace")
        else:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=encoding, errors="replace", line_buffering=True)
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding=encoding, errors="replace", line_buffering=True)
    except Exception:
        pass


_ensure_stdio_utf8()
api = Api()


HTML_TEMPLATE = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Asset Layers</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
<style>
  html, body { height:100%; margin:0; }
  .wrap {
    height:100%;
    display:grid;
    grid-template-columns: 338px 1fr;
    grid-template-rows: 48px 1fr;
    grid-template-areas:
      "bar bar"
      "layers map";
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    background:#0f172a;
    color:#e2e8f0;
  }
  .bar { grid-area: bar; display:flex; gap:12px; align-items:center; padding:8px 12px; flex-wrap:wrap; border-bottom: 2px solid #1f2b46; background:#111f38; }
  .layers { grid-area: layers; border-right:2px solid #1f2b46; background:#0b1222; display:flex; flex-direction:column; }
  .layer-header { padding:12px 16px; font-size:14px; font-weight:600; border-bottom:1px solid #1f2b4666; }
  .layer-header .layer-actions { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .layer-list { flex:1; overflow:auto; padding:6px 8px 10px; }
  .layer-item { display:flex; gap:8px; align-items:center; padding:6px 6px; cursor:pointer; margin-bottom:2px; border-radius:5px; }
  .layer-item:hover { background:#132445; }
  .layer-item input { margin:0; }
  .layer-name { flex:1; font-size:13px; }
  .layer-tree { display:flex; flex-direction:column; gap:4px; }
  .tree-node { font-size:13px; }
  .tree-row { display:flex; align-items:center; gap:6px; padding:4px 6px; border-radius:6px; position:relative; }
  .tree-node.folder .tree-row { font-weight:600; color:#f8fafc; }
  .tree-node.group .tree-row { color:#e2e8f0; }
  .tree-row:hover { background:#132445; }
  .tree-label { flex:1; }
  .tree-children { margin-left:14px; border-left:1px solid #1f2b46; padding-left:10px; }
  .drag-handle { cursor:grab; color:#9aa8d9; user-select:none; font-size:14px; }
  .drag-handle:active { cursor:grabbing; }
  .folder-toggle { width:18px; height:18px; border:none; background:transparent; color:#cbd5f5; cursor:pointer; padding:0; }
  .folder-toggle:focus { outline:none; }
  .tree-row.drop-before::before,
  .tree-row.drop-after::after { content:""; position:absolute; left:6px; right:6px; border-top:2px solid #7aa2ff; }
  .tree-row.drop-before::before { top:0; }
  .tree-row.drop-after::after { bottom:0; }
  .tree-row.drop-inside { background:#1c2e57; }
  .btn-xs { padding:4px 8px; font-size:12px; }
  .layer-footer { border-top:1px solid #1f2b46; padding:12px 14px; font-size:13px; }
  .base-option { display:flex; gap:8px; align-items:center; padding:6px 0; }
  .base-option input { margin:0; }
  .map { grid-area: map; position:relative; background:#0b1222; }
  #map { position:absolute; inset:0; }
  #map.exporting .leaflet-control-zoom { display:none !important; }
  .btn { padding:6px 10px; border:1px solid #354769; background:#1f2b46; border-radius:6px; cursor:pointer; color:#f8fafc; font-size:13px; }
  .btn:active { transform:translateY(1px); }
  .btn[disabled] { opacity:0.45; cursor:not-allowed; }
  .slider { display:flex; align-items:center; gap:8px; }
  .slider input[type=range]{ width:160px; }
  .bar .spacer { flex:1 1 auto; }
  .ai-status { font-size:12px; min-height:18px; color:#94a3b8; }
  .ai-status.error { color:#fca5a5; }
  .leaflet-control-layers { font-size:12px; }
  .leaflet-control-layers label { line-height:1.3; }
  .leaflet-popup-content-wrapper { border-radius:10px; }
  .leaflet-popup-content { font-size:13px; color:#111827; }
  .popup strong { display:block; font-size:14px; margin-bottom:4px; }
  .empty-state { position:absolute; inset:16px; border:1px dashed #2f3b59; border-radius:14px; padding:24px; color:#cbd5f5; background:rgba(9,15,30,0.92); text-align:center; font-size:15px; line-height:1.4; display:flex; align-items:center; justify-content:center; opacity:0; pointer-events:none; transition:opacity 0.25s ease; }
  .empty-state.show { opacity:1; pointer-events:auto; }
  .legend-control { background:rgba(7,12,24,0.92); color:#e2e8f0; padding:12px 14px; border-radius:14px; box-shadow:0 12px 30px rgba(5,8,15,0.65); min-width:220px; border:1px solid #192642; backdrop-filter:blur(6px); }
  .legend-control .legend-title { font-size:13px; letter-spacing:0.08em; text-transform:uppercase; color:#94a3b8; margin-bottom:8px; display:flex; align-items:center; justify-content:space-between; }
  .legend-control .legend-body { display:flex; flex-direction:column; gap:8px; }
  .legend-item { display:flex; gap:10px; align-items:flex-start; }
  .legend-swatch { width:28px; height:20px; border-radius:6px; border:2px solid #1e293b; flex-shrink:0; box-shadow:inset 0 0 0 1px rgba(0,0,0,0.2); }
  .legend-meta { flex:1; }
  .legend-name { font-size:13px; font-weight:600; color:#f1f5f9; line-height:1.3; }
  .legend-count { font-size:12px; color:#94a3b8; }
  .legend-empty { font-size:12px; color:#94a3b8; line-height:1.4; }
  .legend-foot { font-size:11px; text-transform:uppercase; letter-spacing:0.08em; color:#7dd3fc; border-top:1px solid #1f2b46; padding-top:6px; margin-top:2px; }
  .legend-title button.legend-toggle { background:transparent; border:1px solid #1f2b46; color:#7dd3fc; border-radius:999px; width:26px; height:26px; cursor:pointer; display:flex; align-items:center; justify-content:center; font-size:13px; transition:transform 0.2s ease, color 0.2s ease; }
  .legend-title button.legend-toggle:hover { color:#f8fafc; border-color:#2dd4bf; }
  .legend-control.collapsed { padding:10px 12px; }
  .legend-control.collapsed .legend-title { margin-bottom:0; }
  .legend-control.collapsed .legend-body { display:none; }
  .legend-control.collapsed .legend-toggle { transform:rotate(180deg); }
</style>
</head>
<body>
<div class="wrap">
  <div class="bar">
    <button id="homeBtn" class="btn">Home</button>
    <div class="slider">
      <span>Opacity</span>
      <input id="opacity" type="range" min="20" max="100" value="85">
      <span id="opacityValue">85%</span>
    </div>
    <button id="aiStyleBtn" class="btn" title="Create styles for all active layers">Create styles</button>
    <button id="clearStyleBtn" class="btn" title="Remove saved styles from active layers">Clear styles</button>
    <div class="spacer"></div>
    <span id="aiStatus" class="ai-status"></span>
    <button id="exportBtn" class="btn" title="Export current map view to PNG">Export PNG</button>
    <button id="exitBtn" class="btn">Exit</button>
  </div>
  <div class="layers">
    <div class="layer-header" style="display:flex; align-items:center; justify-content:flex-end; gap:8px;">
      <div class="layer-actions">
        <button id="selectAllLayers" class="btn btn-xs" title="Activate every layer">Select all</button>
        <button id="clearAllLayers" class="btn btn-xs" title="Deactivate every layer">Clear all</button>
        <button id="addFolderBtn" class="btn btn-xs" title="Create a folder to organise layers">New Folder</button>
      </div>
    </div>
    <div class="info-block" style="padding:10px 16px; font-size:12px; color:#94a3b8; border-bottom:1px solid #1f2b4666;">
      <div style="color:#fcd34d; font-weight:600; margin-bottom:6px;">Showing original assets (no generalised geometries).</div>
      <div>Drag layers to reorder, drop them onto folders to organise, and use checkboxes to toggle visibility. Folders are saved between sessions.</div>
    </div>
    <div id="layerControls" class="layer-list layer-tree"></div>
    <div class="layer-footer">
      <div style="font-weight:600; margin-bottom:6px;">Base map</div>
      <div id="baseControls"></div>
    </div>
  </div>
  <div class="map">
    <div id="map"></div>
    <div id="emptyStateOverlay" class="empty-state">
      <div>
        <div style="font-size:18px; font-weight:600; color:#f1f5f9; margin-bottom:8px;">No asset data found</div>
        <div>Return to the Mesa window, open the <strong>Activities</strong> tab, and click the <strong>Import</strong> button to load asset data before using Asset maps.</div>
      </div>
    </div>
  </div>
</div>

<script>
let MAP=null;
let BASE_SOURCES=null;
let CURRENT_BASE_KEY=null;
let CURRENT_BASE_LAYER=null;
let COLOR_MAP={};
let FILL_ALPHA=0.85;
let HOME_BOUNDS=null;
let ASSET_LAYERS=[];
let GROUP_LAYERS=[];
let LAYER_BY_GROUP=new Map();
let ACTIVE_GROUPS=new Set();
let HIERARCHY_FLAT=[];
let TREE_ROOTS=[];
let NODE_LOOKUP=new Map();
let DRAG_STATE={ nodeId:null, dropRow:null, dropPosition:null, dropTarget:null };
let PYWEBVIEW_API=null;
let API_PROMISE=null;
let STYLE_QUERY_PATH=null;
let AI_BUSY=false;
let LEGEND_CONTROL=null;
let LEGEND_BODY=null;
let LEGEND_TOGGLE=null;
let LEGEND_COLLAPSED=false;
const BASE_OPACITY=0.85;
const LEGEND_MAX_ITEMS=6;
const LEGEND_STORAGE_KEY='mesaLegendCollapsed';
const DEFAULT_STYLE=Object.freeze({
  fill_color:'#9fa4b0',
  border_color:'#2c3342',
  fill_opacity:0.65,
  border_weight:1.2,
  point_radius:8,
});

function waitForApi(timeoutMs=12000){
  if (PYWEBVIEW_API){
    return Promise.resolve(PYWEBVIEW_API);
  }
  if (window.pywebview && window.pywebview.api){
    PYWEBVIEW_API = window.pywebview.api;
    return Promise.resolve(PYWEBVIEW_API);
  }
  if (API_PROMISE){
    return API_PROMISE;
  }
  API_PROMISE = new Promise((resolve, reject) => {
    let settled = false;
    const onReady = () => {
      if (settled) return;
      settled = true;
      if (window.pywebview && window.pywebview.api){
        PYWEBVIEW_API = window.pywebview.api;
        resolve(PYWEBVIEW_API);
      } else {
        reject(new Error('pywebview API unavailable after ready event'));
      }
    };
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      reject(new Error('pywebview API not ready'));
    }, timeoutMs);
    window.addEventListener('pywebviewready', () => {
      clearTimeout(timer);
      onReady();
    }, { once:true });
  }).finally(() => {
    API_PROMISE = null;
  });
  return API_PROMISE;
}

function uuidv4(){
  if (window.crypto && window.crypto.randomUUID){
    return window.crypto.randomUUID();
  }
  return 'node-' + Math.random().toString(16).slice(2);
}

function escapeHtml(str){
  if (str === undefined || str === null) return '';
  return String(str).replace(/[&<>\"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function clamp(value, min, max){
  if (!Number.isFinite(value)){
    return min;
  }
  return Math.min(max, Math.max(min, value));
}

function computeLayerStyle(meta){
  const styling = meta && meta.styling ? meta.styling : null;
  const fillColor = (styling && styling.fill_color) || DEFAULT_STYLE.fill_color;
  const borderColor = (styling && styling.border_color) || DEFAULT_STYLE.border_color;
  const fillCandidate = Number(styling && styling.fill_opacity);
  const baseFill = Number.isFinite(fillCandidate) ? fillCandidate : DEFAULT_STYLE.fill_opacity;
  const fillOpacity = clamp(baseFill * (FILL_ALPHA / BASE_OPACITY), 0.05, 0.95);
  const weightCandidate = Number(styling && styling.border_weight);
  const borderWeight = clamp(Number.isFinite(weightCandidate) ? weightCandidate : DEFAULT_STYLE.border_weight, 0.3, 5.0);
  const radiusCandidate = Number(styling && styling.point_radius);
  const pointRadius = clamp(
    Number.isFinite(radiusCandidate) ? radiusCandidate : (DEFAULT_STYLE.point_radius || 4),
    1,
    20
  );
  const dashArray = styling && styling.dash_array ? String(styling.dash_array) : null;
  return {
    color: borderColor,
    weight: borderWeight,
    dashArray: dashArray || null,
    fillColor,
    fillOpacity,
    opacity: 0.9,
    radius: pointRadius,
  };
}

function layerStyleFactory(meta){
  return () => computeLayerStyle(meta);
}

function reapplyLayerStyle(layer){
  if (layer && typeof layer.setStyle === 'function'){
    layer.setStyle(computeLayerStyle(layer.__meta || {}));
  }
}

function bindFeature(feature, layer, groupMeta){
  const props = feature && feature.properties ? feature.properties : {};
  const layerName = (groupMeta && (groupMeta.name || groupMeta.title || groupMeta.id)) || 'Asset layer';
  const featureName = props.name_asset_object || props.id_asset_object || null;
  let html = '<div class="popup"><strong>'+escapeHtml(String(layerName))+'</strong>';
  if (featureName && featureName !== layerName){
    html += '<div>Feature: '+escapeHtml(featureName)+'</div>';
  }
  if (props.sensitivity_code_max || props.sensitivity_code){
    html += '<div>Code: '+escapeHtml(props.sensitivity_code_max || props.sensitivity_code)+'</div>';
  }
  if (props.area_km2){
    html += '<div>Area: '+Number(props.area_km2).toLocaleString('en-US',{maximumFractionDigits:2})+' km²</div>';
  }
  html += '</div>';
  layer.bindPopup(html);
}

function buildBaseSources(){
  const common = { maxZoom:19, crossOrigin:true, tileSize:256 };
  const osm = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    ...common,
    attribution:'© OpenStreetMap contributors'
  });
  const topo = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
    ...common,
    subdomains:['a','b','c'],
    maxZoom:17,
    attribution:'© OpenStreetMap, © OpenTopoMap (CC-BY-SA)'
  });
  const esri = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    ...common,
    attribution:'© Esri, Maxar, Earthstar Geographics'
  });
  return { 'OpenStreetMap': osm, 'OSM Topography': topo, 'Satellite (ESRI)': esri };
}

function createAssetLayer(entry){
  const geoLayer = L.geoJSON(null, {
    pane: 'assetsPane',
    renderer: L.canvas({pane:'assetsPane'}),
    style: layerStyleFactory(entry),
    pointToLayer: (feature, latlng) => {
      const style = computeLayerStyle(entry);
      return L.circleMarker(latlng, style);
    },
    onEachFeature: (feature, layer) => bindFeature(feature, layer, entry),
  });
  geoLayer.__meta = entry;
  geoLayer.__loaded = false;
  geoLayer.__pending = null;
  return geoLayer;
}

function ensureLayerData(layer){
  if (!layer) return Promise.resolve(null);
  if (layer.__loaded) return Promise.resolve(layer);
  if (layer.__pending) return layer.__pending;
  const meta = layer.__meta || {};
  layer.__pending = waitForApi().then(api => {
    if (!api || !api.get_asset_layer){
      throw new Error('Asset layer API not available.');
    }
    return api.get_asset_layer(meta.id);
  }).then(res => {
    if (!res || !res.ok || !res.geojson){
      throw new Error((res && res.error) || 'Layer data missing');
    }
    layer.clearLayers();
    layer.addData(res.geojson);
    layer.__loaded = true;
    return layer;
  }).catch(err => {
    console.error('Failed to load asset layer', err);
    throw err;
  }).finally(() => {
    layer.__pending = null;
  });
  return layer.__pending;
}

function setLayerActive(layer, isActive, options){
  if (!layer) return;
  const opts = options || {};
  if (isActive){
    ensureLayerData(layer).then(() => {
      if (!MAP.hasLayer(layer)){
        layer.addTo(MAP);
      }
      syncLayerStacking();
      if (opts.fitBounds !== false){
        const gid = layer.__meta && (layer.__meta.id || layer.__meta.ref_asset_group);
        fitMapToActiveBounds(gid ? [String(gid)] : []);
      }
    }).catch(err => {
      console.error('Layer activation failed', err);
      const gid = layer.__meta && (layer.__meta.id || layer.__meta.ref_asset_group);
      if (gid){
        ACTIVE_GROUPS.delete(String(gid));
        refreshLayerCheckboxes();
      }
    });
  } else if (MAP.hasLayer(layer)) {
    MAP.removeLayer(layer);
    syncLayerStacking();
    if (opts.fitBounds !== false){
      fitMapToActiveBounds();
    }
  } else if (opts.fitBounds !== false) {
    fitMapToActiveBounds();
  }
}

function prepareLayerCollection(assetLayers){
  GROUP_LAYERS = [];
  LAYER_BY_GROUP = new Map();
  ACTIVE_GROUPS = new Set();
  (assetLayers || []).forEach(entry => {
    const layer = createAssetLayer(entry);
    GROUP_LAYERS.push(layer);
    LAYER_BY_GROUP.set(String(entry.id), layer);
  });
}

function normalizeHierarchyNodes(nodes){
  const validGroupIds = new Set((ASSET_LAYERS || []).map(layer => String(layer.id)));
  const sanitized = Array.isArray(nodes) ? nodes.map(item => ({
    node_id: String(item.node_id || uuidv4()),
    parent_id: item.parent_id ? String(item.parent_id) : null,
    node_type: item.node_type === 'folder' ? 'folder' : 'group',
    ref_asset_group: item.ref_asset_group ? String(item.ref_asset_group) : null,
    title: item.title || '',
    sort_order: Number.isFinite(item.sort_order) ? item.sort_order : 0,
    collapsed: Boolean(item.collapsed),
    children: [],
  })) : [];
  const filtered = sanitized.filter(node => node.node_type !== 'group' || validGroupIds.has(node.ref_asset_group));
  const existingGroupNodes = new Set(filtered.filter(node => node.node_type === 'group').map(node => node.ref_asset_group));
  (ASSET_LAYERS || []).forEach(entry => {
    const gid = String(entry.id);
    if (!existingGroupNodes.has(gid)){
      filtered.push({
        node_id: `group:${gid}`,
        parent_id: null,
        node_type: 'group',
        ref_asset_group: gid,
        title: entry.name,
        sort_order: filtered.length,
        collapsed: false,
        children: [],
      });
    }
  });
  return filtered;
}

function rebuildTree(){
  NODE_LOOKUP = new Map();
  TREE_ROOTS = [];
  HIERARCHY_FLAT.forEach(node => {
    node.children = [];
    NODE_LOOKUP.set(node.node_id, node);
  });
  HIERARCHY_FLAT.forEach(node => {
    const parentId = node.parent_id || null;
    if (parentId && NODE_LOOKUP.has(parentId)){
      NODE_LOOKUP.get(parentId).children.push(node);
    } else {
      node.parent_id = null;
      TREE_ROOTS.push(node);
    }
  });
  const sortChildren = list => {
    list.sort((a,b) => (a.sort_order || 0) - (b.sort_order || 0));
    list.forEach(child => sortChildren(child.children || []));
  };
  sortChildren(TREE_ROOTS);
}

function initHierarchy(nodes){
  HIERARCHY_FLAT = normalizeHierarchyNodes(nodes);
  rebuildTree();
}

function updateEmptyState(){
  const overlay = document.getElementById('emptyStateOverlay');
  const hasLayers = Array.isArray(ASSET_LAYERS) && ASSET_LAYERS.length > 0;
  if (overlay){
    overlay.classList.toggle('show', !hasLayers);
  }
}

function readLegendCollapsedPref(){
  try {
    if (typeof window !== 'undefined' && window.localStorage){
      return window.localStorage.getItem(LEGEND_STORAGE_KEY) === '1';
    }
  } catch (err) {
    console.warn('Legend preference read failed', err);
  }
  return false;
}

function writeLegendCollapsedPref(flag){
  try {
    if (typeof window !== 'undefined' && window.localStorage){
      window.localStorage.setItem(LEGEND_STORAGE_KEY, flag ? '1' : '0');
    }
  } catch (err) {
    console.warn('Legend preference write failed', err);
  }
}

function applyLegendCollapsedState(){
  const container = LEGEND_CONTROL && typeof LEGEND_CONTROL.getContainer === 'function'
    ? LEGEND_CONTROL.getContainer()
    : null;
  if (container){
    container.classList.toggle('collapsed', Boolean(LEGEND_COLLAPSED));
  }
  if (LEGEND_TOGGLE){
    LEGEND_TOGGLE.setAttribute('aria-expanded', String(!LEGEND_COLLAPSED));
    LEGEND_TOGGLE.title = LEGEND_COLLAPSED ? 'Expand legend' : 'Collapse legend';
    LEGEND_TOGGLE.textContent = LEGEND_COLLAPSED ? '▸' : '▾';
  }
}

function setLegendCollapsed(flag, options){
  const shouldPersist = !options || options.persist !== false;
  LEGEND_COLLAPSED = Boolean(flag);
  applyLegendCollapsedState();
  if (shouldPersist){
    writeLegendCollapsedPref(LEGEND_COLLAPSED);
  }
}

function toggleLegendCollapsed(){
  setLegendCollapsed(!LEGEND_COLLAPSED);
}

function initLegendControl(){
  if (!MAP || LEGEND_CONTROL){
    return;
  }
  LEGEND_CONTROL = L.control({ position:'bottomright' });
  LEGEND_CONTROL.onAdd = () => {
    const container = L.DomUtil.create('div', 'leaflet-control legend-control');
    container.innerHTML = `
      <div class="legend-title">
        <span>Legend</span>
        <button type="button" class="legend-toggle" aria-expanded="true" title="Collapse legend" aria-label="Collapse legend">▾</button>
      </div>
      <div class="legend-body"></div>
    `;
    LEGEND_BODY = container.querySelector('.legend-body');
    LEGEND_TOGGLE = container.querySelector('.legend-toggle');
    if (LEGEND_TOGGLE){
      LEGEND_TOGGLE.addEventListener('click', evt => {
        evt.preventDefault();
        evt.stopPropagation();
        toggleLegendCollapsed();
      });
    }
    L.DomEvent.disableClickPropagation(container);
    L.DomEvent.disableScrollPropagation(container);
    return container;
  };
  LEGEND_CONTROL.addTo(MAP);
  setLegendCollapsed(readLegendCollapsedPref(), { persist:false });
  updateLegend();
}

function collectLegendEntries(){
  const entries = [];
  getActiveGroupIds().forEach(id => {
    const layer = LAYER_BY_GROUP.get(String(id));
    if (!layer || !layer.__meta){
      return;
    }
    const meta = layer.__meta;
    const style = computeLayerStyle(meta);
    entries.push({
      id: String(id),
      name: meta.name || meta.title || `Layer ${id}`,
      count: Number.isFinite(meta.count) ? Number(meta.count) : null,
      style,
    });
  });
  return entries;
}

function ensureLegendBody(){
  if (LEGEND_BODY){
    return LEGEND_BODY;
  }
  if (LEGEND_CONTROL){
    const container = LEGEND_CONTROL.getContainer();
    if (container){
      LEGEND_BODY = container.querySelector('.legend-body');
    }
  }
  return LEGEND_BODY;
}

function updateLegend(){
  if (!MAP){
    return;
  }
  if (!LEGEND_CONTROL){
    initLegendControl();
  }
  const body = ensureLegendBody();
  if (!body){
    return;
  }
  const entries = collectLegendEntries();
  if (!entries.length){
    body.innerHTML = '<div class="legend-empty">Activate one or more layers to preview their symbology.</div>';
    return;
  }
  const limited = entries.slice(0, LEGEND_MAX_ITEMS);
  const fragments = limited.map(entry => {
    const style = entry.style || {};
    const fill = style.fillColor || '#9fa4b0';
    const stroke = style.color || '#2c3342';
    const countText = entry.count !== null ? `<div class="legend-count">${entry.count.toLocaleString('en-US')} objects</div>` : '';
    return `
      <div class="legend-item">
        <span class="legend-swatch" style="background:${fill}; border-color:${stroke};"></span>
        <div class="legend-meta">
          <div class="legend-name">${escapeHtml(entry.name)}</div>
          ${countText}
        </div>
      </div>
    `;
  }).join('');
  let footer = '';
  if (entries.length > LEGEND_MAX_ITEMS){
    footer = `<div class="legend-foot">+${entries.length - LEGEND_MAX_ITEMS} more layer(s)</div>`;
  }
  body.innerHTML = fragments + footer;
}

function renderLayerTree(){
  const container = document.getElementById('layerControls');
  if (!container) return;
  container.innerHTML = '';
  const hasGroupNodes = Boolean(findFirstGroupNode(TREE_ROOTS));
  if (!hasGroupNodes){
    container.innerHTML = '<p style="font-size:13px; color:#94a3b8;">No asset layers available. Return to the Mesa launcher, open the Activities tab, and click the <strong>Import</strong> button to load asset data before reopening Asset maps.</p>';
    updateEmptyState();
    updateLegend();
    return;
  }
  TREE_ROOTS.forEach(node => container.appendChild(renderTreeNode(node, 0)));
  ensureInitialActivation();
  syncLayerStacking();
  updateEmptyState();
  updateLegend();
}

function renderTreeNode(node, depth){
  const wrapper = document.createElement('div');
  wrapper.className = 'tree-node ' + node.node_type;
  const row = document.createElement('div');
  row.className = 'tree-row';
  row.dataset.nodeId = node.node_id;
  row.style.paddingLeft = (8 + depth * 14) + 'px';
  row.addEventListener('dragover', evt => handleRowDragOver(evt, node, row));
  row.addEventListener('dragleave', () => clearRowDropState(row));
  row.addEventListener('drop', handleRowDrop);

  const handle = document.createElement('span');
  handle.className = 'drag-handle';
  handle.textContent = '≡';
  handle.title = 'Drag to reorder';
  handle.draggable = true;
  handle.addEventListener('dragstart', evt => handleDragStart(evt, node, row));
  handle.addEventListener('dragend', () => clearDragState());
  row.appendChild(handle);

  if (node.node_type === 'folder'){
    const toggle = document.createElement('button');
    toggle.className = 'folder-toggle';
    toggle.textContent = node.collapsed ? '▸' : '▾';
    toggle.addEventListener('click', evt => {
      evt.stopPropagation();
      node.collapsed = !node.collapsed;
      renderLayerTree();
    });
    row.appendChild(toggle);

    const label = document.createElement('span');
    label.className = 'tree-label';
    label.textContent = node.title || 'Folder';
    row.appendChild(label);
  } else {
    row.dataset.groupId = node.ref_asset_group || '';
    const spacer = document.createElement('span');
    spacer.style.width = '18px';
    row.appendChild(spacer);

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = ACTIVE_GROUPS.has(node.ref_asset_group);
    checkbox.addEventListener('click', evt => evt.stopPropagation());
    checkbox.addEventListener('change', () => {
      const shouldFit = checkbox.checked;
      toggleGroupLayer(node.ref_asset_group, checkbox.checked, { fitBounds: shouldFit });
    });
    row.appendChild(checkbox);

    const label = document.createElement('span');
    label.className = 'tree-label';
    const layer = LAYER_BY_GROUP.get(node.ref_asset_group);
    const meta = layer ? layer.__meta : null;
    const countText = meta ? ' (' + Number(meta.count || 0).toLocaleString('en-US') + ')' : '';
    label.textContent = (meta ? meta.name : (node.title || 'Layer')) + countText;
    row.appendChild(label);
  }

  wrapper.appendChild(row);

  if (node.node_type === 'folder' && node.children && node.children.length && !node.collapsed){
    const childWrap = document.createElement('div');
    childWrap.className = 'tree-children';
    node.children.forEach(child => childWrap.appendChild(renderTreeNode(child, depth + 1)));
    wrapper.appendChild(childWrap);
  }
  return wrapper;
}

function handleDragStart(evt, node, row){
  DRAG_STATE = { nodeId: node.node_id, dropRow:null, dropPosition:null, dropTarget:null, dragRow:row };
  evt.dataTransfer.effectAllowed = 'move';
  if (evt.dataTransfer && evt.dataTransfer.setData){
    evt.dataTransfer.setData('text/plain', node.node_id);
  }
  row.classList.add('dragging');
}

function handleRowDragOver(evt, node, row){
  if (!DRAG_STATE.nodeId || DRAG_STATE.nodeId === node.node_id){
    return;
  }
  evt.preventDefault();
  if (DRAG_STATE.dropRow && DRAG_STATE.dropRow !== row){
    clearRowDropState(DRAG_STATE.dropRow);
  }
  const rect = row.getBoundingClientRect();
  const offset = evt.clientY - rect.top;
  const height = rect.height || 1;
  let position = 'inside';
  if (offset < height * 0.25){
    position = 'before';
  } else if (offset > height * 0.75){
    position = 'after';
  } else if (node.node_type !== 'folder'){
    position = offset < height / 2 ? 'before' : 'after';
  }
  if (position === 'inside' && node.node_type !== 'folder'){
    position = 'after';
  }
  clearRowDropState(row);
  if (position === 'inside'){
    row.classList.add('drop-inside');
  } else if (position === 'before'){
    row.classList.add('drop-before');
  } else {
    row.classList.add('drop-after');
  }
  DRAG_STATE.dropRow = row;
  DRAG_STATE.dropPosition = position;
  DRAG_STATE.dropTarget = node;
}

function handleRowDrop(evt){
  if (!DRAG_STATE.nodeId) return;
  evt.preventDefault();
  if (!DRAG_STATE.dropTarget){
    clearDragState();
    return;
  }
  moveNode(DRAG_STATE.nodeId, DRAG_STATE.dropTarget.node_id, DRAG_STATE.dropPosition);
  clearDragState();
}

function handleRootDragOver(evt){
  if (!DRAG_STATE.nodeId) return;
  if (evt.target.closest && evt.target.closest('.tree-row')) return;
  evt.preventDefault();
  if (DRAG_STATE.dropRow){
    clearRowDropState(DRAG_STATE.dropRow);
    DRAG_STATE.dropRow = null;
  }
  DRAG_STATE.dropTarget = null;
  DRAG_STATE.dropPosition = 'root';
}

function handleRootDrop(evt){
  if (!DRAG_STATE.nodeId) return;
  if (evt.target.closest && evt.target.closest('.tree-row')) return;
  evt.preventDefault();
  moveNode(DRAG_STATE.nodeId, null, 'root');
  clearDragState();
}

function clearRowDropState(row){
  if (!row) return;
  row.classList.remove('drop-before', 'drop-after', 'drop-inside');
}

function clearDragState(){
  if (DRAG_STATE.dropRow){
    clearRowDropState(DRAG_STATE.dropRow);
  }
  if (DRAG_STATE.dragRow){
    DRAG_STATE.dragRow.classList.remove('dragging');
  }
  DRAG_STATE = { nodeId:null, dropRow:null, dropPosition:null, dropTarget:null, dragRow:null };
}

function removeNodeFromParent(node){
  const parentId = node.parent_id;
  const siblings = parentId ? (NODE_LOOKUP.get(parentId)?.children || []) : TREE_ROOTS;
  const idx = siblings.indexOf(node);
  if (idx >= 0){
    siblings.splice(idx, 1);
  }
}

function isDescendant(targetNode, ancestorId){
  if (!targetNode) return false;
  if (targetNode.node_id === ancestorId) return true;
  let parentId = targetNode.parent_id;
  while (parentId){
    if (parentId === ancestorId) return true;
    const parent = NODE_LOOKUP.get(parentId);
    parentId = parent ? parent.parent_id : null;
  }
  return false;
}

function moveNode(nodeId, targetId, position){
  const moving = NODE_LOOKUP.get(nodeId);
  if (!moving) return;
  if (targetId){
    const target = NODE_LOOKUP.get(targetId);
    if (!target || nodeId === targetId || isDescendant(target, nodeId)){
      return;
    }
  }
  removeNodeFromParent(moving);
  if (!targetId){
    moving.parent_id = null;
    TREE_ROOTS.push(moving);
  } else {
    const target = NODE_LOOKUP.get(targetId);
    let dropPosition = position;
    if (dropPosition === 'inside' && target.node_type !== 'folder'){
      dropPosition = 'after';
    }
    if (dropPosition === 'inside' && target.node_type === 'folder'){
      target.children = target.children || [];
      target.children.push(moving);
      moving.parent_id = target.node_id;
    } else {
      const parent = target.parent_id ? NODE_LOOKUP.get(target.parent_id) : null;
      const siblings = parent ? parent.children : TREE_ROOTS;
      let insertIndex = siblings.indexOf(target);
      if (insertIndex === -1){
        insertIndex = siblings.length;
      } else if (dropPosition === 'after'){
        insertIndex += 1;
      }
      siblings.splice(insertIndex, 0, moving);
      moving.parent_id = parent ? parent.node_id : null;
    }
  }
  updateSortOrders();
  persistHierarchy();
  renderLayerTree();
}

function updateSortOrders(){
  const apply = (list, parentId) => {
    list.forEach((node, idx) => {
      node.sort_order = idx;
      node.parent_id = parentId;
      if (node.children && node.children.length){
        apply(node.children, node.node_id);
      }
    });
  };
  apply(TREE_ROOTS, null);
}

function serializeHierarchy(){
  const acc = [];
  const walk = (nodes, parentId) => {
    nodes.forEach((node, idx) => {
      acc.push({
        node_id: node.node_id,
        parent_id: parentId,
        node_type: node.node_type,
        ref_asset_group: node.node_type === 'group' ? node.ref_asset_group : null,
        title: node.node_type === 'folder' ? (node.title || '') : (node.title || ''),
        sort_order: idx,
      });
      if (node.children && node.children.length){
        walk(node.children, node.node_id);
      }
    });
  };
  walk(TREE_ROOTS, null);
  return acc;
}

function persistHierarchy(){
  const flat = serializeHierarchy();
  HIERARCHY_FLAT = flat.map(item => ({ ...item }));
  waitForApi().then(api => {
    if (api && api.save_hierarchy){
      return api.save_hierarchy(flat);
    }
    throw new Error('Hierarchy API unavailable');
  }).then(res => {
    if (!res || !res.ok){
      console.error('Failed to save hierarchy', res && res.error);
    }
  }).catch(err => {
    console.error('Failed to save hierarchy', err);
  });
}

function toggleGroupLayer(groupId, enable, options){
  const layer = LAYER_BY_GROUP.get(String(groupId));
  if (!layer) return;
  if (enable){
    ACTIVE_GROUPS.add(String(groupId));
  } else {
    ACTIVE_GROUPS.delete(String(groupId));
  }
  setLayerActive(layer, enable, options);
  updateLegend();
}

function getActiveGroupIds(){
  return Array.from(ACTIVE_GROUPS.values());
}

function collectBoundsForGroups(groupIds){
  let minLat = 90;
  let minLng = 180;
  let maxLat = -90;
  let maxLng = -180;
  let count = 0;
  (groupIds || []).forEach(id => {
    const layer = LAYER_BY_GROUP.get(String(id));
    if (!layer || !layer.__meta || !Array.isArray(layer.__meta.bounds)){
      return;
    }
    const [sw, ne] = layer.__meta.bounds;
    if (!Array.isArray(sw) || !Array.isArray(ne)){
      return;
    }
    const [south, west] = sw;
    const [north, east] = ne;
    if ([south, west, north, east].some(v => typeof v !== 'number' || Number.isNaN(v))){
      return;
    }
    minLat = Math.min(minLat, south);
    minLng = Math.min(minLng, west);
    maxLat = Math.max(maxLat, north);
    maxLng = Math.max(maxLng, east);
    count += 1;
  });
  if (!count){
    return null;
  }
  return [[minLat, minLng], [maxLat, maxLng]];
}

function fitMapToActiveBounds(extraIds){
  if (!MAP){
    return;
  }
  const ids = new Set(ACTIVE_GROUPS);
  (extraIds || []).forEach(id => {
    if (id !== undefined && id !== null){
      ids.add(String(id));
    }
  });
  const bounds = collectBoundsForGroups(Array.from(ids));
  if (bounds){
    MAP.fitBounds(bounds, { padding:[24,24] });
  } else if (HOME_BOUNDS){
    MAP.fitBounds(HOME_BOUNDS, { padding:[24,24] });
  }
}

function refreshLayerCheckboxes(){
  document.querySelectorAll('.tree-node.group .tree-row').forEach(row => {
    const checkbox = row.querySelector('input[type=checkbox]');
    const gid = row.dataset.groupId;
    if (checkbox && gid){
      checkbox.checked = ACTIVE_GROUPS.has(gid);
    }
  });
}

function setAllLayersActive(enable){
  const ids = (ASSET_LAYERS || []).map(entry => String(entry.id));
  ids.forEach(gid => toggleGroupLayer(gid, enable, { fitBounds:false }));
  refreshLayerCheckboxes();
  syncLayerStacking();
  if (enable){
    fitMapToActiveBounds(ids);
  } else if (HOME_BOUNDS){
    MAP.fitBounds(HOME_BOUNDS, { padding:[24,24] });
  } else {
    fitMapToActiveBounds();
  }
}

function handleSelectAllLayers(){
  setAllLayersActive(true);
}

function handleClearAllLayers(){
  setAllLayersActive(false);
}

function applyStyleUpdates(updates){
  if (!updates) return;
  Object.keys(updates).forEach(key => {
    const style = updates[key] || null;
    const layer = LAYER_BY_GROUP.get(String(key));
    if (layer){
      layer.__meta = layer.__meta || {};
      layer.__meta.styling = style;
      reapplyLayerStyle(layer);
    }
    const entry = ASSET_LAYERS.find(item => String(item.id) === String(key));
    if (entry){
      entry.styling = style;
    }
  });
  updateLegend();
}

function setAiStatus(message, isError){
  const node = document.getElementById('aiStatus');
  if (!node) return;
  node.textContent = message || '';
  node.classList.toggle('error', Boolean(isError));
}

function setAiBusy(isBusy){
  AI_BUSY = isBusy;
  ['aiStyleBtn','clearStyleBtn'].forEach(id => {
    const btn = document.getElementById(id);
    if (btn){
      btn.disabled = Boolean(isBusy);
    }
  });
}

function handleCreateAiStyles(){
  if (AI_BUSY) return;
  const active = getActiveGroupIds();
  if (!active.length){
    setAiStatus('Activate at least one layer before creating styles.', true);
    return;
  }
  setAiBusy(true);
  setAiStatus('Requesting styles…');
  waitForApi().then(api => {
    if (!api || !api.generate_ai_styles){
      throw new Error('AI styling API unavailable.');
    }
    return api.generate_ai_styles({ group_ids: active });
  }).then(res => {
    if (!res || !res.ok){
      throw new Error((res && res.error) || 'AI styling failed.');
    }
    applyStyleUpdates(res.styles || {});
    if (res.prompt_file){
      STYLE_QUERY_PATH = res.prompt_file;
    }
    const count = Object.keys(res.styles || {}).length;
    const fallback = Array.isArray(res.fallback_groups) ? res.fallback_groups.length : 0;
    const mode = (res && res.mode) ? String(res.mode) : 'openai';
    const promptPath = STYLE_QUERY_PATH || 'style_query.txt';
    let message = `Updated ${count} layer(s).`;
    if (mode === 'random'){
      message += ' Generated local random styles (no OpenAI key).';
    } else {
      message += ` Prompt saved to ${promptPath}.`;
      if (fallback){
        message += ` (${fallback} layer(s) used default styling.)`;
      }
    }
    setAiStatus(message);
  }).catch(err => {
    setAiStatus(err && err.message ? err.message : 'AI styling failed.', true);
  }).finally(() => {
    setAiBusy(false);
  });
}

function handleClearStyles(){
  if (AI_BUSY) return;
  const active = getActiveGroupIds();
  if (!active.length){
    setAiStatus('Activate at least one layer to clear styles.', true);
    return;
  }
  setAiBusy(true);
  setAiStatus('Clearing styles…');
  waitForApi().then(api => {
    if (!api || !api.clear_styles){
      throw new Error('Clear styles API unavailable.');
    }
    return api.clear_styles({ group_ids: active });
  }).then(res => {
    if (!res || !res.ok){
      throw new Error((res && res.error) || 'Failed to clear styles.');
    }
    applyStyleUpdates(res.styles || {});
    setAiStatus(`Cleared styles for ${active.length} layer(s).`);
  }).catch(err => {
    setAiStatus(err && err.message ? err.message : 'Failed to clear styles.', true);
  }).finally(() => {
    setAiBusy(false);
  });
}

function findFirstGroupNode(nodes){
  for (const node of nodes){
    if (node.node_type === 'group'){
      return node;
    }
    if (node.children && node.children.length){
      const nested = findFirstGroupNode(node.children);
      if (nested) return nested;
    }
  }
  return null;
}

function ensureInitialActivation(){
  if (ACTIVE_GROUPS.size) return;
  const first = findFirstGroupNode(TREE_ROOTS);
  if (first && first.ref_asset_group){
    toggleGroupLayer(first.ref_asset_group, true, { fitBounds:true });
    const checkbox = document.querySelector(`[data-node-id="${first.node_id}"] input[type=checkbox]`);
    if (checkbox){
      checkbox.checked = true;
    }
  }
}

function getGroupDisplayOrder(){
  const order = [];
  const walk = nodes => {
    nodes.forEach(node => {
      if (node.node_type === 'group' && node.ref_asset_group){
        order.push(String(node.ref_asset_group));
      }
      if (node.children && node.children.length){
        walk(node.children);
      }
    });
  };
  walk(TREE_ROOTS || []);
  return order;
}

function syncLayerStacking(){
  if (!MAP) return;
  const order = getGroupDisplayOrder();
  for (let idx = order.length - 1; idx >= 0; idx -= 1){
    const gid = order[idx];
    const layer = LAYER_BY_GROUP.get(gid);
    if (layer && MAP.hasLayer(layer) && typeof layer.bringToFront === 'function'){
      layer.bringToFront();
    }
  }
}

function handleAddFolder(){
  const name = prompt('Folder name');
  if (!name) return;
  const trimmed = name.trim();
  if (!trimmed) return;
  const node = {
    node_id: `folder:${uuidv4()}`,
    parent_id: null,
    node_type: 'folder',
    ref_asset_group: null,
    title: trimmed,
    sort_order: TREE_ROOTS.length,
    collapsed: false,
    children: [],
  };
  TREE_ROOTS.push(node);
  NODE_LOOKUP.set(node.node_id, node);
  updateSortOrders();
  persistHierarchy();
  renderLayerTree();
}

function updateOpacity(){
  GROUP_LAYERS.forEach(layer => {
    reapplyLayerStyle(layer);
  });
  updateLegend();
}

function setBaseLayer(key){
  if (!BASE_SOURCES || !BASE_SOURCES[key] || !MAP){
    return;
  }
  const layer = BASE_SOURCES[key];
  if (CURRENT_BASE_LAYER === layer){
    return;
  }
  if (CURRENT_BASE_LAYER && MAP.hasLayer(CURRENT_BASE_LAYER)){
    MAP.removeLayer(CURRENT_BASE_LAYER);
  }
  CURRENT_BASE_LAYER = layer;
  CURRENT_BASE_KEY = key;
  layer.addTo(MAP);
  renderBaseControls();
}

function renderBaseControls(){
  const container = document.getElementById('baseControls');
  if (!container) return;
  if (!BASE_SOURCES){
    container.innerHTML = '<div style="color:#94a3b8;font-size:12px;">No base maps available.</div>';
    return;
  }
  const entries = Object.keys(BASE_SOURCES);
  if (!entries.length){
    container.innerHTML = '<div style="color:#94a3b8;font-size:12px;">No base maps available.</div>';
    return;
  }
  container.innerHTML = '';
  entries.forEach(key => {
    const label = document.createElement('label');
    label.className = 'base-option';
    const input = document.createElement('input');
    input.type = 'radio';
    input.name = 'baseMap';
    input.value = key;
    input.checked = key === CURRENT_BASE_KEY;
    const text = document.createElement('div');
    text.textContent = key;
    input.addEventListener('change', () => {
      if (input.checked){
        setBaseLayer(key);
      }
    });
    label.appendChild(input);
    label.appendChild(text);
    container.appendChild(label);
  });
}

function boot(){
  MAP = L.map('map', {
    zoomControl:false,
    preferCanvas:true,
    maxBounds: L.latLngBounds([[-85,-180],[85,180]]),
    maxBoundsViscosity: 1.0,
  });
  MAP.createPane('assetsPane');
  L.control.zoom({ position:'topright' }).addTo(MAP);
  L.control.scale({ position:'bottomleft', metric:true, imperial:false }).addTo(MAP);

  BASE_SOURCES = buildBaseSources();
  const baseKeys = Object.keys(BASE_SOURCES);
  if (baseKeys.length){
    CURRENT_BASE_KEY = baseKeys[0];
    CURRENT_BASE_LAYER = BASE_SOURCES[CURRENT_BASE_KEY];
    CURRENT_BASE_LAYER.addTo(MAP);
  }
  renderBaseControls();
  initLegendControl();

  const treeContainer = document.getElementById('layerControls');
  if (treeContainer){
    treeContainer.addEventListener('dragover', handleRootDragOver);
    treeContainer.addEventListener('drop', handleRootDrop);
  }
  const addFolderBtn = document.getElementById('addFolderBtn');
  if (addFolderBtn){
    addFolderBtn.addEventListener('click', handleAddFolder);
  }
  const selectAllBtn = document.getElementById('selectAllLayers');
  if (selectAllBtn){
    selectAllBtn.addEventListener('click', handleSelectAllLayers);
  }
  const clearAllLayersBtn = document.getElementById('clearAllLayers');
  if (clearAllLayersBtn){
    clearAllLayersBtn.addEventListener('click', handleClearAllLayers);
  }

  waitForApi().then(api => {
    if (!api || !api.get_state){
      throw new Error('pywebview API missing get_state');
    }
    return api.get_state();
  }).then(state => {
    COLOR_MAP = state.colors || {};
    HOME_BOUNDS = state.home_bounds || null;
    ASSET_LAYERS = state.asset_layers || [];
    STYLE_QUERY_PATH = state.style_query_file || STYLE_QUERY_PATH;
    prepareLayerCollection(ASSET_LAYERS);
    initHierarchy(state.hierarchy || []);
    renderLayerTree();
    if (HOME_BOUNDS){
      MAP.fitBounds(HOME_BOUNDS, { padding:[24,24] });
    } else {
      MAP.setView([0,0], 2);
    }
  }).catch(err => {
    console.error('Failed to load state', err);
    const fallback = document.getElementById('layerControls');
    if (fallback){
      fallback.innerHTML = '<p style="color:#94a3b8;">Could not load asset layers.<br>' + escapeHtml(err && err.message ? err.message : String(err)) + '</p>';
    }
    updateEmptyState();
  });

  document.getElementById('homeBtn').addEventListener('click', () => {
    if (HOME_BOUNDS){
      MAP.fitBounds(HOME_BOUNDS, { padding:[24,24] });
    }
  });

  const slider = document.getElementById('opacity');
  const label = document.getElementById('opacityValue');
  slider.addEventListener('input', () => {
    FILL_ALPHA = parseInt(slider.value, 10) / 100.0;
    label.textContent = slider.value + '%';
    updateOpacity();
  });

  document.getElementById('exportBtn').addEventListener('click', () => {
    const mapNode = document.getElementById('map');
    mapNode.classList.add('exporting');
    html2canvas(mapNode, {useCORS:true, allowTaint:false, backgroundColor:'#ffffff', scale:3})
      .then(canvas => {
        mapNode.classList.remove('exporting');
        return waitForApi().then(api => {
          if (!api || !api.save_png){
            throw new Error('Save API not available');
          }
          return api.save_png(canvas.toDataURL('image/png'));
        });
      })
      .then(res => {
        if (!res.ok){
          console.error('Export failed', res.error);
        }
      })
      .catch(err => {
        mapNode.classList.remove('exporting');
        console.error('Export failed', err);
      });
  });

  document.getElementById('exitBtn').addEventListener('click', () => {
    waitForApi().then(api => {
      if (api && api.exit_app){
        api.exit_app();
      }
    }).catch(() => {});
  });

  const aiBtn = document.getElementById('aiStyleBtn');
  if (aiBtn){
    aiBtn.addEventListener('click', handleCreateAiStyles);
  }
  const clearBtn = document.getElementById('clearStyleBtn');
  if (clearBtn){
    clearBtn.addEventListener('click', handleClearStyles);
  }
  setAiStatus('');
}
if (document.readyState === 'loading'){
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
</script>
</body>
</html>
"""


def main() -> None:
    window = webview.create_window(
        title="Asset Layers",
        html=HTML_TEMPLATE,
        js_api=api,
        width=1280,
        height=820,
        resizable=True,
    )
    webview.start(gui="edgechromium", debug=False)


if __name__ == "__main__":
    main()
