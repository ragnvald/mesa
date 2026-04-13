"""
GeoNode 5 export helpers for MESA.

Converts MESA GeoParquet outputs to GeoPackage and publishes them
to a GeoNode 5 server via its REST API.

API used:
  POST /api/v2/uploads/upload/          multipart upload → execution_id
  GET  /api/v2/executionrequest/<id>/   poll until status in {finished, failed}
  GET  /api/v2/users/                   connection test (Basic auth)
  GET  /api/v2/datasets/?filter{name}=  existence check
  DELETE /api/v2/datasets/<pk>/         delete before replace
  GET  /api/v2/datasets/<pk>/styles/    get existing style pk
  PATCH /api/v2/styles/<pk>/            update SLD content
  POST /api/v2/maps/                    create empty map
  PATCH /api/v2/maps/<pk>/              add maplayers to map
  Auth: HTTP Basic (username / password)
  Server upload limit: 100 MB per dataset

Sensitivity layers:
  tbl_flat.parquet contains one row per geocode cell per geocode group
  (field: name_gis_geocodegroup).  Each group is published as its own
  GeoNode dataset, styled with the MESA A-E colour palette from config.ini.
  The SLD is saved to output/geonode_styles/ and included in the upload.

Supporting layers (optional, no style):
  tbl_analysis_flat, tbl_analysis_polygons, tbl_lines,
  tbl_segments, tbl_asset_group, tbl_asset_object
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from typing import Callable, Optional

import requests

# ---------------------------------------------------------------------------
# Sensitivity palette (fallback defaults; overridden at runtime from config.ini)
# ---------------------------------------------------------------------------

_DEFAULT_PALETTE: dict[str, str] = {
    "A": "#bd0026",
    "B": "#f03b20",
    "C": "#fd8d3c",
    "D": "#fecc5c",
    "E": "#ffffb2",
    "UNKNOWN": "#BDBDBD",
}

_DEFAULT_DESCRIPTIONS: dict[str, str] = {
    "A": "Very high",
    "B": "High",
    "C": "Moderate",
    "D": "Low",
    "E": "Very low",
}

_OVERLAY_OPACITY = 0.65   # matches report_generate overlay_alpha


# ---------------------------------------------------------------------------
# Supporting (non-sensitivity) layer catalogue  — static
# ---------------------------------------------------------------------------

SUPPORTING_LAYERS: list[dict] = [
    {
        "id": "analysis_results",
        "label": "Analysis results",
        "hint": (
            "Aggregated sensitivity per geocode zone across each study area - "
            "min and max importance, sensitivity and susceptibility scores."
        ),
        "parquet": "tbl_analysis_flat.parquet",
        "layer_name": "mesa_analysis_results",
        "default_checked": False,
        "size_note": None,
    },
    {
        "id": "study_areas",
        "label": "Study areas",
        "hint": "Analysis polygon boundaries that define each study extent.",
        "parquet": "tbl_analysis_polygons.parquet",
        "layer_name": "mesa_study_areas",
        "default_checked": False,
        "size_note": None,
    },
    {
        "id": "routes",
        "label": "Routes",
        "hint": "Line routes being assessed.",
        "parquet": "tbl_lines.parquet",
        "layer_name": "mesa_routes",
        "default_checked": False,
        "size_note": None,
    },
    {
        "id": "segments",
        "label": "Route segments",
        "hint": "Segmented route corridors with per-segment sensitivity scores.",
        "parquet": "tbl_segments.parquet",
        "layer_name": "mesa_route_segments",
        "default_checked": False,
        "size_note": None,
    },
    {
        "id": "asset_groups",
        "label": "Asset group areas",
        "hint": "Convex-hull footprints of each asset group with sensitivity ratings.",
        "parquet": "tbl_asset_group.parquet",
        "layer_name": "mesa_asset_groups",
        "default_checked": False,
        "size_note": None,
    },
    {
        "id": "assets",
        "label": "Individual assets",
        "hint": "All individual asset polygons - large dataset, slow to upload.",
        "parquet": "tbl_asset_object.parquet",
        "layer_name": "mesa_assets",
        "default_checked": False,
        "size_note": "~50 MB - may take several minutes",
    },
]

# Server upload limit in bytes
SERVER_UPLOAD_LIMIT_BYTES = 100 * 1024 * 1024


# ---------------------------------------------------------------------------
# Dynamic sensitivity layers  — one per geocode group in tbl_flat
# ---------------------------------------------------------------------------

def _safe_name(s: str) -> str:
    """Convert an arbitrary string to a safe GeoNode layer name component."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", s).lower().strip("_")


def sensitivity_layers(geoparquet_dir: str) -> list[dict]:
    """
    Scan tbl_flat.parquet and return one layer-dict per unique geocode group,
    ordered by ref_geocodegroup.  Returns [] if the file is absent.

    Each dict includes:
      id             "sensitivity:<name_gis_geocodegroup>"
      filter_field   "name_gis_geocodegroup"
      filter_value   <name_gis_geocodegroup string>
      sld_field      "sensitivity_code_max"
      row_count      number of rows in this group
      available      True
    """
    path = os.path.join(geoparquet_dir, "tbl_flat.parquet")
    if not os.path.isfile(path):
        return []

    try:
        import pyarrow.parquet as pq
        table = pq.read_table(
            path, columns=["ref_geocodegroup", "name_gis_geocodegroup"]
        )
        import pandas as pd
        df = table.to_pandas()
        groups = (
            df.groupby(["ref_geocodegroup", "name_gis_geocodegroup"])
            .size()
            .reset_index(name="rows")
            .sort_values("ref_geocodegroup")
        )
    except Exception:
        return []

    result = []
    for _, row in groups.iterrows():
        gname = str(row["name_gis_geocodegroup"])
        nrows = int(row["rows"])
        result.append({
            "id": f"sensitivity:{gname}",
            "label": gname,
            "hint": (
                f"Sensitivity map for geocode group '{gname}' ({nrows:,} cells). "
                "Coloured automatically with the MESA A-E colour scheme."
            ),
            "parquet": "tbl_flat.parquet",
            "filter_field": "name_gis_geocodegroup",
            "filter_value": gname,
            "layer_name": f"mesa_{_safe_name(gname)}",
            "default_checked": True,
            "size_note": None,
            "sld_field": "sensitivity_code_max",
            "available": True,
            "row_count": nrows,
        })
    return result


def layer_info(geoparquet_dir: str) -> dict:
    """
    Return a dict with two keys:
      "sensitivity"  - list of dicts from sensitivity_layers()
      "supporting"   - list of SUPPORTING_LAYERS dicts enriched with
                       available + row_count
    """
    import pyarrow.parquet as pq

    sens = sensitivity_layers(geoparquet_dir)

    supp = []
    for layer in SUPPORTING_LAYERS:
        path = os.path.join(geoparquet_dir, layer["parquet"])
        available = os.path.isfile(path)
        row_count = None
        if available:
            try:
                row_count = pq.read_metadata(path).num_rows
            except Exception:
                pass
        supp.append({**layer, "available": available, "row_count": row_count})

    return {"sensitivity": sens, "supporting": supp}


# ---------------------------------------------------------------------------
# Connection test
# ---------------------------------------------------------------------------

def test_connection(base_url: str, username: str, password: str) -> tuple[bool, str]:
    """
    Probe the GeoNode server and verify credentials.
    Returns (success, human-readable message).
    """
    url = base_url.rstrip("/") + "/api/v2/users/"
    try:
        r = requests.get(url, auth=(username, password), timeout=12)
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to {base_url} - check the URL."
    except requests.exceptions.Timeout:
        return False, "Connection timed out."
    except Exception as exc:
        return False, str(exc)

    if r.status_code == 200:
        try:
            users = r.json().get("users", [])
            display = users[0].get("username", username) if users else username
            return True, f"Connected as {display}. Server is reachable."
        except Exception:
            return True, "Connected."
    if r.status_code == 401:
        return False, "Authentication failed - wrong username or password."
    return False, f"Unexpected response: HTTP {r.status_code}."


# ---------------------------------------------------------------------------
# Sensitivity palette
# ---------------------------------------------------------------------------

def read_palette(config_path: Optional[str]) -> tuple[dict[str, str], dict[str, str]]:
    """
    Read the A-E colour palette and descriptions from config.ini.
    Returns (colors_map, descriptions_map).  Falls back to defaults on any error.
    """
    colors = dict(_DEFAULT_PALETTE)
    descs = dict(_DEFAULT_DESCRIPTIONS)

    if not config_path or not os.path.isfile(config_path):
        return colors, descs

    try:
        import configparser
        cfg = configparser.ConfigParser(strict=False)
        cfg.read(config_path, encoding="utf-8")
        unknown = cfg.get("VALID_VALUES", "category_colour_unknown", fallback="").strip()
        if unknown:
            colors["UNKNOWN"] = unknown
        for code in ("A", "B", "C", "D", "E"):
            if cfg.has_section(code):
                col = cfg[code].get("category_colour", "").strip()
                if col:
                    colors[code] = col
                desc = cfg[code].get("description", "").strip()
                if desc:
                    descs[code] = desc
    except Exception:
        pass

    return colors, descs


# ---------------------------------------------------------------------------
# SLD generation
# ---------------------------------------------------------------------------

def build_sensitivity_sld(
    layer_name: str,
    field: str,
    colors: dict[str, str],
    descriptions: dict[str, str],
    opacity: float = _OVERLAY_OPACITY,
) -> str:
    """
    Build an SLD 1.0.0 document that styles *field* with the A-E colour palette.
    Rules are ordered E -> D -> C -> B -> A so highest sensitivity paints last.
    """

    def _rule(code: str, label: str, color: str, last: bool = False) -> str:
        fill_op = f"{opacity:.2f}"
        filter_block = (
            "      <ElseFilter/>"
            if last else
            f"      <ogc:Filter>\n"
            f"        <ogc:PropertyIsEqualTo>\n"
            f"          <ogc:PropertyName>{field}</ogc:PropertyName>\n"
            f"          <ogc:Literal>{code}</ogc:Literal>\n"
            f"        </ogc:PropertyIsEqualTo>\n"
            f"      </ogc:Filter>"
        )
        return (
            f"    <Rule>\n"
            f"      <Name>{code}</Name>\n"
            f"      <Title>{label}</Title>\n"
            f"{filter_block}\n"
            f"      <PolygonSymbolizer>\n"
            f"        <Fill>\n"
            f"          <CssParameter name=\"fill\">{color}</CssParameter>\n"
            f"          <CssParameter name=\"fill-opacity\">{fill_op}</CssParameter>\n"
            f"        </Fill>\n"
            f"        <Stroke>\n"
            f"          <CssParameter name=\"stroke\">{color}</CssParameter>\n"
            f"          <CssParameter name=\"stroke-width\">0.0</CssParameter>\n"
            f"          <CssParameter name=\"stroke-opacity\">0.0</CssParameter>\n"
            f"        </Stroke>\n"
            f"      </PolygonSymbolizer>\n"
            f"    </Rule>"
        )

    ordered = ["E", "D", "C", "B", "A"]
    rules_xml = "\n".join(
        _rule(
            code,
            f"{code} - {descriptions.get(code, code)}",
            colors.get(code, colors.get("UNKNOWN", "#BDBDBD")),
        )
        for code in ordered
    )
    rules_xml += "\n" + _rule(
        "UNKNOWN", "No data",
        colors.get("UNKNOWN", "#BDBDBD"),
        last=True,
    )

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<StyledLayerDescriptor version="1.0.0"\n'
        '  xsi:schemaLocation="http://www.opengis.net/sld StyledLayerDescriptor.xsd"\n'
        '  xmlns="http://www.opengis.net/sld"\n'
        '  xmlns:ogc="http://www.opengis.net/ogc"\n'
        '  xmlns:xlink="http://www.w3.org/1999/xlink"\n'
        '  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
        f'  <NamedLayer>\n'
        f'    <Name>{layer_name}</Name>\n'
        f'    <UserStyle>\n'
        f'      <Title>MESA Sensitivity (maximum)</Title>\n'
        f'      <Abstract>A-E colour palette from MESA config.ini</Abstract>\n'
        f'      <FeatureTypeStyle>\n'
        f'{rules_xml}\n'
        f'      </FeatureTypeStyle>\n'
        f'    </UserStyle>\n'
        f'  </NamedLayer>\n'
        f'</StyledLayerDescriptor>\n'
    )


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _sanitize(gdf):
    """Return a copy of gdf safe to write as GeoPackage."""
    import pandas as pd

    gdf = gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    for col in list(gdf.columns):
        if col == "geometry":
            continue
        series = gdf[col]

        non_null = series.dropna()
        if len(non_null) > 0 and isinstance(non_null.iloc[0], (dict, list)):
            gdf[col] = series.apply(
                lambda x: json.dumps(x, ensure_ascii=False)
                if isinstance(x, (dict, list)) else (str(x) if x is not None else None)
            )
            continue

        if hasattr(series, "dtype") and hasattr(series.dtype, "numpy_dtype"):
            try:
                import numpy as np
                if np.issubdtype(series.dtype.numpy_dtype, np.integer):
                    gdf[col] = series.astype("float64")
                    continue
            except Exception:
                pass

        try:
            if isinstance(series.dtype, pd.StringDtype):
                gdf[col] = series.astype("object")
                continue
        except Exception:
            pass

        try:
            if pd.api.types.is_datetime64_any_dtype(series):
                gdf[col] = (
                    series.dt.strftime("%Y-%m-%dT%H:%M:%S")
                    .where(series.notna(), other=None)
                )
        except Exception:
            pass

    return gdf


def _write_gpkg(gdf, layer_name: str, directory: str) -> str:
    """Write a sanitized GeoDataFrame to a GeoPackage. Returns the file path."""
    path = os.path.join(directory, f"{layer_name}.gpkg")
    _sanitize(gdf).to_file(path, driver="GPKG", engine="pyogrio", layer=layer_name)
    return path


# ---------------------------------------------------------------------------
# Dataset existence check and deletion
# ---------------------------------------------------------------------------

def _check_existing_dataset(
    base_url: str,
    username: str,
    password: str,
    layer_name: str,
) -> Optional[dict]:
    """
    Return the GeoNode dataset dict if a dataset with this exact name exists,
    or None if not found.  Dict includes at least {pk, name, alternate}.
    """
    url = base_url.rstrip("/") + f"/api/v2/datasets/?filter{{name}}={layer_name}"
    try:
        r = requests.get(url, auth=(username, password), timeout=12)
        if r.status_code == 200:
            datasets = r.json().get("datasets", [])
            for ds in datasets:
                if ds.get("name") == layer_name or ds.get("alternate", "").endswith(f":{layer_name}"):
                    return ds
    except Exception:
        pass
    return None


def _delete_dataset(
    base_url: str,
    username: str,
    password: str,
    pk: int,
    log: Callable[[str], None],
) -> bool:
    """
    Delete a GeoNode dataset by pk via /api/v2/resources/{pk}/.
    (/api/v2/datasets/{pk}/ returns 405 — DELETE is disabled on that endpoint.)
    """
    url = base_url.rstrip("/") + f"/api/v2/resources/{pk}/"
    try:
        r = requests.delete(url, auth=(username, password), timeout=30)
        if r.status_code in (200, 204):
            log(f"  Deleted existing dataset (pk={pk}).")
            return True
        log(f"  Warning: could not delete existing dataset (HTTP {r.status_code}: {r.text[:100]}).")
    except Exception as exc:
        log(f"  Warning: delete request failed: {exc}")
    return False


# ---------------------------------------------------------------------------
# Upload + poll
# ---------------------------------------------------------------------------

_UPLOAD_PATH = "/api/v2/uploads/upload/"
_EXEC_PATH = "/api/v2/executionrequest/{exec_id}/"
_POLL_INTERVAL_S = 3.0
_POLL_TIMEOUT_S = 300.0


def _pk_from_detail_url(detail_url: str) -> Optional[int]:
    """Extract a numeric pk from a GeoNode detail URL like /catalogue/#/dataset/165."""
    m = re.search(r"/(\d+)/?$", detail_url)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None


def _upload_gpkg(
    base_url: str,
    username: str,
    password: str,
    gpkg_path: str,
    log: Callable[[str], None],
    sld_path: Optional[str] = None,
) -> tuple[bool, str, Optional[str], Optional[int]]:
    """
    POST a GeoPackage (+ optional SLD) to GeoNode and wait for processing.
    Returns (ok, message, detail_url, dataset_pk).
    """
    url = base_url.rstrip("/") + _UPLOAD_PATH
    fname = os.path.basename(gpkg_path)
    fsize_mb = os.path.getsize(gpkg_path) / (1024 * 1024)

    if os.path.getsize(gpkg_path) > SERVER_UPLOAD_LIMIT_BYTES:
        return False, (
            f"File too large ({fsize_mb:.1f} MB). "
            f"Server limit is {SERVER_UPLOAD_LIMIT_BYTES // (1024 * 1024)} MB."
        ), None, None

    style_note = " + SLD style" if sld_path and os.path.isfile(sld_path) else ""
    log(f"  Uploading {fname} ({fsize_mb:.1f} MB){style_note} ...")

    sld_fh = None
    try:
        with open(gpkg_path, "rb") as fh:
            files: dict = {
                "base_file": (fname, fh, "application/geopackage+sqlite3"),
            }
            if sld_path and os.path.isfile(sld_path):
                sld_fh = open(sld_path, "rb")
                files["sld_file"] = (
                    os.path.basename(sld_path), sld_fh,
                    "application/vnd.ogc.sld+xml",
                )
            r = requests.post(
                url,
                auth=(username, password),
                files=files,
                data={
                    "charset": "UTF-8",
                    "permissions": json.dumps({"users": {}, "groups": {}}),
                },
                timeout=180,
            )
    except requests.exceptions.Timeout:
        return False, "Upload timed out.", None, None
    except Exception as exc:
        return False, f"Upload error: {exc}", None, None
    finally:
        if sld_fh:
            try:
                sld_fh.close()
            except Exception:
                pass

    if r.status_code not in (200, 201):
        return False, f"HTTP {r.status_code}: {r.text[:200]}", None, None

    exec_id = r.json().get("execution_id")
    if not exec_id:
        return False, f"No execution_id in response: {r.text[:200]}", None, None

    log("  Waiting for server to process ...")
    return _poll(base_url, username, password, exec_id, log)


def _poll(
    base_url: str,
    username: str,
    password: str,
    exec_id: str,
    log: Callable[[str], None],
) -> tuple[bool, str, Optional[str], Optional[int]]:
    """
    Poll an execution request until finished.
    Returns (ok, message, detail_url, dataset_pk).
    """
    url = base_url.rstrip("/") + _EXEC_PATH.format(exec_id=exec_id)
    deadline = time.monotonic() + _POLL_TIMEOUT_S
    last_status = ""

    while time.monotonic() < deadline:
        try:
            r = requests.get(url, auth=(username, password), timeout=15)
        except Exception as exc:
            return False, f"Polling error: {exc}", None, None

        if r.status_code != 200:
            return False, f"Polling HTTP {r.status_code}", None, None

        req = r.json().get("request", r.json())
        status = req.get("status", "")

        if status != last_status:
            log(f"  Server status: {status}")
            last_status = status

        if status == "finished":
            resources = req.get("output_params", {}).get("resources", [])
            detail_url = None
            dataset_pk = None
            if resources:
                res = resources[0]
                raw_detail = res.get("detail_url", "")
                if raw_detail:
                    detail_url = base_url.rstrip("/") + raw_detail
                dataset_pk = res.get("pk") or res.get("id") or _pk_from_detail_url(raw_detail)
            return True, "Published successfully.", detail_url, dataset_pk

        if status in ("failed", "error", "FAILED"):
            server_log = (req.get("log") or "")[:300]
            return False, f"Server processing failed. {server_log}".rstrip(), None, None

        time.sleep(_POLL_INTERVAL_S)

    return False, "Timed out waiting for server processing.", None, None


# ---------------------------------------------------------------------------
# Style application (post-upload)
#
# Approach: OAuth2 authorization-code flow
#   1. GeoNode login  -> GeoNode sessionid cookie
#   2. GeoServer j_spring_oauth2_geonode_login -> GeoNode authorize page
#   3. POST approval  -> GeoServer callback with ?code=
#   4. GeoServer JSESSIONID set -> GeoServer REST API accessible
#   5. PUT SLD to /geoserver/rest/workspaces/geonode/styles/{name}.sld
#   6. PUT layer default style reference
#   Fallback: Django admin form (updates DB only, no GeoServer push)
# ---------------------------------------------------------------------------

def _apply_style_after_upload(
    base_url: str,
    username: str,
    password: str,
    dataset_pk: int,
    sld_xml: str,
    layer_name: str,
    log: Callable[[str], None],
) -> bool:
    """Apply SLD: try GeoServer REST proxy first, fall back to Django admin."""
    try:
        return _apply_style_with_session(
            base_url, username, password, dataset_pk, sld_xml, layer_name, log
        )
    except Exception as exc:
        log(f"  Style: unexpected error - {exc}")
        return False


def _apply_style_with_session(
    base_url: str,
    username: str,
    password: str,
    dataset_pk: int,
    sld_xml: str,
    layer_name: str,
    log: Callable[[str], None],
) -> bool:
    """
    Apply SLD via GeoServer OAuth2 session (requires GeoNode admin user).
    Falls back to Django admin form (DB only) on OAuth2/REST failure.
    """
    import re as _re

    base = base_url.rstrip("/")
    auth = (username, password)

    # ── Step 1: find the style name for this dataset ──────────────────────
    try:
        r = requests.get(
            f"{base}/api/v2/datasets/{dataset_pk}/styles/",
            auth=auth, timeout=12,
        )
    except Exception as exc:
        log(f"  Style: GET /styles/ failed ({exc}).")
        return False

    if r.status_code != 200:
        log(f"  Style: GET /styles/ returned HTTP {r.status_code}.")
        return False

    raw = r.json()
    styles = raw if isinstance(raw, list) else raw.get("styles", raw.get("results", []))
    if not styles:
        log("  Style: no styles attached to dataset yet.")
        return False

    style_pk   = styles[0].get("pk")
    style_name = styles[0].get("name", layer_name)
    if not style_pk:
        log("  Style: could not determine style pk.")
        return False

    log(f"  Style: found pk={style_pk} name='{style_name}'")

    # ── Step 2: GeoNode session ───────────────────────────────────────────
    gn_session = requests.Session()
    try:
        lp = gn_session.get(f"{base}/account/login/", timeout=10)
        login_csrf = gn_session.cookies.get("csrftoken", "")
        gn_session.post(
            f"{base}/account/login/",
            data={
                "login": username,
                "password": password,
                "csrfmiddlewaretoken": login_csrf,
                "next": "/",
            },
            headers={"Referer": f"{base}/account/login/"},
            timeout=15,
            allow_redirects=True,
        )
    except Exception as exc:
        log(f"  Style: Django login failed ({exc}).")
        return False

    if "sessionid" not in gn_session.cookies:
        log("  Style: login failed.")
        return False

    # ── Step 3: GeoServer session via OAuth2 authorization-code flow ──────
    log("  Style: obtaining GeoServer session via OAuth2 ...")
    try:
        gs_session = _geoserver_oauth2_session(gn_session, base, log)
    except Exception as exc:
        log(f"  Style: OAuth2 flow failed ({exc}); falling back to Django admin ...")
        gs_session = None

    if gs_session is not None:
        ok = _push_sld_to_geoserver(
            gs_session, base, style_name, layer_name, sld_xml, log
        )
        if ok:
            return True

    # ── Step 4: fallback — update Django DB via admin form ────────────────
    log("  Style: GeoServer push failed; updating Django DB via admin (DB only) ...")
    return _update_style_db_via_admin(
        gn_session, base, style_pk, style_name, sld_xml, log, _re
    )


def _geoserver_oauth2_session(
    gn_session: "requests.Session",
    base: str,
    log: Callable[[str], None],
) -> "requests.Session":
    """
    Perform GeoNode OAuth2 authorization-code flow to obtain a GeoServer session.

    GeoServer authenticates via GeoNode as its OAuth2 provider.  A GeoNode admin
    user is mapped to GeoServer ADMIN role, so the resulting JSESSIONID has full
    GeoServer REST API write access (style updates, layer config, etc.).

    Raises RuntimeError if any step fails.
    """
    import re as _re

    gs_session = requests.Session()

    # Step A: GeoServer redirects to GeoNode authorization endpoint
    r1 = gs_session.get(
        f"{base}/geoserver/web/j_spring_oauth2_geonode_login",
        timeout=15, allow_redirects=False,
    )
    authorize_url = r1.headers.get("Location", "")
    if not authorize_url:
        raise RuntimeError(
            f"j_spring_oauth2_geonode_login returned {r1.status_code} without redirect"
        )

    # Step B: GeoNode shows authorization form (user already logged in)
    r2 = gn_session.get(authorize_url, timeout=15, allow_redirects=True)
    auth_csrf_m = _re.search(
        r'name="csrfmiddlewaretoken" value="([^"]+)"', r2.text
    )
    auth_csrf = auth_csrf_m.group(1) if auth_csrf_m else gn_session.cookies.get("csrftoken", "")

    def _qparam(url: str, name: str) -> str:
        m = _re.search(rf"{name}=([^&]+)", url)
        return requests.utils.unquote(m.group(1)) if m else ""

    redirect_uri = _qparam(authorize_url, "redirect_uri")
    client_id    = _qparam(authorize_url, "client_id")
    scope        = _qparam(authorize_url, "scope") or "write"
    state        = _qparam(authorize_url, "state")

    # Step C: POST approval — GeoNode redirects to GeoServer callback with ?code=
    r3 = gn_session.post(
        r2.url,
        data={
            "csrfmiddlewaretoken": auth_csrf,
            "allow": "Authorize",
            "redirect_uri": redirect_uri,
            "scope": scope,
            "client_id": client_id,
            "state": state,
            "response_type": "code",
        },
        headers={"Referer": r2.url},
        timeout=15,
        allow_redirects=False,
    )
    callback_url = r3.headers.get("Location", "")
    if not callback_url:
        raise RuntimeError(
            f"OAuth2 approval POST returned {r3.status_code} without redirect"
        )

    # Step D: GeoServer receives code, exchanges it, sets JSESSIONID
    gs_session.get(callback_url, timeout=15, allow_redirects=True)

    rv = gs_session.get(f"{base}/geoserver/rest/about/version.json", timeout=10)
    if rv.status_code != 200:
        raise RuntimeError(
            f"GeoServer REST returned {rv.status_code} after OAuth2 flow"
        )

    log("  Style: GeoServer session ready.")
    return gs_session


def _push_sld_to_geoserver(
    gs_session: "requests.Session",
    base: str,
    style_name: str,
    layer_name: str,
    sld_xml: str,
    log: Callable[[str], None],
) -> bool:
    """PUT SLD to GeoServer REST and update the layer's default style."""
    sld_bytes = sld_xml.encode("utf-8")
    sld_ct    = "application/vnd.ogc.sld+xml"

    for style_url, desc in [
        (f"{base}/geoserver/rest/workspaces/geonode/styles/{style_name}.sld", "workspace"),
        (f"{base}/geoserver/rest/styles/{style_name}.sld",                    "global"),
    ]:
        try:
            r = gs_session.put(
                style_url,
                data=sld_bytes,
                headers={"Content-Type": sld_ct},
                timeout=30,
            )
            log(f"  Style: PUT {desc} -> HTTP {r.status_code}")
            if r.status_code in (200, 201):
                _set_layer_default_style(gs_session, base, layer_name, style_name, log)
                return True
            if r.status_code == 404:
                create_url = style_url[: style_url.rfind("/")]
                cr = gs_session.post(
                    create_url,
                    data=sld_bytes,
                    headers={"Content-Type": sld_ct},
                    timeout=30,
                )
                log(f"  Style: POST {desc} -> HTTP {cr.status_code}")
                if cr.status_code in (200, 201):
                    _set_layer_default_style(gs_session, base, layer_name, style_name, log)
                    return True
        except Exception as exc:
            log(f"  Style: {desc} error: {exc}")

    return False


def _set_layer_default_style(
    gs_session: "requests.Session",
    base: str,
    layer_name: str,
    style_name: str,
    log: Callable[[str], None],
) -> None:
    """Set style_name as the default style for geonode:{layer_name} in GeoServer."""
    layer_xml = (
        f"<layer><defaultStyle>"
        f"<name>{style_name}</name>"
        f"<workspace>geonode</workspace>"
        f"</defaultStyle></layer>"
    )
    for lyr_url, desc in [
        (f"{base}/geoserver/rest/layers/geonode:{layer_name}", "workspace layer"),
        (f"{base}/geoserver/rest/layers/{layer_name}",          "global layer"),
    ]:
        try:
            r = gs_session.put(
                lyr_url,
                data=layer_xml.encode("utf-8"),
                headers={"Content-Type": "application/xml"},
                timeout=15,
            )
            log(f"  Style: {desc} default -> HTTP {r.status_code}")
            if r.status_code in (200, 201):
                return
        except Exception as exc:
            log(f"  Style: {desc} error: {exc}")


def _update_style_db_via_admin(
    session: "requests.Session",
    base: str,
    style_pk: int,
    style_name: str,
    sld_xml: str,
    log: Callable[[str], None],
    _re,
) -> bool:
    """
    Update the SLD body in GeoNode's Django DB via the admin form.
    Caller must already have a valid session.
    This does NOT push the SLD to GeoServer.
    """
    # GET the admin form to read current sld_url (internal Docker URL)
    # We must preserve this value when posting back.
    admin_url = f"{base}/en-us/admin/layers/style/{style_pk}/change/"
    try:
        form_page = session.get(admin_url, timeout=12)
    except Exception as exc:
        log(f"  Style: admin GET failed ({exc}).")
        return False

    if form_page.status_code != 200:
        # Try without locale prefix
        admin_url = f"{base}/admin/layers/style/{style_pk}/change/"
        try:
            form_page = session.get(admin_url, timeout=12)
        except Exception as exc:
            log(f"  Style: admin GET (no locale) failed ({exc}).")
            return False

    if form_page.status_code != 200:
        log(f"  Style: admin form returned HTTP {form_page.status_code}.")
        return False

    # Extract CSRF token and internal sld_url from the form HTML
    form_csrf_m = _re.search(
        r'name="csrfmiddlewaretoken" value="([^"]+)"', form_page.text
    )
    form_csrf = form_csrf_m.group(1) if form_csrf_m else session.cookies.get("csrftoken", "")

    # sld_url holds the internal GeoServer Docker URL — preserve it
    sld_url_m = (
        _re.search(r'<input[^>]*name="sld_url"[^>]*value="([^"]*)"', form_page.text)
        or _re.search(r'<input[^>]*value="([^"]*)"[^>]*name="sld_url"', form_page.text)
    )
    sld_url_value = sld_url_m.group(1) if sld_url_m else ""

    # Step 4: POST the updated SLD body
    try:
        post_resp = session.post(
            admin_url,
            data={
                "csrfmiddlewaretoken": form_csrf,
                "name": style_name,
                "sld_title": "MESA Sensitivity (maximum)",
                "sld_body": sld_xml,
                "sld_version": "1.0.0",
                "sld_url": sld_url_value,
                "workspace": "geonode",
                "_save": "Save",
            },
            headers={"Referer": admin_url},
            timeout=30,
            allow_redirects=True,
        )
    except Exception as exc:
        log(f"  Style: admin POST failed ({exc}).")
        return False

    if post_resp.status_code == 200 and "errornote" not in post_resp.text.lower():
        log(f"  Style applied via Django admin (pk={style_pk}).")
        return True

    log(f"  Style: admin POST returned HTTP {post_resp.status_code}.")
    if "errornote" in post_resp.text.lower():
        err_m = _re.search(r'class="errornote"[^>]*>(.*?)</p>', post_resp.text, _re.S)
        if err_m:
            log(f"  Style error: {err_m.group(1).strip()[:150]}")
    return False


# ---------------------------------------------------------------------------
# Map creation
# ---------------------------------------------------------------------------

def create_geonode_map(
    base_url: str,
    username: str,
    password: str,
    map_layer_info: list[dict],
    map_title: str,
    log: Callable[[str], None],
) -> Optional[str]:
    """
    Create a GeoNode Map containing the uploaded datasets.

    map_layer_info: list of {layer_name, dataset_pk, alternate}
    Returns the map catalogue URL, or None on failure.
    """
    base = base_url.rstrip("/")

    # Step 1: create empty map
    try:
        r = requests.post(
            f"{base}/api/v2/maps/",
            auth=(username, password),
            json={"title": map_title, "abstract": "MESA sensitivity overview"},
            timeout=30,
        )
    except Exception as exc:
        log(f"  Map: POST /api/v2/maps/ failed ({exc}).")
        return None

    if r.status_code not in (200, 201):
        log(f"  Map: creation failed (HTTP {r.status_code}: {r.text[:200]}).")
        return None

    resp_data = r.json()
    map_obj = resp_data.get("map", resp_data)
    map_pk = map_obj.get("pk")
    if not map_pk:
        log("  Map: created but could not read pk from response.")
        return None

    log(f"  Map created (pk={map_pk}). Adding {len(map_layer_info)} layer(s) ...")

    # Step 2: PATCH map with maplayers
    maplayers = []
    for idx, info in enumerate(map_layer_info, start=1):
        lname = info.get("layer_name", "")
        alternate = info.get("alternate") or f"geonode:{lname}"
        maplayers.append({
            "extra_params": {"msId": f"{alternate}__{idx}"},
            "current_style": lname,
            "dataset": {"alternate": alternate},
        })

    try:
        r2 = requests.patch(
            f"{base}/api/v2/maps/{map_pk}/",
            auth=(username, password),
            json={"maplayers": maplayers},
            timeout=30,
        )
    except Exception as exc:
        log(f"  Map: PATCH maplayers failed ({exc}).")
        r2 = None

    if r2 is not None and r2.status_code in (200, 201, 204):
        log("  Map layers added.")
    else:
        code = r2.status_code if r2 is not None else "?"
        detail = r2.text[:120] if r2 is not None else ""
        log(f"  Map: could not add layers (HTTP {code}: {detail}).")
        log("  Tip: open the map in GeoNode and add layers manually.")

    map_url = f"{base}/catalogue/#/map/{map_pk}"
    return map_url


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def export_layers(
    base_url: str,
    username: str,
    password: str,
    selected_ids: list[str],
    geoparquet_dir: str,
    log: Callable[[str], None],
    cancel_event: threading.Event,
    config_path: Optional[str] = None,
    styles_output_dir: Optional[str] = None,
    all_layers: Optional[list[dict]] = None,
    confirm_cb: Optional[Callable[[str, int], bool]] = None,
    create_map: bool = True,
) -> list[dict]:
    """
    Export selected layers to GeoNode.

    selected_ids contains any mix of:
      - SUPPORTING_LAYERS ids  ("analysis_results", "routes", ...)
      - sensitivity layer ids  ("sensitivity:basic_mosaic", "sensitivity:H3_R6", ...)

    all_layers, if provided, is the flat list of all available layer dicts
    (sensitivity + supporting).  If omitted the function rebuilds it from
    SUPPORTING_LAYERS + sensitivity_layers().

    confirm_cb(layer_name, existing_pk) -> bool:
      Called when a dataset with the same name already exists on GeoNode.
      Return True to delete and replace it, False to skip.
      If not provided, existing layers are always skipped.

    For each selected layer:
      1. Check if dataset already exists on GeoNode; ask user if so
      2. Read GeoParquet (optionally filtered by filter_field/filter_value)
      3. Sanitize -> write temp GeoPackage
      4. If sld_field set: generate SLD from config palette, save to styles_output_dir
      5. Upload GeoPackage -> poll
      6. Apply SLD via PATCH /api/v2/styles/{pk}/
    After all layers: create a GeoNode Map with the published sensitivity layers.

    Returns list of result dicts {id, success, message, url}.
    """
    import geopandas as gpd

    colors, descs = read_palette(config_path)

    # Build the full ordered layer list the caller knows about
    if all_layers is None:
        all_layers = sensitivity_layers(geoparquet_dir) + SUPPORTING_LAYERS

    results: list[dict] = []
    # Track uploaded sensitivity layers for map creation
    uploaded_sens: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="mesa_geonode_") as tmp_dir:
        for layer in all_layers:
            lid = layer["id"]
            if lid not in selected_ids:
                continue
            if cancel_event.is_set():
                log("\nExport cancelled.")
                break

            label = layer["label"]
            parquet_file = os.path.join(geoparquet_dir, layer["parquet"])
            layer_name = layer["layer_name"]
            sld_field = layer.get("sld_field")
            filter_field = layer.get("filter_field")
            filter_value = layer.get("filter_value")
            is_sensitivity = lid.startswith("sensitivity:")

            log(f"\n--- {label} ---")

            if not os.path.isfile(parquet_file):
                msg = "Parquet file not found - layer skipped."
                log(f"  {msg}")
                results.append({"id": lid, "success": False, "message": msg, "url": None})
                continue

            # ── Existence check ──────────────────────────────────────────
            existing = _check_existing_dataset(base_url, username, password, layer_name)
            if existing:
                existing_pk = existing.get("pk")
                log(f"  Layer '{layer_name}' already exists on GeoNode (pk={existing_pk}).")
                if confirm_cb is not None:
                    replace = confirm_cb(layer_name, existing_pk)
                else:
                    replace = False
                if replace:
                    log(f"  Deleting existing layer ...")
                    deleted = _delete_dataset(base_url, username, password, existing_pk, log)
                    if not deleted:
                        log(f"  Delete failed — upload will proceed anyway.")
                    # Brief pause to allow server-side cleanup
                    time.sleep(2)
                else:
                    msg = "Skipped (layer already exists)."
                    log(f"  {msg}")
                    results.append({"id": lid, "success": False, "message": msg, "url": None})
                    continue

            # ── Read + filter ────────────────────────────────────────────
            try:
                log(f"  Reading {layer['parquet']} ...")
                gdf = gpd.read_parquet(parquet_file)
                if filter_field and filter_value is not None:
                    gdf = gdf[gdf[filter_field] == filter_value].copy()
                log(f"  {len(gdf):,} features. Writing GeoPackage ...")
                gpkg_path = _write_gpkg(gdf, layer_name, tmp_dir)
            except Exception as exc:
                msg = f"Preparation failed: {exc}"
                log(f"  {msg}")
                results.append({"id": lid, "success": False, "message": msg, "url": None})
                continue

            # ── Generate SLD ─────────────────────────────────────────────
            sld_xml: Optional[str] = None
            sld_path: Optional[str] = None
            if sld_field:
                try:
                    sld_xml = build_sensitivity_sld(layer_name, sld_field, colors, descs)
                    sld_path = os.path.join(tmp_dir, f"{layer_name}.sld")
                    with open(sld_path, "w", encoding="utf-8") as f:
                        f.write(sld_xml)
                    if styles_output_dir:
                        os.makedirs(styles_output_dir, exist_ok=True)
                        saved = os.path.join(styles_output_dir, f"{layer_name}.sld")
                        with open(saved, "w", encoding="utf-8") as f:
                            f.write(sld_xml)
                        log(f"  SLD saved: {saved}")
                except Exception as exc:
                    log(f"  SLD generation failed ({exc}), uploading without style.")
                    sld_xml = None
                    sld_path = None

            if cancel_event.is_set():
                log("\nExport cancelled.")
                break

            # ── Upload ───────────────────────────────────────────────────
            ok, msg, url, dataset_pk = _upload_gpkg(
                base_url, username, password, gpkg_path, log, sld_path=sld_path
            )
            log(f"  {'Done.' if ok else 'Failed.'} {msg}")
            if url:
                log(f"  View: {url}")

            # ── Apply style post-upload ───────────────────────────────────
            if ok and dataset_pk and sld_xml:
                log("  Applying MESA A-E cartography ...")
                _apply_style_after_upload(
                    base_url, username, password,
                    dataset_pk, sld_xml, layer_name, log,
                )
            elif ok and not dataset_pk and sld_xml:
                log("  Style: could not apply (dataset pk unknown after upload).")

            results.append({"id": lid, "success": ok, "message": msg, "url": url})

            if ok and is_sensitivity:
                alternate = f"geonode:{layer_name}"
                uploaded_sens.append({
                    "layer_name": layer_name,
                    "dataset_pk": dataset_pk,
                    "alternate": alternate,
                })

        # ── Create GeoNode Map ────────────────────────────────────────────
        if create_map and uploaded_sens and not cancel_event.is_set():
            log("\n--- Creating GeoNode map ---")
            map_title = "MESA Sensitivity Map"
            map_url = create_geonode_map(
                base_url, username, password,
                uploaded_sens, map_title, log,
            )
            if map_url:
                log(f"  Map: {map_url}")
                log(f"__MAP__:{map_url}")
            else:
                log("  Map creation failed. Create manually on the GeoNode server.")

    return results
